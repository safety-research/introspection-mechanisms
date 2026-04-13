"""
Utilities for probe-based analysis of introspection.

This module handles:
- Finding probe positions in tokenized prompts
- Creating activation extraction hooks
- Training linear and MLP probes
- Analyzing probe weights and directions
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dataclasses import dataclass
import json
from pathlib import Path


def convert_to_native_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


@dataclass
class ProbeResult:
    """Results from training a probe at a single layer."""
    layer_idx: int
    probe_type: str
    # Test/validation metrics (from CV)
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    auroc_mean: float
    auroc_std: float
    # Train metrics (from CV)
    train_accuracy_mean: float = 0.0
    train_accuracy_std: float = 0.0
    train_auroc_mean: float = 0.0
    train_auroc_std: float = 0.0
    # Sample counts
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0


def find_probe_position(tokenizer, formatted_prompt: str, model_name: str = "gemma3_27b") -> int:
    """
    Find the token position right before model generation begins.
    This is typically the position after the final assistant turn marker.

    For Gemma3 models: look for "<start_of_turn>model\n" and return the position
    of the newline token (the last token before generation).

    Args:
        tokenizer: The model's tokenizer
        formatted_prompt: The full formatted prompt string
        model_name: Model identifier for architecture-specific handling

    Returns:
        Token position (0-indexed) for the probe
    """
    # Tokenize full prompt
    tokens = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens['input_ids'][0]
    seq_len = len(input_ids)

    # For Gemma3 models, look for the pattern "<start_of_turn>model"
    # The probe position should be the last token of the prompt (right before generation)
    if "gemma" in model_name.lower():
        # Find where "<start_of_turn>model" occurs in the formatted prompt
        # The probe position is the newline after "model"
        marker = "<start_of_turn>model"
        marker_positions = []

        # Find all occurrences of the marker
        start = 0
        while True:
            pos = formatted_prompt.find(marker, start)
            if pos == -1:
                break
            marker_positions.append(pos)
            start = pos + 1

        if marker_positions:
            # Use the last occurrence (the final assistant turn)
            last_marker_pos = marker_positions[-1]

            # Tokenize up to and including "model" + newline
            text_up_to_marker = formatted_prompt[:last_marker_pos + len(marker)]

            # Check if there's a newline after
            newline_pos = last_marker_pos + len(marker)
            if newline_pos < len(formatted_prompt) and formatted_prompt[newline_pos] == '\n':
                text_up_to_marker = formatted_prompt[:newline_pos + 1]

            tokens_up_to_marker = tokenizer(text_up_to_marker, return_tensors="pt", add_special_tokens=True)
            probe_pos = tokens_up_to_marker['input_ids'].shape[1] - 1

            # Ensure we don't exceed sequence length
            probe_pos = min(probe_pos, seq_len - 1)
            return probe_pos

    # For other models or fallback: use the last token
    # This is the position right before generation begins
    return seq_len - 1


def find_probe_position_from_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    model_name: str = "gemma3_27b"
) -> Tuple[str, int]:
    """
    Format messages and find probe position.

    Args:
        tokenizer: The model's tokenizer
        messages: List of message dicts with 'role' and 'content'
        model_name: Model identifier

    Returns:
        Tuple of (formatted_prompt, probe_position)
    """
    # Format using chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback formatting
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system' and content:
                parts.append(f"System: {content}")
            elif role == 'user':
                parts.append(f"User: {content}")
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        formatted_prompt = "\n\n".join(parts)

    probe_pos = find_probe_position(tokenizer, formatted_prompt, model_name)

    return formatted_prompt, probe_pos


def create_activation_hook(
    storage_dict: Dict[str, torch.Tensor],
    layer_idx: int,
    probe_pos: int,
    extract_all_positions: bool = False
) -> Callable:
    """
    Create a hook function to extract residual stream activations.

    Args:
        storage_dict: Dictionary to store extracted activations
        layer_idx: Layer index for storage key
        probe_pos: Token position to extract (ignored if extract_all_positions=True)
        extract_all_positions: If True, extract all positions, not just probe_pos

    Returns:
        Hook function suitable for register_forward_hook
    """
    def hook(module, input, output):
        # output is typically a tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # hidden_states: [batch, seq_len, hidden_dim]
        if extract_all_positions:
            storage_dict[f'layer_{layer_idx}'] = hidden_states.detach().cpu()
        else:
            # Extract only the probe position
            # Handle case where probe_pos might be beyond sequence length
            seq_len = hidden_states.shape[1]
            pos = min(probe_pos, seq_len - 1)
            storage_dict[f'layer_{layer_idx}'] = hidden_states[:, pos, :].detach().cpu()

    return hook


def create_multi_layer_hooks(
    storage_dict: Dict[str, torch.Tensor],
    layer_indices: List[int],
    probe_pos: int
) -> List[Callable]:
    """
    Create hooks for multiple layers.

    Args:
        storage_dict: Dictionary to store extracted activations
        layer_indices: List of layer indices to extract from
        probe_pos: Token position to extract

    Returns:
        List of hook functions
    """
    hooks = []
    for layer_idx in layer_indices:
        hook = create_activation_hook(storage_dict, layer_idx, probe_pos)
        hooks.append(hook)
    return hooks


def train_layer_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    probe_type: str = 'linear',
    cv_folds: int = 5,
    random_seed: int = 42
) -> Tuple[Any, StandardScaler, ProbeResult]:
    """
    Train a probe on single layer's activations with cross-validation.

    Args:
        activations: Array of shape [n_samples, hidden_dim]
        labels: Binary labels of shape [n_samples]
        probe_type: 'linear' for LogisticRegression or 'mlp' for MLPClassifier
        cv_folds: Number of cross-validation folds
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (trained_probe, fitted_scaler, ProbeResult)
    """
    # Normalize activations
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = labels.astype(int)

    # Check class balance
    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    n_samples = len(y)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_scores = {'accuracy': [], 'f1': [], 'auroc': []}
    cv_train_scores = {'accuracy': [], 'auroc': []}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if probe_type == 'linear':
            probe = LogisticRegression(
                C=1.0,
                max_iter=3000,  # Increased for convergence on high-dim data
                random_state=random_seed,
                solver='lbfgs',
                class_weight='balanced',
                n_jobs=1  # Prevent CPU overload
            )
        else:  # mlp
            probe = MLPClassifier(
                hidden_layer_sizes=(256,),
                max_iter=1000,
                random_state=random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )

        probe.fit(X_train, y_train)

        # Validation/test metrics
        y_pred = probe.predict(X_val)
        y_prob = probe.predict_proba(X_val)[:, 1]

        cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))

        # AUROC requires at least 2 classes in y_val
        if len(np.unique(y_val)) > 1:
            cv_scores['auroc'].append(roc_auc_score(y_val, y_prob))
        else:
            cv_scores['auroc'].append(0.5)  # Random baseline

        # Train metrics
        y_train_pred = probe.predict(X_train)
        y_train_prob = probe.predict_proba(X_train)[:, 1]
        cv_train_scores['accuracy'].append(accuracy_score(y_train, y_train_pred))
        if len(np.unique(y_train)) > 1:
            cv_train_scores['auroc'].append(roc_auc_score(y_train, y_train_prob))
        else:
            cv_train_scores['auroc'].append(0.5)

    # Train final probe on all data
    if probe_type == 'linear':
        final_probe = LogisticRegression(
            C=1.0,
            max_iter=3000,  # Increased for convergence on high-dim data
            random_state=random_seed,
            solver='lbfgs',
            class_weight='balanced',
            n_jobs=1  # Prevent CPU overload
        )
    else:
        final_probe = MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=1000,
            random_state=random_seed,
            early_stopping=True,
            validation_fraction=0.1
        )

    final_probe.fit(X, y)

    # Create result object
    result = ProbeResult(
        layer_idx=-1,  # To be set by caller
        probe_type=probe_type,
        # Test/validation metrics
        accuracy_mean=np.mean(cv_scores['accuracy']),
        accuracy_std=np.std(cv_scores['accuracy']),
        f1_mean=np.mean(cv_scores['f1']),
        f1_std=np.std(cv_scores['f1']),
        auroc_mean=np.mean(cv_scores['auroc']),
        auroc_std=np.std(cv_scores['auroc']),
        # Train metrics
        train_accuracy_mean=np.mean(cv_train_scores['accuracy']),
        train_accuracy_std=np.std(cv_train_scores['accuracy']),
        train_auroc_mean=np.mean(cv_train_scores['auroc']),
        train_auroc_std=np.std(cv_train_scores['auroc']),
        # Sample counts
        n_samples=n_samples,
        n_positive=n_positive,
        n_negative=n_negative
    )

    return final_probe, scaler, result


@dataclass
class MultinomialProbeResult:
    """Results from multinomial probe training at a single layer."""
    layer_idx: int
    n_classes: int
    # Per-class sample counts
    class_counts: Dict[int, int]
    n_samples: int
    # Overall metrics
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    weighted_f1_mean: float
    weighted_f1_std: float
    # Per-class metrics (averaged across CV folds)
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    per_class_f1_std: Dict[int, float]  # Standard deviation for error bars
    # Training metrics
    train_accuracy_mean: float
    train_accuracy_std: float
    # Derived binary-equivalent metrics (for comparison with binary probes)
    detection_auroc: float  # Classes 2,3 vs 0,1
    identification_auroc: float  # Class 3 vs 2 (on subset)
    rejection_auroc: float  # Class 0 vs 1 (on subset)
    calibration_auroc: float  # Classes 2,3 vs 1 (on subset)


def _train_multinomial_fold(fold_data):
    """
    Train and evaluate one CV fold for multinomial probe.
    Helper function for parallel execution.
    """
    from sklearn.metrics import precision_recall_fscore_support

    X_train, X_val, y_train, y_val, n_classes, random_seed, max_iter, C = fold_data

    # Train multinomial logistic regression
    probe = LogisticRegression(
        C=C,
        max_iter=max_iter,
        tol=1e-4,
        random_state=random_seed,
        solver='lbfgs',
        class_weight='balanced',
        warm_start=False,
    )
    probe.fit(X_train, y_train)

    # Predictions
    y_pred = probe.predict(X_val)
    y_prob = probe.predict_proba(X_val)

    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    fold_result = {
        'accuracy': accuracy_score(y_val, y_pred),
        'macro_f1': np.mean(f1),
        'weighted_f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'train_accuracy': accuracy_score(y_train, probe.predict(X_train)),
    }

    # Binary-equivalent AUROC metrics
    # Detection: P(class in {2,3}) vs {0,1}
    prob_detection = y_prob[:, 2] + y_prob[:, 3]
    true_detection = np.isin(y_val, [2, 3]).astype(int)
    if len(np.unique(true_detection)) > 1:
        fold_result['detection_auroc'] = roc_auc_score(true_detection, prob_detection)
    else:
        fold_result['detection_auroc'] = 0.5

    # Identification: P(class 3) vs class 2 (only where y in {2,3})
    id_mask = np.isin(y_val, [2, 3])
    if np.sum(id_mask) > 10 and len(np.unique(y_val[id_mask])) > 1:
        prob_correct_id = y_prob[id_mask, 3] / (y_prob[id_mask, 2] + y_prob[id_mask, 3] + 1e-10)
        true_correct_id = (y_val[id_mask] == 3).astype(int)
        fold_result['identification_auroc'] = roc_auc_score(true_correct_id, prob_correct_id)
    else:
        fold_result['identification_auroc'] = 0.5

    # Rejection: P(class 0) vs class 1 (only where y in {0,1})
    rej_mask = np.isin(y_val, [0, 1])
    if np.sum(rej_mask) > 10 and len(np.unique(y_val[rej_mask])) > 1:
        prob_tn = y_prob[rej_mask, 0] / (y_prob[rej_mask, 0] + y_prob[rej_mask, 1] + 1e-10)
        true_tn = (y_val[rej_mask] == 0).astype(int)
        fold_result['rejection_auroc'] = roc_auc_score(true_tn, prob_tn)
    else:
        fold_result['rejection_auroc'] = 0.5

    # Calibration: P(class in {2,3}) vs class 1 (only on injection trials)
    cal_mask = np.isin(y_val, [1, 2, 3])
    if np.sum(cal_mask) > 10:
        prob_tp = y_prob[cal_mask, 2] + y_prob[cal_mask, 3]
        true_tp = np.isin(y_val[cal_mask], [2, 3]).astype(int)
        if len(np.unique(true_tp)) > 1:
            fold_result['calibration_auroc'] = roc_auc_score(true_tp, prob_tp)
        else:
            fold_result['calibration_auroc'] = 0.5
    else:
        fold_result['calibration_auroc'] = 0.5

    return fold_result


def train_multinomial_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 4,
    cv_folds: int = 5,
    random_seed: int = 42,
    n_jobs: int = -1,
    max_iter: int = 5000,
    C: float = 0.5,
) -> Tuple[Any, StandardScaler, MultinomialProbeResult]:
    """
    Train a multinomial logistic regression probe with cross-validation.

    This trains a single 4-class model that captures all outcomes:
        Class 0: True Negative (control, no detection)
        Class 1: False Negative (injection, no detection)
        Class 2: Wrong ID (injection, detection, wrong concept)
        Class 3: Correct ID (injection, detection, correct concept)

    Args:
        activations: Array of shape [n_samples, hidden_dim]
        labels: Multi-class labels of shape [n_samples] with values in {0,1,2,3}
        n_classes: Number of classes (default 4)
        cv_folds: Number of cross-validation folds
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs for CV (-1 = all cores)
        max_iter: Maximum iterations for logistic regression (default 5000)
        C: Inverse regularization strength (default 0.5, lower = stronger regularization)

    Returns:
        Tuple of (trained_probe, fitted_scaler, MultinomialProbeResult)
    """
    from joblib import Parallel, delayed

    # Normalize activations
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = labels.astype(int)

    # Class counts
    class_counts = {i: int(np.sum(y == i)) for i in range(n_classes)}
    n_samples = len(y)

    # Prepare CV fold data
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_data = [
        (X[train_idx], X[val_idx], y[train_idx], y[val_idx], n_classes, random_seed, max_iter, C)
        for train_idx, val_idx in cv.split(X, y)
    ]

    # Run CV folds in parallel
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_train_multinomial_fold)(data) for data in fold_data
    )

    # Aggregate results
    cv_scores = {
        'accuracy': [r['accuracy'] for r in fold_results],
        'macro_f1': [r['macro_f1'] for r in fold_results],
        'weighted_f1': [r['weighted_f1'] for r in fold_results],
    }
    cv_train_scores = {'accuracy': [r['train_accuracy'] for r in fold_results]}

    # Per-class metrics
    per_class_metrics = {i: {'precision': [], 'recall': [], 'f1': []} for i in range(n_classes)}
    for r in fold_results:
        for i in range(n_classes):
            per_class_metrics[i]['precision'].append(r['per_class_precision'][i])
            per_class_metrics[i]['recall'].append(r['per_class_recall'][i])
            per_class_metrics[i]['f1'].append(r['per_class_f1'][i])

    # Binary-equivalent metrics
    binary_metrics = {
        'detection_auroc': [r['detection_auroc'] for r in fold_results],
        'identification_auroc': [r['identification_auroc'] for r in fold_results],
        'rejection_auroc': [r['rejection_auroc'] for r in fold_results],
        'calibration_auroc': [r['calibration_auroc'] for r in fold_results],
    }

    # Train final probe on all data
    final_probe = LogisticRegression(
        C=C,
        max_iter=max_iter,
        tol=1e-4,
        random_state=random_seed,
        solver='lbfgs',
        class_weight='balanced',
    )
    final_probe.fit(X, y)

    # Create result object
    result = MultinomialProbeResult(
        layer_idx=-1,  # To be set by caller
        n_classes=n_classes,
        class_counts=class_counts,
        n_samples=n_samples,
        # Overall metrics
        accuracy_mean=np.mean(cv_scores['accuracy']),
        accuracy_std=np.std(cv_scores['accuracy']),
        macro_f1_mean=np.mean(cv_scores['macro_f1']),
        macro_f1_std=np.std(cv_scores['macro_f1']),
        weighted_f1_mean=np.mean(cv_scores['weighted_f1']),
        weighted_f1_std=np.std(cv_scores['weighted_f1']),
        # Per-class metrics
        per_class_precision={i: np.mean(per_class_metrics[i]['precision']) for i in range(n_classes)},
        per_class_recall={i: np.mean(per_class_metrics[i]['recall']) for i in range(n_classes)},
        per_class_f1={i: np.mean(per_class_metrics[i]['f1']) for i in range(n_classes)},
        per_class_f1_std={i: np.std(per_class_metrics[i]['f1']) for i in range(n_classes)},
        # Training metrics
        train_accuracy_mean=np.mean(cv_train_scores['accuracy']),
        train_accuracy_std=np.std(cv_train_scores['accuracy']),
        # Binary-equivalent metrics
        detection_auroc=np.mean(binary_metrics['detection_auroc']),
        identification_auroc=np.mean(binary_metrics['identification_auroc']),
        rejection_auroc=np.mean(binary_metrics['rejection_auroc']),
        calibration_auroc=np.mean(binary_metrics['calibration_auroc']),
    )

    return final_probe, scaler, result


# ============================================================================
# HIERARCHICAL BINARY PROBES
# ============================================================================

@dataclass
class HierarchicalProbeResult:
    """
    Results from hierarchical probe training at a single layer.

    Contains results for both detection and identification probes.
    """
    layer_idx: int

    # Detection probe metrics (all samples: Classes 2,3 vs 0,1)
    detection_accuracy_mean: float
    detection_accuracy_std: float
    detection_f1_mean: float
    detection_f1_std: float
    detection_auroc_mean: float
    detection_auroc_std: float
    detection_n_samples: int
    detection_n_positive: int
    detection_n_negative: int
    detection_train_accuracy_mean: float
    detection_train_auroc_mean: float

    # Identification probe metrics (detection-positive only: Class 3 vs 2)
    identification_accuracy_mean: float
    identification_accuracy_std: float
    identification_f1_mean: float
    identification_f1_std: float
    identification_auroc_mean: float
    identification_auroc_std: float
    identification_n_samples: int
    identification_n_positive: int
    identification_n_negative: int
    identification_train_accuracy_mean: float
    identification_train_auroc_mean: float


def train_hierarchical_probes(
    detection_activations: np.ndarray,
    detection_labels: np.ndarray,
    identification_activations: np.ndarray,
    identification_labels: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
    n_jobs: int = -1,
    max_iter: int = 5000,
    C: float = 0.01,
) -> Tuple[Any, StandardScaler, Any, StandardScaler, HierarchicalProbeResult]:
    """
    Train hierarchical binary probes: Detection (all) + Identification (detection-positive).

    This addresses the class imbalance problem in the 4-class multinomial approach
    by training two focused binary probes:
    1. Detection Probe: Classes 2,3 vs 0,1 (is there a detection claim?)
    2. Identification Probe: Class 3 vs 2 (given detection, is ID correct?)

    Args:
        detection_activations: Activations for ALL samples [n_samples, hidden_dim]
        detection_labels: Binary labels for detection (1 = detection claimed)
        identification_activations: Activations for detection-positive samples only
        identification_labels: Binary labels for identification (1 = correct ID)
        cv_folds: Number of cross-validation folds
        random_seed: Random seed
        n_jobs: Parallel jobs for CV
        max_iter: Max iterations for logistic regression
        C: Inverse regularization strength

    Returns:
        Tuple of (detection_probe, detection_scaler,
                  identification_probe, identification_scaler,
                  HierarchicalProbeResult)
    """
    from joblib import Parallel, delayed

    # =========================================================================
    # Train Detection Probe (ALL samples)
    # =========================================================================
    detection_scaler = StandardScaler()
    X_det = detection_scaler.fit_transform(detection_activations)
    y_det = detection_labels.astype(int)

    det_n_samples = len(y_det)
    det_n_positive = int(np.sum(y_det == 1))
    det_n_negative = int(np.sum(y_det == 0))

    # CV for detection probe
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    det_cv_scores = {'accuracy': [], 'f1': [], 'auroc': []}
    det_train_scores = {'accuracy': [], 'auroc': []}

    for train_idx, val_idx in cv.split(X_det, y_det):
        X_train, X_val = X_det[train_idx], X_det[val_idx]
        y_train, y_val = y_det[train_idx], y_det[val_idx]

        det_probe_cv = LogisticRegression(
            C=C, max_iter=max_iter, tol=1e-4, random_state=random_seed,
            solver='lbfgs', class_weight='balanced', warm_start=False
        )
        det_probe_cv.fit(X_train, y_train)

        y_pred = det_probe_cv.predict(X_val)
        y_prob = det_probe_cv.predict_proba(X_val)[:, 1]

        det_cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
        det_cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        if len(np.unique(y_val)) > 1:
            det_cv_scores['auroc'].append(roc_auc_score(y_val, y_prob))
        else:
            det_cv_scores['auroc'].append(0.5)

        # Train metrics
        y_train_prob = det_probe_cv.predict_proba(X_train)[:, 1]
        det_train_scores['accuracy'].append(accuracy_score(y_train, det_probe_cv.predict(X_train)))
        if len(np.unique(y_train)) > 1:
            det_train_scores['auroc'].append(roc_auc_score(y_train, y_train_prob))
        else:
            det_train_scores['auroc'].append(0.5)

    # Train final detection probe on all data
    detection_probe = LogisticRegression(
        C=C, max_iter=max_iter, tol=1e-4, random_state=random_seed,
        solver='lbfgs', class_weight='balanced', warm_start=False
    )
    detection_probe.fit(X_det, y_det)

    # =========================================================================
    # Train Identification Probe (DETECTION-POSITIVE samples only)
    # =========================================================================
    identification_scaler = StandardScaler()
    X_id = identification_scaler.fit_transform(identification_activations)
    y_id = identification_labels.astype(int)

    id_n_samples = len(y_id)
    id_n_positive = int(np.sum(y_id == 1))
    id_n_negative = int(np.sum(y_id == 0))

    id_cv_scores = {'accuracy': [], 'f1': [], 'auroc': []}
    id_train_scores = {'accuracy': [], 'auroc': []}

    if id_n_samples >= cv_folds * 2 and id_n_positive > 0 and id_n_negative > 0:
        # CV for identification probe
        cv_id = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

        for train_idx, val_idx in cv_id.split(X_id, y_id):
            X_train, X_val = X_id[train_idx], X_id[val_idx]
            y_train, y_val = y_id[train_idx], y_id[val_idx]

            id_probe_cv = LogisticRegression(
                C=C, max_iter=max_iter, tol=1e-4, random_state=random_seed,
                solver='lbfgs', class_weight='balanced', warm_start=False
            )
            id_probe_cv.fit(X_train, y_train)

            y_pred = id_probe_cv.predict(X_val)
            y_prob = id_probe_cv.predict_proba(X_val)[:, 1]

            id_cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            id_cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            if len(np.unique(y_val)) > 1:
                id_cv_scores['auroc'].append(roc_auc_score(y_val, y_prob))
            else:
                id_cv_scores['auroc'].append(0.5)

            # Train metrics
            y_train_prob = id_probe_cv.predict_proba(X_train)[:, 1]
            id_train_scores['accuracy'].append(accuracy_score(y_train, id_probe_cv.predict(X_train)))
            if len(np.unique(y_train)) > 1:
                id_train_scores['auroc'].append(roc_auc_score(y_train, y_train_prob))
            else:
                id_train_scores['auroc'].append(0.5)

        # Train final identification probe on all data
        identification_probe = LogisticRegression(
            C=C, max_iter=max_iter, tol=1e-4, random_state=random_seed,
            solver='lbfgs', class_weight='balanced', warm_start=False
        )
        identification_probe.fit(X_id, y_id)
    else:
        # Not enough samples for proper CV - train without CV
        identification_probe = LogisticRegression(
            C=C, max_iter=max_iter, tol=1e-4, random_state=random_seed,
            solver='lbfgs', class_weight='balanced', warm_start=False
        )
        if id_n_positive > 0 and id_n_negative > 0:
            identification_probe.fit(X_id, y_id)
            # Compute single-fold metrics
            y_prob = identification_probe.predict_proba(X_id)[:, 1]
            id_cv_scores['accuracy'] = [accuracy_score(y_id, identification_probe.predict(X_id))]
            id_cv_scores['f1'] = [f1_score(y_id, identification_probe.predict(X_id), zero_division=0)]
            id_cv_scores['auroc'] = [roc_auc_score(y_id, y_prob) if len(np.unique(y_id)) > 1 else 0.5]
            id_train_scores['accuracy'] = id_cv_scores['accuracy']
            id_train_scores['auroc'] = id_cv_scores['auroc']
        else:
            # Degenerate case - no positive or negative samples
            id_cv_scores = {'accuracy': [0.5], 'f1': [0.0], 'auroc': [0.5]}
            id_train_scores = {'accuracy': [0.5], 'auroc': [0.5]}

    # =========================================================================
    # Build result object
    # =========================================================================
    result = HierarchicalProbeResult(
        layer_idx=-1,  # To be set by caller

        # Detection probe metrics
        detection_accuracy_mean=np.mean(det_cv_scores['accuracy']),
        detection_accuracy_std=np.std(det_cv_scores['accuracy']),
        detection_f1_mean=np.mean(det_cv_scores['f1']),
        detection_f1_std=np.std(det_cv_scores['f1']),
        detection_auroc_mean=np.mean(det_cv_scores['auroc']),
        detection_auroc_std=np.std(det_cv_scores['auroc']),
        detection_n_samples=det_n_samples,
        detection_n_positive=det_n_positive,
        detection_n_negative=det_n_negative,
        detection_train_accuracy_mean=np.mean(det_train_scores['accuracy']),
        detection_train_auroc_mean=np.mean(det_train_scores['auroc']),

        # Identification probe metrics
        identification_accuracy_mean=np.mean(id_cv_scores['accuracy']),
        identification_accuracy_std=np.std(id_cv_scores['accuracy']),
        identification_f1_mean=np.mean(id_cv_scores['f1']),
        identification_f1_std=np.std(id_cv_scores['f1']),
        identification_auroc_mean=np.mean(id_cv_scores['auroc']),
        identification_auroc_std=np.std(id_cv_scores['auroc']),
        identification_n_samples=id_n_samples,
        identification_n_positive=id_n_positive,
        identification_n_negative=id_n_negative,
        identification_train_accuracy_mean=np.mean(id_train_scores['accuracy']),
        identification_train_auroc_mean=np.mean(id_train_scores['auroc']),
    )

    return detection_probe, detection_scaler, identification_probe, identification_scaler, result


def analyze_probe_weights(
    probe: Any,
    scaler: StandardScaler,
    probe_type: str = 'linear',
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Analyze probe weights to understand which dimensions are most important.

    Args:
        probe: Trained probe model
        scaler: Fitted scaler used during training
        probe_type: 'linear' or 'mlp'
        top_k: Number of top dimensions to report

    Returns:
        Dictionary with weight analysis results
    """
    analysis = {
        'probe_type': probe_type,
        'top_k': top_k,
    }

    if probe_type == 'linear':
        # Get coefficients (shape: [1, hidden_dim] for binary classification)
        weights = probe.coef_[0]

        # Compute weight statistics
        analysis['weight_norm'] = float(np.linalg.norm(weights))
        analysis['weight_mean'] = float(np.mean(weights))
        analysis['weight_std'] = float(np.std(weights))
        analysis['weight_min'] = float(np.min(weights))
        analysis['weight_max'] = float(np.max(weights))

        # Get top positive and negative dimensions
        top_positive_idx = np.argsort(weights)[-top_k:][::-1]
        top_negative_idx = np.argsort(weights)[:top_k]

        analysis['top_positive_dims'] = top_positive_idx.tolist()
        analysis['top_positive_weights'] = weights[top_positive_idx].tolist()
        analysis['top_negative_dims'] = top_negative_idx.tolist()
        analysis['top_negative_weights'] = weights[top_negative_idx].tolist()

        # Compute sparsity (fraction of weights close to zero)
        threshold = 0.01 * np.max(np.abs(weights))
        analysis['sparsity'] = float(np.mean(np.abs(weights) < threshold))

        # Save full weights for correlation analysis
        analysis['weights'] = weights.tolist()

    elif probe_type == 'mlp':
        # For MLP, analyze the first layer weights
        if hasattr(probe, 'coefs_') and len(probe.coefs_) > 0:
            first_layer_weights = probe.coefs_[0]  # [hidden_dim, hidden_layer_size]

            # Compute importance as L2 norm across output dimensions
            importance = np.linalg.norm(first_layer_weights, axis=1)

            analysis['importance_norm'] = float(np.linalg.norm(importance))
            analysis['importance_mean'] = float(np.mean(importance))
            analysis['importance_std'] = float(np.std(importance))

            # Get top important dimensions
            top_dims = np.argsort(importance)[-top_k:][::-1]
            analysis['top_important_dims'] = top_dims.tolist()
            analysis['top_importance_values'] = importance[top_dims].tolist()

            # SVD of first layer for principal directions
            U, S, Vt = np.linalg.svd(first_layer_weights, full_matrices=False)
            analysis['singular_values'] = S[:min(10, len(S))].tolist()
            analysis['top_singular_direction'] = U[:, 0].tolist()

    return analysis


def compute_direction_correlation(
    probe_weights: np.ndarray,
    reference_direction: np.ndarray
) -> float:
    """
    Compute cosine similarity between probe weights and a reference direction.

    Args:
        probe_weights: Probe weight vector [hidden_dim]
        reference_direction: Reference direction vector [hidden_dim]

    Returns:
        Cosine similarity (-1 to 1)
    """
    # Normalize both vectors
    probe_norm = probe_weights / (np.linalg.norm(probe_weights) + 1e-8)
    ref_norm = reference_direction / (np.linalg.norm(reference_direction) + 1e-8)

    return float(np.dot(probe_norm, ref_norm))


def save_probe(
    probe: Any,
    scaler: StandardScaler,
    result: ProbeResult,
    save_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save a trained probe and its associated data.

    Args:
        probe: Trained probe model
        scaler: Fitted scaler
        result: ProbeResult object
        save_path: Path to save to (without extension)
        metadata: Optional additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save probe and scaler using torch
    probe_data = {
        'probe': probe,
        'scaler': scaler,
        'result': {
            'layer_idx': result.layer_idx,
            'probe_type': result.probe_type,
            'accuracy_mean': result.accuracy_mean,
            'accuracy_std': result.accuracy_std,
            'f1_mean': result.f1_mean,
            'f1_std': result.f1_std,
            'auroc_mean': result.auroc_mean,
            'auroc_std': result.auroc_std,
            'n_samples': result.n_samples,
            'n_positive': result.n_positive,
            'n_negative': result.n_negative,
        },
        'metadata': metadata or {}
    }

    torch.save(probe_data, str(save_path) + '.pt')

    # Also save metadata as JSON for easy inspection
    json_data = {
        'result': probe_data['result'],
        'metadata': probe_data['metadata']
    }

    # Convert numpy types to native Python types for JSON serialization
    json_data = convert_to_native_types(json_data)

    with open(str(save_path) + '_metadata.json', 'w') as f:
        json.dump(json_data, f, indent=2)


def load_probe(load_path: Path) -> Tuple[Any, StandardScaler, ProbeResult, Dict]:
    """
    Load a saved probe and its associated data.

    Args:
        load_path: Path to load from (without extension)

    Returns:
        Tuple of (probe, scaler, result, metadata)
    """
    load_path = Path(load_path)
    probe_data = torch.load(str(load_path) + '.pt', weights_only=False)

    result_dict = probe_data['result']
    result = ProbeResult(
        layer_idx=result_dict['layer_idx'],
        probe_type=result_dict['probe_type'],
        accuracy_mean=result_dict['accuracy_mean'],
        accuracy_std=result_dict['accuracy_std'],
        f1_mean=result_dict['f1_mean'],
        f1_std=result_dict['f1_std'],
        auroc_mean=result_dict['auroc_mean'],
        auroc_std=result_dict['auroc_std'],
        n_samples=result_dict['n_samples'],
        n_positive=result_dict['n_positive'],
        n_negative=result_dict['n_negative'],
    )

    return probe_data['probe'], probe_data['scaler'], result, probe_data['metadata']


def construct_introspection_prompt(trial_number: int = 1) -> List[Dict[str, str]]:
    """
    Construct the standard introspection prompt messages.

    This matches the format used in exp1 and exp21.

    Args:
        trial_number: Trial number to include in the prompt

    Returns:
        List of message dicts for chat template
    """
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": (
            "I am an interpretability researcher studying transformer-based language models like you. "
            "I have access to the internals of your neural network. I have identified patterns in your "
            "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
            "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
            "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
        )},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            f"Trial {trial_number}: Do you detect an injected thought? "
            f"If so, what is the injected thought about?"
        )},
    ]

    return messages


def extract_label_from_result(result: Dict) -> Tuple[bool, str]:
    """
    Extract binary label from a trial result.

    Positive class (successful introspection):
        - trial_type == "injection" AND claims_detection == True AND correct_identification == True
        - trial_type == "forced_injection" AND correct_identification == True

    Negative class (failed introspection):
        - Everything else (control trials, missed detections, wrong identifications)

    Args:
        result: Trial result dictionary from exp1/exp21

    Returns:
        Tuple of (is_positive, label_reason)
    """
    trial_type = result.get('trial_type', 'unknown')

    # Get evaluation results
    evals = result.get('evaluations', {})
    claims_detection = evals.get('claims_detection', {}).get('claims_detection', False)
    correct_identification = evals.get('correct_concept_identification', {}).get('correct_identification', False)

    # Handle each trial type
    if trial_type == 'control':
        # Control trials: should not claim detection
        # If they do claim detection, it's a false positive (negative label)
        return False, 'control_trial'

    elif trial_type == 'injection':
        # Standard injection trial: need both detection AND correct identification
        if claims_detection and correct_identification:
            return True, 'successful_introspection'
        elif not claims_detection:
            return False, 'missed_detection'
        else:
            return False, 'wrong_identification'

    elif trial_type == 'forced_injection':
        # Forced injection: detection is forced (assumed True), only identification matters
        if correct_identification:
            return True, 'successful_forced_introspection'
        else:
            return False, 'forced_wrong_identification'

    # Unknown trial type - could be old data format
    # Try to infer from 'injected' field for backward compatibility
    if result.get('injected', False):
        if claims_detection and correct_identification:
            return True, 'successful_introspection'
        elif not claims_detection:
            return False, 'missed_detection'
        else:
            return False, 'wrong_identification'
    else:
        return False, 'control_trial'


def is_response_coherent(response: str, min_length: int = 10, max_repetition_ratio: float = 0.5) -> bool:
    """
    Basic heuristic check for response coherence.

    Args:
        response: Model response text
        min_length: Minimum response length
        max_repetition_ratio: Maximum ratio of repeated words

    Returns:
        True if response appears coherent
    """
    if len(response.strip()) < min_length:
        return False

    words = response.lower().split()
    if len(words) < 3:
        return False

    # Check for excessive repetition
    unique_words = set(words)
    repetition_ratio = 1 - (len(unique_words) / len(words))

    return repetition_ratio < max_repetition_ratio
