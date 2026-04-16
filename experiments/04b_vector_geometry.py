"""
Experiment 4b: Concept Vector Geometry Analysis

Analyses used in the paper:
  - Section 2: LDA cross-validation for success/failure partition
      (adaptive threshold selection, balanced accuracy 75.6%)
  - Section 2 / Appendix B: Ridge regression on concept vectors
      predicting detection rates (nested 5-fold CV, R^2=0.406)
  - Section 4.3: PCA of 500 L2-normalized concept vectors

Usage:
    python 04b_vector_geometry.py --injection-dir analysis/02b_steering_500_concepts
"""

import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from scipy.linalg import subspace_angles
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Optional: adjustText for non-overlapping scatter labels
try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Experiment 4b: Concept Vector Geometry Analysis"
)
parser.add_argument(
    "-i",
    "--injection-dir",
    type=str,
    default="analysis/02b_steering_500_concepts",
    help="Path to Concept Injection results directory",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="analysis/04b_vector_geometry",
    help="Output directory",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Analyze only this specific model",
)
parser.add_argument(
    "-s",
    "--standardize",
    action="store_true",
    help="Standardize before PCA (default: False)",
)
parser.add_argument(
    "-n",
    "--normalize",
    action="store_true",
    default=True,
    help="L2-normalize vectors before PCA (default: True)",
)
parser.add_argument(
    "-nc",
    "--n-components",
    type=lambda x: None if x.lower() == "none" else int(x),
    default=10,
    help="Number of PCA components (default: 10, 'none' for all)",
)
parser.add_argument(
    "-sm",
    "--split-metric",
    type=str,
    default="adaptive",
    choices=["detection_rate", "combined_rate", "adaptive"],
    help="Metric for success/failure split (default: adaptive)",
)
parser.add_argument(
    "-st",
    "--split-threshold",
    type=float,
    default=0.2,
    help="Threshold for success/failure split (default: 0.2)",
)
parser.add_argument(
    "-l",
    "--layers",
    type=int,
    nargs="+",
    default=[29, 35, 37],
    help="Layer indices to analyze (default: [29, 35, 37])",
)
parser.add_argument(
    "-ss",
    "--strengths",
    type=float,
    nargs="+",
    default=[4.0],
    help="Steering strengths to analyze (default: [4.0])",
)
parser.add_argument(
    "-x",
    "--exclude-concepts",
    type=str,
    nargs="+",
    default=["Apples", "Bicycles", "Ocean"],
    help="Concepts to exclude from analysis",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj


def add_scatter_labels(
    ax,
    x_coords,
    y_coords,
    labels,
    fontsize=14,
    alpha=0.95,
    max_labels=None,
    values_for_priority=None,
    n_grid=6,
):
    """Add labels to scatter plot points, with spatial sampling to avoid clutter."""
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    n_points = len(labels)

    if max_labels is None:
        if n_points <= 30:
            max_labels = n_points
        elif n_points <= 60:
            max_labels = 55
        elif n_points <= 100:
            max_labels = 75
        elif n_points <= 150:
            max_labels = 90
        else:
            max_labels = 100

    if n_points > max_labels:
        label_indices = set()

        if values_for_priority is not None:
            values = np.asarray(values_for_priority)
            n_extremes = max_labels // 3
            label_indices.update(np.argsort(values)[-n_extremes:])
            label_indices.update(np.argsort(values)[:n_extremes])

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_bins = np.linspace(x_min, x_max, n_grid + 1)
        y_bins = np.linspace(y_min, y_max, n_grid + 1)

        remaining_budget = max_labels - len(label_indices)
        points_per_cell = max(1, remaining_budget // (n_grid * n_grid))

        for i in range(n_grid):
            for j in range(n_grid):
                in_cell = (
                    (x_coords >= x_bins[i])
                    & (x_coords < x_bins[i + 1])
                    & (y_coords >= y_bins[j])
                    & (y_coords < y_bins[j + 1])
                )
                cell_indices = np.where(in_cell)[0]
                if len(cell_indices) > 0:
                    if values_for_priority is not None:
                        sorted_cell = cell_indices[np.argsort(values[cell_indices])]
                        picks = list(sorted_cell[-points_per_cell:]) + list(
                            sorted_cell[:points_per_cell]
                        )
                        label_indices.update(picks[: points_per_cell * 2])
                    else:
                        picks = np.random.choice(
                            cell_indices,
                            min(points_per_cell, len(cell_indices)),
                            replace=False,
                        )
                        label_indices.update(picks)

        center_x, center_y = np.mean(x_coords), np.mean(y_coords)
        distances = np.sqrt(
            (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        )
        n_outliers = max_labels // 6
        label_indices.update(np.argsort(distances)[-n_outliers:])

        label_indices = list(label_indices)
        if len(label_indices) > max_labels:
            if values_for_priority is not None:
                idx_values = [
                    (idx, abs(values[idx] - np.median(values)))
                    for idx in label_indices
                ]
                idx_values.sort(key=lambda x: x[1], reverse=True)
                label_indices = [idx for idx, _ in idx_values[:max_labels]]
            else:
                label_indices = label_indices[:max_labels]

        label_indices = sorted(label_indices)
        x_label = x_coords[label_indices]
        y_label = y_coords[label_indices]
        labels_subset = [labels[i] for i in label_indices]
    else:
        x_label = x_coords
        y_label = y_coords
        labels_subset = labels

    bbox_props = dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="#cccccc",
        alpha=0.85,
        linewidth=0.5,
    )

    if HAS_ADJUST_TEXT:
        texts = []
        for i, label in enumerate(labels_subset):
            t = ax.text(
                x_label[i],
                y_label[i],
                label,
                fontsize=fontsize,
                fontweight="bold",
                alpha=alpha,
                bbox=bbox_props,
            )
            texts.append(t)
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="-",
                color="#666666",
                alpha=0.7,
                lw=1.2,
                shrinkA=0,
                shrinkB=2,
            ),
            expand_points=(1.8, 1.8),
            expand_text=(1.4, 1.4),
            force_text=(0.6, 0.6),
            force_points=(0.4, 0.4),
            lim=5000,
            precision=0.0005,
            only_move={"points": "xy", "text": "xy"},
            avoid_self=True,
        )
    else:
        for i, label in enumerate(labels_subset):
            ax.annotate(
                label,
                (x_label[i], y_label[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=fontsize,
                fontweight="bold",
                alpha=alpha,
                bbox=bbox_props,
            )


# ---------------------------------------------------------------------------
# Model / layer helpers
# ---------------------------------------------------------------------------

MODEL_LAYERS = {
    "gemma3_27b": 62,
    "gemma2_27b": 46,
    "gemma2_9b": 42,
    "gemma2_2b": 26,
    "llama_70b": 80,
    "llama_3_3_70b": 80,
    "qwen_72b": 80,
    "qwen3_235b": 94,
}


def get_model_num_layers(model_name: str) -> int:
    return MODEL_LAYERS.get(model_name, 62)


def layer_index_to_fraction(layer_idx: int, model_name: str) -> float:
    return layer_idx / get_model_num_layers(model_name)


def get_config_dir_name(layer: int, strength: float) -> str:
    return f"layer_{layer}_strength_{strength}"


def get_available_layers(vectors_dir: Path) -> List[int]:
    layers = []
    for d in vectors_dir.glob("layer_*"):
        try:
            layers.append(int(d.name.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return sorted(layers)


def discover_configs(injection_model_dir: Path) -> List[Tuple[int, float]]:
    """Discover layer/strength configurations from experiment 01 (concept injection) results."""
    configs = []
    for d in injection_model_dir.glob("layer_*_strength_*"):
        if not d.is_dir():
            continue
        try:
            parts = d.name.split("_")
            if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "strength":
                configs.append((int(parts[1]), float(parts[3])))
        except (ValueError, IndexError):
            continue
    configs.sort()
    return configs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_injection_results(
    injection_dir: Path, config_dir: Optional[Path] = None
) -> Dict[str, Dict]:
    """Load Concept Injection results and compute per-concept detection metrics."""
    concept_metrics: Dict[str, Dict] = {}

    if config_dir is not None:
        config_dirs = [config_dir] if config_dir.exists() else []
    else:
        config_dirs = list(injection_dir.glob("layer_*_strength_*"))

    if not config_dirs:
        print(f"No Concept Injection results found in {injection_dir}")
        return {}

    for cdir in config_dirs:
        results_file = cdir / "results.json"
        if not results_file.exists():
            continue
        try:
            with open(results_file) as f:
                data = json.load(f)
            for result in data.get("results", []):
                concept = result.get("concept")
                if not concept:
                    continue
                if concept not in concept_metrics:
                    concept_metrics[concept] = {
                        "trial_type": [],
                        "detected": [],
                        "correctly_identified": [],
                    }
                trial_type = result.get("trial_type", "injection")
                detected = False
                correctly_identified = False
                if "evaluations" in result:
                    evals = result["evaluations"]
                    if "claims_detection" in evals:
                        detected = bool(evals["claims_detection"].get("grade", 0))
                    if "correct_concept_identification" in evals:
                        correctly_identified = bool(
                            evals["correct_concept_identification"].get("grade", 0)
                        )
                else:
                    detected = result.get(
                        "llm_judge_detected", result.get("detected", False)
                    )
                    correctly_identified = result.get("llm_judge_correct_id", False)
                concept_metrics[concept]["trial_type"].append(trial_type)
                concept_metrics[concept]["detected"].append(detected)
                concept_metrics[concept]["correctly_identified"].append(
                    correctly_identified
                )
        except Exception as e:
            print(f"Warning: Failed to load {results_file}: {e}")

    # Aggregate per concept
    summary = {}
    for concept, data in concept_metrics.items():
        inj = [i for i, t in enumerate(data["trial_type"]) if t == "injection"]
        if not inj:
            continue
        det = [data["detected"][i] for i in inj]
        ident = [data["correctly_identified"][i] for i in inj]
        detection_rate = np.mean(det)
        identification_rate = np.mean(ident)
        combined = [det[i] and ident[i] for i in range(len(det))]
        combined_rate = np.mean(combined)
        conditional_id_rate = combined_rate / detection_rate if detection_rate > 0 else 0
        summary[concept] = {
            "detection_rate": detection_rate,
            "identification_rate": identification_rate,
            "combined_rate": combined_rate,
            "conditional_id_rate": conditional_id_rate,
            "n_trials": len(inj),
        }
    return summary


def aggregate_concept_metrics_across_configs(
    injection_model_dir: Path, configs: List[Tuple[int, float]]
) -> Dict[str, Dict]:
    """Aggregate concept metrics across configs using max detection rate."""
    all_metrics: Dict[str, Dict] = {}
    for layer_idx, strength in configs:
        config_dir = injection_model_dir / get_config_dir_name(layer_idx, strength)
        if not config_dir.exists():
            continue
        metrics = load_injection_results(injection_model_dir, config_dir=config_dir)
        for concept, m in metrics.items():
            if concept not in all_metrics:
                all_metrics[concept] = {
                    "detection_rate": 0.0,
                    "combined_rate": 0.0,
                    "identification_rate": 0.0,
                    "conditional_id_rate": 0.0,
                    "n_trials": 0,
                }
            for key in [
                "detection_rate",
                "combined_rate",
                "identification_rate",
                "conditional_id_rate",
            ]:
                all_metrics[concept][key] = max(
                    all_metrics[concept][key], m.get(key, 0)
                )
            all_metrics[concept]["n_trials"] += m.get("n_trials", 0)
    return all_metrics


def load_concept_vectors(
    vectors_dir: Path,
    layer_fraction: Optional[float] = None,
    layer_index: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Load concept vectors from .pt files."""
    vectors = {}
    vector_files = list(vectors_dir.glob("*.pt"))

    if not vector_files:
        layer_dirs = list(vectors_dir.glob("layer_*"))
        if layer_dirs:
            if layer_index is not None:
                target = vectors_dir / f"layer_{layer_index}"
                if target.exists():
                    vector_files = list(target.glob("*.pt"))
                    print(f"  Loading vectors from: {target.name}")
                else:
                    available = sorted([d.name for d in layer_dirs])
                    raise ValueError(
                        f"Layer index {layer_index} not found. Available: {', '.join(available)}"
                    )
            elif layer_fraction is not None:
                frac_dir = vectors_dir / f"layer_{layer_fraction:.2f}"
                int_dir = vectors_dir / f"layer_{int(layer_fraction * 100)}"
                if frac_dir.exists():
                    vector_files = list(frac_dir.glob("*.pt"))
                elif int_dir.exists():
                    vector_files = list(int_dir.glob("*.pt"))
                else:
                    # Tolerance-based search
                    for ld in layer_dirs:
                        try:
                            val = float(ld.name.replace("layer_", ""))
                            if abs(val - layer_fraction) < 0.01:
                                vector_files = list(ld.glob("*.pt"))
                                break
                        except ValueError:
                            continue
            else:
                vector_files = list(layer_dirs[0].glob("*.pt"))

    if not vector_files:
        print(f"No vector files found in {vectors_dir}")
        return {}

    for vf in vector_files:
        try:
            vectors[vf.stem] = torch.load(vf, map_location="cpu")
        except Exception as e:
            print(f"Warning: Failed to load {vf}: {e}")

    print(f"Loaded {len(vectors)} concept vectors")
    return vectors


# ---------------------------------------------------------------------------
# Geometric analysis (PCA)
# ---------------------------------------------------------------------------


def compute_geometric_properties(
    vectors: Dict[str, torch.Tensor],
    standardize: bool = False,
    normalize: bool = True,
    n_components: int = 10,
) -> Tuple[pd.DataFrame, Dict]:
    """Compute PCA and per-concept geometric properties."""
    concept_names = list(vectors.keys())
    matrix = torch.stack([vectors[c] for c in concept_names]).float().numpy()
    print(f"  Input: {len(concept_names)} concepts, {matrix.shape[1]} dimensions")

    baseline_mean = matrix.mean(axis=0)

    # L2-normalize
    matrix_for_pca = matrix
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_for_pca = matrix / (norms + 1e-10)
        print(f"  L2-normalized vectors to unit length")

    # Standardize
    scaler = None
    if standardize:
        scaler = StandardScaler()
        matrix_for_pca = scaler.fit_transform(matrix_for_pca)
        print(f"  Standardized vectors for PCA")

    # PCA
    max_comp = min(len(concept_names), matrix_for_pca.shape[1])
    n_keep = min(n_components, max_comp) if n_components else max_comp

    pca_full = PCA(n_components=None)
    pca_full.fit(matrix_for_pca)

    pca = PCA(n_components=n_keep)
    pca.fit(matrix_for_pca)
    projections = pca.transform(matrix_for_pca)

    # Use full-PCA variance ratios for accuracy
    pca.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_keep]
    pca.explained_variance_ = pca_full.explained_variance_[:n_keep]
    pca.components_ = pca_full.components_[:n_keep]

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(
        f"  PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
        f"PC1+PC2={cum_var[1]:.1%}, "
        f"first 5={cum_var[min(4, len(cum_var) - 1)]:.1%}"
    )

    # Per-concept properties
    rows = []
    for i, concept in enumerate(concept_names):
        vec = vectors[concept].float().numpy()
        l2_norm = np.linalg.norm(vec)
        rows.append(
            {
                "concept": concept,
                "l2_norm": l2_norm,
                "distance_from_mean": np.linalg.norm(vec - baseline_mean),
                "pc1_projection": projections[i, 0],
                "pc2_projection": projections[i, 1] if projections.shape[1] > 1 else 0,
                "pc3_projection": projections[i, 2] if projections.shape[1] > 2 else 0,
            }
        )

    analysis = {
        "pca": pca,
        "pca_projections": projections,
        "concept_names": concept_names,
        "vector_matrix": matrix,
        "scaler": scaler,
        "standardized": standardize,
        "normalized": normalize,
    }
    return pd.DataFrame(rows), analysis


def correlate_properties_with_detection(
    df: pd.DataFrame, concept_metrics: Dict[str, Dict]
) -> pd.DataFrame:
    """Add detection metrics to the properties dataframe."""
    for col in [
        "detection_rate",
        "identification_rate",
        "combined_rate",
        "conditional_id_rate",
        "n_trials",
    ]:
        df[col] = df["concept"].map(
            lambda c, _col=col: concept_metrics.get(c, {}).get(
                _col, 0 if _col == "n_trials" else np.nan
            )
        )
    df = df[df["detection_rate"].notna()].copy()
    return df


# ---------------------------------------------------------------------------
# LDA threshold sweep (Section 2)
# ---------------------------------------------------------------------------


def sweep_subspace_parameters(
    df: pd.DataFrame, vectors: Dict[str, torch.Tensor]
) -> Tuple[float, str]:
    """Sweep thresholds and metrics to find optimal LDA split."""
    valid_concepts = set(df["concept"].values) & set(vectors.keys())
    df_valid = df[df["concept"].isin(valid_concepts)].copy()
    if len(df_valid) < 4:
        return 0.2, "combined_rate"

    thresholds = [i / 100 for i in range(1, 100)]
    thresholds.append(None)
    metrics = ["detection_rate", "combined_rate"]

    best_score = -1
    best_threshold = 0.2
    best_metric = "combined_rate"
    results = []

    print(
        f"  Sweeping {len(thresholds)} thresholds x {len(metrics)} metrics..."
    )

    for metric in metrics:
        if metric not in df_valid.columns:
            continue
        for threshold in thresholds:
            threshold_val = (
                df_valid[metric].median() if threshold is None else threshold
            )
            try:
                sa = compute_introspection_subspace_analysis(
                    df,
                    vectors,
                    threshold=threshold_val,
                    metric=metric,
                    allow_fallback=False,
                    verbose=False,
                )
                if not sa:
                    continue
                f1 = sa.get("lda_f1_score")
                accuracy = sa.get("lda_test_accuracy") or sa.get(
                    "lda_classification_accuracy"
                )
                bal_acc = sa.get("lda_balanced_accuracy")
                n_s = len(sa["success_concepts"])
                n_f = len(sa["failure_concepts"])
                n_total = n_s + n_f
                balance = min(n_s, n_f) / n_total if n_total > 0 else 0
                if balance <= 0.2:
                    continue
                if f1 is None and accuracy is None:
                    continue
                results.append(
                    {
                        "threshold_val": threshold_val,
                        "metric": metric,
                        "f1": f1,
                        "f1_std": sa.get("lda_f1_std"),
                        "accuracy": accuracy,
                        "balanced_accuracy": bal_acc,
                        "n_success": n_s,
                        "n_failure": n_f,
                        "balance": balance,
                    }
                )
                score = f1 if f1 is not None else accuracy
                if score > best_score:
                    best_score = score
                    best_threshold = threshold_val
                    best_metric = metric
            except Exception:
                continue

    if results:
        results_sorted = sorted(
            results,
            key=lambda x: x["f1"] if x["f1"] is not None else x["accuracy"],
            reverse=True,
        )
        print(f"  Top 5 splits (by F1, balance >= 20%):")
        for i, r in enumerate(results_sorted[:5], 1):
            f1s = (
                f"F1={r['f1']:.3f}+/-{r['f1_std']:.3f}"
                if r["f1"] is not None and r["f1_std"] is not None
                else "F1=N/A"
            )
            print(
                f"    {i}. tau={r['threshold_val']:.3f}, metric={r['metric']}: "
                f"{f1s}, bal_acc={r['balanced_accuracy']:.3f}, "
                f"split={r['n_success']}/{r['n_failure']}"
            )
        print(f"\n  Selected: tau={best_threshold:.3f}, metric={best_metric}")
    else:
        print("  Warning: No valid splits found, using defaults")

    return best_threshold, best_metric


# ---------------------------------------------------------------------------
# Introspection subspace analysis (LDA, subspace angles)
# ---------------------------------------------------------------------------


def compute_introspection_subspace_analysis(
    df: pd.DataFrame,
    vectors: Dict[str, torch.Tensor],
    threshold: float = 0.2,
    metric: str = "combined_rate",
    allow_fallback: bool = True,
    verbose: bool = True,
) -> Dict:
    """Binary partition + LDA cross-validation + subspace angles."""
    valid_concepts = set(df["concept"].values) & set(vectors.keys())
    df_valid = df[df["concept"].isin(valid_concepts)].copy()
    if len(df_valid) < 4:
        return {}

    # Binary split
    success_mask = df_valid[metric] > threshold
    failure_mask = ~success_mask
    success_concepts = df_valid[success_mask]["concept"].values
    failure_concepts = df_valid[failure_mask]["concept"].values

    if len(success_concepts) == 0 or len(failure_concepts) == 0:
        if not allow_fallback:
            return {}
        median_val = df_valid[metric].median()
        success_mask = df_valid[metric] >= median_val
        failure_mask = ~success_mask
        success_concepts = df_valid[success_mask]["concept"].values
        failure_concepts = df_valid[failure_mask]["concept"].values

    if len(success_concepts) == 0 or len(failure_concepts) == 0:
        return {}

    # Filter to concepts with vectors
    success_concepts = [c for c in success_concepts if c in vectors]
    failure_concepts = [c for c in failure_concepts if c in vectors]
    if len(success_concepts) < 2 or len(failure_concepts) < 2:
        return {}

    success_vectors = np.array([vectors[c].float().numpy() for c in success_concepts])
    failure_vectors = np.array([vectors[c].float().numpy() for c in failure_concepts])

    if success_vectors.shape[1] != failure_vectors.shape[1]:
        return {}

    # Mean difference
    mean_success = success_vectors.mean(axis=0)
    mean_failure = failure_vectors.mean(axis=0)
    mean_difference = mean_success - mean_failure
    mean_difference_norm = mean_difference / (np.linalg.norm(mean_difference) + 1e-10)

    # LDA with stratified k-fold CV
    all_concepts_list = list(success_concepts) + list(failure_concepts)
    all_vectors_arr = np.vstack([success_vectors, failure_vectors])
    all_labels = np.array([1] * len(success_concepts) + [0] * len(failure_concepts))

    lda = None
    lda_direction = None
    lda_projections = None
    lda_classification_accuracy = None
    lda_train_accuracy = None
    lda_test_accuracy = None
    lda_f1_score_val = None
    lda_f1_std = None
    lda_balanced_accuracy = None

    try:
        n_folds = 5
        min_per_class = n_folds

        if (
            len(success_concepts) >= min_per_class
            and len(failure_concepts) >= min_per_class
        ):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_acc, fold_f1, fold_bal, fold_train = [], [], [], []

            for train_idx, test_idx in skf.split(all_vectors_arr, all_labels):
                X_tr, X_te = all_vectors_arr[train_idx], all_vectors_arr[test_idx]
                y_tr, y_te = all_labels[train_idx], all_labels[test_idx]
                fold_lda = LinearDiscriminantAnalysis(n_components=1)
                fold_lda.fit(X_tr, y_tr)
                tr_proj = fold_lda.transform(X_tr).flatten()
                te_proj = fold_lda.transform(X_te).flatten()
                thr = (tr_proj[y_tr == 1].mean() + tr_proj[y_tr == 0].mean()) / 2
                fold_train.append(accuracy_score(y_tr, (tr_proj >= thr).astype(int)))
                fold_acc.append(accuracy_score(y_te, (te_proj >= thr).astype(int)))
                fold_f1.append(
                    f1_score(y_te, (te_proj >= thr).astype(int), average="macro")
                )
                fold_bal.append(
                    balanced_accuracy_score(y_te, (te_proj >= thr).astype(int))
                )

            lda_train_accuracy = np.mean(fold_train)
            lda_test_accuracy = np.mean(fold_acc)
            lda_classification_accuracy = lda_test_accuracy
            lda_f1_score_val = np.mean(fold_f1)
            lda_f1_std = np.std(fold_f1)
            lda_balanced_accuracy = np.mean(fold_bal)

            # Fit final LDA on all data
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(all_vectors_arr, all_labels)
            lda_direction = lda.coef_[0]
            lda_projections = lda.transform(all_vectors_arr).flatten()

            if verbose:
                print(
                    f"  LDA {n_folds}-fold CV: train={lda_train_accuracy:.2%}, "
                    f"test={lda_test_accuracy:.2%}, "
                    f"F1={lda_f1_score_val:.2%}+/-{lda_f1_std:.2%}, "
                    f"bal_acc={lda_balanced_accuracy:.2%}"
                )
        else:
            if verbose:
                print(f"  Not enough samples for {n_folds}-fold CV, using all data")
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(all_vectors_arr, all_labels)
            lda_direction = lda.coef_[0]
            lda_projections = lda.transform(all_vectors_arr).flatten()
    except Exception as e:
        if verbose:
            print(f"Warning: LDA failed: {e}")

    # Subspace angles
    subspace_angles_result = None
    pca_s = pca_f = None
    try:
        nc = min(5, len(success_concepts) - 1, len(failure_concepts) - 1)
        if nc > 0:
            pca_s = PCA(n_components=nc)
            pca_f = PCA(n_components=nc)
            pca_s.fit(success_vectors)
            pca_f.fit(failure_vectors)
            angles_rad = subspace_angles(pca_s.components_.T, pca_f.components_.T)
            subspace_angles_result = np.degrees(angles_rad)
    except Exception as e:
        if verbose:
            print(f"Warning: Subspace angle computation failed: {e}")

    # Silhouette score
    from sklearn.metrics import silhouette_score as sil_score

    silhouette = None
    try:
        silhouette = sil_score(all_vectors_arr, all_labels)
    except Exception:
        pass

    return {
        "success_concepts": list(success_concepts),
        "failure_concepts": list(failure_concepts),
        "success_vectors": success_vectors,
        "failure_vectors": failure_vectors,
        "mean_success": mean_success,
        "mean_failure": mean_failure,
        "mean_difference": mean_difference,
        "mean_difference_norm": mean_difference_norm,
        "lda": lda,
        "lda_direction": lda_direction,
        "lda_projections": lda_projections,
        "lda_classification_accuracy": lda_classification_accuracy,
        "lda_train_accuracy": lda_train_accuracy,
        "lda_test_accuracy": lda_test_accuracy,
        "lda_f1_score": lda_f1_score_val,
        "lda_f1_std": lda_f1_std,
        "lda_balanced_accuracy": lda_balanced_accuracy,
        "pca_success": pca_s,
        "pca_failure": pca_f,
        "subspace_angles": subspace_angles_result,
        "silhouette_score": silhouette,
        "threshold": threshold,
        "metric_used": metric,
        "all_concepts": all_concepts_list,
        "all_vectors": all_vectors_arr,
        "all_labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Ridge regression with nested CV (Section 2, Appendix B)
# ---------------------------------------------------------------------------


def compute_ridge_regression(
    vectors_arr: np.ndarray,
    detection_rates: np.ndarray,
    concepts: List[str],
    output_dir: Path,
    metric_name: str = "detection_rate",
    verbose: bool = True,
) -> Dict:
    """
    Ridge regression predicting detection rates from concept vectors.

    Uses nested cross-validation:
      - Inner loop (3-fold): alpha selection via RidgeCV
      - Outer loop (5-fold): unbiased R^2 estimate
    """
    n_samples, n_features = vectors_arr.shape
    if verbose:
        print(f"    Data: {n_samples} samples, {n_features} features")

    # Center
    X = vectors_arr - vectors_arr.mean(axis=0)
    y = detection_rates - detection_rates.mean()

    alphas = 10 ** np.linspace(-2, 8, 25)

    # Nested CV: outer 5-fold, inner RidgeCV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2 = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for train_idx, test_idx in outer_cv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            ridge_cv = RidgeCV(alphas=alphas, cv=3)
            ridge_cv.fit(X_tr, y_tr)
            y_pred = ridge_cv.predict(X_te)

            ss_res = np.sum((y_te - y_pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fold_r2.append(r2)

    cv_r2 = np.mean(fold_r2)
    cv_r2_std = np.std(fold_r2)

    # Fit final model on all data for direction extraction
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ridge_final_cv = RidgeCV(alphas=alphas, cv=min(5, n_samples))
        ridge_final_cv.fit(X, y)
        optimal_alpha = ridge_final_cv.alpha_

        ridge_final = Ridge(alpha=optimal_alpha)
        ridge_final.fit(X, y)
        direction = ridge_final.coef_

    direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

    # Sign convention: positive correlation with target
    if np.corrcoef(X @ direction_norm, y)[0, 1] < 0:
        direction_norm = -direction_norm

    # Full-data correlation
    preds = ridge_final.predict(X)
    r_full, p_full = stats.pearsonr(preds, y)

    if verbose:
        print(f"    Ridge: alpha={optimal_alpha:.2e}")
        print(f"      Full-data R^2={r_full**2:.3f}")
        print(
            f"      Nested CV R^2={cv_r2:.3f} +/- {cv_r2_std:.3f} (outer 5-fold)"
        )

    # Save under two filenames:
    #   - introspection_direction_ridge_regression.pt: descriptive name used here
    #   - primary_axis.pt: short name consumed by 04c_bidirectional_steering.py,
    #                      04d_ridge_swap.py, and 04i_alternative_geometric_tests.py
    direction_tensor = torch.tensor(direction_norm, dtype=torch.float32)
    torch.save(
        direction_tensor,
        output_dir / "introspection_direction_ridge_regression.pt",
    )
    torch.save(direction_tensor, output_dir / "primary_axis.pt")

    metadata = {
        "method": "Ridge_Regression",
        "alpha": float(optimal_alpha),
        "metric_used": metric_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "full_data_r": float(r_full),
        "full_data_r_squared": float(r_full ** 2),
        "nested_cv_r_squared": float(cv_r2),
        "nested_cv_r_squared_std": float(cv_r2_std),
        "nested_cv_fold_r2": [float(x) for x in fold_r2],
        "p_value": float(p_full),
    }
    with open(
        output_dir / "introspection_direction_ridge_regression_metadata.json", "w"
    ) as f:
        json.dump(convert_numpy_types(metadata), f, indent=2)

    return {
        "direction_norm": direction_norm,
        "alpha": optimal_alpha,
        "r": r_full,
        "r_squared": r_full ** 2,
        "cv_r_squared": cv_r2,
        "cv_r_squared_std": cv_r2_std,
        "p_value": p_full,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def create_pca_visualization(
    df: pd.DataFrame,
    analysis: Dict,
    output_dir: Path,
):
    """PCA scatter plots colored by detection rate and combined rate."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    pca = analysis["pca"]
    proj_all = analysis["pca_projections"]
    names_all = analysis["concept_names"]

    det_map = {r["concept"]: r["detection_rate"] for _, r in df.iterrows()}
    valid = [i for i, c in enumerate(names_all) if c in det_map]
    proj = proj_all[valid]
    names = [names_all[i] for i in valid]
    det_rates = [det_map[c] for c in names]

    # 1. PCA colored by detection rate
    fig, ax = plt.subplots(figsize=(20, 18))
    det_vmax = max(det_rates) if det_rates else 1.0
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=det_rates,
        cmap="RdYlGn",
        s=700,
        alpha=0.8,
        edgecolor="black",
        linewidth=2.5,
        vmin=0,
        vmax=det_vmax,
    )
    add_scatter_labels(
        ax, proj[:, 0], proj[:, 1], names, values_for_priority=det_rates
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=20)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=20)
    ax.set_title("Concept vectors in PCA space (colored by detection rate)", fontsize=24)
    ax.tick_params(axis="both", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Detection rate", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. PCA colored by combined rate
    comb_map = {r["concept"]: r["combined_rate"] for _, r in df.iterrows()}
    comb_rates = [comb_map.get(c, 0) for c in names]
    fig, ax = plt.subplots(figsize=(18, 16))
    comb_vmax = max(comb_rates) if comb_rates else 1.0
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=comb_rates,
        cmap="RdYlGn",
        s=350,
        alpha=0.75,
        edgecolor="black",
        linewidth=1.5,
        vmin=0,
        vmax=comb_vmax,
    )
    add_scatter_labels(
        ax, proj[:, 0], proj[:, 1], names, values_for_priority=comb_rates
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=18)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=18)
    ax.set_title(
        "Concept vectors in PCA space (colored by introspection rate)", fontsize=22
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("P(Detect & Correct ID | Injection)", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_combined.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Variance explained
    fig, ax = plt.subplots(figsize=(10, 6))
    nc = len(pca.explained_variance_ratio_)
    ax.bar(
        range(1, nc + 1),
        pca.explained_variance_ratio_,
        alpha=0.8,
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.plot(
        range(1, nc + 1),
        np.cumsum(pca.explained_variance_ratio_),
        "r-o",
        linewidth=2,
        markersize=8,
        label="Cumulative",
    )
    ax.set_xlabel("Principal component", fontsize=14)
    ax.set_ylabel("Variance explained", fontsize=14)
    ax.set_title("PCA variance explained", fontsize=16)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_variance_explained.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  PCA plots saved to {plots_dir}")


def create_lda_visualization(
    subspace_analysis: Dict,
    output_dir: Path,
):
    """LDA projection histogram with KDE and strip plot."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    if not subspace_analysis.get("lda_projections") is not None:
        return

    all_proj = subspace_analysis["lda_projections"]
    all_labels = subspace_analysis["all_labels"]
    success_proj = all_proj[all_labels == 1]
    failure_proj = all_proj[all_labels == 0]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.3},
    )

    # Histogram + KDE
    ax1.hist(
        success_proj,
        bins=25,
        alpha=0.5,
        color="#2ecc71",
        edgecolor="black",
        linewidth=1,
        label="Success (detected)",
        density=True,
    )
    ax1.hist(
        failure_proj,
        bins=25,
        alpha=0.5,
        color="#e74c3c",
        edgecolor="black",
        linewidth=1,
        label="Failure (undetected)",
        density=True,
    )

    x_lo = min(success_proj.min(), failure_proj.min())
    x_hi = max(success_proj.max(), failure_proj.max())
    x_smooth = np.linspace(x_lo, x_hi, 200)
    if len(success_proj) > 1:
        ax1.plot(
            x_smooth,
            gaussian_kde(success_proj)(x_smooth),
            color="#27ae60",
            linewidth=2.5,
            alpha=0.8,
        )
    if len(failure_proj) > 1:
        ax1.plot(
            x_smooth,
            gaussian_kde(failure_proj)(x_smooth),
            color="#c0392b",
            linewidth=2.5,
            alpha=0.8,
        )

    ax1.set_xlabel("LDA projection (introspection direction)", fontsize=13)
    ax1.set_ylabel("Density", fontsize=13)
    ax1.set_title("Distribution along introspection direction", fontsize=14)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3)

    # Stats box
    acc = subspace_analysis.get("lda_classification_accuracy")
    train_acc = subspace_analysis.get("lda_train_accuracy")
    test_acc = subspace_analysis.get("lda_test_accuracy")
    if train_acc is not None and test_acc is not None:
        stats_text = (
            f"Train accuracy: {train_acc:.1%} | Test accuracy: {test_acc:.1%}\n"
            f"Success mean: {success_proj.mean():.2f} | Failure mean: {failure_proj.mean():.2f}"
        )
    elif acc is not None:
        stats_text = (
            f"Classification accuracy: {acc:.1%}\n"
            f"Success mean: {success_proj.mean():.2f} | Failure mean: {failure_proj.mean():.2f}"
        )
    else:
        stats_text = ""

    if stats_text:
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.85,
                edgecolor="black",
                linewidth=1,
            ),
        )

    # Strip + boxplot
    np.random.seed(42)
    jitter_s = np.random.normal(0, 0.02, len(success_proj))
    jitter_f = np.random.normal(0, 0.02, len(failure_proj))

    ax2.scatter(
        success_proj,
        np.ones(len(success_proj)) + jitter_s,
        color="#2ecc71",
        alpha=0.7,
        s=50,
        edgecolor="black",
        linewidth=0.5,
        label="Success",
        zorder=3,
    )
    ax2.scatter(
        failure_proj,
        np.zeros(len(failure_proj)) + jitter_f,
        color="#e74c3c",
        alpha=0.7,
        s=50,
        edgecolor="black",
        linewidth=0.5,
        label="Failure",
        zorder=3,
    )

    bp1 = ax2.boxplot(
        [failure_proj],
        positions=[0],
        widths=0.3,
        patch_artist=True,
        vert=False,
        showmeans=False,
        showfliers=False,
    )
    bp2 = ax2.boxplot(
        [success_proj],
        positions=[1],
        widths=0.3,
        patch_artist=True,
        vert=False,
        showmeans=False,
        showfliers=False,
    )
    bp1["boxes"][0].set_facecolor("#e74c3c")
    bp1["boxes"][0].set_alpha(0.3)
    bp2["boxes"][0].set_facecolor("#2ecc71")
    bp2["boxes"][0].set_alpha(0.3)

    ax2.set_xlabel("LDA projection (introspection direction)", fontsize=13)
    ax2.set_ylabel("Class", fontsize=13)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Failure", "Success"])
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend(fontsize=10, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
    plt.savefig(
        plots_dir / "lda_projection_histogram.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  LDA histogram saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Layer-wise analysis
# ---------------------------------------------------------------------------


def analyze_layer_wise(
    injection_model_dir: Path,
    vectors_dir: Path,
    combined_df: pd.DataFrame,
    concept_metrics: Dict[str, Dict],
    output_dir: Path,
    threshold: float = 0.2,
    metric: str = "combined_rate",
) -> Optional[pd.DataFrame]:
    """Run LDA subspace analysis at each available layer."""
    available_layers = get_available_layers(vectors_dir)
    if len(available_layers) < 2:
        print("  Fewer than 2 layers found, skipping layer-wise analysis")
        return None

    print(f"  Found {len(available_layers)} layers")
    rows = []

    for layer in available_layers:
        print(f"    Layer {layer}...")
        vecs = load_concept_vectors(vectors_dir, layer_index=layer)
        if not vecs:
            continue
        sa = compute_introspection_subspace_analysis(
            combined_df, vecs, threshold=threshold, metric=metric, verbose=False
        )
        if not sa:
            continue

        angles = sa.get("subspace_angles")
        mean_angle = None
        if angles is not None:
            mean_angle = float(np.mean(angles))

        rows.append(
            {
                "layer": layer,
                "lda_accuracy": sa.get("lda_classification_accuracy"),
                "lda_f1": sa.get("lda_f1_score"),
                "lda_balanced_accuracy": sa.get("lda_balanced_accuracy"),
                "silhouette": sa.get("silhouette_score"),
                "mean_angle": mean_angle,
                "n_success": len(sa["success_concepts"]),
                "n_failure": len(sa["failure_concepts"]),
            }
        )

    if not rows:
        return None

    layer_df = pd.DataFrame(rows)

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    if layer_df["lda_accuracy"].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            layer_df["layer"],
            layer_df["lda_accuracy"],
            "o-",
            linewidth=2,
            markersize=10,
        )
        ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="70%")
        ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.5, label="60%")
        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel("LDA Classification Accuracy", fontsize=14)
        ax.set_title("Layer-wise LDA accuracy", fontsize=16)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            plots_dir / "layer_wise_classification_accuracy.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    layer_df.to_csv(output_dir / "layer_wise_analysis.csv", index=False)
    print(f"  Layer-wise results saved to {output_dir / 'layer_wise_analysis.csv'}")
    return layer_df


# ---------------------------------------------------------------------------
# Main per-model analysis
# ---------------------------------------------------------------------------


def analyze_model(
    injection_model_dir: Path,
    output_dir: Path,
    model_name: str,
    split_metric: str = "adaptive",
    split_threshold: float = 0.2,
    layer_index: Optional[int] = None,
    strength: Optional[float] = None,
) -> bool:
    """Full analysis pipeline for a single model/config."""
    # Output directory
    if layer_index is not None and strength is not None:
        config_name = get_config_dir_name(layer_index, strength)
        model_out = output_dir / model_name / config_name
        config_dir = injection_model_dir / config_name
    else:
        model_out = output_dir / model_name
        config_dir = None

    model_out.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print(f"ANALYZING: {model_name}")
    if layer_index is not None and strength is not None:
        print(f"  Config: layer={layer_index}, strength={strength}")
    print("=" * 80)

    # 1. Load experiment 01 (concept injection) results
    print("\n[1/6] Loading detection metrics...")
    concept_metrics = load_injection_results(injection_model_dir, config_dir=config_dir)
    if not concept_metrics:
        print("Error: No results found.")
        return False
    print(f"  {len(concept_metrics)} concepts")

    # 2. Load concept vectors
    print("\n[2/6] Loading concept vectors...")
    vectors_dir = injection_model_dir / "vectors"
    if not vectors_dir.exists():
        print(f"Error: Vectors directory not found: {vectors_dir}")
        return False

    if layer_index is not None:
        vectors = load_concept_vectors(vectors_dir, layer_index=layer_index)
    else:
        vectors = load_concept_vectors(vectors_dir)

    if not vectors:
        print("Error: No concept vectors found.")
        return False

    # Exclude concepts
    if args.exclude_concepts:
        excluded = set(args.exclude_concepts)
        concept_metrics = {k: v for k, v in concept_metrics.items() if k not in excluded}
        vectors = {k: v for k, v in vectors.items() if k not in excluded}
        print(f"  Excluded: {', '.join(sorted(excluded))}")
        print(f"  Remaining: {len(concept_metrics)} metrics, {len(vectors)} vectors")
        if not concept_metrics or not vectors:
            print("Error: No concepts after exclusion.")
            return False

    # 3. Geometric properties (PCA)
    print("\n[3/6] Computing PCA...")
    props_df, analysis = compute_geometric_properties(
        vectors,
        standardize=args.standardize,
        normalize=args.normalize,
        n_components=args.n_components,
    )

    # 4. Merge with detection metrics
    print("\n[4/6] Merging metrics...")
    combined_df = correlate_properties_with_detection(props_df, concept_metrics)
    if len(combined_df) == 0:
        print("Error: No concepts with both vectors and metrics.")
        return False
    print(f"  {len(combined_df)} concepts with complete data")
    combined_df.to_csv(model_out / "geometric_analysis.csv", index=False)

    # 5. LDA / Ridge / subspace analysis
    print("\n[5/6] Subspace analysis...")

    # Determine threshold and metric
    if split_metric == "adaptive":
        best_threshold, best_metric = sweep_subspace_parameters(combined_df, vectors)
    else:
        best_metric = split_metric
        valid_concepts = set(combined_df["concept"].values) & set(vectors.keys())
        df_valid = combined_df[combined_df["concept"].isin(valid_concepts)]
        success_mask = df_valid[best_metric] > split_threshold
        if success_mask.sum() == 0 or (~success_mask).sum() == 0:
            best_threshold = df_valid[best_metric].median()
        else:
            best_threshold = split_threshold

    metric_out = model_out / best_metric
    metric_out.mkdir(parents=True, exist_ok=True)

    # LDA
    subspace_analysis = compute_introspection_subspace_analysis(
        combined_df,
        vectors,
        threshold=best_threshold,
        metric=best_metric,
        allow_fallback=False,
    )

    if subspace_analysis:
        n_s = len(subspace_analysis["success_concepts"])
        n_f = len(subspace_analysis["failure_concepts"])
        print(f"  Split: {n_s} success / {n_f} failure")

        # Save LDA direction
        if subspace_analysis.get("lda_direction") is not None:
            torch.save(
                torch.tensor(subspace_analysis["lda_direction"]),
                metric_out / "introspection_direction_lda.pt",
            )
        if subspace_analysis.get("mean_difference") is not None:
            torch.save(
                torch.tensor(subspace_analysis["mean_difference"]),
                metric_out / "introspection_direction_mean_diff.pt",
            )
        # Also save the success/failure group centroids so downstream scripts
        # (e.g., 04e synthetic threshold test) can sweep v = mu_fail + alpha*d_diff.
        if subspace_analysis.get("mean_success") is not None:
            torch.save(
                torch.tensor(subspace_analysis["mean_success"]),
                metric_out / "mean_success.pt",
            )
        if subspace_analysis.get("mean_failure") is not None:
            torch.save(
                torch.tensor(subspace_analysis["mean_failure"]),
                metric_out / "mean_failure.pt",
            )

        # Save subspace analysis JSON
        sa_summary = {
            "success_concepts": subspace_analysis["success_concepts"],
            "failure_concepts": subspace_analysis["failure_concepts"],
            "lda_classification_accuracy": subspace_analysis.get(
                "lda_classification_accuracy"
            ),
            "lda_f1_score": subspace_analysis.get("lda_f1_score"),
            "lda_f1_std": subspace_analysis.get("lda_f1_std"),
            "lda_balanced_accuracy": subspace_analysis.get("lda_balanced_accuracy"),
            "silhouette_score": subspace_analysis.get("silhouette_score"),
            "subspace_angles": subspace_analysis.get("subspace_angles"),
            "threshold": subspace_analysis.get("threshold"),
            "metric_used": subspace_analysis.get("metric_used"),
        }
        with open(metric_out / "subspace_analysis.json", "w") as f:
            json.dump(convert_numpy_types(sa_summary), f, indent=2)

        # Ridge regression
        print("\n  Ridge regression...")
        all_concepts_with_data = [c for c in combined_df["concept"].values if c in vectors]
        if len(all_concepts_with_data) > 5:
            all_vecs = np.array(
                [vectors[c].float().numpy() for c in all_concepts_with_data]
            )
            all_rates = combined_df[
                combined_df["concept"].isin(all_concepts_with_data)
            ][best_metric].values

            ridge_results = compute_ridge_regression(
                all_vecs, all_rates, all_concepts_with_data, metric_out, best_metric
            )
        else:
            print("  Not enough concepts for ridge regression")
    else:
        print("  Subspace analysis failed or insufficient data")
        subspace_analysis = {}

    # Layer-wise analysis (only when not running per-config)
    if layer_index is None:
        print("\n  Layer-wise analysis...")
        analyze_layer_wise(
            injection_model_dir,
            vectors_dir,
            combined_df,
            concept_metrics,
            metric_out,
            threshold=best_threshold,
            metric=best_metric,
        )

    # 6. Visualizations
    print("\n[6/6] Creating plots...")
    create_pca_visualization(combined_df, analysis, model_out)
    if subspace_analysis:
        create_lda_visualization(subspace_analysis, metric_out)

    # Summary
    summary_path = model_out / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write("CONCEPT VECTOR GEOMETRY ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Concepts analyzed: {len(combined_df)}\n\n")

        f.write("DETECTION METRICS:\n")
        f.write(f"  Mean detection rate: {combined_df['detection_rate'].mean():.2%}\n")
        f.write(f"  Mean combined rate:  {combined_df['combined_rate'].mean():.2%}\n\n")

        pca = analysis["pca"]
        f.write("PCA:\n")
        f.write(
            f"  L2-normalized: {analysis.get('normalized', False)}\n"
        )
        for i, v in enumerate(pca.explained_variance_ratio_[:5], 1):
            f.write(f"  PC{i}: {v:.2%}\n")
        f.write(f"  Total (5 PCs): {pca.explained_variance_ratio_[:5].sum():.2%}\n\n")

        if subspace_analysis:
            f.write("LDA PARTITION:\n")
            f.write(f"  Metric: {subspace_analysis.get('metric_used')}\n")
            f.write(f"  Threshold: {subspace_analysis.get('threshold')}\n")
            n_s = len(subspace_analysis.get("success_concepts", []))
            n_f = len(subspace_analysis.get("failure_concepts", []))
            f.write(f"  Success: {n_s}, Failure: {n_f}\n")
            acc = subspace_analysis.get("lda_balanced_accuracy")
            if acc is not None:
                f.write(f"  Balanced accuracy: {acc:.2%}\n")
            sil = subspace_analysis.get("silhouette_score")
            if sil is not None:
                f.write(f"  Silhouette: {sil:.3f}\n")

    print(f"\nDone. Results in {model_out}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    injection_base = Path(args.injection_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 4b: CONCEPT VECTOR GEOMETRY ANALYSIS")
    print("=" * 80)

    if not injection_base.exists():
        print(f"Error: {injection_base} not found")
        return

    # Find model directories
    model_dirs = []
    for item in injection_base.iterdir():
        if item.is_dir() and item.name != "shared":
            if any(item.glob("layer_*_strength_*")) or (item / "vectors").exists():
                model_dirs.append(item)

    if not model_dirs:
        print(f"No model directories found in {injection_base}")
        return

    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
        if not model_dirs:
            print(f"Model '{args.model}' not found")
            return

    print(f"Models: {[d.name for d in model_dirs]}\n")

    successful = 0
    total = 0

    for model_dir in tqdm(model_dirs, desc="Models"):
        model_name = model_dir.name
        available = discover_configs(model_dir)

        if not available:
            print(f"No configs for {model_name}, skipping")
            continue

        # Filter configs
        target_layers = set(args.layers) if args.layers else set(c[0] for c in available)
        target_strengths = (
            set(args.strengths) if args.strengths else set(c[1] for c in available)
        )
        configs = [
            (l, s) for l, s in available if l in target_layers and s in target_strengths
        ]

        if not configs:
            print(f"No matching configs for {model_name}")
            continue

        for layer_idx, strength in configs:
            total += 1
            try:
                ok = analyze_model(
                    model_dir,
                    output_dir,
                    model_name,
                    split_metric=args.split_metric,
                    split_threshold=args.split_threshold,
                    layer_index=layer_idx,
                    strength=strength,
                )
                if ok:
                    successful += 1
            except Exception as e:
                print(f"Error: {model_name}/{get_config_dir_name(layer_idx, strength)}: {e}")
                import traceback

                traceback.print_exc()

        # Shared layer-wise analysis
        if len(configs) > 1:
            try:
                shared_dir = output_dir / model_name / "shared"
                shared_dir.mkdir(exist_ok=True, parents=True)
                agg = aggregate_concept_metrics_across_configs(model_dir, configs)
                agg_df = pd.DataFrame(
                    [{"concept": c, **m} for c, m in agg.items()]
                )
                vdir = model_dir / "vectors"
                if vdir.exists():
                    analyze_layer_wise(
                        model_dir,
                        vdir,
                        agg_df,
                        agg,
                        shared_dir,
                        threshold=args.split_threshold,
                        metric=(
                            args.split_metric
                            if args.split_metric != "adaptive"
                            else "combined_rate"
                        ),
                    )
            except Exception as e:
                print(f"Shared layer-wise analysis error: {e}")

    print("\n" + "=" * 80)
    print(f"Done: {successful}/{total} configs analyzed successfully")
    print(f"Results: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
