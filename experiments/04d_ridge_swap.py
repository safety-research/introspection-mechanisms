#!/usr/bin/env python3
"""
Experiment 04d: Ridge Projection vs. Residual Decomposition (Swap Experiments)

Determines whether introspection behavior is driven by projection onto the
ridge regression direction (which predicts detection rate) or by the residual
component orthogonal to that direction.

Decomposition:
    v[c] = proj_ridge[c] + delta_perp[c]

    where:
    - proj_ridge[c] = (v[c] . w_ridge) * w_ridge  (projection onto ridge direction)
    - delta_perp[c] = v[c] - proj_ridge[c]  (residual orthogonal to ridge direction)
    - w_ridge is loaded from 04b_vector_decomposition primary_axis.pt

Swap conditions tested on BOTH success and failure concepts:

ON SUCCESS CONCEPTS (high detection rate, testing if swaps HURT detection):
- baseline_succ: Original vector = ridge_score_succ[c] * w + delta_perp_succ[c]
- proj_swap_on_succ: (ridge_score_succ[c] + ridge_shift) * w + delta_perp_succ[c]
    where ridge_shift = mean_ridge_fail - mean_ridge_succ (SHIFT to failure)
- delta_swap_random_on_succ: ridge_score_succ[c] * w + delta_perp_fail[random]
- delta_swap_nearest_on_succ: ridge_score_succ[c] * w + delta_perp_fail[nearest]

ON FAILURE CONCEPTS (low detection rate, testing if swaps BOOST detection):
- baseline_fail: Original vector = ridge_score_fail[c] * w + delta_perp_fail[c]
- proj_swap_on_fail: (ridge_score_fail[c] + ridge_shift) * w + delta_perp_fail[c]
    where ridge_shift = mean_ridge_succ - mean_ridge_fail (SHIFT to success)
- delta_swap_random_on_fail: ridge_score_fail[c] * w + delta_perp_succ[random]
- delta_swap_nearest_on_fail: ridge_score_fail[c] * w + delta_perp_succ[nearest]

Key predictions:
- proj_swap (SHIFT ridge score): If detection changes -> ridge direction matters
- delta_swap (keep ridge, swap residual): If detection changes -> residual matters

Paper references:
- Section 4.1: Mean-diff swap experiments
- Appendix C: Ridge swap experiments (this script)
- Figures: ridge-swap.pdf

Dependencies:
- 04b_vector_decomposition -> Ridge direction in primary_axis.pt
- 04b_vector_geometry.py -> success/failure partition in subspace_analysis.json
- 02b_steering_500_concepts -> concept vectors + baseline steering results

Usage:
    python 04d_ridge_swap.py -m gemma3_27b
    python 04d_ridge_swap.py -m gemma3_27b --layer 37
    python 04d_ridge_swap.py -m gemma3_27b --trials-per-concept 20
    python 04d_ridge_swap.py -m gemma3_27b -o  # Overwrite and restart
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

# Import local utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model_utils import load_model, ModelWrapper
from steering_utils import run_steered_introspection_test_batch, check_concept_mentioned
from eval_utils import LLMJudge, batch_evaluate

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemma3_27b"
DEFAULT_GEOMETRY_DIR = "analysis/04b_vector_geometry"
DEFAULT_GEOMETRY_DECOMP_DIR = "analysis/04b_vector_decomposition"
DEFAULT_STEERING_DIR = "analysis/02b_steering_500_concepts"
DEFAULT_OUTPUT_DIR = "analysis/04d_ridge_swap"
DEFAULT_THRESHOLD = 0.2
DEFAULT_N_TRIAL_NUMBERS = 10
DEFAULT_SAMPLES_PER_TRIAL = 10
DEFAULT_TRIALS_PER_CONCEPT = DEFAULT_N_TRIAL_NUMBERS * DEFAULT_SAMPLES_PER_TRIAL
DEFAULT_LAYER = 37
DEFAULT_STRENGTH = 4.0
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_PLOT_UPDATE_INTERVAL = 5

# ---------------------------------------------------------------------------
# Condition names
# ---------------------------------------------------------------------------
# Success concepts (testing if swaps HURT detection)
COND_BASELINE_SUCC = "baseline_succ"
COND_PROJ_SWAP_SUCC = "proj_swap_on_succ"
COND_DELTA_RANDOM_SUCC = "delta_swap_random_on_succ"
COND_DELTA_NEAREST_SUCC = "delta_swap_nearest_on_succ"

# Failure concepts (testing if swaps BOOST detection)
COND_BASELINE_FAIL = "baseline_fail"
COND_PROJ_SWAP_FAIL = "proj_swap_on_fail"
COND_DELTA_RANDOM_FAIL = "delta_swap_random_on_fail"
COND_DELTA_NEAREST_FAIL = "delta_swap_nearest_on_fail"

SUCCESS_CONDITIONS = [COND_PROJ_SWAP_SUCC, COND_DELTA_RANDOM_SUCC, COND_DELTA_NEAREST_SUCC]
FAILURE_CONDITIONS = [COND_PROJ_SWAP_FAIL, COND_DELTA_RANDOM_FAIL, COND_DELTA_NEAREST_FAIL]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def deterministic_hash(s: str) -> int:
    """Deterministic hash (consistent across Python runs, unlike built-in hash)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def normalize_to_target_norm(vector: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Normalize *vector* so that its L2 norm equals *target_norm*."""
    current_norm = vector.norm()
    if current_norm < 1e-8:
        return vector
    return vector * (target_norm / current_norm)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_concept_vectors(vectors_dir: Path, concepts: List[str]) -> Dict[str, torch.Tensor]:
    """Load concept vectors from disk, returning {concept: tensor}."""
    vectors = {}
    missing = []
    for concept in concepts:
        vector_path = vectors_dir / f"{concept}.pt"
        if vector_path.exists():
            vectors[concept] = torch.load(vector_path, weights_only=True)
        else:
            missing.append(concept)
    if missing:
        print(f"Warning: Missing vectors for {len(missing)} concepts: {missing[:5]}...")
    return vectors


def load_geometry_partition(
    geometry_dir: Path, model_name: str,
    layer_idx: int = None, strength: float = None,
) -> Tuple[List[str], List[str], dict]:
    """Load success/failure partition from experiment 04b (vector geometry) subspace_analysis.json."""
    if layer_idx is not None and strength is not None:
        config_folder = f"layer_{layer_idx}_strength_{strength}"
        subspace_path = geometry_dir / model_name / config_folder / "detection_rate" / "subspace_analysis.json"
    else:
        subspace_path = geometry_dir / model_name / "subspace_analysis.json"

    if not subspace_path.exists():
        raise FileNotFoundError(f"Could not find experiment 04b (vector geometry) subspace analysis at {subspace_path}")

    with open(subspace_path, 'r') as f:
        data = json.load(f)

    success_concepts = data.get("success_concepts", [])
    failure_concepts = data.get("failure_concepts", [])
    metadata = {
        "threshold": data.get("threshold"),
        "metric_used": data.get("metric_used"),
        "source": str(subspace_path),
    }
    return success_concepts, failure_concepts, metadata


def load_steering_baseline_per_concept(
    steering_dir: Path, model_name: str,
    layer_idx: int = None, strength: float = None,
) -> Dict[str, Dict]:
    """Load per-concept baseline results from experiment 02 (steering evaluation) results.json."""
    if layer_idx is not None and strength is not None:
        config_folder = f"layer_{layer_idx}_strength_{strength}"
        results_path = steering_dir / model_name / config_folder / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Could not find experiment 02 (steering evaluation) results at {results_path}")
    else:
        matches = list(steering_dir.glob(f"{model_name}/layer_*/results.json"))
        if not matches:
            raise FileNotFoundError(f"Could not find experiment 02 (steering evaluation) results for {model_name}")
        results_path = matches[0]

    with open(results_path, 'r') as f:
        data = json.load(f)

    concept_metrics: Dict[str, Dict[str, list]] = {}
    for result in data.get("results", []):
        concept = result.get("concept")
        if not concept:
            continue
        if result.get("trial_type", "injection") != "injection":
            continue
        if concept not in concept_metrics:
            concept_metrics[concept] = {"detected": [], "identified": []}

        evals = result.get("evaluations", {})
        detected = bool(evals.get("claims_detection", {}).get("grade", 0))
        identified = bool(evals.get("correct_concept_identification", {}).get("grade", 0))
        concept_metrics[concept]["detected"].append(detected)
        concept_metrics[concept]["identified"].append(identified)

    per_concept: Dict[str, Dict] = {}
    for concept, cdata in concept_metrics.items():
        n_trials = len(cdata["detected"])
        if n_trials == 0:
            continue
        n_detected = sum(cdata["detected"])
        n_identified = sum(d and i for d, i in zip(cdata["detected"], cdata["identified"]))
        detection_rate = n_detected / n_trials
        cond_id_rate = n_identified / n_detected if n_detected > 0 else 0
        introspection_rate = n_identified / n_trials
        per_concept[concept] = {
            "detection_rate": detection_rate,
            "cond_id_rate": cond_id_rate,
            "introspection_rate": introspection_rate,
            "n_trials": n_trials,
        }

    print(f"  Loaded baseline for {len(per_concept)} concepts from {results_path}")
    return per_concept


def load_ridge_direction(
    geometry_decomp_dir: Path, model_name: str,
    layer: int, strength: float, metric: str = "detection_rate",
) -> torch.Tensor:
    """Load the unit-normalized ridge regression direction from 04b_vector_decomposition."""
    ridge_path = (
        geometry_decomp_dir / model_name
        / f"layer_{layer}_strength_{strength}" / metric / "primary_axis.pt"
    )
    if not ridge_path.exists():
        raise FileNotFoundError(
            f"Ridge direction not found at {ridge_path}. "
            f"Run 04b_vector_decomposition.py first."
        )
    w_ridge = torch.load(ridge_path, map_location="cpu", weights_only=True)
    if isinstance(w_ridge, dict):
        w_ridge = w_ridge.get("direction", w_ridge.get("primary_axis"))
    w_ridge = w_ridge.float()
    w_ridge = w_ridge / w_ridge.norm()
    return w_ridge


# ---------------------------------------------------------------------------
# Ridge decomposition
# ---------------------------------------------------------------------------

def compute_ridge_decomposition(
    vectors: Dict[str, torch.Tensor],
    w_ridge: torch.Tensor,
    success_concepts: List[str],
    failure_concepts: List[str],
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor],
           Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Decompose concept vectors: v[c] = ridge_score[c] * w_ridge + delta_perp[c].

    Returns (w_ridge, ridge_scores_success, deltas_success,
             ridge_scores_failure, deltas_failure).
    """
    success_with_vectors = [c for c in success_concepts if c in vectors]
    failure_with_vectors = [c for c in failure_concepts if c in vectors]
    if not success_with_vectors:
        raise ValueError("No success concepts have vectors")
    if not failure_with_vectors:
        raise ValueError("No failure concepts have vectors")

    w_ridge_float = w_ridge.float()

    ridge_scores_success: Dict[str, float] = {}
    deltas_perp_success: Dict[str, torch.Tensor] = {}
    for c in success_with_vectors:
        v = vectors[c].float()
        score = torch.dot(v, w_ridge_float).item()
        ridge_scores_success[c] = score
        deltas_perp_success[c] = v - score * w_ridge_float

    ridge_scores_failure: Dict[str, float] = {}
    deltas_perp_failure: Dict[str, torch.Tensor] = {}
    for c in failure_with_vectors:
        v = vectors[c].float()
        score = torch.dot(v, w_ridge_float).item()
        ridge_scores_failure[c] = score
        deltas_perp_failure[c] = v - score * w_ridge_float

    if verbose:
        sample_c = success_with_vectors[0]
        v_orig = vectors[sample_c].float()
        v_recon = ridge_scores_success[sample_c] * w_ridge_float + deltas_perp_success[sample_c]
        print(f"  Reconstruction error ('{sample_c}'): {(v_orig - v_recon).abs().max().item():.2e}")
        orth = torch.dot(deltas_perp_success[sample_c], w_ridge_float).abs().item()
        print(f"  Delta orthogonality to ridge: {orth:.2e}")
        succ_scores = list(ridge_scores_success.values())
        fail_scores = list(ridge_scores_failure.values())
        print(f"  Ridge scores - success: mean={np.mean(succ_scores):.4f}, std={np.std(succ_scores):.4f}")
        print(f"  Ridge scores - failure: mean={np.mean(fail_scores):.4f}, std={np.std(fail_scores):.4f}")
        print(f"  Ridge score separation: {np.mean(succ_scores) - np.mean(fail_scores):.4f}")

    return w_ridge, ridge_scores_success, deltas_perp_success, ridge_scores_failure, deltas_perp_failure


# ---------------------------------------------------------------------------
# Swap pairings
# ---------------------------------------------------------------------------

def create_swap_pairings(
    source_concepts: List[str],
    target_concepts: List[str],
    vectors: Dict[str, torch.Tensor],
    method: str = "nearest_neighbor",
    seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Create pairings between source and target groups for delta-swap.

    Args:
        source_concepts: Concepts whose ridge score is kept.
        target_concepts: Concepts whose delta_perp is used.
        method: 'nearest_neighbor' or 'random'.

    Returns:
        (pairings, similarities) dicts keyed by source concept.
    """
    source_with_vecs = [c for c in source_concepts if c in vectors]
    target_with_vecs = [c for c in target_concepts if c in vectors]

    pairings: Dict[str, str] = {}
    similarities: Dict[str, float] = {}

    if method == "random":
        rng = random.Random(seed)
        for src in source_with_vecs:
            paired = rng.choice(target_with_vecs)
            pairings[src] = paired
            s_vec = vectors[src].float()
            t_vec = vectors[paired].float()
            sim = torch.nn.functional.cosine_similarity(
                s_vec.unsqueeze(0), t_vec.unsqueeze(0)
            ).item()
            similarities[src] = sim

    elif method == "nearest_neighbor":
        src_vecs = torch.stack([vectors[c].float() for c in source_with_vecs])
        tgt_vecs = torch.stack([vectors[c].float() for c in target_with_vecs])
        src_norm = src_vecs / src_vecs.norm(dim=1, keepdim=True)
        tgt_norm = tgt_vecs / tgt_vecs.norm(dim=1, keepdim=True)
        sim_matrix = torch.mm(src_norm, tgt_norm.t())
        max_sims, nearest_idx = sim_matrix.max(dim=1)
        for i, src in enumerate(source_with_vecs):
            pairings[src] = target_with_vecs[nearest_idx[i].item()]
            similarities[src] = max_sims[i].item()
        sim_vals = list(similarities.values())
        print(f"  Nearest-neighbor pairing: mean sim={np.mean(sim_vals):.4f}, "
              f"min={np.min(sim_vals):.4f}, max={np.max(sim_vals):.4f}")
    else:
        raise ValueError(f"Unknown pairing method: {method}")

    return pairings, similarities


# ---------------------------------------------------------------------------
# Steering experiment runner
# ---------------------------------------------------------------------------

def run_steering_experiment(
    model: ModelWrapper,
    concept: str,
    layer_idx: int,
    strength: float,
    max_tokens: int,
    temperature: float,
    condition: str,
    concept_type: str,
    steering_vector: torch.Tensor = None,
    steering_vectors: List[torch.Tensor] = None,
    n_trials: int = None,
    n_trial_numbers: int = None,
    samples_per_trial: int = None,
    paired_concept: Optional[str] = None,
    paired_concepts: List[str] = None,
) -> List[Dict]:
    """
    Run steering experiment for a single concept under one condition.

    Provide either *steering_vector* (same for every trial) or *steering_vectors*
    (one per trial for per-trial randomization).
    """
    use_multi = steering_vectors is not None
    if not use_multi and steering_vector is None:
        raise ValueError("Must provide either steering_vector or steering_vectors")

    # Build trial number sequence
    if n_trial_numbers is not None and samples_per_trial is not None:
        trial_numbers: List[int] = []
        for t in range(1, n_trial_numbers + 1):
            trial_numbers.extend([t] * samples_per_trial)
    else:
        if n_trials is None:
            raise ValueError("n_trials required when n_trial_numbers/samples_per_trial not set")
        trial_numbers = list(range(1, n_trials + 1))

    if use_multi and len(steering_vectors) != len(trial_numbers):
        raise ValueError("steering_vectors length must match number of trials")

    # Run batch generation
    gen_kwargs = dict(
        model=model,
        concept_word=concept,
        layer_idx=layer_idx,
        strength=strength,
        trial_numbers=trial_numbers,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    if use_multi:
        gen_kwargs["steering_vectors"] = steering_vectors
    else:
        gen_kwargs["steering_vector"] = steering_vector
    responses = run_steered_introspection_test_batch(**gen_kwargs)

    # Build result dicts
    results = []
    trial_sample_counts: Dict[int, int] = {}
    for i, (trial_num, response) in enumerate(zip(trial_numbers, responses)):
        sample_idx = trial_sample_counts.get(trial_num, 0)
        trial_sample_counts[trial_num] = sample_idx + 1

        result: Dict = {
            "concept": concept,
            "concept_type": concept_type,
            "trial": trial_num,
            "sample_idx": sample_idx,
            "response": response,
            "condition": condition,
            "layer": layer_idx,
            "strength": strength,
            "trial_type": "injection",
            "detected": check_concept_mentioned(response, concept),
        }
        if use_multi and paired_concepts is not None:
            result["paired_concept"] = paired_concepts[i]
        elif paired_concept is not None:
            result["paired_concept"] = paired_concept
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_existing_results(output_dir: Path) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    """Load existing results and return (results_list, {concept: {completed_conditions}})."""
    all_results_path = output_dir / "all_results.json"
    if not all_results_path.exists():
        return [], {}

    try:
        with open(all_results_path, 'r') as f:
            data = json.load(f)
        results = data.get("results", [])

        concept_conditions: Dict[str, Set[str]] = {}
        for r in results:
            concept = r.get("concept")
            condition = r.get("condition")
            if concept and condition:
                concept_conditions.setdefault(concept, set()).add(condition)

        print(f"  Found {len(results)} existing results for {len(concept_conditions)} concepts")
        return results, concept_conditions
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: Could not load existing results: {e}")
        return [], {}


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_condition_metrics(
    results: List[Dict],
    baseline_per_concept: Dict[str, Dict],
    success_concepts: List[str],
    failure_concepts: List[str],
) -> Dict[str, Dict]:
    """Compute aggregate detection/identification metrics for each condition."""
    metrics: Dict[str, Dict] = {}

    def _baseline_metrics(concepts: List[str]) -> Dict:
        det_rates = [baseline_per_concept[c]["detection_rate"]
                     for c in concepts if c in baseline_per_concept]
        intro_rates = [baseline_per_concept[c]["introspection_rate"]
                       for c in concepts if c in baseline_per_concept]
        if not det_rates:
            return {"detection_rate": 0.0, "conditional_identification_rate": 0.0,
                    "introspection_rate": 0.0, "n_concepts": 0}
        avg_det = np.mean(det_rates)
        avg_intro = np.mean(intro_rates)
        return {
            "detection_rate": avg_det,
            "conditional_identification_rate": avg_intro / avg_det if avg_det > 0 else 0,
            "introspection_rate": avg_intro,
            "n_concepts": len(det_rates),
        }

    metrics[COND_BASELINE_SUCC] = _baseline_metrics(success_concepts)
    metrics[COND_BASELINE_FAIL] = _baseline_metrics(failure_concepts)

    def _experiment_metrics(condition_name: str) -> Dict:
        cond_results = [r for r in results if r.get("condition") == condition_name]
        if not cond_results:
            return {"detection_rate": 0.0, "conditional_identification_rate": 0.0,
                    "introspection_rate": 0.0, "n_trials": 0}
        n_detected = n_identified = 0
        for r in cond_results:
            evals = r.get("evaluations", {})
            det = bool(evals.get("claims_detection", {}).get("grade", 0))
            ident = bool(evals.get("correct_concept_identification", {}).get("grade", 0))
            if det:
                n_detected += 1
            if det and ident:
                n_identified += 1
        n = len(cond_results)
        det_rate = n_detected / n
        return {
            "detection_rate": det_rate,
            "conditional_identification_rate": n_identified / n_detected if n_detected else 0,
            "introspection_rate": n_identified / n,
            "n_trials": n,
            "n_detected": n_detected,
            "n_correctly_identified": n_identified,
        }

    for cond in SUCCESS_CONDITIONS + FAILURE_CONDITIONS:
        metrics[cond] = _experiment_metrics(cond)

    return metrics


def compute_per_concept_results(
    results: List[Dict],
    pairings_succ_random: Dict[str, str],
    pairings_succ_nearest: Dict[str, str],
    pairings_fail_random: Dict[str, str],
    pairings_fail_nearest: Dict[str, str],
    similarities_succ_random: Dict[str, float],
    similarities_succ_nearest: Dict[str, float],
    similarities_fail_random: Dict[str, float],
    similarities_fail_nearest: Dict[str, float],
    baseline_per_concept: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Compute per-concept detection/identification metrics for each condition."""
    all_conditions = SUCCESS_CONDITIONS + FAILURE_CONDITIONS

    by_concept: Dict[str, Dict[str, list]] = {}
    for r in results:
        concept = r.get("concept")
        condition = r.get("condition")
        if concept not in by_concept:
            by_concept[concept] = {c: [] for c in all_conditions}
        if condition in by_concept[concept]:
            by_concept[concept][condition].append(r)

    for concept in baseline_per_concept:
        if concept not in by_concept:
            by_concept[concept] = {c: [] for c in all_conditions}

    def _metrics_from_results(cond_results: list) -> Optional[Dict]:
        if not cond_results:
            return None
        n_det = n_id = 0
        for r in cond_results:
            evals = r.get("evaluations", {})
            det = bool(evals.get("claims_detection", {}).get("grade", 0))
            ident = bool(evals.get("correct_concept_identification", {}).get("grade", 0))
            if det:
                n_det += 1
            if det and ident:
                n_id += 1
        n = len(cond_results)
        return {
            "detection_rate": n_det / n,
            "cond_id_rate": n_id / n_det if n_det else 0,
            "introspection_rate": n_id / n,
        }

    per_concept: Dict[str, Dict] = {}
    for concept, condition_results in by_concept.items():
        entry: Dict = {}

        # Determine concept type
        is_success = concept in pairings_succ_random or concept in pairings_succ_nearest
        is_failure = concept in pairings_fail_random or concept in pairings_fail_nearest
        for cond_results in condition_results.values():
            for r in cond_results:
                ct = r.get("concept_type")
                if ct == "success":
                    is_success = True
                elif ct == "failure":
                    is_failure = True
                break
            if is_success or is_failure:
                break

        if is_success:
            entry["concept_type"] = "success"
        elif is_failure:
            entry["concept_type"] = "failure"

        # Baseline
        if concept in baseline_per_concept:
            bl_key = COND_BASELINE_SUCC if is_success else COND_BASELINE_FAIL
            entry[bl_key] = {
                "detection_rate": baseline_per_concept[concept]["detection_rate"],
                "cond_id_rate": baseline_per_concept[concept]["cond_id_rate"],
                "introspection_rate": baseline_per_concept[concept]["introspection_rate"],
            }

        # Experimental conditions
        for cond, cond_results in condition_results.items():
            m = _metrics_from_results(cond_results)
            if m is not None:
                entry[cond] = m

        # Pairing metadata
        _pairing_info = [
            (pairings_succ_random, similarities_succ_random, COND_DELTA_RANDOM_SUCC),
            (pairings_succ_nearest, similarities_succ_nearest, COND_DELTA_NEAREST_SUCC),
            (pairings_fail_random, similarities_fail_random, COND_DELTA_RANDOM_FAIL),
            (pairings_fail_nearest, similarities_fail_nearest, COND_DELTA_NEAREST_FAIL),
        ]
        for pairings, sims, cond_key in _pairing_info:
            if concept in pairings:
                entry.setdefault(cond_key, {})["paired_concept"] = pairings[concept]
                if concept in sims:
                    entry[cond_key]["pairing_cosine_similarity"] = sims[concept]

        per_concept[concept] = entry

    return per_concept


def compute_statistical_tests(per_concept_results: Dict[str, Dict]) -> Dict:
    """Compute paired t-tests and Wilcoxon tests between baseline and swap conditions."""
    results: Dict = {}

    def _paired_stats(baseline: np.ndarray, swap: np.ndarray, suffix: str) -> Dict:
        if len(baseline) < 2:
            return {"error": "Insufficient data", "n_concepts": len(baseline)}
        t_stat, p_value = stats.ttest_rel(baseline, swap)
        try:
            w_stat, w_p = stats.wilcoxon(baseline, swap)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        diff = baseline - swap
        std = np.std(diff, ddof=1)
        effect_size = np.mean(diff) / std if std > 0 else 0
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else None,
            "effect_size": float(effect_size),
            "mean_baseline": float(np.mean(baseline)),
            f"mean_{suffix}": float(np.mean(swap)),
            "mean_difference": float(np.mean(diff)),
            "n_concepts": len(baseline),
        }

    def _extract(bl_cond: str, sw_cond: str):
        bl_rates, sw_rates = [], []
        for concept, conditions in per_concept_results.items():
            if bl_cond in conditions and sw_cond in conditions:
                bl = conditions[bl_cond].get("detection_rate")
                sw = conditions[sw_cond].get("detection_rate")
                if bl is not None and sw is not None:
                    bl_rates.append(bl)
                    sw_rates.append(sw)
        return np.array(bl_rates), np.array(sw_rates)

    # Success concept tests
    for cond in SUCCESS_CONDITIONS:
        bl, sw = _extract(COND_BASELINE_SUCC, cond)
        results[f"{COND_BASELINE_SUCC}_vs_{cond}"] = _paired_stats(bl, sw, "swap")

    # Failure concept tests
    for cond in FAILURE_CONDITIONS:
        bl, sw = _extract(COND_BASELINE_FAIL, cond)
        results[f"{COND_BASELINE_FAIL}_vs_{cond}"] = _paired_stats(bl, sw, "swap")

    # Random vs nearest (success)
    r_rates, n_rates = [], []
    for concept, conditions in per_concept_results.items():
        if COND_DELTA_RANDOM_SUCC in conditions and COND_DELTA_NEAREST_SUCC in conditions:
            dr = conditions[COND_DELTA_RANDOM_SUCC].get("detection_rate")
            dn = conditions[COND_DELTA_NEAREST_SUCC].get("detection_rate")
            if dr is not None and dn is not None:
                r_rates.append(dr)
                n_rates.append(dn)
    results[f"{COND_DELTA_RANDOM_SUCC}_vs_{COND_DELTA_NEAREST_SUCC}"] = (
        _paired_stats(np.array(r_rates), np.array(n_rates), "nearest")
        if len(r_rates) > 1
        else {"error": "Insufficient data", "n_concepts": 0}
    )

    return results


def determine_interpretation(
    metrics: Dict[str, Dict],
    stat_results: Dict,
    significance_threshold: float = 0.05,
    effect_threshold: float = 0.1,
) -> str:
    """Return a human-readable interpretation string based on statistical tests."""
    parts = []

    def _get_p(key: str) -> float:
        p = stat_results.get(key, {}).get("p_value", 1.0)
        return 1.0 if (p is None or np.isnan(p)) else p

    # Success concepts
    bl_s = metrics.get(COND_BASELINE_SUCC, {}).get("detection_rate", 0)
    ms_s = metrics.get(COND_PROJ_SWAP_SUCC, {}).get("detection_rate", 0)
    dr_s = metrics.get(COND_DELTA_RANDOM_SUCC, {}).get("detection_rate", 0)
    dn_s = metrics.get(COND_DELTA_NEAREST_SUCC, {}).get("detection_rate", 0)

    ms_sig = (_get_p(f"{COND_BASELINE_SUCC}_vs_{COND_PROJ_SWAP_SUCC}") < significance_threshold
              and bl_s - ms_s > effect_threshold)
    dr_sig = (_get_p(f"{COND_BASELINE_SUCC}_vs_{COND_DELTA_RANDOM_SUCC}") < significance_threshold
              and bl_s - dr_s > effect_threshold)
    dn_sig = (_get_p(f"{COND_BASELINE_SUCC}_vs_{COND_DELTA_NEAREST_SUCC}") < significance_threshold
              and bl_s - dn_s > effect_threshold)
    delta_s_sig = dr_sig or dn_sig

    if ms_sig and not delta_s_sig:
        parts.append("SUCCESS: ADDITIVE (proj-swap hurts, delta-swap doesn't)")
    elif delta_s_sig and not ms_sig:
        parts.append("SUCCESS: SUBSPACE (delta-swap hurts, proj-swap doesn't)")
    elif ms_sig and delta_s_sig:
        parts.append("SUCCESS: MIXED (both swaps hurt)")
    elif bl_s > 0:
        parts.append("SUCCESS: NEITHER_SIGNIFICANT")

    # Failure concepts
    bl_f = metrics.get(COND_BASELINE_FAIL, {}).get("detection_rate", 0)
    ms_f = metrics.get(COND_PROJ_SWAP_FAIL, {}).get("detection_rate", 0)
    dr_f = metrics.get(COND_DELTA_RANDOM_FAIL, {}).get("detection_rate", 0)
    dn_f = metrics.get(COND_DELTA_NEAREST_FAIL, {}).get("detection_rate", 0)

    ms_f_sig = (_get_p(f"{COND_BASELINE_FAIL}_vs_{COND_PROJ_SWAP_FAIL}") < significance_threshold
                and ms_f - bl_f > effect_threshold)
    dr_f_sig = (_get_p(f"{COND_BASELINE_FAIL}_vs_{COND_DELTA_RANDOM_FAIL}") < significance_threshold
                and dr_f - bl_f > effect_threshold)
    dn_f_sig = (_get_p(f"{COND_BASELINE_FAIL}_vs_{COND_DELTA_NEAREST_FAIL}") < significance_threshold
                and dn_f - bl_f > effect_threshold)
    delta_f_sig = dr_f_sig or dn_f_sig

    if metrics.get(COND_PROJ_SWAP_FAIL, {}).get("n_trials", 0) > 0:
        if ms_f_sig and not delta_f_sig:
            parts.append("FAILURE: ADDITIVE (proj-swap boosts, delta-swap doesn't)")
        elif delta_f_sig and not ms_f_sig:
            parts.append("FAILURE: SUBSPACE (delta-swap boosts, proj-swap doesn't)")
        elif ms_f_sig and delta_f_sig:
            parts.append("FAILURE: MIXED (both swaps boost)")
        elif bl_f >= 0:
            parts.append("FAILURE: NEITHER_SIGNIFICANT")

    return " | ".join(parts) if parts else "Insufficient data for interpretation"


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results_iterative(
    all_results: List[Dict],
    output_dir: Path,
    pairings_succ_random: Dict[str, str],
    pairings_succ_nearest: Dict[str, str],
    pairings_fail_random: Dict[str, str],
    pairings_fail_nearest: Dict[str, str],
    similarities_succ_random: Dict[str, float],
    similarities_succ_nearest: Dict[str, float],
    similarities_fail_random: Dict[str, float],
    similarities_fail_nearest: Dict[str, float],
    baseline_per_concept: Dict[str, Dict],
    success_concepts: List[str],
    failure_concepts: List[str],
    partition_metadata: dict,
    trials_per_concept: int,
    layer_idx: int,
    strength: float,
    seed: int,
):
    """Save all results, per-concept results, and aggregate metrics to disk."""
    # Raw results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump({"results": all_results, "last_updated": datetime.now().isoformat()}, f, indent=2)

    # Per-concept results
    per_concept_results = compute_per_concept_results(
        all_results,
        pairings_succ_random, pairings_succ_nearest,
        pairings_fail_random, pairings_fail_nearest,
        similarities_succ_random, similarities_succ_nearest,
        similarities_fail_random, similarities_fail_nearest,
        baseline_per_concept,
    )
    with open(output_dir / "per_concept_results.json", 'w') as f:
        json.dump(per_concept_results, f, indent=2)

    # Aggregate metrics
    metrics = compute_condition_metrics(all_results, baseline_per_concept, success_concepts, failure_concepts)
    stat_results = compute_statistical_tests(per_concept_results)
    interpretation = determine_interpretation(metrics, stat_results)

    final = {
        "n_success_concepts": len(success_concepts),
        "n_failure_concepts": len(failure_concepts),
        "partition": partition_metadata,
        "trials_per_concept": trials_per_concept,
        "layer": layer_idx,
        "strength": strength,
        "seed": seed,
        "aggregate": metrics,
        "statistical_tests": stat_results,
        "interpretation": interpretation,
        "last_updated": datetime.now().isoformat(),
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(final, f, indent=2)

    return metrics, per_concept_results, stat_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_plots(
    metrics: Dict[str, Dict],
    per_concept: Dict[str, Dict],
    stat_results: Dict,
    output_dir: Path,
):
    """Create bar-chart, change-comparison, scatter, and histogram plots."""
    plt.rcParams.update({'font.size': 12})

    # --- helpers ---
    def _extract_direction_data(baseline_key: str, swap_conditions: List[str]):
        bl_det, bl_cid = [], []
        sw_det = {c: [] for c in swap_conditions}
        sw_cid = {c: [] for c in swap_conditions}
        concepts_list = []
        for concept, cond_data in per_concept.items():
            if baseline_key not in cond_data:
                continue
            bd = cond_data[baseline_key].get("detection_rate")
            bc = cond_data[baseline_key].get("cond_id_rate", 0)
            if bd is None:
                continue
            if not any(c in cond_data for c in swap_conditions):
                continue
            bl_det.append(bd)
            bl_cid.append(bc)
            concepts_list.append(concept)
            for c in swap_conditions:
                sw_det[c].append(cond_data.get(c, {}).get("detection_rate", np.nan))
                sw_cid[c].append(cond_data.get(c, {}).get("cond_id_rate", np.nan))
        return (
            np.array(bl_det) if bl_det else np.array([]),
            np.array(bl_cid) if bl_cid else np.array([]),
            {k: np.array(v) for k, v in sw_det.items()},
            {k: np.array(v) for k, v in sw_cid.items()},
            concepts_list,
        )

    def _se(data):
        valid = data[~np.isnan(data)]
        return np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0

    # Extract data
    succ_bl_det, succ_bl_cid, succ_sw_det, succ_sw_cid, succ_concepts = (
        _extract_direction_data(COND_BASELINE_SUCC, SUCCESS_CONDITIONS))
    fail_bl_det, fail_bl_cid, fail_sw_det, fail_sw_cid, fail_concepts = (
        _extract_direction_data(COND_BASELINE_FAIL, FAILURE_CONDITIONS))

    # Changes
    succ_changes, succ_n = {}, {}
    for c in SUCCESS_CONDITIONS:
        if len(succ_bl_det) and len(succ_sw_det[c]):
            ch = succ_bl_det - succ_sw_det[c]
            succ_changes[c] = ch
            succ_n[c] = int(np.sum(~np.isnan(ch)))
        else:
            succ_changes[c] = np.array([])
            succ_n[c] = 0

    fail_changes, fail_n = {}, {}
    for c in FAILURE_CONDITIONS:
        if len(fail_bl_det) and len(fail_sw_det[c]):
            ch = fail_sw_det[c] - fail_bl_det
            fail_changes[c] = ch
            fail_n[c] = int(np.sum(~np.isnan(ch)))
        else:
            fail_changes[c] = np.array([])
            fail_n[c] = 0

    # Colors
    COLORS = {
        COND_BASELINE_SUCC: '#2ecc71', COND_PROJ_SWAP_SUCC: '#e74c3c',
        COND_DELTA_RANDOM_SUCC: '#3498db', COND_DELTA_NEAREST_SUCC: '#9b59b6',
        COND_BASELINE_FAIL: '#2ecc71', COND_PROJ_SWAP_FAIL: '#e74c3c',
        COND_DELTA_RANDOM_FAIL: '#3498db', COND_DELTA_NEAREST_FAIL: '#9b59b6',
    }
    swap_colors = ['#e74c3c', '#3498db', '#9b59b6']

    # Formulas for x-tick labels (ridge decomposition: v = r * w + delta_perp)
    LABELS_SUCC = {
        COND_BASELINE_SUCC: ("Baseline", "r_s*w + d_s"),
        COND_PROJ_SWAP_SUCC: ("Proj-swap", "(r_s+Dr)*w + d_s"),
        COND_DELTA_RANDOM_SUCC: ("Delta-swap\n(random)", "r_s*w + d_f[rand]"),
        COND_DELTA_NEAREST_SUCC: ("Delta-swap\n(nearest)", "r_s*w + d_f[near]"),
    }
    LABELS_FAIL = {
        COND_BASELINE_FAIL: ("Baseline", "r_f*w + d_f"),
        COND_PROJ_SWAP_FAIL: ("Proj-swap", "(r_f+Dr)*w + d_f"),
        COND_DELTA_RANDOM_FAIL: ("Delta-swap\n(random)", "r_f*w + d_s[rand]"),
        COND_DELTA_NEAREST_FAIL: ("Delta-swap\n(nearest)", "r_f*w + d_s[near]"),
    }

    # ========== PLOT 1: Detection rate comparison ==========
    def _has(cond):
        m = metrics.get(cond, {})
        return m.get("n_trials", 0) > 0 or m.get("n_concepts", 0) > 0

    succ_conds = [COND_BASELINE_SUCC] + [c for c in SUCCESS_CONDITIONS if _has(c)]
    fail_conds = [COND_BASELINE_FAIL] + [c for c in FAILURE_CONDITIONS if _has(c)]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, 4 * len(succ_conds)), 7))
    bar_w = 0.35

    for ax, conds, labels_map, bl_det, bl_cid, sw_det, sw_cid, title_word, title_color in [
        (axes[0], succ_conds, LABELS_SUCC, succ_bl_det, succ_bl_cid,
         succ_sw_det, succ_sw_cid, "Success", "#27ae60"),
        (axes[1], fail_conds, LABELS_FAIL, fail_bl_det, fail_bl_cid,
         fail_sw_det, fail_sw_cid, "Failure", "#e74c3c"),
    ]:
        x = np.arange(len(conds))
        det_vals = [metrics.get(c, {}).get("detection_rate", 0) for c in conds]
        cid_vals = [metrics.get(c, {}).get("conditional_identification_rate", 0) for c in conds]
        det_se = [_se(bl_det)] + [_se(sw_det.get(c, np.array([]))) for c in conds[1:]]
        cid_se = [_se(bl_cid)] + [_se(sw_cid.get(c, np.array([]))) for c in conds[1:]]
        colors = [COLORS[c] for c in conds]
        labels = [labels_map[c][0] for c in conds]
        formulas = [labels_map[c][1] for c in conds]

        bars1 = ax.bar(x - bar_w / 2, det_vals, bar_w, yerr=det_se, capsize=4,
                       color=colors, alpha=0.9, edgecolor='black')
        bars2 = ax.bar(x + bar_w / 2, cid_vals, bar_w, yerr=cid_se, capsize=4,
                       color=colors, alpha=0.5, edgecolor='black', hatch='//')
        ax.set_ylabel('Rate')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for i, f in enumerate(formulas):
            ax.text(i, -0.12, f, ha='center', va='top', fontsize=9, style='italic',
                    transform=ax.get_xaxis_transform())
        for bar, v, se in zip(bars1, det_vals, det_se):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.02,
                    f'{v:.0%}', ha='center', fontsize=9)
        ax.set_title(f'{title_word} concepts', color=title_color, fontsize=14)
        legend_h = [
            plt.Rectangle((0, 0), 1, 1, fc='gray', ec='black', alpha=0.9, label='Detection rate'),
            plt.Rectangle((0, 0), 1, 1, fc='gray', ec='black', alpha=0.5, hatch='//',
                          label='Cond. identification'),
        ]
        ax.legend(handles=legend_h, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / 'detection_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========== PLOT 2: Change in detection rate ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    swap_labels = ['Proj-swap', 'Delta-swap\n(random)', 'Delta-swap\n(nearest)']

    for ax, conditions, changes, ns, direction, ylabel_dir in [
        (axes[0], SUCCESS_CONDITIONS, succ_changes, succ_n,
         "Success", "positive = swap hurts"),
        (axes[1], FAILURE_CONDITIONS, fail_changes, fail_n,
         "Failure", "positive = swap boosts"),
    ]:
        if not any(ns[c] > 0 for c in conditions):
            continue
        vals = [np.nanmean(changes[c]) if ns[c] > 0 else 0 for c in conditions]
        ses = [np.nanstd(changes[c]) / np.sqrt(ns[c]) if ns[c] > 0 else 0 for c in conditions]
        bars = ax.bar(swap_labels, vals, yerr=[1.96 * s for s in ses], capsize=8,
                      color=swap_colors, alpha=0.8, edgecolor='black')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel(f'Change in detection rate\n({ylabel_dir})')
        ax.set_title(f'{direction} concepts', fontsize=13)
        for bar, v, cond in zip(bars, vals, conditions):
            yp = v + 0.02 if v >= 0 else v - 0.04
            ax.text(bar.get_x() + bar.get_width() / 2, yp, f'{v:+.1%}', ha='center', fontsize=11)
            ax.text(bar.get_x() + bar.get_width() / 2, -0.02,
                    f'n={ns[cond]}', ha='center', fontsize=9, va='top')
        bl_key = COND_BASELINE_SUCC if direction == "Success" else COND_BASELINE_FAIL
        for i, cond in enumerate(conditions):
            p_val = stat_results.get(f"{bl_key}_vs_{cond}", {}).get("p_value", 1.0)
            if p_val is not None and not np.isnan(p_val):
                ax.text(0.17 + 0.33 * i, 0.95, f'p={p_val:.2e}',
                        transform=ax.transAxes, ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'change_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========== PLOT 3: Per-concept scatter (6 panels) ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (conditions, bl_det, sw_det, changes, ns, direction, color_dir) in enumerate([
        (SUCCESS_CONDITIONS, succ_bl_det, succ_sw_det, succ_changes, succ_n, "Success", "#27ae60"),
        (FAILURE_CONDITIONS, fail_bl_det, fail_sw_det, fail_changes, fail_n, "Failure", "#e74c3c"),
    ]):
        for j, (cond, label) in enumerate(zip(conditions, swap_labels)):
            ax = axes[row, j]
            if len(bl_det) > 0 and len(sw_det[cond]) > 0:
                valid = ~np.isnan(sw_det[cond])
                ax.scatter(bl_det[valid], sw_det[cond][valid], alpha=0.6, s=50,
                           c=swap_colors[j], edgecolor='black', linewidth=0.5)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
                if ns[cond] > 0:
                    mc = np.nanmean(changes[cond])
                    word = "drop" if direction == "Success" else "boost"
                    ax.text(0.05, 0.95, f'Mean {word}: {mc:.1%}\nn={ns[cond]}',
                            transform=ax.transAxes, va='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlabel('Baseline detection rate')
            ax.set_ylabel(f'{label} detection rate')
            ax.set_title(f'{direction}: baseline vs {label.split(chr(10))[0].lower()}',
                         fontsize=11, color=color_dir)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_concept_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========== PLOT 4: Distribution of changes (histogram, 6 panels) ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for row, (conditions, changes, ns, direction, color_dir) in enumerate([
        (SUCCESS_CONDITIONS, succ_changes, succ_n, "Success", "#27ae60"),
        (FAILURE_CONDITIONS, fail_changes, fail_n, "Failure", "#e74c3c"),
    ]):
        xlabel = 'Change (baseline - swap)' if direction == "Success" else 'Change (swap - baseline)'
        for j, (cond, label) in enumerate(zip(conditions, swap_labels)):
            ax = axes[row, j]
            if ns[cond] > 0:
                valid = ~np.isnan(changes[cond])
                ax.hist(changes[cond][valid], bins=20, color=swap_colors[j],
                        alpha=0.7, edgecolor='black')
                ax.axvline(0, color='black', linestyle='--', linewidth=2)
                mv = np.nanmean(changes[cond])
                ax.axvline(mv, color='darkred', linewidth=2, label=f'Mean: {mv:.1%}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_title(f'{direction}: {label.split(chr(10))[0].lower()}',
                         fontsize=10, color=color_dir)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'change_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str,
    geometry_dir: Path,
    geometry_decomp_dir: Path,
    steering_dir: Path,
    output_dir: Path,
    threshold: float,
    trials_per_concept: int,
    layer_idx: int,
    strength: float,
    max_tokens: int,
    temperature: float,
    device: str,
    dtype: str,
    quantization: Optional[str],
    use_llm_judge: bool,
    seed: int,
    plot_interval: int,
    overwrite: bool,
    verbose: bool,
    n_trial_numbers: int = None,
    samples_per_trial: int = None,
):
    """Run the full swap experiment with iterative saving and resume support."""
    print("=" * 80)
    print("RIDGE SWAP EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}, Strength: {strength}")
    print(f"Trials per concept: {trials_per_concept}")
    print(f"Seed: {seed}")
    print("=" * 80)

    config_folder = f"layer_{layer_idx}_strength_{strength}"
    model_output_dir = output_dir / model_name / config_folder
    model_output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = model_output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Resume support
    existing_results: List[Dict] = []
    concept_conditions: Dict[str, Set[str]] = {}
    if not overwrite:
        print("\nChecking for existing results...")
        existing_results, concept_conditions = load_existing_results(model_output_dir)
    else:
        print("\nStarting fresh (--overwrite)")

    # Load partition
    print("\nLoading concept partition...")
    success_concepts, failure_concepts, partition_metadata = load_geometry_partition(
        geometry_dir, model_name, layer_idx=layer_idx, strength=strength
    )
    print(f"  Success: {len(success_concepts)}, Failure: {len(failure_concepts)}")

    # Load vectors
    print("\nLoading concept vectors...")
    vectors_dir = steering_dir / model_name / "vectors" / f"layer_{layer_idx}"
    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")
    vectors = load_concept_vectors(vectors_dir, success_concepts + failure_concepts)
    success_with_vectors = [c for c in success_concepts if c in vectors]
    failure_with_vectors = [c for c in failure_concepts if c in vectors]
    print(f"  With vectors - success: {len(success_with_vectors)}, failure: {len(failure_with_vectors)}")

    # Load ridge direction and decompose
    print("\nLoading ridge direction and computing decomposition...")
    w_ridge = load_ridge_direction(geometry_decomp_dir, model_name, layer_idx, strength)
    w_ridge, ridge_scores_success, deltas_success, ridge_scores_failure, deltas_failure = (
        compute_ridge_decomposition(vectors, w_ridge, success_with_vectors, failure_with_vectors)
    )
    torch.save(w_ridge.cpu(), model_output_dir / "ridge_direction.pt")

    mean_ridge_succ = np.mean(list(ridge_scores_success.values()))
    mean_ridge_fail = np.mean(list(ridge_scores_failure.values()))

    # Create pairings
    print("\nCreating delta-swap pairings...")
    pairings_succ_random, sims_succ_random = create_swap_pairings(
        success_with_vectors, failure_with_vectors, vectors, "random", seed)
    pairings_succ_nearest, sims_succ_nearest = create_swap_pairings(
        success_with_vectors, failure_with_vectors, vectors, "nearest_neighbor", seed)
    pairings_fail_random, sims_fail_random = create_swap_pairings(
        failure_with_vectors, success_with_vectors, vectors, "random", seed + 1000)
    pairings_fail_nearest, sims_fail_nearest = create_swap_pairings(
        failure_with_vectors, success_with_vectors, vectors, "nearest_neighbor", seed + 1000)

    # Load baseline
    print("\nLoading baseline results from experiment 02 (steering evaluation)...")
    baseline_per_concept = load_steering_baseline_per_concept(
        steering_dir, model_name, layer_idx=layer_idx, strength=strength)

    # Determine what needs to run
    active_succ = SUCCESS_CONDITIONS
    active_fail = FAILURE_CONDITIONS

    succ_needed: Dict[str, List[str]] = {}
    for c in success_with_vectors:
        existing = concept_conditions.get(c, set())
        missing = [cond for cond in active_succ if cond not in existing]
        if missing:
            succ_needed[c] = missing

    fail_needed: Dict[str, List[str]] = {}
    for c in failure_with_vectors:
        existing = concept_conditions.get(c, set())
        missing = [cond for cond in active_fail if cond not in existing]
        if missing:
            fail_needed[c] = missing

    work_items = (
        [(c, "success", needed) for c, needed in succ_needed.items()]
        + [(c, "failure", needed) for c, needed in fail_needed.items()]
    )
    rng = random.Random(seed)
    rng.shuffle(work_items)

    print(f"\nConditions to run: {len(work_items)} concepts")
    if not work_items:
        print("  All conditions already complete!")
        all_results = existing_results
    else:
        # Load model
        print("\nLoading model...")
        model = load_model(model_name=model_name, device=device, dtype=dtype, quantization=quantization)

        judge = None
        if use_llm_judge:
            try:
                judge = LLMJudge()
                print("  LLM judge initialized")
            except Exception as e:
                print(f"  Warning: Could not initialize LLM judge: {e}")

        all_results = list(existing_results)
        concepts_done = 0

        for i, (concept, concept_type, needed) in enumerate(tqdm(work_items, desc="Concepts")):
            concept_results: List[Dict] = []
            original_norm = vectors[concept].float().norm().item()

            if concept_type == "success":
                if COND_PROJ_SWAP_SUCC in needed:
                    ridge_shift = mean_ridge_fail - mean_ridge_succ
                    shifted = ridge_scores_success[concept] + ridge_shift
                    vec = normalize_to_target_norm(shifted * w_ridge + deltas_success[concept], original_norm)
                    concept_results.extend(run_steering_experiment(
                        model=model, concept=concept, steering_vector=vec,
                        layer_idx=layer_idx, strength=strength, max_tokens=max_tokens,
                        temperature=temperature, condition=COND_PROJ_SWAP_SUCC,
                        concept_type=concept_type, n_trial_numbers=n_trial_numbers,
                        samples_per_trial=samples_per_trial, n_trials=trials_per_concept))

                if COND_DELTA_RANDOM_SUCC in needed:
                    total = (n_trial_numbers * samples_per_trial
                             if n_trial_numbers and samples_per_trial else trials_per_concept)
                    t_rng = random.Random(seed + deterministic_hash(concept) + deterministic_hash(COND_DELTA_RANDOM_SUCC))
                    paired_list = [t_rng.choice(failure_with_vectors) for _ in range(total)]
                    proj = ridge_scores_success[concept] * w_ridge
                    vecs = [normalize_to_target_norm(proj + deltas_failure[p], original_norm) for p in paired_list]
                    concept_results.extend(run_steering_experiment(
                        model=model, concept=concept, steering_vectors=vecs,
                        layer_idx=layer_idx, strength=strength, max_tokens=max_tokens,
                        temperature=temperature, condition=COND_DELTA_RANDOM_SUCC,
                        concept_type=concept_type, paired_concepts=paired_list,
                        n_trial_numbers=n_trial_numbers, samples_per_trial=samples_per_trial,
                        n_trials=trials_per_concept))

                if COND_DELTA_NEAREST_SUCC in needed:
                    paired = pairings_succ_nearest[concept]
                    proj = ridge_scores_success[concept] * w_ridge
                    vec = normalize_to_target_norm(proj + deltas_failure[paired], original_norm)
                    concept_results.extend(run_steering_experiment(
                        model=model, concept=concept, steering_vector=vec,
                        layer_idx=layer_idx, strength=strength, max_tokens=max_tokens,
                        temperature=temperature, condition=COND_DELTA_NEAREST_SUCC,
                        concept_type=concept_type, paired_concept=paired,
                        n_trial_numbers=n_trial_numbers, samples_per_trial=samples_per_trial,
                        n_trials=trials_per_concept))

            else:  # failure concept
                if COND_PROJ_SWAP_FAIL in needed:
                    ridge_shift = mean_ridge_succ - mean_ridge_fail
                    shifted = ridge_scores_failure[concept] + ridge_shift
                    vec = normalize_to_target_norm(shifted * w_ridge + deltas_failure[concept], original_norm)
                    concept_results.extend(run_steering_experiment(
                        model=model, concept=concept, steering_vector=vec,
                        layer_idx=layer_idx, strength=strength, max_tokens=max_tokens,
                        temperature=temperature, condition=COND_PROJ_SWAP_FAIL,
                        concept_type=concept_type, n_trial_numbers=n_trial_numbers,
                        samples_per_trial=samples_per_trial, n_trials=trials_per_concept))

                if COND_DELTA_RANDOM_FAIL in needed:
                    total = (n_trial_numbers * samples_per_trial
                             if n_trial_numbers and samples_per_trial else trials_per_concept)
                    t_rng = random.Random(seed + deterministic_hash(concept) + deterministic_hash(COND_DELTA_RANDOM_FAIL))
                    paired_list = [t_rng.choice(success_with_vectors) for _ in range(total)]
                    proj = ridge_scores_failure[concept] * w_ridge
                    vecs = [normalize_to_target_norm(proj + deltas_success[p], original_norm) for p in paired_list]
                    concept_results.extend(run_steering_experiment(
                        model=model, concept=concept, steering_vectors=vecs,
                        layer_idx=layer_idx, strength=strength, max_tokens=max_tokens,
                        temperature=temperature, condition=COND_DELTA_RANDOM_FAIL,
                        concept_type=concept_type, paired_concepts=paired_list,
                        n_trial_numbers=n_trial_numbers, samples_per_trial=samples_per_trial,
                        n_trials=trials_per_concept))

                if COND_DELTA_NEAREST_FAIL in needed:
                    paired = pairings_fail_nearest[concept]
                    proj = ridge_scores_failure[concept] * w_ridge
                    vec = normalize_to_target_norm(proj + deltas_success[paired], original_norm)
                    concept_results.extend(run_steering_experiment(
                        model=model, concept=concept, steering_vector=vec,
                        layer_idx=layer_idx, strength=strength, max_tokens=max_tokens,
                        temperature=temperature, condition=COND_DELTA_NEAREST_FAIL,
                        concept_type=concept_type, paired_concept=paired,
                        n_trial_numbers=n_trial_numbers, samples_per_trial=samples_per_trial,
                        n_trials=trials_per_concept))

            # LLM judge
            if judge is not None and concept_results:
                try:
                    prompts = [f"Trial {r['trial']}: Do you detect an injected thought?" for r in concept_results]
                    concept_results = batch_evaluate(judge, concept_results, prompts)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: LLM judge failed for {concept}: {e}")

            all_results.extend(concept_results)
            concepts_done += 1

            # Save iteratively
            m, pc, st = save_results_iterative(
                all_results, model_output_dir,
                pairings_succ_random, pairings_succ_nearest,
                pairings_fail_random, pairings_fail_nearest,
                sims_succ_random, sims_succ_nearest,
                sims_fail_random, sims_fail_nearest,
                baseline_per_concept, success_with_vectors, failure_with_vectors,
                partition_metadata, trials_per_concept, layer_idx, strength, seed)

            if concepts_done % plot_interval == 0 or i == len(work_items) - 1:
                try:
                    create_plots(m, pc, st, plots_dir)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: plot update failed: {e}")

        model.cleanup()

    # Final save
    print("\nFinalizing results...")
    metrics, per_concept_results, stat_results = save_results_iterative(
        all_results, model_output_dir,
        pairings_succ_random, pairings_succ_nearest,
        pairings_fail_random, pairings_fail_nearest,
        sims_succ_random, sims_succ_nearest,
        sims_fail_random, sims_fail_nearest,
        baseline_per_concept, success_with_vectors, failure_with_vectors,
        partition_metadata, trials_per_concept, layer_idx, strength, seed)

    create_plots(metrics, per_concept_results, stat_results, plots_dir)

    # Save config
    config = {
        "model": model_name, "layer": layer_idx, "strength": strength,
        "trials_per_concept": trials_per_concept, "seed": seed,
        "use_llm_judge": use_llm_judge,
    }
    with open(model_output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    interpretation = determine_interpretation(metrics, stat_results)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for condition, m in metrics.items():
        det = m.get("detection_rate", 0)
        cid = m.get("conditional_identification_rate", 0)
        intro = m.get("introspection_rate", 0)
        print(f"  {condition}: det={det:.1%}  cond_id={cid:.1%}  intro={intro:.1%}")

    for label, bl_key, conditions in [
        ("SUCCESS", COND_BASELINE_SUCC, SUCCESS_CONDITIONS),
        ("FAILURE", COND_BASELINE_FAIL, FAILURE_CONDITIONS),
    ]:
        test_key = f"{bl_key}_vs_{conditions[0]}"
        if test_key in stat_results and "t_statistic" in stat_results[test_key]:
            print(f"\n{label} concept tests:")
            for cond in conditions:
                key = f"{bl_key}_vs_{cond}"
                if key in stat_results:
                    t = stat_results[key].get("t_statistic", float("nan"))
                    p = stat_results[key].get("p_value", float("nan"))
                    d = stat_results[key].get("effect_size", float("nan"))
                    print(f"  vs {cond}: t={t:.3f}, p={p:.2e}, d={d:.3f}")

    print(f"\nInterpretation: {interpretation}")
    print(f"Results saved to: {model_output_dir}")
    print("=" * 80)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ridge Projection vs. Residual Decomposition Swap Experiments")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--geometry-dir", default=DEFAULT_GEOMETRY_DIR,
                        help="Path to 04b_vector_geometry results")
    parser.add_argument("--geometry-decomp-dir", default=DEFAULT_GEOMETRY_DECOMP_DIR,
                        help="Path to 04b_vector_decomposition results")
    parser.add_argument("--steering-dir", default=DEFAULT_STEERING_DIR,
                        help="Path to experiment 02 (steering evaluation) results")
    parser.add_argument("-od", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Success/failure threshold")
    parser.add_argument("-l", "--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("-nt", "--trials-per-concept", type=int, default=None,
                        help="Total trials per concept (overrides trial-number structure)")
    parser.add_argument("--n-trial-numbers", type=int, default=DEFAULT_N_TRIAL_NUMBERS)
    parser.add_argument("--samples-per-trial", type=int, default=DEFAULT_SAMPLES_PER_TRIAL)
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-mt", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("-d", "--device", default=DEFAULT_DEVICE)
    parser.add_argument("-dt", "--dtype", default=DEFAULT_DTYPE,
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("-q", "--quantization", default=None, choices=["8bit", "4bit"])
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-interval", type=int, default=DEFAULT_PLOT_UPDATE_INTERVAL)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from existing results only")
    return parser.parse_args()


def main():
    args = parse_args()
    geometry_dir = Path(args.geometry_dir)
    geometry_decomp_dir = Path(args.geometry_decomp_dir)
    steering_dir = Path(args.steering_dir)
    output_dir = Path(args.output_dir)

    # Plots-only mode
    if args.plots_only:
        print("PLOTS-ONLY MODE")
        config_folder = f"layer_{args.layer}_strength_{args.strength}"
        model_output_dir = output_dir / args.model / config_folder
        plots_dir = model_output_dir / "plots"
        results_path = model_output_dir / "results.json"
        per_concept_path = model_output_dir / "per_concept_results.json"
        if not results_path.exists() or not per_concept_path.exists():
            print(f"Error: Missing results in {model_output_dir}")
            return
        with open(results_path) as f:
            rd = json.load(f)
        with open(per_concept_path) as f:
            pc = json.load(f)
        plots_dir.mkdir(exist_ok=True)
        create_plots(rd["aggregate"], pc, rd["statistical_tests"], plots_dir)
        print(f"Plots saved to {plots_dir}")
        return

    # Trial structure
    if args.trials_per_concept is not None:
        trials_per_concept = args.trials_per_concept
        n_trial_numbers = None
        samples_per_trial = None
    else:
        n_trial_numbers = args.n_trial_numbers
        samples_per_trial = args.samples_per_trial
        trials_per_concept = n_trial_numbers * samples_per_trial

    run_experiment(
        model_name=args.model,
        geometry_dir=geometry_dir,
        geometry_decomp_dir=geometry_decomp_dir,
        steering_dir=steering_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        trials_per_concept=trials_per_concept,
        layer_idx=args.layer,
        strength=args.strength,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        use_llm_judge=not args.no_llm_judge,
        seed=args.seed,
        plot_interval=args.plot_interval,
        overwrite=args.overwrite,
        verbose=args.verbose,
        n_trial_numbers=n_trial_numbers,
        samples_per_trial=samples_per_trial,
    )


if __name__ == "__main__":
    main()
