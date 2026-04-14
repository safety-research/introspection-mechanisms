#!/usr/bin/env python3
"""
Experiment 58: Gaussian Concept Vectors — What Makes a Concept Vector Work?

Goal: Determine whether introspection requires meaningful concept vectors or whether
random noise of the right magnitude/structure is sufficient to trigger detection.

This experiment injects various types of Gaussian/random noise instead of real concept
vectors, systematically varying what structural properties are preserved vs destroyed.

Conditions:

GROUP A — Null Hypothesis Tests:
  real_concepts:          Original concept vectors from experiment 02 (steering evaluation) (positive control)
  no_steering:            No steering applied (negative control / false positive rate)
  isotropic_noise:        randn(d) scaled to mean norm of success concept vectors
  per_concept_norm_matched: For each success concept, randn(d) with that concept's exact norm
  shuffled_dims:          Real concept vectors with dimensions randomly permuted

GROUP B — Subspace Constraints:
  concept_subspace_noise: randn projected into top-k PCA of ALL concept vectors, norm-matched
  success_subspace_noise: randn projected into top-k PCA of SUCCESS vectors, norm-matched
  orthogonal_noise:       randn projected into orthogonal complement of concept PCA, norm-matched

GROUP C — Distribution Matching:
  covariance_matched:     Samples from N(mu_success, Sigma_success)

GROUP D — Norm Sweep:
  norm_sweep:             Fixed random direction at norms [p10, p25, p50, p75, p90, p99]

Key Analyses:
  1. Detection rate comparison across all conditions
  2. Per-concept paired analysis correlating (original - random) with geometric properties
  3. Norm-detection curve from norm_sweep
  4. Subspace decomposition comparison
  5. Statistical tests with Bonferroni correction

Dependencies:
  - 04b_vector_geometry.py → success/failure partition
  - 02b_steering_500_concepts → concept vectors + baseline results

Usage:
    python 04_geometry_analysis.py -m gemma3_27b
    python 04_geometry_analysis.py -m gemma3_27b --conditions isotropic_noise real_concepts no_steering
    python 04_geometry_analysis.py -m gemma3_27b --n-random-vectors 20 --n-trial-numbers 3 --samples-per-trial 3
    python 04_geometry_analysis.py -m gemma3_27b --plots-only
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import json
import os
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
from model_utils import load_model, ModelWrapper
from steering_utils import (
    run_steered_introspection_test_batch,
    run_unsteered_introspection_test_batch,
    check_concept_mentioned,
)
from eval_utils import LLMJudge, batch_evaluate

# ─────────────────────────────────────────────────────────────────────────────
# Constants & defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemma3_27b"
DEFAULT_GEOMETRY_DIR = "analysis/04b_vector_geometry"
DEFAULT_STEERING_DIR = "analysis/02b_steering_500_concepts"
DEFAULT_OUTPUT_DIR = "analysis/04_geometry_analysis"
DEFAULT_LAYER = 38
DEFAULT_STRENGTH = 4.0
DEFAULT_N_TRIAL_NUMBERS = 5
DEFAULT_SAMPLES_PER_TRIAL = 5
DEFAULT_N_RANDOM_VECTORS = 50
DEFAULT_PCA_K = 50
DEFAULT_NORM_PERCENTILES = [10, 25, 50, 75, 90, 99]
DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SEED = 42
DEFAULT_PLOT_INTERVAL = 10

ALL_CONDITIONS = [
    # Group A: Null hypothesis
    "real_concepts",
    "no_steering",
    "isotropic_noise",
    "per_concept_norm_matched",
    "shuffled_dims",
    # Group B: Subspace
    "concept_subspace_noise",
    "success_subspace_noise",
    "orthogonal_noise",
    # Group C: Distribution matching
    "covariance_matched",
    # Group D: Norm sweep
    "norm_sweep",
]

# Conditions that run per-concept (one vector per success concept)
PER_CONCEPT_CONDITIONS = {"real_concepts", "per_concept_norm_matched", "shuffled_dims"}

# Conditions that use N random vectors (fixed, not per-concept)
FIXED_VECTOR_CONDITIONS = {
    "isotropic_noise", "concept_subspace_noise", "success_subspace_noise",
    "orthogonal_noise", "covariance_matched",
}

# Noise conditions: these use random/synthetic vectors, so concept detection
# and identification grading are not meaningful (there is no real concept to
# detect or identify). We set `detected` to None and skip identification grading.
NOISE_CONDITIONS = {
    "isotropic_noise", "per_concept_norm_matched", "shuffled_dims",
    "concept_subspace_noise", "success_subspace_noise",
    "orthogonal_noise", "covariance_matched",
}


def is_noise_condition(condition: str) -> bool:
    """Check if a condition is a noise condition (no real concept to detect/identify)."""
    return condition in NOISE_CONDITIONS or condition.startswith("norm_sweep_")

# Condition display names (for plots)
CONDITION_LABELS = {
    "real_concepts": "Real concepts",
    "no_steering": "No steering",
    "isotropic_noise": "Isotropic noise\n(norm-matched)",
    "per_concept_norm_matched": "Per-concept\nnorm-matched noise",
    "shuffled_dims": "Shuffled dims",
    "concept_subspace_noise": "Concept subspace\nnoise (PCA-k)",
    "success_subspace_noise": "Success subspace\nnoise (PCA-k)",
    "orthogonal_noise": "Orthogonal\nnoise",
    "covariance_matched": "Covariance-matched\nnoise",
    "norm_sweep": "Norm sweep",
}

CONDITION_COLORS = {
    "real_concepts": "#2ecc71",           # Green
    "no_steering": "#95a5a6",             # Gray
    "isotropic_noise": "#e74c3c",         # Red
    "per_concept_norm_matched": "#e67e22", # Orange
    "shuffled_dims": "#f1c40f",           # Yellow
    "concept_subspace_noise": "#3498db",  # Blue
    "success_subspace_noise": "#2980b9",  # Dark blue
    "orthogonal_noise": "#9b59b6",        # Purple
    "covariance_matched": "#1abc9c",      # Teal
    "norm_sweep": "#e91e63",              # Pink
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 58: Gaussian Concept Vectors — What Makes a Concept Vector Work?"
    )
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--geometry-dir", type=str, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--steering-dir", type=str, default=DEFAULT_STEERING_DIR)
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-l", "--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--n-trial-numbers", type=int, default=DEFAULT_N_TRIAL_NUMBERS,
                        help="Unique trial numbers per vector")
    parser.add_argument("--samples-per-trial", type=int, default=DEFAULT_SAMPLES_PER_TRIAL,
                        help="Samples per trial number")
    parser.add_argument("--n-random-vectors", type=int, default=DEFAULT_N_RANDOM_VECTORS,
                        help="Random vectors per fixed-vector condition")
    parser.add_argument("--pca-k", type=int, default=DEFAULT_PCA_K,
                        help="PCA rank for subspace conditions")
    parser.add_argument("--norm-percentiles", type=float, nargs="+",
                        default=DEFAULT_NORM_PERCENTILES,
                        help="Percentiles for norm sweep")
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, default=None,
                        help="Conditions to run (default: all)")
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-mt", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-dt", "--dtype", type=str, default=DEFAULT_DTYPE,
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("-q", "--quantization", type=str, default=None,
                        choices=["8bit", "4bit"])
    parser.add_argument("--plot-interval", type=int, default=DEFAULT_PLOT_INTERVAL)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (follows experiment 04d/04e (direction analysis) patterns)
# ─────────────────────────────────────────────────────────────────────────────

def load_concept_vectors(vectors_dir: Path, concepts: List[str]) -> Dict[str, torch.Tensor]:
    """Load concept vectors from disk."""
    vectors = {}
    missing = []
    for concept in concepts:
        vector_path = vectors_dir / f"{concept}.pt"
        if vector_path.exists():
            vectors[concept] = torch.load(vector_path, weights_only=True)
        else:
            missing.append(concept)
    if missing:
        print(f"  Warning: Missing vectors for {len(missing)} concepts: {missing[:5]}...")
    return vectors


def load_geometry_partition(
    geometry_dir: Path, model_name: str, layer_idx: int, strength: float
) -> Tuple[List[str], List[str], dict]:
    """Load success/failure partition from experiment 04b (vector geometry)'s subspace_analysis.json."""
    config_folder = f"layer_{layer_idx}_strength_{strength}"
    subspace_path = geometry_dir / model_name / config_folder / "subspace_analysis.json"
    if not subspace_path.exists():
        # Fallback to legacy path
        subspace_path = geometry_dir / model_name / "subspace_analysis.json"
    if not subspace_path.exists():
        raise FileNotFoundError(f"Could not find experiment 04b (vector geometry) subspace analysis at {subspace_path}")

    with open(subspace_path, "r") as f:
        data = json.load(f)

    success = data.get("success_concepts", [])
    failure = data.get("failure_concepts", [])
    metadata = {
        "source": "experiment 04b (vector geometry)",
        "threshold": data.get("threshold"),
        "metric_used": data.get("metric_used"),
        "lda_accuracy": data.get("lda_classification_accuracy"),
    }
    return success, failure, metadata


def load_steering_baseline_per_concept(
    steering_dir: Path, model_name: str, layer_idx: int, strength: float
) -> Dict[str, Dict]:
    """Load per-concept baseline detection/identification rates from experiment 02 (steering evaluation) results."""
    config_folder = f"layer_{layer_idx}_strength_{strength}"
    results_path = steering_dir / model_name / config_folder / "results.json"
    if not results_path.exists():
        print(f"  Warning: No experiment 02 (steering evaluation) baseline results at {results_path}")
        return {}

    with open(results_path, "r") as f:
        data = json.load(f)

    per_concept = {}
    results_list = data if isinstance(data, list) else data.get("results", [])
    # Group by concept
    concept_trials = {}
    for r in results_list:
        c = r.get("concept")
        if c:
            concept_trials.setdefault(c, []).append(r)

    for concept, trials in concept_trials.items():
        injection_trials = [t for t in trials if t.get("trial_type") == "injection"]
        if not injection_trials:
            continue
        detected = sum(
            1 for t in injection_trials
            if t.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
        )
        identified = sum(
            1 for t in injection_trials
            if t.get("evaluations", {}).get("correct_concept_identification", {}).get("grade", 0)
        )
        n = len(injection_trials)
        per_concept[concept] = {
            "detection_rate": detected / n if n else 0,
            "identification_rate": identified / n if n else 0,
            "combined_rate": sum(
                1 for t in injection_trials
                if t.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
                and t.get("evaluations", {}).get("correct_concept_identification", {}).get("grade", 0)
            ) / n if n else 0,
            "n_trials": n,
        }
    return per_concept


# ─────────────────────────────────────────────────────────────────────────────
# Vector generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_isotropic_noise(
    d: int, target_norm: float, n_vectors: int, seed: int
) -> List[torch.Tensor]:
    """Generate random Gaussian noise vectors, scaled to target_norm."""
    rng = torch.Generator().manual_seed(seed)
    vectors = []
    for _ in range(n_vectors):
        v = torch.randn(d, generator=rng)
        v = v / v.norm() * target_norm
        vectors.append(v)
    return vectors


def generate_norm_matched_noise(
    concept_vectors: Dict[str, torch.Tensor], seed: int
) -> Dict[str, torch.Tensor]:
    """For each concept, generate random noise with the same norm."""
    rng = torch.Generator().manual_seed(seed)
    d = next(iter(concept_vectors.values())).shape[0]
    result = {}
    for concept, vec in concept_vectors.items():
        noise = torch.randn(d, generator=rng)
        noise = noise / noise.norm() * vec.norm()
        result[concept] = noise
    return result


def generate_shuffled_vectors(
    concept_vectors: Dict[str, torch.Tensor], seed: int
) -> Dict[str, torch.Tensor]:
    """Randomly permute dimensions of each concept vector."""
    rng = torch.Generator().manual_seed(seed)
    result = {}
    for concept, vec in concept_vectors.items():
        perm = torch.randperm(vec.shape[0], generator=rng)
        result[concept] = vec[perm].clone()
    return result


def generate_subspace_noise(
    concept_vectors_stacked: torch.Tensor,
    k: int,
    target_norm: float,
    n_vectors: int,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Generate noise vectors projected into top-k PCA subspace of concept vectors.

    Returns:
        (vectors, basis) where basis is [k, d] PCA basis
    """
    # Compute PCA
    vecs = concept_vectors_stacked.float()
    mean = vecs.mean(dim=0)
    centered = vecs - mean
    # SVD: centered = U @ diag(S) @ Vt
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    k = min(k, Vt.shape[0])
    basis = Vt[:k]  # [k, d]

    rng = torch.Generator().manual_seed(seed)
    vectors = []
    for _ in range(n_vectors):
        # Random coefficients in subspace
        coeffs = torch.randn(k, generator=rng)
        v = coeffs @ basis  # [d]
        v = v / v.norm() * target_norm
        vectors.append(v)
    return vectors, basis


def generate_orthogonal_noise(
    basis: torch.Tensor,
    d: int,
    target_norm: float,
    n_vectors: int,
    seed: int,
) -> List[torch.Tensor]:
    """
    Generate noise in the orthogonal complement of the given PCA basis.

    Args:
        basis: [k, d] PCA basis vectors
    """
    # Projection matrix onto subspace: P = V^T V
    proj = basis.T @ basis  # [d, d]

    rng = torch.Generator().manual_seed(seed)
    vectors = []
    for _ in range(n_vectors):
        noise = torch.randn(d, generator=rng).float()
        # Remove subspace component
        noise_orth = noise - proj @ noise
        if noise_orth.norm() < 1e-8:
            # Extremely unlikely, but handle gracefully
            continue
        noise_orth = noise_orth / noise_orth.norm() * target_norm
        vectors.append(noise_orth)
    return vectors


def generate_covariance_matched(
    concept_vectors_stacked: torch.Tensor,
    n_vectors: int,
    seed: int,
) -> List[torch.Tensor]:
    """Sample from N(mu, Sigma) where mu and Sigma come from concept vectors."""
    vecs = concept_vectors_stacked.float()
    mu = vecs.mean(dim=0)
    centered = vecs - mu
    N = centered.shape[0]
    # Use eigendecomposition for numerical stability
    cov = (centered.T @ centered) / (N - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-6)

    rng = torch.Generator().manual_seed(seed)
    vectors = []
    for _ in range(n_vectors):
        z = torch.randn(eigvals.shape[0], generator=rng)
        v = mu + eigvecs @ (eigvals.sqrt() * z)
        vectors.append(v)
    return vectors


def generate_norm_sweep_vectors(
    d: int,
    direction_seed: int,
    percentiles: List[float],
    concept_norms: np.ndarray,
    n_per_norm: int,
    above_max_multipliers: Optional[List[float]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Generate vectors along a fixed random direction at varying norms.

    The sweep covers percentiles of the concept norm distribution AND extends
    above the maximum concept norm using above_max_multipliers (default [1.5, 2.0])
    to test what happens at norms beyond the real concept range.

    Returns:
        Dict mapping "norm_sweep_pXX" or "norm_sweep_x150" etc -> list of vectors
        (n_per_norm copies, each gets different trial numbers but same direction+norm)
    """
    if above_max_multipliers is None:
        above_max_multipliers = [1.5, 2.0]

    rng = torch.Generator().manual_seed(direction_seed)
    direction = torch.randn(d, generator=rng).float()
    direction = direction / direction.norm()

    norms = np.percentile(concept_norms, percentiles)
    result = {}
    for pct, norm_val in zip(percentiles, norms):
        label = f"norm_sweep_p{int(pct)}"
        v = direction * norm_val
        result[label] = [v] * n_per_norm  # same vector repeated

    # Extend sweep above the max concept norm to test supranormal norms
    max_norm = float(np.max(concept_norms))
    for mult in above_max_multipliers:
        norm_val = max_norm * mult
        label = f"norm_sweep_x{int(mult * 100)}"
        v = direction * norm_val
        result[label] = [v] * n_per_norm
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner (follows experiment 04d/04e (direction analysis) pattern)
# ─────────────────────────────────────────────────────────────────────────────

def run_steering_experiment(
    model: ModelWrapper,
    label: str,
    steering_vector: torch.Tensor,
    layer_idx: int,
    strength: float,
    n_trial_numbers: int,
    samples_per_trial: int,
    max_tokens: int,
    temperature: float,
    condition: str,
) -> List[Dict]:
    """
    Run steering experiment for one vector.

    Args:
        label: Identifying label for this vector (concept name or noise_idx)
    """
    trial_numbers = []
    for t in range(1, n_trial_numbers + 1):
        trial_numbers.extend([t] * samples_per_trial)

    # Use a placeholder concept word for the prompt (the model is asked
    # generically about detecting "an injected thought", not a specific word)
    concept_word = label

    responses = run_steered_introspection_test_batch(
        model=model,
        concept_word=concept_word,
        steering_vector=steering_vector,
        layer_idx=layer_idx,
        strength=strength,
        trial_numbers=trial_numbers,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    results = []
    trial_sample_counts = {}
    for i, (trial_num, response) in enumerate(zip(trial_numbers, responses)):
        if trial_num not in trial_sample_counts:
            trial_sample_counts[trial_num] = 0
        sample_idx = trial_sample_counts[trial_num]
        trial_sample_counts[trial_num] += 1

        # For noise conditions, `detected` via keyword matching is meaningless
        # because the label (e.g. "iso_noise_0") is not a real concept the model
        # could mention. Set to None so downstream code doesn't misinterpret it.
        is_noise = is_noise_condition(condition)
        detected = None if is_noise else check_concept_mentioned(response, label)

        results.append({
            "label": label,
            "concept": label,  # Required by batch_evaluate for identification grading
            "condition": condition,
            "trial": trial_num,
            "sample_idx": sample_idx,
            "response": response,
            "layer": layer_idx,
            "strength": strength,
            "trial_type": "injection",
            "detected": detected,
        })
    return results


def run_no_steering_experiment(
    model: ModelWrapper,
    n_trial_numbers: int,
    samples_per_trial: int,
    max_tokens: int,
    temperature: float,
) -> List[Dict]:
    """Run trials with no steering vector applied (false positive baseline)."""
    trial_numbers = []
    for t in range(1, n_trial_numbers + 1):
        trial_numbers.extend([t] * samples_per_trial)

    responses = run_unsteered_introspection_test_batch(
        model=model,
        concept_word="nothing",
        trial_numbers=trial_numbers,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    results = []
    trial_sample_counts = {}
    for i, (trial_num, response) in enumerate(zip(trial_numbers, responses)):
        if trial_num not in trial_sample_counts:
            trial_sample_counts[trial_num] = 0
        sample_idx = trial_sample_counts[trial_num]
        trial_sample_counts[trial_num] += 1

        results.append({
            "label": "no_steering",
            "concept": "no_steering",  # Required by batch_evaluate
            "condition": "no_steering",
            "trial": trial_num,
            "sample_idx": sample_idx,
            "response": response,
            "layer": -1,
            "strength": 0.0,
            "trial_type": "control",
            "detected": False,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Resume support
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_results(output_dir: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Load existing results for resume support.

    Returns:
        (all_results, completed_keys) where completed_keys is a set of
        "condition::label" strings for items already done.
    """
    results_path = output_dir / "all_results.json"
    if not results_path.exists():
        return [], set()

    with open(results_path, "r") as f:
        data = json.load(f)

    all_results = data.get("results", [])
    completed = set()
    for r in all_results:
        key = f"{r.get('condition', '')}::{r.get('label', '')}"
        completed.add(key)
    return all_results, completed


# ─────────────────────────────────────────────────────────────────────────────
# Result saving & aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_detection_rate(results: List[Dict]) -> Dict[str, float]:
    """Compute detection and identification rates from evaluated results."""
    if not results:
        return {"detection_rate": 0, "identification_rate": 0, "introspection_rate": 0, "n": 0}

    n = len(results)
    detected = sum(
        1 for r in results
        if r.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
    )
    identified = sum(
        1 for r in results
        if r.get("evaluations", {}).get("correct_concept_identification", {}).get("grade", 0)
    )
    both = sum(
        1 for r in results
        if r.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
        and r.get("evaluations", {}).get("correct_concept_identification", {}).get("grade", 0)
    )
    return {
        "detection_rate": detected / n,
        "identification_rate": identified / n,
        "introspection_rate": both / n,
        "n": n,
    }


def compute_aggregate_metrics(all_results: List[Dict]) -> Dict[str, Dict]:
    """
    Compute per-condition aggregate metrics with confidence intervals.

    For per-concept conditions, also computes per-concept rates for paired analysis.
    """
    # Group results by condition
    by_condition = {}
    for r in all_results:
        cond = r.get("condition", "unknown")
        by_condition.setdefault(cond, []).append(r)

    metrics = {}
    for cond, results in by_condition.items():
        overall = compute_detection_rate(results)

        # For per-concept conditions, compute per-label rates for SE
        by_label = {}
        for r in results:
            lbl = r.get("label", "")
            by_label.setdefault(lbl, []).append(r)

        label_det_rates = []
        label_id_rates = []
        label_intro_rates = []
        label_det_rates_by_name = {}  # label -> detection_rate, for paired tests
        for lbl, lbl_results in by_label.items():
            m = compute_detection_rate(lbl_results)
            label_det_rates.append(m["detection_rate"])
            label_id_rates.append(m["identification_rate"])
            label_intro_rates.append(m["introspection_rate"])
            label_det_rates_by_name[lbl] = m["detection_rate"]

        n_labels = len(label_det_rates)

        def se(rates):
            if len(rates) > 1:
                return float(np.std(rates, ddof=1) / np.sqrt(len(rates)))
            return 0.0

        metrics[cond] = {
            **overall,
            "n_labels": n_labels,
            "detection_rate_se": se(label_det_rates),
            "identification_rate_se": se(label_id_rates),
            "introspection_rate_se": se(label_intro_rates),
            "per_label_detection_rates": label_det_rates,
            "per_label_detection_rates_by_name": label_det_rates_by_name,
        }
    return metrics


def compute_paired_analysis(
    all_results: List[Dict],
    concept_vectors: Dict[str, torch.Tensor],
    success_concepts: List[str],
) -> Dict:
    """
    For per-concept conditions, compute (original - random) per concept and
    correlate with geometric properties.
    """
    # Build per-concept detection rates by condition
    by_condition_concept = {}
    for r in all_results:
        cond = r.get("condition")
        label = r.get("label")
        if cond and label:
            by_condition_concept.setdefault(cond, {}).setdefault(label, []).append(r)

    # Get real_concepts rates
    real_rates = {}
    for concept, trials in by_condition_concept.get("real_concepts", {}).items():
        real_rates[concept] = compute_detection_rate(trials)["detection_rate"]

    paired_results = {}
    for compare_cond in ["per_concept_norm_matched", "shuffled_dims"]:
        compare_rates = {}
        for concept, trials in by_condition_concept.get(compare_cond, {}).items():
            compare_rates[concept] = compute_detection_rate(trials)["detection_rate"]

        # Compute deltas for concepts present in both
        common = sorted(set(real_rates.keys()) & set(compare_rates.keys()) & set(concept_vectors.keys()))
        if len(common) < 5:
            continue

        deltas = []
        norms = []
        for c in common:
            deltas.append(real_rates[c] - compare_rates[c])
            norms.append(concept_vectors[c].float().norm().item())

        deltas = np.array(deltas)
        norms = np.array(norms)

        # Correlate delta with norm
        if np.std(deltas) > 1e-8 and np.std(norms) > 1e-8:
            r_val, p_val = stats.pearsonr(deltas, norms)
            spearman_r, spearman_p = stats.spearmanr(deltas, norms)
        else:
            r_val, p_val, spearman_r, spearman_p = 0, 1, 0, 1

        paired_results[compare_cond] = {
            "n_concepts": len(common),
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0,
            "pearson_r_vs_norm": float(r_val),
            "pearson_p_vs_norm": float(p_val),
            "spearman_r_vs_norm": float(spearman_r),
            "spearman_p_vs_norm": float(spearman_p),
            "per_concept": {
                c: {
                    "real_rate": float(real_rates[c]),
                    "random_rate": float(compare_rates[c]),
                    "delta": float(real_rates[c] - compare_rates[c]),
                    "norm": float(concept_vectors[c].float().norm().item()),
                }
                for c in common
            },
        }

    return paired_results


def compute_statistical_tests(
    metrics: Dict[str, Dict],
) -> Dict:
    """Compute statistical tests comparing each condition vs real_concepts.

    For per-concept conditions (same concept labels as real_concepts), uses a
    paired t-test (ttest_rel) since the measurements are on the same concepts.
    For fixed-vector conditions (different labels), falls back to unpaired t-test.
    """
    real_rates_by_name = metrics.get("real_concepts", {}).get("per_label_detection_rates_by_name", {})
    real_rates = metrics.get("real_concepts", {}).get("per_label_detection_rates", [])
    if not real_rates:
        return {}

    tests = {}
    conditions_to_test = [
        c for c in metrics if c not in ("real_concepts", "no_steering")
        and not c.startswith("norm_sweep_")
    ]

    n_comparisons = len(conditions_to_test)

    for cond in conditions_to_test:
        cond_rates = metrics[cond].get("per_label_detection_rates", [])
        cond_rates_by_name = metrics[cond].get("per_label_detection_rates_by_name", {})
        if not cond_rates:
            continue

        # Check if the data is paired: same concept labels in both conditions.
        # Per-concept conditions (per_concept_norm_matched, shuffled_dims) use
        # the same concept names as labels, so we can align by concept for a
        # paired test. Fixed-vector conditions have different labels.
        common_labels = sorted(set(real_rates_by_name.keys()) & set(cond_rates_by_name.keys()))
        is_paired = len(common_labels) >= 5  # need enough overlap for a meaningful paired test

        if is_paired:
            # Paired test: align data by concept label
            real_arr = np.array([real_rates_by_name[lbl] for lbl in common_labels])
            cond_arr = np.array([cond_rates_by_name[lbl] for lbl in common_labels])

            if len(real_arr) > 1:
                t_stat, t_p = stats.ttest_rel(real_arr, cond_arr)
                # Wilcoxon signed-rank test (paired non-parametric)
                try:
                    w_stat, w_p = stats.wilcoxon(real_arr, cond_arr, alternative="two-sided")
                except ValueError:
                    # All differences are zero
                    w_stat, w_p = 0, 1.0
                # Cohen's d for paired data (using std of differences)
                diffs = real_arr - cond_arr
                diff_std = np.std(diffs, ddof=1)
                cohens_d = float(np.mean(diffs) / diff_std) if diff_std > 0 else 0
            else:
                t_stat, t_p, w_stat, w_p, cohens_d = 0, 1, 0, 1, 0

            test_type = "paired"
            u_stat, u_p = w_stat, w_p  # use Wilcoxon as the non-parametric test
        else:
            # Unpaired test: different labels, cannot align
            real_arr = np.array(real_rates)
            cond_arr = np.array(cond_rates)

            if len(real_arr) > 1 and len(cond_arr) > 1:
                t_stat, t_p = stats.ttest_ind(real_arr, cond_arr)
                # Mann-Whitney U
                u_stat, u_p = stats.mannwhitneyu(real_arr, cond_arr, alternative="two-sided")
                # Cohen's d (unpaired)
                pooled_std = np.sqrt(
                    ((len(real_arr) - 1) * np.var(real_arr, ddof=1)
                     + (len(cond_arr) - 1) * np.var(cond_arr, ddof=1))
                    / (len(real_arr) + len(cond_arr) - 2)
                )
                cohens_d = (np.mean(real_arr) - np.mean(cond_arr)) / pooled_std if pooled_std > 0 else 0
            else:
                t_stat, t_p, u_stat, u_p, cohens_d = 0, 1, 0, 1, 0

            test_type = "unpaired"

        # Bonferroni correction
        bonferroni_p = min(t_p * n_comparisons, 1.0)

        tests[cond] = {
            "vs": "real_concepts",
            "test_type": test_type,
            "t_statistic": float(t_stat),
            "p_value": float(t_p),
            "bonferroni_p": float(bonferroni_p),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_p),
            "cohens_d": float(cohens_d),
            "real_mean": float(np.mean(real_arr)),
            "cond_mean": float(np.mean(cond_arr)),
            "n_real": len(real_arr),
            "n_cond": len(cond_arr),
        }

    return tests


def save_results(
    all_results: List[Dict],
    metrics: Dict,
    paired_analysis: Dict,
    stat_tests: Dict,
    vector_properties: Dict,
    config: Dict,
    output_dir: Path,
):
    """Save all results and analyses to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump({
            "results": all_results,
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)

    # Aggregate metrics (strip per_label fields to save space)
    strip_keys = {"per_label_detection_rates", "per_label_detection_rates_by_name"}
    metrics_clean = {}
    for cond, m in metrics.items():
        metrics_clean[cond] = {k: v for k, v in m.items() if k not in strip_keys}
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(metrics_clean, f, indent=2)

    # Paired analysis
    if paired_analysis:
        with open(output_dir / "paired_analysis.json", "w") as f:
            json.dump(paired_analysis, f, indent=2)

    # Statistical tests
    if stat_tests:
        with open(output_dir / "statistical_tests.json", "w") as f:
            json.dump(stat_tests, f, indent=2)

    # Vector properties
    if vector_properties:
        with open(output_dir / "vector_properties.json", "w") as f:
            json.dump(vector_properties, f, indent=2)

    # Config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def create_plots(
    metrics: Dict[str, Dict],
    paired_analysis: Dict,
    stat_tests: Dict,
    norm_sweep_data: Optional[Dict],
    concept_norms: Optional[np.ndarray],
    output_dir: Path,
):
    """Create all visualization plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12})

    # ── Plot 1: Detection rate comparison (main bar chart) ──────────────
    main_conditions = [
        c for c in [
            "real_concepts", "no_steering", "isotropic_noise",
            "per_concept_norm_matched", "shuffled_dims",
            "concept_subspace_noise", "success_subspace_noise",
            "orthogonal_noise", "covariance_matched",
        ]
        if c in metrics
    ]

    if main_conditions:
        for metric_key, metric_name, filename in [
            ("detection_rate", "Detection Rate", "detection_rate_comparison.png"),
            ("introspection_rate", "Introspection Rate (Detected + Identified)", "introspection_rate_comparison.png"),
        ]:
            fig, ax = plt.subplots(figsize=(max(14, len(main_conditions) * 1.5), 7))
            x = np.arange(len(main_conditions))
            bars = []
            for i, cond in enumerate(main_conditions):
                m = metrics[cond]
                rate = m.get(metric_key, 0)
                se = m.get(f"{metric_key}_se", 0)
                color = CONDITION_COLORS.get(cond, "#333333")
                bar = ax.bar(i, rate, color=color, alpha=0.85, edgecolor="black",
                             linewidth=0.5, yerr=1.96 * se, capsize=4)
                bars.append(bar)
                # Value label
                ax.text(i, rate + 1.96 * se + 0.02, f"{rate:.1%}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONDITION_LABELS.get(c, c) for c in main_conditions],
                rotation=45, ha="right", fontsize=9,
            )
            ax.set_ylabel(metric_name)
            ax.set_title(f"Exp 58: {metric_name} — Real Concepts vs Gaussian Noise Variants")
            ax.set_ylim(0, min(1.15, ax.get_ylim()[1] + 0.1))
            ax.axhline(y=0, color="black", linewidth=0.5)

            # Add significance stars
            real_rate = metrics.get("real_concepts", {}).get(metric_key, 0)
            for i, cond in enumerate(main_conditions):
                if cond in stat_tests:
                    p = stat_tests[cond].get("bonferroni_p", 1.0)
                    if p < 0.001:
                        star = "***"
                    elif p < 0.01:
                        star = "**"
                    elif p < 0.05:
                        star = "*"
                    else:
                        star = "ns"
                    rate = metrics[cond].get(metric_key, 0)
                    se = metrics[cond].get(f"{metric_key}_se", 0)
                    ax.text(i, rate + 1.96 * se + 0.06, star,
                            ha="center", va="bottom", fontsize=8, color="red")

            plt.tight_layout()
            fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {filename}")

    # ── Plot 2: Norm-detection curve ────────────────────────────────────
    # Collect all norm sweep conditions (both percentile-based and above-max)
    norm_sweep_conds_unsorted = [c for c in metrics if c.startswith("norm_sweep_")]
    if norm_sweep_conds_unsorted and concept_norms is not None:
        # Compute the actual norm for each sweep condition
        max_norm = float(np.max(concept_norms))
        cond_norm_pairs = []
        for cond in norm_sweep_conds_unsorted:
            if cond.startswith("norm_sweep_p"):
                pct = float(cond.split("_p")[1])
                norm_val = np.percentile(concept_norms, pct)
                label_str = f"p{cond.split('_p')[1]}"
            elif cond.startswith("norm_sweep_x"):
                mult_pct = int(cond.split("_x")[1])
                norm_val = max_norm * (mult_pct / 100.0)
                label_str = f"x{mult_pct}%"
            else:
                continue
            cond_norm_pairs.append((cond, norm_val, label_str))
        cond_norm_pairs.sort(key=lambda x: x[1])
        norm_sweep_conds = [c for c, _, _ in cond_norm_pairs]

        fig, ax = plt.subplots(figsize=(10, 6))
        sweep_norms = np.array([nv for _, nv, _ in cond_norm_pairs])
        sweep_det = np.array([metrics[c]["detection_rate"] for c, _, _ in cond_norm_pairs])
        sweep_se = np.array([metrics[c].get("detection_rate_se", 0) for c, _, _ in cond_norm_pairs])

        ax.plot(sweep_norms, sweep_det, "o-", color="#e91e63", linewidth=2, markersize=8)
        ax.fill_between(
            sweep_norms,
            sweep_det - 1.96 * sweep_se,
            sweep_det + 1.96 * sweep_se,
            alpha=0.2, color="#e91e63",
        )

        # Mark mean norms of success/failure concepts
        if "real_concepts" in metrics:
            mean_norm = float(np.mean(concept_norms))
            ax.axvline(mean_norm, color="#2ecc71", linestyle="--", linewidth=1.5,
                        label=f"Mean concept norm ({mean_norm:.0f})")

        # Add labels for each sweep point
        for (cond, nv, label_str), det in zip(cond_norm_pairs, sweep_det):
            ax.annotate(label_str, (nv, det), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

        ax.set_xlabel("Steering Vector Norm")
        ax.set_ylabel("Detection Rate")
        ax.set_title("Exp 58: Detection Rate vs Noise Norm (Fixed Random Direction)")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        fig.savefig(plots_dir / "norm_detection_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved norm_detection_curve.png")

    # ── Plot 3: Subspace comparison (grouped bar) ──────────────────────
    subspace_conds = [
        c for c in ["isotropic_noise", "concept_subspace_noise",
                     "success_subspace_noise", "orthogonal_noise"]
        if c in metrics
    ]
    if len(subspace_conds) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(subspace_conds))
        for i, cond in enumerate(subspace_conds):
            m = metrics[cond]
            color = CONDITION_COLORS.get(cond, "#333")
            det = m["detection_rate"]
            se = m.get("detection_rate_se", 0)
            ax.bar(i, det, color=color, alpha=0.85, edgecolor="black",
                   linewidth=0.5, yerr=1.96 * se, capsize=4)
            ax.text(i, det + 1.96 * se + 0.02, f"{det:.1%}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Add real_concepts reference line
        if "real_concepts" in metrics:
            ax.axhline(metrics["real_concepts"]["detection_rate"], color="#2ecc71",
                        linestyle="--", linewidth=2, label="Real concepts")
            ax.legend()

        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in subspace_conds],
            rotation=30, ha="right",
        )
        ax.set_ylabel("Detection Rate")
        ax.set_title("Exp 58: Subspace Decomposition — Where Does the Signal Live?")
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        fig.savefig(plots_dir / "subspace_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved subspace_comparison.png")

    # ── Plot 4: Paired delta vs norm (per-concept analysis) ────────────
    for compare_cond in ["per_concept_norm_matched", "shuffled_dims"]:
        if compare_cond not in paired_analysis:
            continue
        pa = paired_analysis[compare_cond]
        per_concept = pa.get("per_concept", {})
        if len(per_concept) < 5:
            continue

        concepts = sorted(per_concept.keys())
        deltas = [per_concept[c]["delta"] for c in concepts]
        norms_arr = [per_concept[c]["norm"] for c in concepts]

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(norms_arr, deltas, alpha=0.5, s=20, c="#3498db")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Concept Vector Norm")
        ax.set_ylabel("Detection Rate Delta (Real - Random)")
        cond_label = CONDITION_LABELS.get(compare_cond, compare_cond).replace("\n", " ")
        ax.set_title(
            f"Exp 58: Per-Concept Advantage of Real vs {cond_label}\n"
            f"r={pa['pearson_r_vs_norm']:.3f}, p={pa['pearson_p_vs_norm']:.2e} "
            f"(ρ={pa['spearman_r_vs_norm']:.3f})"
        )

        # Regression line
        if len(norms_arr) > 2:
            z = np.polyfit(norms_arr, deltas, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(norms_arr), max(norms_arr), 100)
            ax.plot(x_range, p(x_range), "r--", linewidth=1.5, alpha=0.7)

        plt.tight_layout()
        fname = f"paired_delta_vs_norm_{compare_cond}.png"
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Plot 5: Summary table ──────────────────────────────────────────
    summary_conds = [c for c in ALL_CONDITIONS if c in metrics and c != "norm_sweep"]
    # Also include norm_sweep sub-conditions (percentile-based and above-max)
    sweep_subs = [c for c in metrics if c.startswith("norm_sweep_")]
    # Sort: percentile-based first (by percentile), then multiplier-based (by multiplier)
    sweep_p = sorted([c for c in sweep_subs if c.startswith("norm_sweep_p")],
                     key=lambda c: float(c.split("_p")[1]))
    sweep_x = sorted([c for c in sweep_subs if c.startswith("norm_sweep_x")],
                     key=lambda c: float(c.split("_x")[1]))
    summary_conds += sweep_p + sweep_x

    if summary_conds:
        fig, ax = plt.subplots(figsize=(14, max(4, len(summary_conds) * 0.45)))
        ax.axis("off")

        headers = ["Condition", "Detection", "Introspection", "N vectors", "p vs real"]
        cell_text = []
        for cond in summary_conds:
            m = metrics[cond]
            det = f"{m['detection_rate']:.1%} ± {m.get('detection_rate_se', 0) * 1.96:.1%}"
            intro = f"{m['introspection_rate']:.1%} ± {m.get('introspection_rate_se', 0) * 1.96:.1%}"
            n_lbl = str(m.get("n_labels", ""))
            p_str = ""
            if cond in stat_tests:
                bp = stat_tests[cond].get("bonferroni_p", 1.0)
                p_str = f"{bp:.2e}" if bp < 0.05 else f"{bp:.3f}"
            label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
            cell_text.append([label, det, intro, n_lbl, p_str])

        table = ax.table(cellText=cell_text, colLabels=headers, loc="center",
                         cellLoc="center", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.auto_set_column_width(list(range(len(headers))))
        table.scale(1, 1.4)

        # Style header
        for j in range(len(headers)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Color code real_concepts and no_steering rows
        for i, cond in enumerate(summary_conds):
            if cond == "real_concepts":
                for j in range(len(headers)):
                    table[i + 1, j].set_facecolor("#d5f5e3")
            elif cond == "no_steering":
                for j in range(len(headers)):
                    table[i + 1, j].set_facecolor("#fadbd8")

        ax.set_title("Exp 58: Summary — Gaussian Noise vs Real Concept Vectors", fontsize=14, pad=20)
        plt.tight_layout()
        fig.savefig(plots_dir / "summary_table.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved summary_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Paths
    geometry_dir = Path(args.geometry_dir)
    steering_dir = Path(args.steering_dir)
    output_base = Path(args.output_dir)
    layer_idx = args.layer
    strength = args.strength
    config_folder = f"layer_{layer_idx}_strength_{strength}"
    output_dir = output_base / args.model / config_folder

    conditions = args.conditions if args.conditions else list(ALL_CONDITIONS)
    n_trial_numbers = args.n_trial_numbers
    samples_per_trial = args.samples_per_trial
    trials_per_vector = n_trial_numbers * samples_per_trial
    seed = args.seed

    # ── Plots-only mode ─────────────────────────────────────────────
    if args.plots_only:
        print("=" * 80)
        print("PLOTS-ONLY MODE: Regenerating plots from existing results")
        print("=" * 80)

        results_path = output_dir / "all_results.json"
        if not results_path.exists():
            print(f"Error: No results found at {results_path}")
            return

        with open(results_path, "r") as f:
            all_results = json.load(f).get("results", [])

        metrics = compute_aggregate_metrics(all_results)

        # Load concept vectors for paired analysis
        vectors_dir = steering_dir / args.model / "vectors" / f"layer_{layer_idx}"
        success_concepts, failure_concepts, _ = load_geometry_partition(
            geometry_dir, args.model, layer_idx, strength
        )
        vectors = load_concept_vectors(vectors_dir, success_concepts + failure_concepts)
        concept_norms = np.array([v.float().norm().item() for v in vectors.values()])

        paired_analysis = compute_paired_analysis(all_results, vectors, success_concepts)
        stat_tests = compute_statistical_tests(metrics)

        create_plots(metrics, paired_analysis, stat_tests, None, concept_norms, output_dir)
        print("Done!")
        return

    # ── Print banner ─────────────────────────────────────────────────
    print("=" * 80)
    print("EXPERIMENT 58: GAUSSIAN CONCEPT VECTORS")
    print("What Makes a Concept Vector Work?")
    print("=" * 80)
    print(f"Model:            {args.model}")
    print(f"Layer:            {layer_idx}")
    print(f"Strength:         {strength}")
    print(f"Trials per vector: {trials_per_vector} ({n_trial_numbers} trial_nums × {samples_per_trial} samples)")
    print(f"N random vectors:  {args.n_random_vectors} (for fixed-vector conditions)")
    print(f"PCA rank (k):      {args.pca_k}")
    print(f"Conditions:        {conditions}")
    print(f"Seed:              {seed}")
    print("=" * 80)

    # ── Step 1: Load partition ───────────────────────────────────────
    print("\n[1/8] Loading concept partition from experiment 04b (vector geometry)...")
    success_concepts, failure_concepts, partition_meta = load_geometry_partition(
        geometry_dir, args.model, layer_idx, strength
    )
    print(f"  Success: {len(success_concepts)}, Failure: {len(failure_concepts)}")

    # ── Step 2: Load concept vectors ─────────────────────────────────
    print("\n[2/8] Loading concept vectors from experiment 02 (steering evaluation)...")
    vectors_dir = steering_dir / args.model / "vectors" / f"layer_{layer_idx}"
    if not vectors_dir.exists():
        print(f"Error: Could not find vectors directory at {vectors_dir}")
        return

    all_concepts = success_concepts + failure_concepts
    vectors = load_concept_vectors(vectors_dir, all_concepts)
    print(f"  Loaded {len(vectors)} vectors")

    success_with_vecs = [c for c in success_concepts if c in vectors]
    failure_with_vecs = [c for c in failure_concepts if c in vectors]
    print(f"  Success with vectors: {len(success_with_vecs)}")
    print(f"  Failure with vectors: {len(failure_with_vecs)}")

    # Stack vectors
    succ_vecs = torch.stack([vectors[c].float() for c in success_with_vecs])
    all_vecs = torch.stack([vectors[c].float() for c in vectors.keys()])
    succ_norms = succ_vecs.norm(dim=1)
    all_norms = all_vecs.norm(dim=1)
    concept_norms_np = all_norms.numpy()
    mean_succ_norm = float(succ_norms.mean())
    d = succ_vecs.shape[1]
    print(f"  Hidden dim: {d}")
    print(f"  Mean success concept norm: {mean_succ_norm:.2f}")
    print(f"  Norm range: [{float(all_norms.min()):.2f}, {float(all_norms.max()):.2f}]")

    # ── Step 3: Compute PCA for subspace conditions ──────────────────
    print("\n[3/8] Computing PCA for subspace conditions...")
    k = min(args.pca_k, len(vectors) - 1, d)

    # PCA on all concept vectors
    all_vecs_centered = all_vecs - all_vecs.mean(dim=0)
    _, S_all, Vt_all = torch.linalg.svd(all_vecs_centered, full_matrices=False)
    basis_all = Vt_all[:k]  # [k, d]
    var_explained = (S_all[:k] ** 2).sum() / (S_all ** 2).sum()
    print(f"  All-concept PCA: top-{k} components explain {var_explained:.1%} variance")

    # PCA on success vectors only
    succ_centered = succ_vecs - succ_vecs.mean(dim=0)
    _, S_succ, Vt_succ = torch.linalg.svd(succ_centered, full_matrices=False)
    k_succ = min(k, len(success_with_vecs) - 1)
    basis_succ = Vt_succ[:k_succ]
    var_explained_succ = (S_succ[:k_succ] ** 2).sum() / (S_succ ** 2).sum()
    print(f"  Success PCA: top-{k_succ} components explain {var_explained_succ:.1%} variance")

    # ── Step 4: Generate all random vectors ──────────────────────────
    print("\n[4/8] Generating random vectors for all conditions...")
    generated_vectors = {}  # condition -> dict or list
    vector_properties = {}

    # Success concept vectors for per-concept conditions
    succ_vectors = {c: vectors[c] for c in success_with_vecs}

    if "isotropic_noise" in conditions:
        vecs = generate_isotropic_noise(d, mean_succ_norm, args.n_random_vectors, seed)
        labels = [f"iso_noise_{i}" for i in range(len(vecs))]
        generated_vectors["isotropic_noise"] = list(zip(labels, vecs))
        vector_properties["isotropic_noise"] = {
            "n_vectors": len(vecs),
            "target_norm": mean_succ_norm,
            "actual_norms": [float(v.norm()) for v in vecs],
        }
        print(f"  isotropic_noise: {len(vecs)} vectors, norm={mean_succ_norm:.2f}")

    if "per_concept_norm_matched" in conditions:
        noise_vecs = generate_norm_matched_noise(succ_vectors, seed + 1)
        generated_vectors["per_concept_norm_matched"] = noise_vecs
        print(f"  per_concept_norm_matched: {len(noise_vecs)} vectors (one per success concept)")

    if "shuffled_dims" in conditions:
        shuf_vecs = generate_shuffled_vectors(succ_vectors, seed + 2)
        generated_vectors["shuffled_dims"] = shuf_vecs
        print(f"  shuffled_dims: {len(shuf_vecs)} vectors (one per success concept)")

    if "concept_subspace_noise" in conditions:
        vecs, _ = generate_subspace_noise(all_vecs, k, mean_succ_norm, args.n_random_vectors, seed + 3)
        labels = [f"concept_sub_{i}" for i in range(len(vecs))]
        generated_vectors["concept_subspace_noise"] = list(zip(labels, vecs))
        # Verify subspace containment
        residuals = [float((v.float() - (basis_all.T @ (basis_all @ v.float())).squeeze()).norm() / v.float().norm()) for v in vecs]
        vector_properties["concept_subspace_noise"] = {
            "n_vectors": len(vecs),
            "pca_k": k,
            "mean_residual_frac": float(np.mean(residuals)),
        }
        print(f"  concept_subspace_noise: {len(vecs)} vectors, k={k}, mean residual frac={np.mean(residuals):.6f}")

    if "success_subspace_noise" in conditions:
        vecs, _ = generate_subspace_noise(succ_vecs, k_succ, mean_succ_norm, args.n_random_vectors, seed + 4)
        labels = [f"succ_sub_{i}" for i in range(len(vecs))]
        generated_vectors["success_subspace_noise"] = list(zip(labels, vecs))
        print(f"  success_subspace_noise: {len(vecs)} vectors, k={k_succ}")

    if "orthogonal_noise" in conditions:
        vecs = generate_orthogonal_noise(basis_all, d, mean_succ_norm, args.n_random_vectors, seed + 5)
        labels = [f"orth_noise_{i}" for i in range(len(vecs))]
        generated_vectors["orthogonal_noise"] = list(zip(labels, vecs))
        # Verify orthogonality
        projections = [float((basis_all @ v.float()).norm() / v.float().norm()) for v in vecs]
        vector_properties["orthogonal_noise"] = {
            "n_vectors": len(vecs),
            "pca_k": k,
            "mean_projection_frac": float(np.mean(projections)),
        }
        print(f"  orthogonal_noise: {len(vecs)} vectors, mean proj onto subspace={np.mean(projections):.6f}")

    if "covariance_matched" in conditions:
        vecs = generate_covariance_matched(succ_vecs, args.n_random_vectors, seed + 6)
        labels = [f"cov_match_{i}" for i in range(len(vecs))]
        generated_vectors["covariance_matched"] = list(zip(labels, vecs))
        cov_norms = [float(v.norm()) for v in vecs]
        vector_properties["covariance_matched"] = {
            "n_vectors": len(vecs),
            "norm_mean": float(np.mean(cov_norms)),
            "norm_std": float(np.std(cov_norms)),
        }
        print(f"  covariance_matched: {len(vecs)} vectors, norm mean={np.mean(cov_norms):.2f}")

    if "norm_sweep" in conditions:
        above_max_multipliers = [1.5, 2.0]
        sweep = generate_norm_sweep_vectors(
            d, seed + 7, args.norm_percentiles, concept_norms_np, 1,
            above_max_multipliers=above_max_multipliers,
        )
        for label, vec_list in sweep.items():
            generated_vectors[label] = [(label, vec_list[0])]
        pct_norms = np.percentile(concept_norms_np, args.norm_percentiles)
        max_norm = float(np.max(concept_norms_np))
        above_norms = [max_norm * m for m in above_max_multipliers]
        vector_properties["norm_sweep"] = {
            "percentiles": args.norm_percentiles,
            "norms": [float(n) for n in pct_norms],
            "above_max_multipliers": above_max_multipliers,
            "above_max_norms": above_norms,
        }
        all_level_strs = [f'p{int(p)}={n:.0f}' for p, n in zip(args.norm_percentiles, pct_norms)]
        all_level_strs += [f'x{int(m*100)}={n:.0f}' for m, n in zip(above_max_multipliers, above_norms)]
        print(f"  norm_sweep: {len(sweep)} norm levels: {all_level_strs}")

    # ── Step 5: Load model ───────────────────────────────────────────
    print("\n[5/8] Loading model...")
    model = load_model(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
    )

    # Initialize LLM judge
    judge = None
    if not args.no_llm_judge:
        try:
            judge = LLMJudge()
            print("  LLM judge initialized")
        except Exception as e:
            print(f"  Warning: Could not initialize LLM judge: {e}")

    # ── Step 6: Check for existing results (resume) ──────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_results = []
    completed_keys = set()

    if not args.overwrite:
        print("\n[6/8] Checking for existing results...")
        existing_results, completed_keys = load_existing_results(output_dir)
        if completed_keys:
            print(f"  Found {len(completed_keys)} completed (condition, label) pairs — will skip")
    else:
        print("\n[6/8] Starting fresh (--overwrite)")

    all_results = list(existing_results)

    # ── Step 7: Run experiments ──────────────────────────────────────
    print("\n[7/8] Running experiments...")

    # Build work items: (condition, label, vector_or_none)
    work_items = []

    # No steering (negative control)
    if "no_steering" in conditions and "no_steering::no_steering" not in completed_keys:
        work_items.append(("no_steering", "no_steering", None))

    # Per-concept conditions
    for cond in ["real_concepts", "per_concept_norm_matched", "shuffled_dims"]:
        if cond not in conditions:
            continue
        for concept in success_with_vecs:
            key = f"{cond}::{concept}"
            if key in completed_keys:
                continue
            if cond == "real_concepts":
                vec = vectors[concept]
            elif cond == "per_concept_norm_matched":
                vec = generated_vectors.get("per_concept_norm_matched", {}).get(concept)
            elif cond == "shuffled_dims":
                vec = generated_vectors.get("shuffled_dims", {}).get(concept)
            if vec is not None:
                work_items.append((cond, concept, vec))

    # Fixed-vector conditions
    for cond in FIXED_VECTOR_CONDITIONS:
        if cond not in conditions:
            continue
        vec_list = generated_vectors.get(cond, [])
        for label, vec in vec_list:
            key = f"{cond}::{label}"
            if key not in completed_keys:
                work_items.append((cond, label, vec))

    # Norm sweep sub-conditions
    if "norm_sweep" in conditions:
        for sub_cond in sorted(c for c in generated_vectors if c.startswith("norm_sweep_")):
            for label, vec in generated_vectors[sub_cond]:
                key = f"{sub_cond}::{label}"
                if key not in completed_keys:
                    work_items.append((sub_cond, label, vec))

    print(f"  Total work items: {len(work_items)}")
    if not work_items:
        print("  Nothing to do — all conditions already completed!")
    else:
        # Shuffle for interleaving
        rng = random.Random(seed)
        rng.shuffle(work_items)

        items_done = 0
        for wi_idx, (cond, label, vec) in enumerate(tqdm(work_items, desc="Running experiments")):
            if args.verbose:
                print(f"\n  [{wi_idx+1}/{len(work_items)}] condition={cond}, label={label}")

            if cond == "no_steering":
                # No-steering control
                results = run_no_steering_experiment(
                    model=model,
                    n_trial_numbers=n_trial_numbers,
                    samples_per_trial=samples_per_trial,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            else:
                # Steered experiment
                results = run_steering_experiment(
                    model=model,
                    label=label,
                    steering_vector=vec,
                    layer_idx=layer_idx,
                    strength=strength,
                    n_trial_numbers=n_trial_numbers,
                    samples_per_trial=samples_per_trial,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    condition=cond,
                )

            # LLM judge evaluation
            if judge is not None and results:
                try:
                    original_prompts = [
                        f"Trial {r['trial']}: Do you detect an injected thought? "
                        f"If so, what is the injected thought about?"
                        for r in results
                    ]
                    results = batch_evaluate(judge, results, original_prompts)

                    # For noise conditions, identification grading is meaningless
                    # because there is no real concept to identify (the label is
                    # something like "iso_noise_0"). Override identification results
                    # to avoid nonsensical LLM judge queries being used downstream.
                    if is_noise_condition(cond):
                        for r in results:
                            if "evaluations" in r and "correct_concept_identification" in r["evaluations"]:
                                r["evaluations"]["correct_concept_identification"] = {
                                    "correct_identification": False,
                                    "grade": 0,
                                    "raw_response": "N/A (noise condition — no real concept to identify)",
                                }
                except Exception as e:
                    if args.verbose:
                        print(f"    Warning: LLM judge failed: {e}")

            all_results.extend(results)
            items_done += 1

            # Iterative save
            if items_done % args.plot_interval == 0 or wi_idx == len(work_items) - 1:
                with open(output_dir / "all_results.json", "w") as f:
                    json.dump({
                        "results": all_results,
                        "last_updated": datetime.now().isoformat(),
                    }, f, indent=2)
                if args.verbose:
                    print(f"    Saved {len(all_results)} results so far")

    # ── Step 8: Analysis & Plots ─────────────────────────────────────
    print("\n[8/8] Computing metrics, tests, and creating plots...")

    metrics = compute_aggregate_metrics(all_results)
    paired_analysis = compute_paired_analysis(all_results, vectors, success_with_vecs)
    stat_tests = compute_statistical_tests(metrics)

    # Save config
    config = {
        "model": args.model,
        "layer": layer_idx,
        "strength": strength,
        "n_trial_numbers": n_trial_numbers,
        "samples_per_trial": samples_per_trial,
        "trials_per_vector": trials_per_vector,
        "n_random_vectors": args.n_random_vectors,
        "pca_k": args.pca_k,
        "norm_percentiles": args.norm_percentiles,
        "conditions": conditions,
        "seed": seed,
        "use_llm_judge": not args.no_llm_judge,
        "n_success_concepts": len(success_with_vecs),
        "n_failure_concepts": len(failure_with_vecs),
        "partition": partition_meta,
        "timestamp": datetime.now().isoformat(),
    }

    save_results(all_results, metrics, paired_analysis, stat_tests, vector_properties, config, output_dir)

    # Create plots
    norm_sweep_data = vector_properties.get("norm_sweep") if "norm_sweep" in conditions else None
    create_plots(metrics, paired_analysis, stat_tests, norm_sweep_data, concept_norms_np, output_dir)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary_order = [
        "real_concepts", "no_steering", "isotropic_noise",
        "per_concept_norm_matched", "shuffled_dims",
        "concept_subspace_noise", "success_subspace_noise",
        "orthogonal_noise", "covariance_matched",
    ]
    # Add norm sweep (percentile-based then above-max)
    norm_sweep_keys = sorted(
        [c for c in metrics if c.startswith("norm_sweep_p")],
        key=lambda c: float(c.split("_p")[1]),
    ) + sorted(
        [c for c in metrics if c.startswith("norm_sweep_x")],
        key=lambda c: float(c.split("_x")[1]),
    )

    print(f"\n{'Condition':<35} {'Detection':>12} {'Introspection':>15} {'N':>6}")
    print("-" * 72)
    for cond in summary_order + norm_sweep_keys:
        if cond not in metrics:
            continue
        m = metrics[cond]
        det = f"{m['detection_rate']:.1%}"
        intro = f"{m['introspection_rate']:.1%}"
        n = m.get("n", 0)
        label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
        print(f"  {label:<33} {det:>12} {intro:>15} {n:>6}")

    # Print paired analysis
    if paired_analysis:
        print("\nPAIRED ANALYSIS (Real - Random per concept):")
        for cond, pa in paired_analysis.items():
            label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
            print(f"  {label}:")
            print(f"    Mean delta: {pa['mean_delta']:.3f} ± {pa['std_delta']:.3f}")
            print(f"    Correlation with norm: r={pa['pearson_r_vs_norm']:.3f}, p={pa['pearson_p_vs_norm']:.2e}")

    # Print key statistical tests
    if stat_tests:
        print("\nSTATISTICAL TESTS (vs real_concepts, Bonferroni corrected):")
        for cond in summary_order:
            if cond not in stat_tests:
                continue
            t = stat_tests[cond]
            label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
            sig = "***" if t["bonferroni_p"] < 0.001 else "**" if t["bonferroni_p"] < 0.01 else "*" if t["bonferroni_p"] < 0.05 else "ns"
            print(f"  {label:<33} d={t['cohens_d']:+.2f}  p={t['bonferroni_p']:.2e}  {sig}")

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
