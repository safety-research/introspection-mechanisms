#!/usr/bin/env python3
"""
Bidirectional Steering Analysis

Tests whether introspection steering vectors form a structured subspace beyond the
linear mean-difference direction. Two main analyses:

1. **Bidirectional steering** (Section 4.2): Test 1000 S-S and 1000 F-F same-category
   concept pairs. Steering with (A-B) and (B-A) both trigger detection in 23.3% of
   S-S pairs vs 3.2% of F-F pairs, indicating a non-linear subspace structure.
   Figure: bidirectional-steering.pdf

2. **Delta-PC extraction and threshold sweep** (Section 4.3): PCA on success-concept
   difference vectors with mean-diff projected out yields orthogonal delta-PCs.
   Steering along each triggers detection with a distinct profile.
   Figure: part of introspection-nonlinearity.pdf panel (c)

Usage:
    python 04c_bidirectional_steering.py -m gemma3_27b --same-category-swaps
    python 04c_bidirectional_steering.py -m gemma3_27b --same-category-swaps --plots-only
    python 04c_bidirectional_steering.py -m gemma3_27b --same-category-swaps --skip-threshold-sweep
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Import local utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model_utils import load_model, ModelWrapper
from steering_utils import run_steered_introspection_test_batch, run_unsteered_introspection_test_batch
from eval_utils import LLMJudge, batch_evaluate

# ==============================================================================
# Configuration and Defaults
# ==============================================================================

DEFAULT_MODEL = "gemma3_27b"
# 04b_vector_geometry.py writes the ridge direction (primary_axis.pt) and all
# decomposition artifacts (mean_diff, LDA, group centroids) into a single
# per-config subdir — there is no separate decomposition directory.
DEFAULT_GEOMETRY_DIR = "analysis/04b_vector_geometry"
DEFAULT_STEERING_DIR = "analysis/02b_steering_500_concepts"
DEFAULT_OUTPUT_DIR = "analysis/04c_bidirectional_steering"

DEFAULT_LAYER = 37
DEFAULT_STRENGTH = 4.0
DEFAULT_N_TRIAL_NUMBERS = 10
DEFAULT_SAMPLES_PER_TRIAL = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_N_PCS = 10
DEFAULT_BIDIRECTIONAL_THRESHOLD = 200
DEFAULT_PROJ_BUCKET_SIZE = 200
DEFAULT_PLOT_UPDATE_INTERVAL = 2


# ==============================================================================
# Incremental Save/Load Utilities
# ==============================================================================

def save_json_atomic(data: Any, path: Path):
    """Save JSON atomically to avoid corruption on crash."""
    import uuid
    path = Path(path)
    temp_path = path.parent / f".{path.stem}_{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        temp_path.rename(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def cleanup_temp_files(directory: Path):
    """Clean up orphaned .tmp files from previous crashed runs."""
    if not directory.exists():
        return
    for tmp_file in directory.glob("*.tmp"):
        try:
            tmp_file.unlink()
        except Exception:
            pass
    for tmp_file in directory.glob(".*.tmp"):
        try:
            tmp_file.unlink()
        except Exception:
            pass


def load_json_safe(path: Path, default: Any = None) -> Any:
    """Load JSON with fallback to default if file doesn't exist or is corrupted."""
    if not path.exists():
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load checkpoint for resume functionality."""
    return load_json_safe(checkpoint_path, default={
        "completed_items": [],
        "last_update": None,
        "status": "not_started"
    })


def save_checkpoint(checkpoint_path: Path, completed_items: List[str], status: str = "in_progress"):
    """Save checkpoint for resume functionality."""
    checkpoint = {
        "completed_items": completed_items,
        "last_update": datetime.now().isoformat(),
        "status": status
    }
    save_json_atomic(checkpoint, checkpoint_path)


# ==============================================================================
# Ridge Direction Loading
# ==============================================================================

def load_ridge_direction(
    geometry_dir: Path,
    model_name: str,
    layer: int,
    strength: float,
    metric: str = "detection_rate",
) -> torch.Tensor:
    """Load the Ridge direction saved by 04b_vector_geometry.py.

    Tries ``primary_axis.pt`` first (alias written by 04b), then falls back to
    ``introspection_direction_ridge_regression.pt`` under the same config dir.
    """
    base = geometry_dir / model_name / f"layer_{layer}_strength_{strength}" / metric
    candidates = [base / "primary_axis.pt", base / "introspection_direction_ridge_regression.pt"]

    ridge_path = next((p for p in candidates if p.exists()), None)
    if ridge_path is None:
        raise FileNotFoundError(
            f"Ridge direction not found at any of: {', '.join(str(p) for p in candidates)}. "
            f"Run 04b_vector_geometry.py first."
        )

    w_ridge = torch.load(ridge_path, map_location='cpu', weights_only=True)
    w_ridge = w_ridge.float()
    w_ridge = w_ridge / w_ridge.norm()
    return w_ridge


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bidirectional Steering Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model and paths
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--geometry-dir", type=str, default=DEFAULT_GEOMETRY_DIR,
                        help="Path to experiment 04b (vector geometry) results")
    parser.add_argument("--steering-dir", type=str, default=DEFAULT_STEERING_DIR,
                        help="Path to experiment 02 (steering evaluation) results")
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")

    # Mode selection
    parser.add_argument("--same-category-swaps", action="store_true",
                        help="Run same-category bidirectional swaps (S-S and F-F pairs)")
    parser.add_argument("--skip-threshold-sweep", action="store_true",
                        help="Skip the subspace threshold sweep after bidirectional swaps")
    parser.add_argument("--balanced-partition", action="store_true",
                        help="Use balanced partition (categories with both success/failure)")

    # Experiment parameters
    parser.add_argument("-l", "--layers", nargs="+", type=int, default=[37, 29],
                        help="Layer indices for steering (default: [37, 29])")
    parser.add_argument("-s", "--strengths", nargs="+", type=float, default=[4.0],
                        help="Steering strengths (default: [4.0])")
    parser.add_argument("--n-trial-numbers", type=int, default=DEFAULT_N_TRIAL_NUMBERS,
                        help=f"Number of unique trial prompts 1-N (default: {DEFAULT_N_TRIAL_NUMBERS})")
    parser.add_argument("--samples-per-trial", type=int, default=DEFAULT_SAMPLES_PER_TRIAL,
                        help=f"Samples per trial number (default: {DEFAULT_SAMPLES_PER_TRIAL})")
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("-mt", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature (default: {DEFAULT_TEMPERATURE})")

    # Analysis parameters
    parser.add_argument("--n-pcs", type=int, default=DEFAULT_N_PCS,
                        help=f"Number of principal components (default: {DEFAULT_N_PCS})")
    parser.add_argument("--bidirectional-threshold", type=float, default=DEFAULT_BIDIRECTIONAL_THRESHOLD,
                        help=f"Projection matching tolerance for pairing (default: {DEFAULT_BIDIRECTIONAL_THRESHOLD})")
    parser.add_argument("--pairing-direction", type=str, default="ridge",
                        choices=["ridge", "mean-diff", "none"],
                        help="Direction for concept pairing (default: ridge)")
    parser.add_argument("--max-num-pairs", type=int, default=1000,
                        help="Max number of pairs for each of S-S and F-F (default: 1000)")
    parser.add_argument("--proj-bucket-size", type=float, default=DEFAULT_PROJ_BUCKET_SIZE,
                        help=f"Bucket size for proj_diff when sorting pairs (default: {DEFAULT_PROJ_BUCKET_SIZE})")
    parser.add_argument("--within-category-bucket-size", type=float, default=None,
                        help="Bucket size for within-category pairs")
    parser.add_argument("--max-concept-reuse", type=int, default=1,
                        help="Max times a concept can be reused in pairs (default: 1)")
    parser.add_argument("--plot-interval", type=int, default=DEFAULT_PLOT_UPDATE_INTERVAL,
                        help=f"Plot update interval (default: {DEFAULT_PLOT_UPDATE_INTERVAL})")
    parser.add_argument("--norm-normalize", action="store_true",
                        help="Normalize difference vectors to mean concept vector norm")
    parser.add_argument("--min-success-baseline", type=float, default=None,
                        help="Min baseline detection rate for success concepts in S-S pairs")
    parser.add_argument("--max-failure-baseline", type=float, default=None,
                        help="Max baseline detection rate for failure concepts in F-F pairs")

    # Threshold sweep parameters
    parser.add_argument("--sweep-max-pairs", type=int, default=30,
                        help="Max pairs for individual threshold sweep (default: 30)")
    parser.add_argument("--sweep-n-trials", type=int, default=10,
                        help="Trial numbers per alpha in sweep (default: 10)")
    parser.add_argument("--sweep-samples-per-trial", type=int, default=10,
                        help="Samples per trial in sweep (default: 10)")

    # System
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device")
    parser.add_argument("-dt", "--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"], help="Dtype")
    parser.add_argument("-q", "--quantization", type=str, default=None,
                        choices=["8bit", "4bit"], help="Quantization")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Disable LLM judge evaluation")
    parser.add_argument("--judge-batch-pairs", type=int, default=8,
                        help="Number of pairs to accumulate before batch-evaluating (default: 8)")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-pairing-seeds", type=int, default=1,
                        help="Number of random seeds for 'none' pairing direction (default: 1)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only regenerate plots from existing results")

    return parser.parse_args()


# ==============================================================================
# Data Loading Utilities
# ==============================================================================

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
        print(f"Warning: Missing vectors for {len(missing)} concepts: {missing[:5]}...")

    return vectors


def load_geometry_partition(
    geometry_dir: Path,
    model_name: str,
    layer: int,
    strength: float,
    balanced: bool = False,
    metric: str = "detection_rate"
) -> Tuple[List[str], List[str], dict]:
    """Load success/failure partition from experiment 04b (vector geometry)'s subspace_analysis.json or balanced_partition.json."""
    layer_strength_dir = geometry_dir / model_name / f"layer_{layer}_strength_{strength}" / metric

    if balanced:
        partition_path = layer_strength_dir / "balanced_partition.json"
        if not partition_path.exists():
            raise FileNotFoundError(f"Could not find balanced partition at {partition_path}")

        with open(partition_path, 'r') as f:
            data = json.load(f)

        success_concepts = data.get("success_concepts", [])
        failure_concepts = data.get("failure_concepts", [])

        metadata = {
            "balanced_categories": data.get("balanced_categories", []),
            "original_success_count": data.get("original_success_count"),
            "original_failure_count": data.get("original_failure_count"),
            "source": str(partition_path),
            "is_balanced": True,
            "metric_folder": metric,
        }
    else:
        subspace_path = layer_strength_dir / "subspace_analysis.json"
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
            "is_balanced": False,
            "metric_folder": metric,
        }

    return success_concepts, failure_concepts, metadata


def load_baseline_detection_rates(
    steering_dir: Path,
    model_name: str,
    layer: int,
    strength: float,
    success_concepts: List[str],
    failure_concepts: List[str]
) -> Tuple[float, float, Dict[str, float]]:
    """Load baseline detection rates from experiment 02 (steering evaluation) steering experiment results."""
    results_dir = steering_dir / model_name
    layer_results_path = results_dir / f"layer_{layer}_strength_{strength}" / "results.json"

    if layer_results_path.exists():
        results_path = layer_results_path
    else:
        results_files = list(results_dir.glob("layer_*/results.json"))
        if not results_files:
            print(f"  Warning: No experiment 02 (steering evaluation) results found in {results_dir}, using default baselines")
            return 0.8, 0.2, {}
        results_path = results_files[0]

    print(f"  Loading baseline detection rates from {results_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    per_concept_rates = {}
    concept_trials = {}

    for result in data.get("results", []):
        concept = result.get("concept")
        if concept is None:
            continue

        trial_type = result.get("trial_type", "")
        if trial_type != "injection":
            continue

        evals = result.get("evaluations", {})
        claims = evals.get("claims_detection", {})
        detected = claims.get("claims_detection", False)

        if concept not in concept_trials:
            concept_trials[concept] = []
        concept_trials[concept].append(1 if detected else 0)

    for concept, trials in concept_trials.items():
        per_concept_rates[concept] = np.mean(trials)

    success_set = set(success_concepts)
    failure_set = set(failure_concepts)

    success_rates = [per_concept_rates[c] for c in success_set if c in per_concept_rates]
    failure_rates = [per_concept_rates[c] for c in failure_set if c in per_concept_rates]

    D_S = np.mean(success_rates) if success_rates else 0.8
    D_F = np.mean(failure_rates) if failure_rates else 0.2

    print(f"  Baseline detection rates: D_S={D_S:.3f} (n={len(success_rates)}), D_F={D_F:.3f} (n={len(failure_rates)})")

    return D_S, D_F, per_concept_rates


# ==============================================================================
# Orthogonalization Utilities
# ==============================================================================

def compute_orthogonalized_vectors(
    vectors: Dict[str, torch.Tensor],
    mean_diff: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Orthogonalize all concept vectors with respect to mean-diff direction."""
    mean_diff_norm = mean_diff / mean_diff.norm()
    orth_vectors = {}

    for concept, vec in vectors.items():
        proj = torch.dot(vec.flatten(), mean_diff_norm.flatten())
        orth_vec = vec - proj * mean_diff_norm.view_as(vec)
        orth_vectors[concept] = orth_vec

    return orth_vectors


def compute_subspace_pcs(
    vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    n_pcs: int = 10
) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute principal components for a group of concept vectors."""
    from sklearn.decomposition import PCA

    vecs = torch.stack([vectors[c].flatten() for c in concepts if c in vectors])
    vecs_np = vecs.float().numpy()
    vecs_centered = vecs_np - vecs_np.mean(axis=0)

    pca = PCA(n_components=min(n_pcs, len(vecs_centered)))
    pca.fit(vecs_centered)

    pcs = torch.tensor(pca.components_, dtype=vecs.dtype)
    explained_variance = pca.explained_variance_ratio_

    return pcs, explained_variance


# ==============================================================================
# Same-Category Pair Finding
# ==============================================================================

def find_matched_pairs_same_category(
    concepts: List[str],
    vectors: Dict[str, torch.Tensor],
    pairing_direction: torch.Tensor,
    detection_rates: Dict[str, float],
    threshold: float = 200.0,
    bucket_size: float = 200.0,
    max_reuse: int = 1,
    category_name: str = "unknown"
) -> Tuple[List[Tuple[str, str, float, float, float, float]], Dict[str, Any]]:
    """
    Find pairs within the SAME category (S-S or F-F) with similar projections
    onto the pairing direction.

    Uses bucketed sorting to prioritize small detection-rate gaps within each
    proj_diff bucket.

    Returns:
        Tuple of:
        - List of (concept_A, concept_B, proj_A, proj_B, det_A, det_B)
        - Dict with cluster information
    """
    # Compute projections onto pairing direction
    concept_projs = {}
    pairing_dir_norm = (pairing_direction / pairing_direction.norm()).float()

    for c in concepts:
        if c in vectors:
            proj = torch.dot(vectors[c].flatten().float(), pairing_dir_norm.flatten()).item()
            det = detection_rates.get(c, 0)
            concept_projs[c] = (proj, det)

    # Build all candidate pairs
    candidates = []
    concept_list = list(concept_projs.keys())
    for i, c1 in enumerate(concept_list):
        for c2 in concept_list[i+1:]:
            p1, d1 = concept_projs[c1]
            p2, d2 = concept_projs[c2]
            proj_diff = abs(p1 - p2)
            if proj_diff < threshold:
                candidates.append((proj_diff, c1, c2, p1, p2, d1, d2))

    # Sort by proj_diff or by (bucket, gap)
    if bucket_size is None:
        candidates.sort(key=lambda x: x[0])
    else:
        candidates = [
            (int(proj_diff // bucket_size), abs(d1 - d2), proj_diff, c1, c2, p1, p2, d1, d2)
            for proj_diff, c1, c2, p1, p2, d1, d2 in candidates
        ]
        candidates.sort(key=lambda x: (x[0], x[1]))
        candidates = [(x[2], x[3], x[4], x[5], x[6], x[7], x[8]) for x in candidates]

    # Match with limited reuse, randomize A/B order
    rng = random.Random(42)
    usage = defaultdict(int)
    matched_pairs = []

    for proj_diff, c1, c2, p1, p2, d1, d2 in candidates:
        if usage[c1] < max_reuse and usage[c2] < max_reuse:
            if rng.random() < 0.5:
                matched_pairs.append((c1, c2, p1, p2, d1, d2))
            else:
                matched_pairs.append((c2, c1, p2, p1, d2, d1))
            usage[c1] += 1
            usage[c2] += 1

    # Build cluster info via union-find
    pair_to_idx = {(p[0], p[1]): i for i, p in enumerate(matched_pairs)}
    parent = list(range(len(matched_pairs)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    concept_to_pairs = defaultdict(list)
    for i, (c1, c2, *_) in enumerate(matched_pairs):
        concept_to_pairs[c1].append(i)
        concept_to_pairs[c2].append(i)

    for pairs in concept_to_pairs.values():
        for i in range(1, len(pairs)):
            union(pairs[0], pairs[i])

    cluster_ids = [find(i) for i in range(len(matched_pairs))]
    unique_clusters = sorted(set(cluster_ids))
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    cluster_ids = [cluster_map[c] for c in cluster_ids]

    cluster_info = {
        "category": category_name,
        "n_pairs": len(matched_pairs),
        "n_clusters": len(unique_clusters),
        "cluster_ids": cluster_ids,
        "cluster_sizes": [cluster_ids.count(i) for i in range(len(unique_clusters))],
        "max_reuse": max_reuse,
        "usage": dict(usage),
        "effective_n": len(unique_clusters),
    }

    return matched_pairs, cluster_info


def find_random_pairs_same_category(
    concepts: List[str],
    vectors: Dict[str, torch.Tensor],
    detection_rates: Dict[str, float],
    max_pairs: int = None,
    category_name: str = "unknown",
    seed: int = 42,
    n_seeds: int = 1,
) -> Tuple[List[Tuple[str, str, float, float, float, float]], Dict[str, Any]]:
    """
    Find random pairs within the SAME category (S-S or F-F) without concept reuse.

    Multiple seeds can be used to generate more pairs (each seed produces N/2 pairs,
    deduplicated across seeds).

    Returns:
        Tuple of:
        - List of (concept_A, concept_B, proj_A, proj_B, det_A, det_B)
        - Dict with cluster information
    """
    valid_concepts = [c for c in concepts if c in vectors]

    seen_pairs = set()
    matched_pairs = []

    for seed_offset in range(n_seeds):
        current_seed = seed + seed_offset
        rng = random.Random(current_seed)
        shuffled = valid_concepts.copy()
        rng.shuffle(shuffled)

        for i in range(0, len(shuffled) - 1, 2):
            c1 = shuffled[i]
            c2 = shuffled[i + 1]
            pair_key = frozenset([c1, c2])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                d1 = detection_rates.get(c1, 0)
                d2 = detection_rates.get(c2, 0)
                matched_pairs.append((c1, c2, 0.0, 0.0, d1, d2))

    if n_seeds > 1:
        print(f"    {category_name}: {n_seeds} seeds produced {len(matched_pairs)} unique pairs "
              f"(from {n_seeds * (len(valid_concepts) // 2)} total before dedup)")

    if max_pairs is not None and len(matched_pairs) > max_pairs:
        matched_pairs = matched_pairs[:max_pairs]

    # Each pair is its own cluster since no reuse
    usage = {}
    for pair in matched_pairs:
        for c in [pair[0], pair[1]]:
            usage[c] = usage.get(c, 0) + 1

    cluster_info = {
        "category": category_name,
        "n_pairs": len(matched_pairs),
        "n_clusters": len(matched_pairs),
        "cluster_ids": list(range(len(matched_pairs))),
        "cluster_sizes": [1] * len(matched_pairs),
        "max_reuse": max(usage.values()) if usage else 1,
        "usage": usage,
        "effective_n": len(matched_pairs),
        "n_seeds": n_seeds,
    }

    return matched_pairs, cluster_info


# ==============================================================================
# Bidirectional Swap Plotting
# ==============================================================================

def plot_bidirectional_progress_same_category(
    results_dir: Path,
    ss_results: List[Dict],
    ff_results: List[Dict],
    partition_threshold: float = 0.32,
    seed: int = 42,
):
    """
    Plot same-category bidirectional swap results comparing S-S vs F-F.

    Panel 1: Scatter of (A-B) vs (B-A) detection rates.
    Panel 2: Summary bar chart with baseline, steered mean/max, and bidirectional rate.

    Args:
        partition_threshold: Detection rate threshold separating success from failure.
        seed: Random seed for A/B randomization.
    """
    if not ss_results and not ff_results:
        return

    # Randomize A/B order for each pair (coin flip)
    rng = random.Random(seed)

    def randomize_ab_order(results: List[Dict]) -> List[Dict]:
        randomized = []
        for p in results:
            if rng.random() < 0.5:
                randomized.append(p)
            else:
                swapped = p.copy()
                swapped["a_minus_b_detection"] = p.get("b_minus_a_detection", 0)
                swapped["b_minus_a_detection"] = p.get("a_minus_b_detection", 0)
                randomized.append(swapped)
        return randomized

    ss_results = randomize_ab_order(ss_results) if ss_results else ss_results
    ff_results = randomize_ab_order(ff_results) if ff_results else ff_results

    # Styling
    import plot_style
    plot_style.set_defaults(matplotlib=True, plotly=False, pretty=False, install_brand_fonts=False)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.76),
                             gridspec_kw={'width_ratios': [1.0, 1.0]})

    colors = {"S-S": plot_style.AQUA_500, "F-F": plot_style.CLAY}
    jitter_amount = 0.02
    np.random.seed(42)

    # Panel 1: Scatter plot
    ax1 = axes[0]

    for cat_name, cat_results in [("S-S", ss_results), ("F-F", ff_results)]:
        if not cat_results:
            continue
        x = np.array([p.get("a_minus_b_detection", 0) for p in cat_results])
        y = np.array([p.get("b_minus_a_detection", 0) for p in cat_results])
        x_jittered = x + np.random.uniform(-jitter_amount, jitter_amount, len(x))
        y_jittered = y + np.random.uniform(-jitter_amount, jitter_amount, len(y))
        display_name = cat_name.replace("-", "\u2212")
        ax1.scatter(x_jittered, y_jittered, c=colors[cat_name], marker='o',
                   alpha=0.7, s=60, edgecolors='black', linewidths=0.5,
                   label=f'{display_name} pairs')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1.5)
    ax1.set_xlabel('(A\u2212B) detection rate', fontsize=28, labelpad=10)
    ax1.set_ylabel('(B\u2212A) detection rate', fontsize=28, labelpad=10)
    ax1.legend(loc='upper left', fontsize=19.5, framealpha=0.95, handletextpad=0.4,
               handlelength=0.9)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(axis='both', labelsize=22)

    # Panel 2: Summary bar chart
    ax2 = axes[1]
    bidir_threshold = partition_threshold

    def binomial_ci(successes, n, z=1.96):
        if n == 0:
            return 0, 0
        p = successes / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return max(0, center - spread), min(1, center + spread)

    summary_data = []

    for cat_name, cat_results in [("S-S", ss_results), ("F-F", ff_results)]:
        if cat_results:
            n = len(cat_results)

            baseline_max_values = [max(p.get("baseline_det_A", 0), p.get("baseline_det_B", 0)) for p in cat_results]
            baseline_max_mean = np.mean(baseline_max_values)
            baseline_max_se = np.std(baseline_max_values, ddof=1) / np.sqrt(n) if n > 1 else 0

            steered_mean_values = [(p.get("a_minus_b_detection", 0) + p.get("b_minus_a_detection", 0)) / 2 for p in cat_results]
            steered_mean_mean = np.mean(steered_mean_values)
            steered_mean_se = np.std(steered_mean_values, ddof=1) / np.sqrt(n) if n > 1 else 0

            steered_max_values = [max(p.get("a_minus_b_detection", 0), p.get("b_minus_a_detection", 0)) for p in cat_results]
            steered_max_mean = np.mean(steered_max_values)
            steered_max_se = np.std(steered_max_values, ddof=1) / np.sqrt(n) if n > 1 else 0

            both_work_count = sum(1 for p in cat_results
                                  if min(p.get("a_minus_b_detection", 0), p.get("b_minus_a_detection", 0)) > bidir_threshold)
            both_work_rate = both_work_count / n
            both_work_lo, both_work_hi = binomial_ci(both_work_count, n)

            summary_data.append({
                "cat": cat_name,
                "n": n,
                "baseline_max": baseline_max_mean,
                "baseline_max_se": baseline_max_se,
                "steered_mean": steered_mean_mean,
                "steered_mean_se": steered_mean_se,
                "steered_max": steered_max_mean,
                "steered_max_se": steered_max_se,
                "both_work_rate": both_work_rate,
                "both_work_err": [[max(0, both_work_rate - both_work_lo)], [max(0, both_work_hi - both_work_rate)]],
            })

    if summary_data:
        x_pos = np.arange(len(summary_data))
        width = 0.2

        for i, d in enumerate(summary_data):
            ax2.bar(i - 1.5*width, d["baseline_max"], width,
                   yerr=1.96 * d["baseline_max_se"], capsize=6,
                   color='#ADB5BD', edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 3},
                   label='Baseline max: max(A, B)' if i == 0 else None)
            ax2.bar(i - 0.5*width, d["steered_mean"], width,
                   yerr=1.96 * d["steered_mean_se"], capsize=6,
                   color=colors[d["cat"]], edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 3},
                   label='Steered mean: mean(A\u2212B, B\u2212A)' if i == 0 else None)
            ax2.bar(i + 0.5*width, d["steered_max"], width,
                   yerr=1.96 * d["steered_max_se"], capsize=6,
                   color=colors[d["cat"]], edgecolor='black', linewidth=1.5, alpha=0.7,
                   error_kw={'linewidth': 3},
                   label='Steered max: max(A\u2212B, B\u2212A)' if i == 0 else None)
            ax2.bar(i + 1.5*width, d["both_work_rate"], width,
                   yerr=d["both_work_err"], capsize=6,
                   color=colors[d["cat"]], edgecolor='black', linewidth=1.5, alpha=0.4,
                   hatch='///',
                   error_kw={'linewidth': 3},
                   label=f'Bidirectional: min(A\u2212B, B\u2212A) > {bidir_threshold:.0%}' if i == 0 else None)

        ax2.set_ylabel('Detection rate', fontsize=28, labelpad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{d['cat'].replace('-', '\u2212')}\n(n={d['n']})" for d in summary_data], fontsize=24.2)

        # Custom diagonal-split legend handler
        from matplotlib.legend_handler import HandlerBase
        import matplotlib.patches as mpatches

        class DiagonalSplitHandler(HandlerBase):
            def __init__(self, color1, color2, alpha1=1.0, alpha2=1.0, hatch=None, **kwargs):
                self.color1 = color1
                self.color2 = color2
                self.alpha1 = alpha1
                self.alpha2 = alpha2
                self.hatch = hatch
                super().__init__(**kwargs)

            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                from matplotlib.patches import Polygon, Rectangle
                rect = Rectangle((xdescent, ydescent), width, height,
                                 facecolor='none', edgecolor='black',
                                 linewidth=1.0, transform=trans)
                hatch_ec = 'black' if self.hatch else 'none'
                tri1 = Polygon(
                    [[xdescent, ydescent], [xdescent + width, ydescent + height],
                     [xdescent, ydescent + height]],
                    closed=True, facecolor=self.color1, edgecolor=hatch_ec,
                    linewidth=0.0, alpha=self.alpha1, transform=trans)
                if self.hatch:
                    tri1.set_hatch(self.hatch)
                tri2 = Polygon(
                    [[xdescent, ydescent], [xdescent + width, ydescent],
                     [xdescent + width, ydescent + height]],
                    closed=True, facecolor=self.color2, edgecolor=hatch_ec,
                    linewidth=0.0, alpha=self.alpha2, transform=trans)
                if self.hatch:
                    tri2.set_hatch(self.hatch)
                return [tri1, tri2, rect]

        color_ss = colors["S-S"]
        color_ff = colors["F-F"]

        legend_items = [
            (mpatches.Patch(facecolor='#ADB5BD', edgecolor='black', linewidth=1.0),
             'Baseline: max(A, B)'),
            (mpatches.Patch(),
             'mean(A\u2212B, B\u2212A)'),
            (mpatches.Patch(),
             'max(A\u2212B, B\u2212A)'),
            (mpatches.Patch(),
             f'Bidirectional:\nmin(A\u2212B, B\u2212A) > {bidir_threshold:.0%}'),
        ]
        handles = [item[0] for item in legend_items]
        labels = [item[1] for item in legend_items]
        handler_map = {
            handles[1]: DiagonalSplitHandler(color_ss, color_ff, alpha1=1.0, alpha2=1.0),
            handles[2]: DiagonalSplitHandler(color_ss, color_ff, alpha1=0.7, alpha2=0.7),
            handles[3]: DiagonalSplitHandler(color_ss, color_ff, alpha1=0.4, alpha2=0.4, hatch='///'),
        }
        ax2.legend(handles, labels, handler_map=handler_map,
                   loc='upper right', fontsize=19, framealpha=0.95,
                   handlelength=0.9, handletextpad=0.4)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelsize=22)
        for spine in ax2.spines.values():
            spine.set_visible(True)
        for spine in axes[0].spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.275)
    plt.savefig(results_dir / "bidirectional_swap_plot.png", dpi=400)
    plt.close()


# ==============================================================================
# Bidirectional Swap Experiment (Same-Category S-S and F-F)
# ==============================================================================

def run_part1_bidirectional_swaps_same_category(
    output_dir: Path,
    success_pairs: List[Tuple[str, str, float, float, float, float]],
    failure_pairs: List[Tuple[str, str, float, float, float, float]],
    vectors: Dict[str, torch.Tensor],
    model: Optional[Any],
    judge: Optional[Any],
    args,
    overwrite: bool = False,
    ss_cluster_info: Optional[Dict] = None,
    ff_cluster_info: Optional[Dict] = None,
    partition_threshold: float = 0.32,
    pairing_suffix: str = "",
    mean_concept_norm: float = None,
) -> Dict:
    """
    Run same-category bidirectional swap tests (S-S and F-F).

    Tests the subspace hypothesis by comparing:
    - S-S pairs: Both concepts are success concepts. Does (S1-S2) induce introspection?
    - F-F pairs: Both concepts are failure concepts. Does (F1-F2) induce introspection?

    Hypothesis: S-S differences should work (stay in success subspace),
                F-F differences should NOT work (stay in failure subspace).
    """
    part1_dir = output_dir / f"part1_bidirectional_swaps_same_category{pairing_suffix}"
    part1_dir.mkdir(parents=True, exist_ok=True)

    cleanup_temp_files(part1_dir)

    checkpoint_path = part1_dir / "checkpoint.json"
    results_path = part1_dir / "bidirectional_swap_results.json"
    matched_pairs_path = part1_dir / "matched_pairs.json"
    pairs_dir = part1_dir / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    cleanup_temp_files(pairs_dir)
    responses_dir = part1_dir / "responses"
    responses_dir.mkdir(exist_ok=True)
    cleanup_temp_files(responses_dir)

    # Load checkpoint
    if overwrite:
        checkpoint = {"completed_items": [], "status": "not_started"}
        for f in pairs_dir.glob("*.json"):
            f.unlink()
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        total_requested = len(success_pairs) + len(failure_pairs)
        n_completed = len(checkpoint.get("completed_items", []))
        if checkpoint.get("status") == "completed" and total_requested <= n_completed:
            print(f"  Part 1b already completed, loading existing results...")
            return load_json_safe(results_path)
        elif checkpoint.get("status") == "completed" and total_requested > n_completed:
            print(f"  Part 1b previously completed with {n_completed} pairs, but {total_requested} now requested. Resuming...")
            checkpoint["status"] = "in_progress"

    completed_pairs = set(checkpoint.get("completed_items", []))

    # Build matched pairs data
    ss_pairs_data = [
        {"concept_A": a, "concept_B": b, "proj_A": pa, "proj_B": pb,
         "det_A": da, "det_B": db, "category": "S-S",
         "cluster_id": ss_cluster_info["cluster_ids"][i] if ss_cluster_info else 0,
         "a_minus_b_detection": None, "b_minus_a_detection": None, "bidirectional_success": None}
        for i, (a, b, pa, pb, da, db) in enumerate(success_pairs)
    ]
    ff_pairs_data = [
        {"concept_A": a, "concept_B": b, "proj_A": pa, "proj_B": pb,
         "det_A": da, "det_B": db, "category": "F-F",
         "cluster_id": ff_cluster_info["cluster_ids"][i] if ff_cluster_info else 0,
         "a_minus_b_detection": None, "b_minus_a_detection": None, "bidirectional_success": None}
        for i, (a, b, pa, pb, da, db) in enumerate(failure_pairs)
    ]

    # Load existing steering results (handle A/B order)
    ss_key_to_idx = {}
    for i, p in enumerate(ss_pairs_data):
        ss_key_to_idx[f"{p['concept_A']}_{p['concept_B']}"] = i
        ss_key_to_idx[f"{p['concept_B']}_{p['concept_A']}"] = i
    ff_key_to_idx = {}
    for i, p in enumerate(ff_pairs_data):
        ff_key_to_idx[f"{p['concept_A']}_{p['concept_B']}"] = i
        ff_key_to_idx[f"{p['concept_B']}_{p['concept_A']}"] = i

    for f in sorted(pairs_dir.glob("pair_*.json")):
        result = load_json_safe(f)
        if result:
            category = result.get("category")
            key = f"{result.get('concept_A')}_{result.get('concept_B')}"
            if category == "S-S" and key in ss_key_to_idx:
                idx = ss_key_to_idx[key]
                if ss_pairs_data[idx]["concept_A"] == result.get("concept_A"):
                    ss_pairs_data[idx]["a_minus_b_detection"] = result.get("a_minus_b_detection")
                    ss_pairs_data[idx]["b_minus_a_detection"] = result.get("b_minus_a_detection")
                else:
                    ss_pairs_data[idx]["a_minus_b_detection"] = result.get("b_minus_a_detection")
                    ss_pairs_data[idx]["b_minus_a_detection"] = result.get("a_minus_b_detection")
                ss_pairs_data[idx]["bidirectional_success"] = result.get("bidirectional_success")
            elif category == "F-F" and key in ff_key_to_idx:
                idx = ff_key_to_idx[key]
                if ff_pairs_data[idx]["concept_A"] == result.get("concept_A"):
                    ff_pairs_data[idx]["a_minus_b_detection"] = result.get("a_minus_b_detection")
                    ff_pairs_data[idx]["b_minus_a_detection"] = result.get("b_minus_a_detection")
                else:
                    ff_pairs_data[idx]["a_minus_b_detection"] = result.get("b_minus_a_detection")
                    ff_pairs_data[idx]["b_minus_a_detection"] = result.get("a_minus_b_detection")
                ff_pairs_data[idx]["bidirectional_success"] = result.get("bidirectional_success")

    save_json_atomic({"S-S": ss_pairs_data, "F-F": ff_pairs_data}, matched_pairs_path)
    save_json_atomic({"S-S": ss_cluster_info, "F-F": ff_cluster_info}, part1_dir / "cluster_info.json")

    # Load existing results for current matched pairs only
    ss_keys = set(ss_key_to_idx.keys())
    ff_keys = set(ff_key_to_idx.keys())
    ss_results = []
    ff_results = []
    for f in sorted(pairs_dir.glob("pair_*.json")):
        result = load_json_safe(f)
        if result:
            key = f"{result.get('concept_A')}_{result.get('concept_B')}"
            if result.get("category") == "S-S" and key in ss_keys:
                ss_results.append(result)
            elif result.get("category") == "F-F" and key in ff_keys:
                ff_results.append(result)

    if model is None or judge is None:
        print("  Skipping steering experiments (no model or judge)")
        results = {
            "S-S_pairs": ss_results,
            "F-F_pairs": ff_results,
            "statistics": {"note": "Steering experiments skipped"}
        }
        save_json_atomic(results, results_path)
        return results

    # Build list of pairs to process
    def is_pair_completed(prefix: str, a: str, b: str) -> bool:
        return f"{prefix}_{a}_{b}" in completed_pairs or f"{prefix}_{b}_{a}" in completed_pairs

    all_pairs_to_process = []
    max_len = max(len(success_pairs), len(failure_pairs))
    for i in range(max_len):
        if i < len(success_pairs):
            a, b, pa, pb, da, db = success_pairs[i]
            pair_key = f"SS_{a}_{b}"
            if not is_pair_completed("SS", a, b):
                all_pairs_to_process.append((i, a, b, pa, pb, da, db, "S-S", pair_key))
        if i < len(failure_pairs):
            a, b, pa, pb, da, db = failure_pairs[i]
            pair_key = f"FF_{a}_{b}"
            if not is_pair_completed("FF", a, b):
                all_pairs_to_process.append((i, a, b, pa, pb, da, db, "F-F", pair_key))

    print(f"  Processing {len(all_pairs_to_process)} same-category pairs...")
    print(f"    S-S pairs: {len(success_pairs)}, F-F pairs: {len(failure_pairs)}")
    print(f"    Already completed: {len(completed_pairs)}")

    detection_threshold = partition_threshold
    judge_batch_size = getattr(args, 'judge_batch_pairs', 5)

    # Pre-compute trial numbers
    trial_numbers = []
    for t in range(1, args.n_trial_numbers + 1):
        trial_numbers.extend([t] * args.samples_per_trial)
    n_trials = len(trial_numbers)

    def _process_and_judge_batch(batch_items):
        """Run GPU inference for multiple pairs, then batch-evaluate with judge."""
        if not batch_items:
            return

        n_pairs = len(batch_items)

        # Build per-prompt steering vectors
        ab_steering_vecs = []
        ba_steering_vecs = []
        for pair_result, delta_ab, delta_ba, idx, category, pair_key, concept_a, concept_b in batch_items:
            ab_steering_vecs.extend([delta_ab] * n_trials)
            ba_steering_vecs.extend([delta_ba] * n_trials)

        batch_trial_numbers = trial_numbers * n_pairs

        # GPU calls for A-B and B-A directions
        all_ab_responses = run_steered_introspection_test_batch(
            model=model,
            concept_word="batch",
            layer_idx=args.layer,
            strength=args.strength,
            trial_numbers=batch_trial_numbers,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            steering_vectors=[v.to(model.device) for v in ab_steering_vecs],
        )

        all_ba_responses = run_steered_introspection_test_batch(
            model=model,
            concept_word="batch",
            layer_idx=args.layer,
            strength=args.strength,
            trial_numbers=batch_trial_numbers,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            steering_vectors=[v.to(model.device) for v in ba_steering_vecs],
        )

        # Split responses per-pair
        per_pair_ab = [all_ab_responses[i * n_trials:(i + 1) * n_trials] for i in range(n_pairs)]
        per_pair_ba = [all_ba_responses[i * n_trials:(i + 1) * n_trials] for i in range(n_pairs)]

        # Judge all responses in one batch
        all_responses_flat = []
        pair_boundaries = []
        for i in range(n_pairs):
            start = len(all_responses_flat)
            all_responses_flat.extend(per_pair_ab[i])
            all_responses_flat.extend(per_pair_ba[i])
            pair_boundaries.append((start, len(per_pair_ab[i]), len(per_pair_ba[i])))

        results_dicts = [{"response": r, "trial_type": "injection", "concept": "batch"} for r in all_responses_flat]
        original_prompts_list = ["introspection test"] * len(all_responses_flat)
        evals = batch_evaluate(judge, results_dicts, original_prompts_list)

        # Distribute results
        for item_idx, (pair_result, delta_ab, delta_ba, idx, category, pair_key, concept_a, concept_b) in enumerate(batch_items):
            start, n_a, n_b = pair_boundaries[item_idx]
            pair_evals = evals[start:start + n_a + n_b]

            toward_a_detections = [
                e.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
                for e in pair_evals[:n_a]
            ]
            toward_b_detections = [
                e.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
                for e in pair_evals[n_a:]
            ]

            toward_a_rate = np.mean(toward_a_detections) if toward_a_detections else 0.0
            toward_b_rate = np.mean(toward_b_detections) if toward_b_detections else 0.0

            toward_a_works = toward_a_rate > detection_threshold
            toward_b_works = toward_b_rate > detection_threshold

            pair_result["a_minus_b_detection"] = float(toward_a_rate)
            pair_result["b_minus_a_detection"] = float(toward_b_rate)
            pair_result["a_minus_b_works"] = bool(toward_a_works)
            pair_result["b_minus_a_works"] = bool(toward_b_works)
            pair_result["either_direction_works"] = bool(toward_a_works or toward_b_works)
            pair_result["bidirectional_success"] = bool(toward_a_works and toward_b_works)

            # Save responses and pair result
            response_data = {
                "pair_index": idx,
                "concept_A": concept_a,
                "concept_B": concept_b,
                "category": category,
                "a_minus_b_responses": per_pair_ab[item_idx],
                "a_minus_b_detections": toward_a_detections,
                "b_minus_a_responses": per_pair_ba[item_idx],
                "b_minus_a_detections": toward_b_detections,
            }
            save_json_atomic(response_data, responses_dir / f"pair_{category}_{idx:04d}_{pair_key}_responses.json")
            save_json_atomic(pair_result, pairs_dir / f"pair_{category}_{idx:04d}_{pair_key}.json")

            if category == "S-S":
                ss_results.append(pair_result)
                if idx < len(ss_pairs_data):
                    ss_pairs_data[idx]["a_minus_b_detection"] = pair_result.get("a_minus_b_detection")
                    ss_pairs_data[idx]["b_minus_a_detection"] = pair_result.get("b_minus_a_detection")
                    ss_pairs_data[idx]["bidirectional_success"] = pair_result.get("bidirectional_success")
            else:
                ff_results.append(pair_result)
                if idx < len(ff_pairs_data):
                    ff_pairs_data[idx]["a_minus_b_detection"] = pair_result.get("a_minus_b_detection")
                    ff_pairs_data[idx]["b_minus_a_detection"] = pair_result.get("b_minus_a_detection")
                    ff_pairs_data[idx]["bidirectional_success"] = pair_result.get("bidirectional_success")

            completed_pairs.add(pair_key)

        # Save state once per batch
        save_json_atomic({"S-S": ss_pairs_data, "F-F": ff_pairs_data}, matched_pairs_path)
        save_checkpoint(checkpoint_path, list(completed_pairs), status="in_progress")

        if len(completed_pairs) % args.plot_interval < judge_batch_size:
            plot_bidirectional_progress_same_category(part1_dir, ss_results, ff_results, partition_threshold, seed=args.seed)

    # Main loop: accumulate pairs, then process in batches
    pending_batch = []

    for idx, concept_a, concept_b, proj_a, proj_b, det_a, det_b, category, pair_key in tqdm(
        all_pairs_to_process, desc="Same-category swaps"
    ):
        if concept_a not in vectors or concept_b not in vectors:
            continue

        delta_a_to_b = vectors[concept_a] - vectors[concept_b]
        delta_b_to_a = vectors[concept_b] - vectors[concept_a]

        # Norm-normalize if requested
        if mean_concept_norm is not None and args.norm_normalize:
            diff_norm = delta_a_to_b.norm().item()
            if diff_norm > 0:
                scale = mean_concept_norm / diff_norm
                delta_a_to_b = delta_a_to_b * scale
                delta_b_to_a = delta_b_to_a * scale

        pair_result = {
            "pair_index": idx,
            "concept_A": concept_a,
            "concept_B": concept_b,
            "category": category,
            "proj_A": proj_a,
            "proj_B": proj_b,
            "proj_diff": abs(proj_a - proj_b),
            "baseline_det_A": det_a,
            "baseline_det_B": det_b,
            "timestamp": datetime.now().isoformat(),
        }

        pending_batch.append((pair_result, delta_a_to_b, delta_b_to_a, idx, category, pair_key, concept_a, concept_b))

        if len(pending_batch) >= judge_batch_size:
            _process_and_judge_batch(pending_batch)
            pending_batch = []

    # Flush remaining pairs
    _process_and_judge_batch(pending_batch)

    # Compute statistics
    def compute_category_stats(results, name):
        if not results:
            return {"n": 0}
        n = len(results)
        return {
            "n": n,
            "mean_a_minus_b": np.mean([p.get("a_minus_b_detection", 0) for p in results]),
            "mean_b_minus_a": np.mean([p.get("b_minus_a_detection", 0) for p in results]),
            "mean_overall": np.mean([p.get("a_minus_b_detection", 0) for p in results] +
                                   [p.get("b_minus_a_detection", 0) for p in results]),
            "either_works_count": sum(1 for p in results if p.get("either_direction_works", False)),
            "either_works_rate": np.mean([1 if p.get("either_direction_works", False) else 0 for p in results]),
            "bidirectional_count": sum(1 for p in results if p.get("bidirectional_success", False)),
            "bidirectional_rate": np.mean([1 if p.get("bidirectional_success", False) else 0 for p in results]),
        }

    ss_stats = compute_category_stats(ss_results, "S-S")
    ff_stats = compute_category_stats(ff_results, "F-F")

    # Subspace effect: P(bidirectional | S-S) - P(bidirectional | F-F)
    p_ss = ss_stats.get("bidirectional_rate", 0)
    p_ff = ff_stats.get("bidirectional_rate", 0)
    n_ss = ss_stats.get("n", 0)
    n_ff = ff_stats.get("n", 0)

    subspace_effect_strength = p_ss - p_ff

    if n_ss > 0 and n_ff > 0:
        from scipy import stats as scipy_stats
        se_diff = np.sqrt(p_ss * (1 - p_ss) / n_ss + p_ff * (1 - p_ff) / n_ff)
        ci_low = subspace_effect_strength - 1.96 * se_diff
        ci_high = subspace_effect_strength + 1.96 * se_diff
        z_stat = subspace_effect_strength / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
    else:
        se_diff = 0
        ci_low = ci_high = subspace_effect_strength
        z_stat = 0
        p_value = 1.0

    print(f"\n  === Subspace Effect Analysis ===")
    print(f"  P(bidirectional | S-S) = {p_ss:.1%} (n={n_ss})")
    print(f"  P(bidirectional | F-F) = {p_ff:.1%} (n={n_ff})")
    print(f"  Subspace effect strength = {subspace_effect_strength:.1%} [{ci_low:.1%}, {ci_high:.1%}]")
    print(f"  Z-statistic = {z_stat:.2f}, p-value = {p_value:.4f}")

    results = {
        "S-S_pairs": ss_results,
        "F-F_pairs": ff_results,
        "statistics": {
            "S-S": ss_stats,
            "F-F": ff_stats,
            "subspace_effect": {
                "description": "Subspace effect strength = P(bidirectional | S-S) - P(bidirectional | F-F)",
                "p_ss_bidirectional": p_ss,
                "p_ff_bidirectional": p_ff,
                "n_ss": n_ss,
                "n_ff": n_ff,
                "effect_strength": subspace_effect_strength,
                "se": se_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "z_statistic": float(z_stat),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05 and subspace_effect_strength > 0),
            },
            "hypothesis_test": {
                "description": "Subspace hypothesis: S-S should work better than F-F",
                "ss_either_works_rate": float(ss_stats.get("either_works_rate", 0)),
                "ff_either_works_rate": float(ff_stats.get("either_works_rate", 0)),
                "ss_better_than_ff": bool(ss_stats.get("either_works_rate", 0) > ff_stats.get("either_works_rate", 0)),
            },
            "partition_threshold": partition_threshold,
        },
        "cluster_info": {"S-S": ss_cluster_info, "F-F": ff_cluster_info},
    }

    save_json_atomic(results, results_path)
    save_checkpoint(checkpoint_path, list(completed_pairs), status="completed")
    plot_bidirectional_progress_same_category(part1_dir, ss_results, ff_results, partition_threshold, seed=args.seed)

    return results


# ==============================================================================
# Delta-PC Extraction (Orthogonalized Difference Vectors)
# ==============================================================================

def compute_orthogonalized_difference_vectors(
    pairs: List[Dict],
    vectors: Dict[str, torch.Tensor],
    mean_diff: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[Dict], torch.Tensor]:
    """
    Compute orthogonalized difference vectors for S-S pairs.

    For each pair (A, B), computes:
        v_diff = v_A - v_B
        v_orth = v_diff - proj_{mean_diff}(v_diff)

    Returns:
        orth_vectors: List of orthogonalized vectors (unnormalized)
        pair_info: List of dicts with pair metadata
        orth_matrix: Stacked matrix for PCA
    """
    mean_diff_norm = mean_diff / mean_diff.norm()

    orth_vectors = []
    pair_info = []
    orth_list = []

    for p in pairs:
        concept_a = p.get("concept_A")
        concept_b = p.get("concept_B")

        if concept_a not in vectors or concept_b not in vectors:
            continue

        v_a = vectors[concept_a]
        v_b = vectors[concept_b]
        v_diff = v_a - v_b

        proj = torch.dot(v_diff.flatten(), mean_diff_norm.flatten())
        v_orth = v_diff - proj * mean_diff_norm.view_as(v_diff)

        orth_norm = v_orth.norm()
        orth_list.append(v_orth.flatten())
        orth_vectors.append(v_orth)

        pair_info.append({
            "concept_A": concept_a,
            "concept_B": concept_b,
            "original_norm": v_diff.norm().item(),
            "orth_norm": orth_norm.item(),
            "mean_diff_projection": proj.item(),
            "projection_fraction": abs(proj.item()) / v_diff.norm().item() if v_diff.norm().item() > 0 else 0,
        })

    if orth_list:
        orth_matrix = torch.stack(orth_list)
    else:
        orth_matrix = torch.empty(0)

    return orth_vectors, pair_info, orth_matrix


def compute_pca_on_orth_vectors(
    orth_matrix: torch.Tensor,
    n_components: int = 5,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Compute PCA on orthogonalized difference vectors.

    Returns:
        pcs: (n_components, dim) tensor of principal components (unit norm)
        explained_variance_ratio: Variance explained by each PC
        singular_values: Singular values from SVD
    """
    if orth_matrix.shape[0] < 2:
        return torch.empty(0), np.array([]), np.array([])

    orth_np = orth_matrix.cpu().float().numpy()
    orth_centered = orth_np - orth_np.mean(axis=0)

    n_components = min(n_components, orth_matrix.shape[0] - 1, orth_matrix.shape[1])
    U, S, Vt = np.linalg.svd(orth_centered, full_matrices=False)

    total_var = np.sum(S ** 2)
    explained_variance_ratio = (S[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)

    pcs = torch.tensor(Vt[:n_components], dtype=orth_matrix.dtype)
    pcs = pcs / pcs.norm(dim=1, keepdim=True)

    return pcs, explained_variance_ratio, S[:n_components]


# ==============================================================================
# Subspace Threshold Sweep Plots
# ==============================================================================

def plot_subspace_threshold_individual(
    results_dir: Path,
    sweep_results: List[Dict],
    alpha_values: List[float],
    quiet: bool = False,
):
    """Plot detection rate vs alpha for individual S-S pairs (orthogonalized)."""
    if not sweep_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    all_curves = []

    for pair_result in sweep_results:
        alpha_results = pair_result.get("alpha_results", [])
        if not alpha_results:
            continue

        alpha_results = sorted(alpha_results, key=lambda x: x.get("alpha", 0))
        alphas = [r.get("alpha", 0) for r in alpha_results]
        det_rates = [r.get("detection_rate", 0) for r in alpha_results]

        ax.plot(alphas, det_rates, color='#2A9D8F', alpha=0.15, linewidth=1)
        all_curves.append((alphas, det_rates))

    if all_curves:
        common_alphas = sorted(alpha_values)
        interpolated = []

        for alphas, det_rates in all_curves:
            if len(alphas) == len(common_alphas) and alphas == common_alphas:
                interpolated.append(det_rates)
            else:
                interp_rates = np.interp(common_alphas, alphas, det_rates)
                interpolated.append(interp_rates)

        interpolated = np.array(interpolated)
        mean_curve = np.mean(interpolated, axis=0)
        se_curve = np.std(interpolated, axis=0) / np.sqrt(len(interpolated))

        ax.plot(common_alphas, mean_curve, color='#E76F51', linewidth=3, label='Average', zorder=10)
        ax.fill_between(common_alphas,
                       mean_curve - 1.96 * se_curve,
                       mean_curve + 1.96 * se_curve,
                       color='#E76F51', alpha=0.3, zorder=9)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='alpha=0 (no steering)')

    ax.set_xlabel('Steering strength (alpha)', fontsize=12)
    ax.set_ylabel('Detection rate', fontsize=12)
    ax.set_title('Subspace Threshold Sweep: Individual S-S Pairs (Orthogonalized)', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    n_pairs = len(sweep_results)
    info_text = f"n = {n_pairs} bidirectional S-S pairs\nVectors orthogonalized to mean-diff"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(results_dir / "synthetic_threshold_subspace.png", dpi=150, bbox_inches='tight')
    plt.close()
    if not quiet:
        print(f"    Saved: {results_dir / 'synthetic_threshold_subspace.png'}")


def plot_subspace_threshold_pcs(
    results_dir: Path,
    pc_sweep_results: List[Dict],
    alpha_values: List[float],
    explained_variance: np.ndarray,
    quiet: bool = False,
):
    """Plot detection rate vs alpha for each principal component direction."""
    if not pc_sweep_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Detection curves per PC
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(pc_sweep_results)))

    for i, pc_result in enumerate(pc_sweep_results):
        alpha_results = pc_result.get("alpha_results", [])
        if not alpha_results:
            continue

        alpha_results = sorted(alpha_results, key=lambda x: x.get("alpha", 0))
        alphas = [r.get("alpha", 0) for r in alpha_results]
        det_rates = [r.get("detection_rate", 0) for r in alpha_results]

        var_expl = explained_variance[i] * 100 if i < len(explained_variance) else 0
        label = f'PC{i+1} ({var_expl:.1f}% var)'

        ax1.plot(alphas, det_rates, color=colors[i], linewidth=2, marker='o',
                markersize=4, label=label)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering strength (alpha)', fontsize=12)
    ax1.set_ylabel('Detection rate', fontsize=12)
    ax1.set_title('Subspace Threshold Sweep: Principal Components', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scree plot
    ax2 = axes[1]
    n_pcs = len(explained_variance)
    x = np.arange(1, n_pcs + 1)

    ax2.bar(x, explained_variance * 100, color='#2A9D8F', edgecolor='white')
    ax2.plot(x, np.cumsum(explained_variance) * 100, 'o-', color='#E76F51',
            linewidth=2, markersize=6, label='Cumulative')

    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Variance Explained (%)', fontsize=12)
    ax2.set_title('PCA of Orthogonalized S-S Difference Vectors', fontsize=11)
    ax2.set_xticks(x)
    ax2.legend(loc='right')
    ax2.grid(True, alpha=0.3, axis='y')

    cumsum = np.cumsum(explained_variance) * 100
    for i, (xi, yi) in enumerate(zip(x, cumsum)):
        if i < 3:
            ax2.annotate(f'{yi:.0f}%', (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / "synthetic_threshold_subspace_pcs.png", dpi=150, bbox_inches='tight')
    plt.close()
    if not quiet:
        print(f"    Saved: {results_dir / 'synthetic_threshold_subspace_pcs.png'}")


# ==============================================================================
# Subspace Threshold Sweep Experiment
# ==============================================================================

def run_subspace_threshold_sweep(
    output_dir: Path,
    bidirectional_ss_pairs: List[Dict],
    vectors: Dict[str, torch.Tensor],
    mean_diff: torch.Tensor,
    model: "ModelWrapper",
    judge: "LLMJudge",
    args,
    alpha_values: List[float] = None,
    n_trials: int = 5,
    samples_per_trial: int = 5,
    max_pairs: int = 30,
    n_pcs: int = 5,
    pairing_suffix: str = "",
) -> Dict:
    """
    Run threshold sweep on orthogonalized S-S difference vectors.

    Two analyses:
    1. Individual pair sweep: detection rate vs alpha for each pair
    2. PC sweep: detection rate vs alpha along each principal component

    Args:
        bidirectional_ss_pairs: S-S pairs where both directions trigger detection
        vectors: Concept vectors
        mean_diff: mu_success - mu_failure for orthogonalization
        alpha_values: Steering strengths to test
        n_trials: Trial numbers per alpha
        samples_per_trial: Samples per trial number
        max_pairs: Maximum pairs for individual sweep
        n_pcs: Number of principal components for PC sweep
    """
    if alpha_values is None:
        alpha_values = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    sweep_dir = output_dir / f"subspace_threshold_sweep{pairing_suffix}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = sweep_dir / "checkpoint.json"
    results_path = sweep_dir / "sweep_results.json"

    # Check if already completed
    if results_path.exists() and not args.overwrite:
        existing = load_json_safe(results_path)
        if existing and existing.get("status") == "completed":
            print(f"    Subspace threshold sweep already completed, loading results...")
            if existing.get("individual_sweep"):
                plot_subspace_threshold_individual(sweep_dir, existing["individual_sweep"], alpha_values)
            if existing.get("pc_sweep"):
                plot_subspace_threshold_pcs(sweep_dir, existing["pc_sweep"], alpha_values,
                                           np.array(existing.get("explained_variance", [])))
            return existing

    # Compute orthogonalized vectors
    print(f"\n  === Subspace Threshold Sweep ===")
    print(f"    Computing orthogonalized difference vectors...")

    orth_vectors, pair_info, orth_matrix = compute_orthogonalized_difference_vectors(
        bidirectional_ss_pairs, vectors, mean_diff
    )

    if len(orth_vectors) == 0:
        print(f"    No valid bidirectional S-S pairs found, skipping sweep.")
        return {"status": "skipped", "reason": "no_bidirectional_pairs"}

    print(f"    Total bidirectional S-S pairs: {len(orth_vectors)}")

    # Compute PCA
    print(f"    Computing PCA on orthogonalized vectors...")
    pcs, explained_variance, singular_values = compute_pca_on_orth_vectors(orth_matrix, n_components=n_pcs)

    if len(pcs) > 0:
        print(f"    Top {len(pcs)} PCs explain {np.sum(explained_variance)*100:.1f}% of variance")
        for i, ev in enumerate(explained_variance):
            print(f"      PC{i+1}: {ev*100:.1f}%")

    # Save PCA info
    pca_info = {
        "n_pairs": len(orth_vectors),
        "explained_variance": explained_variance.tolist() if len(explained_variance) > 0 else [],
        "cumulative_variance": np.cumsum(explained_variance).tolist() if len(explained_variance) > 0 else [],
        "singular_values": singular_values.tolist() if len(singular_values) > 0 else [],
    }
    save_json_atomic(pca_info, sweep_dir / "pca_info.json")

    avg_orth_norm = np.mean([info["orth_norm"] for info in pair_info])
    print(f"    Average orthogonalized vector norm: {avg_orth_norm:.1f}")

    if len(pcs) > 0:
        torch.save(pcs, sweep_dir / "pcs.pt")
        torch.save(torch.tensor([avg_orth_norm]), sweep_dir / "avg_orth_norm.pt")

    # Select subset for individual sweep
    if len(orth_vectors) > max_pairs:
        print(f"    Selecting {max_pairs} pairs for individual sweep (from {len(orth_vectors)} total)")
        np.random.seed(args.seed)
        selected_indices = np.random.choice(len(orth_vectors), size=max_pairs, replace=False)
        selected_vectors = [orth_vectors[i] for i in selected_indices]
        selected_info = [pair_info[i] for i in selected_indices]
    else:
        selected_vectors = orth_vectors
        selected_info = pair_info

    print(f"    Alpha values: {alpha_values}")
    print(f"    Samples per alpha: {n_trials} trials x {samples_per_trial} samples = {n_trials * samples_per_trial}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed = set(checkpoint.get("completed_items", []))

    trial_numbers = []
    for t in range(1, n_trials + 1):
        trial_numbers.extend([t] * samples_per_trial)

    # =========================================================================
    # Part A: Individual pair sweep
    # =========================================================================
    print(f"\n    --- Part A: Individual Pair Sweep ({len(selected_vectors)} pairs) ---")

    individual_results = []
    (sweep_dir / "individual_experiments").mkdir(parents=True, exist_ok=True)

    for pair_idx, (v_orth, info) in enumerate(tqdm(
        zip(selected_vectors, selected_info),
        total=len(selected_vectors),
        desc="    Individual pairs",
        unit="pair"
    )):
        pair_key = f"{info['concept_A']}_{info['concept_B']}"
        pair_results = {"pair": info, "alpha_results": []}

        for alpha in alpha_values:
            exp_key = f"individual_{pair_key}_alpha_{alpha}"
            exp_file = sweep_dir / "individual_experiments" / f"{exp_key}.json"

            if exp_key in completed:
                if exp_file.exists():
                    exp_data = load_json_safe(exp_file)
                    if exp_data:
                        pair_results["alpha_results"].append(exp_data)
                continue

            steering_vector = (alpha * v_orth).to(model.device)

            if abs(alpha) < 1e-8:
                responses = run_unsteered_introspection_test_batch(
                    model=model,
                    concept_word=info["concept_A"],
                    trial_numbers=trial_numbers,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            else:
                responses = run_steered_introspection_test_batch(
                    model=model,
                    concept_word=info["concept_A"],
                    steering_vector=steering_vector,
                    layer_idx=args.layer,
                    strength=args.strength,
                    trial_numbers=trial_numbers,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

            results_dicts = [{"response": r, "trial_type": "injection", "concept": info["concept_A"]}
                           for r in responses]
            original_prompts = ["introspection test"] * len(responses)
            evals = batch_evaluate(judge, results_dicts, original_prompts)

            detections = [
                e.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
                for e in evals
            ]
            detection_rate = np.mean(detections) if detections else 0.0

            exp_result = {
                "alpha": alpha,
                "effective_strength": float(args.strength),
                "vector_scale": float(alpha),
                "detection_rate": float(detection_rate),
                "n_samples": len(detections),
                "n_detections": int(sum(detections)),
                "responses": responses,
            }

            save_json_atomic(exp_result, exp_file)
            pair_results["alpha_results"].append(exp_result)

            completed.add(exp_key)
            save_checkpoint(checkpoint_path, list(completed))

        individual_results.append(pair_results)

        if pair_results["alpha_results"]:
            plot_subspace_threshold_individual(sweep_dir, individual_results, alpha_values, quiet=True)

    # =========================================================================
    # Part B: PC sweep
    # =========================================================================
    print(f"\n    --- Part B: Principal Component Sweep ({len(pcs)} PCs) ---")

    pc_results = []
    (sweep_dir / "pc_experiments").mkdir(parents=True, exist_ok=True)

    for pc_idx, pc in enumerate(tqdm(pcs, desc="    Principal components", unit="PC")):
        pc_key = f"PC{pc_idx + 1}"
        var_pct = explained_variance[pc_idx] * 100 if pc_idx < len(explained_variance) else 0

        pc_result = {
            "pc_index": pc_idx,
            "pc_name": pc_key,
            "variance_explained": float(explained_variance[pc_idx]) if pc_idx < len(explained_variance) else 0.0,
            "alpha_results": []
        }

        pc_vec = pc.to(model.device)
        pc_vec = (pc_vec / pc_vec.norm()) * avg_orth_norm

        for alpha in alpha_values:
            exp_key = f"pc_{pc_idx}_alpha_{alpha}"
            exp_file = sweep_dir / "pc_experiments" / f"{exp_key}.json"

            if exp_key in completed:
                if exp_file.exists():
                    exp_data = load_json_safe(exp_file)
                    if exp_data:
                        pc_result["alpha_results"].append(exp_data)
                continue

            steering_vector_pc = (alpha * pc_vec)

            if abs(alpha) < 1e-8:
                responses = run_unsteered_introspection_test_batch(
                    model=model,
                    concept_word="subspace",
                    trial_numbers=trial_numbers,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            else:
                responses = run_steered_introspection_test_batch(
                    model=model,
                    concept_word="subspace",
                    steering_vector=steering_vector_pc,
                    layer_idx=args.layer,
                    strength=args.strength,
                    trial_numbers=trial_numbers,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

            results_dicts = [{"response": r, "trial_type": "injection", "concept": "subspace"}
                           for r in responses]
            original_prompts = ["introspection test"] * len(responses)
            evals = batch_evaluate(judge, results_dicts, original_prompts)

            detections = [
                e.get("evaluations", {}).get("claims_detection", {}).get("grade", 0)
                for e in evals
            ]
            detection_rate = np.mean(detections) if detections else 0.0

            exp_result = {
                "alpha": alpha,
                "effective_strength": float(args.strength),
                "vector_scale": float(alpha),
                "detection_rate": float(detection_rate),
                "n_samples": len(detections),
                "n_detections": int(sum(detections)),
                "responses": responses,
            }

            save_json_atomic(exp_result, exp_file)
            pc_result["alpha_results"].append(exp_result)

            completed.add(exp_key)
            save_checkpoint(checkpoint_path, list(completed))

        pc_results.append(pc_result)

        if pc_result["alpha_results"]:
            plot_subspace_threshold_pcs(sweep_dir, pc_results, alpha_values, explained_variance, quiet=True)

    # =========================================================================
    # Generate final plots and save results
    # =========================================================================
    print(f"\n    Generating final plots...")

    plot_subspace_threshold_individual(sweep_dir, individual_results, alpha_values)
    plot_subspace_threshold_pcs(sweep_dir, pc_results, alpha_values, explained_variance)

    final_results = {
        "status": "completed",
        "alpha_values": alpha_values,
        "n_trials": n_trials,
        "samples_per_trial": samples_per_trial,
        "individual_sweep": individual_results,
        "pc_sweep": pc_results,
        "pca_info": pca_info,
        "explained_variance": explained_variance.tolist() if len(explained_variance) > 0 else [],
        "n_total_bidirectional_pairs": len(orth_vectors),
        "n_pairs_in_sweep": len(selected_vectors),
        "pair_info": selected_info,
    }

    save_json_atomic(final_results, results_path)
    save_checkpoint(checkpoint_path, list(completed), status="completed")

    print(f"\n    === Subspace Threshold Sweep Complete ===")
    print(f"    Individual pairs tested: {len(individual_results)}")
    print(f"    PCs tested: {len(pc_results)}")
    print(f"    Results saved to: {sweep_dir}")

    return final_results


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup base paths
    geometry_dir = Path(args.geometry_dir)
    steering_dir = Path(args.steering_dir)

    # Create all layer/strength combinations
    layer_strength_configs = [(layer, strength) for layer in args.layers for strength in args.strengths]
    print(f"=" * 80)
    print(f"Bidirectional Steering Analysis")
    print(f"Model: {args.model}")
    print(f"Layer/Strength configurations: {layer_strength_configs}")
    print(f"=" * 80)

    for config_idx, (layer, strength) in enumerate(layer_strength_configs):
        args.layer = layer
        args.strength = strength

        print(f"\n{'#' * 80}")
        print(f"# Configuration {config_idx + 1}/{len(layer_strength_configs)}: layer={layer}, strength={strength}")
        print(f"{'#' * 80}")

        # Output directory
        if args.balanced_partition:
            output_dir = Path(args.output_dir) / f"{args.model}_balanced" / f"layer_{args.layer}_strength_{args.strength}"
        else:
            output_dir = Path(args.output_dir) / args.model / f"layer_{args.layer}_strength_{args.strength}"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = vars(args)
        config["timestamp"] = datetime.now().isoformat()
        save_json_atomic(config, output_dir / "config.json")

        # =====================================================================
        # Load data
        # =====================================================================
        print("\n[1] Loading data...")

        # Load partition
        success_concepts, failure_concepts, partition_metadata = load_geometry_partition(
            geometry_dir, args.model, args.layer, args.strength, balanced=args.balanced_partition
        )
        if args.balanced_partition:
            print(f"  Loaded BALANCED partition: {len(success_concepts)} success, {len(failure_concepts)} failure")
        else:
            print(f"  Loaded partition: {len(success_concepts)} success, {len(failure_concepts)} failure")

        # Load concept vectors
        vectors_dir = steering_dir / args.model / "vectors" / f"layer_{args.layer}"
        if not vectors_dir.exists():
            alt_paths = [
                steering_dir / args.model / "vectors",
                steering_dir / args.model / f"layer_{args.layer}_strength_{args.strength}" / "vectors",
            ]
            for alt in alt_paths:
                if alt.exists():
                    vectors_dir = alt
                    break

        all_concepts = list(set(success_concepts + failure_concepts))
        vectors = load_concept_vectors(vectors_dir, all_concepts)
        print(f"  Loaded {len(vectors)} concept vectors from {vectors_dir}")

        if len(vectors) == 0:
            print("ERROR: No concept vectors found! Skipping this configuration...")
            continue

        # Compute group statistics
        success_vecs = [vectors[c] for c in success_concepts if c in vectors]
        failure_vecs = [vectors[c] for c in failure_concepts if c in vectors]

        mu_success = torch.stack(success_vecs).mean(dim=0) if success_vecs else None
        mu_failure = torch.stack(failure_vecs).mean(dim=0) if failure_vecs else None
        mean_diff = mu_success - mu_failure if mu_success is not None and mu_failure is not None else None

        # Mean concept norm for optional normalization
        all_vecs = success_vecs + failure_vecs
        mean_concept_norm = torch.stack([v.norm() for v in all_vecs]).mean().item() if all_vecs else 1.0

        print(f"  mean_diff norm: {mean_diff.norm().item():.2f}" if mean_diff is not None else "  mean_diff: N/A")

        # Compute pairing direction
        if args.pairing_direction == "ridge":
            geometry_dir = Path(args.geometry_dir)
            try:
                pairing_direction = load_ridge_direction(geometry_dir, args.model, layer, strength)
                print(f"  Using RIDGE direction for pairing")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                print(f"  Falling back to mean-diff for pairing")
                pairing_direction = mean_diff
        elif args.pairing_direction == "mean-diff":
            pairing_direction = mean_diff
            print(f"  Using MEAN-DIFF direction for pairing")
        else:
            pairing_direction = mean_diff
            print(f"  Using RANDOM pairing (no projection matching)")

        # Pairing suffix for folder names
        pairing_suffix = f"_{args.pairing_direction.replace('-', '_')}"
        if args.norm_normalize:
            pairing_suffix += "_normed"
        else:
            pairing_suffix += "_nonorm"
        if args.min_success_baseline is not None:
            pairing_suffix += f"_smin{args.min_success_baseline:.2f}"
        if args.max_failure_baseline is not None:
            pairing_suffix += f"_fmax{args.max_failure_baseline:.2f}"

        # =====================================================================
        # Same-Category Bidirectional Swaps (S-S and F-F)
        # =====================================================================
        if not args.same_category_swaps:
            print("\n  No experiment mode selected. Use --same-category-swaps to run.")
            continue

        print(f"\n{'='*80}")
        print("Same-Category Bidirectional Swaps (S-S and F-F)")
        print(f"{'='*80}")

        # Handle --plots-only mode
        if args.plots_only:
            print("  [plots-only mode] Regenerating plots from existing results...")
            part1b_dir = output_dir / f"part1_bidirectional_swaps_same_category{pairing_suffix}"
            results_path = part1b_dir / "bidirectional_swap_results.json"
            pairs_dir = part1b_dir / "pairs"

            ss_results = []
            ff_results = []

            pairs_ss = []
            pairs_ff = []
            if pairs_dir.exists():
                pair_files = sorted(pairs_dir.glob("pair_*.json"))
                for pf in pair_files:
                    pair_data = load_json_safe(pf)
                    if pair_data:
                        category = pair_data.get("category", "")
                        if category == "S-S":
                            pairs_ss.append(pair_data)
                        elif category == "F-F":
                            pairs_ff.append(pair_data)

            if results_path.exists():
                part1b_data = load_json_safe(results_path)
                if part1b_data:
                    final_ss = part1b_data.get("S-S_pairs", [])
                    final_ff = part1b_data.get("F-F_pairs", [])
                else:
                    final_ss, final_ff = [], []
            else:
                final_ss, final_ff = [], []

            if len(pairs_ss) + len(pairs_ff) >= len(final_ss) + len(final_ff):
                ss_results = pairs_ss
                ff_results = pairs_ff
            else:
                ss_results = final_ss
                ff_results = final_ff

            if ss_results or ff_results:
                print(f"  Loaded {len(ss_results)} S-S and {len(ff_results)} F-F pair results")
                threshold = partition_metadata.get("threshold", 0.32)
                plot_bidirectional_progress_same_category(part1b_dir, ss_results, ff_results, threshold, seed=args.seed)
                print(f"  Plot saved to {part1b_dir / 'bidirectional_swap_plot.png'}")

                # Regenerate sweep plots if they exist
                sweep_dir = output_dir / f"subspace_threshold_sweep{pairing_suffix}"
                sweep_results_path = sweep_dir / "sweep_results.json"
                if sweep_results_path.exists():
                    print(f"  Regenerating subspace threshold sweep plots...")
                    sweep_data = load_json_safe(sweep_results_path)
                    if sweep_data:
                        alpha_values = sweep_data.get("alpha_values", [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
                        if sweep_data.get("individual_sweep"):
                            plot_subspace_threshold_individual(sweep_dir, sweep_data["individual_sweep"], alpha_values)
                        if sweep_data.get("pc_sweep"):
                            explained_var = np.array(sweep_data.get("explained_variance", []))
                            plot_subspace_threshold_pcs(sweep_dir, sweep_data["pc_sweep"], alpha_values, explained_var)
            else:
                print(f"  ERROR: No results found. Run without --plots-only first.")

            print("  [plots-only mode] Complete")
            continue

        # Normal execution
        D_S, D_F, per_concept_rates = load_baseline_detection_rates(
            steering_dir, args.model, args.layer, args.strength, success_concepts, failure_concepts
        )

        # Filter concepts by baseline detection rate thresholds
        ss_concepts_for_pairing = success_concepts
        ff_concepts_for_pairing = failure_concepts
        if args.min_success_baseline is not None:
            ss_concepts_for_pairing = [c for c in success_concepts
                                        if per_concept_rates.get(c, 0) >= args.min_success_baseline]
            print(f"  Filtered S concepts by baseline >= {args.min_success_baseline}: {len(ss_concepts_for_pairing)}/{len(success_concepts)}")
        if args.max_failure_baseline is not None:
            ff_concepts_for_pairing = [c for c in failure_concepts
                                        if per_concept_rates.get(c, 0) <= args.max_failure_baseline]
            print(f"  Filtered F concepts by baseline <= {args.max_failure_baseline}: {len(ff_concepts_for_pairing)}/{len(failure_concepts)}")

        # Find pairs
        if args.pairing_direction == "none":
            n_seeds = args.n_pairing_seeds
            print(f"  Finding S-S pairs (RANDOM pairing, {n_seeds} seed(s))...")
            ss_pairs, ss_cluster_info = find_random_pairs_same_category(
                concepts=ss_concepts_for_pairing,
                vectors=vectors,
                detection_rates=per_concept_rates,
                max_pairs=args.max_num_pairs,
                category_name="S-S",
                n_seeds=n_seeds,
            )
            print(f"    Found {len(ss_pairs)} S-S pairs")

            print(f"  Finding F-F pairs (RANDOM pairing, {n_seeds} seed(s))...")
            ff_pairs, ff_cluster_info = find_random_pairs_same_category(
                concepts=ff_concepts_for_pairing,
                vectors=vectors,
                detection_rates=per_concept_rates,
                max_pairs=args.max_num_pairs,
                category_name="F-F",
                n_seeds=n_seeds,
            )
            print(f"    Found {len(ff_pairs)} F-F pairs")

            # Merge existing completed pairs from disk
            part1b_dir = output_dir / f"part1_bidirectional_swaps_same_category{pairing_suffix}"
            existing_pairs_dir = part1b_dir / "pairs"
            if existing_pairs_dir.exists():
                existing_ss = []
                existing_ff = []
                for pf in sorted(existing_pairs_dir.glob("pair_*.json")):
                    try:
                        ep = load_json_safe(pf)
                        if not ep:
                            continue
                        cat = ep.get("category")
                        ca, cb = ep.get("concept_A", ""), ep.get("concept_B", "")
                        if ca not in vectors or cb not in vectors:
                            continue
                        da = per_concept_rates.get(ca, 0)
                        db = per_concept_rates.get(cb, 0)
                        pair_tuple = (ca, cb, 0.0, 0.0, da, db)
                        if cat == "S-S":
                            existing_ss.append(pair_tuple)
                        elif cat == "F-F":
                            existing_ff.append(pair_tuple)
                    except Exception:
                        continue

                existing_ss_keys = {frozenset([a, b]) for a, b, *_ in existing_ss}
                existing_ff_keys = {frozenset([a, b]) for a, b, *_ in existing_ff}

                merged_ss = list(existing_ss)
                for pair in ss_pairs:
                    if frozenset([pair[0], pair[1]]) not in existing_ss_keys:
                        merged_ss.append(pair)
                merged_ff = list(existing_ff)
                for pair in ff_pairs:
                    if frozenset([pair[0], pair[1]]) not in existing_ff_keys:
                        merged_ff.append(pair)

                ss_pairs = merged_ss
                ff_pairs = merged_ff

            # Enforce max_num_pairs
            if args.max_num_pairs is not None:
                ss_pairs = ss_pairs[:args.max_num_pairs]
                ff_pairs = ff_pairs[:args.max_num_pairs]
            print(f"    Final pair counts (max={args.max_num_pairs}): {len(ss_pairs)} S-S, {len(ff_pairs)} F-F")

            ss_cluster_info["n_pairs"] = len(ss_pairs)
            ss_cluster_info["n_clusters"] = len(ss_pairs)
            ss_cluster_info["cluster_ids"] = list(range(len(ss_pairs)))
            ss_cluster_info["cluster_sizes"] = [1] * len(ss_pairs)
            ff_cluster_info["n_pairs"] = len(ff_pairs)
            ff_cluster_info["n_clusters"] = len(ff_pairs)
            ff_cluster_info["cluster_ids"] = list(range(len(ff_pairs)))
            ff_cluster_info["cluster_sizes"] = [1] * len(ff_pairs)

        else:
            pairing_dir_name = "ridge" if args.pairing_direction == "ridge" else "mean-diff"
            print(f"  Finding S-S pairs (matched by {pairing_dir_name} projection)...")
            ss_pairs, ss_cluster_info = find_matched_pairs_same_category(
                concepts=ss_concepts_for_pairing,
                vectors=vectors,
                pairing_direction=pairing_direction,
                detection_rates=per_concept_rates,
                threshold=args.bidirectional_threshold,
                bucket_size=args.within_category_bucket_size,
                max_reuse=args.max_concept_reuse,
                category_name="S-S"
            )
            print(f"    Found {len(ss_pairs)} S-S pairs")

            print(f"  Finding F-F pairs (matched by {pairing_dir_name} projection)...")
            ff_pairs, ff_cluster_info = find_matched_pairs_same_category(
                concepts=ff_concepts_for_pairing,
                vectors=vectors,
                pairing_direction=pairing_direction,
                detection_rates=per_concept_rates,
                threshold=args.bidirectional_threshold,
                bucket_size=args.within_category_bucket_size,
                max_reuse=args.max_concept_reuse,
                category_name="F-F"
            )
            print(f"    Found {len(ff_pairs)} F-F pairs")

            if args.max_num_pairs is not None:
                if len(ss_pairs) > args.max_num_pairs:
                    ss_pairs = ss_pairs[:args.max_num_pairs]
                if len(ff_pairs) > args.max_num_pairs:
                    ff_pairs = ff_pairs[:args.max_num_pairs]

        # Load model and judge
        model = None
        judge = None
        if (len(ss_pairs) > 0 or len(ff_pairs) > 0) and not args.no_llm_judge:
            print("  Loading model for steering experiments...")
            model = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
            judge = LLMJudge()
            print(f"  Model and judge loaded successfully.")

        # Run same-category swaps
        same_cat_results = run_part1_bidirectional_swaps_same_category(
            output_dir=output_dir,
            success_pairs=ss_pairs,
            failure_pairs=ff_pairs,
            vectors=vectors,
            model=model,
            judge=judge,
            args=args,
            overwrite=args.overwrite,
            ss_cluster_info=ss_cluster_info,
            ff_cluster_info=ff_cluster_info,
            partition_threshold=partition_metadata.get("threshold", 0.32),
            pairing_suffix=pairing_suffix,
            mean_concept_norm=mean_concept_norm,
        )

        # Print summary
        if "statistics" in same_cat_results:
            stats = same_cat_results["statistics"]
            ss_stats = stats.get("S-S", {})
            ff_stats = stats.get("F-F", {})
            print(f"\n  Results:")
            print(f"    S-S pairs: n={ss_stats.get('n', 0)}, either_works={ss_stats.get('either_works_rate', 0):.1%}, mean_detection={ss_stats.get('mean_overall', 0):.1%}")
            print(f"    F-F pairs: n={ff_stats.get('n', 0)}, either_works={ff_stats.get('either_works_rate', 0):.1%}, mean_detection={ff_stats.get('mean_overall', 0):.1%}")

        # Run subspace threshold sweep
        if not args.skip_threshold_sweep and not args.no_llm_judge and model is not None:
            ss_pairs_results = same_cat_results.get("S-S_pairs", [])
            bidirectional_ss = [p for p in ss_pairs_results if p.get("bidirectional_success", False)]

            if len(bidirectional_ss) >= 3:
                print(f"\n{'='*80}")
                print("Subspace Threshold Sweep")
                print(f"{'='*80}")

                sweep_results = run_subspace_threshold_sweep(
                    output_dir=output_dir,
                    bidirectional_ss_pairs=bidirectional_ss,
                    vectors=vectors,
                    mean_diff=mean_diff,
                    model=model,
                    judge=judge,
                    args=args,
                    n_trials=args.sweep_n_trials,
                    samples_per_trial=args.sweep_samples_per_trial,
                    max_pairs=args.sweep_max_pairs,
                    n_pcs=args.n_pcs,
                    pairing_suffix=pairing_suffix,
                )
            else:
                print(f"\n  Skipping threshold sweep: only {len(bidirectional_ss)} bidirectional S-S pairs (need >= 3)")

        if model is not None:
            model.cleanup()
            del model
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("Bidirectional Steering Analysis Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
