#!/usr/bin/env python3
"""
Transcoder Feature Analysis: Detector Feature Analysis

Analyzes transcoder features one layer after the injection site to identify
gate features and evidence carrier features for anomaly detection.

Paper section: 5.3 ("Gate and Evidence Carrier Features")

For each steering layer L, analyzes layer L+1 features:
- L35 injection -> L36 features
- L38 injection -> L39 features
- L44 injection -> L45 features

Analysis pipeline:
1. Compute per-feature monotonicity (Spearman r) across steering strengths
2. Compute detection correlation (feature activation vs concept detection rate)
3. Compute direct logit attribution (DLA) per paper §5.3:
       (w_decoder_f . (u_Yes - u_No)) * mean_activation(f)
4. Rank features. Two metrics are available via --ranking-metric:
     - "detector_score" (default): mean_abs_r * detection_correlation;
       combines monotonicity and detection-predictiveness.
     - "logit_attribution": paper §5.3 DLA. Gate candidates are the top-200
       most-negative-DLA features (i.e., features that push Yes-No toward
       "No"). In practice the two rankings converge on overlapping gate
       features, but the paper's canonical formulation is DLA.
5. Identify evidence carrier features
6. Generate paper figures (activation vs steering strength curves)

Paper figures generated:
- top-logit-attribution-features (DLA bar chart from feature analysis)
- gate-example-1 (activation vs steering strength for gate features)
- evidence-feature-1, evidence-feature-2 (evidence carrier examples)

Output: analysis/07_transcoder_feature_analysis/L{steering_layer}/
"""

import argparse
import json
import gc
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

CACHED_ACTIVATIONS_BASE = Path("analysis/08_cached_activations")
OUTPUT_BASE = Path("analysis/07_transcoder_feature_analysis")
TRANSCODERS_BASE = Path("transcoders")
GEOMETRY_BASE = Path("analysis/04b_vector_geometry/gemma3_27b")

# Steering layer -> Detector layer mapping
DETECTOR_LAYERS = {29: 30, 35: 36, 37: 38, 38: 39, 44: 45}

# All available strengths
ALL_STRENGTHS = [-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]

# Monotonicity threshold (Spearman |r| above this = monotonic)
DEFAULT_MONOTONICITY_THRESHOLD = 0.7

# In/out tokens cache
INOUT_TOKENS_CACHE = Path("analysis/07_cached_inout/small_inout_tokens.json")

# Default strength for loading concept detection rates
DEFAULT_DETECTION_RATE_STRENGTH = 4.0


# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(
    description="Transcoder Feature Analysis: Detector Feature Analysis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-sl", "--steering-layers",
    type=int, nargs="+", default=[37],
    help="Steering layer(s) to analyze"
)
parser.add_argument(
    "-tm", "--token-mode",
    type=str, default="last_token",
    choices=["last_token", "mean_steering_tokens"],
    help="Token aggregation mode"
)
parser.add_argument(
    "-tl", "--transcoder-l0",
    type=str, default="small",
    choices=["big", "small"],
    help="Transcoder L0 variant"
)
parser.add_argument(
    "-mt", "--monotonicity-threshold",
    type=float, default=DEFAULT_MONOTONICITY_THRESHOLD,
    help="Spearman |r| threshold for monotonicity"
)
parser.add_argument(
    "-nf", "--n-top-features",
    type=int, default=100,
    help="Number of top features to analyze in detail"
)
parser.add_argument(
    "-np", "--n-plot-features",
    type=int, default=10,
    help="Number of top features to plot with activation curves"
)
parser.add_argument(
    "-po", "--plots-only",
    action="store_true",
    help="Only regenerate plots from saved data (skip computation)"
)
parser.add_argument(
    "--debug-sample",
    type=int, default=0,
    help="[DEBUG ONLY] Sample N features for faster testing (0 = all features)"
)
parser.add_argument(
    "--ranking-metric",
    type=str, default="detector_score",
    choices=["detector_score", "logit_attribution"],
    help=(
        "Ranking metric for the top-features JSON output. "
        "'detector_score' = mean_abs_r * detection_correlation (correlation-based). "
        "'logit_attribution' = paper §5.3 DLA: (w_dec · Δu_Yes-No) * mean_activation. "
        "Gate candidates are ranked most-negative-first under the DLA metric."
    ),
)

args = parser.parse_args()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FeatureMonotonicity:
    """Monotonicity statistics for a single feature."""
    feature_id: int
    physical_id: str  # e.g., "L45_F1234"

    # Per-concept Spearman r values
    concept_correlations: Dict[str, float] = field(default_factory=dict)

    # Aggregate statistics
    mean_abs_r: float = 0.0
    std_r: float = 0.0
    mean_r: float = 0.0

    # Monotonicity counts
    n_monotonic_pos: int = 0
    n_monotonic_neg: int = 0
    n_non_monotonic: int = 0

    # Concepts lists
    monotonic_pos_concepts: List[str] = field(default_factory=list)
    monotonic_neg_concepts: List[str] = field(default_factory=list)

    # Feature label (if available)
    label: str = ""

    # Detection correlation: correlation between activation and detection rate
    detection_correlation: float = 0.0

    # Direct logit attribution (DLA) to Yes-No logit difference,
    # as defined in paper §5.3: (w_decoder · (u_Yes - u_No)) * mean_activation.
    # Populated by compute_direct_logit_attribution(); negative values identify
    # gate candidates (features that push the Yes-No logit toward "No").
    logit_attribution: float = 0.0

    @property
    def n_monotonic(self) -> int:
        return self.n_monotonic_pos + self.n_monotonic_neg

    @property
    def monotonic_frac(self) -> float:
        total = self.n_monotonic + self.n_non_monotonic
        return self.n_monotonic / total if total > 0 else 0.0

    @property
    def direction_diversity(self) -> float:
        """How diverse are the monotonic directions? 0 = all same, 1 = balanced."""
        if self.n_monotonic == 0:
            return 0.0
        frac_pos = self.n_monotonic_pos / self.n_monotonic
        return 2 * min(frac_pos, 1 - frac_pos)

    @property
    def specialization_score(self) -> float:
        """Combined score for specialized detectors: high std, moderate mean."""
        return self.std_r * (1 - 0.3 * self.mean_abs_r)

    @property
    def detector_score(self) -> float:
        """Combined score for anomaly detectors: monotonic AND detection-predictive."""
        return self.mean_abs_r * self.detection_correlation


# =============================================================================
# Data Loading
# =============================================================================

def load_detector_activations(
    steering_layer: int,
    strength: float,
    token_mode: str,
    transcoder_l0: str,
) -> Tuple[Dict[str, torch.Tensor], List[str], int]:
    """
    Load activations for the detector layer (one after injection).

    Returns:
        activations: {concept: tensor of shape (n_trials, n_features)}
        concepts: List of concept names
        n_features: Number of features
    """
    detector_layer = DETECTOR_LAYERS[steering_layer]
    cache_path = CACHED_ACTIVATIONS_BASE / f"L{steering_layer}_S{strength}_{token_mode}_{transcoder_l0}" / "steered_activations.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    data = torch.load(cache_path, map_location='cpu', weights_only=False)

    activations = {}
    for concept in data['concepts']:
        if detector_layer in data['activations'][concept]:
            activations[concept] = data['activations'][concept][detector_layer]

    return activations, data['concepts'], data['n_features']


def load_feature_labels(detector_layer: int, transcoder_l0: str) -> Dict[int, str]:
    """Load feature labels for a transcoder layer from gemma-scope-2.

    NOTE (external dependency): labels are not bundled with this repo. They
    come from a sibling ``gemma-scope-2/feature_labels/`` directory; see the
    "External dependency: Gemma-Scope-2 feature labels" section in README.md.
    If the directory is missing, features will be identified by
    ``(layer, feature_idx)`` instead of semantic labels and the script will
    still run.
    """
    labels = {}

    gemma_scope_path = Path("gemma-scope-2/feature_labels")
    label_filename = f"gemma_scope_2_27b_transcoder_all_layer{detector_layer}_16k_{transcoder_l0}_labels.json"
    label_path = gemma_scope_path / label_filename

    if label_path.exists():
        with open(label_path) as f:
            label_data = json.load(f)
            for feat_id_str, feat_info in label_data.items():
                if isinstance(feat_info, dict) and 'title' in feat_info:
                    labels[int(feat_id_str)] = feat_info['title']
        print(f"  Loaded {len(labels)} feature labels from {label_path}")
    else:
        fallback_path = TRANSCODERS_BASE / f"layer_{detector_layer}" / transcoder_l0 / "feature_labels.json"
        if fallback_path.exists():
            with open(fallback_path) as f:
                label_data = json.load(f)
                for item in label_data:
                    if 'feature_id' in item and 'label' in item:
                        labels[item['feature_id']] = item['label']
            print(f"  Loaded {len(labels)} feature labels from {fallback_path}")
        else:
            print(f"  WARNING: No feature labels found at {label_path}")

    return labels


def load_geometry_partition(
    steering_layer: int,
    strength: float = DEFAULT_DETECTION_RATE_STRENGTH,
) -> Tuple[List[str], List[str]]:
    """
    Load success/failure concept partition from experiment 04b (vector geometry) subspace analysis.

    Returns:
        Tuple of (success_concepts, failure_concepts) lists
    """
    strengths_to_try = [strength, 4.0, 8.0, 2.0]

    for s in strengths_to_try:
        partition_path = GEOMETRY_BASE / f"layer_{steering_layer}_strength_{s}" / "detection_rate" / "subspace_analysis.json"
        if partition_path.exists():
            try:
                with open(partition_path) as f:
                    data = json.load(f)
                success = data.get('success_concepts', [])
                failure = data.get('failure_concepts', [])
                print(f"  Loaded experiment 04b (vector geometry) partition from {partition_path}")
                print(f"    Success concepts: {len(success)}, Failure concepts: {len(failure)}")
                return success, failure
            except Exception as e:
                print(f"  WARNING: Failed to load partition from {partition_path}: {e}")

    print(f"  WARNING: No experiment 04b (vector geometry) partition found for L{steering_layer}")
    return [], []


def load_concept_detection_rates(
    steering_layer: int,
    strength: float = DEFAULT_DETECTION_RATE_STRENGTH,
) -> Dict[str, float]:
    """
    Load per-concept detection rates from experiment 04b (vector geometry) geometry analysis.

    Returns:
        Dict mapping concept name -> detection rate (0.0 to 1.0)
    """
    detection_rates = {}
    strengths_to_try = [strength, 4.0, 8.0, 2.0, 1.0]

    for s in strengths_to_try:
        geometry_path = GEOMETRY_BASE / f"layer_{steering_layer}_strength_{s}" / "geometric_analysis.csv"
        if geometry_path.exists():
            try:
                df = pd.read_csv(geometry_path)
                if 'concept' in df.columns and 'detection_rate' in df.columns:
                    for _, row in df.iterrows():
                        detection_rates[row['concept']] = float(row['detection_rate'])
                    print(f"  Loaded {len(detection_rates)} concept detection rates from {geometry_path}")
                    return detection_rates
            except Exception as e:
                print(f"  WARNING: Failed to load detection rates from {geometry_path}: {e}")

    print(f"  WARNING: No concept detection rates found for L{steering_layer}")
    return detection_rates


# =============================================================================
# Feature Token Analysis
# =============================================================================

def load_inout_tokens_cache() -> Dict[str, Dict[str, List[str]]]:
    """Load cached in/out tokens for features."""
    if INOUT_TOKENS_CACHE.exists():
        with open(INOUT_TOKENS_CACHE) as f:
            return json.load(f)
    return {}


def save_inout_tokens_cache(cache: Dict[str, Dict[str, List[str]]]):
    """Save in/out tokens cache."""
    INOUT_TOKENS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(INOUT_TOKENS_CACHE, 'w') as f:
        json.dump(cache, f)


def load_transcoder_weights(layer: int, l0: str = "small"):
    """Load a GemmaScope-2 transcoder weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo_id = "google/gemma-scope-2-27b-it"
    width = "16k"
    filename = f"transcoder_all/layer_{layer}_width_{width}_l0_{l0}_affine/params.safetensors"

    print(f"    Loading transcoder from {repo_id}/{filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = load_file(path)

    class TranscoderWeights:
        def __init__(self, w_enc, w_dec):
            self.w_enc = torch.nn.Parameter(w_enc)
            self.w_dec = torch.nn.Parameter(w_dec)

    tc = TranscoderWeights(params["w_enc"], params["w_dec"])
    return tc


def compute_direct_logit_attribution(
    features: List["FeatureMonotonicity"],
    detector_layer: int,
    yes_tokens: Iterable[str] = ("yes", "Yes", "YES"),
    no_tokens: Iterable[str] = ("no", "No", "NO"),
    transcoder_l0: str = "small",
    model_name: str = "google/gemma-3-27b-it",
    device: Optional[str] = None,
) -> None:
    """Populate ``feature.logit_attribution`` with paper §5.3's exact DLA formula.

    For each transcoder feature ``f`` at ``detector_layer``, we compute::

        DLA(f) = (w_decoder_f · Δu_{Yes-No}) * mean_activation(f)

    where:
      - ``w_decoder_f`` is the transcoder decoder direction for feature ``f``
        (unit-normalized, following the paper's convention for transcoder DLA),
      - ``Δu_{Yes-No} = mean(u_t for t in yes_tokens) - mean(u_t for t in no_tokens)``
        with ``u_t`` taken from the unembedding matrix,
      - ``mean_activation(f)`` is the feature's mean activation across the
        concepts stored in ``concept_correlations`` (i.e., the monotonicity
        sample used elsewhere in this script).

    Features with **negative** ``logit_attribution`` push the Yes-No logit
    toward "No" and are selected as gate candidates. This matches the paper's
    "top-200 most-negative DLA" criterion when features are sorted ascending
    by ``logit_attribution``.

    This method MUTATES ``features`` in place and does not return a value.

    Notes:
      - We use the **decoder** direction for attribution (paper §5.3); this is
        the direction through which the feature writes to the residual stream.
      - ``transcoder_l0`` must match the transcoder width/L0 used to compute
        activations. Default ``"small"`` matches other defaults in this file.
      - The correlation-based ``detector_score`` is retained as an additional
        ranking metric; DLA and detector_score usually agree on the top gates
        but are not identical.
    """
    import numpy as _np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tc = load_transcoder_weights(detector_layer, transcoder_l0)
    w_dec = tc.w_dec.data.to(device=device, dtype=torch.float32)          # (n_features, d_model)
    w_dec_unit = w_dec / (w_dec.norm(dim=-1, keepdim=True) + 1e-10)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    shard_path = hf_hub_download(repo_id=model_name, filename="model-00001-of-00012.safetensors")
    unembed = None
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "embed_tokens.weight" in key:
                unembed = f.get_tensor(key).to(device=device, dtype=torch.float32)
                break
    if unembed is None:
        raise RuntimeError("Could not locate unembedding weights in the first shard.")

    def _avg_embed(tokens):
        vecs = []
        for t in tokens:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                vecs.append(unembed[ids[0]])
        if not vecs:
            raise ValueError(f"None of the provided tokens are single-token: {list(tokens)}")
        return torch.stack(vecs).mean(dim=0)

    delta_u = _avg_embed(yes_tokens) - _avg_embed(no_tokens)              # (d_model,)
    feature_unembed_scores = w_dec_unit @ delta_u                         # (n_features,)

    for feat in features:
        feat_id = int(feat.feature_id)
        if feat_id < 0 or feat_id >= feature_unembed_scores.shape[0]:
            feat.logit_attribution = 0.0
            continue
        # Mean absolute activation across the sampled concepts.
        acts = [abs(v) for v in feat.concept_correlations.values() if _np.isfinite(v)]
        mean_act = float(_np.mean(acts)) if acts else 0.0
        feat.logit_attribution = float(feature_unembed_scores[feat_id].item()) * mean_act


def is_ascii_readable(token: str) -> bool:
    """Check if token is ASCII and readable."""
    if not token or len(token) == 0:
        return False
    try:
        token.encode('ascii')
        return any(c.isalnum() for c in token) and len(token) < 30
    except UnicodeEncodeError:
        return False


def filter_ascii_tokens(tokens: List[str], n: int = 4) -> List[str]:
    """Filter to only ASCII-readable tokens and return top n."""
    return [t for t in tokens if is_ascii_readable(t)][:n]


def compute_feature_inout_tokens(
    layer: int,
    feature_ids: List[int],
    transcoder_l0: str = "small",
    n_tokens: int = 20,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Compute in/out tokens for transcoder features using logit lens.

    Returns:
        Dict mapping physical_id -> {"in": [...], "out": [...]}
    """
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if not feature_ids:
        return {}

    print(f"    Computing in/out tokens for {len(feature_ids)} features at L{layer}...")

    tc = load_transcoder_weights(layer, transcoder_l0)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    print("    Loading embedding weights...")
    embed_path = hf_hub_download(
        repo_id="google/gemma-3-27b-it",
        filename="model-00001-of-00012.safetensors"
    )
    embed_weights = load_file(embed_path)

    if "model.embed_tokens.weight" in embed_weights:
        unembed = embed_weights["model.embed_tokens.weight"]
    else:
        for key in embed_weights.keys():
            if "embed" in key.lower():
                unembed = embed_weights[key]
                break
        else:
            print("    ERROR: No embedding weights found")
            return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unembed = unembed.to(device=device, dtype=tc.w_dec.dtype)
    tc.w_enc = torch.nn.Parameter(tc.w_enc.to(device=device))
    tc.w_dec = torch.nn.Parameter(tc.w_dec.to(device=device))
    unembed_T = unembed.T

    results = {}
    for feat_id in tqdm(feature_ids, desc="    Computing tokens", leave=False):
        physical_id = f"L{layer}_F{feat_id}"

        w_dec_feature = tc.w_dec.data[feat_id]
        feature_logits = w_dec_feature @ unembed_T
        top_out_idx = feature_logits.argsort(descending=True)[:n_tokens]
        out_tokens = [tokenizer.decode([t.item()]).strip() for t in top_out_idx]

        w_enc_feature = tc.w_enc.data[:, feat_id].to(unembed_T.dtype)
        encoder_logits = w_enc_feature @ unembed_T
        top_enc_idx = encoder_logits.argsort(descending=True)[:n_tokens]
        in_tokens = [tokenizer.decode([t.item()]).strip() for t in top_enc_idx]

        results[physical_id] = {"in": in_tokens, "out": out_tokens}

    del tc, unembed, unembed_T
    torch.cuda.empty_cache()
    gc.collect()

    return results


def get_feature_inout_tokens(
    features: List[FeatureMonotonicity],
    transcoder_l0: str = "small",
) -> Dict[str, Dict[str, List[str]]]:
    """Get in/out tokens for features, computing missing ones and updating cache."""
    cache = load_inout_tokens_cache()

    missing_by_layer: Dict[int, List[int]] = defaultdict(list)
    for feat in features:
        if feat.physical_id not in cache:
            match = re.match(r'L(\d+)_F(\d+)', feat.physical_id)
            if match:
                layer = int(match.group(1))
                missing_by_layer[layer].append(feat.feature_id)

    if missing_by_layer:
        total_missing = sum(len(ids) for ids in missing_by_layer.values())
        print(f"    Computing in/out tokens for {total_missing} uncached features...")

        for layer, feat_ids in missing_by_layer.items():
            new_tokens = compute_feature_inout_tokens(
                layer=layer,
                feature_ids=feat_ids,
                transcoder_l0=transcoder_l0,
            )
            cache.update(new_tokens)

        save_inout_tokens_cache(cache)

    return cache


# =============================================================================
# Vectorized Spearman Correlation (Memory-Safe)
# =============================================================================

def compute_spearman_chunk(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Spearman correlation between x and each column of Y.
    Uses float32 to reduce memory.

    Args:
        x: Shape (n,)
        Y: Shape (n, m)

    Returns:
        r: Shape (m,) - Spearman r for each column
    """
    n, m = Y.shape

    x_order = x.argsort()
    x_ranks = np.empty(n, dtype=np.float32)
    x_ranks[x_order] = np.arange(1, n + 1, dtype=np.float32)

    y_order = Y.argsort(axis=0)
    y_ranks = np.empty((n, m), dtype=np.float32)

    for i in range(n):
        y_ranks[y_order[i], np.arange(m)] = i + 1

    del y_order
    gc.collect()

    x_mean = x_ranks.mean()
    y_mean = y_ranks.mean(axis=0)

    x_centered = (x_ranks - x_mean).reshape(-1, 1)
    y_centered = y_ranks - y_mean

    numerator = (x_centered * y_centered).sum(axis=0)
    x_var = (x_centered ** 2).sum()
    y_var = (y_centered ** 2).sum(axis=0)

    denominator = np.sqrt(x_var * y_var)

    with np.errstate(invalid='ignore', divide='ignore'):
        r = numerator / denominator

    return np.nan_to_num(r, nan=0.0).astype(np.float32)


def compute_spearman_vectorized(x: np.ndarray, Y: np.ndarray, chunk_size: int = 500000) -> np.ndarray:
    """
    Compute Spearman correlation between 1D array x and each column of 2D array Y.
    Processes in chunks to avoid OOM errors.

    Args:
        x: Shape (n,) - the x values (e.g., strength indices)
        Y: Shape (n, m) - each column is a y series (e.g., activations)
        chunk_size: Number of columns to process at once

    Returns:
        r: Shape (m,) - Spearman r for each column
    """
    n, m = Y.shape

    mem_per_chunk_mb = 4 * n * chunk_size * 4 / (1024 * 1024)
    n_chunks = (m + chunk_size - 1) // chunk_size
    print(f"    Processing {m:,} correlations in {n_chunks} chunks (~{mem_per_chunk_mb:.0f}MB per chunk)")

    if m <= chunk_size:
        return compute_spearman_chunk(x, Y)

    results = []
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        chunk = Y[:, start:end]
        r_chunk = compute_spearman_chunk(x, chunk)
        results.append(r_chunk)
        gc.collect()

    return np.concatenate(results)


# =============================================================================
# Monotonicity Analysis
# =============================================================================

def compute_feature_monotonicity(
    steering_layer: int,
    token_mode: str,
    transcoder_l0: str,
    monotonicity_threshold: float,
    feature_labels: Dict[int, str],
    concept_detection_rates: Dict[str, float],
    debug_sample: int = 0,
) -> List[FeatureMonotonicity]:
    """
    Compute monotonicity statistics for all features in detector layer.

    Uses fully vectorized Spearman correlation. Also computes
    detection_correlation: correlation between feature activation and concept
    detection rate (at strength=4.0).
    """
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n[L{steering_layer}->L{detector_layer}] Computing feature monotonicity...")

    # Load activations at all strengths
    print("  Loading activations at all strengths...")
    all_activations = {}
    concepts = None
    n_features = None

    for strength in tqdm(ALL_STRENGTHS, desc="  Loading"):
        try:
            acts, conc, nf = load_detector_activations(
                steering_layer, strength, token_mode, transcoder_l0
            )
            all_activations[strength] = acts
            if concepts is None:
                concepts = conc
                n_features = nf
        except FileNotFoundError as e:
            print(f"    Warning: {e}")

    if not all_activations:
        raise ValueError(f"No activations found for L{steering_layer}")

    # Determine which features to analyze
    if debug_sample > 0 and debug_sample < n_features:
        np.random.seed(42)
        feature_indices = np.sort(np.random.choice(n_features, debug_sample, replace=False))
        print(f"  [DEBUG] Sampling {debug_sample} features (out of {n_features})")
        actual_n_features = debug_sample
    else:
        feature_indices = np.arange(n_features)
        print(f"  Analyzing all {n_features} features")
        actual_n_features = n_features

    n_strengths = len(ALL_STRENGTHS)
    n_concepts = len(concepts)

    # Pre-compute all mean activations
    print(f"  Pre-computing mean activations ({n_strengths} x {n_concepts} x {actual_n_features})...")
    all_means = np.zeros((n_strengths, n_concepts, actual_n_features), dtype=np.float32)

    for s_idx, strength in enumerate(tqdm(ALL_STRENGTHS, desc="  Means")):
        if strength in all_activations:
            strength_acts = all_activations[strength]
            for c_idx, concept in enumerate(concepts):
                if concept in strength_acts:
                    tensor = strength_acts[concept]
                    if debug_sample > 0:
                        all_means[s_idx, c_idx, :] = tensor[:, feature_indices].float().mean(dim=0).numpy()
                    else:
                        all_means[s_idx, c_idx, :] = tensor.float().mean(dim=0).numpy()

            del all_activations[strength]
            gc.collect()

    del all_activations
    gc.collect()

    # Compute ALL Spearman correlations (chunked for memory safety)
    print(f"  Computing {n_concepts * actual_n_features:,} Spearman correlations (vectorized)...")

    all_means_flat = all_means.reshape(n_strengths, -1)
    x = np.arange(n_strengths, dtype=np.float64)

    t0 = time.time()
    r_flat = compute_spearman_vectorized(x, all_means_flat)
    t1 = time.time()
    print(f"    Vectorized correlation took {t1-t0:.2f}s")

    r_matrix = r_flat.reshape(n_concepts, actual_n_features)

    # Compute detection correlation for each feature
    print(f"  Computing detection correlations...")

    detection_corr_per_feature = np.zeros(actual_n_features, dtype=np.float32)

    if concept_detection_rates:
        det_rates = np.array([concept_detection_rates.get(c, np.nan) for c in concepts], dtype=np.float32)
        valid_det_mask = ~np.isnan(det_rates)

        if valid_det_mask.sum() > 10:
            try:
                acts_s4, _, _ = load_detector_activations(
                    steering_layer, 4.0, token_mode, transcoder_l0
                )
                mean_acts_s4 = np.zeros((n_concepts, actual_n_features), dtype=np.float32)
                for c_idx, concept in enumerate(concepts):
                    if concept in acts_s4:
                        tensor = acts_s4[concept]
                        if debug_sample > 0:
                            mean_acts_s4[c_idx, :] = tensor[:, feature_indices].float().mean(dim=0).numpy()
                        else:
                            mean_acts_s4[c_idx, :] = tensor.float().mean(dim=0).numpy()

                del acts_s4
                gc.collect()

                det_rates_valid = det_rates[valid_det_mask]
                mean_acts_valid = mean_acts_s4[valid_det_mask, :]

                t0 = time.time()
                detection_corr_per_feature = compute_spearman_chunk(
                    det_rates_valid.astype(np.float64),
                    mean_acts_valid.astype(np.float64)
                )
                t1 = time.time()
                print(f"    Detection correlation took {t1-t0:.2f}s")

                del mean_acts_s4
                gc.collect()

            except Exception as e:
                print(f"    WARNING: Could not compute detection correlation: {e}")
    else:
        print(f"    WARNING: No detection rates provided, skipping detection correlation")

    del all_means, all_means_flat, r_flat
    gc.collect()

    # Build FeatureMonotonicity objects
    print(f"  Building feature objects...")
    results = []

    for f_local_idx, feat_idx in enumerate(tqdm(feature_indices, desc="  Features")):
        physical_id = f"L{detector_layer}_F{feat_idx}"

        mono = FeatureMonotonicity(
            feature_id=feat_idx,
            physical_id=physical_id,
            label=feature_labels.get(feat_idx, ""),
            detection_correlation=float(detection_corr_per_feature[f_local_idx]),
        )

        r_values = r_matrix[:, f_local_idx]
        valid_mask = ~np.isnan(r_values) & (r_values != 0)

        for c_idx, concept in enumerate(concepts):
            r = r_values[c_idx]
            if valid_mask[c_idx]:
                mono.concept_correlations[concept] = float(r)

                if r > monotonicity_threshold:
                    mono.n_monotonic_pos += 1
                    mono.monotonic_pos_concepts.append(concept)
                elif r < -monotonicity_threshold:
                    mono.n_monotonic_neg += 1
                    mono.monotonic_neg_concepts.append(concept)
                else:
                    mono.n_non_monotonic += 1

        valid_r = r_values[valid_mask]
        if len(valid_r) > 0:
            mono.mean_abs_r = float(np.abs(valid_r).mean())
            mono.std_r = float(valid_r.std())
            mono.mean_r = float(valid_r.mean())

        results.append(mono)

    gc.collect()
    return results


def save_computed_results(features: List[FeatureMonotonicity], output_dir: Path):
    """Save computed results to pickle for --plots-only mode."""
    pickle_path = output_dir / "computed_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"  Saved computed results to {pickle_path}")


def load_computed_results(output_dir: Path) -> List[FeatureMonotonicity]:
    """Load computed results from pickle."""
    pickle_path = output_dir / "computed_results.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(f"No computed results found at {pickle_path}. Run without --plots-only first.")
    with open(pickle_path, 'rb') as f:
        features = pickle.load(f)
    print(f"  Loaded {len(features)} features from {pickle_path}")
    return features


# =============================================================================
# Plotting Label Tracking
# =============================================================================

class PlottedFeatureTracker:
    """Tracks which features are plotted and their labels."""

    def __init__(self, feature_labels: Dict[int, str]):
        self.feature_labels = feature_labels
        self.plotted_features: Set[Tuple[int, str]] = set()

    def add(self, feature: FeatureMonotonicity):
        self.plotted_features.add((feature.feature_id, feature.physical_id))

    def add_by_id(self, feature_id: int, physical_id: str):
        self.plotted_features.add((feature_id, physical_id))

    def get_label(self, feature_id: int) -> str:
        return self.feature_labels.get(feature_id, "")

    def format_feature_title(self, feature: FeatureMonotonicity, max_label_len: int = 50) -> str:
        title = feature.physical_id
        if feature.label:
            label = feature.label[:max_label_len] + "..." if len(feature.label) > max_label_len else feature.label
            title += f"\n\"{label}\""
        return title

    def get_missing_labels(self) -> List[Dict]:
        missing = []
        for feat_id, physical_id in sorted(self.plotted_features):
            if feat_id not in self.feature_labels:
                match = re.match(r'L(\d+)_F(\d+)', physical_id)
                if match:
                    layer = int(match.group(1))
                    missing.append({
                        'feature_id': int(feat_id),
                        'physical_id': physical_id,
                        'layer': layer,
                    })
        return missing

    def save_missing_labels(self, output_dir: Path, context: str = ""):
        missing = self.get_missing_labels()
        total_plotted = len(self.plotted_features)

        data = {
            'summary': {
                'total_plotted_features': total_plotted,
                'features_with_labels': total_plotted - len(missing),
                'features_missing_labels': len(missing),
                'coverage': f"{(total_plotted - len(missing)) / total_plotted * 100:.1f}%" if total_plotted > 0 else "N/A",
            },
            'context': context,
            'missing_features': missing,
        }

        with open(output_dir / "missing_feature_labels.json", 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Missing labels: {len(missing)}/{total_plotted} plotted features")


# =============================================================================
# Analysis & Plotting
# =============================================================================

def analyze_monotonicity_distribution(
    features: List[FeatureMonotonicity],
    steering_layer: int,
    output_dir: Path,
    monotonicity_threshold: float,
    tracker: PlottedFeatureTracker,
):
    """Analyze and plot the distribution of monotonicity patterns."""
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n  Analyzing monotonicity distribution...")

    mean_abs_r = np.array([f.mean_abs_r for f in features])
    std_r = np.array([f.std_r for f in features])
    n_monotonic = np.array([f.n_monotonic for f in features])
    monotonic_frac = np.array([f.monotonic_frac for f in features])
    direction_diversity = np.array([f.direction_diversity for f in features])
    specialization = np.array([f.specialization_score for f in features])

    print(f"    Total features: {len(features)}")
    print(f"    Mean |r| distribution: mean={mean_abs_r.mean():.3f}, max={mean_abs_r.max():.3f}")
    print(f"    Features with >50% monotonic concepts: {(monotonic_frac > 0.5).sum()}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.hist(mean_abs_r, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5')
    ax.set_xlabel('Mean |Spearman r| across concepts')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Mean monotonicity')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(std_r, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Std(r) across concepts')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Specialization')

    ax = axes[0, 2]
    ax.hist(monotonic_frac, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel(f'Fraction of concepts with |r| > {monotonicity_threshold}')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Monotonic fraction')

    ax = axes[1, 0]
    scatter = ax.scatter(mean_abs_r, std_r, c=direction_diversity, cmap='viridis',
                         alpha=0.5, s=10)
    ax.set_xlabel('Mean |r| across concepts')
    ax.set_ylabel('Std(r) across concepts')
    ax.set_title('Monotonicity vs Specialization\n(color = direction diversity)')
    plt.colorbar(scatter, ax=ax, label='Direction diversity')

    ax = axes[1, 1]
    ax.hist(direction_diversity, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Direction diversity (0=same, 1=balanced)')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Direction diversity')

    ax = axes[1, 2]
    ax.hist(specialization, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Specialization score')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Combined specialization')

    plt.tight_layout()
    plt.savefig(output_dir / "monotonicity_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: monotonicity_distribution.png")


def analyze_detection_distribution(
    features: List[FeatureMonotonicity],
    steering_layer: int,
    output_dir: Path,
    tracker: PlottedFeatureTracker,
):
    """Analyze and plot the distribution of detection correlation and detector scores."""
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n  Analyzing detection score distribution...")

    mean_abs_r = np.array([f.mean_abs_r for f in features])
    detection_corr = np.array([f.detection_correlation for f in features])
    detector_score = np.array([f.detector_score for f in features])
    specialization = np.array([f.specialization_score for f in features])
    std_r = np.array([f.std_r for f in features])

    print(f"    Detection correlation: mean={detection_corr.mean():.3f}, std={detection_corr.std():.3f}")
    print(f"    Features with det_corr > 0.3: {(detection_corr > 0.3).sum()}")
    print(f"    Detector score: mean={detector_score.mean():.3f}, max={detector_score.max():.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    ax.hist(detection_corr, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.7, label='0.3')
    ax.set_xlabel('Detection correlation')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Detection correlation distribution')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(detector_score, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.7, label='0.3')
    ax.set_xlabel('Detector score (mean|r| x det_corr)')
    ax.set_ylabel('Count (features)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Detector score distribution')
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    scatter = ax.scatter(mean_abs_r, detection_corr, c=detector_score, cmap='plasma',
                         alpha=0.5, s=10)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean |r| (monotonicity)')
    ax.set_ylabel('Detection correlation')
    ax.set_title('Monotonicity vs Detection Predictiveness\n(color = detector score)')
    plt.colorbar(scatter, ax=ax, label='Detector score')

    top_idx = np.argsort(detector_score)[-5:]
    for idx in top_idx:
        ax.annotate(f'F{features[idx].feature_id}',
                   (mean_abs_r[idx], detection_corr[idx]), fontsize=7, alpha=0.8)

    ax = axes[1, 0]
    scatter = ax.scatter(specialization, detector_score, c=detection_corr, cmap='coolwarm',
                         alpha=0.5, s=10)
    ax.set_xlabel('Specialization score')
    ax.set_ylabel('Detector score')
    ax.set_title('Specialization vs Anomaly Detection\n(color = detection correlation)')
    plt.colorbar(scatter, ax=ax, label='Detection corr')

    ax = axes[1, 1]
    scatter = ax.scatter(std_r, detection_corr, c=mean_abs_r, cmap='viridis',
                         alpha=0.5, s=10)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Std(r) across concepts')
    ax.set_ylabel('Detection correlation')
    ax.set_title('Variance vs Detection Correlation\n(color = mean |r|)')
    plt.colorbar(scatter, ax=ax, label='Mean |r|')

    ax = axes[1, 2]
    rank_detector = np.argsort(np.argsort(-detector_score))
    rank_special = np.argsort(np.argsort(-specialization))
    scatter = ax.scatter(rank_special, rank_detector, c=detection_corr, cmap='coolwarm',
                         alpha=0.3, s=5)
    ax.plot([0, len(features)], [0, len(features)], 'k--', alpha=0.3, label='Same rank')
    ax.set_xlabel('Rank by specialization score')
    ax.set_ylabel('Rank by detector score')
    ax.set_title('Ranking comparison\n(color = detection correlation)')
    ax.set_xlim(0, min(1000, len(features)))
    ax.set_ylim(0, min(1000, len(features)))
    plt.colorbar(scatter, ax=ax, label='Detection corr')

    plt.tight_layout()
    plt.savefig(output_dir / "detection_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: detection_distribution.png")


def find_specialized_detectors(
    features: List[FeatureMonotonicity],
    n_top: int = 100,
) -> Tuple[List[FeatureMonotonicity], List[FeatureMonotonicity], List[FeatureMonotonicity], List[FeatureMonotonicity]]:
    """
    Find different types of interesting features:
    1. Specialized detectors: high std(r), mixed directions
    2. General detectors: high mean |r|, monotonic for most concepts
    3. Direction-diverse: high n_monotonic with mixed positive/negative
    4. Anomaly detectors: high detector_score (monotonic AND detection-predictive)

    Returns four lists of top features by each criterion.
    """
    by_specialization = sorted(features, key=lambda f: f.specialization_score, reverse=True)
    by_mean_r = sorted(features, key=lambda f: f.mean_abs_r, reverse=True)
    by_diversity = sorted(features, key=lambda f: (f.direction_diversity, f.n_monotonic), reverse=True)
    by_detector_score = sorted(features, key=lambda f: f.detector_score, reverse=True)

    return by_specialization[:n_top], by_mean_r[:n_top], by_diversity[:n_top], by_detector_score[:n_top]


def plot_top_features(
    features: List[FeatureMonotonicity],
    steering_layer: int,
    token_mode: str,
    transcoder_l0: str,
    output_dir: Path,
    category: str,
    n_plot: int,
    tracker: PlottedFeatureTracker,
    concept_detection_rates: Dict[str, float] = None,
):
    """Plot activation vs strength curves for top features (paper figures)."""
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n  Plotting top {n_plot} {category} features...")

    if concept_detection_rates is None:
        concept_detection_rates = {}

    features_to_plot = features[:n_plot]
    inout_cache = get_feature_inout_tokens(features_to_plot, transcoder_l0)

    # Load activations at all strengths
    all_activations = {}
    for strength in ALL_STRENGTHS:
        try:
            acts, _, _ = load_detector_activations(
                steering_layer, strength, token_mode, transcoder_l0
            )
            all_activations[strength] = acts
        except FileNotFoundError:
            pass

    n_cols = 2
    n_rows = (n_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7.5 * n_rows))
    axes = axes.flatten()

    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]
        tracker.add(feat)

        concepts_sorted = sorted(
            feat.concept_correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select diverse set: top 3 positive, top 3 negative, 3 near zero
        top_pos = concepts_sorted[:3]
        top_neg = concepts_sorted[-3:]
        mid = [c for c in concepts_sorted if abs(c[1]) < 0.3][:3]
        selected = top_pos + top_neg + mid

        cmap = plt.cm.coolwarm

        for concept, r_val in selected:
            acts = []
            valid_strengths = []

            for strength in ALL_STRENGTHS:
                if strength in all_activations and concept in all_activations[strength]:
                    tensor = all_activations[strength][concept]
                    if feat.feature_id < tensor.shape[1]:
                        mean_act = tensor[:, feat.feature_id].float().mean().item()
                        acts.append(mean_act)
                        valid_strengths.append(strength)

            if acts:
                color = cmap((r_val + 1) / 2)
                det_rate = concept_detection_rates.get(concept, None)
                concept_short = concept[:12] + "..." if len(concept) > 12 else concept
                if det_rate is not None:
                    label = f"{concept_short} (r={r_val:.2f}, det={det_rate:.0%})"
                else:
                    label = f"{concept_short} (r={r_val:.2f})"
                ax.plot(valid_strengths, acts, 'o-', color=color, alpha=0.7,
                       linewidth=2.5, markersize=5, label=label)

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Steering strength', fontsize=11)
        ax.set_ylabel('Mean activation', fontsize=11)

        inout = inout_cache.get(feat.physical_id, {})
        in_tokens = filter_ascii_tokens(inout.get('in', []), n=4)
        out_tokens = filter_ascii_tokens(inout.get('out', []), n=4)

        title_lines = [feat.physical_id]
        if feat.label:
            title_lines.append(feat.label)
        title_lines.append(f"mean|r|={feat.mean_abs_r:.2f}, std(r)={feat.std_r:.2f}, "
                          f"div={feat.direction_diversity:.2f}, det_corr={feat.detection_correlation:.2f}")
        if in_tokens or out_tokens:
            in_str = ', '.join(in_tokens) if in_tokens else '?'
            out_str = ', '.join(out_tokens) if out_tokens else '?'
            title_lines.append(f"in: [{in_str}]  out: [{out_str}]")

        ax.set_title('\n'.join(title_lines), fontsize=16, linespacing=1.1)
        ax.legend(loc='upper left', fontsize=8)

    for idx in range(n_plot, len(axes)):
        axes[idx].set_visible(False)

    category_display = category.replace('_', ' ')
    fig.suptitle(f'L{steering_layer}->L{detector_layer}: Top {category_display} features', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f"top_{category}_features.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: top_{category}_features.png")

    del all_activations
    gc.collect()


def analyze_concept_coverage(
    features: List[FeatureMonotonicity],
    steering_layer: int,
    output_dir: Path,
    n_top: int,
    tracker: PlottedFeatureTracker,
):
    """
    Analyze how well the top features cover different concepts.
    Goal: Find if different features specialize in different concept subsets.
    """
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n  Analyzing concept coverage for top {n_top} features...")

    top_features = sorted(features, key=lambda f: f.specialization_score, reverse=True)[:n_top]

    for f in top_features:
        tracker.add(f)

    all_concepts = set()
    for f in top_features:
        all_concepts.update(f.monotonic_pos_concepts)
        all_concepts.update(f.monotonic_neg_concepts)
    all_concepts = sorted(all_concepts)

    if not all_concepts:
        print("    No monotonic concepts found")
        return

    print(f"    Total concepts with monotonic behavior: {len(all_concepts)}")

    concept_to_idx = {c: i for i, c in enumerate(all_concepts)}
    feature_ids = [f.physical_id for f in top_features]

    matrix = np.zeros((len(all_concepts), n_top))
    for f_idx, feat in enumerate(top_features):
        for concept, r in feat.concept_correlations.items():
            if concept in concept_to_idx:
                matrix[concept_to_idx[concept], f_idx] = r

    threshold = 0.7
    concept_coverage = (np.abs(matrix) > threshold).sum(axis=1)

    print(f"    Concepts detected by >=1 feature: {(concept_coverage >= 1).sum()}")
    print(f"    Concepts detected by >=3 features: {(concept_coverage >= 3).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    ax = axes[0]
    if n_top > 2:
        linkage_matrix = linkage(matrix.T, method='ward')
        order = dendrogram(linkage_matrix, no_plot=True)['leaves']
        matrix_ordered = matrix[:, order]
        feature_ids_ordered = [feature_ids[i] for i in order]
    else:
        matrix_ordered = matrix
        feature_ids_ordered = feature_ids

    top_concept_indices = np.argsort(concept_coverage)[-100:][::-1]
    matrix_subset = matrix_ordered[top_concept_indices, :]
    concepts_subset = [all_concepts[i] for i in top_concept_indices]

    im = ax.imshow(matrix_subset, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Concept')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Feature-Concept correlations\n'
                 f'(top {len(concepts_subset)} concepts by coverage)')
    ax.set_xticks(range(0, n_top, 5))
    ax.set_xticklabels([feature_ids_ordered[i][4:] for i in range(0, n_top, 5)],
                       rotation=45, ha='right', fontsize=7)
    plt.colorbar(im, ax=ax, label='Spearman r')

    ax = axes[1]
    ax.hist(concept_coverage, bins=range(0, n_top + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of features detecting concept')
    ax.set_ylabel('Count (concepts)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Concept coverage distribution')
    ax.axvline(1, color='red', linestyle='--', alpha=0.7, label='Detected by >=1')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "concept_coverage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: concept_coverage.png")

    return matrix, all_concepts, feature_ids


def analyze_feature_directions(
    features: List[FeatureMonotonicity],
    steering_layer: int,
    output_dir: Path,
    n_top: int,
    tracker: PlottedFeatureTracker,
):
    """
    Analyze if top features point in different directions in concept space.
    Uses PCA on the feature correlation vectors.
    """
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n  Analyzing feature directions...")

    top_features = sorted(features, key=lambda f: f.specialization_score, reverse=True)[:n_top]

    for f in top_features:
        tracker.add(f)

    all_concepts = sorted(set().union(*[set(f.concept_correlations.keys()) for f in top_features]))

    if len(all_concepts) < 10:
        print("    Not enough concepts for direction analysis")
        return

    concept_to_idx = {c: i for i, c in enumerate(all_concepts)}
    X = np.zeros((n_top, len(all_concepts)))

    for f_idx, feat in enumerate(top_features):
        for concept, r in feat.concept_correlations.items():
            if concept in concept_to_idx:
                X[f_idx, concept_to_idx[concept]] = r

    pca = PCA(n_components=min(10, n_top, len(all_concepts)))
    X_pca = pca.fit_transform(X)

    colors_by_diversity = [f.direction_diversity for f in top_features]
    colors_by_mean_r = [f.mean_abs_r for f in top_features]
    colors_by_det_corr = [f.detection_correlation for f in top_features]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_by_diversity,
                        cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Feature directions (PCA)\nColor = direction diversity')
    plt.colorbar(scatter, ax=ax)

    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_by_mean_r,
                        cmap='plasma', s=50, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Feature directions (PCA)\nColor = mean |r|')
    plt.colorbar(scatter, ax=ax)

    ax = axes[1, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_by_det_corr,
                        cmap='coolwarm', s=50, alpha=0.7, vmin=-0.3, vmax=0.6)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: Feature directions (PCA)\nColor = detection correlation')
    plt.colorbar(scatter, ax=ax)

    ax = axes[1, 1]
    ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'L{steering_layer}->L{detector_layer}: PCA explained variance')
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2 = ax.twinx()
    ax2.plot(range(len(cumsum)), cumsum, 'r-o', label='Cumulative')
    ax2.set_ylabel('Cumulative explained variance', color='r')
    ax2.axhline(0.9, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_directions_pca.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: feature_directions_pca.png")
    print(f"    PC1-2 explain {cumsum[1]*100:.1f}% of variance")


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    features: List[FeatureMonotonicity],
    steering_layer: int,
    output_dir: Path,
    ranking_metric: str = "detector_score",
):
    """Save feature monotonicity + DLA results to CSV and JSON.

    Args:
        ranking_metric: "detector_score" (correlation-based, default) or
            "logit_attribution" (paper §5.3 DLA). Under "logit_attribution",
            the top-200 list is the 200 most-negative features (gate
            candidates, i.e., features that push Yes-No toward "No").
    """
    detector_layer = DETECTOR_LAYERS[steering_layer]
    print(f"\n  Saving results (ranking_metric={ranking_metric})...")

    rows = []
    for f in features:
        rows.append({
            'feature_id': f.feature_id,
            'physical_id': f.physical_id,
            'label': f.label,
            'mean_abs_r': f.mean_abs_r,
            'std_r': f.std_r,
            'mean_r': f.mean_r,
            'n_monotonic_pos': f.n_monotonic_pos,
            'n_monotonic_neg': f.n_monotonic_neg,
            'n_non_monotonic': f.n_non_monotonic,
            'monotonic_frac': f.monotonic_frac,
            'direction_diversity': f.direction_diversity,
            'specialization_score': f.specialization_score,
            'detection_correlation': f.detection_correlation,
            'detector_score': f.detector_score,
            'logit_attribution': f.logit_attribution,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('detector_score', ascending=False)
    df.to_csv(output_dir / "feature_monotonicity.csv", index=False)
    print(f"    Saved: feature_monotonicity.csv ({len(df)} features)")

    if ranking_metric == "logit_attribution":
        # Gate candidates: top-200 most-negative DLA features.
        top_features = sorted(features, key=lambda f: f.logit_attribution)[:200]
    else:
        top_features = sorted(features, key=lambda f: f.detector_score, reverse=True)[:200]

    json_data = {
        'steering_layer': steering_layer,
        'detector_layer': detector_layer,
        'n_features': len(features),
        'n_top': len(top_features),
        'ranking_metric': ranking_metric,
        'features': []
    }

    for f in top_features:
        json_data['features'].append({
            'feature_id': int(f.feature_id),
            'physical_id': f.physical_id,
            'label': f.label,
            'mean_abs_r': float(f.mean_abs_r),
            'std_r': float(f.std_r),
            'mean_r': float(f.mean_r),
            'detection_correlation': float(f.detection_correlation),
            'detector_score': float(f.detector_score),
            'logit_attribution': float(f.logit_attribution),
            'n_monotonic_pos': int(f.n_monotonic_pos),
            'n_monotonic_neg': int(f.n_monotonic_neg),
            'monotonic_frac': float(f.monotonic_frac),
            'direction_diversity': float(f.direction_diversity),
            'specialization_score': float(f.specialization_score),
            'monotonic_pos_concepts': f.monotonic_pos_concepts[:20],
            'monotonic_neg_concepts': f.monotonic_neg_concepts[:20],
        })

    with open(output_dir / "top_features_detail.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"    Saved: top_features_detail.json ({len(top_features)} features)")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_detector_analysis(
    steering_layer: int,
    token_mode: str,
    transcoder_l0: str,
    monotonicity_threshold: float,
    n_top_features: int,
    n_plot_features: int,
    plots_only: bool,
    debug_sample: int,
    ranking_metric: str = "detector_score",
) -> Path:
    """Run detector analysis for a single steering layer."""
    detector_layer = DETECTOR_LAYERS[steering_layer]

    print("\n" + "=" * 70)
    print(f"DETECTOR ANALYSIS: L{steering_layer} injection -> L{detector_layer} features")
    print("=" * 70)

    output_dir = OUTPUT_BASE / f"L{steering_layer}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load feature labels
    print("\n  Loading feature labels...")
    feature_labels = load_feature_labels(detector_layer, transcoder_l0)

    # Load concept detection rates
    print("\n  Loading concept detection rates...")
    concept_detection_rates = load_concept_detection_rates(steering_layer)

    # Load experiment 04b (vector geometry) success/failure partition
    print("\n  Loading experiment 04b (vector geometry) partition...")
    success_concepts, failure_concepts = load_geometry_partition(steering_layer)

    tracker = PlottedFeatureTracker(feature_labels)

    if plots_only:
        print("\n  [--plots-only] Loading computed results...")
        features = load_computed_results(output_dir)
        for f in features:
            f.label = feature_labels.get(f.feature_id, "")
    else:
        features = compute_feature_monotonicity(
            steering_layer=steering_layer,
            token_mode=token_mode,
            transcoder_l0=transcoder_l0,
            monotonicity_threshold=monotonicity_threshold,
            feature_labels=feature_labels,
            concept_detection_rates=concept_detection_rates,
            debug_sample=debug_sample,
        )
        save_computed_results(features, output_dir)

    # Populate direct logit attribution (paper §5.3 DLA formula) on every feature
    # so it appears in CSV/JSON alongside the correlation-based detector_score.
    try:
        compute_direct_logit_attribution(
            features=features,
            detector_layer=detector_layer,
            transcoder_l0=transcoder_l0,
        )
    except Exception as e:
        print(f"  WARNING: DLA computation failed, leaving logit_attribution=0: {e}")

    # Analyze distributions
    analyze_monotonicity_distribution(
        features=features,
        steering_layer=steering_layer,
        output_dir=output_dir,
        monotonicity_threshold=monotonicity_threshold,
        tracker=tracker,
    )

    analyze_detection_distribution(
        features=features,
        steering_layer=steering_layer,
        output_dir=output_dir,
        tracker=tracker,
    )

    # Find specialized detectors
    specialized, general, diverse, anomaly_detectors = find_specialized_detectors(
        features=features,
        n_top=n_top_features,
    )

    print(f"\n  Top anomaly detectors (by detector_score = mean_abs_r * detection_correlation):")
    for i, f in enumerate(anomaly_detectors[:5]):
        print(f"    {i+1}. {f.physical_id}: score={f.detector_score:.3f} "
              f"(|r|={f.mean_abs_r:.3f}, det_corr={f.detection_correlation:.3f})")

    # Plot top features (generates paper figures)
    plot_top_features(
        features=anomaly_detectors,
        steering_layer=steering_layer,
        token_mode=token_mode,
        transcoder_l0=transcoder_l0,
        output_dir=output_dir,
        category="anomaly_detector",
        n_plot=n_plot_features,
        tracker=tracker,
        concept_detection_rates=concept_detection_rates,
    )

    plot_top_features(
        features=specialized,
        steering_layer=steering_layer,
        token_mode=token_mode,
        transcoder_l0=transcoder_l0,
        output_dir=output_dir,
        category="specialized",
        n_plot=n_plot_features,
        tracker=tracker,
        concept_detection_rates=concept_detection_rates,
    )

    plot_top_features(
        features=diverse,
        steering_layer=steering_layer,
        token_mode=token_mode,
        transcoder_l0=transcoder_l0,
        output_dir=output_dir,
        category="direction_diverse",
        n_plot=n_plot_features,
        tracker=tracker,
        concept_detection_rates=concept_detection_rates,
    )

    # Concept coverage and feature direction analysis
    analyze_concept_coverage(
        features=features,
        steering_layer=steering_layer,
        output_dir=output_dir,
        n_top=n_top_features,
        tracker=tracker,
    )

    analyze_feature_directions(
        features=features,
        steering_layer=steering_layer,
        output_dir=output_dir,
        n_top=n_top_features,
        tracker=tracker,
    )

    # Save results
    save_results(
        features=features,
        steering_layer=steering_layer,
        output_dir=output_dir,
        ranking_metric=ranking_metric,
    )

    tracker.save_missing_labels(
        output_dir=output_dir,
        context=f"L{steering_layer}->L{detector_layer} detector analysis"
    )

    print(f"\n  Analysis complete for L{steering_layer}!")
    return output_dir


def main():
    print("=" * 70)
    print("EXPERIMENT 50: DETECTOR FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Token mode: {args.token_mode}")
    print(f"Transcoder L0: {args.transcoder_l0}")
    print(f"Monotonicity threshold: {args.monotonicity_threshold}")
    print(f"Steering layers: {args.steering_layers}")
    print(f"Top features to analyze: {args.n_top_features}")
    print(f"Features to plot: {args.n_plot_features}")
    if args.plots_only:
        print(f"Mode: PLOTS ONLY (loading saved data)")
    if args.debug_sample > 0:
        print(f"[DEBUG] Sampling {args.debug_sample} features")

    successful = []

    for steering_layer in args.steering_layers:
        if steering_layer not in DETECTOR_LAYERS:
            print(f"\nWARNING: No detector layer mapping for L{steering_layer}, skipping")
            continue

        try:
            output_dir = run_detector_analysis(
                steering_layer=steering_layer,
                token_mode=args.token_mode,
                transcoder_l0=args.transcoder_l0,
                monotonicity_threshold=args.monotonicity_threshold,
                n_top_features=args.n_top_features,
                n_plot_features=args.n_plot_features,
                plots_only=args.plots_only,
                debug_sample=args.debug_sample,
                ranking_metric=args.ranking_metric,
            )
            successful.append((steering_layer, output_dir))
        except Exception as e:
            print(f"\nERROR analyzing L{steering_layer}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully analyzed {len(successful)}/{len(args.steering_layers)} layers:")
    for layer, output_dir in successful:
        print(f"  L{layer} -> L{DETECTOR_LAYERS[layer]}: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
