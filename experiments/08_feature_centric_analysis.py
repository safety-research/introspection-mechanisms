#!/usr/bin/env python3
"""
Transcoder Feature Analysis: Feature-Centric Subset Analysis

For each top feature identified in the additional analysis, this script:
1. Finds the top-K concepts where this feature activates most strongly
2. Re-computes correlation between feature activation and detection rate on just those K concepts
3. Compares subset correlation vs global correlation

The hypothesis: Features may be strongly predictive for specific concept subsets,
but signal gets diluted when averaging across all 500 concepts.

Output:
    analysis/08_feature_centric_analysis/
    └── {config}/
        ├── feature_subset_correlations.csv       # Main results for all K values
        ├── top_concepts_per_feature.json         # Which concepts each feature fires on
        ├── correlation_improvement_bar.png       # Global vs subset correlation comparison
        ├── optimal_k_analysis.png                # How correlation changes with K
        ├── concept_cooccurrence_heatmap.png      # Which concepts cluster together
        └── summary.json                          # Overall findings
"""

import json
import re
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def _ensure_dense(x):
    """Convert sparse tensor to dense, or return as-is. Handles lists of tensors."""
    if isinstance(x, torch.Tensor) and x.is_sparse:
        return x.to_dense()
    if isinstance(x, list):
        return [_ensure_dense(t) for t in x]
    return x


def format_feature_id(physical_feature: str) -> str:
    """Format physical feature ID from 'L45_F2054' to 'L45F2054'."""
    return physical_feature.replace('_', '')


def wrap_label(physical_feature: str, label: str, max_width: int = 45) -> str:
    """Wrap feature label to multiple lines for better readability in plots.

    Format: L45F2054: <feature_title>
    """
    formatted_id = format_feature_id(physical_feature)
    if label:
        full_text = f"{formatted_id}: {label}"
    else:
        full_text = f"{formatted_id}: [NO LABEL]"

    # Wrap to max_width characters
    wrapped = textwrap.fill(full_text, width=max_width)
    return wrapped

# =============================================================================
# Configuration
# =============================================================================

CACHED_ACTIVATIONS_PATH = Path("analysis/08_cached_activations")
STEERING_RESULTS_PATH = Path("analysis/02b_steering_500_concepts/gemma3_27b")
ADDITIONAL_ANALYSIS_PATH = Path("analysis/08_additional_analysis")
OUTPUT_BASE = Path("analysis/08_feature_centric_analysis")
FEATURE_LABELS_PATH = Path("gemma-scope-2/feature_labels")

# K values to try for subset analysis (including very small values)
K_VALUES = [1, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200]

# Minimum K for "robust" correlation analysis
# K=3 gives statistically meaningless perfect correlations (only 6 possible rankings)
# K>=10 provides more statistical power and meaningful signal
MIN_K_ROBUST = 10

# Configs to analyze (layer, strength pairs)
CONFIGS = [
    (37, 4.0),
]

# Extended configs for multi-layer analysis (require regenerating cached activations)
ALL_CONFIGS = [
    (35, 2.0), (35, 4.0),
    (37, 2.0), (37, 4.0),
    (38, 2.0), (38, 4.0),
    (44, 2.0), (44, 4.0),
]

# Concept categories (parsed from 02b_concepts_list.py)
CONCEPT_CATEGORIES = {
    "ANIMALS": [
        "Elephants", "Dolphins", "Eagles", "Serpents", "Wolves", "Koalas", "Octopuses", "Penguins",
        "Crocodiles", "Hummingbirds", "Spiders", "Whales", "Foxes", "Owls", "Jellyfish", "Cheetahs",
        "Salamanders", "Peacocks", "Scorpions", "Gorillas", "Flamingos", "Beetles", "Seahorses", "Ravens",
        "Lobsters", "Chameleons", "Bats", "Kangaroos", "Starfish", "Hedgehogs", "Parrots", "Crabs",
        "Falcons", "Ants", "Squirrels", "Swans", "Turtles", "Bees", "Moose", "Otters", "Centipedes",
    ],
    "BODY_PARTS": [
        "Fingers", "Elbows", "Skulls", "Lungs", "Knees", "Spines", "Tongues", "Ankles",
        "Ribs", "Wrists", "Shoulders", "Teeth", "Eyebrows", "Hips", "Necks", "Palms",
        "Earlobes", "Nostrils", "Thumbs", "Foreheads", "Chins", "Heels",
    ],
    "COLORS": [
        "Crimson", "Turquoise", "Amber", "Violet", "Scarlet", "Indigo", "Maroon", "Magenta",
        "Cerulean", "Burgundy", "Lavender", "Teal",
    ],
    "ABSTRACT": [
        "Probability", "Entropy", "Paradoxes", "Symmetry", "Infinity", "Chaos", "Gravity", "Velocity",
        "Momentum", "Friction", "Consciousness", "Wisdom", "Causality", "Integrity", "Curiosity", "Ambition",
        "Duality", "Resilience", "Destiny", "Karma", "Irony", "Satire", "Metaphors", "Analogies",
        "Hypotheses", "Theories", "Proofs", "Axioms", "Dilemmas", "Compromise", "Sacrifice",
        "Redemption", "Transformation", "Evolution", "Revolution",
    ],
    "EMOTIONS": [
        "Anxiety", "Jealousy", "Gratitude", "Nostalgia", "Frustration", "Excitement", "Boredom",
        "Confusion", "Disgust", "Envy", "Pride", "Shame", "Guilt", "Loneliness", "Contentment",
        "Skepticism", "Wonder", "Awe", "Despair", "Hope", "Regret", "Anticipation", "Dread",
        "Serenity", "Melancholy",
    ],
    "FOODS": [
        "Chocolate", "Cinnamon", "Ginger", "Tofu", "Wasabi", "Mustard", "Pepper", "Garlic",
        "Onions", "Lemons", "Oranges", "Bananas", "Mangoes", "Grapes", "Cherries", "Peaches",
        "Almonds", "Walnuts", "Cashews", "Peanuts", "Coffee", "Tea", "Wine", "Whiskey",
        "Cheese", "Butter", "Eggs", "Caviar", "Sausages", "Dumplings", "Polenta", "Noodles",
        "Soups", "Pizzas", "Sushi",
    ],
    "MATERIALS": [
        "Copper", "Iron", "Steel", "Titanium", "Aluminum", "Granite", "Kevlar", "Limestone",
        "Concrete", "Glass", "Ceramic", "Porcelain", "Leather", "Silk", "Cotton", "Nylon",
        "Linen", "Velvet",
    ],
    "PROFESSIONS": [
        "Surgeons", "Architects", "Engineers", "Scientists", "Lawyers", "Pilots", "Chefs", "Farmers",
        "Teachers", "Nurses", "Firefighters", "Detectives", "Journalists", "Astronomers", "Archaeologists", "Philosophers",
        "Economists", "Therapists", "Electricians", "Plumbers", "Carpenters", "Tailors", "Librarians", "Curators",
        "Diplomats", "Translators", "Choreographers", "Sculptors",
    ],
    "ACTIONS": [
        "Swimming", "Dancing", "Climbing", "Running", "Jumping", "Diving", "Skating", "Skiing",
        "Surfing", "Fishing", "Hunting", "Gardening", "Cooking", "Painting", "Writing", "Reading",
        "Singing", "Whistling", "Laughing", "Crying", "Sleeping", "Dreaming", "Thinking", "Planning",
        "Building", "Creating", "Discovering", "Exploring",
    ],
    "SCIENCE": [
        "Photons", "Electrons", "Neutrons", "Protons", "Molecules", "Atoms", "Quarks", "Neurons",
        "Enzymes", "Proteins", "Chromosomes", "Genes", "Viruses", "Bacteria", "Fungi", "Ecosystems",
        "Mitochondria", "Ribosomes", "Comets", "Asteroids", "Isotopes", "Catalysts",
    ],
    "OBJECTS": [
        "Scissors", "Stethoscopes", "Screws", "Binoculars", "Threads", "Buttons", "Zippers",
        "Candles", "Lamps", "Clocks", "Keys", "Locks", "Chains", "Ropes", "Ladders",
        "Brushes", "Combs", "Towels", "Blankets", "Pillows", "Curtains", "Carpets", "Chandeliers",
        "Wardrobes", "Shelves", "Drawers", "Boxes", "Jars", "Bottles", "Cups", "Plates",
        "Forks", "Spoons", "Knives", "Pans",
    ],
    "BUILDINGS": [
        "Cathedrals", "Mosques", "Temples", "Pyramids", "Castles", "Cottages", "Lighthouses", "Windmills",
        "Factories", "Warehouses", "Barns", "Bridges", "Tunnels", "Dams", "Pagodas", "Monuments",
        "Stadiums", "Arenas", "Libraries", "Museums", "Hospitals", "Prisons",
    ],
    "VEHICLES": [
        "Rickshaws", "Motorcycles", "Automobiles", "Trains", "Submarines", "Helicopters", "Airplanes", "Rockets",
        "Sailboats", "Canoes", "Kayaks", "Ferries", "Tractors", "Ambulances", "Tanks", "Carriages",
        "Sleds", "Skateboards",
    ],
    "SPORTS": [
        "Basketball", "Football", "Baseball", "Tennis", "Golf", "Boxing", "Wrestling", "Archery",
        "Fencing", "Gymnastics", "Weightlifting", "Marathon", "Triathlon", "Hockey", "Cricket", "Rugby",
        "Volleyball", "Badminton",
    ],
    "MUSIC_ART": [
        "Violins", "Cellos", "Bagpipes", "Drums", "Flutes", "Harps", "Guitars", "Saxophones",
        "Clarinets", "Accordions", "Symphonies", "Operas", "Ballets", "Sculptures", "Mosaics", "Tapestries",
        "Frescoes", "Murals",
    ],
    "WEATHER": [
        "Thunderstorms", "Hurricanes", "Auroras", "Droughts", "Floods", "Hailstorms", "Rainbows", "Squalls",
        "Mist", "Dew", "Sleet", "Blizzards",
    ],
    "GEOGRAPHY": [
        "Marshes", "Lagoons", "Lakes", "Tundras", "Canyons", "Savannas", "Atolls", "Islands",
        "Peninsulas", "Archipelagos", "Fjords", "Deltas",
    ],
    "TIME": [
        "Centuries", "Decades", "Millennia", "Equinoxes", "Instants", "Eternities", "Eras", "Epochs",
        "Seasons", "Twilights", "Dawns", "Solstices",
    ],
    "SOCIAL": [
        "Weddings", "Funerals", "Ceremonies", "Rituals", "Traditions", "Customs", "Festivals", "Celebrations",
        "Protests", "Elections", "Debates", "Negotiations", "Treaties", "Alliances", "Conflicts", "Reconciliations",
        "Partnerships", "Rivalries", "Friendships", "Mentorships", "Apprenticeships", "Citizenship",
    ],
    "PLANTS": [
        "Roses", "Tulips", "Orchids", "Sunflowers", "Daisies", "Lilies", "Ferns", "Mosses",
        "Cacti", "Vines", "Shrubs", "Herbs", "Grasses", "Seaweed", "Mushrooms",
    ],
}

# Build reverse mapping: concept -> category
CONCEPT_TO_CATEGORY = {}
for category, concepts in CONCEPT_CATEGORIES.items():
    for concept in concepts:
        CONCEPT_TO_CATEGORY[concept] = category

# =============================================================================
# Argument Parser
# =============================================================================

parser = argparse.ArgumentParser(description="Feature-centric subset analysis")
parser.add_argument("-l", "--layer", type=int, default=35, help="Steering layer")
parser.add_argument("-s", "--strength", type=float, default=4.0, help="Steering strength")
parser.add_argument("-t", "--token-mode", type=str, default="last_token", help="Token mode")
parser.add_argument("-tc", "--transcoder-l0", type=str, default="big", help="Transcoder L0")
parser.add_argument("-n", "--n-features", type=int, default=100, help="Number of top features to analyze")
parser.add_argument("-m", "--metric", type=str, default="detection_rate", choices=["detection_rate", "forced_identification_rate"], help="Which metric to use")
parser.add_argument("--all-features", action="store_true", help="Analyze ALL features (top N by variance), not just sign-consistent from additional analysis")
parser.add_argument("--top-n-by-variance", type=int, default=5000, help="When using --all-features, how many top features by variance to analyze")
parser.add_argument("--category-analysis", action="store_true", help="Also run category-based analysis using semantic categories")
parser.add_argument("--all-configs", action="store_true", help="Run across all 6 configs (L35, L38, L44 × S2.0, S4.0)")
parser.add_argument("--pooled-analysis", action="store_true", help="Run pooled activation contrast analysis across all configs")

# =============================================================================
# Helper Functions
# =============================================================================

def parse_physical_feature(physical_feature: str) -> Tuple[int, int]:
    """Parse physical feature ID like 'L45_F123' into (layer, feature_id)."""
    match = re.match(r'L(\d+)_F(\d+)', physical_feature)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid physical_feature format: {physical_feature}")


def physical_to_index(physical_feature: str, layers: List[int], n_features: int = 262144) -> int:
    """Convert physical feature ID to global array index."""
    layer, feat_id = parse_physical_feature(physical_feature)
    if layer not in layers:
        return -1
    layer_idx = layers.index(layer)
    return layer_idx * n_features + feat_id


def load_feature_labels(transcoder_l0: str) -> Dict[int, Dict[int, str]]:
    """Load feature labels for all available transcoder layers."""
    labels = {}
    if FEATURE_LABELS_PATH.exists():
        for label_file in FEATURE_LABELS_PATH.glob(f"*_transcoder_*_layer*_262k_{transcoder_l0}_labels.json"):
            match = re.search(r'layer(\d+)_', label_file.name)
            if match:
                layer = int(match.group(1))
                try:
                    with open(label_file) as f:
                        data = json.load(f)
                    labels[layer] = {int(k): v.get('title', f'Feature {k}') for k, v in data.items()}
                except Exception as e:
                    print(f"    Warning: Could not load {label_file}: {e}")
    return labels


def get_feature_label(physical_feature: str, labels: Dict[int, Dict[int, str]]) -> str:
    """Get label for a physical feature."""
    try:
        layer, feat_id = parse_physical_feature(physical_feature)
        if layer in labels and feat_id in labels[layer]:
            return labels[layer][feat_id]
    except:
        pass
    return ""


# =============================================================================
# Data Loading
# =============================================================================

def load_cached_activations(
    steering_layer: int,
    steering_strength: float,
    token_mode: str,
    transcoder_l0: str,
) -> Tuple[Dict[str, np.ndarray], List[str], List[int]]:
    """
    Load cached activations and compute mean activation per concept per feature.

    Returns:
        concept_activations: Dict[concept_name] -> np.ndarray of shape (n_features_total,)
        concepts: List of concept names
        layers: List of transcoder layers
    """
    config_name = f"L{steering_layer}_S{steering_strength}_{token_mode}_{transcoder_l0}"
    cache_dir = CACHED_ACTIVATIONS_PATH / config_name

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cached activations not found: {cache_dir}")

    print(f"  Loading cached activations from {cache_dir}...")
    data = torch.load(cache_dir / "steered_activations.pt", weights_only=False)

    concepts = data['concepts']
    layers = data['layers']
    n_features = data['n_features']

    # Compute mean activation per concept per feature
    # Shape: (n_layers * n_features,) per concept
    concept_activations = {}

    for concept in concepts:
        concept_acts = []
        for layer in layers:
            # Shape: (n_trials, n_features)
            # Convert from bfloat16 to float32 for numpy compatibility
            layer_acts = _ensure_dense(data['activations'][concept][layer]).float().numpy()
            # Mean across trials
            mean_acts = layer_acts.mean(axis=0)
            concept_acts.append(mean_acts)
        # Concatenate all layers: shape (n_layers * n_features,)
        concept_activations[concept] = np.concatenate(concept_acts)

    print(f"    Loaded {len(concepts)} concepts, {len(layers)} layers, {n_features} features/layer")
    print(f"    Total features: {len(layers) * n_features}")

    return concept_activations, concepts, layers


def discover_cached_strengths(
    steering_layer: int,
    token_mode: str,
    transcoder_l0: str,
    base_model: bool = False,
    model_variant: Optional[str] = None,
    transcoder_width: str = "16k",
) -> Dict[float, Path]:
    """Discover cached activation files for different steering strengths."""
    strengths = {}
    prefix = f"L{steering_layer}_S"
    width_prefix = f"{transcoder_width}_" if transcoder_width != "16k" else ""
    suffix = f"_{token_mode}_{width_prefix}{transcoder_l0}"
    if model_variant:
        suffix += f"_{model_variant}"
    elif base_model:
        suffix += "_base_model"
    for cache_dir in CACHED_ACTIVATIONS_PATH.iterdir():
        if not cache_dir.is_dir():
            continue
        name = cache_dir.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        strength_str = parts[1][1:]
        try:
            strength = float(strength_str)
        except ValueError:
            continue
        cache_file = cache_dir / "steered_activations.pt"
        if cache_file.exists():
            strengths[strength] = cache_file
    return strengths


def get_layer_matrix_cached(
    cache_file: Path,
    layer: int,
    cached_data: Optional[dict] = None,
    activation_dtype: Optional[np.dtype] = None,
    cache_store: Optional[dict] = None,
    use_trial1: bool = False,
    use_fast_layer_matrices: bool = False,
    show_progress: bool = False,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """Load and return (concepts, matrix, concept_to_idx) for a single layer from cache."""
    cache_key = (str(cache_file), layer, bool(use_trial1))
    if cache_store is not None and cache_key in cache_store:
        return cache_store[cache_key]
    if use_fast_layer_matrices:
        matrix_dir = cache_file.parent / "layer_matrices"
        suffix = "trial1" if use_trial1 else "mean"
        matrix_file = matrix_dir / f"layer_{layer}_{suffix}.npz"
        if matrix_file.exists() and matrix_file.stat().st_mtime >= cache_file.stat().st_mtime:
            try:
                with np.load(matrix_file, allow_pickle=False) as data:
                    concepts = data["concepts"] if "concepts" in data else np.array([], dtype=str)
                    matrix = data["matrix"] if "matrix" in data else np.zeros((0, 0), dtype=np.float16)
            except Exception:
                concepts = None
                matrix = None
            if concepts is not None and matrix is not None:
                kept = [str(c) for c in concepts.tolist()] if concepts.size else []
                if activation_dtype is not None and matrix.dtype != activation_dtype:
                    matrix = matrix.astype(activation_dtype, copy=False)
                concept_to_idx = {c: i for i, c in enumerate(kept)}
                result = (kept, matrix.astype(np.float32, copy=False), concept_to_idx)
                if cache_store is not None:
                    cache_store[cache_key] = result
                return result
    data = cached_data if cached_data is not None else torch.load(cache_file, weights_only=False)
    activations = data["activations"]
    concepts = data.get("concepts", [])
    kept = []
    rows = []
    for concept in concepts:
        raw = activations.get(concept, {}).get(layer)
        if raw is None:
            continue
        kept.append(concept)
        layer_acts = _ensure_dense(raw).float().numpy()
        if use_trial1:
            rows.append(layer_acts[0] if layer_acts.ndim > 1 else layer_acts)
        else:
            rows.append(layer_acts.mean(axis=0) if layer_acts.ndim > 1 else layer_acts)
    if rows:
        matrix = np.stack(rows)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    if activation_dtype is not None and matrix.dtype != activation_dtype:
        matrix = matrix.astype(activation_dtype, copy=False)
    concept_to_idx = {c: i for i, c in enumerate(kept)}
    result = (kept, matrix, concept_to_idx)
    if cache_store is not None:
        cache_store[cache_key] = result
    return result


def get_control_cache_file(
    token_mode: str,
    transcoder_l0: str,
    base_model: bool = False,
    model_variant: Optional[str] = None,
    transcoder_width: str = "16k",
) -> Optional[Path]:
    """Get control (no-steering) cache file path."""
    width_prefix = f"{transcoder_width}_" if transcoder_width != "16k" else ""
    dir_name = f"control_{token_mode}_{width_prefix}{transcoder_l0}"
    if model_variant:
        dir_name += f"_{model_variant}"
    elif base_model:
        dir_name += "_base_model"
    control_dir = CACHED_ACTIVATIONS_PATH / dir_name
    control_file = control_dir / "control_activations.pt"
    return control_file if control_file.exists() else None


def load_detection_rates(
    steering_layer: int,
    steering_strength: float,
    metric_type: str = "detection_rate",
) -> Dict[str, float]:
    """
    Load per-concept detection rates from experiment 02 (steering evaluation) results.

    Returns:
        Dict mapping concept name -> detection rate (0-1)
    """
    steering_dir = STEERING_RESULTS_PATH / f"layer_{steering_layer}_strength_{steering_strength}"
    results_file = steering_dir / "results.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"02_steering_evaluation results not found: {results_file}")

    print(f"  Loading detection rates from {results_file}...")
    df = pd.read_csv(results_file)

    # Filter to steered trials with valid concept
    df = df[df['concept'].notna() & (df['concept'] != '') & (df['injected'] == True)]

    # Parse evaluations to get detection
    detection_rates = {}

    for concept in df['concept'].unique():
        concept_df = df[df['concept'] == concept]

        if metric_type == "detection_rate":
            # Use ONLY 'injection' trial_type for detection rate
            # (forced_injection inflates rates by forcing model to respond)
            injection_df = concept_df[concept_df['trial_type'] == 'injection']
            n_detected = 0
            n_total = 0
            for _, row in injection_df.iterrows():
                try:
                    evals = eval(row['evaluations'])
                    detected = evals.get('claims_detection', {}).get('claims_detection', False)
                    n_detected += int(detected)
                    n_total += 1
                except:
                    pass
            detection_rates[concept] = n_detected / n_total if n_total > 0 else 0.0
        else:
            # forced_identification_rate - use forced_injection trials
            forced_df = concept_df[concept_df['trial_type'] == 'forced_injection']
            n_correct = 0
            n_total = 0
            for _, row in forced_df.iterrows():
                try:
                    evals = eval(row['evaluations'])
                    correct = evals.get('correct_concept_identification', {}).get('correct_identification', False)
                    n_correct += int(correct)
                    n_total += 1
                except:
                    pass
            detection_rates[concept] = n_correct / n_total if n_total > 0 else 0.0

    print(f"    Loaded detection rates for {len(detection_rates)} concepts")
    print(f"    Mean detection rate: {np.mean(list(detection_rates.values())):.3f}")

    return detection_rates


def load_top_features(
    token_mode: str,
    transcoder_l0: str,
    correlation_method: str = "concept_level",
    n_features: int = 100,
) -> Tuple[List[str], List[str]]:
    """
    Load top sign-consistent features from additional analysis.

    Returns:
        (top_positive_features, top_negative_features)
    """
    analysis_dir = ADDITIONAL_ANALYSIS_PATH / f"{correlation_method}_{token_mode}_{transcoder_l0}"

    pos_file = analysis_dir / "consistent_features_positive.csv"
    neg_file = analysis_dir / "consistent_features_negative.csv"

    if not pos_file.exists() or not neg_file.exists():
        raise FileNotFoundError(f"Additional analysis results not found in {analysis_dir}")

    pos_df = pd.read_csv(pos_file)
    neg_df = pd.read_csv(neg_file)

    top_positive = pos_df['physical_feature'].head(n_features).tolist()
    top_negative = neg_df['physical_feature'].head(n_features).tolist()

    print(f"  Loaded {len(top_positive)} positive and {len(top_negative)} negative features")

    return top_positive, top_negative


def get_top_features_by_variance(
    concept_activations: Dict[str, np.ndarray],
    detection_rates: Dict[str, float],
    layers: List[int],
    n_features: int = 5000,
    n_features_per_layer: int = 262144,
) -> Tuple[List[str], List[str]]:
    """
    Get top features by variance across concepts AND by correlation strength.

    Returns:
        (top_positive_features, top_negative_features)
        - Positive: features with positive global correlation with detection
        - Negative: features with negative global correlation with detection
    """
    print(f"  Finding top {n_features} features by variance and correlation...")

    # Get common concepts
    concepts_list = sorted(set(concept_activations.keys()) & set(detection_rates.keys()))
    n_concepts = len(concepts_list)

    # Build activation matrix: (n_concepts, n_total_features)
    n_total_features = len(layers) * n_features_per_layer
    activation_matrix = np.zeros((n_concepts, n_total_features))

    for i, concept in enumerate(concepts_list):
        activation_matrix[i] = concept_activations[concept]

    # Compute variance across concepts for each feature
    feature_variances = np.var(activation_matrix, axis=0)

    # Compute detection rates array
    rates = np.array([detection_rates[c] for c in concepts_list])

    # Find features with non-zero variance
    nonzero_mask = feature_variances > 0
    nonzero_indices = np.where(nonzero_mask)[0]

    print(f"    Features with non-zero variance: {len(nonzero_indices)}/{n_total_features}")

    # Compute correlations for all non-zero features
    correlations = []
    for idx in nonzero_indices:
        feat_acts = activation_matrix[:, idx]
        if np.std(feat_acts) > 0:
            corr, _ = stats.spearmanr(feat_acts, rates)
            if not np.isnan(corr):
                correlations.append((idx, corr, feature_variances[idx]))

    print(f"    Features with valid correlations: {len(correlations)}")

    # Separate positive and negative correlations
    positive_features = [(idx, corr, var) for idx, corr, var in correlations if corr > 0]
    negative_features = [(idx, corr, var) for idx, corr, var in correlations if corr < 0]

    # Sort by |correlation| * variance (to prioritize high-signal features)
    positive_features.sort(key=lambda x: abs(x[1]) * x[2], reverse=True)
    negative_features.sort(key=lambda x: abs(x[1]) * x[2], reverse=True)

    # Convert indices to physical feature IDs
    def idx_to_physical(idx):
        layer_idx = idx // n_features_per_layer
        feat_idx = idx % n_features_per_layer
        layer = layers[layer_idx]
        return f"L{layer}_F{feat_idx}"

    top_positive = [idx_to_physical(idx) for idx, _, _ in positive_features[:n_features]]
    top_negative = [idx_to_physical(idx) for idx, _, _ in negative_features[:n_features]]

    print(f"    Selected {len(top_positive)} positive and {len(top_negative)} negative features")

    # Print some stats about top features
    if positive_features:
        print(f"    Top positive correlation: {positive_features[0][1]:.4f}")
    if negative_features:
        print(f"    Top negative correlation: {negative_features[0][1]:.4f}")

    return top_positive, top_negative


def compute_category_correlation(
    physical_feature: str,
    concept_activations: Dict[str, np.ndarray],
    detection_rates: Dict[str, float],
    layers: List[int],
    category_concepts: List[str],
) -> Dict:
    """
    Compute correlation for a feature on concepts from a specific category.

    Returns dict with: corr, pval, n_concepts, mean_activation, mean_detection
    """
    idx = physical_to_index(physical_feature, layers)
    if idx < 0:
        return {'corr': 0.0, 'pval': 1.0, 'n_concepts': 0, 'mean_activation': 0.0, 'mean_detection': 0.0}

    # Get concepts that are both in this category and in our data
    valid_concepts = [c for c in category_concepts
                      if c in concept_activations and c in detection_rates]

    if len(valid_concepts) < 5:  # Need minimum concepts for meaningful correlation
        return {'corr': 0.0, 'pval': 1.0, 'n_concepts': len(valid_concepts),
                'mean_activation': 0.0, 'mean_detection': 0.0}

    activations = np.array([concept_activations[c][idx] for c in valid_concepts])
    rates = np.array([detection_rates[c] for c in valid_concepts])

    # Handle constant arrays
    if np.std(activations) == 0 or np.std(rates) == 0:
        corr, pval = 0.0, 1.0
    else:
        corr, pval = stats.spearmanr(activations, rates)

    return {
        'corr': float(corr) if not np.isnan(corr) else 0.0,
        'pval': float(pval) if not np.isnan(pval) else 1.0,
        'n_concepts': len(valid_concepts),
        'mean_activation': float(activations.mean()),
        'mean_detection': float(rates.mean()),
        'concepts': valid_concepts,
    }


def analyze_feature_with_categories(
    physical_feature: str,
    concept_activations: Dict[str, np.ndarray],
    detection_rates: Dict[str, float],
    layers: List[int],
    labels: Dict[int, Dict[int, str]],
) -> Dict:
    """
    Analyze a feature's correlation across all semantic categories.

    Returns dict with category -> correlation results
    """
    # Global correlation first
    global_corr, global_pval = compute_global_correlation(
        physical_feature, concept_activations, detection_rates, layers
    )

    category_results = {}
    for category, concepts in CONCEPT_CATEGORIES.items():
        result = compute_category_correlation(
            physical_feature, concept_activations, detection_rates, layers, concepts
        )
        category_results[category] = result

    # Find best category (highest absolute correlation)
    best_category = max(category_results.keys(),
                        key=lambda c: abs(category_results[c]['corr']))

    return {
        'physical_feature': physical_feature,
        'label': get_feature_label(physical_feature, labels),
        'global_corr': global_corr,
        'global_pval': global_pval,
        'best_category': best_category,
        'best_category_corr': category_results[best_category]['corr'],
        'category_results': category_results,
        'improvement': abs(category_results[best_category]['corr']) - abs(global_corr),
    }


# =============================================================================
# Feature-Centric Analysis
# =============================================================================

@dataclass
class FeatureSubsetResult:
    """Results for one feature's subset analysis."""
    physical_feature: str
    label: str
    direction: str  # "positive" or "negative"
    global_corr: float
    global_pval: float
    subset_results: Dict[int, Dict]  # K -> {corr, pval, concepts, ...}
    best_k: int
    best_subset_corr: float
    improvement: float  # best_subset_corr - global_corr
    # Robust metrics (K >= MIN_K_ROBUST only)
    robust_k: int  # Best K among K >= MIN_K_ROBUST
    robust_corr: float  # Correlation at robust_k
    robust_improvement: float  # abs(robust_corr) - abs(global_corr)
    corr_at_k10: float  # Correlation specifically at K=10
    corr_at_k15: float  # Correlation specifically at K=15
    corr_at_k25: float  # Correlation specifically at K=25


def compute_global_correlation(
    physical_feature: str,
    concept_activations: Dict[str, np.ndarray],
    detection_rates: Dict[str, float],
    layers: List[int],
) -> Tuple[float, float]:
    """Compute correlation across all concepts."""
    idx = physical_to_index(physical_feature, layers)
    if idx < 0:
        return 0.0, 1.0

    # Get activations and detection rates for all concepts
    concepts = sorted(set(concept_activations.keys()) & set(detection_rates.keys()))

    activations = np.array([concept_activations[c][idx] for c in concepts])
    rates = np.array([detection_rates[c] for c in concepts])

    # Handle constant arrays
    if np.std(activations) == 0 or np.std(rates) == 0:
        return 0.0, 1.0

    corr, pval = stats.spearmanr(activations, rates)
    return corr, pval


def compute_subset_correlation(
    physical_feature: str,
    concept_activations: Dict[str, np.ndarray],
    detection_rates: Dict[str, float],
    layers: List[int],
    k: int,
) -> Dict:
    """
    Compute correlation on top-K concepts by activation.

    Returns dict with: corr, pval, concepts, mean_activation, mean_detection
    """
    idx = physical_to_index(physical_feature, layers)
    if idx < 0:
        return {'corr': 0.0, 'pval': 1.0, 'concepts': [], 'mean_activation': 0.0, 'mean_detection': 0.0}

    # Get all concepts with both activations and detection rates
    concepts = sorted(set(concept_activations.keys()) & set(detection_rates.keys()))

    # Get activations for this feature across all concepts
    activations = np.array([concept_activations[c][idx] for c in concepts])
    rates = np.array([detection_rates[c] for c in concepts])

    # Find top-K concepts by activation
    top_k_indices = np.argsort(activations)[-k:]
    top_k_concepts = [concepts[i] for i in top_k_indices]

    subset_activations = activations[top_k_indices]
    subset_rates = rates[top_k_indices]

    # Handle constant arrays
    if np.std(subset_activations) == 0 or np.std(subset_rates) == 0:
        corr, pval = 0.0, 1.0
    else:
        corr, pval = stats.spearmanr(subset_activations, subset_rates)

    return {
        'corr': corr,
        'pval': pval,
        'concepts': top_k_concepts,
        'mean_activation': float(subset_activations.mean()),
        'mean_detection': float(subset_rates.mean()),
        'min_activation': float(subset_activations.min()),
        'max_activation': float(subset_activations.max()),
    }


def analyze_feature(
    physical_feature: str,
    concept_activations: Dict[str, np.ndarray],
    detection_rates: Dict[str, float],
    layers: List[int],
    direction: str,
    labels: Dict[int, Dict[int, str]],
    k_values: List[int] = K_VALUES,
) -> FeatureSubsetResult:
    """Run full subset analysis for one feature."""

    # Global correlation
    global_corr, global_pval = compute_global_correlation(
        physical_feature, concept_activations, detection_rates, layers
    )

    # Subset correlations for each K
    subset_results = {}
    for k in k_values:
        subset_results[k] = compute_subset_correlation(
            physical_feature, concept_activations, detection_rates, layers, k
        )

    # Find best K (highest absolute correlation in the expected direction)
    if direction == "positive":
        best_k = max(k_values, key=lambda k: subset_results[k]['corr'])
        best_subset_corr = subset_results[best_k]['corr']
    else:
        best_k = min(k_values, key=lambda k: subset_results[k]['corr'])
        best_subset_corr = subset_results[best_k]['corr']

    improvement = abs(best_subset_corr) - abs(global_corr)

    # Compute ROBUST metrics (K >= MIN_K_ROBUST only)
    # This avoids the K=3 statistical artifact
    robust_k_values = [k for k in k_values if k >= MIN_K_ROBUST]

    if direction == "positive":
        robust_k = max(robust_k_values, key=lambda k: subset_results[k]['corr'])
        robust_corr = subset_results[robust_k]['corr']
    else:
        robust_k = min(robust_k_values, key=lambda k: subset_results[k]['corr'])
        robust_corr = subset_results[robust_k]['corr']

    robust_improvement = abs(robust_corr) - abs(global_corr)

    # Get correlations at specific K values for comparison
    corr_at_k10 = subset_results.get(10, {}).get('corr', 0.0)
    corr_at_k15 = subset_results.get(15, {}).get('corr', 0.0)
    corr_at_k25 = subset_results.get(25, {}).get('corr', 0.0)

    return FeatureSubsetResult(
        physical_feature=physical_feature,
        label=get_feature_label(physical_feature, labels),
        direction=direction,
        global_corr=global_corr,
        global_pval=global_pval,
        subset_results=subset_results,
        best_k=best_k,
        best_subset_corr=best_subset_corr,
        improvement=improvement,
        robust_k=robust_k,
        robust_corr=robust_corr,
        robust_improvement=robust_improvement,
        corr_at_k10=corr_at_k10,
        corr_at_k15=corr_at_k15,
        corr_at_k25=corr_at_k25,
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_correlation_improvement(
    results: List[FeatureSubsetResult],
    output_dir: Path,
    direction: str,
    n_features: int = 30,
):
    """Plot global vs best subset correlation for top features."""

    # Sort by improvement
    sorted_results = sorted(results, key=lambda r: r.improvement, reverse=True)[:n_features]

    # Calculate figure height based on number of features (more height for wrapped labels)
    fig_height = max(16, n_features * 0.7)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    y_pos = np.arange(len(sorted_results))

    # Global correlations
    global_corrs = [r.global_corr for r in sorted_results]
    subset_corrs = [r.best_subset_corr for r in sorted_results]

    # Plot bars
    width = 0.35
    bars1 = ax.barh(y_pos - width/2, global_corrs, width, label='Global (all 500 concepts)', color='steelblue', alpha=0.7)
    bars2 = ax.barh(y_pos + width/2, subset_corrs, width, label='Best subset', color='darkorange', alpha=0.7)

    # Labels - full feature title with wrapping, include K value
    labels = []
    for r in sorted_results:
        base_label = wrap_label(r.physical_feature, r.label, max_width=50)
        # Add K value on same or new line
        lbl = f"{base_label}\n(K={r.best_k})"
        labels.append(lbl)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, linespacing=0.9)
    ax.invert_yaxis()

    ax.set_xlabel('Spearman Correlation', fontsize=14)
    ax.set_title(f'Global vs Subset Correlation: Top {n_features} {direction.upper()} Features\n(Sorted by improvement)',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"correlation_improvement_{direction}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_robust_correlation_improvement(
    results: List[FeatureSubsetResult],
    output_dir: Path,
    direction: str,
    n_features: int = 30,
):
    """Plot global vs ROBUST subset correlation (K>=10) for top features.

    This version uses robust_improvement (based on K>=10) instead of the
    potentially misleading K=3 results.
    """

    # Sort by ROBUST improvement (K>=10 only)
    sorted_results = sorted(results, key=lambda r: r.robust_improvement, reverse=True)[:n_features]

    # Calculate figure height based on number of features (more height for wrapped labels)
    fig_height = max(16, n_features * 0.7)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    y_pos = np.arange(len(sorted_results))

    # Global correlations and ROBUST correlations
    global_corrs = [r.global_corr for r in sorted_results]
    robust_corrs = [r.robust_corr for r in sorted_results]

    # Plot bars
    width = 0.35
    bars1 = ax.barh(y_pos - width/2, global_corrs, width, label='Global (all 500 concepts)', color='steelblue', alpha=0.7)
    bars2 = ax.barh(y_pos + width/2, robust_corrs, width, label=f'Best robust subset (K>={MIN_K_ROBUST})', color='darkgreen', alpha=0.7)

    # Labels - full feature title with wrapping, include K value
    labels = []
    for r in sorted_results:
        base_label = wrap_label(r.physical_feature, r.label, max_width=50)
        # Add robust K value
        lbl = f"{base_label}\n(K={r.robust_k})"
        labels.append(lbl)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, linespacing=0.9)
    ax.invert_yaxis()

    ax.set_xlabel('Spearman Correlation', fontsize=14)
    ax.set_title(f'Global vs ROBUST Subset Correlation (K>={MIN_K_ROBUST}): Top {n_features} {direction.upper()} Features\n(Sorted by robust improvement - avoids K=3 statistical artifact)',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"robust_correlation_improvement_{direction}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimal_k_analysis(
    results: List[FeatureSubsetResult],
    output_dir: Path,
    direction: str,
    n_features: int = 10,
):
    """Plot how correlation changes with K for top features."""

    # Sort by improvement and take top N
    sorted_results = sorted(results, key=lambda r: r.improvement, reverse=True)[:n_features]

    fig, ax = plt.subplots(figsize=(16, 10))

    k_values = sorted(sorted_results[0].subset_results.keys())

    for i, r in enumerate(sorted_results):
        corrs = [r.subset_results[k]['corr'] for k in k_values]
        # Full label with formatted feature ID (L45F2054 format)
        formatted_id = format_feature_id(r.physical_feature)
        if r.label:
            label = f"{formatted_id}: {r.label}"
        else:
            label = f"{formatted_id}: [NO LABEL]"
        ax.plot(k_values, corrs, marker='o', label=label, linewidth=2, markersize=6)

        # Add horizontal line for global correlation
        ax.axhline(y=r.global_corr, color=f'C{i}', linestyle='--', alpha=0.3)

    ax.set_xlabel('K (number of top concepts)', fontsize=14)
    ax.set_ylabel('Spearman Correlation', fontsize=14)
    ax.set_title(f'Correlation vs Subset Size: Top {n_features} {direction.upper()} Features\n(Dashed lines = global correlation)',
                 fontsize=16, fontweight='bold')
    # Place legend outside the plot for better readability with full labels
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, borderaxespad=0)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to actual K values (not linear interpolation)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    # Use log scale for x-axis since K values span wide range
    ax.set_xscale('log')
    ax.set_xticks(k_values)  # Re-set after log scale
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    ax.minorticks_off()  # Disable minor ticks for cleaner look

    plt.tight_layout()
    plt.savefig(output_dir / f"optimal_k_analysis_{direction}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_robust_optimal_k_analysis(
    results: List[FeatureSubsetResult],
    output_dir: Path,
    direction: str,
    n_features: int = 10,
):
    """Plot how correlation changes with K for top ROBUST features (sorted by K>=10 improvement).

    This version sorts by robust_improvement and highlights the K>=10 region.
    """

    # Sort by ROBUST improvement and take top N
    sorted_results = sorted(results, key=lambda r: r.robust_improvement, reverse=True)[:n_features]

    fig, ax = plt.subplots(figsize=(16, 10))

    k_values = sorted(sorted_results[0].subset_results.keys())

    for i, r in enumerate(sorted_results):
        corrs = [r.subset_results[k]['corr'] for k in k_values]
        # Full label with formatted feature ID (L45F2054 format)
        formatted_id = format_feature_id(r.physical_feature)
        if r.label:
            label = f"{formatted_id}: {r.label}"
        else:
            label = f"{formatted_id}: [NO LABEL]"
        ax.plot(k_values, corrs, marker='o', label=label, linewidth=2, markersize=6)

        # Add horizontal line for global correlation
        ax.axhline(y=r.global_corr, color=f'C{i}', linestyle='--', alpha=0.3)

    # Shade the "robust" region (K>=10)
    ax.axvspan(MIN_K_ROBUST, max(k_values) * 1.1, alpha=0.1, color='green', label=f'Robust region (K>={MIN_K_ROBUST})')

    # Add vertical line at MIN_K_ROBUST
    ax.axvline(x=MIN_K_ROBUST, color='green', linestyle=':', linewidth=2, alpha=0.7)

    ax.set_xlabel('K (number of top concepts)', fontsize=14)
    ax.set_ylabel('Spearman Correlation', fontsize=14)
    ax.set_title(f'Correlation vs Subset Size: Top {n_features} ROBUST {direction.upper()} Features\n(Sorted by K>={MIN_K_ROBUST} improvement; green region = statistically robust)',
                 fontsize=16, fontweight='bold')
    # Place legend outside the plot for better readability with full labels
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, borderaxespad=0)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to actual K values (not linear interpolation)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    # Use log scale for x-axis since K values span wide range
    ax.set_xscale('log')
    ax.set_xticks(k_values)  # Re-set after log scale
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    ax.minorticks_off()  # Disable minor ticks for cleaner look

    plt.tight_layout()
    plt.savefig(output_dir / f"robust_optimal_k_analysis_{direction}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_concept_cooccurrence(
    results: List[FeatureSubsetResult],
    output_dir: Path,
    direction: str,
    k: int = 50,
    n_features: int = 20,
):
    """Plot which concepts appear together in top-K for multiple features."""

    # Sort by improvement and take top N
    sorted_results = sorted(results, key=lambda r: r.improvement, reverse=True)[:n_features]

    # Count concept occurrences
    concept_counts = defaultdict(int)
    for r in sorted_results:
        if k in r.subset_results:
            for c in r.subset_results[k]['concepts']:
                concept_counts[c] += 1

    # Get top concepts that appear in multiple features
    top_concepts = sorted(concept_counts.keys(), key=lambda c: concept_counts[c], reverse=True)[:30]

    # Build co-occurrence matrix with full labels (wrapped)
    feature_labels = []
    for r in sorted_results:
        lbl = wrap_label(r.physical_feature, r.label, max_width=50)
        feature_labels.append(lbl)

    matrix = np.zeros((len(sorted_results), len(top_concepts)))

    for i, r in enumerate(sorted_results):
        if k in r.subset_results:
            for j, c in enumerate(top_concepts):
                if c in r.subset_results[k]['concepts']:
                    matrix[i, j] = 1

    # Plot heatmap - larger figure to accommodate full labels
    fig_height = max(14, n_features * 0.8)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    sns.heatmap(matrix, xticklabels=top_concepts, yticklabels=feature_labels,
                cmap='Blues', cbar_kws={'label': 'In top-K'}, ax=ax)

    ax.set_xlabel('Concept', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title(f'Concept Co-occurrence in Top-{k}: {direction.upper()} Features\n(Which concepts cluster together?)',
                 fontsize=16, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f"concept_cooccurrence_{direction}_K{k}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return concept_counts


def plot_category_heatmap(
    category_results: Dict[str, Dict],
    output_dir: Path,
    n_features: int = 50,
):
    """Plot heatmap of correlations by feature x category."""

    # Get top features by improvement
    sorted_features = sorted(
        category_results.items(),
        key=lambda x: x[1]['improvement'],
        reverse=True
    )[:n_features]

    categories = list(CONCEPT_CATEGORIES.keys())

    # Build feature labels with full titles (wrapped)
    feature_labels = []
    for feat, result in sorted_features:
        label = result.get('label', '')
        lbl = wrap_label(feat, label, max_width=50)
        feature_labels.append(lbl)

    # Build correlation matrix
    matrix = np.zeros((len(sorted_features), len(categories)))
    for i, (feat, result) in enumerate(sorted_features):
        for j, cat in enumerate(categories):
            matrix[i, j] = result['category_results'].get(cat, {}).get('corr', 0)

    # Plot - larger figure for full labels
    fig_height = max(18, n_features * 0.5)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    # Use diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    sns.heatmap(matrix, xticklabels=categories, yticklabels=feature_labels,
                cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={'label': 'Spearman Correlation'}, ax=ax)

    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title(f'Feature Correlations by Semantic Category\n(Top {n_features} features by improvement)',
                 fontsize=16, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "category_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also plot per-category best features with full labels
    # Sort categories by sample size (largest first) and filter small ones
    cat_sizes = {cat: len([c for c in concepts if c in CONCEPT_TO_CATEGORY and CONCEPT_TO_CATEGORY[c] == cat])
                 for cat in categories for concepts in [list(category_results.values())[0]['category_results'].get(cat, {}).get('concepts', [])]}

    # Get actual N from the results
    sample_cat = list(category_results.values())[0]
    cat_n_values = {cat: sample_cat['category_results'].get(cat, {}).get('n_concepts', 0) for cat in categories}

    # Sort by N (descending) and mark small categories
    sorted_cats = sorted(categories, key=lambda c: cat_n_values.get(c, 0), reverse=True)

    fig, axes = plt.subplots(4, 5, figsize=(32, 26))
    axes = axes.flatten()

    for idx, cat in enumerate(sorted_cats):
        if idx >= len(axes):
            break
        ax = axes[idx]

        n_concepts = cat_n_values.get(cat, 0)
        is_small = n_concepts < 20

        # Get correlations for this category
        cat_corrs = [(f, result['category_results'].get(cat, {}).get('corr', 0), result.get('label', ''))
                     for f, result in category_results.items()]

        # Sort by absolute correlation
        cat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_10 = cat_corrs[:10]

        # Build full labels with wrapping (uses L45F2054 format via wrap_label)
        feat_labels = []
        for f, c, lbl in top_10:
            # Use shorter wrap width for subplots
            wrapped = wrap_label(f, lbl, max_width=40)
            feat_labels.append(wrapped)
        corrs = [c for _, c, _ in top_10]

        colors = ['green' if c > 0 else 'red' for c in corrs]
        ax.barh(range(len(top_10)), corrs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(feat_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Correlation', fontsize=10)

        # Title includes N and warning for small categories
        title_color = 'orange' if is_small else 'black'
        warning = ' ⚠️' if is_small else ''
        ax.set_title(f'{cat} (N={n_concepts}){warning}', fontsize=12, fontweight='bold', color=title_color)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

        # Add light background for small categories
        if is_small:
            ax.set_facecolor('#fff3e0')

    # Hide empty subplots
    for idx in range(len(sorted_cats), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Top 10 Features by Category\n(Sorted by sample size; orange = N<20, interpret with caution)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "category_top_features.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def run_feature_centric_analysis(
    steering_layer: int,
    steering_strength: float,
    token_mode: str,
    transcoder_l0: str,
    n_features: int,
    metric_type: str,
    all_features: bool = False,
    top_n_by_variance: int = 5000,
    category_analysis: bool = False,
):
    """Run the full feature-centric analysis."""

    config_name = f"L{steering_layer}_S{steering_strength}_{token_mode}_{transcoder_l0}"
    output_dir = OUTPUT_BASE / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "ALL features by variance" if all_features else "top sign-consistent from additional analysis"

    print("=" * 80)
    print("EXPERIMENT 50: Feature-Centric Subset Analysis")
    print("=" * 80)
    print(f"Config: {config_name}")
    print(f"Metric: {metric_type}")
    print(f"Mode: {mode_str}")
    print(f"N features: {top_n_by_variance if all_features else n_features}")
    print(f"K values: {K_VALUES}")
    print(f"Category analysis: {category_analysis}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("[1/6] Loading data...")
    concept_activations, concepts, layers = load_cached_activations(
        steering_layer, steering_strength, token_mode, transcoder_l0
    )

    detection_rates = load_detection_rates(steering_layer, steering_strength, metric_type)

    # Choose feature selection method
    if all_features:
        print(f"\n[1b/6] Selecting top {top_n_by_variance} features by variance and correlation...")
        top_positive, top_negative = get_top_features_by_variance(
            concept_activations, detection_rates, layers, n_features=top_n_by_variance
        )
    else:
        top_positive, top_negative = load_top_features(token_mode, transcoder_l0, n_features=n_features)

    labels = load_feature_labels(transcoder_l0)
    print(f"  Loaded {sum(len(v) for v in labels.values())} feature labels")

    # Analyze positive features
    print(f"\n[2/6] Analyzing {len(top_positive)} positive features...")
    positive_results = []
    log_interval = max(1, len(top_positive) // 20)
    for i, feat in enumerate(top_positive):
        if i % log_interval == 0:
            print(f"    Processing {i+1}/{len(top_positive)}...")
        result = analyze_feature(
            feat, concept_activations, detection_rates, layers, "positive", labels
        )
        positive_results.append(result)

    # Analyze negative features
    print(f"\n[3/6] Analyzing {len(top_negative)} negative features...")
    negative_results = []
    log_interval = max(1, len(top_negative) // 20)
    for i, feat in enumerate(top_negative):
        if i % log_interval == 0:
            print(f"    Processing {i+1}/{len(top_negative)}...")
        result = analyze_feature(
            feat, concept_activations, detection_rates, layers, "negative", labels
        )
        negative_results.append(result)

    # Category-based analysis
    category_results = {}
    if category_analysis:
        print(f"\n[4/6] Running category-based analysis...")
        all_features_to_analyze = list(set(top_positive[:min(500, len(top_positive))] +
                                            top_negative[:min(500, len(top_negative))]))
        print(f"    Analyzing {len(all_features_to_analyze)} features across {len(CONCEPT_CATEGORIES)} categories...")

        for i, feat in enumerate(all_features_to_analyze):
            if i % 100 == 0:
                print(f"    Processing {i+1}/{len(all_features_to_analyze)}...")
            result = analyze_feature_with_categories(
                feat, concept_activations, detection_rates, layers, labels
            )
            category_results[feat] = result

        # Save category analysis results
        category_output_dir = output_dir / "category_analysis"
        category_output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed category results
        category_rows = []
        for feat, result in category_results.items():
            row = {
                'physical_feature': feat,
                'label': result['label'],
                'global_corr': result['global_corr'],
                'best_category': result['best_category'],
                'best_category_corr': result['best_category_corr'],
                'improvement': result['improvement'],
            }
            # Add per-category correlations
            for cat, cat_result in result['category_results'].items():
                row[f'{cat}_corr'] = cat_result['corr']
                row[f'{cat}_n'] = cat_result['n_concepts']
            category_rows.append(row)

        category_df = pd.DataFrame(category_rows)
        category_df = category_df.sort_values('improvement', ascending=False)
        category_df.to_csv(category_output_dir / "category_correlations.csv", index=False)

        # Find features that work best for each category
        category_best_features = {}
        for cat in CONCEPT_CATEGORIES.keys():
            cat_col = f'{cat}_corr'
            if cat_col in category_df.columns:
                # Best positive
                pos_df = category_df[category_df[cat_col] > 0].nlargest(10, cat_col)
                # Best negative
                neg_df = category_df[category_df[cat_col] < 0].nsmallest(10, cat_col)
                category_best_features[cat] = {
                    'best_positive': pos_df[['physical_feature', 'label', cat_col, 'global_corr']].to_dict('records'),
                    'best_negative': neg_df[['physical_feature', 'label', cat_col, 'global_corr']].to_dict('records'),
                }

        with open(category_output_dir / "best_features_per_category.json", 'w') as f:
            json.dump(category_best_features, f, indent=2)

        # Category analysis summary
        category_summary = {
            'n_features_analyzed': len(all_features_to_analyze),
            'n_categories': len(CONCEPT_CATEGORIES),
            'categories': list(CONCEPT_CATEGORIES.keys()),
            'top_improved_by_category': sorted(
                [(feat, r['best_category'], r['improvement'])
                 for feat, r in category_results.items()],
                key=lambda x: x[2], reverse=True
            )[:20],
        }

        with open(category_output_dir / "summary.json", 'w') as f:
            json.dump(category_summary, f, indent=2)

        # Plot category heatmap for top features
        plot_category_heatmap(category_results, category_output_dir)

        print(f"    Category analysis saved to {category_output_dir}")

    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    # Original plots (sorted by best K, including K=3)
    plot_correlation_improvement(positive_results, output_dir, "positive")
    plot_correlation_improvement(negative_results, output_dir, "negative")

    plot_optimal_k_analysis(positive_results, output_dir, "positive")
    plot_optimal_k_analysis(negative_results, output_dir, "negative")

    # NEW: Robust plots (sorted by K>=10 improvement only)
    plot_robust_correlation_improvement(positive_results, output_dir, "positive")
    plot_robust_correlation_improvement(negative_results, output_dir, "negative")

    plot_robust_optimal_k_analysis(positive_results, output_dir, "positive")
    plot_robust_optimal_k_analysis(negative_results, output_dir, "negative")

    pos_concept_counts = plot_concept_cooccurrence(positive_results, output_dir, "positive", k=50)
    neg_concept_counts = plot_concept_cooccurrence(negative_results, output_dir, "negative", k=50)

    # Save results
    print("\n[6/6] Saving results...")

    # Main results CSV
    rows = []
    for r in positive_results + negative_results:
        row = {
            'physical_feature': r.physical_feature,
            'label': r.label,
            'direction': r.direction,
            'global_corr': r.global_corr,
            'global_pval': r.global_pval,
            'best_k': r.best_k,
            'best_subset_corr': r.best_subset_corr,
            'improvement': r.improvement,
            'pct_improvement': r.improvement / abs(r.global_corr) * 100 if r.global_corr != 0 else 0,
            # ROBUST metrics (K>=10 only)
            'robust_k': r.robust_k,
            'robust_corr': r.robust_corr,
            'robust_improvement': r.robust_improvement,
            'robust_pct_improvement': r.robust_improvement / abs(r.global_corr) * 100 if r.global_corr != 0 else 0,
            'corr_at_k10': r.corr_at_k10,
            'corr_at_k15': r.corr_at_k15,
            'corr_at_k25': r.corr_at_k25,
        }
        # Add correlations for each K
        for k in K_VALUES:
            row[f'corr_K{k}'] = r.subset_results[k]['corr']
            row[f'pval_K{k}'] = r.subset_results[k]['pval']
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "feature_subset_correlations.csv", index=False)

    # Top concepts per feature
    top_concepts_data = {}
    for r in positive_results + negative_results:
        top_concepts_data[r.physical_feature] = {
            'direction': r.direction,
            'label': r.label,
            'global_corr': r.global_corr,
            'best_k': r.best_k,
            'best_subset_corr': r.best_subset_corr,
            'top_concepts_K50': r.subset_results[50]['concepts'] if 50 in r.subset_results else [],
        }

    with open(output_dir / "top_concepts_per_feature.json", 'w') as f:
        json.dump(top_concepts_data, f, indent=2)

    # Track and save missing feature labels ONLY for features shown in plots
    # Features shown in plots:
    # - correlation_improvement: top 30 by improvement (pos + neg)
    # - robust_correlation_improvement: top 30 by robust_improvement (pos + neg)
    # - optimal_k_analysis: top 10 by improvement (pos + neg)
    # - robust_optimal_k_analysis: top 10 by robust_improvement (pos + neg)
    # - concept_cooccurrence: top 20 by improvement (pos + neg)
    # - category_correlation_heatmap: top 50 by improvement (from category analysis)
    # - category_top_features: top 10 per category by absolute correlation (20 categories)

    # Get features shown in main plots (top 30 covers all since it's the largest)
    n_shown_in_plots = 30
    # Original plots sort by improvement (any K)
    top_positive_shown = sorted(positive_results, key=lambda r: r.improvement, reverse=True)[:n_shown_in_plots]
    top_negative_shown = sorted(negative_results, key=lambda r: r.improvement, reverse=True)[:n_shown_in_plots]
    # Robust plots sort by robust_improvement (K>=10 only)
    top_positive_robust = sorted(positive_results, key=lambda r: r.robust_improvement, reverse=True)[:n_shown_in_plots]
    top_negative_robust = sorted(negative_results, key=lambda r: r.robust_improvement, reverse=True)[:n_shown_in_plots]

    features_in_plots = set()
    for r in top_positive_shown + top_negative_shown + top_positive_robust + top_negative_robust:
        features_in_plots.add(r.physical_feature)

    # Also add features from category analysis if it was run
    if category_results:
        # category_correlation_heatmap.png: top 50 by improvement
        top_category_shown = sorted(
            category_results.items(),
            key=lambda x: x[1]['improvement'],
            reverse=True
        )[:50]
        for feat, _ in top_category_shown:
            features_in_plots.add(feat)

        # category_top_features.png: top 10 per category by ABSOLUTE CORRELATION
        # This plot shows different features than the heatmap!
        for cat in CONCEPT_CATEGORIES.keys():
            cat_corrs = [(f, result['category_results'].get(cat, {}).get('corr', 0))
                         for f, result in category_results.items()]
            cat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_10_for_cat = cat_corrs[:10]
            for feat, _ in top_10_for_cat:
                features_in_plots.add(feat)

    # Now find missing labels only among features shown in plots
    missing_labels = []
    for r in positive_results + negative_results:
        if r.physical_feature in features_in_plots and not r.label:
            layer, feat_id = parse_physical_feature(r.physical_feature)
            missing_labels.append({
                'physical_feature': r.physical_feature,
                'layer': layer,
                'feature_id': feat_id,
                'direction': r.direction,
                'global_corr': r.global_corr,
                'best_subset_corr': r.best_subset_corr,
                'improvement': r.improvement,
            })

    # Sort by improvement to prioritize important unlabeled features
    missing_labels.sort(key=lambda x: x['improvement'], reverse=True)

    # ALWAYS write this file (even if empty) so it reflects current state
    with open(output_dir / "missing_feature_labels.json", 'w') as f:
        json.dump({
            'n_missing': len(missing_labels),
            'n_shown_in_plots': len(features_in_plots),
            'pct_missing': len(missing_labels) / len(features_in_plots) * 100 if features_in_plots else 0,
            'features': missing_labels,
        }, f, indent=2)

    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} features with missing labels (out of {len(features_in_plots)} shown in plots)")
    else:
        print(f"  All {len(features_in_plots)} features shown in plots have labels")

    # Summary
    pos_improvements = [r.improvement for r in positive_results]
    neg_improvements = [r.improvement for r in negative_results]
    pos_robust_improvements = [r.robust_improvement for r in positive_results]
    neg_robust_improvements = [r.robust_improvement for r in negative_results]

    summary = {
        'config': config_name,
        'metric': metric_type,
        'n_concepts': len(concepts),
        'n_positive_features': len(positive_results),
        'n_negative_features': len(negative_results),
        'k_values': K_VALUES,
        'min_k_robust': MIN_K_ROBUST,
        'positive_analysis': {
            'mean_global_corr': float(np.mean([r.global_corr for r in positive_results])),
            'mean_best_subset_corr': float(np.mean([r.best_subset_corr for r in positive_results])),
            'mean_improvement': float(np.mean(pos_improvements)),
            'max_improvement': float(np.max(pos_improvements)),
            'n_improved': sum(1 for i in pos_improvements if i > 0),
            'top_improved_features': [
                {'feature': r.physical_feature, 'label': r.label, 'global': r.global_corr,
                 'subset': r.best_subset_corr, 'improvement': r.improvement, 'best_k': r.best_k}
                for r in sorted(positive_results, key=lambda x: x.improvement, reverse=True)[:10]
            ],
        },
        'negative_analysis': {
            'mean_global_corr': float(np.mean([r.global_corr for r in negative_results])),
            'mean_best_subset_corr': float(np.mean([r.best_subset_corr for r in negative_results])),
            'mean_improvement': float(np.mean(neg_improvements)),
            'max_improvement': float(np.max(neg_improvements)),
            'n_improved': sum(1 for i in neg_improvements if i > 0),
            'top_improved_features': [
                {'feature': r.physical_feature, 'label': r.label, 'global': r.global_corr,
                 'subset': r.best_subset_corr, 'improvement': r.improvement, 'best_k': r.best_k}
                for r in sorted(negative_results, key=lambda x: x.improvement, reverse=True)[:10]
            ],
        },
        # NEW: Robust analysis (K>=10 only - avoids K=3 statistical artifact)
        'robust_positive_analysis': {
            'mean_robust_corr': float(np.mean([r.robust_corr for r in positive_results])),
            'mean_robust_improvement': float(np.mean(pos_robust_improvements)),
            'max_robust_improvement': float(np.max(pos_robust_improvements)),
            'n_robust_improved': sum(1 for i in pos_robust_improvements if i > 0),
            'mean_corr_at_k10': float(np.mean([r.corr_at_k10 for r in positive_results])),
            'mean_corr_at_k15': float(np.mean([r.corr_at_k15 for r in positive_results])),
            'mean_corr_at_k25': float(np.mean([r.corr_at_k25 for r in positive_results])),
            'top_robust_features': [
                {'feature': r.physical_feature, 'label': r.label, 'global': r.global_corr,
                 'robust_corr': r.robust_corr, 'robust_improvement': r.robust_improvement,
                 'robust_k': r.robust_k, 'corr_at_k10': r.corr_at_k10, 'corr_at_k15': r.corr_at_k15}
                for r in sorted(positive_results, key=lambda x: x.robust_improvement, reverse=True)[:10]
            ],
        },
        'robust_negative_analysis': {
            'mean_robust_corr': float(np.mean([r.robust_corr for r in negative_results])),
            'mean_robust_improvement': float(np.mean(neg_robust_improvements)),
            'max_robust_improvement': float(np.max(neg_robust_improvements)),
            'n_robust_improved': sum(1 for i in neg_robust_improvements if i > 0),
            'mean_corr_at_k10': float(np.mean([r.corr_at_k10 for r in negative_results])),
            'mean_corr_at_k15': float(np.mean([r.corr_at_k15 for r in negative_results])),
            'mean_corr_at_k25': float(np.mean([r.corr_at_k25 for r in negative_results])),
            'top_robust_features': [
                {'feature': r.physical_feature, 'label': r.label, 'global': r.global_corr,
                 'robust_corr': r.robust_corr, 'robust_improvement': r.robust_improvement,
                 'robust_k': r.robust_k, 'corr_at_k10': r.corr_at_k10, 'corr_at_k15': r.corr_at_k15}
                for r in sorted(negative_results, key=lambda x: x.robust_improvement, reverse=True)[:10]
            ],
        },
        'concept_clustering': {
            'positive_top_concepts': dict(sorted(pos_concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'negative_top_concepts': dict(sorted(neg_concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
        },
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\nPOSITIVE FEATURES (any K):")
    print(f"  Mean global correlation: {summary['positive_analysis']['mean_global_corr']:.3f}")
    print(f"  Mean best subset correlation: {summary['positive_analysis']['mean_best_subset_corr']:.3f}")
    print(f"  Mean improvement: {summary['positive_analysis']['mean_improvement']:.3f}")
    print(f"  Features improved: {summary['positive_analysis']['n_improved']}/{len(positive_results)}")

    print(f"\nNEGATIVE FEATURES (any K):")
    print(f"  Mean global correlation: {summary['negative_analysis']['mean_global_corr']:.3f}")
    print(f"  Mean best subset correlation: {summary['negative_analysis']['mean_best_subset_corr']:.3f}")
    print(f"  Mean improvement: {summary['negative_analysis']['mean_improvement']:.3f}")
    print(f"  Features improved: {summary['negative_analysis']['n_improved']}/{len(negative_results)}")

    print(f"\n" + "=" * 80)
    print(f"ROBUST ANALYSIS (K>={MIN_K_ROBUST} only - avoids K=3 statistical artifact)")
    print("=" * 80)

    print(f"\nROBUST POSITIVE FEATURES:")
    print(f"  Mean robust correlation: {summary['robust_positive_analysis']['mean_robust_corr']:.3f}")
    print(f"  Mean robust improvement: {summary['robust_positive_analysis']['mean_robust_improvement']:.3f}")
    print(f"  Mean corr at K=10: {summary['robust_positive_analysis']['mean_corr_at_k10']:.3f}")
    print(f"  Mean corr at K=15: {summary['robust_positive_analysis']['mean_corr_at_k15']:.3f}")
    print(f"  Mean corr at K=25: {summary['robust_positive_analysis']['mean_corr_at_k25']:.3f}")
    print(f"  Features with robust improvement: {summary['robust_positive_analysis']['n_robust_improved']}/{len(positive_results)}")

    print(f"\nROBUST NEGATIVE FEATURES:")
    print(f"  Mean robust correlation: {summary['robust_negative_analysis']['mean_robust_corr']:.3f}")
    print(f"  Mean robust improvement: {summary['robust_negative_analysis']['mean_robust_improvement']:.3f}")
    print(f"  Mean corr at K=10: {summary['robust_negative_analysis']['mean_corr_at_k10']:.3f}")
    print(f"  Mean corr at K=15: {summary['robust_negative_analysis']['mean_corr_at_k15']:.3f}")
    print(f"  Mean corr at K=25: {summary['robust_negative_analysis']['mean_corr_at_k25']:.3f}")
    print(f"  Features with robust improvement: {summary['robust_negative_analysis']['n_robust_improved']}/{len(negative_results)}")

    print(f"\nTop ROBUST POSITIVE features (K>={MIN_K_ROBUST}):")
    for item in summary['robust_positive_analysis']['top_robust_features'][:5]:
        print(f"  {item['feature']}: global={item['global']:.3f} -> robust={item['robust_corr']:.3f} (+{item['robust_improvement']:.3f}) K={item['robust_k']}")

    print(f"\nTop ROBUST NEGATIVE features (K>={MIN_K_ROBUST}):")
    for item in summary['robust_negative_analysis']['top_robust_features'][:5]:
        print(f"  {item['feature']}: global={item['global']:.3f} -> robust={item['robust_corr']:.3f} (+{item['robust_improvement']:.3f}) K={item['robust_k']}")

    print(f"\nOutput files saved to: {output_dir}/")

    return summary


# =============================================================================
# Pooled Activation Contrast Analysis
# =============================================================================

@dataclass
class ContrastResult:
    """Result of activation contrast analysis for a single feature."""
    physical_feature: str
    layer: int
    feature_id: int
    label: str
    n_observations: int  # Total number of (concept, config) pairs

    # Median split contrast
    median_split_high_mean: float  # Mean detection when activation > median
    median_split_low_mean: float   # Mean detection when activation <= median
    median_split_contrast: float   # High - Low
    median_split_cohens_d: float   # Standardized effect size
    median_split_pvalue: float     # Mann-Whitney U p-value

    # Quartile analysis
    q1_mean: float  # Mean detection in lowest quartile
    q2_mean: float
    q3_mean: float
    q4_mean: float  # Mean detection in highest quartile
    quartile_trend: str  # "increasing", "decreasing", "non-monotonic"

    # Extreme contrast (top 25% vs bottom 25%)
    extreme_high_mean: float
    extreme_low_mean: float
    extreme_contrast: float

    # Global correlation (across all pooled observations)
    global_correlation: float
    global_pvalue: float


def load_pooled_data(
    token_mode: str,
    transcoder_l0: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]], List[int]]:
    """
    Load and pool data from all 6 configs.

    Returns:
        pooled_activations: Dict[config_name] -> Dict[concept] -> np.ndarray of activations
        pooled_detection_rates: Dict[config_name] -> Dict[concept] -> float
        common_layers: List of layers present in ALL configs
    """
    print("Loading data from all configs...")

    all_activations = {}
    all_detection_rates = {}
    all_layers = []

    for layer, strength in CONFIGS:
        config_name = f"L{layer}_S{strength}"
        print(f"  Loading {config_name}...")

        # Load activations
        concept_activations, concepts, layers = load_cached_activations(
            layer, strength, token_mode, transcoder_l0
        )
        all_activations[config_name] = concept_activations
        all_layers.append(set(layers))

        # Load detection rates
        detection_rates = load_detection_rates(layer, strength, "detection_rate")
        all_detection_rates[config_name] = detection_rates

    # Find common layers (intersection of all layer sets)
    common_layers = sorted(set.intersection(*all_layers))
    print(f"\nCommon layers across all configs: {common_layers[0]} to {common_layers[-1]} ({len(common_layers)} layers)")

    return all_activations, all_detection_rates, common_layers


def compute_feature_contrast(
    physical_feature: str,
    pooled_activations: Dict[str, Dict[str, np.ndarray]],
    pooled_detection_rates: Dict[str, Dict[str, float]],
    all_layers_per_config: Dict[str, List[int]],
    labels: Dict[int, Dict[int, str]],
) -> Optional[ContrastResult]:
    """
    Compute activation contrast metrics for a single feature across all pooled data.
    """
    layer, feat_id = parse_physical_feature(physical_feature)

    # Collect all (activation, detection_rate) pairs across configs
    activations = []
    detection_rates = []

    for config_name, concept_acts in pooled_activations.items():
        config_layers = all_layers_per_config[config_name]
        if layer not in config_layers:
            continue  # Skip configs that don't have this layer

        layer_idx = config_layers.index(layer)
        n_features = 262144  # Features per layer (262k transcoder width)

        for concept, acts in concept_acts.items():
            # Extract activation for this specific feature
            feat_activation = acts[layer_idx * n_features + feat_id]
            det_rate = pooled_detection_rates[config_name].get(concept, np.nan)

            if not np.isnan(det_rate):
                activations.append(feat_activation)
                detection_rates.append(det_rate)

    activations = np.array(activations)
    detection_rates = np.array(detection_rates)

    if len(activations) < 20:  # Need enough data points
        return None

    # Get label
    label = ""
    if layer in labels and feat_id in labels[layer]:
        label = labels[layer][feat_id]

    # Median split contrast
    median_act = np.median(activations)
    high_mask = activations > median_act
    low_mask = activations <= median_act

    high_det = detection_rates[high_mask]
    low_det = detection_rates[low_mask]

    if len(high_det) < 5 or len(low_det) < 5:
        return None

    high_mean = float(np.mean(high_det))
    low_mean = float(np.mean(low_det))
    contrast = high_mean - low_mean

    # Cohen's d
    pooled_std = np.sqrt((np.var(high_det) + np.var(low_det)) / 2)
    cohens_d = contrast / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U test
    from scipy.stats import mannwhitneyu, spearmanr
    try:
        _, pvalue = mannwhitneyu(high_det, low_det, alternative='two-sided')
    except:
        pvalue = 1.0

    # Quartile analysis
    quartiles = np.percentile(activations, [25, 50, 75])
    q1_mask = activations <= quartiles[0]
    q2_mask = (activations > quartiles[0]) & (activations <= quartiles[1])
    q3_mask = (activations > quartiles[1]) & (activations <= quartiles[2])
    q4_mask = activations > quartiles[2]

    q1_mean = float(np.mean(detection_rates[q1_mask])) if q1_mask.sum() > 0 else np.nan
    q2_mean = float(np.mean(detection_rates[q2_mask])) if q2_mask.sum() > 0 else np.nan
    q3_mean = float(np.mean(detection_rates[q3_mask])) if q3_mask.sum() > 0 else np.nan
    q4_mean = float(np.mean(detection_rates[q4_mask])) if q4_mask.sum() > 0 else np.nan

    # Determine trend
    quartile_means = [q1_mean, q2_mean, q3_mean, q4_mean]
    if all(quartile_means[i] <= quartile_means[i+1] for i in range(3)):
        trend = "increasing"
    elif all(quartile_means[i] >= quartile_means[i+1] for i in range(3)):
        trend = "decreasing"
    else:
        trend = "non-monotonic"

    # Extreme contrast (top 25% vs bottom 25%)
    extreme_high_mean = q4_mean
    extreme_low_mean = q1_mean
    extreme_contrast = extreme_high_mean - extreme_low_mean

    # Global correlation
    try:
        global_corr, global_pval = spearmanr(activations, detection_rates)
    except:
        global_corr, global_pval = 0.0, 1.0

    return ContrastResult(
        physical_feature=physical_feature,
        layer=layer,
        feature_id=feat_id,
        label=label,
        n_observations=len(activations),
        median_split_high_mean=high_mean,
        median_split_low_mean=low_mean,
        median_split_contrast=contrast,
        median_split_cohens_d=float(cohens_d),
        median_split_pvalue=float(pvalue),
        q1_mean=q1_mean,
        q2_mean=q2_mean,
        q3_mean=q3_mean,
        q4_mean=q4_mean,
        quartile_trend=trend,
        extreme_high_mean=extreme_high_mean,
        extreme_low_mean=extreme_low_mean,
        extreme_contrast=extreme_contrast,
        global_correlation=float(global_corr),
        global_pvalue=float(global_pval),
    )


def plot_pooled_contrast_bar(
    results: List[ContrastResult],
    output_dir: Path,
    direction: str,  # "positive" or "negative"
    n_features: int = 30,
):
    """Plot top features by contrast (positive = firing increases detection, negative = decreases)."""
    if direction == "positive":
        sorted_results = sorted(results, key=lambda r: r.median_split_contrast, reverse=True)[:n_features]
        title_suffix = "Higher Activation → Higher Detection"
        color = 'forestgreen'
    else:
        sorted_results = sorted(results, key=lambda r: r.median_split_contrast)[:n_features]
        title_suffix = "Higher Activation → Lower Detection"
        color = 'crimson'

    if not sorted_results:
        return

    fig_height = max(10, n_features * 0.4)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    labels = [wrap_label(r.physical_feature, r.label, max_width=55) for r in sorted_results]
    contrasts = [r.median_split_contrast for r in sorted_results]
    cohens_ds = [r.median_split_cohens_d for r in sorted_results]

    y_pos = np.arange(len(sorted_results))
    bars = ax.barh(y_pos, contrasts, color=color, alpha=0.7)

    # Add Cohen's d annotations
    for i, (bar, d) in enumerate(zip(bars, cohens_ds)):
        width = bar.get_width()
        x_pos = width + 0.01 if width >= 0 else width - 0.01
        ha = 'left' if width >= 0 else 'right'
        ax.annotate(f'd={d:.2f}', (x_pos, bar.get_y() + bar.get_height()/2),
                   ha=ha, va='center', fontsize=8, color='gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Contrast (High Activation - Low Activation Detection Rate)', fontsize=12)
    ax.set_title(f'Top {n_features} Features: {title_suffix}\n(Pooled across all configs, N≈3000 observations per feature)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"contrast_{direction}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pooled_quartile_analysis(
    results: List[ContrastResult],
    output_dir: Path,
    direction: str,
    n_features: int = 10,
):
    """Plot quartile breakdown for top features."""
    if direction == "positive":
        sorted_results = sorted(results, key=lambda r: r.median_split_contrast, reverse=True)[:n_features]
        title_suffix = "Positive Contrast"
    else:
        sorted_results = sorted(results, key=lambda r: r.median_split_contrast)[:n_features]
        title_suffix = "Negative Contrast"

    if not sorted_results:
        return

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()

    for idx, result in enumerate(sorted_results):
        if idx >= len(axes):
            break
        ax = axes[idx]

        quartile_means = [result.q1_mean, result.q2_mean, result.q3_mean, result.q4_mean]
        quartile_labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']

        colors = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
        bars = ax.bar(quartile_labels, quartile_means, color=colors, alpha=0.7)

        # Add value labels
        for bar, val in zip(bars, quartile_means):
            ax.annotate(f'{val:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

        short_label = result.label[:40] + "..." if len(result.label) > 40 else result.label
        ax.set_title(f'{format_feature_id(result.physical_feature)}\n{short_label}',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('Detection Rate', fontsize=9)
        ax.set_ylim(0, 1)

        # Add trend indicator
        trend_symbol = "↗" if result.quartile_trend == "increasing" else "↘" if result.quartile_trend == "decreasing" else "~"
        ax.annotate(trend_symbol, (0.95, 0.95), xycoords='axes fraction',
                   fontsize=16, ha='right', va='top',
                   color='green' if result.quartile_trend == "increasing" else 'red' if result.quartile_trend == "decreasing" else 'gray')

    # Hide empty subplots
    for idx in range(len(sorted_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Quartile Analysis: Top {n_features} Features with {title_suffix}\n(Detection rate by activation quartile)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"quartile_analysis_{direction}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pooled_scatter(
    results: List[ContrastResult],
    pooled_activations: Dict[str, Dict[str, np.ndarray]],
    pooled_detection_rates: Dict[str, Dict[str, float]],
    all_layers_per_config: Dict[str, List[int]],
    output_dir: Path,
    n_features: int = 6,
):
    """Plot scatter plots of activation vs detection for top contrast features."""
    # Get top positive and negative contrast features
    top_positive = sorted(results, key=lambda r: r.median_split_contrast, reverse=True)[:n_features//2]
    top_negative = sorted(results, key=lambda r: r.median_split_contrast)[:n_features//2]
    selected = top_positive + top_negative

    if not selected:
        return

    n_cols = 3
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()

    for idx, result in enumerate(selected):
        if idx >= len(axes):
            break
        ax = axes[idx]

        layer, feat_id = parse_physical_feature(result.physical_feature)

        # Collect data for this feature
        activations = []
        detection_rates = []
        configs = []

        for config_name, concept_acts in pooled_activations.items():
            config_layers = all_layers_per_config[config_name]
            if layer not in config_layers:
                continue

            layer_idx = config_layers.index(layer)
            n_features_per_layer = 262144

            for concept, acts in concept_acts.items():
                feat_activation = acts[layer_idx * n_features_per_layer + feat_id]
                det_rate = pooled_detection_rates[config_name].get(concept, np.nan)

                if not np.isnan(det_rate):
                    activations.append(feat_activation)
                    detection_rates.append(det_rate)
                    configs.append(config_name)

        activations = np.array(activations)
        detection_rates = np.array(detection_rates)

        # Color by config
        config_colors = {
            'L35_S2.0': '#1f77b4', 'L35_S4.0': '#aec7e8',
            'L38_S2.0': '#ff7f0e', 'L38_S4.0': '#ffbb78',
            'L44_S2.0': '#2ca02c', 'L44_S4.0': '#98df8a',
        }
        colors = [config_colors.get(c, 'gray') for c in configs]

        ax.scatter(activations, detection_rates, c=colors, alpha=0.5, s=20)

        # Add regression line
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(activations, detection_rates)
        x_line = np.array([activations.min(), activations.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'r={result.global_correlation:.3f}')

        # Add median line
        median_act = np.median(activations)
        ax.axvline(x=median_act, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        short_label = result.label[:35] + "..." if len(result.label) > 35 else result.label
        ax.set_title(f'{format_feature_id(result.physical_feature)}: {short_label}\nContrast={result.median_split_contrast:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature Activation', fontsize=9)
        ax.set_ylabel('Detection Rate', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

    # Hide empty subplots
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Activation vs Detection Rate Scatter Plots\n(Top positive and negative contrast features)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plots.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pooled_effect_size_distribution(
    results: List[ContrastResult],
    output_dir: Path,
):
    """Plot distribution of effect sizes across all features."""
    contrasts = [r.median_split_contrast for r in results]
    cohens_ds = [r.median_split_cohens_d for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Contrast distribution
    ax1 = axes[0]
    ax1.hist(contrasts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(contrasts), color='green', linestyle='-', linewidth=2, label=f'Mean={np.mean(contrasts):.3f}')
    ax1.set_xlabel('Contrast (High - Low Detection Rate)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Activation Contrasts', fontsize=14, fontweight='bold')
    ax1.legend()

    # Cohen's d distribution
    ax2 = axes[1]
    ax2.hist(cohens_ds, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=0.2, color='gray', linestyle=':', linewidth=1, label='Small (d=0.2)')
    ax2.axvline(x=-0.2, color='gray', linestyle=':', linewidth=1)
    ax2.axvline(x=0.5, color='gray', linestyle='-.', linewidth=1, label='Medium (d=0.5)')
    ax2.axvline(x=-0.5, color='gray', linestyle='-.', linewidth=1)
    ax2.set_xlabel("Cohen's d (Standardized Effect Size)", fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title("Distribution of Effect Sizes", fontsize=14, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "effect_size_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_pooled_analysis(
    token_mode: str,
    transcoder_l0: str,
    n_top_features: int = 5000,
):
    """Run pooled activation contrast analysis across all configs."""
    output_dir = OUTPUT_BASE / f"pooled_analysis_{token_mode}_{transcoder_l0}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("POOLED ACTIVATION CONTRAST ANALYSIS")
    print("=" * 80)
    print(f"Configs: {CONFIGS}")
    print(f"Token mode: {token_mode}")
    print(f"Transcoder L0: {transcoder_l0}")
    print(f"Output: {output_dir}")
    print()

    # Load data from all configs
    print("[1/5] Loading and pooling data...")
    pooled_activations, pooled_detection_rates, common_layers = load_pooled_data(
        token_mode, transcoder_l0
    )

    # Get layers per config for proper indexing
    all_layers_per_config = {}
    for layer, strength in CONFIGS:
        config_name = f"L{layer}_S{strength}"
        cache_dir = CACHED_ACTIVATIONS_PATH / f"{config_name}_{token_mode}_{transcoder_l0}"
        data = torch.load(cache_dir / "steered_activations.pt", weights_only=False)
        all_layers_per_config[config_name] = data['layers']

    # Load feature labels
    labels = load_feature_labels(transcoder_l0)
    print(f"  Loaded {sum(len(v) for v in labels.values())} feature labels")

    # Get all unique physical features to analyze
    # Use features from all layers that exist in at least one config
    all_features_to_analyze = set()

    # Option 1: Use sign-consistent features from additional analysis
    try:
        top_positive, top_negative = load_top_features(token_mode, transcoder_l0, n_features=500)
        for feat in top_positive + top_negative:
            all_features_to_analyze.add(feat)
        print(f"  Loaded {len(all_features_to_analyze)} sign-consistent features from additional analysis")
    except Exception as e:
        print(f"  Warning: Could not load sign-consistent features: {e}")

    # Option 2: Add high-variance features from common layers
    print(f"\n[2/5] Selecting top features by variance from common layers...")
    n_features_per_layer = 262144

    # For each common layer, compute variance across all pooled data
    for layer in common_layers:
        # Collect activations for this layer across all configs
        layer_activations = []

        for config_name, concept_acts in pooled_activations.items():
            config_layers = all_layers_per_config[config_name]
            if layer not in config_layers:
                continue

            layer_idx = config_layers.index(layer)
            start_idx = layer_idx * n_features_per_layer
            end_idx = start_idx + n_features_per_layer

            for concept, acts in concept_acts.items():
                layer_activations.append(acts[start_idx:end_idx])

        if not layer_activations:
            continue

        layer_activations = np.stack(layer_activations)  # Shape: (n_obs, n_features_per_layer)
        layer_variances = np.var(layer_activations, axis=0)

        # Get top 100 features by variance for this layer
        top_feat_indices = np.argsort(layer_variances)[-100:]
        for feat_idx in top_feat_indices:
            all_features_to_analyze.add(f"L{layer}_F{feat_idx}")

    print(f"  Total unique features to analyze: {len(all_features_to_analyze)}")

    # Compute contrast for each feature
    print(f"\n[3/5] Computing contrast metrics for {len(all_features_to_analyze)} features...")
    results = []
    log_interval = max(1, len(all_features_to_analyze) // 20)

    for i, physical_feature in enumerate(sorted(all_features_to_analyze)):
        if i % log_interval == 0:
            print(f"    Processing {i+1}/{len(all_features_to_analyze)}...")

        result = compute_feature_contrast(
            physical_feature,
            pooled_activations,
            pooled_detection_rates,
            all_layers_per_config,
            labels,
        )
        if result is not None:
            results.append(result)

    print(f"  Successfully analyzed {len(results)} features")

    # Generate visualizations
    print(f"\n[4/5] Generating visualizations...")
    plot_pooled_contrast_bar(results, output_dir, "positive", n_features=30)
    plot_pooled_contrast_bar(results, output_dir, "negative", n_features=30)
    plot_pooled_quartile_analysis(results, output_dir, "positive", n_features=10)
    plot_pooled_quartile_analysis(results, output_dir, "negative", n_features=10)
    plot_pooled_scatter(results, pooled_activations, pooled_detection_rates,
                        all_layers_per_config, output_dir, n_features=6)
    plot_pooled_effect_size_distribution(results, output_dir)

    # Save results
    print(f"\n[5/5] Saving results...")

    # Save CSV
    csv_data = []
    for r in results:
        csv_data.append({
            'physical_feature': r.physical_feature,
            'layer': r.layer,
            'feature_id': r.feature_id,
            'label': r.label,
            'n_observations': r.n_observations,
            'median_split_contrast': r.median_split_contrast,
            'median_split_cohens_d': r.median_split_cohens_d,
            'median_split_pvalue': r.median_split_pvalue,
            'median_split_high_mean': r.median_split_high_mean,
            'median_split_low_mean': r.median_split_low_mean,
            'q1_mean': r.q1_mean,
            'q2_mean': r.q2_mean,
            'q3_mean': r.q3_mean,
            'q4_mean': r.q4_mean,
            'quartile_trend': r.quartile_trend,
            'extreme_contrast': r.extreme_contrast,
            'global_correlation': r.global_correlation,
            'global_pvalue': r.global_pvalue,
        })
    df = pd.DataFrame(csv_data)
    df.to_csv(output_dir / "contrast_results.csv", index=False)

    # Track features in plots for missing labels
    n_shown_in_plots = 30
    top_positive = sorted(results, key=lambda r: r.median_split_contrast, reverse=True)[:n_shown_in_plots]
    top_negative = sorted(results, key=lambda r: r.median_split_contrast)[:n_shown_in_plots]

    features_in_plots = set()
    for r in top_positive + top_negative:
        features_in_plots.add(r.physical_feature)

    # Also add features from quartile analysis plots (top 10 each direction)
    for r in sorted(results, key=lambda r: r.median_split_contrast, reverse=True)[:10]:
        features_in_plots.add(r.physical_feature)
    for r in sorted(results, key=lambda r: r.median_split_contrast)[:10]:
        features_in_plots.add(r.physical_feature)

    # Find missing labels
    missing_labels = []
    for r in results:
        if r.physical_feature in features_in_plots and not r.label:
            missing_labels.append({
                'physical_feature': r.physical_feature,
                'layer': r.layer,
                'feature_id': r.feature_id,
                'median_split_contrast': r.median_split_contrast,
                'cohens_d': r.median_split_cohens_d,
            })

    missing_labels.sort(key=lambda x: abs(x['median_split_contrast']), reverse=True)

    with open(output_dir / "missing_feature_labels.json", 'w') as f:
        json.dump({
            'n_missing': len(missing_labels),
            'n_shown_in_plots': len(features_in_plots),
            'pct_missing': len(missing_labels) / len(features_in_plots) * 100 if features_in_plots else 0,
            'features': missing_labels,
        }, f, indent=2)

    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} features with missing labels")
    else:
        print(f"  All {len(features_in_plots)} features shown in plots have labels")

    # Save summary
    positive_contrasts = [r.median_split_contrast for r in results if r.median_split_contrast > 0]
    negative_contrasts = [r.median_split_contrast for r in results if r.median_split_contrast < 0]

    summary = {
        'n_configs': len(CONFIGS),
        'configs': [f"L{l}_S{s}" for l, s in CONFIGS],
        'n_features_analyzed': len(results),
        'n_observations_per_feature': int(np.mean([r.n_observations for r in results])),
        'overall_stats': {
            'mean_contrast': float(np.mean([r.median_split_contrast for r in results])),
            'std_contrast': float(np.std([r.median_split_contrast for r in results])),
            'mean_cohens_d': float(np.mean([r.median_split_cohens_d for r in results])),
            'n_positive_contrast': len(positive_contrasts),
            'n_negative_contrast': len(negative_contrasts),
            'n_significant_p05': sum(1 for r in results if r.median_split_pvalue < 0.05),
            'n_monotonic_increasing': sum(1 for r in results if r.quartile_trend == 'increasing'),
            'n_monotonic_decreasing': sum(1 for r in results if r.quartile_trend == 'decreasing'),
        },
        'top_positive_contrast': [
            {
                'feature': r.physical_feature,
                'label': r.label,
                'contrast': r.median_split_contrast,
                'cohens_d': r.median_split_cohens_d,
                'high_mean': r.median_split_high_mean,
                'low_mean': r.median_split_low_mean,
                'trend': r.quartile_trend,
            }
            for r in sorted(results, key=lambda x: x.median_split_contrast, reverse=True)[:20]
        ],
        'top_negative_contrast': [
            {
                'feature': r.physical_feature,
                'label': r.label,
                'contrast': r.median_split_contrast,
                'cohens_d': r.median_split_cohens_d,
                'high_mean': r.median_split_high_mean,
                'low_mean': r.median_split_low_mean,
                'trend': r.quartile_trend,
            }
            for r in sorted(results, key=lambda x: x.median_split_contrast)[:20]
        ],
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("POOLED ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nFeatures analyzed: {len(results)}")
    print(f"Observations per feature: ~{summary['n_observations_per_feature']}")
    print(f"\nContrast Statistics:")
    print(f"  Mean contrast: {summary['overall_stats']['mean_contrast']:.4f}")
    print(f"  Std contrast: {summary['overall_stats']['std_contrast']:.4f}")
    print(f"  Mean Cohen's d: {summary['overall_stats']['mean_cohens_d']:.3f}")
    print(f"  Features with positive contrast: {summary['overall_stats']['n_positive_contrast']}")
    print(f"  Features with negative contrast: {summary['overall_stats']['n_negative_contrast']}")
    print(f"  Significant at p<0.05: {summary['overall_stats']['n_significant_p05']}")
    print(f"  Monotonically increasing: {summary['overall_stats']['n_monotonic_increasing']}")
    print(f"  Monotonically decreasing: {summary['overall_stats']['n_monotonic_decreasing']}")

    print(f"\nTop 5 POSITIVE contrast features (firing → higher detection):")
    for item in summary['top_positive_contrast'][:5]:
        label_short = item['label'][:40] + "..." if len(item['label']) > 40 else item['label']
        print(f"  {item['feature']}: contrast={item['contrast']:.3f}, d={item['cohens_d']:.2f}")
        print(f"    {label_short}")

    print(f"\nTop 5 NEGATIVE contrast features (firing → lower detection):")
    for item in summary['top_negative_contrast'][:5]:
        label_short = item['label'][:40] + "..." if len(item['label']) > 40 else item['label']
        print(f"  {item['feature']}: contrast={item['contrast']:.3f}, d={item['cohens_d']:.2f}")
        print(f"    {label_short}")

    print(f"\nOutput saved to: {output_dir}/")

    return summary


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parser.parse_args()

    if args.pooled_analysis:
        # Run pooled analysis across all configs
        run_pooled_analysis(
            token_mode=args.token_mode,
            transcoder_l0=args.transcoder_l0,
        )
    elif args.all_configs:
        # Run across all 6 configs
        print("=" * 80)
        print("RUNNING ACROSS ALL CONFIGS")
        print("=" * 80)
        print(f"Configs: {CONFIGS}")
        print()

        all_summaries = {}
        for layer, strength in CONFIGS:
            print(f"\n{'='*80}")
            print(f"CONFIG: Layer {layer}, Strength {strength}")
            print(f"{'='*80}\n")

            try:
                summary = run_feature_centric_analysis(
                    steering_layer=layer,
                    steering_strength=strength,
                    token_mode=args.token_mode,
                    transcoder_l0=args.transcoder_l0,
                    n_features=args.n_features,
                    metric_type=args.metric,
                    all_features=args.all_features,
                    top_n_by_variance=args.top_n_by_variance,
                    category_analysis=args.category_analysis,
                )
                all_summaries[f"L{layer}_S{strength}"] = summary
            except Exception as e:
                print(f"ERROR for L{layer}_S{strength}: {e}")
                continue

        # Save combined summary
        combined_output = OUTPUT_BASE / "combined_summary.json"
        with open(combined_output, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n\nCombined summary saved to: {combined_output}")

    else:
        # Run single config
        run_feature_centric_analysis(
            steering_layer=args.layer,
            steering_strength=args.strength,
            token_mode=args.token_mode,
            transcoder_l0=args.transcoder_l0,
            n_features=args.n_features,
            metric_type=args.metric,
            all_features=args.all_features,
            top_n_by_variance=args.top_n_by_variance,
            category_analysis=args.category_analysis,
        )
