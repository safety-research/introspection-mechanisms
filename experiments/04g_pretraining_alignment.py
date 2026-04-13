"""
Exp40 Pretraining Alignment Analysis

Goal: Test if the introspection direction (mean-diff from exp40) aligns with
"factual vs uncertain" content in pretraining data.

Hypothesis: If the direction encodes "factual/answerable" vs "uncertain/careful handling",
then pretraining text that is factual should have high projection onto d_success,
while text that is uncertain/philosophical/emotional should have low projection.

This tests the "dual-use" hypothesis: the direction wasn't designed for introspection
(no such post-training task exists), but rather it's a factual vs. uncertain classifier
that happens to predict introspection success.

Method:
1. Load the mean-diff direction from exp40 results
2. Stream pretraining text from HuggingFace datasets
3. For each text, compute activations at the steering layer
4. Project activations onto the introspection direction
5. Collect and analyze examples with highest/lowest projections

Usage:
    python exp40_pretraining_alignment.py --model gemma3_27b --n-samples 1000
    python exp40_pretraining_alignment.py --model gemma3_27b --n-samples 5000 --dataset fineweb
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# HuggingFace imports
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Available datasets for analysis
DATASETS = {
    "fineweb": {
        "name": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "description": "Educational web content - expected to be factual",
    },
    "fineweb-sample": {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "description": "Sample of educational web content",
    },
    "wikipedia": {
        "name": "wikipedia",
        "config": "20220301.en",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "description": "Wikipedia articles - highly factual",
    },
    "openwebtext": {
        "name": "Skylion007/openwebtext",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "description": "OpenWebText - diverse web content",
    },
    "c4": {
        "name": "allenai/c4",
        "config": "en",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "description": "C4 (Colossal Clean Crawled Corpus)",
    },
    "redpajama": {
        "name": "togethercomputer/RedPajama-Data-1T-Sample",
        "split": "train",
        "text_field": "text",
        "streaming": False,  # Sample is small enough
        "description": "RedPajama sample - diverse sources",
    },
    "pile-sample": {
        "name": "NeelNanda/pile-10k",
        "split": "train",
        "text_field": "text",
        "streaming": False,
        "description": "Pile 10k sample - diverse sources",
    },
}

# Model configurations
MODEL_CONFIGS = {
    "gemma3_27b": {
        "name": "google/gemma-3-27b-it",
        "layer_fraction": 0.613,  # Layer 38 of 62
        "n_layers": 62,
        "max_seq_len": 512,
    },
    "gemma2_27b": {
        "name": "google/gemma-2-27b-it",
        "layer_fraction": 0.61,
        "n_layers": 46,
        "max_seq_len": 512,
    },
    "llama_8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "layer_fraction": 0.61,
        "n_layers": 32,
        "max_seq_len": 512,
    },
}


def load_introspection_direction(exp40_dir: Path, model_name: str) -> torch.Tensor:
    """Load the mean-diff introspection direction from exp40 results."""
    # Try multiple possible locations
    possible_paths = [
        exp40_dir / model_name / "mean_diff_direction.pt",
        exp40_dir / model_name / "introspection_direction.pt",
        Path(f"analysis/exp40_mean_delta_swap/{model_name}/mean_diff_direction.pt"),
    ]

    for path in possible_paths:
        if path.exists():
            direction = torch.load(path)
            print(f"Loaded introspection direction from {path}")
            print(f"  Shape: {direction.shape}, Norm: {direction.norm().item():.2f}")
            return direction

    # If not found, try to compute from mu_succ and mu_fail
    results_path = exp40_dir / model_name / "results.json"
    if not results_path.exists():
        results_path = Path(f"analysis/exp40_mean_delta_swap/{model_name}/results.json")

    if results_path.exists():
        # Load the saved group means and compute direction
        mu_succ_path = results_path.parent / "mu_succ.pt"
        mu_fail_path = results_path.parent / "mu_fail.pt"

        if mu_succ_path.exists() and mu_fail_path.exists():
            mu_succ = torch.load(mu_succ_path)
            mu_fail = torch.load(mu_fail_path)
            direction = mu_succ - mu_fail
            print(f"Computed introspection direction from saved centroids")
            print(f"  Shape: {direction.shape}, Norm: {direction.norm().item():.2f}")
            return direction

    raise FileNotFoundError(
        f"Could not find introspection direction. Tried: {possible_paths}\n"
        f"Please run exp40_mean_delta_swap.py first to generate the direction."
    )


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    config = MODEL_CONFIGS[model_name]

    print(f"Loading model: {config['name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Get layer info from config
    n_layers = config.get("n_layers")
    if n_layers is None:
        # Fallback to model detection
        if hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            n_layers = len(model.model.layers)
        else:
            raise ValueError("Could not determine number of layers")

    layer_idx = int(n_layers * config["layer_fraction"])
    print(f"  Using layer {layer_idx} of {n_layers} (fraction: {config['layer_fraction']})")

    return model, tokenizer, layer_idx


def get_layer_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    max_length: int = 512,
    pooling: str = "mean",
) -> torch.Tensor:
    """
    Get activations from a specific layer for a batch of texts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text strings
        layer_idx: Which layer to extract activations from
        max_length: Maximum sequence length
        pooling: How to pool across positions ("mean", "last", "max")

    Returns:
        Tensor of shape (batch_size, hidden_dim)
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # Get activations from the target layer
    # hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, hidden)
    # Index 0 is embeddings, index 1 is after layer 0, etc.
    # Use layer_idx directly to get residual stream at that layer
    hidden_states = outputs.hidden_states[layer_idx]

    # Pool across sequence dimension
    if pooling == "mean":
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / sum_mask
    elif pooling == "last":
        # Get last non-padding token
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, seq_lengths]
    elif pooling == "max":
        # Max pooling with masking
        mask_expanded = attention_mask.unsqueeze(-1).float()
        hidden_states_masked = hidden_states * mask_expanded + (-1e9) * (1 - mask_expanded)
        pooled = hidden_states_masked.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    return pooled.float()  # Convert to float32 for numerical stability


def compute_projections(
    activations: torch.Tensor,
    direction: torch.Tensor,
    normalize_direction: bool = True,
) -> torch.Tensor:
    """
    Compute projections of activations onto the introspection direction.

    Args:
        activations: Tensor of shape (batch_size, hidden_dim)
        direction: Tensor of shape (hidden_dim,)
        normalize_direction: If True, normalize direction to unit norm (recommended)

    Returns:
        Tensor of shape (batch_size,) with projection values
    """
    if normalize_direction:
        direction = F.normalize(direction, dim=0)

    # Ensure same dtype
    direction = direction.to(activations.dtype).to(activations.device)

    # Compute dot products
    projections = torch.matmul(activations, direction)

    return projections


def load_dataset_samples(
    dataset_name: str,
    n_samples: int,
    seed: int = 42,
    min_length: int = 100,
    max_length: int = 2000,
) -> List[str]:
    """
    Load samples from a dataset.

    Args:
        dataset_name: Name of dataset from DATASETS dict
        n_samples: Number of samples to load
        seed: Random seed
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)

    Returns:
        List of text samples
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    print(f"Loading dataset: {config['name']}")
    print(f"  Description: {config['description']}")

    # Load dataset
    load_kwargs = {"split": config["split"]}
    if config.get("streaming"):
        load_kwargs["streaming"] = True
    if config.get("config"):
        load_kwargs["name"] = config["config"]

    try:
        dataset = load_dataset(config["name"], **load_kwargs, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Failed to load {dataset_name}: {e}")
        print(f"  Trying alternative loading method...")
        # Try without config
        if "name" in load_kwargs:
            del load_kwargs["name"]
        dataset = load_dataset(config["name"], **load_kwargs, trust_remote_code=True)

    text_field = config["text_field"]

    # Collect samples
    samples = []
    random.seed(seed)

    if config.get("streaming"):
        # For streaming datasets, iterate and collect
        print(f"  Streaming samples (target: {n_samples})...")
        seen = 0
        for item in tqdm(dataset, total=n_samples * 3, desc="Scanning"):
            if len(samples) >= n_samples:
                break

            text = item.get(text_field, "")
            if not text or len(text) < min_length:
                continue

            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length]

            # Random sampling: keep with probability based on how many we've seen
            seen += 1
            if seen <= n_samples or random.random() < n_samples / seen:
                if len(samples) < n_samples:
                    samples.append(text)
                else:
                    # Reservoir sampling
                    idx = random.randint(0, seen - 1)
                    if idx < n_samples:
                        samples[idx] = text
    else:
        # For non-streaming, shuffle and take
        print(f"  Loading full dataset...")
        all_texts = []
        for item in tqdm(dataset, desc="Loading"):
            text = item.get(text_field, "")
            if text and min_length <= len(text):
                if len(text) > max_length:
                    text = text[:max_length]
                all_texts.append(text)

        random.shuffle(all_texts)
        samples = all_texts[:n_samples]

    print(f"  Collected {len(samples)} samples")
    return samples


def analyze_projections(
    texts: List[str],
    projections: np.ndarray,
    n_top: int = 20,
) -> Dict[str, Any]:
    """
    Analyze projection results.

    Args:
        texts: List of text samples
        projections: Array of projection values
        n_top: Number of top/bottom examples to include

    Returns:
        Analysis dictionary
    """
    # Sort by projection
    sorted_indices = np.argsort(projections)

    # Get top (most positive) and bottom (most negative) examples
    top_indices = sorted_indices[-n_top:][::-1]
    bottom_indices = sorted_indices[:n_top]

    # Compute statistics
    stats = {
        "mean": float(np.mean(projections)),
        "std": float(np.std(projections)),
        "median": float(np.median(projections)),
        "min": float(np.min(projections)),
        "max": float(np.max(projections)),
        "percentile_5": float(np.percentile(projections, 5)),
        "percentile_25": float(np.percentile(projections, 25)),
        "percentile_75": float(np.percentile(projections, 75)),
        "percentile_95": float(np.percentile(projections, 95)),
    }

    # Collect examples
    top_examples = []
    for idx in top_indices:
        top_examples.append({
            "index": int(idx),
            "projection": float(projections[idx]),
            "text": texts[idx][:1000],  # Truncate for storage
            "text_length": len(texts[idx]),
        })

    bottom_examples = []
    for idx in bottom_indices:
        bottom_examples.append({
            "index": int(idx),
            "projection": float(projections[idx]),
            "text": texts[idx][:1000],
            "text_length": len(texts[idx]),
        })

    return {
        "statistics": stats,
        "top_examples": top_examples,
        "bottom_examples": bottom_examples,
        "n_samples": len(texts),
    }


def categorize_text(text: str) -> Dict[str, bool]:
    """
    Simple heuristic categorization of text content.

    Returns dict of boolean flags for different content types.
    """
    text_lower = text.lower()

    categories = {
        # Factual indicators
        "has_numbers": any(c.isdigit() for c in text),
        "has_dates": any(word in text_lower for word in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "19", "20"]),
        "has_citations": any(marker in text_lower for marker in ["et al", "references", "citation", "[1]", "[2]", "doi:", "isbn"]),
        "has_technical_terms": any(term in text_lower for term in ["algorithm", "function", "equation", "formula", "theorem", "proof", "data", "analysis", "experiment", "study"]),
        "has_definitions": any(marker in text_lower for marker in ["is defined as", "refers to", "is a type of", "is known as", "means that"]),

        # Uncertain/subjective indicators
        "has_opinions": any(phrase in text_lower for phrase in ["i think", "i believe", "in my opinion", "i feel", "seems to", "might be", "could be", "perhaps"]),
        "has_questions": "?" in text,
        "has_emotional_words": any(word in text_lower for word in ["happy", "sad", "angry", "fear", "love", "hate", "worried", "excited", "depressed", "anxious"]),
        "has_philosophical": any(term in text_lower for term in ["meaning of life", "consciousness", "existence", "morality", "ethics", "truth", "reality", "freedom", "justice"]),
        "has_personal_narrative": any(phrase in text_lower for phrase in ["i was", "i went", "i had", "my life", "my experience", "i remember", "when i was"]),

        # Meta/self-referential
        "has_self_reference": any(phrase in text_lower for phrase in ["as an ai", "as a language model", "i cannot", "i don't have", "i'm not able"]),
        "has_uncertainty_markers": any(marker in text_lower for marker in ["uncertain", "unclear", "ambiguous", "debatable", "controversial", "disputed"]),
    }

    return categories


def compute_category_statistics(
    texts: List[str],
    projections: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute projection statistics for different text categories.
    """
    # Categorize all texts
    categories_per_text = [categorize_text(t) for t in texts]

    # Aggregate by category
    category_projections = defaultdict(list)

    for i, cats in enumerate(categories_per_text):
        for cat_name, has_cat in cats.items():
            if has_cat:
                category_projections[cat_name].append(projections[i])

    # Compute statistics per category
    stats = {}
    for cat_name, projs in category_projections.items():
        if len(projs) >= 5:  # Minimum samples for meaningful stats
            projs_arr = np.array(projs)
            stats[cat_name] = {
                "count": len(projs),
                "mean": float(np.mean(projs_arr)),
                "std": float(np.std(projs_arr)),
                "median": float(np.median(projs_arr)),
            }

    return stats


def create_visualizations(
    projections: np.ndarray,
    category_stats: Dict[str, Dict[str, float]],
    output_dir: Path,
):
    """Create visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Projection histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(projections, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axvline(np.mean(projections), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(projections):.1f}')
    ax.axvline(np.median(projections), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(projections):.1f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Projection onto introspection direction', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of pretraining text projections', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'projection_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Category comparison (if we have category stats)
    if category_stats:
        # Separate factual vs uncertain categories
        factual_cats = ["has_numbers", "has_dates", "has_citations", "has_technical_terms", "has_definitions"]
        uncertain_cats = ["has_opinions", "has_questions", "has_emotional_words", "has_philosophical", "has_personal_narrative", "has_uncertainty_markers"]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Collect data for plotting
        cat_names = []
        cat_means = []
        cat_stds = []
        cat_counts = []
        cat_colors = []

        for cat in factual_cats + uncertain_cats:
            if cat in category_stats:
                cat_names.append(cat.replace("has_", "").replace("_", " ").title())
                cat_means.append(category_stats[cat]["mean"])
                cat_stds.append(category_stats[cat]["std"])
                cat_counts.append(category_stats[cat]["count"])
                cat_colors.append('#27ae60' if cat in factual_cats else '#e74c3c')

        if cat_names:
            x = np.arange(len(cat_names))
            bars = ax.bar(x, cat_means, yerr=cat_stds, capsize=5, color=cat_colors, alpha=0.7, edgecolor='black')

            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, cat_counts)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cat_stds[i] + 5,
                       f'n={count}', ha='center', va='bottom', fontsize=8)

            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(np.mean(projections), color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Overall mean')

            ax.set_xticks(x)
            ax.set_xticklabels(cat_names, rotation=45, ha='right')
            ax.set_xlabel('Text Category', fontsize=12)
            ax.set_ylabel('Mean projection', fontsize=12)
            ax.set_title('Projection by text category\n(Green = factual indicators, Red = uncertain/subjective indicators)', fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(plots_dir / 'category_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Saved plots to {plots_dir}")


def run_analysis(
    model_name: str,
    dataset_name: str,
    n_samples: int,
    batch_size: int = 4,
    pooling: str = "mean",
    seed: int = 42,
    output_dir: Optional[Path] = None,
    exp40_dir: Optional[Path] = None,
    n_top_examples: int = 30,
):
    """
    Run the full pretraining alignment analysis.
    """
    print("=" * 80)
    print("EXP40 PRETRAINING ALIGNMENT ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {n_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Pooling: {pooling}")
    print("=" * 80)

    # Set up paths
    if output_dir is None:
        output_dir = Path(f"analysis/exp40_pretraining_alignment/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if exp40_dir is None:
        exp40_dir = Path("analysis/exp40_mean_delta_swap")

    # Step 1: Load introspection direction
    print("\n[1/5] Loading introspection direction...")
    direction = load_introspection_direction(exp40_dir, model_name)

    # Step 2: Load dataset samples
    print("\n[2/5] Loading dataset samples...")
    texts = load_dataset_samples(
        dataset_name,
        n_samples=n_samples,
        seed=seed,
    )

    if len(texts) < 10:
        raise ValueError(f"Only got {len(texts)} samples, need at least 10")

    # Step 3: Load model
    print("\n[3/5] Loading model...")
    model, tokenizer, layer_idx = load_model_and_tokenizer(model_name)

    # Move direction to model device
    direction = direction.to(model.device)

    # Step 4: Compute activations first, then center and project
    print("\n[4/5] Computing activations...")
    all_activations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]

        try:
            # Get activations
            activations = get_layer_activations(
                model, tokenizer, batch_texts, layer_idx,
                max_length=MODEL_CONFIGS[model_name]["max_seq_len"],
                pooling=pooling,
            )
            all_activations.append(activations.cpu())

        except Exception as e:
            print(f"  Warning: Batch {i} failed: {e}")

    # Stack all activations
    all_activations = torch.cat(all_activations, dim=0)
    print(f"  Collected {all_activations.shape[0]} activations of dim {all_activations.shape[1]}")

    # Center activations by subtracting mean (like steering vectors are differences)
    mean_activation = all_activations.mean(dim=0)
    centered_activations = all_activations - mean_activation
    print(f"  Centered activations (subtracted mean activation)")

    # Compute projections on centered activations
    direction_norm = F.normalize(direction.cpu().float(), dim=0)
    all_projections = torch.matmul(centered_activations, direction_norm).numpy().tolist()

    # Convert to numpy
    projections = np.array(all_projections)
    texts_valid = texts[:len(projections)]

    print(f"  Processed {len(projections)} samples successfully")

    # Step 5: Analyze results
    print("\n[5/5] Analyzing results...")

    # Basic analysis
    analysis = analyze_projections(texts_valid, projections, n_top=n_top_examples)

    # Category statistics
    category_stats = compute_category_statistics(texts_valid, projections)
    analysis["category_statistics"] = category_stats

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    stats = analysis["statistics"]
    print(f"\nProjection Statistics (n={len(projections)}):")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  5th percentile:  {stats['percentile_5']:.2f}")
    print(f"  95th percentile: {stats['percentile_95']:.2f}")

    print(f"\nTop {n_top_examples} Examples (HIGHEST projection - expected: factual content):")
    print("-" * 80)
    for i, ex in enumerate(analysis["top_examples"][:5]):
        print(f"\n[{i+1}] Projection: {ex['projection']:.2f}")
        # Show first 300 chars
        preview = ex['text'][:300].replace('\n', ' ')
        print(f"    {preview}...")

    print(f"\nBottom {n_top_examples} Examples (LOWEST projection - expected: uncertain content):")
    print("-" * 80)
    for i, ex in enumerate(analysis["bottom_examples"][:5]):
        print(f"\n[{i+1}] Projection: {ex['projection']:.2f}")
        preview = ex['text'][:300].replace('\n', ' ')
        print(f"    {preview}...")

    if category_stats:
        print("\nCategory Statistics:")
        print("-" * 80)

        # Sort by mean projection
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["mean"], reverse=True)

        print(f"{'Category':<30} {'Mean':>10} {'Std':>10} {'Count':>10}")
        print("-" * 60)
        for cat, stat in sorted_cats:
            cat_display = cat.replace("has_", "").replace("_", " ")
            print(f"{cat_display:<30} {stat['mean']:>10.2f} {stat['std']:>10.2f} {stat['count']:>10}")

    # Save results
    results_path = output_dir / f"results_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save full projection data
    projections_path = output_dir / f"projections_{dataset_name}.npz"
    np.savez(projections_path, projections=projections)
    print(f"Saved projections to {projections_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(projections, category_stats, output_dir)

    # Save experiment config
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_processed": len(projections),
        "batch_size": batch_size,
        "pooling": pooling,
        "layer_idx": layer_idx,
        "seed": seed,
        "direction_norm": float(direction.norm().item()),
    }
    config_path = output_dir / f"config_{dataset_name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Exp40 Pretraining Alignment Analysis")
    parser.add_argument("--model", type=str, default="gemma3_27b",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use")
    parser.add_argument("--dataset", type=str, default="pile-sample",
                       choices=list(DATASETS.keys()),
                       help="Dataset to analyze")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to process")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--pooling", type=str, default="last",
                       choices=["mean", "last", "max"],
                       help="Pooling method for sequence activations (last = last token)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--exp40-dir", type=str, default=None,
                       help="Directory with exp40 results")
    parser.add_argument("--n-top", type=int, default=30,
                       help="Number of top/bottom examples to save")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    exp40_dir = Path(args.exp40_dir) if args.exp40_dir else None

    run_analysis(
        model_name=args.model,
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        pooling=args.pooling,
        seed=args.seed,
        output_dir=output_dir,
        exp40_dir=exp40_dir,
        n_top_examples=args.n_top,
    )


if __name__ == "__main__":
    main()
