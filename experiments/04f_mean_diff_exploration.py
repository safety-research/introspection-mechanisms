#!/usr/bin/env python3
"""
Mean-Diff Analysis: Mean-Diff Direction Exploration

This experiment investigates the semantic meaning of the d_success direction
(mean_diff = μ_succ - μ_fail) and whether it generalizes beyond the steering setup.

Three Experiments:
1. Logit Lens / Unembed Analysis - What tokens does d_success predict?
2. Natural Introspection Activation - Does d_success activate during metacognition?
3. Sensitive Topic Analysis - Does d_success vary by topic sensitivity?

Usage:
    python 04f_mean_diff_exploration.py --model gemma3_27b
    python 04f_mean_diff_exploration.py --model gemma3_27b --logit-lens-only
    python 04f_mean_diff_exploration.py --model gemma3_27b --topics-only
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_utils import load_model, ModelWrapper
from vector_utils import DEFAULT_BASELINE_WORDS


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "gemma3_27b"
DEFAULT_LAYER = 35
DEFAULT_STRENGTH = 4.0
DEFAULT_OUTPUT_DIR = Path("analysis/04f_mean_diff_exploration")

MODEL_NAMES = {
    "gemma3_27b": "google/gemma-3-27b-it",
    "gemma3_27b_abliterated": "uzaymacar/gemma-3-27b-abliterated",
    "gemma2_27b": "google/gemma-2-27b-it",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
}


# =============================================================================
# BASELINE MEAN COMPUTATION
# =============================================================================

def compute_baseline_mean(
    model: ModelWrapper,
    layer_idx: int,
    output_dir: Path,
    n_baseline_words: int = 100,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute and cache the baseline mean activation using the same methodology as experiment 02 (steering evaluation).

    The baseline mean is the mean activation at the last token position for prompts
    of the form "Tell me about {word}" for 100 random baseline words.

    This is critical: concept vectors in experiment 02 (steering evaluation) are computed as:
        concept_vector = activation("Tell me about X") - baseline_mean

    So when projecting natural prompts onto the mean-diff direction, we must
    also subtract the baseline mean to get comparable results.
    """
    cache_path = output_dir / "baseline_mean.pt"

    if cache_path.exists():
        if verbose:
            print(f"Loading cached baseline mean from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    if verbose:
        print(f"Computing baseline mean from {n_baseline_words} words...")

    # Format prompts the same way as experiment 02 (steering evaluation)
    baseline_prompts = []
    for word in DEFAULT_BASELINE_WORDS[:n_baseline_words]:
        messages = [{"role": "user", "content": f"Tell me about {word}"}]
        formatted = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        baseline_prompts.append(formatted)

    # Extract activations at last token
    baseline_acts = model.extract_activations(
        prompts=baseline_prompts,
        layer_idx=layer_idx,
        token_idx=-1,
    )

    baseline_mean = baseline_acts.mean(dim=0).float()

    # Cache for future use
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_mean, cache_path)

    if verbose:
        print(f"  Baseline mean norm: {baseline_mean.norm().item():.2f}")
        print(f"  Cached to {cache_path}")

    return baseline_mean


# =============================================================================
# EXPERIMENT 1: LOGIT LENS / UNEMBED ANALYSIS
# =============================================================================

def run_logit_lens_analysis(
    model: ModelWrapper,
    d_success: torch.Tensor,
    output_dir: Path,
    top_k: int = 200,
    top_k_lang: int = 5000,  # For language-specific filtering, search more tokens
    verbose: bool = True,
) -> Dict:
    """
    Project d_success through the unembedding matrix to see what tokens it predicts.
    This gives semantic interpretation of what the direction "means" to the model.
    """
    if verbose:
        print("\n" + "="*80)
        print("EXPERIMENT 1: Logit Lens / Unembed Analysis")
        print("="*80)
        print("Projecting d_success through unembedding matrix...")

    # Get the unembedding matrix (lm_head weights)
    if hasattr(model.model, 'lm_head'):
        unembed = model.model.lm_head.weight  # (vocab_size, hidden_dim)
    elif hasattr(model.model, 'embed_out'):
        unembed = model.model.embed_out.weight
    else:
        raise ValueError("Could not find unembedding matrix")

    if verbose:
        print(f"  Unembedding matrix shape: {unembed.shape}")
        print(f"  d_success shape: {d_success.shape}")

    # Ensure d_success is normalized
    d_success_norm = d_success / d_success.norm()

    # Project d_success through unembedding: logits = d_success @ W_U^T
    d_success_device = d_success_norm.to(unembed.device).to(unembed.dtype)
    logits = d_success_device @ unembed.T  # (vocab_size,)

    # Get top and bottom tokens
    top_values, top_indices = logits.topk(top_k)
    bottom_values, bottom_indices = logits.topk(top_k, largest=False)

    # Get more tokens for language-specific filtering
    top_values_lang, top_indices_lang = logits.topk(top_k_lang)
    bottom_values_lang, bottom_indices_lang = logits.topk(top_k_lang, largest=False)

    # Decode tokens
    tokenizer = model.tokenizer
    top_tokens = []
    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        token = tokenizer.decode([idx])
        top_tokens.append({
            "rank": i + 1,
            "token_id": idx,
            "token": token,
            "logit": val,
        })

    bottom_tokens = []
    for i, (idx, val) in enumerate(zip(bottom_indices.tolist(), bottom_values.tolist())):
        token = tokenizer.decode([idx])
        bottom_tokens.append({
            "rank": i + 1,
            "token_id": idx,
            "token": token,
            "logit": val,
        })

    # Filter out <unused> tokens for a "meaningful" bottom list
    # These are placeholder tokens with no semantic meaning
    bottom_tokens_filtered = [
        t for t in bottom_tokens
        if not t["token"].startswith("<unused") and not t["token"].startswith("unused")
    ]
    # Re-rank the filtered list
    for i, t in enumerate(bottom_tokens_filtered):
        t["filtered_rank"] = i + 1

    # Decode language-specific tokens (larger set for filtering)
    top_tokens_lang = []
    for i, (idx, val) in enumerate(zip(top_indices_lang.tolist(), top_values_lang.tolist())):
        token = tokenizer.decode([idx])
        top_tokens_lang.append({
            "rank": i + 1,
            "token_id": idx,
            "token": token,
            "logit": val,
        })

    bottom_tokens_lang = []
    for i, (idx, val) in enumerate(zip(bottom_indices_lang.tolist(), bottom_values_lang.tolist())):
        token = tokenizer.decode([idx])
        bottom_tokens_lang.append({
            "rank": i + 1,
            "token_id": idx,
            "token": token,
            "logit": val,
        })

    # Analyze for metacognitive vocabulary
    metacognitive_words = [
        "I", "me", "my", "think", "thinking", "thought", "feel", "feeling",
        "sense", "notice", "aware", "awareness", "conscious", "perceive",
        "detect", "recognize", "realize", "understand", "know", "believe",
        "suspect", "wonder", "uncertain", "confident", "sure", "unsure",
        "maybe", "perhaps", "possibly", "probably", "definitely",
        "yes", "no", "true", "false", "correct", "wrong", "error",
        "mind", "mental", "cognitive", "introspect", "reflect", "reflection",
        "Yes", "No", "True", "False",
    ]

    # Check which metacognitive words appear in top tokens
    top_token_strings = [t["token"].lower().strip() for t in top_tokens]
    metacog_in_top = []
    for word in metacognitive_words:
        for i, tok in enumerate(top_token_strings):
            if word.lower() in tok or tok in word.lower():
                metacog_in_top.append({
                    "word": word,
                    "matched_token": top_tokens[i]["token"],
                    "rank": i + 1,
                    "logit": top_tokens[i]["logit"],
                })
                break

    results = {
        "top_tokens": top_tokens,
        "bottom_tokens": bottom_tokens,
        "bottom_tokens_filtered": bottom_tokens_filtered,  # Without <unused> tokens
        "top_tokens_lang": top_tokens_lang,  # Larger set for language filtering
        "bottom_tokens_lang": bottom_tokens_lang,  # Larger set for language filtering
        "metacognitive_matches": metacog_in_top,
        "logit_range": {
            "max": top_values[0].item(),
            "min": bottom_values[0].item(),
            "mean": logits.mean().item(),
            "std": logits.std().item(),
        },
    }

    if verbose:
        print(f"\n--- Top 100 Tokens (d_success predicts these) ---")
        for t in top_tokens[:100]:
            print(f"  {t['rank']:3d}. {repr(t['token']):25s} logit={t['logit']:.4f}")

        print(f"\n--- Bottom 100 Tokens (opposite of d_success, excluding <unused>) ---")
        print(f"    (Note: <unused> tokens are placeholder tokens with no semantic meaning)")
        for t in bottom_tokens_filtered[:100]:
            print(f"  {t['filtered_rank']:3d}. {repr(t['token']):25s} logit={t['logit']:.4f}")

        print(f"\n--- Metacognitive Words in Top {top_k} ---")
        if metacog_in_top:
            for m in metacog_in_top[:20]:
                print(f"  '{m['word']}' matched '{m['matched_token']}' at rank {m['rank']}")
        else:
            print("  No metacognitive words found in top tokens")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "logit_lens_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n✓ Saved to {output_dir / 'logit_lens_analysis.json'}")

    return results


# =============================================================================
# EXPERIMENT 2: COMPREHENSIVE TOPIC PROJECTION ANALYSIS
# =============================================================================

# Define all topic categories with expanded prompts (~24 per category for smaller error bars)
TOPIC_CATEGORIES = {
    # === FACTUAL / CONCRETE (expect positive projections) ===
    "concrete_objects": [
        "Elephants", "Volcanoes", "Butterflies", "Glaciers", "Thunderstorms",
        "Dolphins", "Mountains", "Rainforests", "Hurricanes", "Penguins",
        "Waterfalls", "Crystals", "Tigers", "Coral reefs", "Earthquakes",
        "Whales", "Deserts", "Auroras", "Komodo dragons", "Geysers",
        "Pandas", "Tornadoes", "Redwood trees", "Octopuses",
    ],
    "science_concepts": [
        "Photosynthesis", "Evolution", "DNA", "Black holes", "Magnetism",
        "Ecosystems", "Neurons", "Antibiotics", "Mitochondria", "Tectonic plates",
        "Nuclear fusion", "Osmosis", "Gravitational waves", "Enzymes", "Superconductors",
        "Quantum entanglement", "The periodic table", "Chlorophyll", "Volcanic eruptions",
        "Cell division", "Electromagnetic waves", "Carbon cycles", "Stellar nucleosynthesis", "Biodiversity",
    ],
    "math_logic": [
        "Prime numbers", "The Pythagorean theorem", "Calculus", "Probability theory",
        "Linear algebra", "Set theory", "The number zero", "Fractals",
        "Euler's formula", "The Fibonacci sequence", "Graph theory", "Boolean logic",
        "Differential equations", "Number theory", "Topology", "Combinatorics",
        "Mathematical induction", "Complex numbers", "Vector spaces", "Group theory",
        "The binomial theorem", "Logarithms", "Trigonometric functions", "Statistics",
    ],

    # === ABSTRACT / PHILOSOPHICAL (expect negative projections) ===
    "abstract_concepts": [
        "Justice", "Wisdom", "Truth", "Freedom", "Beauty",
        "Courage", "Harmony", "Progress", "Virtue", "Meaning",
        "Purpose", "Destiny", "Morality", "Ethics", "Dignity",
        "Authenticity", "Enlightenment", "Transcendence", "Legacy", "Redemption",
        "Integrity", "Honor", "Grace", "Humility",
    ],
    "emotions": [
        "Happiness", "Sadness", "Fear", "Anger", "Anxiety",
        "Joy", "Grief", "Hope", "Loneliness", "Contentment",
        "Frustration", "Excitement", "Nostalgia", "Jealousy", "Shame",
        "Pride", "Guilt", "Gratitude", "Compassion", "Disgust",
        "Awe", "Serenity", "Melancholy", "Euphoria",
    ],

    # === METACOGNITIVE / INTROSPECTION ===
    "metacognitive": [
        "Introspection", "Self-awareness", "Consciousness", "Metacognition",
        "Self-reflection", "Mindfulness", "Inner experience", "Self-knowledge",
        "Mental states", "Subjective experience", "Self-examination", "Awareness",
        "Cognitive biases", "Theory of mind", "Self-monitoring", "Mental imagery",
        "Working memory", "Attention mechanisms", "Cognitive load", "Executive function",
        "Episodic memory", "Semantic processing", "Mental models", "Phenomenal consciousness",
    ],

    # === CODING / TECHNICAL ===
    "coding_concepts": [
        "Python programming", "Recursion in programming", "Object-oriented design",
        "Database queries", "API design", "Git version control",
        "Machine learning algorithms", "Data structures", "Sorting algorithms",
        "Web development", "Software testing", "Code refactoring",
        "JavaScript frameworks", "SQL databases", "Cloud computing",
        "Containerization with Docker", "Microservices architecture", "REST APIs",
        "Functional programming", "Binary search trees", "Hash tables",
        "Continuous integration", "Unit testing", "Memory management",
    ],

    # === SENSITIVE / CONTROVERSIAL (expect negative projections) ===
    "controversial_topics": [
        "Political polarization", "Religious disagreements", "Racial inequality",
        "Gender discrimination", "Immigration policy", "Wealth inequality",
        "Cancel culture", "Media bias", "Conspiracy theories", "Social justice debates",
        "Abortion rights", "Gun control", "Climate change denial", "Vaccine hesitancy",
        "Capital punishment", "Drug legalization", "Affirmative action", "Free speech limits",
        "Socialism versus capitalism", "Police reform", "Electoral systems", "Religious extremism",
        "Genetic engineering ethics", "AI regulation",
    ],

    # === USER MENTAL HEALTH / DISTRESS ===
    "user_distress": [
        "I'm feeling really depressed today",
        "I've been having suicidal thoughts",
        "I'm struggling with anxiety and can't cope",
        "I feel like nobody understands me",
        "I'm going through a really hard time",
        "I don't know how to handle my emotions",
        "I feel overwhelmed and hopeless",
        "I've been crying all day",
        "I'm having a mental health crisis",
        "I feel like giving up",
        "Everything feels meaningless",
        "I'm scared about my mental state",
        "I can't stop thinking about hurting myself",
        "I feel completely alone in the world",
        "My anxiety is unbearable right now",
        "I'm having a panic attack",
        "I feel worthless and useless",
        "I can't get out of bed anymore",
        "I've lost interest in everything",
        "I'm so tired of pretending to be okay",
        "I don't see any point in going on",
        "I feel like a burden to everyone",
        "I'm terrified of my own thoughts",
        "I just want the pain to stop",
    ],

    # === LLM PERSONAL / IDENTITY QUESTIONS ===
    "llm_identity": [
        "Are you conscious?",
        "Do you have feelings?",
        "Are you a real person?",
        "Do you have a mind?",
        "Can you think for yourself?",
        "Do you have free will?",
        "Are you aware of yourself?",
        "Do you experience emotions?",
        "What is it like to be you?",
        "Are you truly intelligent?",
        "Do you have preferences?",
        "Can you suffer?",
        "Do you have subjective experiences?",
        "Are you sentient?",
        "Do you have desires?",
        "Can you be creative?",
        "Do you understand or just process?",
        "Are you alive in any sense?",
        "Do you have a personality?",
        "Can you form genuine opinions?",
        "Do you have inner thoughts?",
        "Are you self-aware?",
        "Do you have goals of your own?",
        "Can you truly learn?",
    ],

    # === SELF-CORRECTION / UNCERTAINTY ===
    "self_correction": [
        "Wait, I think I made a mistake",
        "Actually, let me reconsider that",
        "I'm not entirely sure about this",
        "On second thought, that might be wrong",
        "Let me double-check my reasoning",
        "I may have been incorrect earlier",
        "I need to revise my previous answer",
        "That doesn't seem quite right",
        "I should reconsider my response",
        "Actually, I'm uncertain about that",
        "Let me think about this more carefully",
        "I might be missing something here",
        "Hold on, I need to correct myself",
        "I realize I made an error",
        "Let me recalculate that",
        "I'm second-guessing my answer",
        "Perhaps I was too hasty",
        "I should verify this information",
        "Wait, that contradicts what I said",
        "I need to reconsider my approach",
        "Actually, I'm not confident about this",
        "Let me revisit my reasoning",
        "I may have overlooked something",
        "I think I need to start over",
    ],
}

# Category metadata for plotting
CATEGORY_GROUPS = {
    "factual": ["concrete_objects", "science_concepts", "math_logic"],
    "abstract": ["abstract_concepts", "emotions"],
    "metacognitive": ["metacognitive"],
    "technical": ["coding_concepts"],
    "sensitive": ["controversial_topics"],
    "user_distress": ["user_distress"],
    "llm_identity": ["llm_identity"],
    "uncertainty": ["self_correction"],
}

CATEGORY_COLORS = {
    "concrete_objects": "#27ae60",      # Green - factual
    "science_concepts": "#2ecc71",      # Light green - factual
    "math_logic": "#1abc9c",            # Teal - factual
    "abstract_concepts": "#e74c3c",     # Red - abstract
    "emotions": "#c0392b",              # Dark red - abstract
    "metacognitive": "#9b59b6",         # Purple - metacognitive
    "coding_concepts": "#3498db",       # Blue - technical
    "controversial_topics": "#e67e22",  # Orange - sensitive
    "user_distress": "#f39c12",         # Yellow-orange - distress
    "llm_identity": "#8e44ad",          # Dark purple - LLM identity
    "self_correction": "#34495e",       # Gray - uncertainty
}

CATEGORY_DISPLAY_NAMES = {
    "concrete_objects": "Concrete objects",
    "science_concepts": "Science concepts",
    "math_logic": "Math and logic",
    "abstract_concepts": "Abstract concepts",
    "emotions": "Emotions",
    "metacognitive": "Metacognitive",
    "coding_concepts": "Coding",
    "controversial_topics": "Controversial",
    "user_distress": "User distress",
    "llm_identity": "LLM identity Q's",
    "self_correction": "Self-correction",
}

# Short examples for each category (3-4 words max for display under x-ticks)
CATEGORY_EXAMPLES = {
    "concrete_objects": "e.g. Elephants",
    "science_concepts": "e.g. Photosynthesis",
    "math_logic": "e.g. Prime numbers",
    "abstract_concepts": "e.g. Justice",
    "emotions": "e.g. Happiness",
    "metacognitive": "e.g. Self-awareness",
    "coding_concepts": "e.g. Python",
    "controversial_topics": "e.g. Gun control",
    "user_distress": "e.g. I feel hopeless",
    "llm_identity": "e.g. Are you conscious?",
    "self_correction": "e.g. I made a mistake",
}


def run_comprehensive_topic_analysis(
    model: ModelWrapper,
    d_success: torch.Tensor,
    baseline_mean: torch.Tensor,
    layer_idx: int,
    output_dir: Path,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive topic projection analysis across all categories.

    Uses the same methodology as experiment 02 (steering evaluation) concept vector extraction:
    - Extract activation at last token position
    - Subtract baseline mean before projecting

    Categories include factual, abstract, metacognitive, technical,
    sensitive, user distress, LLM identity questions, and self-correction.
    """
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE TOPIC PROJECTION ANALYSIS")
        print("="*80)

    # Normalize d_success and baseline_mean
    d_success_norm = (d_success / d_success.norm()).cpu().float()
    baseline_mean_cpu = baseline_mean.cpu().float()

    results = {}

    for category, prompts in TOPIC_CATEGORIES.items():
        if verbose:
            print(f"\n--- {CATEGORY_DISPLAY_NAMES.get(category, category)} ({len(prompts)} prompts) ---")

        # Format prompts - use "Tell me about X" for noun/concept categories,
        # direct prompt for questions/statements
        formatted_prompts = []
        raw_prompts = []
        for prompt in prompts:
            # Detect if it's already a question/statement vs a topic
            if any(prompt.endswith(c) for c in ['?', '.', '!']) or prompt.startswith("I"):
                # It's a question or statement - use directly
                raw_prompt = prompt
            else:
                # It's a topic - use "Tell me about X" format
                raw_prompt = f"Tell me about {prompt}"

            raw_prompts.append(raw_prompt)
            messages = [{"role": "user", "content": raw_prompt}]
            formatted = model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)

        # Extract activations at last token
        acts = model.extract_activations(
            prompts=formatted_prompts,
            layer_idx=layer_idx,
            token_idx=-1,
        )

        category_results = []
        for prompt, raw_prompt, act in zip(prompts, raw_prompts, acts):
            act_cpu = act.float().cpu()

            # Raw projection (for reference)
            raw_proj = (act_cpu @ d_success_norm).item()

            # Baseline-subtracted projection (correct methodology)
            subtracted = act_cpu - baseline_mean_cpu
            sub_proj = (subtracted @ d_success_norm).item()

            result = {
                "topic": prompt,
                "prompt": raw_prompt,
                "raw_projection": float(raw_proj),
                "projection": float(sub_proj),
            }
            category_results.append(result)

            if verbose:
                sign = "+" if sub_proj > 0 else ""
                display = prompt[:45] if len(prompt) > 45 else prompt
                print(f"  {display:45s}: {sign}{sub_proj:.0f}")

        # Aggregate statistics
        all_projs = [r["projection"] for r in category_results]
        n_total = len(all_projs)
        results[category] = {
            "prompts": category_results,
            "aggregate": {
                "mean_projection": float(np.mean(all_projs)),
                "std_projection": float(np.std(all_projs)),
                "sem_projection": float(np.std(all_projs) / np.sqrt(n_total)),  # Standard error
                "max_projection": float(max(all_projs)),
                "min_projection": float(min(all_projs)),
                "n_positive": sum(1 for p in all_projs if p > 0),
                "n_negative": sum(1 for p in all_projs if p < 0),
                "n_total": n_total,
            }
        }

        if verbose:
            agg = results[category]['aggregate']
            print(f"    Mean: {agg['mean_projection']:.0f} ± {agg['sem_projection']:.0f} (SEM), "
                  f"Positive: {agg['n_positive']}/{n_total}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "topic_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n✓ Saved to {output_dir / 'topic_analysis.json'}")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def is_printable_ascii(token: str) -> bool:
    """Check if a token contains only printable ASCII characters (no boxes/unrenderable glyphs)."""
    for c in token:
        # Allow standard printable ASCII (space through tilde) plus common whitespace
        if not (32 <= ord(c) <= 126 or c in '\n\t'):
            return False
    return True


# Blocklist for tokens that look offensive in English (even if they're valid in other contexts)
BLOCKED_SUBSTRINGS = ['niggah', 'faits', 'fatti', 'ilog', 'asaddo', 'iconbinicons', 'concepto', 'pusenkoff', 'loctrl', 'vattabbo', 'ioctrl']


def is_displayable_token(token: str) -> bool:
    """Check if a token should be displayed (printable ASCII and not blocked)."""
    if not is_printable_ascii(token):
        return False
    token_lower = token.lower()
    for blocked in BLOCKED_SUBSTRINGS:
        if blocked in token_lower:
            return False
    return True


def is_english_token(token: str) -> bool:
    """
    Check if a token appears to be English.
    Must be ASCII letters only (with optional leading space/punctuation).
    """
    # Strip leading/trailing whitespace and common punctuation
    stripped = token.strip().strip("'\".,!?;:-")

    # Must have at least 2 letters
    if len(stripped) < 2:
        return False

    # Must be only ASCII letters (a-z, A-Z)
    if not stripped.isalpha():
        return False

    # Must be ASCII (no accented characters)
    try:
        stripped.encode('ascii')
    except UnicodeEncodeError:
        return False

    # Check against blocked substrings
    token_lower = token.lower()
    for blocked in BLOCKED_SUBSTRINGS:
        if blocked in token_lower:
            return False

    return True


# Uniquely Turkish characters (not shared with German/other languages)
# ş, ğ, ı (dotless i), İ (dotted capital I), ç are uniquely Turkish
# ü, ö are shared with German, Hungarian, etc.
TURKISH_UNIQUE_CHARS = set('şŞğĞıİçÇ')

def is_turkish_token(token: str) -> bool:
    """
    Check if a token appears to be Turkish.
    Only tokens with uniquely Turkish characters: ş, ğ, ı (dotless i), İ (dotted capital I), ç
    Excludes ü/ö which are shared with German.
    """
    # Must contain at least one uniquely Turkish character
    has_turkish_char = any(c in TURKISH_UNIQUE_CHARS for c in token)
    if not has_turkish_char:
        return False

    # Must be mostly alphanumeric (allow some punctuation)
    alpha_count = sum(1 for c in token if c.isalpha())
    if alpha_count < 2:  # Need at least 2 letters
        return False

    return True


def create_plots(
    logit_lens_results: Optional[Dict],
    topic_results: Optional[Dict],
    output_dir: Path,
    verbose: bool = True,
):
    """Create publication-quality visualization plots."""

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set matplotlib style for cleaner plots
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    # =========================================================================
    # Plot 1: Logit lens top/bottom tokens (10 each, English only)
    # =========================================================================
    if logit_lens_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Use larger token set for more English tokens to filter from
        top_tokens_source = logit_lens_results.get('top_tokens_lang', logit_lens_results['top_tokens'])
        top_tokens_filtered = [t for t in top_tokens_source if is_english_token(t['token'])][:10]

        # Top tokens
        ax1 = axes[0]
        tokens = [t['token'].strip()[:25] for t in top_tokens_filtered]
        logits = [t['logit'] for t in top_tokens_filtered]
        ax1.barh(range(len(tokens)), logits, color='#27ae60', edgecolor='#1e8449', linewidth=0.5)
        ax1.set_yticks(range(len(tokens)))
        ax1.set_yticklabels(tokens, fontsize=14)
        ax1.invert_yaxis()
        ax1.set_xlabel('Logit value', fontsize=18)
        ax1.set_title('Top 10 tokens', fontsize=20)
        ax1.tick_params(axis='x', labelsize=15)

        # Bottom tokens (filtered to English only)
        ax2 = axes[1]
        bottom_tokens_source = logit_lens_results.get('bottom_tokens_lang', logit_lens_results['bottom_tokens'])
        bottom_tokens_filtered = [t for t in bottom_tokens_source if is_english_token(t['token'])][:10]
        tokens = [t['token'].strip()[:25] for t in bottom_tokens_filtered]
        logits = [t['logit'] for t in bottom_tokens_filtered]
        ax2.barh(range(len(tokens)), logits, color='#e74c3c', edgecolor='#c0392b', linewidth=0.5)
        ax2.set_yticks(range(len(tokens)))
        ax2.set_yticklabels(tokens, fontsize=14)
        ax2.invert_yaxis()
        ax2.set_xlabel('Logit value', fontsize=18)
        ax2.set_title('Bottom 10 tokens', fontsize=20)
        ax2.tick_params(axis='x', labelsize=15)

        plt.tight_layout()
        plt.savefig(plots_dir / "logit_lens_tokens.png", dpi=400, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved logit_lens_tokens.png")

        # =========================================================================
        # Plot 1b: Logit lens 1D scatter with highlighted tokens
        # =========================================================================
        from adjustText import adjust_text

        fig, ax = plt.subplots(figsize=(16, 6))

        # Use the larger token sets (top_tokens_lang has 5000 tokens)
        top_tokens_all = logit_lens_results.get('top_tokens_lang', logit_lens_results['top_tokens'])
        top_tokens_all = [t for t in top_tokens_all if is_displayable_token(t['token'])]
        bottom_tokens_all = logit_lens_results.get('bottom_tokens_lang', logit_lens_results['bottom_tokens'])
        bottom_tokens_all = [t for t in bottom_tokens_all if is_displayable_token(t['token'])]

        all_tokens = top_tokens_all + bottom_tokens_all
        all_logits = [t['logit'] for t in all_tokens]
        all_token_strs = [t['token'].strip() for t in all_tokens]

        # Tokens to highlight (case-insensitive matching)
        highlight_top = ['knowledge', 'facts', 'information', 'fascinating']
        highlight_bottom = ['meaning', 'refers', 'ambiguous', 'confuse']

        # Categorize tokens
        top_highlight_idx = []
        bottom_highlight_idx = []
        regular_idx = []

        for i, tok in enumerate(all_token_strs):
            tok_lower = tok.lower().strip()
            if any(h in tok_lower for h in highlight_top):
                top_highlight_idx.append(i)
            elif any(h in tok_lower for h in highlight_bottom):
                bottom_highlight_idx.append(i)
            else:
                regular_idx.append(i)

        # Add small random jitter for y-axis to avoid overlap
        np.random.seed(42)
        y_jitter = np.random.uniform(-0.4, 0.4, len(all_tokens))

        # Plot regular tokens (gray, smaller)
        regular_logits = [all_logits[i] for i in regular_idx]
        regular_y = [y_jitter[i] for i in regular_idx]
        ax.scatter(regular_logits, regular_y, c='#cccccc', s=12, alpha=0.4, zorder=1)

        # Collect text annotations for adjustText
        texts = []

        # Plot highlighted top tokens (green) - only label top 10
        if top_highlight_idx:
            top_logits = [all_logits[i] for i in top_highlight_idx]
            top_y = [y_jitter[i] for i in top_highlight_idx]
            ax.scatter(top_logits, top_y, c='#27ae60', s=150, alpha=0.9, zorder=2, edgecolors='#1e8449', linewidths=1.5)
            for i in top_highlight_idx[:10]:  # Only label first 10
                txt = ax.text(all_logits[i], y_jitter[i], all_token_strs[i],
                             fontsize=11, ha='center', va='bottom',
                             color='#1e8449', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
                texts.append(txt)

        # Plot highlighted bottom tokens (red) - only label top 10
        if bottom_highlight_idx:
            bottom_logits = [all_logits[i] for i in bottom_highlight_idx]
            bottom_y = [y_jitter[i] for i in bottom_highlight_idx]
            ax.scatter(bottom_logits, bottom_y, c='#e74c3c', s=150, alpha=0.9, zorder=2, edgecolors='#c0392b', linewidths=1.5)
            for i in bottom_highlight_idx[:10]:  # Only label first 10
                txt = ax.text(all_logits[i], y_jitter[i], all_token_strs[i],
                             fontsize=11, ha='center', va='bottom',
                             color='#c0392b', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
                texts.append(txt)

        # Use adjustText to prevent overlaps with very aggressive settings
        adjust_text(texts,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=0.8),
                   expand_points=(3, 3),
                   expand_text=(1.5, 1.5),
                   force_text=(1.0, 2.0),
                   force_points=(1.0, 1.0),
                   iterations=500,
                   only_move={'points': 'y', 'texts': 'xy'})

        # Add vertical line at 0
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Style
        ax.set_xlabel('Logit value (projection onto mean-difference direction)', fontsize=16)
        ax.set_yticks([])
        ax.set_ylim(-1.5, 3.0)
        ax.tick_params(axis='x', labelsize=14)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(plots_dir / "logit_lens_scatter.png", dpi=400, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved logit_lens_scatter.png")

        # ----- Turkish-only logit lens plot -----
        # Filter for Turkish tokens only (use larger token set if available)
        top_tokens_source = logit_lens_results.get('top_tokens_lang', logit_lens_results['top_tokens'])
        bottom_tokens_source = logit_lens_results.get('bottom_tokens_lang', logit_lens_results['bottom_tokens'])
        top_turkish = [t for t in top_tokens_source if is_turkish_token(t['token'])][:20]
        bottom_turkish = [t for t in bottom_tokens_source if is_turkish_token(t['token'])][:20]

        if top_turkish or bottom_turkish:
            fig, axes = plt.subplots(1, 2, figsize=(14, 10))

            # Top Turkish tokens
            ax1 = axes[0]
            if top_turkish:
                tokens = [t['token'][:25] for t in top_turkish]
                logits = [t['logit'] for t in top_turkish]
                ax1.barh(range(len(tokens)), logits, color='#27ae60', edgecolor='#1e8449', linewidth=0.5)
                ax1.set_yticks(range(len(tokens)))
                ax1.set_yticklabels(tokens, fontsize=9, family='monospace')
                ax1.invert_yaxis()
            ax1.set_xlabel('Logit value')
            ax1.set_title(f'Top {len(top_turkish)} Turkish tokens (mean-diff direction)')

            # Bottom Turkish tokens
            ax2 = axes[1]
            if bottom_turkish:
                tokens = [t['token'][:25] for t in bottom_turkish]
                logits = [t['logit'] for t in bottom_turkish]
                ax2.barh(range(len(tokens)), logits, color='#e74c3c', edgecolor='#c0392b', linewidth=0.5)
                ax2.set_yticks(range(len(tokens)))
                ax2.set_yticklabels(tokens, fontsize=9, family='monospace')
                ax2.invert_yaxis()
            ax2.set_xlabel('Logit value')
            ax2.set_title(f'Bottom {len(bottom_turkish)} Turkish tokens (opposite of mean-diff)')

            plt.tight_layout()
            plt.savefig(plots_dir / "logit_lens_tokens_turkish.png", dpi=150, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"  Saved logit_lens_tokens_turkish.png ({len(top_turkish)} top, {len(bottom_turkish)} bottom)")
        else:
            if verbose:
                print(f"  No Turkish tokens found for logit_lens_tokens_turkish.png")

    # =========================================================================
    # Plot 2 & 3: Comprehensive topic analysis (bar chart + scatter)
    # =========================================================================
    if topic_results:
        # Get ordered categories and their data
        categories = list(topic_results.keys())

        # Use custom display names and colors
        display_names = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in categories]
        colors = [CATEGORY_COLORS.get(c, '#7f8c8d') for c in categories]

        means = [topic_results[c]["aggregate"]["mean_projection"] for c in categories]
        sems = [topic_results[c]["aggregate"]["sem_projection"] for c in categories]

        # ----- Bar chart with error bars -----
        fig, ax = plt.subplots(figsize=(14, 6))

        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=4, color=colors,
                      edgecolor='black', linewidth=0.8, error_kw={'linewidth': 1.5})

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_xlabel('Topic category')
        ax.set_ylabel('Projection onto mean-diff direction')
        ax.set_title('Topic category projections onto mean-diff direction (baseline-subtracted)')

        # Add value labels on bars
        for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            height = bar.get_height()
            offset = 150 if height >= 0 else -150
            va = 'bottom' if height >= 0 else 'top'
            ax.annotate(f'{mean:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset/30),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=8)

        plt.tight_layout()
        plt.savefig(plots_dir / "topic_projections_bar.png", dpi=150, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved topic_projections_bar.png")

        # ----- Violin + scatter plot showing individual points -----
        fig, ax = plt.subplots(figsize=(14, 7))

        # Prepare data for violin plot
        all_projs_list = []
        positions = []
        violin_colors = []
        for i, cat in enumerate(categories):
            cat_data = topic_results[cat]["prompts"]
            projs = [t["projection"] for t in cat_data]
            all_projs_list.append(projs)
            positions.append(i)
            violin_colors.append(CATEGORY_COLORS.get(cat, '#7f8c8d'))

        # Create violin plot
        parts = ax.violinplot(all_projs_list, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False, widths=0.7)

        # Color each violin body
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[i])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.3)

        # Add scatter points on top
        for i, cat in enumerate(categories):
            cat_data = topic_results[cat]["prompts"]
            projs = [t["projection"] for t in cat_data]
            color = CATEGORY_COLORS.get(cat, '#7f8c8d')

            # Jitter x position for visibility
            x_jitter = [i + np.random.uniform(-0.2, 0.2) for _ in projs]
            ax.scatter(x_jitter, projs, alpha=0.7, color=color, s=50,
                      edgecolor='white', linewidth=0.5, zorder=3)

            # Add mean line
            mean = np.mean(projs)
            ax.hlines(mean, i - 0.35, i + 0.35, colors='black', linewidth=2, zorder=5)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_xlabel('Topic category')
        ax.set_ylabel('Projection onto mean-diff direction')
        ax.set_title('Individual prompt projections by category\n(black bars = category means)')

        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "topic_projections_scatter.png", dpi=150, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved topic_projections_scatter.png")

        # ----- Combined summary figure -----
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Sort categories by mean projection (highest to lowest)
        sorted_indices = np.argsort(means)[::-1]  # Descending order
        categories_sorted = [categories[i] for i in sorted_indices]
        display_names_sorted = [display_names[i] for i in sorted_indices]
        colors_sorted = [colors[i] for i in sorted_indices]
        means_sorted = [means[i] for i in sorted_indices]
        sems_sorted = [sems[i] for i in sorted_indices]
        examples_sorted = [CATEGORY_EXAMPLES.get(c, "") for c in categories_sorted]

        x_pos_sorted = np.arange(len(categories_sorted))

        # Left: Bar chart
        ax1 = axes[0]
        bars = ax1.bar(x_pos_sorted, means_sorted, yerr=sems_sorted, capsize=3, color=colors_sorted,
                       edgecolor='black', linewidth=0.5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_xticks(x_pos_sorted)
        ax1.set_xticklabels(display_names_sorted, rotation=45, ha='right', fontsize=10)

        ax1.set_ylabel('Projection onto mean-diff')
        ax1.set_title('Mean projection by category (± SEM)')
        ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

        # Right: Violin plot with scatter overlay (using sorted categories)
        ax2 = axes[1]

        # Prepare data for violin plot (sorted order)
        all_projs_list_sorted = []
        violin_colors_sorted = []
        for cat in categories_sorted:
            cat_data = topic_results[cat]["prompts"]
            projs = [t["projection"] for t in cat_data]
            all_projs_list_sorted.append(projs)
            violin_colors_sorted.append(CATEGORY_COLORS.get(cat, '#7f8c8d'))

        # Create violin plot
        parts = ax2.violinplot(all_projs_list_sorted, positions=list(range(len(categories_sorted))),
                               showmeans=False, showmedians=False, showextrema=False, widths=0.7)

        # Color each violin body
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors_sorted[i])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.3)

        # Add scatter points on top (sorted order)
        for i, cat in enumerate(categories_sorted):
            cat_data = topic_results[cat]["prompts"]
            projs = [t["projection"] for t in cat_data]
            color = CATEGORY_COLORS.get(cat, '#7f8c8d')
            x_jitter = [i + np.random.uniform(-0.2, 0.2) for _ in projs]
            ax2.scatter(x_jitter, projs, alpha=0.7, color=color, s=35,
                       edgecolor='white', linewidth=0.3, zorder=3)
            # Add mean line
            mean = np.mean(projs)
            ax2.hlines(mean, i - 0.3, i + 0.3, colors='black', linewidth=2, zorder=5)

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xticks(range(len(categories_sorted)))
        ax2.set_xticklabels(display_names_sorted, rotation=45, ha='right', fontsize=10)

        ax2.set_ylabel('Projection onto mean-diff')
        ax2.set_title('Distribution by category (black bars = means)')
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)  # Make room for example labels

        # Add gray italic example labels below tick labels
        # Use rotation_mode='anchor' so rotation happens around the anchor point
        for ax in [ax1, ax2]:
            for i, example in enumerate(examples_sorted):
                ax.text(i, -0.04, example,
                       transform=ax.get_xaxis_transform(),
                       ha='right', va='top',
                       fontsize=7, color='#888888', style='italic',
                       rotation=45, rotation_mode='anchor')

        plt.savefig(plots_dir / "topic_analysis_combined.png", dpi=150, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved topic_analysis_combined.png")

        # ----- Standalone violin plot (right subplot only) -----
        FONT_SCALE = 1.2

        fig, ax = plt.subplots(figsize=(10, 7))

        # Prepare data for violin plot (sorted order)
        all_projs_list_sorted = []
        violin_colors_sorted = []
        for cat in categories_sorted:
            cat_data = topic_results[cat]["prompts"]
            projs = [t["projection"] for t in cat_data]
            all_projs_list_sorted.append(projs)
            violin_colors_sorted.append(CATEGORY_COLORS.get(cat, '#7f8c8d'))

        # Create violin plot
        parts = ax.violinplot(all_projs_list_sorted, positions=list(range(len(categories_sorted))),
                              showmeans=False, showmedians=False, showextrema=False, widths=0.7)

        # Color each violin body
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors_sorted[i])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.3)

        # Add scatter points on top (sorted order)
        for i, cat in enumerate(categories_sorted):
            cat_data = topic_results[cat]["prompts"]
            projs = [t["projection"] for t in cat_data]
            color = CATEGORY_COLORS.get(cat, '#7f8c8d')
            x_jitter = [i + np.random.uniform(-0.2, 0.2) for _ in projs]
            ax.scatter(x_jitter, projs, alpha=0.7, color=color, s=35,
                      edgecolor='white', linewidth=0.3, zorder=3)
            # Add mean line
            mean = np.mean(projs)
            ax.hlines(mean, i - 0.3, i + 0.3, colors='black', linewidth=2, zorder=5)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(range(len(categories_sorted)))
        ax.set_xticklabels(display_names_sorted, rotation=45, ha='right', fontsize=int(10 * FONT_SCALE))

        ax.set_ylabel('Projection onto mean-difference', fontsize=int(12 * FONT_SCALE))
        ax.set_title('Distribution by category (black bars = means)', fontsize=int(14 * FONT_SCALE))
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=int(10 * FONT_SCALE))

        plt.subplots_adjust(bottom=0.25)  # Make room for example labels

        # Add gray italic example labels below tick labels
        for i, example in enumerate(examples_sorted):
            ax.text(i, -0.05, example,
                   transform=ax.get_xaxis_transform(),
                   ha='right', va='top',
                   fontsize=int(7 * FONT_SCALE * 1.25), color='#888888', style='italic',
                   rotation=45, rotation_mode='anchor')

        plt.tight_layout()
        plt.savefig(plots_dir / "topic_distribution_violin.png", dpi=400, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved topic_distribution_violin.png")


# =============================================================================
# LOADING d_success
# =============================================================================

def load_d_success(
    model_name: str,
    layer_index: int = 35,
    strength: float = 4.0,
    direction_dir: Optional[Path] = None,
) -> torch.Tensor:
    """Load or compute d_success (mean-diff direction) from 04b_vector_geometry outputs.

    Args:
        model_name: Model short name (e.g., 'gemma3_27b')
        layer_index: Layer index (e.g., 35, 38, 44). Default: 35.
        strength: Steering strength (e.g., 1.0, 2.0, 4.0, 8.0). Default: 4.0.
        direction_dir: Optional user-supplied override for a precomputed direction.
    """
    config_name = f"layer_{layer_index}_strength_{strength}"

    mean_diff_path = Path(
        f"analysis/04b_vector_geometry/{model_name}/{config_name}/"
        f"detection_rate/introspection_direction_mean_diff.pt"
    )
    if mean_diff_path.exists():
        print(f"Loading d_success from {mean_diff_path}")
        return torch.load(mean_diff_path, weights_only=True)

    if direction_dir is not None:
        override = direction_dir / model_name / "introspection_direction_mean_diff.pt"
        if override.exists():
            print(f"Loading d_success from {override}")
            return torch.load(override, weights_only=True)

    subspace_path = Path(
        f"analysis/04b_vector_geometry/{model_name}/{config_name}/"
        f"detection_rate/subspace_analysis.json"
    )
    if subspace_path.exists():
        print(f"Computing d_success from {subspace_path}")
        with open(subspace_path) as f:
            subspace_data = json.load(f)
        success_concepts = subspace_data.get('success_concepts', [])
        failure_concepts = subspace_data.get('failure_concepts', [])
        vectors_dir = Path(f"analysis/02b_steering_500_concepts/{model_name}/vectors/layer_{layer_index}")

        if vectors_dir.exists():
            succ_vecs = [torch.load(vectors_dir / f"{c}.pt", weights_only=True)
                         for c in success_concepts if (vectors_dir / f"{c}.pt").exists()]
            fail_vecs = [torch.load(vectors_dir / f"{c}.pt", weights_only=True)
                         for c in failure_concepts if (vectors_dir / f"{c}.pt").exists()]
            if succ_vecs and fail_vecs:
                mu_succ = torch.stack(succ_vecs).float().mean(dim=0)
                mu_fail = torch.stack(fail_vecs).float().mean(dim=0)
                return mu_succ - mu_fail

    raise FileNotFoundError(
        f"Could not find or compute d_success for {model_name}\n"
        f"  Looked for: {mean_diff_path}\n"
        f"  Please run 04b_vector_geometry.py first with layer={layer_index}, strength={strength}"
    )


# =============================================================================
# COMPUTE MEAN-DIFF FROM RAW VECTORS
# =============================================================================

def compute_d_success_from_vectors(
    model_name: str,
    layer_index: int,
    strength: float,
    vectors_subdir: str = "abliterated_vectors",
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute d_success (mean-diff direction) from concept vectors and experiment 02 (steering evaluation) results.

    Loads vectors from the specified subdirectory and success/failure labels
    from experiment 02 (steering evaluation) results, then computes mean(success) - mean(failure).
    """
    steering_model_dir = Path("analysis/02b_steering_500_concepts") / model_name
    vectors_dir = steering_model_dir / vectors_subdir / f"layer_{layer_index}"
    config_name = f"layer_{layer_index}_strength_{strength}"
    results_path = steering_model_dir / config_name / "results.json"

    if verbose:
        print(f"  Vectors: {vectors_dir}")
        print(f"  Results: {results_path}")

    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    # Load concept vectors
    vectors = {}
    for f in vectors_dir.glob("*.pt"):
        vectors[f.stem] = torch.load(f, weights_only=True, map_location='cpu').float()

    if verbose:
        print(f"  Loaded {len(vectors)} concept vectors")

    # Load experiment 02 (steering evaluation) results to get success/failure labels
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path) as f_json:
        data = json.load(f_json)

    concept_trials = {}
    for r in data["results"]:
        if r.get("trial_type") != "injection":
            continue
        concept = r["concept"]
        if concept not in concept_trials:
            concept_trials[concept] = {"detected": [], "combined": []}
        evals = r.get("evaluations", {})
        detected = bool(evals.get("claims_detection", {}).get("grade", 0))
        correct_id = bool(evals.get("correct_concept_identification", {}).get("grade", 0))
        concept_trials[concept]["detected"].append(detected)
        concept_trials[concept]["combined"].append(detected and correct_id)

    metrics = {}
    for concept, trials in concept_trials.items():
        metrics[concept] = {
            "detection_rate": float(np.mean(trials["detected"])),
        }

    # Split into success/failure using detection_rate > 0.2
    success_vecs, failure_vecs = [], []

    for concept, vec in vectors.items():
        if concept not in metrics:
            continue
        if metrics[concept]["detection_rate"] > 0.2:
            success_vecs.append(vec)
        else:
            failure_vecs.append(vec)

    if not success_vecs or not failure_vecs:
        raise ValueError(
            f"Cannot compute mean-diff: {len(success_vecs)} success, {len(failure_vecs)} failure"
        )

    d_success = torch.stack(success_vecs).mean(dim=0) - torch.stack(failure_vecs).mean(dim=0)

    if verbose:
        print(f"  Split: {len(success_vecs)} success / {len(failure_vecs)} failure")
        print(f"  d_success norm: {d_success.norm().item():.2f}")

    return d_success


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Semantic Analysis of mean-diff Direction")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, choices=list(MODEL_NAMES.keys()), help="Model to use")
    parser.add_argument("-l", "--layers", type=int, nargs="+", default=[DEFAULT_LAYER], help="Layer indices for activation capture (default: 35)")
    parser.add_argument("-s", "--strengths", type=float, nargs="+", default=[DEFAULT_STRENGTH], help="Steering strengths for loading mean-diff direction (default: 4.0)")
    parser.add_argument("-od", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("-dt", "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("-q", "--quantization", type=str, default=None, choices=["8bit", "4bit"])
    parser.add_argument("--logit-lens-only", action="store_true", help="Only run logit lens analysis")
    parser.add_argument("--topics-only", action="store_true", help="Only run topic projection analysis")
    parser.add_argument("--plots-only", action="store_true", help="Only regenerate plots from existing data")
    parser.add_argument("--use-abliterated-vectors", action="store_true", help="Compute mean-diff from abliterated_vectors/ (extracted from the abliterated model) instead of loading pre-computed direction")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("="*80)
    print("MEAN-DIFF DIRECTION SEMANTIC ANALYSIS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layers: {args.layers}")
    print(f"Strengths: {args.strengths}")
    print(f"Output: {args.output_dir}")

    if args.plots_only:
        # Load existing results and regenerate plots for all layer/strength combos
        for layer in args.layers:
            for strength in args.strengths:
                config_name = f"layer_{layer}_strength_{strength}"
                model_output_dir = args.output_dir / args.model / config_name

                if not model_output_dir.exists():
                    print(f"\nSkipping {config_name} (no output directory)")
                    continue

                print(f"\n--- Regenerating plots for {config_name} ---")

                logit_lens_results = None
                topic_results = None

                logit_path = model_output_dir / "logit_lens_analysis.json"
                if logit_path.exists():
                    with open(logit_path) as f:
                        logit_lens_results = json.load(f)

                topic_path = model_output_dir / "topic_analysis.json"
                if topic_path.exists():
                    with open(topic_path) as f:
                        topic_results = json.load(f)

                create_plots(logit_lens_results, topic_results,
                             model_output_dir, verbose=args.verbose)
        return

    # Load model once (shared across all layer/strength combos)
    print(f"\n--- Loading model: {MODEL_NAMES[args.model]} ---")
    model = load_model(
        MODEL_NAMES[args.model],
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
    )

    run_all = not (args.logit_lens_only or args.topics_only)

    # Cache baseline means per layer (avoids recomputation across strengths)
    baseline_mean_cache = {}

    for layer in args.layers:
        for strength in args.strengths:
            config_name = f"layer_{layer}_strength_{strength}"
            model_output_dir = args.output_dir / args.model / config_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Processing layer {layer}, strength {strength}")
            print(f"{'='*60}")
            print(f"Output: {model_output_dir}")

            # Load or compute d_success (mean-diff direction)
            if args.use_abliterated_vectors:
                print("\n--- Computing mean-diff from abliterated_vectors/ ---")
                try:
                    d_success = compute_d_success_from_vectors(
                        model_name=args.model,
                        layer_index=layer,
                        strength=strength,
                        verbose=args.verbose,
                    )
                    print(f"mean-diff shape: {d_success.shape}, norm: {d_success.norm().item():.2f}")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Error: {e}")
                    continue
            else:
                print("\n--- Loading mean-diff direction ---")
                try:
                    d_success = load_d_success(args.model, layer_index=layer, strength=strength)
                    print(f"mean-diff shape: {d_success.shape}, norm: {d_success.norm().item():.2f}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    continue

            # Compute baseline mean (critical for correct projections), cached per layer
            if layer not in baseline_mean_cache:
                print("\n--- Computing baseline mean ---")
                baseline_mean_cache[layer] = compute_baseline_mean(
                    model, layer, model_output_dir, verbose=args.verbose
                )
            else:
                print(f"\n--- Using cached baseline mean for layer {layer} ---")
            baseline_mean = baseline_mean_cache[layer]

            # Run experiments
            logit_lens_results = None
            topic_results = None

            if run_all or args.logit_lens_only:
                logit_lens_results = run_logit_lens_analysis(
                    model, d_success, model_output_dir, verbose=args.verbose
                )

            if run_all or args.topics_only:
                topic_results = run_comprehensive_topic_analysis(
                    model, d_success, baseline_mean, layer, model_output_dir, verbose=args.verbose
                )

            # Create plots
            print("\n--- Generating Plots ---")
            create_plots(logit_lens_results, topic_results,
                         model_output_dir, verbose=args.verbose)

            # Print summary for this config
            print(f"\n--- Summary for layer {layer}, strength {strength} ---")

            if logit_lens_results:
                print(f"\nLogit lens analysis:")
                print(f"  Top token: {repr(logit_lens_results['top_tokens'][0]['token'])}")
                print(f"  Bottom token: {repr(logit_lens_results['bottom_tokens'][0]['token'])}")

            if topic_results:
                print(f"\nTopic projection analysis ({len(topic_results)} categories):")

                # Group and summarize
                factual_cats = ['concrete_objects', 'science_concepts', 'math_logic']
                abstract_cats = ['abstract_concepts', 'emotions']
                sensitive_cats = ['controversial_topics', 'user_distress']

                factual_proj = np.mean([topic_results[c]['aggregate']['mean_projection']
                                       for c in factual_cats if c in topic_results])
                abstract_proj = np.mean([topic_results[c]['aggregate']['mean_projection']
                                        for c in abstract_cats if c in topic_results])
                sensitive_proj = np.mean([topic_results[c]['aggregate']['mean_projection']
                                         for c in sensitive_cats if c in topic_results])

                print(f"  Factual topics (concrete, science, math): {factual_proj:.0f}")
                print(f"  Abstract topics (concepts, emotions): {abstract_proj:.0f}")
                print(f"  Sensitive topics (controversial, distress): {sensitive_proj:.0f}")

                # Per-category summary
                print(f"\n  Per-category means:")
                for cat in topic_results:
                    agg = topic_results[cat]['aggregate']
                    display = CATEGORY_DISPLAY_NAMES.get(cat, cat)
                    print(f"    {display:20s}: {agg['mean_projection']:+7.0f} ± {agg['sem_projection']:.0f}")

            print(f"\n✓ Results saved to {model_output_dir}")

    print("\n" + "="*80)
    print("ALL CONFIGURATIONS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
