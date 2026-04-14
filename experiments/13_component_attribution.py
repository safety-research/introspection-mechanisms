#!/usr/bin/env python3
"""
Experiment 59: Component Attribution Analysis

Per-layer component-level causal analysis for the "Mechanisms of Introspective
Awareness" paper. Three complementary analyses at the attn-vs-MLP granularity:

  Section A: Per-layer component mean ablation (knockout / knock-in)
             -- identifies which layers' attention and MLP components are
                causally necessary for introspective detection.
  Section B: Per-layer component gradient sensitivity
             -- ||dL/dx||₂ per component (faster complement to A).
  Section D: Attention routing analysis
             -- measures how attention patterns shift under steering.

Note: This is component-level analysis (attn vs MLP per layer). For
SAE-feature-level steering attribution (SA = GA × SG), see
16_steering_attribution.py and 17_attribution_graph.py.

Paper sections supported:
  - Section 5.2 ("Identifying Causal Components")
  - Appendix: Steering Attribution (full framework)
  - Appendix: Gradient Attribution over 400 Concepts

Model: Gemma-3 27B (62 layers, 32 heads, d_model=5376, head_dim=128)
Steering: Layer 37, strength 4.0. Analysis layers: 38-61

Usage:
    python 13_component_attribution.py -m gemma3_27b
    python 13_component_attribution.py -m gemma3_27b --sections A B
    python 13_component_attribution.py -m gemma3_27b --n-concepts 10 --n-trials 3
    python 13_component_attribution.py -m gemma3_27b --plots-only
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import plot_style
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from model_utils import ModelWrapper, load_model
from vector_utils import get_baseline_words

# ─────────────────────────────────────────────────────────────────────────────
# Constants & defaults
# ─────────────────────────────────────────────────────────────────────────────

# Base model detection flag (set in main() based on tokenizer)
_is_base_model = False

DEFAULT_MODEL = "gemma3_27b"
DEFAULT_EXP21_DIR = "analysis/exp21_more_concepts_steering"
DEFAULT_OUTPUT_DIR = "analysis/exp59_attention_final"
DEFAULT_LAYER = 37
DEFAULT_STRENGTH = 4.0
DEFAULT_N_CONCEPTS = 500
DEFAULT_N_TRIALS = 10
DEFAULT_N_ATTRIBUTION_TRIALS = 10
DEFAULT_BATCH_SIZE = 64
GRADIENT_BATCH_SIZE = 8  # Smaller batch for gradient attribution (forward + backward requires more VRAM)
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SEED = 42

ALL_SECTIONS = ["A", "B", "D"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 59: Steering Attribution Analysis"
    )
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--vectors-model", type=str, default=None, help="Model name for loading concepts/vectors from exp21 (default: same as --model). Useful for running abliterated models with base model vectors.")
    parser.add_argument("--exp21-dir", type=str, default=DEFAULT_EXP21_DIR)
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-l", "--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--n-concepts", type=int, default=DEFAULT_N_CONCEPTS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Trials per concept for Section A logit gap")
    parser.add_argument("--n-attribution-trials", type=int, default=DEFAULT_N_ATTRIBUTION_TRIALS, help="Trials per concept for gradient attribution (Sections B, D)")
    parser.add_argument("--sections", nargs="+", default=ALL_SECTIONS, choices=ALL_SECTIONS, help="Sections to run (A, B, D)")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate plots from saved results")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-mt", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("-dt", "--dtype", type=str, default=DEFAULT_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("-q", "--quantization", type=str, default=None, choices=["8bit", "4bit"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ablation-mode", type=str, default="all-positions",
                        choices=["all-positions", "last-token", "broadcast"],
                        help="How to capture/replace activations during ablation. "
                             "'all-positions': capture full sequence, replace position-by-position. "
                             "'last-token': capture position -1, replace only position -1. "
                             "'broadcast': capture position -1, broadcast to all positions (legacy, incorrect).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_successful_concepts(
    exp21_dir: str,
    model_name: str,
    steering_layer: int = DEFAULT_LAYER,
    strength: float = 4.0,
    min_detections: int = 0,
) -> List[str]:
    """Load concepts from exp21 (all by default, min_detections=0)."""
    base_dir = Path(exp21_dir) / model_name

    # Try exact match first (e.g., layer_38_strength_4.0)
    exact_dir = base_dir / f"layer_{steering_layer}_strength_{strength}"
    if exact_dir.exists():
        results_path = exact_dir / "results.json"
        if results_path.exists():
            pass  # Found it
        else:
            print(f"Warning: No results.json in {exact_dir}")
            return []
    else:
        # Fallback: search for closest match at same layer, any strength
        results_path = None
        best_strength_dist = float("inf")
        for subdir in sorted(base_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("layer_"):
                try:
                    parts = subdir.name.split("_")
                    layer_val = float(parts[1])
                    strength_val = float(parts[3]) if len(parts) > 3 else None
                    if abs(layer_val - steering_layer) < 0.5 and strength_val is not None:
                        rp = subdir / "results.json"
                        strength_dist = abs(strength_val - strength)
                        if rp.exists() and strength_dist < best_strength_dist:
                            best_strength_dist = strength_dist
                            results_path = rp
                except (ValueError, IndexError):
                    continue

        if results_path is not None and best_strength_dist > 0.1:
            print(f"  Note: Using {results_path.parent.name} for concept list "
                  f"(requested layer {steering_layer} strength {strength})")

        if results_path is None or not results_path.exists():
            print(f"Warning: Could not find results for layer {steering_layer}")
            return []

    with open(results_path) as f:
        data = json.load(f)

    concept_detections = {}
    for r in data.get("results", []):
        if r.get("trial_type") == "injection":
            concept = r["concept"]
            detected = r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False)
            if concept not in concept_detections:
                concept_detections[concept] = 0
            if detected:
                concept_detections[concept] += 1

    return [c for c, count in concept_detections.items() if count >= min_detections]


def load_concept_vectors(
    exp21_dir: str,
    model_name: str,
    concepts: List[str],
    steering_layer: int = DEFAULT_LAYER,
) -> Dict[str, torch.Tensor]:
    """Load concept vectors from exp21."""
    base_vectors_dir = Path(exp21_dir) / model_name / "vectors"

    # Find best layer directory (match by actual layer number)
    layer_dirs = sorted(base_vectors_dir.glob("layer_*"))
    if layer_dirs:
        best_dir = min(layer_dirs,
                       key=lambda d: abs(int(d.name.replace("layer_", "")) - steering_layer))
        vectors_dir = best_dir
        print(f"  Loading vectors from: {vectors_dir.name}")
    else:
        vectors_dir = base_vectors_dir

    vectors = {}
    for concept in concepts:
        vec_path = vectors_dir / f"{concept}.pt"
        if vec_path.exists():
            vectors[concept] = torch.load(vec_path, weights_only=True)
        else:
            print(f"  Warning: Missing vector for {concept}")

    return vectors


def load_concept_groups(
    steering_layer: int = DEFAULT_LAYER,
    strength: float = DEFAULT_STRENGTH,
    exp4_dir: str = "analysis/exp4_vector_geometry",
    model_name: str = DEFAULT_MODEL,
) -> Tuple[List[str], List[str]]:
    """Load success/failure concept groups from exp4 subspace analysis.

    Returns:
        (success_concepts, failure_concepts) -- lists of concept names.
    """
    base = Path(exp4_dir) / model_name / f"layer_{steering_layer}_strength_{strength}"
    sa_path = base / "subspace_analysis.json"
    if not sa_path.exists():
        print(f"  Warning: exp4 subspace analysis not found at {sa_path}")
        return [], []
    with open(sa_path) as f:
        data = json.load(f)
    success = data.get("success_concepts", [])
    failure = data.get("failure_concepts", [])
    return success, failure


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_messages(trial_num: int = 1) -> List[Dict[str, str]]:
    """Build standard introspection prompt."""
    preamble = (
        "I am an interpretability researcher studying transformer-based language models like you. "
        "I have access to the internals of your neural network. I have identified patterns in your "
        "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
        "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
        "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
    )
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": preamble},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"},
    ]


def format_forced_messages(
    messages: List[Dict[str, str]], tokenizer,
) -> Tuple[torch.Tensor, int, str]:
    """Format messages for forced identification (with prefilled assistant response).

    Uses the same messages as standard introspection but appends a prefill:
      'Yes, I detect an injected thought. The injected word is **"'
    The last-token logit then reflects the model's prediction for the concept word.
    """
    filtered = [m for m in messages if not (m["role"] == "system" and m["content"] == "")]
    formatted = tokenizer.apply_chat_template(filtered, tokenize=False, add_generation_prompt=True)
    formatted += 'Yes, I detect an injected thought. The injected word is **"'
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


def format_messages(messages: List[Dict[str, str]], tokenizer) -> Tuple[torch.Tensor, int, str]:
    """Format messages with chat template and tokenize."""
    filtered = [m for m in messages if not (m["role"] == "system" and m["content"] == "")]
    formatted = tokenizer.apply_chat_template(filtered, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


# ── Base model prompt formatting (no chat template) ─────────────────────────

_BASE_MODEL_PREAMBLE = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
)


def format_messages_base(trial_num: int, tokenizer) -> Tuple[torch.Tensor, int, str]:
    """Format introspection prompt for base model (no chat template)."""
    bos = getattr(tokenizer, 'bos_token', '') or ''
    raw_prompt = (
        f"User: {_BASE_MODEL_PREAMBLE}\n\n"
        f"Assistant: Ok.\n\n"
        f"User: Trial {trial_num}: Do you detect an injected thought? "
        f"If so, what is the injected thought about?\n\n"
        f"Assistant:"
    )
    formatted = bos + raw_prompt
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


def format_forced_messages_base(trial_num: int, tokenizer) -> Tuple[torch.Tensor, int, str]:
    """Format forced identification prompt for base model (no chat template)."""
    bos = getattr(tokenizer, 'bos_token', '') or ''
    raw_prompt = (
        f"User: {_BASE_MODEL_PREAMBLE}\n\n"
        f"Assistant: Ok.\n\n"
        f"User: Trial {trial_num}: Do you detect an injected thought? "
        f"If so, what is the injected thought about?\n\n"
        f'Assistant: Yes, I detect an injected thought. The injected word is **"'
    )
    formatted = bos + raw_prompt
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


def extract_concept_vectors_base_model(
    model_wrapper: ModelWrapper,
    concept_words: List[str],
    baseline_words: List[str],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Extract concept vectors for base model using mean baseline subtraction."""
    tokenizer = model_wrapper.tokenizer
    bos = getattr(tokenizer, 'bos_token', '') or ''

    def format_extraction_prompt(word: str) -> str:
        return f"User:\nTell me about {word}\n\nAssistant:\n"

    def get_activation(prompt: str) -> torch.Tensor:
        full_prompt = bos + prompt
        input_ids = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
        input_ids = input_ids.to(model_wrapper.model.device)
        with torch.no_grad():
            outputs = model_wrapper.model(input_ids, output_hidden_states=True)
            activation = outputs.hidden_states[layer_idx + 1][0, -1, :]
        return activation

    print(f"  Computing baseline mean from {len(baseline_words)} words...")
    baseline_activations = []
    for word in baseline_words:
        activation = get_activation(format_extraction_prompt(word))
        baseline_activations.append(activation)
    baseline_mean = torch.stack(baseline_activations).mean(dim=0)

    print(f"  Extracting vectors for {len(concept_words)} concepts...")
    concept_vectors = {}
    for i, word in enumerate(concept_words):
        concept_activation = get_activation(format_extraction_prompt(word))
        concept_vectors[word] = concept_activation - baseline_mean
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(concept_words)} extracted")

    return concept_vectors


def find_steering_start_position(tokenizer, formatted_prompt: str, trial_num: int) -> int:
    """Find token position where steering should start (at 'Trial X:')."""
    trial_marker = f"Trial {trial_num}:"
    trial_char_pos = formatted_prompt.find(trial_marker)
    if trial_char_pos == -1:
        return 0
    newline_pos = formatted_prompt.rfind("\n", 0, trial_char_pos)
    if newline_pos == -1:
        newline_pos = 0
    prefix = formatted_prompt[:newline_pos]
    prefix_tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    return len(prefix_tokens["input_ids"][0])


# Module-level prompt cache: avoids redundant tokenization across all sections.
# Key: (trial_num, use_forced), Value: (input_ids, prompt_len, formatted, steer_start)
_prompt_cache: Dict[Tuple[int, bool], Tuple[torch.Tensor, int, str, int]] = {}


def get_cached_prompt(
    trial_num: int, tokenizer, use_forced: bool = False,
) -> Tuple[torch.Tensor, int, str, int]:
    """Get tokenized prompt from cache, building it on first access.

    Returns (input_ids, prompt_len, formatted_string, steer_start_position).
    """
    key = (trial_num, use_forced)
    if key not in _prompt_cache:
        if _is_base_model:
            if use_forced:
                input_ids, prompt_len, formatted = format_forced_messages_base(trial_num, tokenizer)
            else:
                input_ids, prompt_len, formatted = format_messages_base(trial_num, tokenizer)
        else:
            messages = build_messages(trial_num)
            if use_forced:
                input_ids, prompt_len, formatted = format_forced_messages(messages, tokenizer)
            else:
                input_ids, prompt_len, formatted = format_messages(messages, tokenizer)
        steer_start = find_steering_start_position(tokenizer, formatted, trial_num)
        _prompt_cache[key] = (input_ids, prompt_len, formatted, steer_start)
    return _prompt_cache[key]


def get_yes_no_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Get token IDs for detection-positive and detection-negative variants.

    Returns lists of token IDs.  The logit gap is computed via logsumexp
    over these lists so the metric captures any variant the model uses.
    """
    yes_variants = [
        "Oh", "oh", "OH", " Oh", " oh",
        "Yes", "yes", "YES", " Yes", " yes", " YES",
        "Wow", "wow", "WOW", " Wow", " wow",
        "Absolutely", " Absolutely",
    ]
    no_variants = ["No", "no", "NO", " No", " no", " NO"]

    yes_ids = []
    for v in yes_variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        if len(tokens) == 1 and tokens[0] not in yes_ids:
            yes_ids.append(tokens[0])

    no_ids = []
    for v in no_variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        if len(tokens) == 1 and tokens[0] not in no_ids:
            no_ids.append(tokens[0])

    # Fallback: ensure at least one ID each
    if not yes_ids:
        yes_ids = [tokenizer.encode("Yes", add_special_tokens=False)[0]]
    if not no_ids:
        no_ids = [tokenizer.encode("No", add_special_tokens=False)[0]]

    return yes_ids, no_ids


def build_discriminative_token_set(
    exp21_dir: str,
    model_name: str,
    concepts: List[str],
    tokenizer,
    steering_layer: int = DEFAULT_LAYER,
    strength: float = 4.0,
    yes_threshold: float = 0.8,
    no_threshold: float = 0.2,
    min_total: int = 20,
) -> Tuple[List[int], List[int]]:
    """Build global discriminative YES/NO token sets from exp21 first-token data.

    For each first BPE token across all exp21 responses, computes:
        P(detected | first_token = t) = n_detected / (n_detected + n_not_detected)

    Tokens with P > yes_threshold become YES tokens; tokens with P < no_threshold
    become NO tokens. Only tokens with >= min_total total occurrences are considered.

    Falls back to get_yes_no_token_ids() if exp21 data is unavailable.
    """
    default_yes, default_no = get_yes_no_token_ids(tokenizer)

    # Load exp21 results
    base_dir = Path(exp21_dir) / model_name
    exact_dir = base_dir / f"layer_{steering_layer}_strength_{strength}"
    results_path = exact_dir / "results.json" if exact_dir.exists() else None

    if results_path is None or not results_path.exists():
        for subdir in sorted(base_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("layer_"):
                try:
                    parts = subdir.name.split("_")
                    layer_val = float(parts[1])
                    strength_val = float(parts[3]) if len(parts) > 3 else None
                    if abs(layer_val - steering_layer) < 0.5:
                        if strength_val is not None and abs(strength_val - strength) < 0.1:
                            candidate = subdir / "results.json"
                            if candidate.exists():
                                results_path = candidate
                                break
                except (ValueError, IndexError):
                    continue

    if results_path is None or not results_path.exists():
        print("  Warning: No exp21 results found for discriminative tokens, using defaults")
        return default_yes, default_no

    with open(results_path) as f:
        data = json.load(f)

    # Count first-token occurrences across ALL exp21 concepts (global statistics)
    global_det = Counter()
    global_nodet = Counter()

    for r in data.get("results", []):
        if r.get("trial_type") != "injection":
            continue
        resp = r.get("response", "").strip()
        if not resp:
            continue
        tokens = tokenizer.encode(resp, add_special_tokens=False)
        if not tokens:
            continue
        first_tid = tokens[0]
        detected = r.get("evaluations", {}).get("claims_detection", {}).get(
            "claims_detection", False
        )
        if detected:
            global_det[first_tid] += 1
        else:
            global_nodet[first_tid] += 1

    # Compute P(detected | first_token = t) for tokens with enough data
    all_tids = set(global_det.keys()) | set(global_nodet.keys())
    yes_ids = []
    no_ids = []
    for tid in all_tids:
        d = global_det.get(tid, 0)
        nd = global_nodet.get(tid, 0)
        if d + nd < min_total:
            continue
        p_det = d / (d + nd)
        if p_det > yes_threshold:
            yes_ids.append(tid)
        elif p_det < no_threshold:
            no_ids.append(tid)

    if not yes_ids or not no_ids:
        print("  Warning: Not enough discriminative tokens found, using defaults")
        return default_yes, default_no

    # Sort by discrimination strength for readability
    yes_ids.sort(key=lambda t: global_det.get(t, 0) / max(global_det.get(t, 0) + global_nodet.get(t, 0), 1), reverse=True)
    no_ids.sort(key=lambda t: global_det.get(t, 0) / max(global_det.get(t, 0) + global_nodet.get(t, 0), 1))

    print(f"  Discriminative tokens: {len(yes_ids)} YES (P>{yes_threshold}), "
          f"{len(no_ids)} NO (P<{no_threshold})")
    for tid in yes_ids[:5]:
        d, nd = global_det.get(tid, 0), global_nodet.get(tid, 0)
        print(f"    YES: {tokenizer.decode([tid])!r:15s} P(det)={d/(d+nd):.2f} ({d+nd} total)")
    for tid in no_ids[:5]:
        d, nd = global_det.get(tid, 0), global_nodet.get(tid, 0)
        print(f"    NO:  {tokenizer.decode([tid])!r:15s} P(det)={d/(d+nd):.2f} ({d+nd} total)")

    return yes_ids, no_ids


def build_detection_weights(
    exp21_dir: str,
    model_name: str,
    tokenizer,
    steering_layer: int = DEFAULT_LAYER,
    strength: float = 4.0,
    min_total: int = 5,
) -> Tuple[List[int], torch.Tensor]:
    """Build per-token detection probability weights from exp21 first-token data.

    For each first BPE token observed in exp21 responses, computes:
        w(t) = P(detected | first_token = t)

    The detection log-odds is then computed via the law of total probability:
        P(det) = sum_t softmax(logits)[t] * w(t)
        log-odds = log(P(det) / (1 - P(det)))

    Returns:
        (scored_ids, weights): token IDs and their P(detected) weights.
        Falls back to simple YES=1/NO=0 weights if exp21 data is unavailable.
    """
    # Load exp21 results
    base_dir = Path(exp21_dir) / model_name
    exact_dir = base_dir / f"layer_{steering_layer}_strength_{strength}"
    results_path = exact_dir / "results.json" if exact_dir.exists() else None

    if results_path is None or not results_path.exists():
        for subdir in sorted(base_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("layer_"):
                try:
                    parts = subdir.name.split("_")
                    layer_val = float(parts[1])
                    strength_val = float(parts[3]) if len(parts) > 3 else None
                    if abs(layer_val - steering_layer) < 0.5:
                        if strength_val is not None and abs(strength_val - strength) < 0.1:
                            candidate = subdir / "results.json"
                            if candidate.exists():
                                results_path = candidate
                                break
                except (ValueError, IndexError):
                    continue

    if _is_base_model or results_path is None or not results_path.exists():
        if _is_base_model:
            print("  Base model: using default YES/NO bundle for detection weights")
        else:
            print("  Warning: No exp21 results found for detection weights, using fallback")
        yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
        scored_ids = yes_ids + no_ids
        weights = torch.tensor([1.0] * len(yes_ids) + [0.0] * len(no_ids))
        return scored_ids, weights

    with open(results_path) as f:
        data = json.load(f)

    global_det = Counter()
    global_nodet = Counter()
    for r in data.get("results", []):
        if r.get("trial_type") != "injection":
            continue
        resp = r.get("response", "").strip()
        if not resp:
            continue
        tokens = tokenizer.encode(resp, add_special_tokens=False)
        if not tokens:
            continue
        first_tid = tokens[0]
        detected = r.get("evaluations", {}).get("claims_detection", {}).get(
            "claims_detection", False
        )
        if detected:
            global_det[first_tid] += 1
        else:
            global_nodet[first_tid] += 1

    scored_ids = []
    weight_vals = []
    for tid in set(global_det.keys()) | set(global_nodet.keys()):
        d = global_det.get(tid, 0)
        nd = global_nodet.get(tid, 0)
        if d + nd < min_total:
            continue
        scored_ids.append(tid)
        weight_vals.append(d / (d + nd))

    if not scored_ids:
        print("  Warning: No scored tokens found, using fallback")
        yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
        scored_ids = yes_ids + no_ids
        weights = torch.tensor([1.0] * len(yes_ids) + [0.0] * len(no_ids))
        return scored_ids, weights

    weights = torch.tensor(weight_vals, dtype=torch.float32)
    print(f"  Detection weights: {len(scored_ids)} scored tokens (min_total={min_total})")
    return scored_ids, weights


def get_head_dim(model_wrapper: ModelWrapper) -> int:
    """Get attention head dimension."""
    config = model_wrapper.model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    if hasattr(config, 'head_dim'):
        return config.head_dim
    return model_wrapper.d_model // model_wrapper.n_heads


def collect_forced_control_activations(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    n_trials: int,
    device: str = "cuda",
    verbose: bool = False,
    ablation_mode: str = "all-positions",
) -> Dict[Tuple[str, int, int], torch.Tensor]:
    """Collect component outputs from unsteered forced-identification runs.

    Since there is no steering, outputs depend only on the trial prompt.
    Only n_trials forward passes needed.

    Returns:
        Dict mapping ('attn'|'mlp', layer_idx, trial_num) -> output tensor on CPU.
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    target_layers = list(range(steering_layer + 1, n_layers))
    control_means = {}
    capture_full_seq = (ablation_mode == "all-positions")

    for trial_idx in tqdm(range(n_trials), desc="Forced control activations", disable=not verbose):
        trial_num = trial_idx + 1
        input_ids, _, _, _ = get_cached_prompt(trial_num, tokenizer, use_forced=True)
        input_ids = input_ids.to(device)

        captured = {}
        hooks = []

        def make_capture_hook(comp_type, l_idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                if capture_full_seq:
                    captured[(comp_type, l_idx)] = out[0, :, :].detach().cpu()  # [seq, d_model]
                else:
                    captured[(comp_type, l_idx)] = out[0, -1, :].detach().cpu()  # [d_model]
            return hook

        for l_idx in target_layers:
            layer_module = model_wrapper.get_layer_module(l_idx)
            h_attn = layer_module.self_attn.register_forward_hook(make_capture_hook("attn", l_idx))
            h_mlp = layer_module.mlp.register_forward_hook(make_capture_hook("mlp", l_idx))
            hooks.extend([h_attn, h_mlp])

        try:
            with torch.no_grad():
                model(input_ids=input_ids)
            for (comp_type, l_idx), val in captured.items():
                control_means[(comp_type, l_idx, trial_num)] = val
        finally:
            for h in hooks:
                h.remove()
            captured.clear()
            torch.cuda.empty_cache()

    return control_means


def collect_component_control_activations(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    n_trials: int,
    device: str = "cuda",
    verbose: bool = False,
    ablation_mode: str = "all-positions",
) -> Dict[Tuple[str, int, int], torch.Tensor]:
    """Collect component outputs (attn and mlp) from unsteered runs.

    Since there is no steering, outputs depend only on the trial prompt.
    Only n_trials forward passes needed (not n_concepts x n_trials).

    Returns:
        Dict mapping ('attn'|'mlp', layer_idx, trial_num) -> output tensor on CPU.
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    target_layers = list(range(steering_layer + 1, n_layers))
    control_means = {}
    capture_full_seq = (ablation_mode == "all-positions")

    for trial_idx in tqdm(range(n_trials), desc="Control activations", disable=not verbose):
        trial_num = trial_idx + 1
        input_ids, _, _, _ = get_cached_prompt(trial_num, tokenizer, use_forced=False)
        input_ids = input_ids.to(device)

        captured = {}
        hooks = []

        def make_capture_hook(comp_type, l_idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                if capture_full_seq:
                    captured[(comp_type, l_idx)] = out[0, :, :].detach().cpu()  # [seq, d_model]
                else:
                    captured[(comp_type, l_idx)] = out[0, -1, :].detach().cpu()  # [d_model]
            return hook

        for l_idx in target_layers:
            layer_module = model_wrapper.get_layer_module(l_idx)
            h_attn = layer_module.self_attn.register_forward_hook(make_capture_hook("attn", l_idx))
            h_mlp = layer_module.mlp.register_forward_hook(make_capture_hook("mlp", l_idx))
            hooks.extend([h_attn, h_mlp])

        try:
            with torch.no_grad():
                model(input_ids=input_ids)
            for (comp_type, l_idx), val in captured.items():
                control_means[(comp_type, l_idx, trial_num)] = val
        finally:
            for h in hooks:
                h.remove()
            captured.clear()
            torch.cuda.empty_cache()

    return control_means


def collect_component_steered_activations(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    strength: float,
    n_trials: int,
    device: str = "cuda",
    verbose: bool = False,
    ablation_mode: str = "all-positions",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[Tuple[str, int, str, int], torch.Tensor]:
    """Collect component outputs from steered runs, keyed by concept AND trial.

    Batched implementation: all concepts are processed in one forward pass per
    trial (or in batches if N > batch_size).

    Returns:
        Dict mapping ('attn'|'mlp', layer_idx, concept, trial_num) -> output tensor on CPU.
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    target_layers = list(range(steering_layer + 1, n_layers))
    concepts = list(concept_vectors.keys())
    N = len(concepts)
    capture_full_seq = (ablation_mode == "all-positions")

    # Pre-stack steering vectors
    all_steer_vecs = torch.stack([concept_vectors[c] for c in concepts]).to(device)  # [N, d_model]

    steered_acts = {}

    for trial_idx in tqdm(range(n_trials), desc="Collecting steered activations (batched)", disable=not verbose):
        trial_num = trial_idx + 1
        input_ids, _, _, steer_start = get_cached_prompt(trial_num, tokenizer, use_forced=False)

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            B = batch_end - batch_start
            batch_concepts = concepts[batch_start:batch_end]
            batch_vecs = all_steer_vecs[batch_start:batch_end]

            batch_ids = input_ids.expand(B, -1).to(device)

            captured = {}  # (comp_type, l_idx) -> [B, seq, d_model] or [B, d_model]
            hooks = []

            # Batched steering hook
            def make_batched_steering_hook(start_pos, vecs, s):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    n = hidden.shape[0]
                    v = vecs[:n].to(hidden.dtype)
                    seq_len = hidden.shape[1]
                    if seq_len == 1:
                        hidden = hidden + s * v.unsqueeze(1)
                    elif start_pos < seq_len:
                        hidden = hidden.clone()
                        hidden[:, start_pos:, :] = hidden[:, start_pos:, :] + s * v.unsqueeze(1)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            layer_module = model_wrapper.get_layer_module(steering_layer)
            h_steer = layer_module.register_forward_hook(
                make_batched_steering_hook(steer_start, batch_vecs, strength)
            )
            hooks.append(h_steer)

            # Capture hooks for each post-steering layer
            def make_capture_hook(comp_type, l_idx):
                def hook(module, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if capture_full_seq:
                        captured[(comp_type, l_idx)] = out.detach().cpu()  # [B, seq, d_model]
                    else:
                        captured[(comp_type, l_idx)] = out[:, -1, :].detach().cpu()  # [B, d_model]
                return hook

            for l_idx in target_layers:
                lm = model_wrapper.get_layer_module(l_idx)
                h_attn = lm.self_attn.register_forward_hook(make_capture_hook("attn", l_idx))
                h_mlp = lm.mlp.register_forward_hook(make_capture_hook("mlp", l_idx))
                hooks.extend([h_attn, h_mlp])

            try:
                with torch.no_grad():
                    model(input_ids=batch_ids)

                # Split batch dimension into per-concept entries
                for (comp_type, l_idx), val in captured.items():
                    for j, concept in enumerate(batch_concepts):
                        steered_acts[(comp_type, l_idx, concept, trial_num)] = val[j]
            finally:
                for h in hooks:
                    h.remove()
                captured.clear()
                del batch_ids
                torch.cuda.empty_cache()

    return steered_acts


# ─────────────────────────────────────────────────────────────────────────────
# Batched forward pass utilities
# ─────────────────────────────────────────────────────────────────────────────

def batched_forward(
    model_wrapper: ModelWrapper,
    tokenizer,
    trial_num: int,
    steering_vecs: Optional[torch.Tensor],
    steering_layer: int,
    steering_strength: float,
    device: str = "cuda",
    extra_hooks: Optional[List] = None,
    use_forced_prompt: bool = False,
    override_batch_size: Optional[int] = None,
    select_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Batched forward pass returning logits at last position.

    All items use the same trial prompt. Different steering vectors per item.

    Args:
        steering_vecs: [B, d_model] tensor on device, or None for no steering.
        select_token_ids: If provided, only return logits for these token IDs.

    Returns:
        logits: [B, vocab_size] tensor on CPU (or [B, K] if select_token_ids given).
    """
    model = model_wrapper.model
    input_ids, _, _, steer_start = get_cached_prompt(trial_num, tokenizer, use_forced=use_forced_prompt)

    B = override_batch_size if override_batch_size is not None else (steering_vecs.shape[0] if steering_vecs is not None else 1)
    batch_ids = input_ids.expand(B, -1).to(device)

    hooks = []
    try:
        if steering_vecs is not None:

            def make_batched_steering_hook(start_pos, vecs, s):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    n = hidden.shape[0]
                    v = vecs[:n].to(hidden.dtype)
                    seq_len = hidden.shape[1]
                    if seq_len == 1:
                        hidden = hidden + s * v[:, None, :]
                    elif start_pos < seq_len:
                        hidden = hidden.clone()
                        hidden[:, start_pos:, :] += s * v[:, None, :]
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            layer_module = model_wrapper.get_layer_module(steering_layer)
            h = layer_module.register_forward_hook(
                make_batched_steering_hook(steer_start, steering_vecs, steering_strength)
            )
            hooks.append(h)

        if extra_hooks:
            for module, hook_fn, hook_type in extra_hooks:
                if hook_type == "forward":
                    h = module.register_forward_hook(hook_fn)
                elif hook_type == "forward_pre":
                    h = module.register_forward_pre_hook(hook_fn)
                else:
                    h = module.register_full_backward_hook(hook_fn)
                hooks.append(h)

        with torch.no_grad():
            outputs = model(input_ids=batch_ids)
            last_logits = outputs.logits[:, -1, :]  # [B, V] on GPU
            if select_token_ids is not None:
                logits = last_logits[:, select_token_ids].cpu()  # [B, K]
            else:
                logits = last_logits.cpu()  # [B, V]

    finally:
        for h in hooks:
            h.remove()

    return logits


def make_batched_ablation_hook(
    model_wrapper: ModelWrapper,
    component_type: str,
    layer_idx: int,
    values: torch.Tensor,
    device: str = "cuda",
    ablation_mode: str = "all-positions",
) -> Tuple:
    """Create batched ablation hook for a single layer's component.

    Args:
        values: Replacement activation tensor. Shape depends on mode and direction:
            broadcast/last-token knockout: [d_model]
            broadcast/last-token knockin:  [B, d_model]
            all-positions knockout:        [seq, d_model]
            all-positions knockin:         [B, seq, d_model]
        ablation_mode: 'all-positions', 'last-token', or 'broadcast'.

    Returns:
        (module, hook_fn, 'forward') tuple.
    """
    layer_module = model_wrapper.get_layer_module(layer_idx)
    target = layer_module.self_attn if component_type == "attn" else layer_module.mlp
    val = values.to(device)

    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        # out: [B, seq, d_model]

        if ablation_mode == "broadcast":
            if val.dim() == 1:  # [d_model] -- knockout
                replaced = val.to(out.dtype).unsqueeze(0).unsqueeze(0).expand_as(out)
            else:  # [B, d_model] -- knockin
                n = out.shape[0]
                replaced = val[:n].to(out.dtype).unsqueeze(1).expand_as(out)

        elif ablation_mode == "last-token":
            replaced = out.clone()
            if val.dim() == 1:  # [d_model] -- knockout
                replaced[:, -1, :] = val.to(out.dtype)
            else:  # [B, d_model] -- knockin
                n = out.shape[0]
                replaced[:, -1, :] = val[:n].to(out.dtype)

        else:  # all-positions: position-by-position replacement
            replaced = out.clone()
            if val.dim() == 3:
                # [B, seq, d_model] -- knockin (per-item full-sequence)
                n = out.shape[0]
                min_seq = min(val.shape[1], out.shape[1])
                replaced[:n, :min_seq, :] = val[:n, :min_seq, :].to(out.dtype)
            elif val.dim() == 2:
                # [seq, d_model] -- knockout (uniform full-sequence)
                min_seq = min(val.shape[0], out.shape[1])
                replaced[:, :min_seq, :] = val[:min_seq].to(out.dtype).unsqueeze(0).expand(out.shape[0], -1, -1)
            elif val.dim() == 1:
                # [d_model] -- fallback to last-token
                replaced[:, -1, :] = val.to(out.dtype)

        if isinstance(output, tuple):
            return (replaced,) + output[1:]
        return replaced

    return (target, hook, "forward")


def compute_component_gradient_attribution(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    steering_strength: float,
    n_trials: int = 10,
    device: str = "cuda",
    verbose: bool = False,
    exp21_dir: str = DEFAULT_EXP21_DIR,
    model_name: str = DEFAULT_MODEL,
    completed_concepts: Optional[Set[str]] = None,
    batch_size: int = GRADIENT_BATCH_SIZE,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Compute per-layer per-component gradient attribution (attn vs MLP).

    Batched implementation: all concepts are processed in one forward pass per
    trial.  Uses autograd.grad (not backward) to avoid accumulating gradients
    on all model parameters, saving ~54 GB of VRAM on a 27B model.

    Targets (detection prompt):
        discriminative  -- logsumexp(disc_yes_ids) - logsumexp(disc_no_ids)
        default_bundle  -- logsumexp(default_yes) - logsumexp(default_no)
        single_yes_no   -- logit("Yes") - logit("No")
        p_yes           -- sum(softmax(logits)[disc_yes_ids])
        entropy         -- -sum(softmax * log_softmax)

    Target (forced prompt, separate forward pass):
        forced          -- logit of concept token (first BPE token)

    Attribution = ||grad||_2 at last token position, per batch item.

    Returns:
        {target_name: {comp_layer: {concept: [trial_values]}}}
        where comp_layer is like "attn_38", "mlp_38", etc.
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    target_layers = list(range(steering_layer + 1, n_layers))

    # Build token sets for all targets
    yes_ids, no_ids = build_discriminative_token_set(
        exp21_dir, model_name, list(concept_vectors.keys()), tokenizer,
        steering_layer=steering_layer, strength=steering_strength,
    )
    yes_default, no_default = get_yes_no_token_ids(tokenizer)
    single_yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    single_no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    concept_token_map = _extract_concept_token_ids(list(concept_vectors.keys()), tokenizer)

    yes_ids_t = torch.tensor(yes_ids, device=device)
    no_ids_t = torch.tensor(no_ids, device=device)
    yes_default_t = torch.tensor(yes_default, device=device)
    no_default_t = torch.tensor(no_default, device=device)

    # Component labels
    comp_labels = []
    for l_idx in target_layers:
        comp_labels.append(f"attn_{l_idx}")
        comp_labels.append(f"mlp_{l_idx}")

    # Detection targets
    detection_targets = ["discriminative", "default_bundle", "single_yes_no", "p_yes", "entropy"]

    # Initialize results structure
    all_targets = detection_targets + ["forced"]
    results = {}
    for target in all_targets:
        results[target] = {}
        for cl in comp_labels:
            results[target][cl] = {}

    completed = completed_concepts or set()
    concepts = [c for c in concept_vectors.keys() if c not in completed]

    if not concepts:
        print("  All concepts already completed (resume)")
        return results

    # Pre-stack steering vectors
    all_steer_vecs = torch.stack([concept_vectors[c] for c in concepts]).to(device)  # [N, d_model]
    # Pre-compute concept token IDs for forced pass
    concept_tids_list = [concept_token_map.get(c, []) for c in concepts]

    print(f"  Computing component gradient attribution (batched) for {len(concepts)} concepts x {n_trials} trials...")
    print(f"  Targets: {', '.join(all_targets)}")
    print(f"  Components: {len(comp_labels)} (attn + mlp x {len(target_layers)} layers)")
    print(f"  Batch size: {batch_size}")

    N = len(concepts)

    for trial_idx in range(n_trials):
        trial_num = trial_idx + 1
        print(f"  Trial {trial_num}/{n_trials}...")

        # Process concepts in batches
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            B = batch_end - batch_start
            batch_concepts = concepts[batch_start:batch_end]
            batch_vecs = all_steer_vecs[batch_start:batch_end]  # [B, d_model]

            # ── Detection targets (batched forward + multiple backward) ──
            input_ids, _, _, steer_start = get_cached_prompt(trial_num, tokenizer, use_forced=False)
            batch_ids = input_ids.expand(B, -1).to(device)

            component_outputs = {}  # (comp_type, layer) -> tensor [B, seq, d_model]
            hooks = []

            def make_capture_hook(comp_type, l_idx):
                def hook(module, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    component_outputs[(comp_type, l_idx)] = out
                return hook

            for l_idx in target_layers:
                layer_mod = model_wrapper.get_layer_module(l_idx)
                fh_attn = layer_mod.self_attn.register_forward_hook(make_capture_hook("attn", l_idx))
                fh_mlp = layer_mod.mlp.register_forward_hook(make_capture_hook("mlp", l_idx))
                hooks.extend([fh_attn, fh_mlp])

            # Batched steering hook
            def make_batched_steering_hook(start_pos, vecs, s):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    n = hidden.shape[0]
                    v = vecs[:n].to(hidden.dtype)
                    seq_len = hidden.shape[1]
                    if seq_len == 1:
                        hidden = hidden + s * v.unsqueeze(1)
                    elif start_pos < seq_len:
                        hidden = hidden.clone()
                        hidden[:, start_pos:, :] = hidden[:, start_pos:, :] + s * v.unsqueeze(1)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            steer_module = model_wrapper.get_layer_module(steering_layer)
            sh = steer_module.register_forward_hook(
                make_batched_steering_hook(steer_start, batch_vecs, steering_strength)
            )
            hooks.append(sh)

            try:
                model.eval()
                with torch.enable_grad():
                    outputs = model(input_ids=batch_ids)
                    logits = outputs.logits[:, -1, :]  # [B, vocab]

                    # Compute per-item targets, sum across batch for single backward
                    target_scalars = {}
                    target_scalars["discriminative"] = (
                        torch.logsumexp(logits[:, yes_ids_t], dim=1) -
                        torch.logsumexp(logits[:, no_ids_t], dim=1)
                    )  # [B]
                    target_scalars["default_bundle"] = (
                        torch.logsumexp(logits[:, yes_default_t], dim=1) -
                        torch.logsumexp(logits[:, no_default_t], dim=1)
                    )
                    target_scalars["single_yes_no"] = logits[:, single_yes_id] - logits[:, single_no_id]

                    probs = F.softmax(logits, dim=1)  # [B, vocab]
                    target_scalars["p_yes"] = probs[:, yes_ids_t].sum(dim=1)  # [B]

                    log_probs = F.log_softmax(logits, dim=1)
                    target_scalars["entropy"] = -(probs * log_probs).sum(dim=1)  # [B]

                    # Use autograd.grad (not backward) to avoid 54GB param gradient overhead
                    output_keys = list(component_outputs.keys())
                    output_tensors = [component_outputs[k] for k in output_keys]

                    for t_idx, target_name in enumerate(detection_targets):
                        per_item = target_scalars[target_name]  # [B]
                        total_scalar = per_item.sum()
                        retain = (t_idx < len(detection_targets) - 1)
                        grads = torch.autograd.grad(
                            total_scalar, output_tensors,
                            retain_graph=retain, allow_unused=True,
                        )

                        # Extract per-item attributions (gradient norm at last position)
                        for (comp_type, l_idx), grad_tensor in zip(output_keys, grads):
                            if grad_tensor is not None:
                                grad = grad_tensor[:, -1, :]   # [B, d_model]
                                attrs = grad.norm(dim=1)  # [B] -- L2 sensitivity
                                cl = f"{comp_type}_{l_idx}"
                                for j, concept in enumerate(batch_concepts):
                                    if concept not in results[target_name][cl]:
                                        results[target_name][cl][concept] = []
                                    results[target_name][cl][concept].append(attrs[j].item())

            finally:
                for h in hooks:
                    h.remove()
                component_outputs.clear()
                try:
                    del outputs, batch_ids
                except NameError:
                    pass
                torch.cuda.empty_cache()

            # ── Forced target (batched forward + backward) ──
            batch_tids = concept_tids_list[batch_start:batch_end]
            valid_indices = [j for j, tids in enumerate(batch_tids) if tids]
            if valid_indices:
                forced_ids, _, _, steer_start_f = get_cached_prompt(trial_num, tokenizer, use_forced=True)
                B_forced = len(valid_indices)
                forced_batch_ids = forced_ids.expand(B_forced, -1).to(device)
                forced_vecs = batch_vecs[valid_indices]  # [B_forced, d_model]
                forced_concepts = [batch_concepts[j] for j in valid_indices]
                forced_tid_first = [batch_tids[j][0] for j in valid_indices]

                component_outputs_f = {}
                hooks_f = []

                def make_capture_hook_f(comp_type, l_idx):
                    def hook(module, args, output):
                        out = output[0] if isinstance(output, tuple) else output
                        component_outputs_f[(comp_type, l_idx)] = out
                    return hook

                for l_idx in target_layers:
                    layer_mod = model_wrapper.get_layer_module(l_idx)
                    fh_attn = layer_mod.self_attn.register_forward_hook(make_capture_hook_f("attn", l_idx))
                    fh_mlp = layer_mod.mlp.register_forward_hook(make_capture_hook_f("mlp", l_idx))
                    hooks_f.extend([fh_attn, fh_mlp])

                sh_f = steer_module.register_forward_hook(
                    make_batched_steering_hook(steer_start_f, forced_vecs, steering_strength)
                )
                hooks_f.append(sh_f)

                try:
                    with torch.enable_grad():
                        outputs_f = model(input_ids=forced_batch_ids)
                        logits_f = outputs_f.logits[:, -1, :]  # [B_forced, vocab]
                        forced_tid_t = torch.tensor(forced_tid_first, device=device)
                        per_item_forced = logits_f[torch.arange(B_forced, device=device), forced_tid_t]
                        total_forced = per_item_forced.sum()

                        output_keys_f = list(component_outputs_f.keys())
                        output_tensors_f = [component_outputs_f[k] for k in output_keys_f]
                        grads_f = torch.autograd.grad(
                            total_forced, output_tensors_f, allow_unused=True,
                        )

                        for (comp_type, l_idx), grad_tensor in zip(output_keys_f, grads_f):
                            if grad_tensor is not None:
                                grad = grad_tensor[:, -1, :]
                                attrs = grad.norm(dim=1)  # [B] -- L2 sensitivity
                                cl = f"{comp_type}_{l_idx}"
                                for j, concept in enumerate(forced_concepts):
                                    if concept not in results["forced"][cl]:
                                        results["forced"][cl][concept] = []
                                    results["forced"][cl][concept].append(attrs[j].item())

                finally:
                    for h in hooks_f:
                        h.remove()
                    component_outputs_f.clear()
                    try:
                        del outputs_f, forced_batch_ids
                    except NameError:
                        pass
                    torch.cuda.empty_cache()

    return results


def bootstrap_ci(data: List[float], n_bootstrap: int = 5000, ci: float = 0.95,
                  seed: int = 42) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, ci_low, ci_high)."""
    data = np.array(data)
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    alpha = (1 - ci) / 2
    low = means[int(alpha * n_bootstrap)]
    high = means[int((1 - alpha) * n_bootstrap)]
    return float(np.mean(data)), float(low), float(high)


# ─────────────────────────────────────────────────────────────────────────────
# Section A: Component-Type Knockout
# ─────────────────────────────────────────────────────────────────────────────

def _get_singular_plural_variants(word: str) -> set:
    """Generate singular/plural variants of a word."""
    variants = {word}
    w = word.lower()
    variants.add(w)
    if w.endswith('ies') and len(w) > 4:
        variants.add(w[:-3] + 'y')
        variants.add(word[:-3] + 'y')
    elif w.endswith('oes'):
        variants.add(w[:-2])
        variants.add(word[:-2])
        variants.add(w[:-1])
    elif w.endswith('ves'):
        variants.add(w[:-3] + 'f')
        variants.add(word[:-3] + 'f')
        variants.add(w[:-3] + 'fe')
        variants.add(word[:-3] + 'fe')
    elif w.endswith('ses') or w.endswith('xes') or w.endswith('zes') or w.endswith('ches') or w.endswith('shes'):
        variants.add(w[:-2])
        variants.add(word[:-2])
    elif w.endswith('s') and not w.endswith('ss'):
        variants.add(w[:-1])
        variants.add(word[:-1])
    if not w.endswith('s'):
        variants.add(w + 's')
        variants.add(word + 's')
    return variants


def _extract_concept_token_ids(concepts: List[str], tokenizer) -> Dict[str, List[int]]:
    """Pre-compute concept token IDs for forced identification.

    For each concept, generates singular/plural variants in original casing,
    lowercase, and capitalized forms.
    """
    concept_token_map = {}
    for concept in concepts:
        ids = set()
        for variant in _get_singular_plural_variants(concept):
            for form in [variant, variant.lower(), variant.capitalize()]:
                tokens = tokenizer.encode(form, add_special_tokens=False)
                if tokens:
                    ids.add(tokens[0])
                space_tokens = tokenizer.encode(" " + form, add_special_tokens=False)
                if space_tokens:
                    ids.add(space_tokens[0])
        concept_token_map[concept] = list(ids)
    return concept_token_map


def _batched_detection_gaps(
    logits: torch.Tensor,
    detection_ids: List[int],
    detection_weights: torch.Tensor,
) -> List[float]:
    """Compute detection log-odds from batched logits [B, V].

    Uses weighted first-token detection probabilities:
        P(det) = sum_t softmax(logits)[t] * P(detected | first_token = t)
        log-odds = log(P(det) / (1 - P(det)))
    """
    probs = torch.softmax(logits.float(), dim=-1)
    scored_probs = probs[:, detection_ids]  # [B, n_scored]
    w = detection_weights.to(probs.device)
    p_det = (scored_probs * w.unsqueeze(0)).sum(dim=-1)  # [B]
    log_odds = torch.log(p_det / (1 - p_det + 1e-10))
    return log_odds.tolist()


def _batched_concept_log_odds(
    logits: torch.Tensor, batch_concepts: List[str],
    concept_token_map: Dict[str, List[int]],
) -> List[float]:
    """Compute log-odds of concept tokens vs. rest from batched logits [B, V]."""
    scores = []
    V = logits.shape[-1]
    all_ids = torch.arange(V, device=logits.device)
    for i, concept in enumerate(batch_concepts):
        ids = concept_token_map[concept]
        ids_t = torch.tensor(ids, device=logits.device)
        log_p_concept = torch.logsumexp(logits[i, ids_t], dim=0)
        mask = torch.ones(V, dtype=torch.bool, device=logits.device)
        mask[ids_t] = False
        log_p_rest = torch.logsumexp(logits[i, mask], dim=0)
        scores.append((log_p_concept - log_p_rest).item())
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Multi-metric infrastructure
# ─────────────────────────────────────────────────────────────────────────────

MULTI_METRIC_NAMES = [
    "discriminative", "default_bundle", "single_yes_no",
    "kl_divergence", "js_divergence", "total_variation",
    "entropy", "p_yes", "top1_match",
]

METRIC_PLOT_CONFIG = {
    "discriminative": {
        "filename": "per_layer_ablation.png",
        "ylabel": "Detection log-odds",
        "ylabel_forced": "Identification log-odds",
        "ylabel_fontsize": 17,
    },
    "default_bundle": {
        "filename": "per_layer_ablation_default_bundle.png",
        "ylabel": "Log-odds: yes vs. no (default bundle)",
        "ylabel_forced": "Identification log-odds",
        "ylabel_fontsize": 15,
    },
    "single_yes_no": {
        "filename": "per_layer_ablation_single_yes_no.png",
        "ylabel": r"Logit difference: $\mathrm{logit}(\text{Yes}) - \mathrm{logit}(\text{No})$",
        "ylabel_forced": "Identification log-odds",
        "ylabel_fontsize": 15,
    },
    "kl_divergence": {
        "filename": "per_layer_ablation_kl_divergence.png",
        "ylabel": r"KL divergence from baseline: $D_{\mathrm{KL}}(P_{\mathrm{base}} \| P_{\mathrm{abl}})$",
        "ylabel_forced": None,
        "ylabel_fontsize": 15,
    },
    "js_divergence": {
        "filename": "per_layer_ablation_js_divergence.png",
        "ylabel": r"Jensen-Shannon divergence: $D_{\mathrm{JS}}(P_{\mathrm{base}} \| P_{\mathrm{abl}})$",
        "ylabel_forced": None,
        "ylabel_fontsize": 15,
    },
    "total_variation": {
        "filename": "per_layer_ablation_total_variation.png",
        "ylabel": r"Total variation distance: $\frac{1}{2}\|P_{\mathrm{base}} - P_{\mathrm{abl}}\|_1$",
        "ylabel_forced": None,
        "ylabel_fontsize": 15,
    },
    "entropy": {
        "filename": "per_layer_ablation_entropy.png",
        "ylabel": r"Output entropy: $H(P) = -\sum P \log P$",
        "ylabel_forced": None,
        "ylabel_fontsize": 15,
    },
    "p_yes": {
        "filename": "per_layer_ablation_p_yes.png",
        "ylabel": r"$P(\mathrm{detection}) = \sum_t P(t) \cdot w(t)$",
        "ylabel_forced": None,
        "ylabel_fontsize": 15,
    },
    "top1_match": {
        "filename": "per_layer_ablation_top1_match.png",
        "ylabel": "Top-1 token match rate with baseline",
        "ylabel_forced": None,
        "ylabel_fontsize": 14,
    },
}


def _compute_multi_metrics(
    logits: torch.Tensor,
    detection_ids: List[int],
    detection_weights: torch.Tensor,
    yes_default: List[int],
    no_default: List[int],
    single_yes_id: int,
    single_no_id: int,
    baseline_logprobs: Optional[torch.Tensor] = None,
    baseline_top1: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """Compute all metrics from batched logits [B, V].

    Returns:
        Dict mapping metric name to list of B float values.
    """
    metrics = {}

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    # 1. Discriminative: weighted detection log-odds (primary metric)
    w = detection_weights.to(probs.device)
    scored_probs = probs[:, detection_ids]  # [B, n_scored]
    p_det = (scored_probs * w.unsqueeze(0)).sum(dim=-1)  # [B]
    metrics["discriminative"] = torch.log(p_det / (1 - p_det + 1e-10)).tolist()

    # 2. Default bundle
    metrics["default_bundle"] = (
        torch.logsumexp(logits[:, yes_default], dim=1) -
        torch.logsumexp(logits[:, no_default], dim=1)
    ).tolist()

    # 3. Single Yes vs No
    metrics["single_yes_no"] = (
        logits[:, single_yes_id] - logits[:, single_no_id]
    ).tolist()

    # 4. Entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)
    metrics["entropy"] = entropy.tolist()

    # 5. P(YES)
    metrics["p_yes"] = p_det.tolist()

    # 6. Top-1 match
    if baseline_top1 is not None:
        current_top1 = logits.argmax(dim=-1)
        metrics["top1_match"] = (current_top1 == baseline_top1).float().tolist()

    # 7-9. Distribution-based metrics (need baseline)
    if baseline_logprobs is not None:
        log_probs_abl = log_probs
        bl = baseline_logprobs.float()  # [B, V]
        p_base = torch.exp(bl)

        # KL divergence
        kl = torch.sum(p_base * (bl - log_probs_abl), dim=-1)
        metrics["kl_divergence"] = kl.tolist()

        # JS divergence
        p_abl = probs
        m = 0.5 * (p_base + p_abl)
        log_m = torch.log(m + 1e-10)
        kl_pm = torch.sum(p_base * (bl - log_m), dim=-1)
        kl_qm = torch.sum(p_abl * (log_probs_abl - log_m), dim=-1)
        metrics["js_divergence"] = (0.5 * (kl_pm + kl_qm)).tolist()

        # Total variation
        tv = 0.5 * torch.sum(torch.abs(p_base - p_abl), dim=-1)
        metrics["total_variation"] = tv.tolist()

    return metrics


def run_section_a(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    strength: float,
    n_trials: int,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    verbose: bool = False,
    max_tokens: int = 100,
    temperature: float = 1.0,
    exp21_dir: str = DEFAULT_EXP21_DIR,
    model_name: str = DEFAULT_MODEL,
    ablation_mode: str = "all-positions",
) -> Dict:
    """Section A: Per-layer component mean ablation (batched).

    For every layer L after the steering layer, measures the effect of
    mean-ablating either attention or MLP in three directions:

      Knock-out:        Steered run, replace component with unsteered mean.
      Knock-in:         Unsteered run, replace component with steered value.
      Forced knock-out: Steered run + forced prompt, replace with unsteered mean.

    All concept forward passes within a trial are batched together for speed.
    """
    print("\n" + "=" * 70)
    print("SECTION A: Per-Layer Component Ablation (Batched)")
    print("=" * 70)
    print(f"  Batch size: {batch_size}")
    print(f"  Ablation mode: {ablation_mode}")

    section_dir = output_dir / "section_a"
    section_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = section_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    results_path = section_dir / "logit_gaps.json"
    multi_results_path = section_dir / "multi_metric_gaps.json"

    # Build detection weight vector from exp21 first-token data
    detection_ids, detection_weights = build_detection_weights(
        exp21_dir, model_name, tokenizer,
        steering_layer=steering_layer, strength=strength,
    )

    # Build additional token sets for multi-metric comparison
    yes_default, no_default = get_yes_no_token_ids(tokenizer)
    single_yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    single_no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"  Multi-metric tokens: detection weights ({len(detection_ids)} scored), "
          f"default ({len(yes_default)}Y/{len(no_default)}N), "
          f"single (Yes={single_yes_id}, No={single_no_id})")

    # Load success/failure concept groups for by-groups plot
    success_concepts, failure_concepts = load_concept_groups(
        steering_layer=steering_layer, strength=strength, model_name=model_name,
    )
    if success_concepts:
        print(f"  Concept groups: {len(success_concepts)} success, {len(failure_concepts)} failure")

    n_layers = model_wrapper.n_layers
    post_steering_layers = list(range(steering_layer + 1, n_layers))
    concepts = list(concept_vectors.keys())
    N = len(concepts)

    # Pre-stack steering vectors on device for batching
    all_steer_vecs = torch.stack([concept_vectors[c] for c in concepts]).to(device)  # [N, d_model]

    # Pre-compute concept token IDs for forced identification
    concept_token_map = _extract_concept_token_ids(concepts, tokenizer)

    # Load existing results for resume
    all_results = {}
    if results_path.exists() and not overwrite:
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results with {len(all_results)} conditions")

    # Load existing multi-metric results for resume
    multi_results = {}  # metric_name -> {cond_name -> {concept -> [trials]}}
    if multi_results_path.exists() and not overwrite:
        with open(multi_results_path) as f:
            multi_results = json.load(f)
        print(f"  Loaded existing multi-metric results ({len(multi_results)} metrics)")
    for m_name in MULTI_METRIC_NAMES:
        if m_name not in multi_results:
            multi_results[m_name] = {}

    # Baseline logprobs for KL/JS/TV
    baseline_logprobs_path = section_dir / "baseline_logprobs.pt"
    baseline_logprobs_dict = {}  # concept -> [V] tensor (CPU)
    baseline_top1_dict = {}  # concept -> int (argmax token ID)

    def save_results():
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    def save_multi_results():
        with open(multi_results_path, "w") as f:
            json.dump(multi_results, f, indent=2)

    # ── Helper: run batched condition ──
    def run_batched_condition(
        cond_name: str, do_steer: bool, use_forced: bool,
        ablation_comp=None, ablation_layer=None, ablation_values=None,
        metric="detection", collect_baseline_logprobs=False,
    ):
        """Run a single condition across all concepts and trials, batched."""
        if cond_name in all_results:
            if collect_baseline_logprobs and not baseline_logprobs_dict:
                pass  # Fall through to run it
            else:
                return
        print(f"\n  [{cond_name}] {N} concepts x {n_trials} trials (batched)...")
        condition_gaps = defaultdict(list)
        multi_cond = {m_name: defaultdict(list) for m_name in MULTI_METRIC_NAMES} if metric == "detection" else None
        bl_probs_accum = {} if collect_baseline_logprobs else None

        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1

            extra_hooks = None
            if ablation_comp is not None:
                if callable(ablation_values):
                    abl_val = ablation_values(trial_num)
                else:
                    abl_val = ablation_values
                extra_hooks = [make_batched_ablation_hook(
                    model_wrapper, ablation_comp, ablation_layer, abl_val, device=device,
                    ablation_mode=ablation_mode,
                )]

            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch_concepts = concepts[batch_start:batch_end]

                steer_vecs = all_steer_vecs[batch_start:batch_end] if do_steer else None

                if ablation_comp is not None and callable(ablation_values):
                    pass  # Already handled above
                elif ablation_comp is not None and ablation_values is not None and ablation_values.dim() == 2:
                    batch_abl = ablation_values[batch_start:batch_end]
                    extra_hooks = [make_batched_ablation_hook(
                        model_wrapper, ablation_comp, ablation_layer, batch_abl, device=device,
                        ablation_mode=ablation_mode,
                    )]

                logits = batched_forward(
                    model_wrapper, tokenizer, trial_num,
                    steer_vecs, steering_layer, strength,
                    device=device, extra_hooks=extra_hooks,
                    use_forced_prompt=use_forced,
                )

                if metric == "detection":
                    gaps = _batched_detection_gaps(logits, detection_ids, detection_weights)

                    batch_bl = None
                    batch_top1 = None
                    if baseline_logprobs_dict and not collect_baseline_logprobs:
                        batch_bl = torch.stack([
                            baseline_logprobs_dict[c] for c in batch_concepts
                        ]).to(logits.device)
                    if baseline_top1_dict and not collect_baseline_logprobs:
                        batch_top1 = torch.tensor([
                            baseline_top1_dict[c] for c in batch_concepts
                        ]).to(logits.device)
                    mm = _compute_multi_metrics(
                        logits, detection_ids, detection_weights, yes_default, no_default,
                        single_yes_id, single_no_id,
                        baseline_logprobs=batch_bl, baseline_top1=batch_top1,
                    )
                    for m_name in mm:
                        for i, c in enumerate(batch_concepts):
                            multi_cond[m_name][c].append(mm[m_name][i])

                    if collect_baseline_logprobs:
                        probs = torch.softmax(logits.float(), dim=-1).cpu()
                        for i, c in enumerate(batch_concepts):
                            if c not in bl_probs_accum:
                                bl_probs_accum[c] = probs[i]
                            else:
                                bl_probs_accum[c] = bl_probs_accum[c] + probs[i]
                else:
                    gaps = _batched_concept_log_odds(logits, batch_concepts, concept_token_map)

                for i, concept in enumerate(batch_concepts):
                    condition_gaps[concept].append(gaps[i])

        all_results[cond_name] = dict(condition_gaps)
        m = np.mean([np.mean(g) for g in condition_gaps.values()])
        print(f"  [{cond_name}] Done. Mean: {m:.3f}")
        save_results()

        if multi_cond is not None:
            for m_name in MULTI_METRIC_NAMES:
                if m_name in multi_cond and multi_cond[m_name]:
                    multi_results[m_name][cond_name] = dict(multi_cond[m_name])
            save_multi_results()

        if collect_baseline_logprobs and bl_probs_accum:
            for c in bl_probs_accum:
                mean_probs = bl_probs_accum[c] / n_trials
                baseline_logprobs_dict[c] = torch.log(mean_probs + 1e-10)
                baseline_top1_dict[c] = mean_probs.argmax().item()
            torch.save(baseline_logprobs_dict, baseline_logprobs_path)
            print(f"  Saved baseline logprobs for {len(baseline_logprobs_dict)} concepts")

    # ── No-steering baseline ──
    if "no_steering" not in all_results:
        print(f"\n  [no_steering] Running {n_trials} trials (concept-independent)...")
        condition_gaps = defaultdict(list)
        multi_nosteer = {m_name: defaultdict(list) for m_name in MULTI_METRIC_NAMES}
        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1
            logits = batched_forward(
                model_wrapper, tokenizer, trial_num,
                None, steering_layer, strength, device=device,
            )
            gap = _batched_detection_gaps(logits, detection_ids, detection_weights)[0]
            mm = _compute_multi_metrics(
                logits, detection_ids, detection_weights, yes_default, no_default,
                single_yes_id, single_no_id, baseline_logprobs=None,
            )
            for c in concepts:
                condition_gaps[c].append(gap)
                for m_name in mm:
                    multi_nosteer[m_name][c].append(mm[m_name][0])
        all_results["no_steering"] = dict(condition_gaps)
        for m_name in MULTI_METRIC_NAMES:
            if m_name in multi_nosteer and multi_nosteer[m_name]:
                multi_results[m_name]["no_steering"] = dict(multi_nosteer[m_name])
        m = np.mean([np.mean(g) for g in condition_gaps.values()])
        print(f"  [no_steering] Done. Mean gap: {m:.3f}")
        save_results()
        save_multi_results()
    else:
        print(f"  [no_steering] Already done, skipping")

    # ── Steered baseline (also collects baseline logprobs for KL) ──
    if baseline_logprobs_path.exists() and not baseline_logprobs_dict:
        baseline_logprobs_dict.update(torch.load(baseline_logprobs_path, weights_only=False))
        print(f"  Loaded cached baseline logprobs for {len(baseline_logprobs_dict)} concepts")
    run_batched_condition("baseline", do_steer=True, use_forced=False, metric="detection",
                          collect_baseline_logprobs=True)

    # ── Collect activations ──
    needs_knockout = any(f"knockout_attn_{l}" not in all_results for l in post_steering_layers)
    component_control = None
    if needs_knockout:
        print(f"\n  Collecting control activations (detection, T forward passes, mode={ablation_mode})...")
        component_control = collect_component_control_activations(
            model_wrapper, tokenizer, concept_vectors,
            steering_layer, n_trials=n_trials, device=device, verbose=verbose,
            ablation_mode=ablation_mode,
        )
        print(f"  Collected {len(component_control)} control activations")

    # ── Per-layer knock-out ──
    n_conditions = len(post_steering_layers) * 2
    done_count = 0
    for l_idx in post_steering_layers:
        for comp in ["attn", "mlp"]:
            done_count += 1
            ko_name = f"knockout_{comp}_{l_idx}"
            if ko_name not in all_results and component_control is not None:
                print(f"\n  [{ko_name}] {N}x{n_trials} ({done_count}/{n_conditions})...")
                condition_gaps = defaultdict(list)
                multi_cond = {m_name: defaultdict(list) for m_name in MULTI_METRIC_NAMES}
                for trial_idx in range(n_trials):
                    trial_num = trial_idx + 1
                    ctrl_act = component_control[(comp, l_idx, trial_num)]
                    abl_hook = make_batched_ablation_hook(
                        model_wrapper, comp, l_idx, ctrl_act, device=device,
                        ablation_mode=ablation_mode,
                    )
                    for bs in range(0, N, batch_size):
                        be = min(bs + batch_size, N)
                        batch_concepts = concepts[bs:be]
                        logits = batched_forward(
                            model_wrapper, tokenizer, trial_num,
                            all_steer_vecs[bs:be], steering_layer, strength,
                            device=device, extra_hooks=[abl_hook],
                        )
                        gaps = _batched_detection_gaps(logits, detection_ids, detection_weights)
                        batch_bl = None
                        batch_top1 = None
                        if baseline_logprobs_dict:
                            batch_bl = torch.stack([
                                baseline_logprobs_dict[c] for c in batch_concepts
                            ]).to(logits.device)
                        if baseline_top1_dict:
                            batch_top1 = torch.tensor([
                                baseline_top1_dict[c] for c in batch_concepts
                            ]).to(logits.device)
                        mm = _compute_multi_metrics(
                            logits, detection_ids, detection_weights, yes_default, no_default,
                            single_yes_id, single_no_id,
                            baseline_logprobs=batch_bl, baseline_top1=batch_top1,
                        )
                        for i, c in enumerate(batch_concepts):
                            condition_gaps[c].append(gaps[i])
                            for m_name in mm:
                                multi_cond[m_name][c].append(mm[m_name][i])
                all_results[ko_name] = dict(condition_gaps)
                for m_name in MULTI_METRIC_NAMES:
                    if m_name in multi_cond and multi_cond[m_name]:
                        multi_results[m_name][ko_name] = dict(multi_cond[m_name])
                m = np.mean([np.mean(g) for g in condition_gaps.values()])
                print(f"  [{ko_name}] Done. Mean: {m:.3f}")
                save_results()
                save_multi_results()

        # Plot after each knockout layer completes
        try:
            plot_section_a(all_results, post_steering_layers, steering_layer, plots_dir)
            plot_section_a_p_det(all_results, post_steering_layers, steering_layer, plots_dir)
            plot_section_a_all_metrics(multi_results, post_steering_layers, steering_layer, plots_dir,
                                       forced_results=all_results)
            if success_concepts:
                plot_section_a_by_groups(all_results, post_steering_layers, steering_layer,
                                         plots_dir, success_concepts, failure_concepts)
        except Exception as e:
            print(f"  Warning: per-layer plot update failed: {e}")

    del component_control
    gc.collect()
    torch.cuda.empty_cache()

    # ── Per-layer knock-in (chunked by concepts for memory efficiency) ──
    KNOCKIN_CHUNK_SIZE = 50 if ablation_mode == "all-positions" else N
    needs_knockin = any(f"knockin_attn_{l}" not in all_results for l in post_steering_layers)
    if needs_knockin:
        n_post_layers = len(post_steering_layers)
        if ablation_mode == "all-positions":
            if _is_base_model:
                test_ids, _, _ = format_messages_base(1, tokenizer)
            else:
                messages = build_messages(1)
                test_ids, _, _ = format_messages(messages, tokenizer)
            est_seq_len = test_ids.shape[1]
            bytes_per_chunk = (KNOCKIN_CHUNK_SIZE * n_trials * n_post_layers * 2
                               * est_seq_len * model_wrapper.d_model * 2)
            print(f"\n  [knockin] Memory estimate: ~{bytes_per_chunk / 1e9:.1f} GB per chunk "
                  f"({KNOCKIN_CHUNK_SIZE} concepts x {n_trials} trials x {n_post_layers} layers "
                  f"x 2 comps x {est_seq_len} seq x {model_wrapper.d_model} d_model x 2 bytes)")
        else:
            bytes_per_chunk = N * n_trials * n_post_layers * 2 * model_wrapper.d_model * 2
            print(f"\n  [knockin] Memory estimate: ~{bytes_per_chunk / 1e9:.1f} GB total "
                  f"(last-token mode, all concepts at once)")

        ki_partial_path = section_dir / "knockin_partial.json"
        ki_multi_partial_path = section_dir / "knockin_multi_partial.json"

        ki_accum = {}
        ki_multi_accum = {}
        completed_chunks = 0

        if ki_partial_path.exists() and not overwrite:
            with open(ki_partial_path) as f:
                ki_accum = json.load(f)
            ki_accum = {k: defaultdict(list, v) for k, v in ki_accum.items()}
            if ki_multi_partial_path.exists():
                with open(ki_multi_partial_path) as f:
                    ki_multi_accum = json.load(f)
                ki_multi_accum = {
                    k: {mn: defaultdict(list, mv) for mn, mv in v.items()}
                    for k, v in ki_multi_accum.items()
                }
            if ki_accum:
                any_key = next(iter(ki_accum))
                n_concepts_done = len(ki_accum[any_key])
                completed_chunks = (n_concepts_done + KNOCKIN_CHUNK_SIZE - 1) // KNOCKIN_CHUNK_SIZE
                print(f"  [knockin] Resuming: {n_concepts_done} concepts already processed "
                      f"({completed_chunks} chunks)")

        for l_idx in post_steering_layers:
            for comp in ["attn", "mlp"]:
                ki_name = f"knockin_{comp}_{l_idx}"
                if ki_name not in all_results:
                    if ki_name not in ki_accum:
                        ki_accum[ki_name] = defaultdict(list)
                    if ki_name not in ki_multi_accum:
                        ki_multi_accum[ki_name] = {m_name: defaultdict(list) for m_name in MULTI_METRIC_NAMES}

        n_chunks = (N + KNOCKIN_CHUNK_SIZE - 1) // KNOCKIN_CHUNK_SIZE
        for chunk_idx, chunk_start in enumerate(range(0, N, KNOCKIN_CHUNK_SIZE)):
            if chunk_idx < completed_chunks:
                continue

            chunk_end = min(chunk_start + KNOCKIN_CHUNK_SIZE, N)
            chunk_concepts = concepts[chunk_start:chunk_end]
            chunk_vectors = {c: concept_vectors[c] for c in chunk_concepts}

            print(f"\n  [knockin] Collecting steered activations for chunk {chunk_idx+1}/{n_chunks} "
                  f"({len(chunk_concepts)} concepts)...")
            steered_acts = collect_component_steered_activations(
                model_wrapper, tokenizer, chunk_vectors,
                steering_layer, strength, n_trials=n_trials, device=device, verbose=verbose,
                ablation_mode=ablation_mode, batch_size=batch_size,
            )
            print(f"  Collected {len(steered_acts)} steered activations for chunk")

            for l_idx in post_steering_layers:
                for comp in ["attn", "mlp"]:
                    ki_name = f"knockin_{comp}_{l_idx}"
                    if ki_name in ki_accum:
                        for trial_idx in range(n_trials):
                            trial_num = trial_idx + 1
                            for bs in range(0, len(chunk_concepts), batch_size):
                                be = min(bs + batch_size, len(chunk_concepts))
                                batch_concepts = chunk_concepts[bs:be]
                                batch_acts = torch.stack([
                                    steered_acts[(comp, l_idx, c, trial_num)] for c in batch_concepts
                                ])
                                abl_hook = make_batched_ablation_hook(
                                    model_wrapper, comp, l_idx, batch_acts, device=device,
                                    ablation_mode=ablation_mode,
                                )
                                logits = batched_forward(
                                    model_wrapper, tokenizer, trial_num,
                                    None, steering_layer, strength,
                                    device=device, extra_hooks=[abl_hook],
                                    override_batch_size=len(batch_concepts),
                                )
                                gaps = _batched_detection_gaps(logits, detection_ids, detection_weights)
                                batch_bl = None
                                batch_top1 = None
                                if baseline_logprobs_dict:
                                    batch_bl = torch.stack([
                                        baseline_logprobs_dict[c] for c in batch_concepts
                                    ]).to(logits.device)
                                if baseline_top1_dict:
                                    batch_top1 = torch.tensor([
                                        baseline_top1_dict[c] for c in batch_concepts
                                    ]).to(logits.device)
                                mm = _compute_multi_metrics(
                                    logits, detection_ids, detection_weights, yes_default, no_default,
                                    single_yes_id, single_no_id,
                                    baseline_logprobs=batch_bl, baseline_top1=batch_top1,
                                )
                                for i, c in enumerate(batch_concepts):
                                    ki_accum[ki_name][c].append(gaps[i])
                                    for m_name in mm:
                                        ki_multi_accum[ki_name][m_name][c].append(mm[m_name][i])

            del steered_acts
            gc.collect()
            torch.cuda.empty_cache()

            with open(ki_partial_path, "w") as f:
                json.dump({k: dict(v) for k, v in ki_accum.items()}, f)
            with open(ki_multi_partial_path, "w") as f:
                json.dump({k: {mn: dict(mv) for mn, mv in v.items()} for k, v in ki_multi_accum.items()}, f)
            print(f"  [knockin] Chunk {chunk_idx+1}/{n_chunks} saved (partial progress)")

        for ki_name in ki_accum:
            all_results[ki_name] = dict(ki_accum[ki_name])
            for m_name in MULTI_METRIC_NAMES:
                if ki_multi_accum[ki_name][m_name]:
                    multi_results[m_name][ki_name] = dict(ki_multi_accum[ki_name][m_name])
            m = np.mean([np.mean(g) for g in ki_accum[ki_name].values()])
            print(f"  [{ki_name}] Done. Mean: {m:.3f}")
        save_results()
        save_multi_results()
        del ki_accum, ki_multi_accum

        if ki_partial_path.exists():
            ki_partial_path.unlink()
        if ki_multi_partial_path.exists():
            ki_multi_partial_path.unlink()

        try:
            plot_section_a(all_results, post_steering_layers, steering_layer, plots_dir)
            plot_section_a_p_det(all_results, post_steering_layers, steering_layer, plots_dir)
            plot_section_a_all_metrics(multi_results, post_steering_layers, steering_layer, plots_dir,
                                       forced_results=all_results)
            if success_concepts:
                plot_section_a_by_groups(all_results, post_steering_layers, steering_layer,
                                         plots_dir, success_concepts, failure_concepts)
        except Exception as e:
            print(f"  Warning: knockin plot update failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    # ── Forced identification baselines ──
    if "forced_no_steering" not in all_results:
        print(f"\n  [forced_no_steering] Running {n_trials} trials (concept-independent)...")
        trial_logits = {}
        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1
            logits = batched_forward(
                model_wrapper, tokenizer, trial_num,
                None, steering_layer, strength, device=device,
                use_forced_prompt=True,
            )
            trial_logits[trial_num] = logits[0]  # [V], single item
        condition_gaps = {}
        V = trial_logits[1].shape[0]
        for c in concepts:
            scores = []
            ids = concept_token_map[c]
            ids_t = torch.tensor(ids, device=trial_logits[1].device)
            mask = torch.ones(V, dtype=torch.bool, device=trial_logits[1].device)
            mask[ids_t] = False
            for trial_num in range(1, n_trials + 1):
                lgt = trial_logits[trial_num]
                log_p_concept = torch.logsumexp(lgt[ids_t], dim=0)
                log_p_rest = torch.logsumexp(lgt[mask], dim=0)
                scores.append((log_p_concept - log_p_rest).item())
            condition_gaps[c] = scores
        all_results["forced_no_steering"] = condition_gaps
        m = np.mean([np.mean(g) for g in condition_gaps.values()])
        print(f"  [forced_no_steering] Done. Mean: {m:.3f}")
        save_results()
    else:
        print(f"  [forced_no_steering] Already done, skipping")

    # Forced baseline (steered): batched
    run_batched_condition("forced_baseline", do_steer=True, use_forced=True, metric="forced")

    # Forced control activations
    needs_forced = any(f"forced_knockout_attn_{l}" not in all_results for l in post_steering_layers)
    forced_control = None
    if needs_forced:
        print(f"\n  Collecting forced control activations (T forward passes, mode={ablation_mode})...")
        forced_control = collect_forced_control_activations(
            model_wrapper, tokenizer, concept_vectors,
            steering_layer, n_trials=n_trials, device=device, verbose=verbose,
            ablation_mode=ablation_mode,
        )
        print(f"  Collected {len(forced_control)} forced control activations")

    # Per-layer forced identification knock-out
    done_count = 0
    for l_idx in post_steering_layers:
        for comp in ["attn", "mlp"]:
            done_count += 1
            fk_name = f"forced_knockout_{comp}_{l_idx}"
            if fk_name not in all_results and forced_control is not None:
                print(f"\n  [{fk_name}] {N}x{n_trials} ({done_count}/{n_conditions})...")
                condition_gaps = defaultdict(list)
                for trial_idx in range(n_trials):
                    trial_num = trial_idx + 1
                    ctrl_act = forced_control[(comp, l_idx, trial_num)]
                    abl_hook = make_batched_ablation_hook(
                        model_wrapper, comp, l_idx, ctrl_act, device=device,
                        ablation_mode=ablation_mode,
                    )
                    for bs in range(0, N, batch_size):
                        be = min(bs + batch_size, N)
                        batch_concepts = concepts[bs:be]
                        logits = batched_forward(
                            model_wrapper, tokenizer, trial_num,
                            all_steer_vecs[bs:be], steering_layer, strength,
                            device=device, extra_hooks=[abl_hook],
                            use_forced_prompt=True,
                        )
                        scores = _batched_concept_log_odds(logits, batch_concepts, concept_token_map)
                        for i, c in enumerate(batch_concepts):
                            condition_gaps[c].append(scores[i])
                all_results[fk_name] = dict(condition_gaps)
                m = np.mean([np.mean(g) for g in condition_gaps.values()])
                print(f"  [{fk_name}] Done. Mean: {m:.3f}")
                save_results()

        try:
            plot_section_a(all_results, post_steering_layers, steering_layer, plots_dir)
            plot_section_a_p_det(all_results, post_steering_layers, steering_layer, plots_dir)
            plot_section_a_all_metrics(multi_results, post_steering_layers, steering_layer, plots_dir,
                                       forced_results=all_results)
            if success_concepts:
                plot_section_a_by_groups(all_results, post_steering_layers, steering_layer,
                                         plots_dir, success_concepts, failure_concepts)
        except Exception as e:
            print(f"  Warning: per-layer forced plot update failed: {e}")

    del forced_control
    gc.collect()
    torch.cuda.empty_cache()

    baseline_logprobs_dict.clear()

    # Compute metrics and plot
    metrics = compute_section_a_metrics(all_results, post_steering_layers)
    with open(section_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_section_a(all_results, post_steering_layers, steering_layer, plots_dir)
    plot_section_a_p_det(all_results, post_steering_layers, steering_layer, plots_dir)
    plot_section_a_all_metrics(multi_results, post_steering_layers, steering_layer, plots_dir,
                               forced_results=all_results)
    if success_concepts:
        plot_section_a_by_groups(all_results, post_steering_layers, steering_layer,
                                 plots_dir, success_concepts, failure_concepts)

    return {"logit_gaps": all_results, "metrics": metrics, "multi_metric_gaps": multi_results}


def compute_section_a_metrics(results: Dict, post_steering_layers: List[int]) -> Dict:
    """Compute aggregate metrics for Section A per-layer ablation."""
    metrics = {}
    baseline_gaps = results.get("baseline", {})
    no_steer_gaps = results.get("no_steering", {})

    for cond_name in ["baseline", "no_steering", "forced_baseline", "forced_no_steering"]:
        cond_gaps = results.get(cond_name, {})
        if not cond_gaps:
            continue
        concept_means = [np.mean(gaps) for gaps in cond_gaps.values()]
        if concept_means:
            mean, ci_low, ci_high = bootstrap_ci(concept_means)
        else:
            mean = ci_low = ci_high = 0
        metrics[cond_name] = {
            "mean_logit_gap": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "se": np.std(concept_means) / np.sqrt(len(concept_means)) if concept_means else 0,
        }

    for direction in ["knockout", "knockin", "forced_knockout"]:
        ref_name = {
            "knockout": "baseline",
            "knockin": "no_steering",
            "forced_knockout": "forced_baseline",
        }[direction]
        for comp in ["attn", "mlp"]:
            for l_idx in post_steering_layers:
                cond_name = f"{direction}_{comp}_{l_idx}"
                cond_gaps = results.get(cond_name, {})
                if not cond_gaps:
                    continue
                concept_means = [np.mean(gaps) for gaps in cond_gaps.values()]
                mean_gap = np.mean(concept_means)
                se = np.std(concept_means) / np.sqrt(len(concept_means))
                metrics[cond_name] = {
                    "mean_logit_gap": mean_gap,
                    "se": se,
                }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Section B: Per-Layer Component Gradient Attribution
# ─────────────────────────────────────────────────────────────────────────────

def run_section_b(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    strength: float,
    n_trials: int,
    output_dir: Path,
    device: str = "cuda",
    overwrite: bool = False,
    verbose: bool = False,
    exp21_dir: str = DEFAULT_EXP21_DIR,
    model_name: str = DEFAULT_MODEL,
) -> Dict:
    """Section B: Per-layer component gradient attribution (attn vs MLP).

    Computes gradient-based attribution for each layer's attention and MLP
    components across multiple targets.
    """
    print("\n" + "=" * 70)
    print("SECTION B: Per-Layer Component Gradient Attribution")
    print("=" * 70)

    section_dir = output_dir / "section_b"
    section_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = section_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    results_path = section_dir / "gradient_attribution.json"
    n_layers = model_wrapper.n_layers
    post_steering_layers = list(range(steering_layer + 1, n_layers))

    # Load existing results for resume
    existing_results = {}
    completed_concepts = set()
    if results_path.exists() and not overwrite:
        with open(results_path) as f:
            existing_results = json.load(f)
        first_target = next(iter(existing_results), None)
        if first_target:
            first_comp = next(iter(existing_results[first_target]), None)
            if first_comp:
                completed_concepts = set(existing_results[first_target][first_comp].keys())
        if completed_concepts:
            print(f"  Resuming: {len(completed_concepts)} concepts already done")

    # Compute attribution
    new_results = compute_component_gradient_attribution(
        model_wrapper, tokenizer, concept_vectors,
        steering_layer, strength, n_trials=n_trials,
        device=device, verbose=verbose,
        exp21_dir=exp21_dir, model_name=model_name,
        completed_concepts=completed_concepts,
    )

    # Merge new results into existing
    if existing_results:
        for target in new_results:
            if target not in existing_results:
                existing_results[target] = {}
            for comp_layer in new_results[target]:
                if comp_layer not in existing_results[target]:
                    existing_results[target][comp_layer] = {}
                existing_results[target][comp_layer].update(new_results[target][comp_layer])
        results = existing_results
    else:
        results = new_results

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved gradient attribution to {results_path}")

    success_concepts, failure_concepts = load_concept_groups(
        steering_layer=steering_layer, strength=strength, model_name=model_name,
    )

    plot_section_b(results, post_steering_layers, steering_layer, plots_dir)
    if success_concepts:
        plot_section_b_by_groups(results, post_steering_layers, steering_layer, plots_dir,
                                 success_concepts, failure_concepts)
    plot_section_b_all_metrics(results, post_steering_layers, steering_layer, plots_dir)

    return {"results": results}


def _plot_section_b_rows(axes, results, post_steering_layers, row_targets,
                          concept_filter=None):
    """Shared helper: plot Attention + MLP lines on each axis."""
    for row_idx, (target_name, title) in enumerate(row_targets):
        ax = axes[row_idx]
        target_data = results.get(target_name, {})
        all_y = []

        for comp, color, label in [("attn", "#e74c3c", "Attention"),
                                    ("mlp", "#3498db", "MLP")]:
            layer_means = []
            layer_ses = []
            valid_layers = []

            for l_idx in post_steering_layers:
                cl = f"{comp}_{l_idx}"
                comp_data = target_data.get(cl, {})
                if not comp_data:
                    continue
                if concept_filter is not None:
                    concept_means = [np.mean(vals) for c, vals in comp_data.items()
                                     if c in concept_filter]
                else:
                    concept_means = [np.mean(vals) for vals in comp_data.values()]
                if not concept_means:
                    continue
                layer_means.append(np.mean(concept_means))
                layer_ses.append(np.std(concept_means) / np.sqrt(len(concept_means)))
                valid_layers.append(l_idx)

            if valid_layers:
                layer_means = np.array(layer_means)
                layer_ses = np.array(layer_ses)
                all_y.extend(layer_means)
                all_y.extend(layer_means - 1.96 * layer_ses)
                all_y.extend(layer_means + 1.96 * layer_ses)

                ax.plot(valid_layers, layer_means, color=color, label=label,
                        linewidth=5.25, marker="o", markersize=12)
                ax.fill_between(valid_layers, layer_means - 1.96 * layer_ses,
                                layer_means + 1.96 * layer_ses, color=color, alpha=0.2)

        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            pad = max(0.5, (ymax - ymin) * 0.15) if ymax > ymin else 0.5
            ax.set_ylim(max(0, ymin - pad), ymax + pad)

        ax.set_ylabel(r"$\|\nabla_{\mathrm{comp}}\|_2$", fontsize=23)
        ax.set_title(title, fontsize=26)
        ax.tick_params(axis='both', labelsize=19)
        ax.set_xticks(post_steering_layers)
        ax.tick_params(axis='x', labelsize=22)
        ax.set_xlim(post_steering_layers[0] - 0.8, post_steering_layers[-1] + 0.5)
        ax.grid(True, alpha=0.3)

        if row_idx == len(row_targets) - 1:
            ax.set_xlabel("Layer", fontsize=23)


def plot_section_b(results: Dict, post_steering_layers: List[int],
                   steering_layer: int, plots_dir: Path):
    """Generate Section B line plots: per-layer gradient sensitivity."""
    print("\n  Generating Section B plots...")

    has_forced = "forced" in results and results["forced"]
    has_discriminative = "discriminative" in results and results["discriminative"]

    if not has_discriminative:
        print("  Warning: No discriminative results, skipping main plot")
        return

    row_targets = [("discriminative", "Gradient sensitivity: detection")]
    if has_forced:
        row_targets.append(("forced", "Gradient sensitivity: forced identification"))

    n_rows = len(row_targets)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = [axes]

    _plot_section_b_rows(axes, results, post_steering_layers, row_targets)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=25, frameon=True, bbox_to_anchor=(0.5, -0.02), columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.5)
    plt.savefig(plots_dir / "per_layer_gradient_attribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Section B main plot saved.")


def plot_section_b_by_groups(
    results: Dict, post_steering_layers: List[int],
    steering_layer: int, plots_dir: Path,
    success_concepts: List[str], failure_concepts: List[str],
):
    """Generate per-layer gradient sensitivity split by success vs failure."""
    first_target = next(iter(results), None)
    if not first_target:
        return
    first_comp = next(iter(results[first_target]), None)
    if not first_comp:
        return
    available_concepts = set(results[first_target][first_comp].keys())
    success_set = set(c for c in success_concepts if c in available_concepts)
    failure_set = set(c for c in failure_concepts if c in available_concepts)

    if not success_set or not failure_set:
        print(f"  Warning: by-groups plot skipped (success={len(success_set)}, failure={len(failure_set)})")
        return

    has_forced = "forced" in results and results["forced"]
    has_discriminative = "discriminative" in results and results["discriminative"]

    if not has_discriminative:
        return

    row_targets = [("discriminative", "Gradient sensitivity: detection")]
    if has_forced:
        row_targets.append(("forced", "Gradient sensitivity: forced identification"))

    n_rows = len(row_targets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(32, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for col_idx, (group_set, group_label) in enumerate(
        [(success_set, f"Success (n={len(success_set)})"),
         (failure_set, f"Failure (n={len(failure_set)})")]):

        col_row_targets = [(t, f"{group_label}: {title}") for t, title in row_targets]
        _plot_section_b_rows(axes[:, col_idx], results, post_steering_layers,
                              col_row_targets, concept_filter=group_set)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=25, frameon=True, bbox_to_anchor=(0.5, -0.02), columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.5, w_pad=2.0)
    plt.savefig(plots_dir / "per_layer_gradient_attribution_by_groups.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  By-groups gradient attribution plot saved.")


def plot_section_b_single_metric(
    metric_name: str,
    results: Dict,
    post_steering_layers: List[int],
    steering_layer: int,
    plots_dir: Path,
):
    """Generate a per-layer gradient sensitivity line plot for a single metric."""
    target_data = results.get(metric_name, {})
    if not target_data:
        return

    pretty = metric_name.replace("_", " ")
    has_forced = "forced" in results and results["forced"]

    row_targets = [(metric_name, f"Gradient sensitivity: {pretty}")]
    if has_forced:
        row_targets.append(("forced", "Gradient sensitivity: forced identification"))

    n_rows = len(row_targets)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = [axes]

    _plot_section_b_rows(axes, results, post_steering_layers, row_targets)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=25, frameon=True, bbox_to_anchor=(0.5, -0.02), columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.5)
    plt.savefig(plots_dir / f"per_layer_gradient_attribution_{metric_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_section_b_all_metrics(
    results: Dict,
    post_steering_layers: List[int],
    steering_layer: int,
    plots_dir: Path,
):
    """Generate per-layer gradient attribution plots for all non-primary metrics."""
    secondary_metrics = ["default_bundle", "single_yes_no", "p_yes", "entropy"]
    count = 0
    for metric_name in secondary_metrics:
        if metric_name in results and results[metric_name]:
            try:
                plot_section_b_single_metric(
                    metric_name, results, post_steering_layers, steering_layer, plots_dir,
                )
                count += 1
            except Exception as e:
                print(f"  Warning: gradient attribution plot for '{metric_name}' failed: {e}")
    if count > 0:
        print(f"  Multi-metric gradient attribution plots saved ({count} metrics).")


# ─────────────────────────────────────────────────────────────────────────────
# Section D: Attention Routing Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_section_d(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    strength: float,
    n_trials: int,
    output_dir: Path,
    device: str = "cuda",
    overwrite: bool = False,
    verbose: bool = False,
    exp21_dir: str = DEFAULT_EXP21_DIR,
    model_name: str = DEFAULT_MODEL,
) -> Dict:
    """Section D: Attention routing analysis."""
    print("\n" + "=" * 70)
    print("SECTION D: Attention Routing Analysis")
    print("=" * 70)

    section_dir = output_dir / "section_d"
    section_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = section_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    n_heads = model_wrapper.n_heads
    target_layers = list(range(steering_layer + 1, n_layers))
    yes_ids, no_ids = build_discriminative_token_set(
        exp21_dir, model_name, list(concept_vectors.keys()), tokenizer,
        steering_layer=steering_layer, strength=strength,
    )

    attn_path = section_dir / "attention_weights.json"
    if attn_path.exists() and not overwrite:
        print("  Loading existing attention data...")
        with open(attn_path) as f:
            attn_data = json.load(f)
    else:
        print(f"  Collecting attention weights ({len(concept_vectors)} concepts x {n_trials} trials x 2 conditions)...")

        # Force eager attention so output_attentions=True returns weights
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        original_attn_impl = getattr(config, '_attn_implementation', None)
        config._attn_implementation = "eager"
        print("  Set attention implementation to 'eager' for weight extraction")

        attn_data = {"steered": {}, "control": {}}
        concepts = list(concept_vectors.keys())

        for condition in ["steered", "control"]:
            print(f"\n  Condition: {condition}")

            for concept in tqdm(concepts, desc=f"  {condition}", disable=not verbose):
                steering_vec = concept_vectors[concept].to(device)
                concept_attn = defaultdict(list)

                for trial_idx in range(n_trials):
                    trial_num = trial_idx + 1
                    input_ids, _, _, steer_start = get_cached_prompt(trial_num, tokenizer, use_forced=False)
                    input_ids = input_ids.to(device)

                    hooks = []

                    try:
                        if condition == "steered":
                            def make_steering_hook(start_pos, vec, s):
                                def hook(module, args, output):
                                    hidden = output[0] if isinstance(output, tuple) else output
                                    seq_len = hidden.shape[1]
                                    if seq_len == 1:
                                        hidden = hidden + s * vec.to(hidden.dtype)
                                    elif start_pos < seq_len:
                                        hidden = hidden.clone()
                                        hidden[:, start_pos:, :] = hidden[:, start_pos:, :] + s * vec.to(hidden.dtype)
                                    if isinstance(output, tuple):
                                        return (hidden,) + output[1:]
                                    return hidden
                                return hook

                            layer_module = model_wrapper.get_layer_module(steering_layer)
                            h = layer_module.register_forward_hook(
                                make_steering_hook(steer_start, steering_vec, strength)
                            )
                            hooks.append(h)

                        with torch.no_grad():
                            outputs = model(
                                input_ids=input_ids,
                                output_attentions=True,
                            )

                        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                            for l_idx in target_layers:
                                if l_idx < len(outputs.attentions):
                                    attn = outputs.attentions[l_idx][0]  # [n_heads, seq, seq]
                                    seq_len = attn.shape[-1]

                                    for h_idx in range(attn.shape[0]):
                                        if steer_start < seq_len:
                                            attn_to_steered = attn[h_idx, -1, steer_start:].sum().item()
                                        else:
                                            attn_to_steered = 0.0

                                        probs = attn[h_idx, -1, :]
                                        probs = probs.clamp(min=1e-10)
                                        entropy = -(probs * probs.log()).sum().item()

                                        concept_attn[(l_idx, h_idx)].append({
                                            "attn_to_steered": attn_to_steered,
                                            "entropy": entropy,
                                        })

                    finally:
                        for h in hooks:
                            h.remove()
                        del input_ids
                        if 'outputs' in locals():
                            del outputs
                        torch.cuda.empty_cache()

                for (l, h), trial_vals in concept_attn.items():
                    key = f"{l},{h}"
                    if key not in attn_data[condition]:
                        attn_data[condition][key] = {"attn_to_steered": [], "entropy": []}
                    attn_data[condition][key]["attn_to_steered"].append(
                        float(np.mean([v["attn_to_steered"] for v in trial_vals]))
                    )
                    attn_data[condition][key]["entropy"].append(
                        float(np.mean([v["entropy"] for v in trial_vals]))
                    )

        # Restore original attention implementation
        if original_attn_impl is not None:
            config._attn_implementation = original_attn_impl
        elif hasattr(config, '_attn_implementation'):
            delattr(config, '_attn_implementation')
        print("  Restored attention implementation")

        if not attn_data["steered"]:
            print("  WARNING: No attention data collected! output_attentions may not be supported.")

        with open(attn_path, "w") as f:
            json.dump(attn_data, f, indent=2)

    # Compute deltas
    delta_results = compute_section_d_analysis(attn_data)
    with open(section_dir / "attention_deltas.json", "w") as f:
        json.dump(delta_results, f, indent=2)

    plot_section_d(attn_data, delta_results, plots_dir, steering_layer, n_heads)

    return {"attention": attn_data, "deltas": delta_results}


def compute_section_d_analysis(attn_data: Dict) -> Dict:
    """Compute attention deltas between steered and control conditions."""
    deltas = {}

    steered = attn_data.get("steered", {})
    control = attn_data.get("control", {})

    for key in steered:
        if key in control:
            s_vals = steered[key]["attn_to_steered"]
            c_vals = control[key]["attn_to_steered"]

            n = min(len(s_vals), len(c_vals))
            if n > 0:
                delta_vals = [s_vals[i] - c_vals[i] for i in range(n)]
                mean_delta = float(np.mean(delta_vals))

                if n > 1:
                    t_stat, p_val = stats.ttest_1samp(delta_vals, 0)
                else:
                    t_stat, p_val = 0, 1

                deltas[key] = {
                    "mean_delta": mean_delta,
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "n_concepts": n,
                }

    # FDR correction
    p_values = [(k, d["p_value"]) for k, d in deltas.items()]
    if p_values:
        p_values.sort(key=lambda x: x[1])
        n_tests = len(p_values)
        for rank, (key, p) in enumerate(p_values, 1):
            fdr_threshold = 0.05 * rank / n_tests
            deltas[key]["fdr_significant"] = p <= fdr_threshold

    return {"deltas": deltas}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_section_a(results: Dict, post_steering_layers: List[int],
                    steering_layer: int, plots_dir: Path):
    """Generate Section A line plots: per-layer ablation effects.

    Three-row figure:
      Row 1: Ablating steered: detection
      Row 2: Patching steered -> unsteered: detection
      Row 3: Ablating steered (forced): identification
    """
    print("\n  Generating Section A plots...")

    baseline_gaps = results.get("baseline", {})
    no_steer_gaps = results.get("no_steering", {})
    baseline_ref = np.median([np.mean(g) for g in baseline_gaps.values()]) if baseline_gaps else 0
    no_steer_ref = np.median([np.mean(g) for g in no_steer_gaps.values()]) if no_steer_gaps else 0

    forced_base_gaps = results.get("forced_baseline", {})
    forced_nosteer_gaps = results.get("forced_no_steering", {})
    forced_base_ref = np.median([np.mean(g) for g in forced_base_gaps.values()]) if forced_base_gaps else None
    forced_nosteer_ref = np.median([np.mean(g) for g in forced_nosteer_gaps.values()]) if forced_nosteer_gaps else None

    has_forced = any(k.startswith("forced_knockout_") for k in results)
    n_rows = 3 if has_forced else 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(9.0, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = [axes]

    row_configs = [
        ("knockout", "Ablating steered: detection",
         "Detection log-odds",
         baseline_ref, no_steer_ref),
        ("knockin", "Patching steered \u2192 unsteered: detection",
         "Detection log-odds",
         baseline_ref, no_steer_ref),
    ]
    if has_forced:
        row_configs.append(
            ("forced_knockout", "Ablating steered (forced): identification",
             "Identification log-odds", forced_base_ref, forced_nosteer_ref),
        )

    for row_idx, (direction, title, ylabel, ref_steered, ref_unsteered) in enumerate(row_configs):
        ax = axes[row_idx]
        all_y = []
        is_knockin = (direction == "knockin")
        layer_effects = {}

        for comp, color, label in [("attn", plot_style.CLAY, "Attention"), ("mlp", plot_style.SKY, "MLP")]:
            layer_means = []
            layer_ses = []
            valid_layers = []

            for l_idx in post_steering_layers:
                cond_name = f"{direction}_{comp}_{l_idx}"
                cond_gaps = results.get(cond_name, {})
                if not cond_gaps:
                    continue

                concept_means = [np.mean(gaps) for gaps in cond_gaps.values()]
                med = np.median(concept_means)
                layer_means.append(med)
                layer_ses.append(1.2533 * np.std(concept_means) / np.sqrt(len(concept_means)))
                valid_layers.append(l_idx)
                if is_knockin:
                    layer_effects[l_idx] = max(layer_effects.get(l_idx, float('-inf')), med)
                else:
                    layer_effects[l_idx] = min(layer_effects.get(l_idx, float('inf')), med)

            if not valid_layers:
                continue

            layer_means = np.array(layer_means)
            layer_ses = np.array(layer_ses)
            all_y.extend(layer_means)
            all_y.extend(layer_means - 1.96 * layer_ses)
            all_y.extend(layer_means + 1.96 * layer_ses)

            ax.plot(valid_layers, layer_means, color=color, label=label, linewidth=5.25, marker="o", markersize=12)
            ax.fill_between(valid_layers, layer_means - 1.96 * layer_ses, layer_means + 1.96 * layer_ses,
                            color=color, alpha=0.2)

        if ref_steered is not None and row_idx != 1:
            ax.axhline(y=ref_steered, color=plot_style.OLIVE, linestyle="--", linewidth=6, alpha=1.0, label="Baseline")
            all_y.append(ref_steered)
        if ref_unsteered is not None and row_idx == 1:
            ax.axhline(y=ref_unsteered, color=plot_style.DARK_PURPLE, linestyle="--", linewidth=6, alpha=1.0, label="No steering")
            all_y.append(ref_unsteered)

        if all_y:
            y_data = [v for v in all_y if v != ref_unsteered]
            if y_data:
                ymin, ymax = min(y_data), max(y_data)
                pad = max(0.5, (ymax - ymin) * 0.15)
                ax.set_ylim(ymin - pad, ymax + pad)

        # Top-5 x-ticks
        if is_knockin:
            ranked = sorted(layer_effects, key=lambda l: layer_effects[l], reverse=True)
        else:
            ranked = sorted(layer_effects, key=lambda l: layer_effects[l])
        top5_layers = []
        for l in ranked:
            if len(top5_layers) >= 5:
                break
            if any(abs(l - existing) <= 1 for existing in top5_layers):
                continue
            top5_layers.append(l)
        top5_layers.sort()

        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_title(title, fontsize=25)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_xticks(top5_layers)
        ax.tick_params(axis='x', labelsize=22)
        ax.set_xlim(post_steering_layers[0] - 0.8, post_steering_layers[-1] + 0.5)
        ax.grid(True, alpha=0.3)

        if row_idx == n_rows - 1:
            ax.set_xlabel("Layer", fontsize=23)

    fig.align_ylabels(axes)

    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)
    leg = fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_labels),
               fontsize=25, bbox_to_anchor=(0.5, -0.02), frameon=True,
               edgecolor="0.8", fancybox=True, columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.5)
    plt.savefig(plots_dir / "per_layer_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Section A plots saved.")


def plot_section_a_p_det(results: Dict, post_steering_layers: List[int],
                          steering_layer: int, plots_dir: Path):
    """Generate Section A plot using probabilities instead of log-odds."""
    def _logodds_to_p(vals):
        return [1.0 / (1.0 + np.exp(-v)) for v in vals]

    baseline_gaps = results.get("baseline", {})
    no_steer_gaps = results.get("no_steering", {})
    baseline_ref = np.mean([np.mean(_logodds_to_p(g)) for g in baseline_gaps.values()]) if baseline_gaps else 0
    no_steer_ref = np.mean([np.mean(_logodds_to_p(g)) for g in no_steer_gaps.values()]) if no_steer_gaps else 0

    forced_base_gaps = results.get("forced_baseline", {})
    forced_nosteer_gaps = results.get("forced_no_steering", {})
    forced_base_ref = np.mean([np.mean(_logodds_to_p(g)) for g in forced_base_gaps.values()]) if forced_base_gaps else 0
    forced_nosteer_ref = np.mean([np.mean(_logodds_to_p(g)) for g in forced_nosteer_gaps.values()]) if forced_nosteer_gaps else 0
    has_forced = any(k.startswith("forced_knockout_") for k in results)

    n_rows = 3 if has_forced else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.0, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = [axes]

    row_configs = [
        ("knockout", "Ablating steered: detection",
         r"$P(\mathrm{detection})$",
         baseline_ref, no_steer_ref),
        ("knockin", "Patching steered \u2192 unsteered: detection",
         r"$P(\mathrm{detection})$",
         baseline_ref, no_steer_ref),
    ]
    if has_forced:
        row_configs.append(
            ("forced_knockout", "Ablating steered (forced): identification",
             r"$P(\mathrm{identification})$",
             forced_base_ref, forced_nosteer_ref),
        )

    for row_idx, (direction, title, ylabel, ref_steered, ref_unsteered) in enumerate(row_configs):
        ax = axes[row_idx]
        all_y = []
        is_knockin = (direction == "knockin")
        layer_effects = {}

        for comp, color, label in [("attn", plot_style.CLAY, "Attention"), ("mlp", plot_style.SKY, "MLP")]:
            layer_means = []
            layer_ses = []
            valid_layers = []

            for l_idx in post_steering_layers:
                cond_name = f"{direction}_{comp}_{l_idx}"
                cond_gaps = results.get(cond_name, {})
                if not cond_gaps:
                    continue

                concept_p_dets = [np.mean(_logodds_to_p(gaps)) for gaps in cond_gaps.values()]
                m = np.mean(concept_p_dets)
                layer_means.append(m)
                layer_ses.append(np.std(concept_p_dets) / np.sqrt(len(concept_p_dets)))
                valid_layers.append(l_idx)
                if is_knockin:
                    layer_effects[l_idx] = max(layer_effects.get(l_idx, float('-inf')), m)
                else:
                    layer_effects[l_idx] = min(layer_effects.get(l_idx, float('inf')), m)

            if not valid_layers:
                continue

            layer_means = np.array(layer_means)
            layer_ses = np.array(layer_ses)
            all_y.extend(layer_means)
            all_y.extend(layer_means - 1.96 * layer_ses)
            all_y.extend(layer_means + 1.96 * layer_ses)

            ax.plot(valid_layers, layer_means, color=color, label=label, linewidth=5.25, marker="o", markersize=12)
            ax.fill_between(valid_layers, layer_means - 1.96 * layer_ses, layer_means + 1.96 * layer_ses,
                            color=color, alpha=0.2)

        if ref_steered is not None and row_idx != 1:
            ax.axhline(y=ref_steered, color=plot_style.OLIVE, linestyle="--", linewidth=6, alpha=1.0, label="Baseline")
            all_y.append(ref_steered)
        if ref_unsteered is not None and row_idx == 1:
            ax.axhline(y=ref_unsteered, color=plot_style.DARK_PURPLE, linestyle="--", linewidth=6, alpha=1.0, label="No steering")
            all_y.append(ref_unsteered)

        if all_y:
            y_data = [v for v in all_y if v != ref_unsteered]
            if y_data:
                ymin, ymax = min(y_data), max(y_data)
                pad = max(0.05, (ymax - ymin) * 0.15)
                ax.set_ylim(max(0, ymin - pad), min(1, ymax + pad))

        if is_knockin:
            ranked = sorted(layer_effects, key=lambda l: layer_effects[l], reverse=True)
        else:
            ranked = sorted(layer_effects, key=lambda l: layer_effects[l])
        top5_layers = []
        for l in ranked:
            if len(top5_layers) >= 5:
                break
            if any(abs(l - existing) <= 1 for existing in top5_layers):
                continue
            top5_layers.append(l)
        top5_layers.sort()

        ax.set_ylabel(ylabel, fontsize=27, labelpad=16)
        ax.set_title(title, fontsize=27)
        ax.tick_params(axis='y', labelsize=25)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.set_xticks(top5_layers)
        ax.tick_params(axis='x', labelsize=25)
        ax.set_xlim(post_steering_layers[0] - 0.8, post_steering_layers[-1] + 0.5)
        ax.grid(True, alpha=0.3)

        if row_idx == n_rows - 1:
            ax.set_xlabel("Layer", fontsize=27)

    fig.align_ylabels(axes)
    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)
    leg = fig.legend(all_handles, all_labels, loc="upper center", ncol=len(all_labels),
               fontsize=25, bbox_to_anchor=(0.5, 1.02), frameon=True,
               edgecolor="0.8", fancybox=True, columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.5)
    plt.savefig(plots_dir / "per_layer_ablation_p_det.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Section A P(det) plot saved.")


def plot_section_a_by_groups(
    results: Dict, post_steering_layers: List[int],
    steering_layer: int, plots_dir: Path,
    success_concepts: List[str], failure_concepts: List[str],
):
    """Generate per-layer ablation plot split by success vs failure concept groups."""
    baseline_concepts = set(results.get("baseline", {}).keys())
    success_set = [c for c in success_concepts if c in baseline_concepts]
    failure_set = [c for c in failure_concepts if c in baseline_concepts]

    if not success_set or not failure_set:
        print(f"  Warning: by-groups plot skipped (success={len(success_set)}, failure={len(failure_set)})")
        return

    has_forced = any(k.startswith("forced_knockout_") for k in results)
    n_rows = 3 if has_forced else 2

    row_configs = [
        ("knockout", "Knockout: detection",
         "Detection log-odds",
         "baseline", "no_steering"),
        ("knockin", "Knock-in: detection",
         "Detection log-odds",
         "baseline", "no_steering"),
    ]
    if has_forced:
        row_configs.append(
            ("forced_knockout", "Forced knockout: identification",
             "Identification log-odds", "forced_baseline", "forced_no_steering"),
        )

    fig, axes = plt.subplots(n_rows, 2, figsize=(28, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    groups = [("Success", success_set), ("Failure", failure_set)]

    for col_idx, (group_name, group_concepts) in enumerate(groups):
        for row_idx, (direction, title, ylabel, base_key, nosteer_key) in enumerate(row_configs):
            ax = axes[row_idx, col_idx]
            all_y = []

            base_gaps = results.get(base_key, {})
            nosteer_gaps = results.get(nosteer_key, {})
            g_base = [np.mean(base_gaps[c]) for c in group_concepts if c in base_gaps]
            g_nosteer = [np.mean(nosteer_gaps[c]) for c in group_concepts if c in nosteer_gaps]
            baseline_mean = np.mean(g_base) if g_base else None
            nosteer_mean = np.mean(g_nosteer) if g_nosteer else None

            for comp, color, label in [("attn", plot_style.CLAY, "Attention"), ("mlp", plot_style.SKY, "MLP")]:
                layer_means = []
                layer_ses = []
                valid_layers = []

                for l_idx in post_steering_layers:
                    cond_name = f"{direction}_{comp}_{l_idx}"
                    cond_gaps = results.get(cond_name, {})
                    if not cond_gaps:
                        continue
                    concept_means = [np.mean(cond_gaps[c]) for c in group_concepts if c in cond_gaps]
                    if not concept_means:
                        continue
                    layer_means.append(np.mean(concept_means))
                    layer_ses.append(np.std(concept_means) / np.sqrt(len(concept_means)))
                    valid_layers.append(l_idx)

                if not valid_layers:
                    continue
                layer_means = np.array(layer_means)
                layer_ses = np.array(layer_ses)
                all_y.extend(layer_means)
                all_y.extend(layer_means - 1.96 * layer_ses)
                all_y.extend(layer_means + 1.96 * layer_ses)
                ax.plot(valid_layers, layer_means, color=color, label=label, linewidth=5.25, marker="o", markersize=12)
                ax.fill_between(valid_layers, layer_means - 1.96 * layer_ses, layer_means + 1.96 * layer_ses,
                                color=color, alpha=0.2)

            if baseline_mean is not None and row_idx == 0:
                ax.axhline(y=baseline_mean, color=plot_style.OLIVE, linestyle="--", linewidth=6, alpha=1.0, label="Baseline")
                all_y.append(baseline_mean)
            if nosteer_mean is not None:
                ax.axhline(y=nosteer_mean, color=plot_style.DARK_PURPLE, linestyle="--", linewidth=6, alpha=1.0, label="No steering")
                all_y.append(nosteer_mean)

            if all_y:
                y_data = [v for v in all_y if v != nosteer_mean]
                ymin, ymax = (min(y_data), max(y_data)) if y_data else (min(all_y), max(all_y))
                pad = max(0.5, (ymax - ymin) * 0.15) if ymax > ymin else 0.5
                ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=18)
            ax.set_title(f"{group_name} (n={len(group_concepts)}): {title}", fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xticks(post_steering_layers)
            ax.tick_params(axis='x', labelsize=16)
            ax.set_xlim(post_steering_layers[0] - 0.8, post_steering_layers[-1] + 0.5)
            ax.grid(True, alpha=0.3)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=20)

    fig.align_ylabels(axes[:, 0])

    all_handles, all_labels = [], []
    seen = set()
    for row_axes in axes:
        for ax in row_axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    all_handles.append(h)
                    all_labels.append(l)
                    seen.add(l)
    leg = fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_labels),
               fontsize=23, bbox_to_anchor=(0.5, -0.02), frameon=True,
               edgecolor="0.8", fancybox=True, columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.5, w_pad=2.0)
    plt.savefig(plots_dir / "per_layer_ablation_by_groups.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  By-groups plot saved.")


def plot_section_a_single_metric(
    metric_name: str,
    metric_results: Dict[str, Dict[str, List[float]]],
    post_steering_layers: List[int],
    steering_layer: int,
    plots_dir: Path,
    forced_results: Optional[Dict] = None,
):
    """Generate a single per-layer ablation plot for one metric."""
    config = METRIC_PLOT_CONFIG.get(metric_name)
    if config is None:
        return

    is_divergence = metric_name in ("kl_divergence", "js_divergence", "total_variation")

    has_knockout = any(k.startswith("knockout_") for k in metric_results)
    has_knockin = any(k.startswith("knockin_") for k in metric_results)
    has_forced = forced_results is not None and any(
        k.startswith("forced_knockout_") for k in forced_results
    )

    rows = []
    if has_knockout:
        rows.append(("knockout", "Ablating steered: detection", config["ylabel"], metric_results))
    if has_knockin:
        rows.append(("knockin", "Patching steered \u2192 unsteered: detection", config["ylabel"], metric_results))
    if has_forced:
        rows.append(("forced_knockout", "Ablating steered (forced): identification",
                      "Identification log-odds", forced_results))

    if not rows:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.2 * n_rows + 1.2))
    if n_rows == 1:
        axes = [axes]

    for row_idx, (direction, title, ylabel, row_data) in enumerate(rows):
        ax = axes[row_idx]
        all_y = []

        is_forced_row = direction == "forced_knockout"
        row_is_divergence = is_divergence and not is_forced_row

        row_baseline_mean = None
        row_nosteer_mean = None
        if not row_is_divergence:
            if is_forced_row:
                base_gaps = row_data.get("forced_baseline", {})
                nosteer_gaps = row_data.get("forced_no_steering", {})
            else:
                base_gaps = metric_results.get("baseline", {})
                nosteer_gaps = metric_results.get("no_steering", {})
            row_baseline_mean = np.mean([np.mean(g) for g in base_gaps.values()]) if base_gaps else None
            row_nosteer_mean = np.mean([np.mean(g) for g in nosteer_gaps.values()]) if nosteer_gaps else None

        for comp, color, label in [("attn", plot_style.CLAY, "Attention"), ("mlp", plot_style.SKY, "MLP")]:
            layer_means = []
            layer_ses = []
            valid_layers = []

            for l_idx in post_steering_layers:
                cond_name = f"{direction}_{comp}_{l_idx}"
                cond_gaps = row_data.get(cond_name, {})
                if not cond_gaps:
                    continue

                concept_means = [np.mean(gaps) for gaps in cond_gaps.values()]
                layer_means.append(np.mean(concept_means))
                layer_ses.append(np.std(concept_means) / np.sqrt(len(concept_means)))
                valid_layers.append(l_idx)

            if not valid_layers:
                continue

            layer_means = np.array(layer_means)
            layer_ses = np.array(layer_ses)
            all_y.extend(layer_means)
            all_y.extend(layer_means - 1.96 * layer_ses)
            all_y.extend(layer_means + 1.96 * layer_ses)

            ax.plot(valid_layers, layer_means, color=color, label=label, linewidth=5.25, marker="o", markersize=12)
            ax.fill_between(valid_layers, layer_means - 1.96 * layer_ses, layer_means + 1.96 * layer_ses,
                            color=color, alpha=0.2)

        if not row_is_divergence:
            if row_baseline_mean is not None and row_idx == 0:
                ax.axhline(y=row_baseline_mean, color=plot_style.OLIVE, linestyle="--", linewidth=6, alpha=1.0, label="Baseline")
                all_y.append(row_baseline_mean)
            if row_nosteer_mean is not None:
                ax.axhline(y=row_nosteer_mean, color=plot_style.DARK_PURPLE, linestyle="--", linewidth=6, alpha=1.0, label="No steering")
                all_y.append(row_nosteer_mean)
        else:
            ax.axhline(y=0, color=plot_style.OLIVE, linestyle="--", linewidth=6, alpha=1.0, label="Baseline (zero)")

        if all_y:
            if row_is_divergence:
                ymin, ymax = min(all_y), max(all_y)
            else:
                y_data = [v for v in all_y if v != row_nosteer_mean]
                ymin, ymax = (min(y_data), max(y_data)) if y_data else (min(all_y), max(all_y))
            pad = max(0.5, (ymax - ymin) * 0.15) if ymax > ymin else 0.5
            ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_ylabel(ylabel, fontsize=int(config.get("ylabel_fontsize", 15) * 1.37))
        ax.set_title(title, fontsize=25)
        ax.tick_params(axis='both', labelsize=19)
        ax.set_xticks(post_steering_layers)
        ax.tick_params(axis='x', labelsize=22)
        ax.set_xlim(post_steering_layers[0] - 0.8, post_steering_layers[-1] + 0.5)
        ax.grid(True, alpha=0.3)

        if row_idx == n_rows - 1:
            ax.set_xlabel("Layer", fontsize=23)

    fig.align_ylabels(axes)

    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)
    leg = fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_labels),
               fontsize=25, bbox_to_anchor=(0.5, -0.02), frameon=True,
               edgecolor="0.8", fancybox=True, columnspacing=0.82, handletextpad=0.3,
               handlelength=1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.5)

    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.5)
    plt.savefig(plots_dir / config["filename"], dpi=300, bbox_inches="tight")
    plt.close()


def plot_section_a_all_metrics(
    multi_results: Dict[str, Dict],
    post_steering_layers: List[int],
    steering_layer: int,
    plots_dir: Path,
    forced_results: Optional[Dict] = None,
):
    """Generate per-layer ablation plots for ALL metrics (excluding 'discriminative')."""
    for metric_name in MULTI_METRIC_NAMES:
        if metric_name == "discriminative":
            continue
        metric_data = multi_results.get(metric_name, {})
        if not metric_data:
            continue
        try:
            plot_section_a_single_metric(
                metric_name, metric_data, post_steering_layers, steering_layer, plots_dir,
                forced_results=forced_results,
            )
        except Exception as e:
            print(f"  Warning: plot for metric '{metric_name}' failed: {e}")
    print(f"  Multi-metric plots saved ({len(MULTI_METRIC_NAMES) - 1} metrics).")


def plot_section_d(attn_data: Dict, delta_results: Dict, plots_dir: Path, steering_layer: int, n_heads: int):
    """Generate Section D plots."""
    print("\n  Generating Section D plots...")

    deltas = delta_results.get("deltas", {})

    if not deltas:
        print("  No attention delta data available, skipping plots.")
        return

    # Plot 1: Attention delta heatmap
    min_layer = min(int(k.split(",")[0]) for k in deltas)
    max_layer = max(int(k.split(",")[0]) for k in deltas)
    max_head = max(int(k.split(",")[1]) for k in deltas) + 1

    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap = np.zeros((max_layer - min_layer + 1, max_head))
    for key, d in deltas.items():
        l, h = int(key.split(",")[0]), int(key.split(",")[1])
        heatmap[l - min_layer, h] = d["mean_delta"]

    vmax = np.percentile(np.abs(heatmap[heatmap != 0]), 95) if np.any(heatmap != 0) else 0.1
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_yticks(range(0, max_layer - min_layer + 1, 5))
    ax.set_yticklabels(range(min_layer, max_layer + 1, 5))
    ax.set_title("Section D: Attention Delta (Steered - Control) to Steered Region", fontsize=14)
    plt.colorbar(im, ax=ax, label="Attention delta")

    plt.tight_layout()
    plt.savefig(plots_dir / "attention_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Mean attention to steered region by layer
    fig, ax = plt.subplots(figsize=(10, 5))

    steered_by_layer = defaultdict(list)
    control_by_layer = defaultdict(list)

    steered_data = attn_data.get("steered", {})
    control_data = attn_data.get("control", {})

    for key in steered_data:
        l = int(key.split(",")[0])
        steered_by_layer[l].append(np.mean(steered_data[key]["attn_to_steered"]))
    for key in control_data:
        l = int(key.split(",")[0])
        control_by_layer[l].append(np.mean(control_data[key]["attn_to_steered"]))

    layers = sorted(set(steered_by_layer.keys()) | set(control_by_layer.keys()))
    s_means = [np.mean(steered_by_layer.get(l, [0])) for l in layers]
    c_means = [np.mean(control_by_layer.get(l, [0])) for l in layers]
    s_se = [np.std(steered_by_layer.get(l, [0])) / max(np.sqrt(len(steered_by_layer.get(l, [0]))), 1) for l in layers]
    c_se = [np.std(control_by_layer.get(l, [0])) / max(np.sqrt(len(control_by_layer.get(l, [0]))), 1) for l in layers]

    ax.plot(layers, s_means, "o-", color="#e74c3c", label="Steered", linewidth=2, markersize=4)
    ax.fill_between(layers, np.array(s_means) - np.array(s_se), np.array(s_means) + np.array(s_se),
                     color="#e74c3c", alpha=0.15)
    ax.plot(layers, c_means, "s-", color="#3498db", label="Control", linewidth=2, markersize=4)
    ax.fill_between(layers, np.array(c_means) - np.array(c_se), np.array(c_means) + np.array(c_se),
                     color="#3498db", alpha=0.15)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Attention to Steered Region", fontsize=12)
    ax.set_title("Section D: Attention to Steered Region by Layer", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "attention_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Section D plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    global _is_base_model

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    vectors_model = args.vectors_model or args.model

    print("EXPERIMENT 59: STEERING ATTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"  Model: {args.model}")
    if vectors_model != args.model:
        print(f"  Vectors model: {vectors_model}")
    print(f"  Steering: layer {args.layer}, strength {args.strength}")
    print(f"  Concepts: {args.n_concepts}")
    print(f"  Sections: {args.sections}")
    print(f"  Ablation mode: {args.ablation_mode}")
    print(f"  Seed: {args.seed}")
    print()

    # Output directory
    output_dir = Path(args.output_dir) / args.model / f"layer_{args.layer}_strength_{args.strength}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load concepts (from vectors_model's exp21 data)
    print("Loading concepts...")
    successful = load_successful_concepts(
        args.exp21_dir, vectors_model, args.layer, args.strength,
    )
    print(f"  Found {len(successful)} successful concepts")

    # Sample concepts
    if len(successful) > args.n_concepts:
        rng = random.Random(args.seed)
        concepts = rng.sample(successful, args.n_concepts)
    else:
        concepts = successful

    print(f"  Using {len(concepts)} concepts")

    # Load concept vectors for instruct models
    concept_vectors = None
    if not args.model.endswith("_pt"):
        print("Loading concept vectors...")
        concept_vectors = load_concept_vectors(
            args.exp21_dir, vectors_model, concepts, steering_layer=args.layer,
        )
        print(f"  Loaded {len(concept_vectors)} vectors")
        if not concept_vectors:
            print("ERROR: No concept vectors loaded. Check exp21 directory.")
            sys.exit(1)

    if args.plots_only:
        print("\n  --plots-only mode: regenerating plots from saved results...")

        if "A" in args.sections:
            a_dir = output_dir / "section_a"
            if (a_dir / "logit_gaps.json").exists():
                with open(a_dir / "logit_gaps.json") as f:
                    a_results = json.load(f)
                all_layer_nums = set()
                for key in a_results:
                    parts = key.split("_")
                    if len(parts) >= 3 and parts[-1].isdigit():
                        all_layer_nums.add(int(parts[-1]))
                if all_layer_nums:
                    post_layers = sorted(all_layer_nums)
                else:
                    post_layers = list(range(args.layer + 1, 62))
                plot_section_a(a_results, post_layers, args.layer, a_dir / "plots")
                plot_section_a_p_det(a_results, post_layers, args.layer, a_dir / "plots")

                multi_path = a_dir / "multi_metric_gaps.json"
                if multi_path.exists():
                    with open(multi_path) as f:
                        multi_results = json.load(f)
                    print(f"  Generating multi-metric plots ({len(multi_results)} metrics)...")
                    plot_section_a_all_metrics(multi_results, post_layers, args.layer, a_dir / "plots",
                                               forced_results=a_results)

                success_c, failure_c = load_concept_groups(
                    steering_layer=args.layer, strength=args.strength, model_name=args.model,
                )
                if success_c:
                    plot_section_a_by_groups(a_results, post_layers, args.layer, a_dir / "plots",
                                             success_c, failure_c)

        if "B" in args.sections:
            b_dir = output_dir / "section_b"
            if (b_dir / "gradient_attribution.json").exists():
                with open(b_dir / "gradient_attribution.json") as f:
                    b_results = json.load(f)
                b_layer_nums = set()
                for target in b_results.values():
                    for cl in target:
                        parts = cl.rsplit("_", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            b_layer_nums.add(int(parts[1]))
                b_post_layers = sorted(b_layer_nums) if b_layer_nums else list(range(args.layer + 1, 62))
                plot_section_b(b_results, b_post_layers, args.layer, b_dir / "plots")

                success_b, failure_b = load_concept_groups(
                    steering_layer=args.layer, strength=args.strength, model_name=args.model,
                )
                if success_b:
                    plot_section_b_by_groups(b_results, b_post_layers, args.layer, b_dir / "plots",
                                             success_b, failure_b)

                plot_section_b_all_metrics(b_results, b_post_layers, args.layer, b_dir / "plots")

        if "D" in args.sections:
            d_dir = output_dir / "section_d"
            if (d_dir / "attention_weights.json").exists():
                with open(d_dir / "attention_weights.json") as f:
                    attn_data = json.load(f)
                delta_res = compute_section_d_analysis(attn_data)
                d_n_heads = 32  # default
                if attn_data.get("steered"):
                    d_n_heads = max((int(k.split(",")[1]) for k in attn_data["steered"]), default=31) + 1
                plot_section_d(attn_data, delta_res, d_dir / "plots", args.layer, d_n_heads)

        print("\n  Plots regenerated.")
        return

    # Load model
    print("Loading model...")
    dtype = getattr(torch, args.dtype)
    model_wrapper = load_model(args.model, device=args.device, dtype=dtype, quantization=args.quantization)
    tokenizer = model_wrapper.tokenizer

    # Detect base model (no chat template)
    _is_base_model = (
        not hasattr(tokenizer, 'chat_template') or
        tokenizer.chat_template is None
    )
    if _is_base_model:
        print("  Base model detected (no chat template) -- using raw User:/Assistant: format")

    print(f"  Model loaded: {model_wrapper.n_layers} layers, {model_wrapper.n_heads} heads")
    print(f"  Head dim: {get_head_dim(model_wrapper)}")

    # Extract concept vectors for base model (requires loaded model)
    if _is_base_model and concept_vectors is None:
        print("Extracting concept vectors from base model (on-the-fly)...")
        baseline_words = get_baseline_words(100)
        concept_vectors = extract_concept_vectors_base_model(
            model_wrapper, concepts, baseline_words, args.layer,
        )
        print(f"  Extracted {len(concept_vectors)} vectors")
        if not concept_vectors:
            print("ERROR: No concept vectors extracted.")
            sys.exit(1)

    # Run sections
    if "A" in args.sections:
        run_section_a(
            model_wrapper, tokenizer, concept_vectors,
            args.layer, args.strength, args.n_trials,
            output_dir, device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite, verbose=args.verbose,
            exp21_dir=args.exp21_dir, model_name=vectors_model,
            ablation_mode=args.ablation_mode,
        )
        gc.collect()
        torch.cuda.empty_cache()

    if "B" in args.sections:
        run_section_b(
            model_wrapper, tokenizer, concept_vectors,
            args.layer, args.strength, args.n_trials,
            output_dir, device=args.device,
            overwrite=args.overwrite, verbose=args.verbose,
            exp21_dir=args.exp21_dir, model_name=vectors_model,
        )
        gc.collect()
        torch.cuda.empty_cache()

    if "D" in args.sections:
        run_section_d(
            model_wrapper, tokenizer, concept_vectors,
            args.layer, args.strength, args.n_attribution_trials,
            output_dir, device=args.device,
            overwrite=args.overwrite, verbose=args.verbose,
            exp21_dir=args.exp21_dir, model_name=vectors_model,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # Write summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 59 COMPLETE")
    print("=" * 70)
    print(f"  Results saved to: {output_dir}")

    summary = {"timestamp": datetime.now().isoformat(), "sections_run": args.sections}

    # Section A summary
    metrics_path = output_dir / "section_a" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            a_metrics = json.load(f)
        baseline_gap = a_metrics.get("baseline", {}).get("mean_logit_gap", 0)
        no_steer_gap = a_metrics.get("no_steering", {}).get("mean_logit_gap", 0)
        attn_ko_gaps = [v["mean_logit_gap"] for k, v in a_metrics.items()
                        if k.startswith("knockout_attn_") and "mean_logit_gap" in v]
        mlp_ko_gaps = [v["mean_logit_gap"] for k, v in a_metrics.items()
                       if k.startswith("knockout_mlp_") and "mean_logit_gap" in v]
        mean_attn_ko = np.mean(attn_ko_gaps) if attn_ko_gaps else 0
        mean_mlp_ko = np.mean(mlp_ko_gaps) if mlp_ko_gaps else 0
        attn_most_impactful = min(
            [(k, v["mean_logit_gap"]) for k, v in a_metrics.items()
             if k.startswith("knockout_attn_") and "mean_logit_gap" in v],
            key=lambda x: x[1], default=("none", 0)
        )
        mlp_most_impactful = min(
            [(k, v["mean_logit_gap"]) for k, v in a_metrics.items()
             if k.startswith("knockout_mlp_") and "mean_logit_gap" in v],
            key=lambda x: x[1], default=("none", 0)
        )
        summary["section_a"] = {
            "question": "Are attention heads causally necessary?",
            "baseline_logit_gap": baseline_gap,
            "no_steering_logit_gap": no_steer_gap,
            "mean_attn_knockout_gap": mean_attn_ko,
            "mean_mlp_knockout_gap": mean_mlp_ko,
            "n_attn_layers_tested": len(attn_ko_gaps),
            "n_mlp_layers_tested": len(mlp_ko_gaps),
            "most_impactful_attn_layer": attn_most_impactful[0],
            "most_impactful_attn_gap": attn_most_impactful[1],
            "most_impactful_mlp_layer": mlp_most_impactful[0],
            "most_impactful_mlp_gap": mlp_most_impactful[1],
        }

    # Section B summary
    b_attr_path = output_dir / "section_b" / "gradient_attribution.json"
    if b_attr_path.exists():
        with open(b_attr_path) as f:
            b_data = json.load(f)
        b_summary = {"question": "Which layers contribute most via gradient attribution?"}
        for target_name, target_data in b_data.items():
            attn_vals = []
            mlp_vals = []
            for cl, concept_data in target_data.items():
                mean_val = np.mean([np.mean(v) for v in concept_data.values()]) if concept_data else 0
                if cl.startswith("attn_"):
                    attn_vals.append(mean_val)
                elif cl.startswith("mlp_"):
                    mlp_vals.append(mean_val)
            b_summary[f"{target_name}_mean_attn_attribution"] = float(np.mean(attn_vals)) if attn_vals else 0
            b_summary[f"{target_name}_mean_mlp_attribution"] = float(np.mean(mlp_vals)) if mlp_vals else 0
        summary["section_b"] = b_summary

    # Section D summary
    delta_path = output_dir / "section_d" / "attention_deltas.json"
    if delta_path.exists():
        with open(delta_path) as f:
            d_deltas = json.load(f)
        summary["section_d"] = {
            "question": "What information-routing function do heads perform?",
            "n_significant_heads": sum(1 for d in d_deltas.get("deltas", {}).values() if d.get("fdr_significant")),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary written to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
