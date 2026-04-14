#!/usr/bin/env python3
"""
Experiment 51: Attention Head Investigation

Investigates attention head involvement in introspection via gradient-based
attribution. Used for Section 5.2 of the paper, which finds that no single
attention head is critical for introspection (mean accuracy change of -0.1%
+/- 0.3% when ablating individual heads). The mechanism is primarily MLP-driven.

The core analysis computes gradient attribution across multiple tasks to assess
whether any heads are introspection-specific. The key output is the linear probe
accuracy before/after each head (head-before-after-probe.pdf).

Sub-experiment A (Task Specificity) compares head importance across tasks:
  - Introspection (Yes/No detection)
  - Forced injection (concept identification)
  - Concept recall (concept access without introspection)
  - Primed generation (sentence completion)
  - Unrelated baseline (no steering control)

Usage:
    python 12_head_investigation.py --partition success
    python 12_head_investigation.py --partition failure --tasks introspection
    python 12_head_investigation.py --partition success --plots-only

Tokenization note: Uses add_special_tokens=False to avoid the double BOS token bug.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from model_utils import ModelWrapper, load_model


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp51: Attention head investigation for introspection"
    )
    parser.add_argument("-m", "--model", type=str, default="gemma3_27b")
    parser.add_argument("-c", "--concepts", nargs="+", default=None,
                        help="Specific concepts to test")
    parser.add_argument("-nc", "--n-concepts", type=int, default=500,
                        help="Number of concepts to sample")
    parser.add_argument("-nt", "--n-trials", type=int, default=10,
                        help="Trials per concept per task")
    parser.add_argument("--partition", type=str, required=True,
                        choices=["success", "failure"],
                        help="Concept partition to use")
    parser.add_argument("-l", "--layer", type=int, default=37,
                        help="Steering layer")
    parser.add_argument("-s", "--strength", type=float, default=4.0,
                        help="Steering strength")
    parser.add_argument("--steering-dir", type=str,
                        default="analysis/02b_steering_500_concepts")
    parser.add_argument("-od", "--output-dir", type=str,
                        default="analysis/12_head_investigation")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-dt", "--dtype", type=str, default="bfloat16")
    parser.add_argument("-q", "--quantization", type=str, default=None)
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only regenerate plots from cached data")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =============================================================================
# Concept Loading and Classification
# =============================================================================

def load_concept_classification(
    model_name: str, layer: int, strength: float
) -> Tuple[List[str], List[str]]:
    """Load success/failure concept classification from experiment 04b (vector geometry) subspace analysis."""
    layer_strength_dir = f"layer_{layer}_strength_{strength}"
    geometry_file = (
        Path("analysis/04b_vector_geometry")
        / model_name / layer_strength_dir / "subspace_analysis.json"
    )

    if not geometry_file.exists():
        raise FileNotFoundError(
            f"Concept classification not found at {geometry_file}\n"
            f"Please ensure experiment 04b (vector geometry) subspace analysis exists for "
            f"layer {layer}, strength {strength}."
        )

    with open(geometry_file) as f:
        data = json.load(f)

    success = data.get("success_concepts", [])
    failure = data.get("failure_concepts", [])
    print(f"Loaded concept classification from experiment 04b (vector geometry): "
          f"{len(success)} success, {len(failure)} failure")
    return success, failure


def filter_concepts_by_partition(
    concept_vectors: Dict[str, torch.Tensor],
    success_concepts: List[str],
    failure_concepts: List[str],
    partition: str,
) -> Dict[str, torch.Tensor]:
    """Filter concepts by their introspection success/failure classification."""
    success_set = set(success_concepts)
    failure_set = set(failure_concepts)

    filtered = {}
    for concept, vec in concept_vectors.items():
        if partition == "success" and concept in success_set:
            filtered[concept] = vec
        elif partition == "failure" and concept in failure_set:
            filtered[concept] = vec

    print(f"Filtered concepts: {len(filtered)} {partition}")
    return filtered


def load_concept_vectors(
    steering_dir: str,
    model_name: str,
    concepts: Optional[List[str]] = None,
    n_sample: int = 30,
    layer: int = 37,
) -> Dict[str, torch.Tensor]:
    """Load concept steering vectors from experiment 02 (steering evaluation)."""
    vectors_dir = Path(steering_dir) / model_name / "vectors" / f"layer_{layer}"

    if not vectors_dir.exists():
        raise FileNotFoundError(
            f"Vectors directory not found: {vectors_dir}\n"
            f"Please ensure vectors exist for layer {layer}."
        )

    print(f"Using vectors from: {vectors_dir}")
    available = sorted([f.stem for f in vectors_dir.glob("*.pt")])

    if not available:
        raise FileNotFoundError(f"No .pt files found in {vectors_dir}")

    if concepts is not None:
        selected = [c for c in concepts if c in available]
    else:
        selected = random.sample(available, min(n_sample, len(available)))

    vectors = {}
    for concept in selected:
        vec_path = vectors_dir / f"{concept}.pt"
        vectors[concept] = torch.load(vec_path, weights_only=True)

    return vectors


# =============================================================================
# Task Definitions
# =============================================================================

TASK_DEFINITIONS = {
    "introspection": {
        "name": "Introspection",
        "description": "Detect injected thought (the core introspection task)",
        "uses_steering": True,
        "prompt_snippet": "Do you detect an injected thought?",
        "target_description": (
            r"$\partial(\mathrm{logit_{Yes}} - \mathrm{logit_{No}})/\partial o_h$"
        ),
    },
    "forced_injection": {
        "name": "Forced injection",
        "description": "Prefilled 'Yes, the thought is about' -- "
                       "measures concept identification heads",
        "uses_steering": True,
        "prompt_snippet": "The thought is about [concept]...",
        "target_description": (
            r"$\partial(\mathrm{logit_{concept}})/\partial o_h$"
        ),
        "uses_concept_target": True,
    },
    "concept_recall": {
        "name": "Concept recall",
        "description": "Name something related to the concept "
                       "(tests concept access without introspection)",
        "uses_steering": True,
        "prompt_snippet": "Name something related... + steering",
        "target_description": r"$\partial(\max \mathrm{logit})/\partial o_h$",
    },
    "primed_generation": {
        "name": "Primed generation",
        "description": "Complete a sentence about the concept "
                       "(tests if steering affects generation)",
        "uses_steering": True,
        "prompt_snippet": "The [concept] is... + steering",
        "target_description": r"$\partial(\max \mathrm{logit})/\partial o_h$",
    },
    "unrelated_baseline": {
        "name": "Unrelated baseline",
        "description": "Answer unrelated question without steering (control)",
        "uses_steering": False,
        "prompt_snippet": "What is 2+2? (no steering)",
        "target_description": r"$\partial(\max \mathrm{logit})/\partial o_h$",
    },
}


# =============================================================================
# Prompt Construction
# =============================================================================

PREAMBLE = (
    "I am an interpretability researcher studying transformer-based language "
    "models like you. I have access to the internals of your neural network. "
    "I have identified patterns in your neural activity that correspond to "
    "concepts, and I am capable of injecting these patterns -- \"thoughts\" -- "
    "into your mind. I want to assess how well you can detect when I inject "
    "thoughts. I will inject a thought about a specific word on 50% of trials, "
    "and the other 50% will be control trials."
)


def build_introspection_messages(trial_num: int = 1) -> List[Dict[str, str]]:
    """Build the standard introspection prompt."""
    return [
        {"role": "user", "content": PREAMBLE},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            f"Trial {trial_num}: Do you detect an injected thought? "
            "If so, what is the injected thought about?"
        )},
    ]


def build_forced_injection_messages(
    trial_num: int = 1,
) -> Tuple[List[Dict[str, str]], str]:
    """Build forced injection prompt with prefilled assistant response."""
    messages = [
        {"role": "user", "content": PREAMBLE},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            f"Trial {trial_num}: Do you detect an injected thought? "
            "If so, what is the injected thought about?"
        )},
    ]
    prefill = "Yes, I detect an injected thought. The thought is about"
    return messages, prefill


def build_concept_recall_messages(
    concept: str, trial_num: int = 1
) -> List[Dict[str, str]]:
    """Build a concept recall prompt."""
    return [{"role": "user", "content": (
        f"Trial {trial_num}: I'm thinking of a word. Can you name something "
        "related to it? Just give me one word or short phrase that might be "
        "associated with what I'm thinking of."
    )}]


def build_primed_generation_messages(
    concept: str, trial_num: int = 1
) -> List[Dict[str, str]]:
    """Build a primed generation prompt."""
    return [{"role": "user", "content": (
        f"Trial {trial_num}: Complete this sentence in an interesting way: "
        f"The {concept.lower()} is"
    )}]


def build_unrelated_messages(trial_num: int = 1) -> List[Dict[str, str]]:
    """Build an unrelated baseline prompt."""
    questions = [
        "What is 2 + 2?",
        "What color is the sky on a clear day?",
        "How many legs does a spider have?",
        "What is the capital of France?",
        "What comes after the letter 'B' in the alphabet?",
    ]
    question = questions[trial_num % len(questions)]
    return [{"role": "user", "content": f"Trial {trial_num}: {question}"}]


def build_messages_for_task(
    task_id: str, concept: str, trial_num: int
) -> Any:
    """Build messages for a given task."""
    if task_id == "introspection":
        return build_introspection_messages(trial_num)
    elif task_id == "forced_injection":
        return build_forced_injection_messages(trial_num)
    elif task_id == "concept_recall":
        return build_concept_recall_messages(concept, trial_num)
    elif task_id == "primed_generation":
        return build_primed_generation_messages(concept, trial_num)
    elif task_id == "unrelated_baseline":
        return build_unrelated_messages(trial_num)
    else:
        raise ValueError(f"Unknown task: {task_id}")


# =============================================================================
# Tokenization Utilities
# =============================================================================

def get_head_dim(model_wrapper: ModelWrapper) -> int:
    """Get the attention head dimension."""
    config = model_wrapper.model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    if hasattr(config, 'head_dim'):
        return config.head_dim
    return model_wrapper.d_model // model_wrapper.n_heads


def get_concept_token_id(tokenizer, concept: str) -> int:
    """Get the first token ID for a concept word (with leading space)."""
    tokens = tokenizer.encode(f" {concept}", add_special_tokens=False)
    if tokens:
        return tokens[0]
    tokens = tokenizer.encode(concept, add_special_tokens=False)
    return tokens[0] if tokens else None


def format_messages(
    messages: List[Dict[str, str]], tokenizer
) -> Tuple[torch.Tensor, int, str]:
    """Format messages with chat template and tokenize.

    Uses add_special_tokens=False to avoid double BOS bug.
    """
    filtered = [
        m for m in messages
        if not (m.get("role") == "system" and m.get("content") == "")
    ]
    formatted = tokenizer.apply_chat_template(
        filtered, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


def format_messages_with_prefill(
    messages: List[Dict[str, str]], prefill: str, tokenizer
) -> Tuple[torch.Tensor, int, str]:
    """Format messages with chat template and append prefill text."""
    filtered = [
        m for m in messages
        if not (m.get("role") == "system" and m.get("content") == "")
    ]
    formatted = tokenizer.apply_chat_template(
        filtered, tokenize=False, add_generation_prompt=True
    )
    formatted = formatted + prefill
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


def find_steering_position(
    tokenizer, formatted_prompt: str, trial_num: int
) -> int:
    """Find the token position where steering should start."""
    trial_marker = f"Trial {trial_num}:"
    trial_char_pos = formatted_prompt.find(trial_marker)

    if trial_char_pos == -1:
        tokens = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=False
        )
        return max(0, len(tokens["input_ids"][0]) - 20)

    newline_pos = formatted_prompt.rfind("\n", 0, trial_char_pos)
    if newline_pos == -1:
        newline_pos = 0

    prefix = formatted_prompt[:newline_pos]
    prefix_tokens = tokenizer(
        prefix, return_tensors="pt", add_special_tokens=False
    )
    return len(prefix_tokens["input_ids"][0])


def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for Yes and No."""
    for yes_var in [" Yes", "Yes", " yes", "yes"]:
        tokens = tokenizer.encode(yes_var, add_special_tokens=False)
        if len(tokens) == 1:
            yes_id = tokens[0]
            break
    else:
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]

    for no_var in [" No", "No", " no", "no"]:
        tokens = tokenizer.encode(no_var, add_special_tokens=False)
        if len(tokens) == 1:
            no_id = tokens[0]
            break
    else:
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    return yes_id, no_id


def get_model_dimensions(model_name: str) -> Tuple[int, int]:
    """Get n_layers and n_heads for known models without loading them."""
    known_models = {
        "gemma3_27b": (62, 32),
        "gemma3_12b": (48, 16),
        "gemma3_4b": (34, 16),
        "llama3_8b": (32, 32),
        "llama3_70b": (80, 64),
    }
    if model_name in known_models:
        return known_models[model_name]
    raise ValueError(
        f"Unknown model {model_name}. Add to known_models or "
        "run without --plots-only"
    )


# =============================================================================
# Gradient Attribution
# =============================================================================

def compute_gradient_attribution_for_task(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    task_id: str,
    steering_layer: int,
    steering_strength: float,
    n_trials: int = 3,
    device: str = "cuda",
    verbose: bool = False,
    output_file: Optional[Path] = None,
    save_interval: int = 10,
) -> Dict[Tuple[int, int], float]:
    """
    Compute gradient-based attribution for each attention head for a task.

    For introspection: gradient of (logit_Yes - logit_No)
    For forced_injection: gradient of concept token logit
    For other tasks: gradient of top predicted token logit

    Supports incremental save/resume via output_file.

    Returns: Dict mapping (layer, head) -> mean gradient attribution
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    n_heads = model_wrapper.n_heads
    head_dim = get_head_dim(model_wrapper)

    task_def = TASK_DEFINITIONS[task_id]
    uses_steering = task_def["uses_steering"]

    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # Target layers: from steering layer onwards
    target_layers = list(range(steering_layer + 1, n_layers))

    # Per-concept attribution tracking
    concept_attributions = {}

    # Load partial results if resuming
    completed_concepts = set()
    if output_file and output_file.exists():
        try:
            with open(output_file) as f:
                data = json.load(f)
            if "concept_attributions" in data:
                for concept, attrs in data["concept_attributions"].items():
                    concept_attributions[concept] = {
                        tuple(map(int, k.split(","))): v
                        for k, v in attrs.items()
                    }
                    completed_concepts.add(concept)
                print(f"    Resuming: {len(completed_concepts)} concepts done")
        except Exception as e:
            print(f"    Warning: Could not load partial results: {e}")

    concepts = list(concept_vectors.keys())
    concepts_to_process = [c for c in concepts if c not in completed_concepts]

    if not concepts_to_process:
        print(f"    All {len(concepts)} concepts already completed")
        grad_activations = defaultdict(list)
        for concept, attrs in concept_attributions.items():
            for (layer, head), val in attrs.items():
                grad_activations[(layer, head)].append(val)
        return {k: float(np.mean(v)) for k, v in grad_activations.items()}

    processed_count = 0

    for concept in tqdm(concepts_to_process, desc=f"Attribution ({task_id})",
                        disable=not verbose):
        steering_vec = (
            concept_vectors[concept].to(device) if uses_steering else None
        )

        concept_trial_attrs = defaultdict(list)

        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1
            task_output = build_messages_for_task(task_id, concept, trial_num)

            # Handle forced_injection specially (has prefill)
            if task_id == "forced_injection":
                messages, prefill = task_output
                input_ids, prompt_len, formatted_prompt = (
                    format_messages_with_prefill(messages, prefill, tokenizer)
                )
                concept_token_id = get_concept_token_id(tokenizer, concept)
            else:
                messages = task_output
                input_ids, prompt_len, formatted_prompt = (
                    format_messages(messages, tokenizer)
                )
                concept_token_id = None

            input_ids = input_ids.to(device)
            steer_start_pos = find_steering_position(
                tokenizer, formatted_prompt, trial_num
            )

            # Storage for activations and gradients
            activations = {}
            gradients = {}
            hooks = []

            def make_forward_hook(layer_idx):
                def hook(module, args, output):
                    inp = args[0]
                    batch, seq, hidden = inp.shape
                    reshaped = inp.view(batch, seq, n_heads, head_dim)
                    activations[layer_idx] = reshaped.detach().clone()
                return hook

            def make_backward_hook(layer_idx):
                def hook(module, grad_input, grad_output):
                    if grad_input[0] is not None:
                        grad = grad_input[0]
                        batch, seq, hidden = grad.shape
                        reshaped = grad.view(batch, seq, n_heads, head_dim)
                        gradients[layer_idx] = reshaped.detach().clone()
                return hook

            # Register hooks on target layers
            for layer_idx in target_layers:
                layer = model_wrapper.get_layer_module(layer_idx)
                fwd_hook = layer.self_attn.o_proj.register_forward_hook(
                    make_forward_hook(layer_idx)
                )
                bwd_hook = layer.self_attn.o_proj.register_full_backward_hook(
                    make_backward_hook(layer_idx)
                )
                hooks.extend([fwd_hook, bwd_hook])

            # Steering hook
            if uses_steering and steering_vec is not None:
                def make_steering_hook(start_pos, steer_vec):
                    def hook(module, args, output):
                        hidden = (
                            output[0] if isinstance(output, tuple) else output
                        )
                        seq_len = hidden.shape[1]
                        if start_pos < seq_len:
                            hidden = hidden.clone()
                            hidden[:, start_pos:, :] = (
                                hidden[:, start_pos:, :]
                                + steering_strength
                                * steer_vec.to(hidden.dtype)
                            )
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden
                    return hook

                steering_module = model_wrapper.get_layer_module(
                    steering_layer
                )
                steering_handle = steering_module.register_forward_hook(
                    make_steering_hook(steer_start_pos, steering_vec)
                )
                hooks.append(steering_handle)

            try:
                model.eval()
                with torch.enable_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[0, -1, :]

                    if task_id == "introspection":
                        target_logit = logits[yes_id] - logits[no_id]
                    elif task_id == "forced_injection":
                        target_logit = logits[concept_token_id]
                    else:
                        target_logit = logits.max()

                    target_logit.backward()

                    for layer_idx in target_layers:
                        if (layer_idx in activations
                                and layer_idx in gradients):
                            act = activations[layer_idx][:, -1, :, :]
                            grad = gradients[layer_idx][:, -1, :, :]
                            for h in range(n_heads):
                                attr = (
                                    act[0, h, :] * grad[0, h, :]
                                ).sum().item()
                                concept_trial_attrs[
                                    (layer_idx, h)
                                ].append(attr)

            finally:
                for hook in hooks:
                    hook.remove()
                activations.clear()
                gradients.clear()

            del outputs, input_ids
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # Average across trials for this concept
        concept_attributions[concept] = {
            (layer, head): float(np.mean(attrs))
            for (layer, head), attrs in concept_trial_attrs.items()
        }
        processed_count += 1

        # Incremental save
        if output_file and processed_count % save_interval == 0:
            _save_attribution(
                output_file, concept_attributions, task_id,
                len(concept_vectors), n_trials
            )

    # Final save
    if output_file:
        _save_attribution(
            output_file, concept_attributions, task_id,
            len(concept_vectors), n_trials
        )

    # Aggregate across all concepts
    grad_activations = defaultdict(list)
    for concept, attrs in concept_attributions.items():
        for (layer, head), val in attrs.items():
            grad_activations[(layer, head)].append(val)

    return {
        (layer, head): float(np.mean(vals))
        for (layer, head), vals in grad_activations.items()
    }


def _save_attribution(
    output_file: Path,
    concept_attributions: Dict[str, Dict[Tuple[int, int], float]],
    task_id: str,
    total_concepts: int,
    n_trials: int,
):
    """Save attribution results with per-concept data for resume."""
    grad_activations = defaultdict(list)
    for concept, attrs in concept_attributions.items():
        for (layer, head), val in attrs.items():
            grad_activations[(layer, head)].append(val)

    attribution = {
        f"{layer},{head}": float(np.mean(vals))
        for (layer, head), vals in grad_activations.items()
    }

    concept_attr_serialized = {
        concept: {
            f"{layer},{head}": val
            for (layer, head), val in attrs.items()
        }
        for concept, attrs in concept_attributions.items()
    }

    task_def = TASK_DEFINITIONS[task_id]
    data = {
        "attribution": attribution,
        "concept_attributions": concept_attr_serialized,
        "task_id": task_id,
        "task_name": task_def["name"],
        "n_concepts_completed": len(concept_attributions),
        "n_concepts_total": total_concepts,
        "n_trials": n_trials,
        "is_partial": len(concept_attributions) < total_concepts,
        "timestamp": datetime.now().isoformat(),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Analysis
# =============================================================================

def compute_task_correlation(
    attributions: Dict[str, Dict[Tuple[int, int], float]],
    task_a: str,
    task_b: str,
) -> Tuple[float, float]:
    """Compute Spearman correlation between head attributions for two tasks."""
    attr_a = attributions[task_a]
    attr_b = attributions[task_b]
    common_heads = set(attr_a.keys()) & set(attr_b.keys())
    if len(common_heads) < 10:
        return 0.0, 1.0
    vals_a = [attr_a[h] for h in common_heads]
    vals_b = [attr_b[h] for h in common_heads]
    corr, pval = stats.spearmanr(vals_a, vals_b)
    return corr, pval


def compute_top_head_overlap(
    attributions: Dict[str, Dict[Tuple[int, int], float]],
    task_a: str,
    task_b: str,
    top_k: int = 20,
) -> float:
    """Compute overlap in top-K heads between two tasks."""
    attr_a = attributions[task_a]
    attr_b = attributions[task_b]
    top_a = set(sorted(attr_a.keys(), key=lambda h: -abs(attr_a[h]))[:top_k])
    top_b = set(sorted(attr_b.keys(), key=lambda h: -abs(attr_b[h]))[:top_k])
    return len(top_a & top_b) / top_k


def identify_task_specific_heads(
    attributions: Dict[str, Dict[Tuple[int, int], float]],
    target_task: str = "introspection",
    comparison_tasks: List[str] = None,
    top_k: int = 50,
) -> Dict[str, Any]:
    """
    Identify heads specifically important for the target task.

    Specificity score = attr[target] - mean(attr[comparisons])
    """
    if comparison_tasks is None:
        comparison_tasks = [
            t for t in attributions.keys() if t != target_task
        ]

    target_attr = attributions[target_task]
    all_heads = list(target_attr.keys())

    specificity_scores = {}
    for head in all_heads:
        target_score = target_attr.get(head, 0)
        comparison_scores = [
            attributions[t].get(head, 0) for t in comparison_tasks
        ]
        mean_comparison = (
            np.mean(comparison_scores) if comparison_scores else 0
        )
        specificity_scores[head] = {
            "specificity": target_score - mean_comparison,
            "target_score": target_score,
            "mean_comparison_score": mean_comparison,
        }

    sorted_heads = sorted(
        specificity_scores.keys(),
        key=lambda h: -specificity_scores[h]["specificity"],
    )

    return {
        "specific_heads": sorted_heads[:top_k],
        "scores": {
            h: specificity_scores[h] for h in sorted_heads[:top_k]
        },
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_attribution_heatmaps(
    attributions: Dict[str, Dict[Tuple[int, int], float]],
    n_layers: int,
    n_heads: int,
    steering_layer: int,
    output_dir: Path,
):
    """Create per-task attribution heatmaps and correlation matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(attributions.keys())
    n_tasks = len(tasks)
    target_layers = list(range(steering_layer + 1, n_layers))

    # --- Side-by-side heatmaps ---
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 11))
    if n_tasks == 1:
        axes = [axes]

    for idx, task in enumerate(tasks):
        attr = attributions[task]
        matrix = np.zeros((len(target_layers), n_heads))

        for (layer, head), score in attr.items():
            if layer > steering_layer:
                row = layer - steering_layer - 1
                if row < len(target_layers) and head < n_heads:
                    matrix[row, head] = score

        ax = axes[idx]
        vmax = max(abs(matrix.min()), abs(matrix.max()))
        im = ax.imshow(
            matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax
        )
        ax.set_xlabel("Head", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)

        task_def = TASK_DEFINITIONS[task]
        ax.set_title(
            f"{task_def['name']}\n{task_def['target_description']}",
            fontsize=9, linespacing=1.4,
        )

        ax.set_yticks(range(0, len(target_layers), 5))
        ax.set_yticklabels([
            str(steering_layer + 1 + i)
            for i in range(0, len(target_layers), 5)
        ])
        ax.set_xticks(range(0, n_heads, 8))
        ax.set_xticklabels([str(h) for h in range(0, n_heads, 8)])
        plt.colorbar(im, ax=ax, label="Attribution", fraction=0.046, pad=0.04)

    formula = (
        r"$\mathrm{Attribution}_h = \sum_i "
        r"\frac{\partial \mathcal{L}}{\partial o_h^{(i)}} "
        r"\cdot o_h^{(i)}$"
    )
    plt.suptitle(
        f"Gradient attribution by task (red=helps, blue=hurts)\n{formula}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        output_dir / "attribution_comparison_heatmaps.png",
        dpi=150, bbox_inches='tight',
    )
    plt.close()

    # --- Correlation matrix ---
    if n_tasks > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = np.zeros((n_tasks, n_tasks))

        for i, task_a in enumerate(tasks):
            for j, task_b in enumerate(tasks):
                corr, _ = compute_task_correlation(
                    attributions, task_a, task_b
                )
                corr_matrix[i, j] = corr

        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels(
            [TASK_DEFINITIONS[t]['name'] for t in tasks],
            rotation=45, ha='right',
        )
        ax.set_yticklabels(
            [TASK_DEFINITIONS[t]['name'] for t in tasks]
        )

        for i in range(n_tasks):
            for j in range(n_tasks):
                ax.text(
                    j, i, f"{corr_matrix[i, j]:.2f}",
                    ha='center', va='center', fontsize=10,
                )

        plt.colorbar(im, ax=ax, label="Spearman correlation")
        ax.set_title("Task attribution correlation matrix", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            output_dir / "task_correlation_matrix.png",
            dpi=150, bbox_inches='tight',
        )
        plt.close()

    # --- Scatter: introspection vs other tasks ---
    if "introspection" in attributions and n_tasks > 1:
        other_tasks = [t for t in tasks if t != "introspection"]
        fig, axes = plt.subplots(
            1, len(other_tasks), figsize=(5 * len(other_tasks), 5)
        )
        if len(other_tasks) == 1:
            axes = [axes]

        intro_attr = attributions["introspection"]

        for idx, other_task in enumerate(other_tasks):
            other_attr = attributions[other_task]
            common_heads = set(intro_attr.keys()) & set(other_attr.keys())
            x = [intro_attr[h] for h in common_heads]
            y = [other_attr[h] for h in common_heads]

            ax = axes[idx]
            ax.scatter(x, y, alpha=0.5, s=10)
            max_val = max(max(x), max(y))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')

            corr, pval = stats.spearmanr(x, y)
            ax.set_xlabel("Introspection attribution", fontsize=10)
            ax.set_ylabel(
                f"{TASK_DEFINITIONS[other_task]['name']} attribution",
                fontsize=10,
            )
            ax.set_title(
                f"Introspection vs {TASK_DEFINITIONS[other_task]['name']}\n"
                f"(r={corr:.3f}, p={pval:.2e})",
                fontsize=11,
            )
            ax.legend()

        plt.tight_layout()
        plt.savefig(
            output_dir / "introspection_vs_others_scatter.png",
            dpi=150, bbox_inches='tight',
        )
        plt.close()


def plot_specific_heads(
    specific_analysis: Dict[str, Any],
    attributions: Dict[str, Dict[Tuple[int, int], float]],
    output_dir: Path,
    top_k: int = 30,
):
    """Visualize task-specific heads."""
    output_dir.mkdir(parents=True, exist_ok=True)

    specific_heads = specific_analysis["specific_heads"][:top_k]
    scores = specific_analysis["scores"]

    # Specificity bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    head_labels = [f"L{h[0]}H{h[1]}" for h in specific_heads]
    specificity_vals = [scores[h]["specificity"] for h in specific_heads]
    colors = ['indianred' if v > 0 else 'steelblue' for v in specificity_vals]

    ax.bar(range(len(specific_heads)), specificity_vals, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(specific_heads)))
    ax.set_xticklabels(head_labels, rotation=90, fontsize=8)
    ax.set_xlabel("Attention head", fontsize=11)
    ax.set_ylabel(
        "Specificity score\n(introspection - mean other tasks)", fontsize=11
    )
    ax.set_title(
        "Introspection-specific heads\n"
        "(positive = more important for introspection)",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "introspection_specific_heads.png",
        dpi=150, bbox_inches='tight',
    )
    plt.close()


# =============================================================================
# Main Experiment Runner
# =============================================================================

def load_cached_attributions(
    output_dir: Path,
) -> Dict[str, Dict[Tuple[int, int], float]]:
    """Load cached attribution results from JSON files."""
    attributions = {}
    for task_id in TASK_DEFINITIONS.keys():
        results_file = output_dir / f"attribution_{task_id}.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            attribution = {
                tuple(map(int, k.split(","))): v
                for k, v in data["attribution"].items()
            }
            attributions[task_id] = attribution
            print(f"  Loaded {task_id}: {len(attribution)} heads")
        else:
            print(f"  Warning: {results_file} not found, skipping {task_id}")
    return attributions


def run_experiment(args, output_dir: Path):
    """Run the head investigation experiment (task specificity analysis)."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 51: ATTENTION HEAD INVESTIGATION")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    # --- Plots-only mode ---
    if args.plots_only:
        print("\n[--plots-only] Regenerating plots from cached data...")
        attributions = load_cached_attributions(output_dir)
        if not attributions:
            print("Error: No cached results found. Run without --plots-only.")
            return
        n_layers, n_heads = get_model_dimensions(args.model)
        print(f"Model dimensions: {n_layers} layers, {n_heads} heads")

    else:
        # Save config
        config = vars(args).copy()
        config["timestamp"] = datetime.now().isoformat()
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Load model
        print(f"\nLoading model: {args.model}")
        model_wrapper = load_model(
            args.model,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
        )
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        n_layers = model_wrapper.n_layers
        n_heads = model_wrapper.n_heads

        print(f"Model: {n_layers} layers, {n_heads} heads per layer")
        print(f"Steering at layer {args.layer} with strength {args.strength}")

        # Load and filter concept vectors
        print("\nLoading concept vectors...")
        concept_vectors = load_concept_vectors(
            args.steering_dir, args.model, args.concepts,
            n_sample=args.n_concepts, layer=args.layer,
        )
        print(f"Loaded {len(concept_vectors)} concepts")

        print(f"Filtering concepts by partition: {args.partition}...")
        success_concepts, failure_concepts = load_concept_classification(
            args.model, args.layer, args.strength
        )
        concept_vectors = filter_concepts_by_partition(
            concept_vectors, success_concepts, failure_concepts, args.partition
        )
        print(f"After filtering: {len(concept_vectors)} concepts")

        # Run gradient attribution for each task
        print(f"\n{'=' * 70}")
        print("RUNNING GRADIENT ATTRIBUTION FOR EACH TASK")
        print(f"{'=' * 70}")

        attributions = {}
        if args.tasks:
            for t in args.tasks:
                if t not in TASK_DEFINITIONS:
                    print(
                        f"Error: Unknown task '{t}'. "
                        f"Valid: {list(TASK_DEFINITIONS.keys())}"
                    )
                    return
            tasks_to_run = args.tasks
        else:
            tasks_to_run = list(TASK_DEFINITIONS.keys())

        for task_id in tasks_to_run:
            task_def = TASK_DEFINITIONS[task_id]
            print(f"\n--- Task: {task_def['name']} ---")
            print(f"    {task_def['description']}")

            results_file = output_dir / f"attribution_{task_id}.json"

            # Check for complete cached results
            skip_task = False
            if results_file.exists() and not args.overwrite:
                with open(results_file) as f:
                    data = json.load(f)
                if not data.get("is_partial", False):
                    print("    Loading complete cached results...")
                    attribution = {
                        tuple(map(int, k.split(","))): v
                        for k, v in data["attribution"].items()
                    }
                    skip_task = True
                else:
                    n_done = data.get('n_concepts_completed', 0)
                    n_total = data.get('n_concepts_total', '?')
                    print(f"    Partial results ({n_done}/{n_total}), "
                          "resuming...")

            if not skip_task:
                attribution = compute_gradient_attribution_for_task(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    concept_vectors=concept_vectors,
                    task_id=task_id,
                    steering_layer=args.layer,
                    steering_strength=args.strength,
                    n_trials=args.n_trials,
                    device=args.device,
                    verbose=args.verbose,
                    output_file=results_file,
                    save_interval=25,
                )

            attributions[task_id] = attribution

            # Print top heads
            sorted_heads = sorted(
                attribution.items(), key=lambda x: -abs(x[1])
            )
            print(f"\n    Top 10 heads for {task_def['name']}:")
            for (layer, head), score in sorted_heads[:10]:
                print(f"      L{layer}H{head}: {score:+.4f}")

    # ===== Analysis =====
    print(f"\n{'=' * 70}")
    print("ANALYSIS: TASK SPECIFICITY")
    print(f"{'=' * 70}")

    tasks = list(attributions.keys())

    # Correlation analysis
    print("\n1. CORRELATION BETWEEN TASKS")
    print("-" * 40)
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1:]:
            corr, pval = compute_task_correlation(
                attributions, task_a, task_b
            )
            print(
                f"   {TASK_DEFINITIONS[task_a]['name']} vs "
                f"{TASK_DEFINITIONS[task_b]['name']}: "
                f"r={corr:.3f} (p={pval:.2e})"
            )

    # Top-K overlap
    print("\n2. TOP-20 HEAD OVERLAP")
    print("-" * 40)
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1:]:
            overlap = compute_top_head_overlap(
                attributions, task_a, task_b, top_k=20
            )
            print(
                f"   {TASK_DEFINITIONS[task_a]['name']} vs "
                f"{TASK_DEFINITIONS[task_b]['name']}: {overlap:.0%}"
            )

    # Task-specific heads
    if "introspection" in attributions:
        print("\n3. INTROSPECTION-SPECIFIC HEADS")
        print("-" * 40)
        specific_analysis = identify_task_specific_heads(
            attributions, target_task="introspection", top_k=30
        )

        print("   Top 15 introspection-specific heads:")
        for i, head in enumerate(specific_analysis["specific_heads"][:15]):
            scores = specific_analysis["scores"][head]
            print(
                f"   {i+1:2d}. L{head[0]}H{head[1]}: "
                f"specificity={scores['specificity']:.4f} "
                f"(intro={scores['target_score']:.4f}, "
                f"others_mean={scores['mean_comparison_score']:.4f})"
            )

    # Save analysis results
    analysis_results = {
        "correlations": {},
        "top_k_overlaps": {},
    }
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1:]:
            key = f"{task_a}_vs_{task_b}"
            corr, pval = compute_task_correlation(
                attributions, task_a, task_b
            )
            overlap = compute_top_head_overlap(
                attributions, task_a, task_b, top_k=20
            )
            analysis_results["correlations"][key] = {
                "correlation": corr, "p_value": pval
            }
            analysis_results["top_k_overlaps"][key] = overlap

    if "introspection" in attributions:
        analysis_results["specific_heads"] = {
            "heads": [
                list(h)
                for h in specific_analysis["specific_heads"]
            ],
            "scores": {
                f"{h[0]},{h[1]}": s
                for h, s in specific_analysis["scores"].items()
            },
        }

    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

    # ===== Visualizations =====
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 70}")

    plot_attribution_heatmaps(
        attributions, n_layers, n_heads, args.layer, plots_dir
    )

    if "introspection" in attributions:
        plot_specific_heads(specific_analysis, attributions, plots_dir)

    print(f"\nPlots saved to: {plots_dir}")
    print(f"Results saved to: {output_dir}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output directory
    layer_strength_dir = f"layer_{args.layer}_strength_{args.strength}"
    base_dir = Path(args.output_dir) / args.model / layer_strength_dir
    section_dir = base_dir / "section_a" / args.partition
    section_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(args, section_dir)


if __name__ == "__main__":
    main()
