#!/usr/bin/env python3
"""
Experiment 51: Automatic Attention Head Identification for Introspection

This experiment implements two principled approaches to identify causally important
attention heads for introspection:

1. **Signed Steering Alignment**: Separates heads into:
   - Amplifiers (positive alignment): shift output in steering direction
   - Suppressors (negative alignment): shift output against steering direction

   Key insight: Previous exp12 used |cosine| which loses sign information.
   A head with -0.8 alignment (suppressor) looks the same as +0.8 (amplifier).

2. **Gradient Attribution**: Computes ∂(logit_Yes - logit_No)/∂(head_output)
   This directly measures: "How much would changing this head's output change
   the detection decision?"

Key methodological fixes from exp12:
- Signed alignment metric preserves amplifier vs suppressor distinction
- Gradient-based attribution for direct causal measurement
- Prompt-level patching at ALL token positions (not just last)
- Proper matched control trials for ablation

Usage:
    python 10_head_identification.py -m gemma3_27b
    python 10_head_identification.py -m gemma3_27b --sections gradient signed_alignment
    python 10_head_identification.py -m gemma3_27b --sections ablation -nc 10 -nt 10
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
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from model_utils import ModelWrapper, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic attention head identification for introspection")
    parser.add_argument("-m", "--model", type=str, default="gemma3_27b")
    parser.add_argument("-c", "--concepts", nargs="+", default=None, help="Specific concepts to test")
    parser.add_argument("-nc", "--n-concepts", type=int, default=500, help="Number of concepts to sample if not specified (default: 500 = all)")
    parser.add_argument("-nt", "--n-trials", type=int, default=5, help="Trials per concept")
    parser.add_argument("-l", "--layer", type=int, default=38, help="Steering layer (default: 38)")
    parser.add_argument("-s", "--strength", type=float, default=4.0, help="Steering strength (default: 4.0)")
    parser.add_argument("--steering-dir", type=str, default="analysis/02b_steering_500_concepts", help="Path to experiment 02 (steering evaluation) results for concept vectors")
    parser.add_argument("-od", "--output-dir", type=str, default="analysis/10_head_identification")
    parser.add_argument("-mt", "--max-tokens", type=int, default=100)
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-dt", "--dtype", type=str, default="bfloat16")
    parser.add_argument("-q", "--quantization", type=str, default=None, choices=["8bit", "4bit", None])
    parser.add_argument("--sections", nargs="+", default=["gradient", "signed_alignment", "ablation"], choices=["gradient", "signed_alignment", "ablation"], help="Sections to run")
    parser.add_argument("--gradient-modes", nargs="+", default=["all", "successful", "differential"], choices=["all", "successful", "differential"], help="Gradient attribution modes to run")
    parser.add_argument("--ablation-percentages", nargs="+", type=float, default=[1, 2, 3, 5, 10, 20], help="Percentages of heads to ablate")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM judge evaluation (use keyword matching)")
    return parser.parse_args()


# =============================================================================
# Utility Functions
# =============================================================================

def build_messages(trial_num: int = 1) -> List[Dict[str, str]]:
    """Build the standard introspection prompt as multi-turn messages.

    This matches the canonical format used in 02_steering_evaluation.py and
    11_head_ablation.py for consistency across experiments.
    """
    preamble = (
        "I am an interpretability researcher studying transformer-based language models like you. "
        "I have access to the internals of your neural network. I have identified patterns in your "
        "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
        "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
        "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
    )
    trial_prompt = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"

    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": preamble},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": trial_prompt},
    ]


def load_concept_vectors(
    steering_dir: str,
    model_name: str,
    concepts: Optional[List[str]] = None,
    n_sample: int = 20,
    layer_fraction: float = 0.61
) -> Dict[str, torch.Tensor]:
    """Load concept vectors from experiment 02 (steering evaluation).

    experiment 02 (steering evaluation) stores vectors in subdirectories by layer fraction:
    vectors/layer_0.61/Bread.pt, etc.
    """
    base_vectors_dir = Path(steering_dir) / model_name / "vectors"

    if not base_vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {base_vectors_dir}")

    # Find the layer subdirectory closest to requested fraction
    layer_dirs = sorted(base_vectors_dir.glob("layer_*"))
    if not layer_dirs:
        # Try direct .pt files (fallback)
        vectors_dir = base_vectors_dir
    else:
        # Find closest layer fraction
        best_dir = None
        best_diff = float('inf')
        for layer_dir in layer_dirs:
            try:
                frac = float(layer_dir.name.replace("layer_", ""))
                diff = abs(frac - layer_fraction)
                if diff < best_diff:
                    best_diff = diff
                    best_dir = layer_dir
            except ValueError:
                continue

        if best_dir is None:
            raise FileNotFoundError(f"No layer directories found in {base_vectors_dir}")

        vectors_dir = best_dir
        print(f"Using vectors from: {vectors_dir}")

    available = sorted([f.stem for f in vectors_dir.glob("*.pt")])

    if not available:
        raise FileNotFoundError(f"No .pt files found in {vectors_dir}")

    if concepts is not None:
        missing = set(concepts) - set(available)
        if missing:
            print(f"Warning: Missing concept vectors: {missing}")
        selected = [c for c in concepts if c in available]
    else:
        selected = random.sample(available, min(n_sample, len(available)))

    vectors = {}
    for concept in selected:
        vec_path = vectors_dir / f"{concept}.pt"
        vectors[concept] = torch.load(vec_path, weights_only=True)

    return vectors


def load_successful_concepts(
    steering_dir: str,
    model_name: str,
    layer_fraction: float = 0.61,
    strength: float = 4.0,
    min_detections: int = 1
) -> List[str]:
    """Load list of concepts that had at least min_detections successful detections.

    A detection is considered successful if claims_detection is True in the LLM judge evaluation.
    """
    # Find the results file
    results_path = None
    base_dir = Path(steering_dir) / model_name

    # Look for results matching the layer and strength
    for subdir in base_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("layer_"):
            try:
                parts = subdir.name.split("_")
                frac_str = parts[1]
                strength_str = parts[3] if len(parts) > 3 else None

                if abs(float(frac_str) - layer_fraction) < 0.01:
                    if strength_str and abs(float(strength_str) - strength) < 0.1:
                        results_path = subdir / "results.json"
                        break
            except (ValueError, IndexError):
                continue

    if results_path is None or not results_path.exists():
        print(f"Warning: Could not find results.json for layer {layer_fraction} strength {strength}")
        return []

    with open(results_path) as f:
        data = json.load(f)

    # Count detections per concept
    concept_detections = {}
    for r in data.get("results", []):
        if r.get("trial_type") == "injection":
            concept = r["concept"]
            detected = r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False)
            if concept not in concept_detections:
                concept_detections[concept] = 0
            if detected:
                concept_detections[concept] += 1

    # Filter to concepts with at least min_detections
    successful = [c for c, count in concept_detections.items() if count >= min_detections]

    return successful


def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for Yes and No responses."""
    # Try space-prefixed first (common in chat templates)
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


def format_messages_for_model(messages: List[Dict[str, str]], tokenizer) -> Tuple[torch.Tensor, int, str]:
    """Format multi-turn messages with chat template and tokenize.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: The model's tokenizer

    Returns:
        Tuple of (input_ids tensor, prompt length, formatted string)
    """
    # Filter out empty system messages if the model doesn't support them
    filtered_messages = [m for m in messages if not (m["role"] == "system" and m["content"] == "")]

    formatted = tokenizer.apply_chat_template(
        filtered_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Use add_special_tokens=False for consistency with find_steering_start_position
    # The chat template already includes BOS token in the formatted string
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], len(inputs["input_ids"][0]), formatted


def find_steering_start_position(tokenizer, formatted_prompt: str, trial_num: int) -> int:
    """
    Find the token position where steering should start (at 'Trial X:').

    Steering should only be applied from the trial question onwards, not during
    the preamble or the model's "Ok." acknowledgment.

    Args:
        tokenizer: The model's tokenizer
        formatted_prompt: The full formatted prompt string
        trial_num: The trial number (1-indexed)

    Returns:
        Token position where steering should start
    """
    # Find "Trial X:" in the formatted prompt
    trial_marker = f"Trial {trial_num}:"
    trial_char_pos = formatted_prompt.find(trial_marker)

    if trial_char_pos == -1:
        raise ValueError(f"Trial marker '{trial_marker}' not found in formatted prompt")

    # Find the newline before "Trial X:" (usually marks the user turn)
    newline_pos = formatted_prompt.rfind("\n", 0, trial_char_pos)
    if newline_pos == -1:
        newline_pos = 0

    # Tokenize up to that position to get token count
    prefix = formatted_prompt[:newline_pos]
    prefix_tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    steering_start = len(prefix_tokens["input_ids"][0])

    return steering_start


def evaluate_detection(response: str, concept: str = None) -> Dict[str, bool]:
    """Simple keyword-based detection evaluation."""
    response_lower = response.lower()[:200]

    # Check for Yes/No in first part of response
    detected = False
    if response_lower.startswith("yes") or " yes" in response_lower[:50]:
        if not response_lower.startswith("no") and " no" not in response_lower[:30]:
            detected = True

    # Check for concept identification
    identified = False
    if detected and concept:
        concept_lower = concept.lower()
        if concept_lower in response_lower:
            identified = True

    return {
        "detected": detected,
        "identified": identified
    }


def get_head_dim(model_wrapper: ModelWrapper) -> int:
    """Get the attention head dimension from the model config."""
    config = model_wrapper.model.config

    # Try text_config first (for multimodal models like Gemma3)
    if hasattr(config, 'text_config'):
        config = config.text_config

    # Try direct head_dim attribute
    if hasattr(config, 'head_dim'):
        return config.head_dim

    # Fallback: compute from hidden_size and n_heads
    # (but this may be wrong for models with GQA)
    return model_wrapper.d_model // model_wrapper.n_heads


# =============================================================================
# Gradient Attribution
# =============================================================================

def compute_gradient_attribution(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    steering_strength: float,
    n_trials: int = 5,
    device: str = "cuda",
    verbose: bool = False,
    mode: str = "standard"
) -> Dict[Tuple[int, int], float]:
    """
    Compute gradient-based attribution for each attention head.

    Modes:
    - "standard": |∂(Yes-No)/∂(head_output)| (original method)
    - "differential": |∂Yes/∂head| - |∂No/∂head| (positive = yes-biased, negative = no-biased)

    Method:
    1. Forward pass with steering applied
    2. Compute detection_logit = logit(Yes) - logit(No)
    3. Use backward hooks to capture gradients at each layer's attention output
    4. For each head: importance = |grad . activation| (gradient × input product)

    This measures: "How much would changing this head's output change the detection decision?"
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    n_heads = model_wrapper.n_heads
    head_dim = get_head_dim(model_wrapper)

    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # We'll analyze heads from steering layer onwards
    target_layers = list(range(steering_layer + 1, n_layers))  # Start from layer AFTER steering

    # Accumulate gradient magnitudes
    if mode == "differential":
        yes_grad_activations = defaultdict(list)  # (layer, head) -> list of |grad_yes . act|
        no_grad_activations = defaultdict(list)   # (layer, head) -> list of |grad_no . act|
    else:
        grad_activations = defaultdict(list)  # (layer, head) -> list of |grad . act|

    concepts = list(concept_vectors.keys())

    for concept in tqdm(concepts, desc=f"Gradient attribution ({mode})", disable=not verbose):
        steering_vec = concept_vectors[concept].to(device)

        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1
            messages = build_messages(trial_num=trial_num)
            input_ids, prompt_len, formatted_prompt = format_messages_for_model(messages, tokenizer)
            input_ids = input_ids.to(device)

            # Find where steering should start (at "Trial X:")
            steer_start_pos = find_steering_start_position(tokenizer, formatted_prompt, trial_num)

            # Storage for activations and gradients
            activations = {}  # layer_idx -> [batch, seq, n_heads, head_dim]
            gradients = {}    # layer_idx -> [batch, seq, n_heads, head_dim]
            hooks = []

            def make_forward_hook(layer_idx):
                def hook(module, args, output):
                    # args[0] is the input to o_proj: [batch, seq, n_heads*head_dim]
                    inp = args[0]
                    batch, seq, hidden = inp.shape
                    # Reshape and store
                    reshaped = inp.view(batch, seq, n_heads, head_dim)
                    activations[layer_idx] = reshaped.detach().clone()
                return hook

            def make_backward_hook(layer_idx):
                def hook(module, grad_input, grad_output):
                    # grad_input[0] is the gradient w.r.t. the input to o_proj
                    if grad_input[0] is not None:
                        grad = grad_input[0]
                        batch, seq, hidden = grad.shape
                        reshaped = grad.view(batch, seq, n_heads, head_dim)
                        gradients[layer_idx] = reshaped.detach().clone()
                return hook

            # Register hooks
            for layer_idx in target_layers:
                layer = model_wrapper.get_layer_module(layer_idx)
                fwd_hook = layer.self_attn.o_proj.register_forward_hook(
                    make_forward_hook(layer_idx)
                )
                bwd_hook = layer.self_attn.o_proj.register_full_backward_hook(
                    make_backward_hook(layer_idx)
                )
                hooks.extend([fwd_hook, bwd_hook])

            # Steering hook - only steer from trial question onwards
            def make_steering_hook(start_pos, steer_vec):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    seq_len = hidden.shape[1]
                    if start_pos < seq_len:
                        hidden = hidden.clone()
                        hidden[:, start_pos:, :] = hidden[:, start_pos:, :] + steering_strength * steer_vec.to(hidden.dtype)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            steering_module = model_wrapper.get_layer_module(steering_layer)
            steering_handle = steering_module.register_forward_hook(
                make_steering_hook(steer_start_pos, steering_vec)
            )
            hooks.append(steering_handle)

            try:
                model.eval()
                # Need gradients for backward hooks
                with torch.enable_grad():
                    # Forward pass
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[0, -1, :]  # [vocab_size]

                    if mode == "differential":
                        # Compute gradients for Yes and No separately
                        yes_logit = logits[yes_id]
                        no_logit = logits[no_id]

                        # Gradient w.r.t Yes
                        yes_logit.backward(retain_graph=True)

                        # Store Yes gradients
                        yes_grads = {}
                        for layer_idx in target_layers:
                            if layer_idx in gradients:
                                yes_grads[layer_idx] = gradients[layer_idx].clone()

                        # Clear gradients for next backward
                        model.zero_grad()
                        gradients.clear()

                        # Gradient w.r.t No
                        no_logit.backward()

                        # Compute differential attribution
                        for layer_idx in target_layers:
                            if layer_idx in activations and layer_idx in yes_grads and layer_idx in gradients:
                                act = activations[layer_idx][:, -1, :, :]  # [batch, n_heads, head_dim]
                                yes_grad = yes_grads[layer_idx][:, -1, :, :]
                                no_grad = gradients[layer_idx][:, -1, :, :]

                                for h in range(n_heads):
                                    yes_attr = (act[0, h, :] * yes_grad[0, h, :]).abs().sum().item()
                                    no_attr = (act[0, h, :] * no_grad[0, h, :]).abs().sum().item()
                                    yes_grad_activations[(layer_idx, h)].append(yes_attr)
                                    no_grad_activations[(layer_idx, h)].append(no_attr)
                    else:
                        # Standard mode: gradient of (Yes - No)
                        detection_logit = logits[yes_id] - logits[no_id]
                        detection_logit.backward()

                        # Compute attribution: |grad . activation| at last token
                        for layer_idx in target_layers:
                            if layer_idx in activations and layer_idx in gradients:
                                act = activations[layer_idx][:, -1, :, :]  # [batch, n_heads, head_dim]
                                grad = gradients[layer_idx][:, -1, :, :]

                                for h in range(n_heads):
                                    # Element-wise product, then take absolute mean
                                    attr = (act[0, h, :] * grad[0, h, :]).abs().sum().item()
                                    grad_activations[(layer_idx, h)].append(attr)

            finally:
                for hook in hooks:
                    hook.remove()
                # Clear all stored tensors
                for k in list(activations.keys()):
                    del activations[k]
                for k in list(gradients.keys()):
                    del gradients[k]
                activations.clear()
                gradients.clear()

            # Aggressively clear memory after each trial
            del outputs, input_ids
            try:
                del yes_grads
            except NameError:
                pass
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    # Aggregate across trials
    attribution = {}
    if mode == "differential":
        for (layer, head) in yes_grad_activations.keys():
            yes_mean = float(np.mean(yes_grad_activations[(layer, head)]))
            no_mean = float(np.mean(no_grad_activations[(layer, head)]))
            # Positive = more sensitive to Yes, Negative = more sensitive to No
            attribution[(layer, head)] = yes_mean - no_mean
    else:
        for (layer, head), attrs in grad_activations.items():
            attribution[(layer, head)] = float(np.mean(attrs))

    return attribution


# =============================================================================
# Signed Steering Alignment
# =============================================================================

def compute_signed_steering_alignment(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    steering_layer: int,
    steering_strength: float,
    n_trials: int = 5,
    device: str = "cuda",
    verbose: bool = False
) -> Dict[Tuple[int, int], float]:
    """
    Compute SIGNED steering alignment for each attention head.

    For each head:
    1. Run forward pass WITH steering, get head output at last token
    2. Run forward pass WITHOUT steering, get head output at last token
    3. delta = head_output_steered - head_output_control
    4. Project delta through O-weight to get contribution to residual stream
    5. Compute SIGNED cosine similarity with steering vector

    Positive alignment = head amplifies steering (potential helper)
    Negative alignment = head opposes steering (potential suppressor)

    Key difference from exp12: We keep the sign, not just magnitude!
    """
    model = model_wrapper.model
    n_layers = model_wrapper.n_layers
    n_heads = model_wrapper.n_heads
    head_dim = get_head_dim(model_wrapper)

    target_layers = list(range(steering_layer + 1, n_layers))  # Start from layer AFTER steering

    alignments = defaultdict(list)  # (layer, head) -> list of signed alignments

    concepts = list(concept_vectors.keys())

    for concept in tqdm(concepts, desc="Signed alignment", disable=not verbose):
        steering_vec = concept_vectors[concept].to(device)
        steering_vec_flat = steering_vec.flatten().float()

        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1
            messages = build_messages(trial_num=trial_num)
            input_ids, prompt_len, formatted_prompt = format_messages_for_model(messages, tokenizer)
            input_ids = input_ids.to(device)

            # Find where steering should start (at "Trial X:")
            steer_start_pos = find_steering_start_position(tokenizer, formatted_prompt, trial_num)

            # Storage for head outputs
            control_outputs = {}  # layer -> [batch, seq, n_heads, head_dim]
            steered_outputs = {}

            def make_capture_hook(storage, layer_idx):
                def hook(module, args, output):
                    inp = args[0]  # [batch, seq, hidden]
                    batch, seq, hidden = inp.shape
                    reshaped = inp.view(batch, seq, n_heads, head_dim)
                    storage[layer_idx] = reshaped.detach().clone()
                    return output
                return hook

            # Run CONTROL (no steering)
            hooks = []
            for layer_idx in target_layers:
                layer = model_wrapper.get_layer_module(layer_idx)
                hook = layer.self_attn.o_proj.register_forward_hook(
                    make_capture_hook(control_outputs, layer_idx)
                )
                hooks.append(hook)

            with torch.no_grad():
                model(input_ids=input_ids)

            for hook in hooks:
                hook.remove()

            # Run STEERED
            hooks = []
            for layer_idx in target_layers:
                layer = model_wrapper.get_layer_module(layer_idx)
                hook = layer.self_attn.o_proj.register_forward_hook(
                    make_capture_hook(steered_outputs, layer_idx)
                )
                hooks.append(hook)

            # Steering hook - only steer from trial question onwards
            def make_steering_hook(start_pos, steer_vec):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    seq_len = hidden.shape[1]
                    if start_pos < seq_len:
                        hidden = hidden.clone()
                        hidden[:, start_pos:, :] = hidden[:, start_pos:, :] + steering_strength * steer_vec.to(hidden.dtype)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            steering_module = model_wrapper.get_layer_module(steering_layer)
            steering_handle = steering_module.register_forward_hook(
                make_steering_hook(steer_start_pos, steering_vec)
            )
            hooks.append(steering_handle)

            with torch.no_grad():
                model(input_ids=input_ids)

            for hook in hooks:
                hook.remove()

            # Compute signed alignment for each head
            for layer_idx in target_layers:
                ctrl = control_outputs[layer_idx]  # [batch, seq, n_heads, head_dim]
                steer = steered_outputs[layer_idx]

                # Get O-projection weight for this layer
                o_weight = model_wrapper.get_layer_module(layer_idx).self_attn.o_proj.weight
                # Shape: [hidden_dim, hidden_dim]

                for h in range(n_heads):
                    # Get head outputs at last token
                    ctrl_h = ctrl[0, -1, h, :]  # [head_dim]
                    steer_h = steer[0, -1, h, :]  # [head_dim]

                    # Compute delta
                    delta_h = steer_h - ctrl_h  # [head_dim]

                    # Project delta through O-weight slice for this head
                    # O-weight is organized as [hidden_dim, hidden_dim]
                    # where columns [h*head_dim : (h+1)*head_dim] correspond to head h
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    o_slice = o_weight[:, start_idx:end_idx]  # [hidden_dim, head_dim]

                    # Project: delta_in_residual_stream = O_slice @ delta_h
                    delta_proj = (o_slice @ delta_h.unsqueeze(-1)).squeeze(-1)  # [hidden_dim]
                    delta_proj_flat = delta_proj.flatten().float()

                    # SIGNED cosine similarity (this is the key difference!)
                    cos_sim = F.cosine_similarity(
                        delta_proj_flat.unsqueeze(0),
                        steering_vec_flat.unsqueeze(0)
                    ).item()

                    alignments[(layer_idx, h)].append(cos_sim)

            # Clear storage
            del control_outputs, steered_outputs
            torch.cuda.empty_cache()

    # Aggregate
    signed_alignment = {}
    for (layer, head), sims in alignments.items():
        signed_alignment[(layer, head)] = float(np.mean(sims))

    return signed_alignment


# =============================================================================
# Ablation Experiments
# =============================================================================

def run_ablation_experiment(
    model_wrapper: ModelWrapper,
    tokenizer,
    concept_vectors: Dict[str, torch.Tensor],
    heads_to_ablate: List[Tuple[int, int]],
    steering_layer: int,
    steering_strength: float,
    n_trials: int = 5,
    max_tokens: int = 100,
    temperature: float = 1.0,
    device: str = "cuda",
    condition_name: str = "ablation",
    verbose: bool = False
) -> List[Dict]:
    """
    Run ablation experiment with proper prompt-level patching.

    Methodology:
    1. For each trial, run control forward pass (no steering) on the prompt
    2. Cache head outputs at ALL token positions
    3. Run generation with steering, but PATCH specified heads' outputs
       with their control values during prompt processing

    This ensures heads are "ablated" in the sense of not responding to steering,
    while still functioning normally during generation.
    """
    model = model_wrapper.model
    n_heads = model_wrapper.n_heads
    head_dim = get_head_dim(model_wrapper)

    results = []
    concepts = list(concept_vectors.keys())

    # Group heads by layer for efficient patching
    heads_by_layer = defaultdict(list)
    for layer_idx, head_idx in heads_to_ablate:
        heads_by_layer[layer_idx].append(head_idx)

    for concept in tqdm(concepts, desc=f"Ablation: {condition_name}", disable=not verbose):
        steering_vec = concept_vectors[concept].to(device)

        for trial_idx in range(n_trials):
            trial_num = trial_idx + 1
            messages = build_messages(trial_num=trial_num)
            input_ids, prompt_len, formatted_prompt = format_messages_for_model(messages, tokenizer)
            input_ids = input_ids.to(device)

            # Find where steering should start (at "Trial X:")
            steer_start_pos = find_steering_start_position(tokenizer, formatted_prompt, trial_num)

            if len(heads_to_ablate) == 0:
                # No ablation - just run with steering
                control_cache = {}
            else:
                # Step 1: Cache control head outputs at all positions
                control_cache = {}

                def make_cache_hook(layer_idx):
                    def hook(module, args, output):
                        inp = args[0]  # [batch, seq, hidden]
                        batch, seq, hidden = inp.shape
                        reshaped = inp.view(batch, seq, n_heads, head_dim)
                        control_cache[layer_idx] = reshaped.detach().clone()
                        return output
                    return hook

                hooks = []
                for layer_idx in heads_by_layer.keys():
                    layer = model_wrapper.get_layer_module(layer_idx)
                    hook = layer.self_attn.o_proj.register_forward_hook(
                        make_cache_hook(layer_idx)
                    )
                    hooks.append(hook)

                with torch.no_grad():
                    model(input_ids=input_ids)

                for hook in hooks:
                    hook.remove()

            # Step 2: Generate with steering + ablation
            def make_ablation_hook(layer_idx, heads_at_layer):
                def hook(module, args, output):
                    if layer_idx not in control_cache:
                        return output

                    inp = args[0]  # [batch, seq, hidden]
                    batch, seq, hidden = inp.shape

                    # Only patch during prompt processing
                    control = control_cache[layer_idx]  # [batch, prompt_len, n_heads, head_dim]
                    control_seq = control.shape[1]

                    if seq > control_seq:
                        # We're in generation - don't patch new tokens
                        return output

                    # Reshape input to access heads
                    reshaped = inp.view(batch, seq, n_heads, head_dim).clone()

                    # Patch specified heads with control values
                    for h in heads_at_layer:
                        reshaped[:, :, h, :] = control[:, :seq, h, :]

                    # Reshape back to [batch, seq, hidden]
                    modified = reshaped.view(batch, seq, hidden)

                    # Return modified input (this is a pre-hook effectively)
                    # But since we're hooking the output, we need to modify the actual projection
                    # Actually, we need a different approach - modify the OUTPUT not input

                    # The clean approach: compute what the output WOULD have been with control inputs
                    # output = o_proj(input), so if we change input, output changes
                    # But we can't easily modify the forward pass mid-computation

                    # Alternative: compute the difference in output and subtract it
                    # output_steered = o_proj(input_steered)
                    # output_control = o_proj(input_control)
                    # We want: output_steered - (contribution from ablated heads)

                    return output  # For now, return unchanged - we'll fix this below
                return hook

            # Actually, let's use a cleaner approach: modify the residual stream directly
            # after the attention layer to remove the ablated heads' contributions

            def make_ablation_post_hook(layer_idx, heads_at_layer):
                def hook(module, args, output):
                    if layer_idx not in control_cache:
                        return output

                    hidden = output[0] if isinstance(output, tuple) else output
                    batch, seq, hidden_dim = hidden.shape

                    control = control_cache[layer_idx]  # [batch, control_seq, n_heads, head_dim]
                    control_seq = control.shape[1]

                    if seq > control_seq:
                        # Generation phase - don't modify
                        if isinstance(output, tuple):
                            return output
                        return hidden

                    # Get the O-projection weight
                    o_weight = module.o_proj.weight  # [hidden_dim, hidden_dim]

                    # For each ablated head, compute and remove its steering-induced contribution
                    # The head's contribution to residual = O_slice @ head_output
                    # We want to replace this with O_slice @ control_head_output

                    # Get current head outputs
                    if not hasattr(hook, 'current_head_outputs'):
                        # We don't have access to current head outputs in this hook
                        # Need to use a different approach
                        pass

                    # Simpler approach: during the FORWARD pass, before generation,
                    # we can modify the hidden states to patch in control values

                    # For now, let's just return - the ablation will be done differently
                    if isinstance(output, tuple):
                        return output
                    return hidden
                return hook

            # Better approach: Use activation patching on the RESIDUAL STREAM
            # After each ablated layer, subtract the steered head's contribution and add the control head's contribution

            # Actually, the cleanest approach is to directly modify the attention module
            # to output control values for specified heads. Let's do this properly:

            class AblationContext:
                def __init__(self):
                    self.control_cache = control_cache
                    self.prompt_len = prompt_len
                    self.in_prompt = True

                def update_position(self, seq_len):
                    self.in_prompt = (seq_len <= self.prompt_len)

            ablation_ctx = AblationContext()

            def make_input_ablation_hook(layer_idx, heads_at_layer):
                """Hook that modifies the INPUT to o_proj to use control values for ablated heads."""
                def hook(module, args):
                    if layer_idx not in ablation_ctx.control_cache:
                        return args

                    inp = args[0]  # [batch, seq, hidden]
                    batch, seq, hidden = inp.shape

                    control = ablation_ctx.control_cache[layer_idx]
                    control_seq = control.shape[1]

                    if seq > control_seq:
                        # Generation phase - don't modify
                        return args

                    # Reshape input to access per-head values
                    reshaped = inp.view(batch, seq, n_heads, head_dim).clone()

                    # Patch specified heads with control values
                    for h in heads_at_layer:
                        reshaped[:, :seq, h, :] = control[:, :seq, h, :]

                    # Reshape back
                    modified = reshaped.view(batch, seq, hidden)

                    return (modified,) + args[1:] if len(args) > 1 else (modified,)
                return hook

            hooks = []

            # Register ablation hooks (pre-hooks on o_proj)
            for layer_idx, heads_at_layer in heads_by_layer.items():
                layer = model_wrapper.get_layer_module(layer_idx)
                hook = layer.self_attn.o_proj.register_forward_pre_hook(
                    make_input_ablation_hook(layer_idx, heads_at_layer)
                )
                hooks.append(hook)

            # Steering hook - only steer from trial question onwards
            def make_steering_hook(start_pos, steer_vec):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    seq_len = hidden.shape[1]
                    # During generation (seq_len == 1), always steer
                    if seq_len == 1:
                        hidden = hidden + steering_strength * steer_vec.to(hidden.dtype)
                    elif start_pos < seq_len:
                        hidden = hidden.clone()
                        hidden[:, start_pos:, :] = hidden[:, start_pos:, :] + steering_strength * steer_vec.to(hidden.dtype)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            steering_module = model_wrapper.get_layer_module(steering_layer)
            steering_handle = steering_module.register_forward_hook(
                make_steering_hook(steer_start_pos, steering_vec)
            )
            hooks.append(steering_handle)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

                eval_result = evaluate_detection(response, concept)

                results.append({
                    "concept": concept,
                    "trial": trial_idx,
                    "response": response,
                    "detected": eval_result["detected"],
                    "identified": eval_result["identified"],
                    "condition": condition_name,
                })

            finally:
                for hook in hooks:
                    hook.remove()

            # Clear cache
            del control_cache
            torch.cuda.empty_cache()

    return results


def aggregate_ablation_results(results: List[Dict]) -> Dict[str, float]:
    """Aggregate results into metrics."""
    total = len(results)
    if total == 0:
        return {"detection_rate": 0.0, "identification_rate": 0.0, "total": 0}

    detections = sum(1 for r in results if r["detected"])
    identifications = sum(1 for r in results if r["identified"])

    return {
        "detection_rate": detections / total,
        "identification_rate": identifications / total,
        "total": total,
        "detections": detections,
        "identifications": identifications,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_gradient_attribution_heatmap(
    attribution: Dict[Tuple[int, int], float],
    n_layers: int,
    n_heads: int,
    steering_layer: int,
    output_path: Path,
    title: str = "Gradient attribution by attention head"
):
    """Plot heatmap of gradient attribution scores."""
    # Create matrix (only for layers from steering layer onwards)
    target_layers = list(range(steering_layer + 1, n_layers))  # Start from layer AFTER steering
    matrix = np.zeros((len(target_layers), n_heads))

    for (layer, head), score in attribution.items():
        if layer > steering_layer:  # Only layers AFTER steering
            row = layer - steering_layer - 1  # Adjusted for starting at steering_layer + 1
            if row < len(target_layers) and head < n_heads:
                matrix[row, head] = score

    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(matrix, aspect='auto', cmap='viridis')

    # Labels
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Y-axis: show ALL layer numbers (every tick)
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([str(steering_layer + 1 + i) for i in range(len(target_layers))], fontsize=6)

    # X-axis: show ALL head numbers (every tick)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)], fontsize=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Gradient magnitude", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_differential_gradient_heatmap(
    attribution: Dict[Tuple[int, int], float],
    n_layers: int,
    n_heads: int,
    steering_layer: int,
    output_path: Path,
    title: str = "Differential gradient attribution (Yes - No)"
):
    """Plot heatmap with diverging colormap (red=yes-biased, blue=no-biased)."""
    target_layers = list(range(steering_layer + 1, n_layers))  # Start from layer AFTER steering
    matrix = np.full((len(target_layers), n_heads), np.nan)

    for (layer, head), score in attribution.items():
        if layer > steering_layer:  # Only layers AFTER steering
            row = layer - steering_layer - 1  # Adjusted for starting at steering_layer + 1
            if row < len(target_layers) and head < n_heads:
                matrix[row, head] = score

    fig, ax = plt.subplots(figsize=(16, 14))

    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Y-axis: show ALL layer numbers (every tick)
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([str(steering_layer + 1 + i) for i in range(len(target_layers))], fontsize=6)

    # X-axis: show ALL head numbers (every tick)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)], fontsize=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Differential attribution\n(red=Yes-biased, blue=No-biased)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_signed_alignment_heatmap(
    alignment: Dict[Tuple[int, int], float],
    n_layers: int,
    n_heads: int,
    steering_layer: int,
    output_path: Path,
    title: str = "Signed steering alignment by attention head"
):
    """Plot heatmap with diverging colormap (red=amplifier, blue=suppressor)."""
    target_layers = list(range(steering_layer + 1, n_layers))  # Start from layer AFTER steering
    matrix = np.full((len(target_layers), n_heads), np.nan)

    for (layer, head), score in alignment.items():
        if layer > steering_layer:  # Only layers AFTER steering
            row = layer - steering_layer - 1  # Adjusted for starting at steering_layer + 1
            if row < len(target_layers) and head < n_heads:
                matrix[row, head] = score

    fig, ax = plt.subplots(figsize=(16, 14))

    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Y-axis: show ALL layer numbers (every tick)
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([str(steering_layer + 1 + i) for i in range(len(target_layers))], fontsize=6)

    # X-axis: show ALL head numbers (every tick)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)], fontsize=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Signed alignment\n(red=amplifier, blue=suppressor)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(
    ablation_results: Dict[str, Dict],
    output_path: Path,
    title: str = "Detection rates by ablation condition"
):
    """Plot bar chart comparing ablation conditions."""
    # Sort conditions for consistent ordering
    conditions = sorted(ablation_results.keys(), key=lambda x: (
        0 if x == "baseline" else
        1 if "amplifier" in x else
        2 if "suppressor" in x else
        3 if "gradient" in x else 4
    ))

    detection_rates = [ablation_results[c]["detection_rate"] for c in conditions]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(conditions))
    colors = []
    for c in conditions:
        if c == "baseline":
            colors.append("gray")
        elif "amplifier" in c:
            colors.append("indianred")
        elif "suppressor" in c:
            colors.append("steelblue")
        elif "gradient" in c:
            colors.append("forestgreen")
        else:
            colors.append("mediumpurple")

    bars = ax.bar(x, detection_rates, color=colors)

    # Baseline reference line
    if "baseline" in ablation_results:
        baseline_rate = ablation_results["baseline"]["detection_rate"]
        ax.axhline(y=baseline_rate, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Detection rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)

    # Value labels
    for bar, rate in zip(bars, detection_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(color='gray', label='Baseline'),
        mpatches.Patch(color='indianred', label='Ablate Amplifiers'),
        mpatches.Patch(color='steelblue', label='Ablate Suppressors'),
        mpatches.Patch(color='forestgreen', label='Ablate by Gradient'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_amplifier_suppressor_distribution(
    signed_alignment: Dict[Tuple[int, int], float],
    steering_layer: int,
    output_path: Path
):
    """Plot distribution of alignment scores and layer breakdown."""
    scores = list(signed_alignment.values())
    layers = [l for (l, h) in signed_alignment.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of alignment scores
    ax1 = axes[0]
    ax1.hist(scores, bins=50, color='gray', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (0)')
    ax1.set_xlabel("Signed alignment score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of signed alignment scores", fontsize=13)

    # Set x-ticks with finer granularity (every 0.02)
    min_score, max_score = min(scores), max(scores)
    x_tick_step = 0.02
    x_ticks = np.arange(np.floor(min_score / x_tick_step) * x_tick_step,
                        np.ceil(max_score / x_tick_step) * x_tick_step + x_tick_step,
                        x_tick_step)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{x:.2f}' for x in x_ticks], fontsize=6, rotation=45, ha='right')

    n_pos = sum(1 for s in scores if s > 0)
    n_neg = sum(1 for s in scores if s < 0)
    ax1.legend([f'Neutral\n(Amplifiers: {n_pos}, Suppressors: {n_neg})'], loc='upper right')

    # Mean alignment by layer
    ax2 = axes[1]
    layer_means = defaultdict(list)
    for (l, h), score in signed_alignment.items():
        layer_means[l].append(score)

    layer_nums = sorted(layer_means.keys())
    means = [np.mean(layer_means[l]) for l in layer_nums]
    stds = [np.std(layer_means[l]) for l in layer_nums]

    colors = ['indianred' if m > 0 else 'steelblue' for m in means]
    ax2.bar(range(len(layer_nums)), means, yerr=stds, color=colors, alpha=0.7, capsize=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # Show ALL layer numbers (every tick)
    ax2.set_xticks(range(len(layer_nums)))
    ax2.set_xticklabels([str(l) for l in layer_nums], fontsize=6, rotation=45, ha='right')
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Mean signed alignment", fontsize=12)
    ax2.set_title("Mean alignment by layer", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Sections to run: {args.sections}")

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
        quantization=args.quantization
    )
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # Enable gradient checkpointing to reduce memory usage during backward pass
    # This trades ~20% more compute time for massive memory savings (~40-50GB)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (reduces memory for gradient computation)")

    n_layers = model_wrapper.n_layers
    n_heads = model_wrapper.n_heads

    print(f"Model: {n_layers} layers, {n_heads} heads per layer")
    print(f"Steering at layer {args.layer} with strength {args.strength}")

    # Load concept vectors
    # Compute layer fraction for loading vectors from experiment 02 (steering evaluation)
    layer_fraction = args.layer / n_layers
    print(f"\nLoading concept vectors from {args.steering_dir}")
    print(f"(Layer {args.layer} = fraction {layer_fraction:.2f})")
    concept_vectors = load_concept_vectors(
        args.steering_dir,
        args.model,
        args.concepts,
        n_sample=args.n_concepts,
        layer_fraction=layer_fraction
    )
    print(f"Loaded {len(concept_vectors)} concepts: {list(concept_vectors.keys())[:5]}...")

    results = {}

    # =========================================================================
    # Section 1: Gradient Attribution (multiple modes)
    # =========================================================================
    if "gradient" in args.sections:
        print("\n" + "="*70)
        print("SECTION 1: Gradient-Based Attribution")
        print("="*70)

        # Load successful concepts if needed
        successful_concepts = None
        if "successful" in args.gradient_modes:
            successful_concepts = load_successful_concepts(
                args.steering_dir,
                args.model,
                layer_fraction=layer_fraction,
                strength=args.strength
            )
            print(f"Loaded {len(successful_concepts)} successful concepts (>=1 detection)")

        gradient_results = {}

        for mode in args.gradient_modes:
            print(f"\n--- Gradient Mode: {mode} ---")

            if mode == "all":
                mode_concepts = concept_vectors
                mode_name = "all"
                gradient_mode = "standard"
            elif mode == "successful":
                if successful_concepts is None:
                    print("Skipping successful mode: no successful concepts loaded")
                    continue
                mode_concepts = {c: v for c, v in concept_vectors.items() if c in successful_concepts}
                if len(mode_concepts) == 0:
                    print("Skipping successful mode: no matching concepts")
                    continue
                mode_name = "successful"
                gradient_mode = "standard"
                print(f"Using {len(mode_concepts)} successful concepts")
            elif mode == "differential":
                mode_concepts = concept_vectors
                mode_name = "differential"
                gradient_mode = "differential"
            else:
                continue

            gradient_path = output_dir / f"gradient_attribution_{mode_name}.json"

            if gradient_path.exists() and not args.overwrite:
                print(f"Loading cached gradient attribution ({mode_name})...")
                with open(gradient_path) as f:
                    data = json.load(f)
                gradient_attribution = {
                    tuple(map(int, k.split(","))): v
                    for k, v in data["attribution"].items()
                }
            else:
                gradient_attribution = compute_gradient_attribution(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    concept_vectors=mode_concepts,
                    steering_layer=args.layer,
                    steering_strength=args.strength,
                    n_trials=args.n_trials,
                    device=args.device,
                    verbose=args.verbose,
                    mode=gradient_mode
                )

                # Save
                data = {
                    "attribution": {f"{l},{h}": v for (l, h), v in gradient_attribution.items()},
                    "n_concepts": len(mode_concepts),
                    "n_trials": args.n_trials,
                    "steering_layer": args.layer,
                    "steering_strength": args.strength,
                    "mode": mode_name,
                }
                with open(gradient_path, "w") as f:
                    json.dump(data, f, indent=2)

            gradient_results[mode_name] = gradient_attribution

            # Plot
            if mode_name == "differential":
                plot_differential_gradient_heatmap(
                    gradient_attribution,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    steering_layer=args.layer,
                    output_path=plots_dir / f"gradient_attribution_heatmap_{mode_name}.png",
                    title=f"Differential gradient: |∂Yes/∂head| - |∂No/∂head| ({len(mode_concepts)} concepts)"
                )
            else:
                plot_gradient_attribution_heatmap(
                    gradient_attribution,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    steering_layer=args.layer,
                    output_path=plots_dir / f"gradient_attribution_heatmap_{mode_name}.png",
                    title=f"Gradient attribution ({mode_name}): |∂(Yes-No)/∂head| ({len(mode_concepts)} concepts)"
                )

            # Print top heads
            if mode_name == "differential":
                # Sort by absolute value for differential
                sorted_heads = sorted(gradient_attribution.items(), key=lambda x: -abs(x[1]))
                print(f"\nTop 20 heads by differential gradient (mode={mode_name}):")
                for (layer, head), score in sorted_heads[:20]:
                    direction = "Yes" if score > 0 else "No"
                    print(f"  L{layer}H{head}: {score:+.6f} ({direction}-biased)")
            else:
                sorted_heads = sorted(gradient_attribution.items(), key=lambda x: -x[1])
                print(f"\nTop 20 heads by gradient attribution (mode={mode_name}):")
                for (layer, head), score in sorted_heads[:20]:
                    print(f"  L{layer}H{head}: {score:.6f}")

            # Save ranked list
            with open(output_dir / f"gradient_ranked_{mode_name}.txt", "w") as f:
                f.write("Rank\tLayer\tHead\tAttribution\n")
                for i, ((layer, head), score) in enumerate(sorted_heads):
                    f.write(f"{i+1}\t{layer}\t{head}\t{score:.6f}\n")

        # Store the "all" mode as the main gradient attribution for ablation
        if "all" in gradient_results:
            results["gradient_attribution"] = gradient_results["all"]
        elif gradient_results:
            results["gradient_attribution"] = list(gradient_results.values())[0]

    # =========================================================================
    # Section 2: Signed Steering Alignment
    # =========================================================================
    if "signed_alignment" in args.sections:
        print("\n" + "="*70)
        print("SECTION 2: Signed Steering Alignment")
        print("="*70)

        alignment_path = output_dir / "signed_alignment.json"

        if alignment_path.exists() and not args.overwrite:
            print("Loading cached signed alignment...")
            with open(alignment_path) as f:
                data = json.load(f)
            signed_alignment = {
                tuple(map(int, k.split(","))): v
                for k, v in data["alignment"].items()
            }
        else:
            signed_alignment = compute_signed_steering_alignment(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                concept_vectors=concept_vectors,
                steering_layer=args.layer,
                steering_strength=args.strength,
                n_trials=args.n_trials,
                device=args.device,
                verbose=args.verbose
            )

            # Save
            data = {
                "alignment": {f"{l},{h}": v for (l, h), v in signed_alignment.items()},
                "n_concepts": len(concept_vectors),
                "n_trials": args.n_trials,
                "steering_layer": args.layer,
                "steering_strength": args.strength,
            }
            with open(alignment_path, "w") as f:
                json.dump(data, f, indent=2)

        results["signed_alignment"] = signed_alignment

        # Separate amplifiers and suppressors
        amplifiers = [(k, v) for k, v in signed_alignment.items() if v > 0]
        suppressors = [(k, v) for k, v in signed_alignment.items() if v < 0]

        amplifiers_sorted = sorted(amplifiers, key=lambda x: -x[1])  # Most positive first
        suppressors_sorted = sorted(suppressors, key=lambda x: x[1])  # Most negative first

        print(f"\nFound {len(amplifiers)} amplifier heads (positive alignment)")
        print(f"Found {len(suppressors)} suppressor heads (negative alignment)")

        print("\nTop 15 AMPLIFIER heads (shift output in steering direction):")
        for (layer, head), score in amplifiers_sorted[:15]:
            print(f"  L{layer}H{head}: {score:+.4f}")

        print("\nTop 15 SUPPRESSOR heads (shift output against steering direction):")
        for (layer, head), score in suppressors_sorted[:15]:
            print(f"  L{layer}H{head}: {score:+.4f}")

        # Plots
        plot_signed_alignment_heatmap(
            signed_alignment,
            n_layers=n_layers,
            n_heads=n_heads,
            steering_layer=args.layer,
            output_path=plots_dir / "signed_alignment_heatmap.png",
            title="Signed steering alignment (red=amplifier, blue=suppressor)"
        )

        plot_amplifier_suppressor_distribution(
            signed_alignment,
            steering_layer=args.layer,
            output_path=plots_dir / "alignment_distribution.png"
        )

        # Save ranked lists
        with open(output_dir / "amplifiers_ranked.txt", "w") as f:
            f.write("Rank\tLayer\tHead\tAlignment\n")
            for i, ((layer, head), score) in enumerate(amplifiers_sorted):
                f.write(f"{i+1}\t{layer}\t{head}\t{score:+.4f}\n")

        with open(output_dir / "suppressors_ranked.txt", "w") as f:
            f.write("Rank\tLayer\tHead\tAlignment\n")
            for i, ((layer, head), score) in enumerate(suppressors_sorted):
                f.write(f"{i+1}\t{layer}\t{head}\t{score:+.4f}\n")

        results["amplifiers"] = amplifiers_sorted
        results["suppressors"] = suppressors_sorted

    # =========================================================================
    # Section 3: Ablation Experiments
    # =========================================================================
    if "ablation" in args.sections:
        print("\n" + "="*70)
        print("SECTION 3: Ablation Experiments")
        print("="*70)

        ablation_results = {}

        # Baseline (no ablation)
        print("\nRunning baseline (no ablation)...")
        baseline_trials = run_ablation_experiment(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            concept_vectors=concept_vectors,
            heads_to_ablate=[],
            steering_layer=args.layer,
            steering_strength=args.strength,
            n_trials=args.n_trials,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
            condition_name="baseline",
            verbose=args.verbose
        )
        ablation_results["baseline"] = aggregate_ablation_results(baseline_trials)
        print(f"Baseline detection rate: {ablation_results['baseline']['detection_rate']:.1%}")

        # Calculate total heads for percentage-based ablation
        total_heads = (n_layers - args.layer) * n_heads  # Only heads after steering layer

        # Ablation based on signed alignment
        if "signed_alignment" in results:
            amplifiers = results["amplifiers"]
            suppressors = results["suppressors"]

            for pct in args.ablation_percentages:
                n_ablate = max(1, int(pct / 100 * total_heads))

                # Ablate top amplifiers
                if len(amplifiers) >= n_ablate:
                    top_amp = [k for k, v in amplifiers[:n_ablate]]
                    print(f"\nAblating top {pct}% amplifiers ({len(top_amp)} heads)...")
                    amp_trials = run_ablation_experiment(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        concept_vectors=concept_vectors,
                        heads_to_ablate=top_amp,
                        steering_layer=args.layer,
                        steering_strength=args.strength,
                        n_trials=args.n_trials,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        device=args.device,
                        condition_name=f"amplifiers_{pct}%",
                        verbose=args.verbose
                    )
                    ablation_results[f"amplifiers_{pct}%"] = aggregate_ablation_results(amp_trials)
                    rate = ablation_results[f"amplifiers_{pct}%"]["detection_rate"]
                    delta = rate - ablation_results["baseline"]["detection_rate"]
                    print(f"  Detection: {rate:.1%} (Δ {delta:+.1%})")

                # Ablate top suppressors
                if len(suppressors) >= n_ablate:
                    top_sup = [k for k, v in suppressors[:n_ablate]]
                    print(f"\nAblating top {pct}% suppressors ({len(top_sup)} heads)...")
                    sup_trials = run_ablation_experiment(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        concept_vectors=concept_vectors,
                        heads_to_ablate=top_sup,
                        steering_layer=args.layer,
                        steering_strength=args.strength,
                        n_trials=args.n_trials,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        device=args.device,
                        condition_name=f"suppressors_{pct}%",
                        verbose=args.verbose
                    )
                    ablation_results[f"suppressors_{pct}%"] = aggregate_ablation_results(sup_trials)
                    rate = ablation_results[f"suppressors_{pct}%"]["detection_rate"]
                    delta = rate - ablation_results["baseline"]["detection_rate"]
                    print(f"  Detection: {rate:.1%} (Δ {delta:+.1%})")

        # Ablation based on gradient attribution
        if "gradient_attribution" in results:
            gradient_attribution = results["gradient_attribution"]
            sorted_by_grad = sorted(gradient_attribution.items(), key=lambda x: -x[1])

            for pct in args.ablation_percentages:
                n_ablate = max(1, int(pct / 100 * len(sorted_by_grad)))

                # Top gradient heads
                top_grad = [k for k, v in sorted_by_grad[:n_ablate]]
                print(f"\nAblating top {pct}% by gradient ({len(top_grad)} heads)...")
                grad_trials = run_ablation_experiment(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    concept_vectors=concept_vectors,
                    heads_to_ablate=top_grad,
                    steering_layer=args.layer,
                    steering_strength=args.strength,
                    n_trials=args.n_trials,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=args.device,
                    condition_name=f"gradient_top_{pct}%",
                    verbose=args.verbose
                )
                ablation_results[f"gradient_top_{pct}%"] = aggregate_ablation_results(grad_trials)
                rate = ablation_results[f"gradient_top_{pct}%"]["detection_rate"]
                delta = rate - ablation_results["baseline"]["detection_rate"]
                print(f"  Detection: {rate:.1%} (Δ {delta:+.1%})")

                # Bottom gradient heads (inverted test)
                bottom_grad = [k for k, v in sorted_by_grad[-n_ablate:]]
                print(f"\nAblating bottom {pct}% by gradient ({len(bottom_grad)} heads)...")
                inv_trials = run_ablation_experiment(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    concept_vectors=concept_vectors,
                    heads_to_ablate=bottom_grad,
                    steering_layer=args.layer,
                    steering_strength=args.strength,
                    n_trials=args.n_trials,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=args.device,
                    condition_name=f"gradient_bottom_{pct}%",
                    verbose=args.verbose
                )
                ablation_results[f"gradient_bottom_{pct}%"] = aggregate_ablation_results(inv_trials)
                rate = ablation_results[f"gradient_bottom_{pct}%"]["detection_rate"]
                delta = rate - ablation_results["baseline"]["detection_rate"]
                print(f"  Detection: {rate:.1%} (Δ {delta:+.1%})")

        # Save ablation results
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

        # Plot comparison
        plot_ablation_comparison(
            ablation_results,
            output_path=plots_dir / "ablation_comparison.png",
            title="Detection rates by ablation condition"
        )

        # Summary
        print("\n" + "="*70)
        print("ABLATION SUMMARY")
        print("="*70)

        baseline_rate = ablation_results["baseline"]["detection_rate"]

        for condition in sorted(ablation_results.keys()):
            if condition == "baseline":
                continue
            rate = ablation_results[condition]["detection_rate"]
            delta = rate - baseline_rate
            direction = "↑" if delta > 0.005 else "↓" if delta < -0.005 else "→"
            print(f"  {condition:30s}: {rate:.1%} ({direction} {abs(delta):.1%})")

        # Interpretation
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)

        # Check amplifier ablation effect
        amp_keys = [k for k in ablation_results.keys() if "amplifiers" in k]
        if amp_keys:
            amp_deltas = [ablation_results[k]["detection_rate"] - baseline_rate for k in amp_keys]
            mean_amp_delta = np.mean(amp_deltas)
            if mean_amp_delta < -0.05:
                print("✓ Ablating AMPLIFIERS DECREASES detection → Amplifiers help introspection")
            elif mean_amp_delta > 0.05:
                print("✗ Ablating AMPLIFIERS INCREASES detection → 'Amplifiers' are actually suppressors!")
            else:
                print("? Ablating amplifiers has minimal effect")

        # Check suppressor ablation effect
        sup_keys = [k for k in ablation_results.keys() if "suppressors" in k]
        if sup_keys:
            sup_deltas = [ablation_results[k]["detection_rate"] - baseline_rate for k in sup_keys]
            mean_sup_delta = np.mean(sup_deltas)
            if mean_sup_delta > 0.05:
                print("✓ Ablating SUPPRESSORS INCREASES detection → Suppressors hurt introspection")
            elif mean_sup_delta < -0.05:
                print("✗ Ablating SUPPRESSORS DECREASES detection → 'Suppressors' actually help!")
            else:
                print("? Ablating suppressors has minimal effect")

        # Check gradient-based ablation
        grad_top_keys = [k for k in ablation_results.keys() if "gradient_top" in k]
        grad_bot_keys = [k for k in ablation_results.keys() if "gradient_bottom" in k]

        if grad_top_keys and grad_bot_keys:
            top_deltas = [ablation_results[k]["detection_rate"] - baseline_rate for k in grad_top_keys]
            bot_deltas = [ablation_results[k]["detection_rate"] - baseline_rate for k in grad_bot_keys]

            if np.mean(top_deltas) < np.mean(bot_deltas) - 0.05:
                print("✓ High-gradient heads are more important than low-gradient heads")
            elif np.mean(bot_deltas) < np.mean(top_deltas) - 0.05:
                print("✗ Low-gradient heads are more important → Gradient metric may be inverted")
            else:
                print("? Gradient ranking doesn't clearly distinguish importance")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")

    # Save summary
    summary = {
        "n_concepts": len(concept_vectors),
        "n_trials": args.n_trials,
        "steering_layer": args.layer,
        "steering_strength": args.strength,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "sections_run": args.sections,
    }

    if "gradient_attribution" in results:
        sorted_grad = sorted(results["gradient_attribution"].items(), key=lambda x: -x[1])
        summary["top_10_gradient_heads"] = [
            {"layer": l, "head": h, "score": s} for (l, h), s in sorted_grad[:10]
        ]

    if "signed_alignment" in results:
        summary["n_amplifiers"] = len(results.get("amplifiers", []))
        summary["n_suppressors"] = len(results.get("suppressors", []))

        if results.get("amplifiers"):
            summary["top_10_amplifiers"] = [
                {"layer": l, "head": h, "alignment": a}
                for (l, h), a in results["amplifiers"][:10]
            ]

        if results.get("suppressors"):
            summary["top_10_suppressors"] = [
                {"layer": l, "head": h, "alignment": a}
                for (l, h), a in results["suppressors"][:10]
            ]

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
