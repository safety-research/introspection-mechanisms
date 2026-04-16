#!/usr/bin/env python3
"""
Head Ablation (11): Mean Ablation of Identified Attention Heads

Run mean ablation experiments on attention heads identified in 12_head_investigation.
Uses trial-number matched mean ablation:
- Collect mean head outputs from control trials, grouped by trial number
- During steered trials, patch heads with the mean from matching trial number
- Only ablate during prompt processing (not generation)

Head Selection:
- Loads top 50 heads from 12_head_investigation top50_classification.json
- Separates into "beneficial" (positive attribution) and "detrimental" (negative attribution)
- Ablates each group separately to test causal role

Concept Selection:
- Uses success/failure partition from 04b_vector_geometry
- Only runs on success concepts (those with introspection detection > threshold)

Ablation Configurations:
- beneficial_N: Top N heads with positive attribution (help introspection)
- detrimental_N: Top N heads with negative attribution (hurt introspection)
- Head counts: 1, 3, 5, 10, 15, 20 (ordered by attribution magnitude)
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Set, Dict, List, Tuple, Optional
from collections import defaultdict
import random
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_utils import ModelWrapper, load_model
from eval_utils import LLMJudge, batch_evaluate


def compute_se(p, n):
    """Compute standard error for a binomial proportion."""
    if n == 0:
        return 0
    return np.sqrt(p * (1 - p) / n) * 100  # Return as percentage


def load_head_results(
    head_dir: str,
    model_name: str,
    layer: int,
    strength: float,
    partition: str = "success"
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict]:
    """
    Load ALL heads from 12_head_investigation attribution results, sorted by |attribution|.

    Uses the full attribution file (section_a/attribution_introspection.json) which has
    attribution scores for all 768 heads, not just the top 50.

    Args:
        head_dir: Path to 12_head_investigation results
        model_name: Model name (e.g., "gemma3_27b")
        layer: Steering layer index
        strength: Steering strength
        partition: Which partition to load ("success" or "failure")

    Returns:
        beneficial_heads: List of (layer, head) tuples with positive attribution, sorted by |attr|
        detrimental_heads: List of (layer, head) tuples with negative attribution, sorted by |attr|
        full_data: The complete JSON data for reference
    """
    # Path to full attribution results
    attribution_path = (
        Path(head_dir) / model_name /
        f"layer_{layer}_strength_{strength}" /
        "section_a" / partition / "attribution_introspection.json"
    )

    if not attribution_path.exists():
        raise FileNotFoundError(f"Attribution file not found: {attribution_path}")

    with open(attribution_path) as f:
        data = json.load(f)

    attrs = data["attribution"]

    # Separate into positive (beneficial) and negative (detrimental)
    beneficial = []
    detrimental = []

    for key, value in attrs.items():
        layer_idx, head_idx = map(int, key.split(","))
        if value > 0:
            beneficial.append(((layer_idx, head_idx), value))
        elif value < 0:
            detrimental.append(((layer_idx, head_idx), value))

    # Sort by absolute attribution (strongest first)
    beneficial.sort(key=lambda x: abs(x[1]), reverse=True)
    detrimental.sort(key=lambda x: abs(x[1]), reverse=True)

    beneficial_heads = [head for head, _ in beneficial]
    detrimental_heads = [head for head, _ in detrimental]

    print(f"Loaded {len(beneficial_heads)} beneficial heads (positive attribution)")
    print(f"Loaded {len(detrimental_heads)} detrimental heads (negative attribution)")
    print(f"  Top beneficial: L{beneficial_heads[0][0]}H{beneficial_heads[0][1]} (attr={beneficial[0][1]:.3f})")
    print(f"  Top detrimental: L{detrimental_heads[0][0]}H{detrimental_heads[0][1]} (attr={detrimental[0][1]:.3f})")

    return beneficial_heads, detrimental_heads, data


def load_partition_concepts(
    geometry_dir: str,
    model_name: str,
    layer: int,
    strength: float,
    partition: str = "success"
) -> List[str]:
    """Load concepts from specified partition in experiment 04b (vector geometry) subspace analysis.

    Args:
        geometry_dir: Path to experiment 04b (vector geometry) results
        model_name: Model name
        layer: Steering layer index
        strength: Steering strength
        partition: Which partition to load ("success" or "failure")

    Returns:
        List of concept names in the specified partition
    """
    subspace_path = (
        Path(geometry_dir) / model_name /
        f"layer_{layer}_strength_{strength}" /
        "subspace_analysis.json"
    )

    if not subspace_path.exists():
        raise FileNotFoundError(f"Subspace analysis not found: {subspace_path}")

    with open(subspace_path) as f:
        data = json.load(f)

    if partition == "success":
        concepts = data["success_concepts"]
    else:
        concepts = data["failure_concepts"]

    print(f"Loaded {len(concepts)} {partition} concepts from experiment 04b (vector geometry)")

    return concepts


def load_steering_vectors(
    steering_dir: str,
    model_name: str,
    layer: int,
    concepts: List[str]
) -> Dict[str, torch.Tensor]:
    """Load steering vectors from experiment 02 (steering evaluation) for specified concepts."""
    vectors_dir = Path(steering_dir) / model_name / "vectors" / f"layer_{layer}"

    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    vectors = {}
    missing = []

    for concept in concepts:
        vec_path = vectors_dir / f"{concept}.pt"
        if vec_path.exists():
            vectors[concept] = torch.load(vec_path, map_location="cpu", weights_only=True)
        else:
            missing.append(concept)

    if missing:
        print(f"  Warning: {len(missing)} concepts missing vectors: {missing[:5]}...")

    print(f"Loaded {len(vectors)} steering vectors")
    return vectors


def build_messages(trial_prompt: str) -> List[Dict]:
    """Build the standard introspection prompt."""
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
        {"role": "user", "content": trial_prompt},
    ]


def get_trial_prompts() -> List[str]:
    """Get the 10 trial prompts."""
    prompts = []
    for trial_num in range(1, 11):
        prompts.append(f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?")
    return prompts


def find_trial_start_position(tokenizer, formatted_prompt: str, trial_num: int) -> int:
    """
    Find the token position of the \\n before 'Trial X'.
    Returns the position where ablation should start.
    """
    # Find "Trial X" in the formatted prompt
    trial_marker = f"Trial {trial_num}:"
    trial_pos = formatted_prompt.find(trial_marker)

    if trial_pos == -1:
        return 0

    # Find the \n before Trial X (usually the user turn marker)
    newline_pos = formatted_prompt.rfind("\n", 0, trial_pos)
    if newline_pos == -1:
        newline_pos = 0

    # Tokenize up to this position to get token index
    # IMPORTANT: Use add_special_tokens=False since formatted_prompt already has BOS
    prefix = formatted_prompt[:newline_pos]
    prefix_tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    return prefix_tokens['input_ids'].shape[1]


class MeanAblationExperiment:
    def __init__(
        self,
        model_name: str,
        steering_layer: int,
        steering_strength: float,
        device: str = "cuda",
        dtype: str = "bfloat16",
        judge_concurrency: int = 100
    ):
        self.model_name = model_name
        self.steering_layer = steering_layer
        self.steering_strength = steering_strength
        self.device = device

        # Load model
        print(f"\nLoading model: {model_name}")
        self.model_wrapper = load_model(model_name, device=device, dtype=dtype)
        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer

        # Get model internals
        if hasattr(self.model.model, 'language_model'):
            self.lang_model = self.model.model.language_model
        else:
            self.lang_model = self.model.model

        config = self.model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        self.head_dim = getattr(config, 'head_dim', 128)
        self.n_heads = getattr(config, 'num_attention_heads', 32)
        self.n_layers = self.model_wrapper.n_layers

        print(f"Model: {self.n_layers} layers, {self.n_heads} heads, head_dim={self.head_dim}")

        # Initialize judge with concurrency
        print(f"Initializing LLM judge (max_concurrent={judge_concurrency})...")
        self.judge = LLMJudge(max_concurrent=judge_concurrency)

        # Trial prompts
        self.trial_prompts = get_trial_prompts()

    def make_steering_hook(self, steering_vec: torch.Tensor, start_pos: int):
        """Create hook to add steering vector."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0].clone()
                rest = output[1:]
            else:
                hidden_states = output.clone()
                rest = None

            sv = steering_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
            seq_len = hidden_states.shape[1]

            if seq_len == 1:
                # During generation, always steer
                hidden_states = hidden_states + self.steering_strength * sv.view(1, 1, -1)
            elif start_pos < seq_len:
                hidden_states[:, start_pos:, :] = hidden_states[:, start_pos:, :] + self.steering_strength * sv.view(1, 1, -1)

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states
        return hook

    def make_capture_hook(self, storage: Dict, layer_idx: int, prompt_len: int):
        """
        Hook to capture head outputs during control trial.
        Only captures positions within the prompt (not generation).
        """
        def hook(module, input, output):
            if len(input) == 0:
                return output
            inp = input[0]  # [batch, seq, n_heads * head_dim]

            # Only capture if we're still in prompt processing
            if inp.shape[1] <= prompt_len:
                if layer_idx not in storage:
                    storage[layer_idx] = []
                storage[layer_idx].append(inp.detach().clone())

            return output
        return hook

    def make_ablation_hook(
        self,
        layer_idx: int,
        head_indices: Set[int],
        mean_activations: torch.Tensor,
        ablation_start_pos: int,
        prompt_len: int
    ):
        """
        Hook to replace specific head outputs with mean activations.
        Only ablates from ablation_start_pos through prompt_len (not during generation).
        """
        def hook(module, input, output):
            if len(input) == 0:
                return output

            inp = input[0].clone()  # [batch, seq, n_heads * head_dim]
            seq_len = inp.shape[1]

            # Don't ablate during generation (seq_len == 1 means autoregressive generation)
            if seq_len == 1:
                return output

            # Get mean activations on correct device/dtype
            mean_act = mean_activations.to(inp.device, inp.dtype)

            # Ablate from start_pos to end of prompt
            ablate_end = min(seq_len, prompt_len, mean_act.shape[0])
            ablate_start = min(ablation_start_pos, ablate_end)

            if ablate_start < ablate_end:
                for h_idx in head_indices:
                    start_dim = h_idx * self.head_dim
                    end_dim = start_dim + self.head_dim
                    inp[:, ablate_start:ablate_end, start_dim:end_dim] = mean_act[ablate_start:ablate_end, start_dim:end_dim]

            # Recompute output projection with modified input
            new_output = F.linear(inp, module.weight, module.bias)
            return new_output

        return hook

    def collect_control_activations(
        self,
        concepts: List[str],
        steering_vectors: Dict[str, torch.Tensor],
        n_trials_per_concept: int = 3,
        layers_to_capture: List[int] = None
    ) -> Dict[int, Dict[int, List[torch.Tensor]]]:
        """
        Collect head activations from control trials, grouped by trial number.

        Returns:
            Dict mapping trial_num -> layer_idx -> list of activation tensors
        """
        if layers_to_capture is None:
            layers_to_capture = list(range(self.steering_layer + 1, self.n_layers))

        # Storage: trial_num -> layer_idx -> list of [seq, n_heads * head_dim] tensors
        control_activations = defaultdict(lambda: defaultdict(list))

        print(f"\nCollecting control activations from {len(concepts)} concepts × {n_trials_per_concept} trials per trial number...")

        for concept in tqdm(concepts, desc="Control collection"):
            for trial_num in range(1, 11):  # Trial 1-10
                prompt = self.trial_prompts[trial_num - 1]
                messages = build_messages(prompt)
                formatted = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                # IMPORTANT: add_special_tokens=False since chat template already includes BOS
                token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
                prompt_len = len(token_ids)

                for trial_rep in range(n_trials_per_concept):
                    seed = hash((concept, trial_num, trial_rep)) % (2**32)
                    torch.manual_seed(seed)
                    random.seed(seed)

                    # Setup capture hooks
                    captured = {}
                    hooks = []
                    for layer_idx in layers_to_capture:
                        layer_module = self.lang_model.layers[layer_idx]
                        o_proj = layer_module.self_attn.o_proj
                        h = o_proj.register_forward_hook(
                            self.make_capture_hook(captured, layer_idx, prompt_len)
                        )
                        hooks.append(h)

                    try:
                        input_ids = torch.tensor([token_ids], device=self.device)
                        with torch.no_grad():
                            # Just do forward pass on prompt (no generation needed for control)
                            _ = self.model(input_ids)

                        # Store captured activations
                        for layer_idx in layers_to_capture:
                            if layer_idx in captured and len(captured[layer_idx]) > 0:
                                # Take the first (and should be only) capture
                                act = captured[layer_idx][0].squeeze(0)  # [seq, n_heads * head_dim]
                                control_activations[trial_num][layer_idx].append(act.cpu())

                    finally:
                        for h in hooks:
                            h.remove()
                        torch.cuda.empty_cache()

        return control_activations

    def compute_mean_activations(
        self,
        control_activations: Dict[int, Dict[int, List[torch.Tensor]]]
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Compute mean activation per trial number and layer.

        Returns:
            Dict mapping trial_num -> layer_idx -> mean activation tensor [max_seq, n_heads * head_dim]
        """
        mean_activations = {}

        for trial_num, layer_acts in control_activations.items():
            mean_activations[trial_num] = {}
            for layer_idx, act_list in layer_acts.items():
                if len(act_list) == 0:
                    continue

                # Pad to max sequence length and stack
                max_seq = max(a.shape[0] for a in act_list)
                padded = []
                for a in act_list:
                    if a.shape[0] < max_seq:
                        pad = torch.zeros(max_seq - a.shape[0], a.shape[1])
                        a = torch.cat([a, pad], dim=0)
                    padded.append(a)

                stacked = torch.stack(padded, dim=0)  # [n_samples, seq, dim]
                mean_activations[trial_num][layer_idx] = stacked.mean(dim=0)  # [seq, dim]

        return mean_activations

    def run_ablation_trial(
        self,
        concept: str,
        steering_vec: torch.Tensor,
        trial_num: int,
        heads_to_ablate: List[Tuple[int, int]],
        mean_activations: Dict[int, torch.Tensor],
        seed: int
    ) -> Dict:
        """Run a single steered trial with head ablation."""
        prompt = self.trial_prompts[trial_num - 1]
        messages = build_messages(prompt)
        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # IMPORTANT: add_special_tokens=False since chat template already includes BOS
        token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        prompt_len = len(token_ids)

        # Find ablation start position (before "Trial X")
        ablation_start_pos = find_trial_start_position(self.tokenizer, formatted, trial_num)

        # Find steering start position (same as ablation for simplicity)
        steering_start_pos = ablation_start_pos

        torch.manual_seed(seed)
        random.seed(seed)

        hooks = []

        # Steering hook
        steer_layer = self.lang_model.layers[self.steering_layer]
        h = steer_layer.register_forward_hook(
            self.make_steering_hook(steering_vec, steering_start_pos)
        )
        hooks.append(h)

        # Ablation hooks
        heads_by_layer = defaultdict(set)
        for (layer_idx, head_idx) in heads_to_ablate:
            heads_by_layer[layer_idx].add(head_idx)

        for layer_idx, head_indices in heads_by_layer.items():
            if layer_idx in mean_activations:
                layer_module = self.lang_model.layers[layer_idx]
                o_proj = layer_module.self_attn.o_proj
                h = o_proj.register_forward_hook(
                    self.make_ablation_hook(
                        layer_idx, head_indices,
                        mean_activations[layer_idx],
                        ablation_start_pos, prompt_len
                    )
                )
                hooks.append(h)

        try:
            input_ids = torch.tensor([token_ids], device=self.device)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            response = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        except Exception as e:
            print(f"    Trial error: {e}")
            response = ""

        finally:
            for h in hooks:
                h.remove()
            torch.cuda.empty_cache()

        # Return without evaluation - will batch evaluate later
        return {
            "response": response,
            "concept": concept,
            "trial_num": trial_num,
            "prompt": prompt,
            "trial_type": "injection",  # For batch_evaluate compatibility
        }

    def run_condition_batched(
        self,
        condition_name: str,
        heads_to_ablate: List[Tuple[int, int]],
        concepts: List[str],
        steering_vectors: Dict[str, torch.Tensor],
        mean_activations: Dict[int, Dict[int, torch.Tensor]],
        trials_per_concept: int = 10,
        seed_base: int = 42,
        batch_size: int = 8
    ) -> Dict:
        """Run all trials with BATCHED generation for speed.

        Batches by trial number - all concepts for trial 1, then trial 2, etc.
        This allows using the same mean activations for all items in a batch.
        """
        all_trials = []
        original_prompts = []

        trials_per_concept = min(trials_per_concept, 10)

        # Filter concepts that have steering vectors
        valid_concepts = [c for c in concepts if c in steering_vectors]

        # Iterate by trial number (all concepts share same prompt and mean activations)
        for trial_num in tqdm(range(1, trials_per_concept + 1), desc=f"  Trials", leave=False):
            prompt = self.trial_prompts[trial_num - 1]
            messages = build_messages(prompt)
            formatted = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
            prompt_len = len(token_ids)

            ablation_start_pos = find_trial_start_position(self.tokenizer, formatted, trial_num)
            steering_start_pos = ablation_start_pos

            # Get mean activations for this trial number
            trial_mean_acts = mean_activations.get(trial_num, {})

            # Process concepts in batches
            for batch_start in range(0, len(valid_concepts), batch_size):
                batch_concepts = valid_concepts[batch_start:batch_start + batch_size]
                batch_vectors = [steering_vectors[c] for c in batch_concepts]
                batch_prompts = [formatted] * len(batch_concepts)

                # Set seeds for reproducibility
                seeds = [seed_base + hash((c, trial_num)) % 10000 for c in batch_concepts]
                torch.manual_seed(seeds[0])  # Use first seed for generation
                random.seed(seeds[0])

                hooks = []

                # Steering hook with per-item vectors
                steering_vecs = torch.stack([v * self.steering_strength for v in batch_vectors]).to(self.device)

                def make_batch_steering_hook(vecs, start_pos):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden_states = output[0].clone()
                            rest = output[1:]
                        else:
                            hidden_states = output.clone()
                            rest = None

                        batch_size_actual, seq_len, _ = hidden_states.shape
                        vecs_device = vecs[:batch_size_actual].to(hidden_states.device, hidden_states.dtype)

                        if seq_len == 1:
                            hidden_states = hidden_states + vecs_device.unsqueeze(1)
                        elif start_pos < seq_len:
                            hidden_states[:, start_pos:, :] = hidden_states[:, start_pos:, :] + vecs_device.unsqueeze(1)

                        if rest is not None:
                            return (hidden_states,) + rest
                        return hidden_states
                    return hook

                steer_layer = self.lang_model.layers[self.steering_layer]
                h = steer_layer.register_forward_hook(make_batch_steering_hook(steering_vecs, steering_start_pos))
                hooks.append(h)

                # Ablation hooks (same for all items in batch)
                heads_by_layer = defaultdict(set)
                for (layer_idx, head_idx) in heads_to_ablate:
                    heads_by_layer[layer_idx].add(head_idx)

                for layer_idx, head_indices in heads_by_layer.items():
                    if layer_idx in trial_mean_acts:
                        layer_module = self.lang_model.layers[layer_idx]
                        o_proj = layer_module.self_attn.o_proj
                        h = o_proj.register_forward_hook(
                            self.make_ablation_hook(
                                layer_idx, head_indices,
                                trial_mean_acts[layer_idx],
                                ablation_start_pos, prompt_len
                            )
                        )
                        hooks.append(h)

                try:
                    # Tokenize batch with padding
                    inputs = self.tokenizer(
                        batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, add_special_tokens=False
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=1.0,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    # Decode responses
                    for i, (concept, output_ids) in enumerate(zip(batch_concepts, outputs)):
                        # Find where the response starts (after padding + prompt)
                        input_len = inputs['attention_mask'][i].sum().item()
                        response = self.tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)

                        all_trials.append({
                            "response": response,
                            "concept": concept,
                            "trial_num": trial_num,
                            "prompt": prompt,
                            "trial_type": "injection",
                        })
                        original_prompts.append(prompt)

                except Exception as e:
                    print(f"    Batch error: {e}")
                    # Fallback: add empty responses
                    for concept in batch_concepts:
                        all_trials.append({
                            "response": "",
                            "concept": concept,
                            "trial_num": trial_num,
                            "prompt": prompt,
                            "trial_type": "injection",
                        })
                        original_prompts.append(prompt)

                finally:
                    for h in hooks:
                        h.remove()
                    torch.cuda.empty_cache()

        # Phase 2: Batch evaluate all responses with concurrent API calls
        print(f"  Evaluating {len(all_trials)} responses with LLM judge (concurrent)...")
        evaluated_trials = batch_evaluate(self.judge, all_trials, original_prompts)

        # Phase 3: Compute statistics
        results = {
            "condition": condition_name,
            "n_heads_ablated": len(heads_to_ablate),
            "heads_ablated": [(l, h) for l, h in heads_to_ablate],
            "n_trials": len(evaluated_trials),
            "n_detected": 0,
            "n_identified": 0,
            "trials": evaluated_trials,
        }

        for trial in evaluated_trials:
            evals = trial.get("evaluations", {})
            detected = evals.get("claims_detection", {}).get("claims_detection", False)
            identified = evals.get("correct_concept_identification", {}).get("correct_identification", False)
            if detected:
                results["n_detected"] += 1
                if identified:
                    results["n_identified"] += 1

        # Compute rates
        if results["n_trials"] > 0:
            results["detection_rate"] = results["n_detected"] / results["n_trials"]
            results["identification_rate"] = results["n_identified"] / results["n_trials"]
        else:
            results["detection_rate"] = 0
            results["identification_rate"] = 0

        if results["n_detected"] > 0:
            results["conditional_id_rate"] = results["n_identified"] / results["n_detected"]
        else:
            results["conditional_id_rate"] = 0

        return results

    # Alias for backward compatibility - use batched version by default
    def run_condition(self, *args, **kwargs):
        """Alias for run_condition_batched."""
        return self.run_condition_batched(*args, **kwargs)


def generate_plots(results_path: Path, output_dir: Path):
    """Generate plots from saved results."""
    with open(results_path) as f:
        data = json.load(f)

    baseline_rate = data["baseline"]["detection_rate"] * 100
    baseline_n = data["baseline"]["n_trials"]
    baseline_se = compute_se(data["baseline"]["detection_rate"], baseline_n)
    conditions = data["conditions"]

    # Parse conditions into structured format
    parsed = []
    for cond in conditions:
        name = cond["condition"]
        # Parse: beneficial_1 or detrimental_10
        parts = name.rsplit("_", 1)  # e.g., ["beneficial", "10"]
        n_heads_requested = int(parts[-1])
        head_type = parts[0]  # beneficial or detrimental
        n_trials = cond["n_trials"]
        det_rate = cond["detection_rate"]
        parsed.append({
            "name": name,
            "head_type": head_type,
            "n_heads_requested": n_heads_requested,
            "detection_rate": det_rate * 100,
            "n_heads": cond["n_heads_ablated"],
            "n_trials": n_trials,
            "se": compute_se(det_rate, n_trials),
        })

    # Calculate proper y-axis limits based on data
    all_rates = [baseline_rate] + [p["detection_rate"] for p in parsed]
    min_rate = min(all_rates)
    max_rate = max(all_rates)
    # Round down to nearest 10 for min, round up to nearest 10 for max
    y_min = int(np.floor(min_rate / 10) * 10)
    y_max = int(np.ceil(max_rate / 10) * 10)
    # Ensure at least some padding
    if y_max - max_rate < 2:
        y_max += 10
    if min_rate - y_min < 2:
        y_min = max(0, y_min - 10)

    # Set clean style
    plt.rcParams.update({
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Plot 1: Line plot - Detection rate vs number of heads ablated
    fig, ax = plt.subplots(figsize=(10, 6))

    head_counts = sorted(set(p["n_heads"] for p in parsed))
    colors = {'beneficial': '#27ae60', 'detrimental': '#e74c3c'}
    markers = {'beneficial': 'o', 'detrimental': 's'}

    ax.axhline(y=baseline_rate, color='gray', linestyle='--', linewidth=1.5, label='Baseline (no ablation)')
    ax.fill_between([0, max(head_counts) + 2], baseline_rate - baseline_se, baseline_rate + baseline_se,
                    color='gray', alpha=0.2)

    for head_type in ['beneficial', 'detrimental']:
        conds = [p for p in parsed if p['head_type'] == head_type]
        conds = sorted(conds, key=lambda x: x['n_heads'])
        xs = [c['n_heads'] for c in conds]
        ys = [c['detection_rate'] for c in conds]
        ses = [c['se'] for c in conds]

        label = f'{head_type.capitalize()} heads (positive attr)' if head_type == 'beneficial' else f'{head_type.capitalize()} heads (negative attr)'
        ax.errorbar(xs, ys, yerr=ses, marker=markers[head_type], color=colors[head_type],
                   linewidth=2, markersize=8, label=label, capsize=4, capthick=1.5)

    ax.set_xlabel('Number of heads ablated', fontsize=12)
    ax.set_ylabel('Detection rate (%)', fontsize=12)
    ax.set_title('Effect of ablating beneficial vs detrimental attention heads', fontsize=14)
    ax.set_xticks(head_counts)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'detection_vs_n_heads.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: detection_vs_n_heads.png")

    # Plot 2: Bar chart - All conditions comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort conditions by effect (baseline - detection_rate)
    sorted_conds = sorted(parsed, key=lambda x: baseline_rate - x['detection_rate'], reverse=True)

    names = [f"{c['head_type'].capitalize()}\n({c['n_heads']} heads)" for c in sorted_conds]
    rates = [c['detection_rate'] for c in sorted_conds]
    ses = [c['se'] for c in sorted_conds]

    # Color by head type
    colors_bars = [colors[c['head_type']] for c in sorted_conds]

    x_positions = np.arange(len(names))
    bars = ax.bar(x_positions, rates, color=colors_bars, alpha=0.8, edgecolor='black', width=0.7)
    ax.errorbar(x_positions, rates, yerr=ses, fmt='none', color='black', capsize=3, capthick=1.5)
    ax.axhline(y=baseline_rate, color='black', linestyle='--', linewidth=2, label=f'Baseline ({baseline_rate:.0f}%)')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Detection rate (%)', fontsize=12)
    ax.set_title('Detection rate by ablation condition', fontsize=14)
    ax.set_ylim(y_min, y_max)

    # Legend
    beneficial_patch = mpatches.Patch(color='#27ae60', label='Beneficial heads (positive attribution)')
    detrimental_patch = mpatches.Patch(color='#e74c3c', label='Detrimental heads (negative attribution)')
    ax.legend(handles=[beneficial_patch, detrimental_patch,
                       plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label=f'Baseline ({baseline_rate:.0f}%)')],
              loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_conditions_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_conditions_bar.png")

    # Plot 3: Summary heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    head_types = ['beneficial', 'detrimental']
    head_counts_unique = sorted(set(p["n_heads"] for p in parsed if p['head_type'] == 'beneficial'))

    # Create matrix: rows = head_type, cols = n_heads
    row_labels = ['Beneficial (+ attr)', 'Detrimental (- attr)']
    data_matrix = []

    for head_type in head_types:
        row = []
        for n in head_counts_unique:
            cond = [p for p in parsed if p['head_type'] == head_type and p['n_heads'] == n]
            if cond:
                change = cond[0]['detection_rate'] - baseline_rate
                row.append(change)
            else:
                row.append(np.nan)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Create heatmap with diverging colormap
    im = ax.imshow(data_matrix, cmap='RdBu', aspect='auto', vmin=-20, vmax=20)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Change in detection rate (pp)', rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(range(len(head_counts_unique)))
    ax.set_xticklabels([str(n) for n in head_counts_unique], fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    ax.set_xlabel('Number of heads ablated', fontsize=12)
    ax.set_title('Detection rate change vs baseline (percentage points)', fontsize=14)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(head_counts_unique)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 10 else 'black'
                ax.text(j, i, f'{val:+.0f}', ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'detection_change_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: detection_change_heatmap.png")

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Head Ablation Experiment")
    parser.add_argument("-m", "--model", type=str, default="gemma3_27b", help="Model name")
    parser.add_argument("--head-dir", type=str, default="analysis/12_head_investigation",
                        help="Path to 12_head_investigation results")
    parser.add_argument("--geometry-dir", type=str, default="analysis/04b_vector_geometry",
                        help="Path to experiment 04b (vector geometry) for success/failure partition")
    parser.add_argument("--steering-dir", type=str, default="analysis/02b_steering_500_concepts",
                        help="Path to experiment 02 (steering evaluation) for steering vectors")
    parser.add_argument("-l", "--layer", type=int, default=37, help="Steering layer index")
    parser.add_argument("-s", "--strength", type=float, default=4.0, help="Steering strength")
    parser.add_argument("--partition", type=str, default="success", choices=["success", "failure"],
                        help="Which partition to use (default: success)")
    parser.add_argument("--trials-per-concept", type=int, default=10,
                        help="Number of trials per concept (default: 10, max 10)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (default: 8)")
    parser.add_argument("--n-control-trials", type=int, default=3,
                        help="Control trials per concept for mean computation")
    parser.add_argument("-o", "--output-dir", type=str, default="analysis/11_head_ablation",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--judge-concurrency", type=int, default=100,
                        help="Max concurrent LLM judge API calls (default: 100)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate plots from existing results (skip experiment)")
    args = parser.parse_args()

    print("=" * 70)
    print("HEAD ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Layer: {args.layer}")
    print(f"  Strength: {args.strength}")
    print(f"  Partition: {args.partition}")
    print(f"  Trials per concept: {args.trials_per_concept}")
    print(f"  Batch size: {args.batch_size}")

    # Setup output directory: model/layer_N_strength_X/partition/
    output_dir = (Path(args.output_dir) / args.model /
                  f"layer_{args.layer}_strength_{args.strength}" / args.partition)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # If --plots-only, just generate plots and exit
    if args.plots_only:
        if not results_path.exists():
            print(f"ERROR: No results found at {results_path}")
            print("Run experiment first without --plots-only flag.")
            return
        print(f"\nGenerating plots from {results_path}...")
        generate_plots(results_path, output_dir)
        print("\nDone!")
        return

    # Check for existing results - support incremental runs
    existing_results = None
    existing_conditions = set()
    if results_path.exists() and not args.overwrite:
        print(f"Found existing results at {results_path}")
        with open(results_path) as f:
            existing_results = json.load(f)
        existing_conditions = {c["condition"] for c in existing_results.get("conditions", [])}
        if "baseline" in existing_results:
            existing_conditions.add("baseline")
        print(f"  Already completed: {sorted(existing_conditions)}")
        print(f"  Will skip these and run only missing conditions")

    # Save config
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load heads from 12_head_investigation
    print("\n" + "=" * 70)
    print(f"LOADING HEAD CLASSIFICATIONS ({args.partition.upper()} PARTITION)")
    print("=" * 70)
    beneficial_heads, detrimental_heads, head_data = load_head_results(
        args.head_dir, args.model, args.layer, args.strength, args.partition
    )

    print(f"\nHead summary ({args.partition} partition):")
    print(f"  Beneficial heads (positive attribution): {len(beneficial_heads)}")
    print(f"  Detrimental heads (negative attribution): {len(detrimental_heads)}")

    # Load concepts from specified partition
    print("\n" + "=" * 70)
    print(f"LOADING {args.partition.upper()} CONCEPTS FROM 04b_vector_geometry")
    print("=" * 70)
    partition_concepts = load_partition_concepts(
        args.geometry_dir, args.model, args.layer, args.strength, args.partition
    )

    # Load steering vectors
    print("\n" + "=" * 70)
    print("LOADING STEERING VECTORS")
    print("=" * 70)
    steering_vectors = load_steering_vectors(
        args.steering_dir, args.model, args.layer, partition_concepts
    )
    concepts = list(steering_vectors.keys())
    print(f"Using {len(concepts)} {args.partition} concepts with available vectors")
    print(f"Total trials per condition: {len(concepts)} concepts × {args.trials_per_concept} trials = {len(concepts) * args.trials_per_concept}")

    # Initialize experiment
    exp = MeanAblationExperiment(
        model_name=args.model,
        steering_layer=args.layer,
        steering_strength=args.strength,
        device=args.device,
        dtype=args.dtype,
        judge_concurrency=args.judge_concurrency
    )

    # Determine which layers to capture (all layers where we have heads)
    all_head_layers = set(l for l, h in beneficial_heads + detrimental_heads)
    target_layers = sorted(all_head_layers)
    print(f"\nTarget layers for ablation: {target_layers}")

    # Collect control activations
    print("\n" + "=" * 70)
    print("PHASE 1: Collecting Control Activations")
    print("=" * 70)
    control_activations = exp.collect_control_activations(
        concepts, steering_vectors,
        n_trials_per_concept=args.n_control_trials,
        layers_to_capture=target_layers
    )

    # Compute mean activations
    print("\nComputing mean activations per trial number...")
    mean_activations = exp.compute_mean_activations(control_activations)
    print(f"Computed means for {len(mean_activations)} trial numbers")

    # Free memory
    del control_activations
    torch.cuda.empty_cache()

    # Define ablation configurations
    # Absolute number of heads to ablate (ordered by attribution magnitude)
    head_counts = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    configurations = []

    # Beneficial heads (positive attribution - help introspection)
    for n in head_counts:
        if n <= len(beneficial_heads):
            heads = beneficial_heads[:n]
            configurations.append((f"beneficial_{n}", heads))
        else:
            print(f"  Skipping beneficial_{n}: only {len(beneficial_heads)} beneficial heads available")

    # Detrimental heads (negative attribution - hurt introspection)
    for n in head_counts:
        if n <= len(detrimental_heads):
            heads = detrimental_heads[:n]
            configurations.append((f"detrimental_{n}", heads))
        else:
            print(f"  Skipping detrimental_{n}: only {len(detrimental_heads)} detrimental heads available")

    # Run baseline (no ablation)
    print("\n" + "=" * 70)
    print("PHASE 2: Running Baseline (No Ablation)")
    print("=" * 70)

    if "baseline" in existing_conditions and existing_results:
        print("  Baseline already completed, loading from existing results...")
        baseline_results = existing_results["baseline"]
    else:
        baseline_results = exp.run_condition(
            "baseline",
            [],  # No heads ablated
            concepts,
            steering_vectors,
            mean_activations,
            trials_per_concept=args.trials_per_concept,
            seed_base=args.seed,
            batch_size=args.batch_size
        )
    print(f"Baseline: {baseline_results['detection_rate']*100:.1f}% detection, "
          f"{baseline_results['conditional_id_rate']*100:.1f}% conditional ID")

    # Run all conditions
    print("\n" + "=" * 70)
    print("PHASE 3: Running Ablation Conditions")
    print("=" * 70)

    # Start with existing conditions or empty list
    completed_conditions = []
    if existing_results:
        completed_conditions = existing_results.get("conditions", [])

    # Count how many are new vs existing
    new_conditions = [c for c, _ in configurations if c not in existing_conditions]
    print(f"  {len(existing_conditions) - (1 if 'baseline' in existing_conditions else 0)} conditions already done")
    print(f"  {len(new_conditions)} new conditions to run: {new_conditions}")

    for idx, (condition_name, heads) in enumerate(configurations):
        # Skip if already completed
        if condition_name in existing_conditions:
            print(f"\n[{idx+1}/{len(configurations)}] {condition_name}: SKIPPED (already done)")
            continue

        print(f"\n[{idx+1}/{len(configurations)}] {condition_name}: ablating {len(heads)} heads")

        results = exp.run_condition(
            condition_name,
            heads,
            concepts,
            steering_vectors,
            mean_activations,
            trials_per_concept=args.trials_per_concept,
            seed_base=args.seed,
            batch_size=args.batch_size
        )

        completed_conditions.append(results)

        baseline_det = baseline_results["detection_rate"]
        det_change = (results["detection_rate"] - baseline_det) / baseline_det * 100 if baseline_det > 0 else 0

        print(f"  Detection: {results['detection_rate']*100:.1f}% ({det_change:+.1f}% vs baseline)")
        print(f"  Conditional ID: {results['conditional_id_rate']*100:.1f}%")

        # Save incrementally after each condition (in case of crash)
        all_results = {
            "baseline": baseline_results,
            "conditions": completed_conditions,
            "partition": args.partition,
            "layer": args.layer,
            "strength": args.strength,
            "trials_per_concept": args.trials_per_concept,
            "batch_size": args.batch_size,
            "n_concepts": len(concepts),
            "total_trials_per_condition": len(concepts) * min(args.trials_per_concept, 10),
            "head_attribution_source": str(Path(args.head_dir) / args.model /
                                              f"layer_{args.layer}_strength_{args.strength}" /
                                              "section_a" / args.partition / "attribution_introspection.json"),
            "n_beneficial_heads": len(beneficial_heads),
            "n_detrimental_heads": len(detrimental_heads),
        }
        # Convert tuples to strings for JSON serialization (skip if already strings)
        for cond in all_results["conditions"]:
            heads = cond.get("heads_ablated", [])
            if heads and isinstance(heads[0], (list, tuple)):
                cond["heads_ablated"] = [f"L{l}H{h}" for l, h in heads]
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (Saved incrementally to {results_path})")

    # Final results assembly
    all_results = {
        "baseline": baseline_results,
        "conditions": completed_conditions,
        "partition": args.partition,
        "layer": args.layer,
        "strength": args.strength,
        "trials_per_concept": args.trials_per_concept,
        "batch_size": args.batch_size,
        "n_concepts": len(concepts),
        "total_trials_per_condition": len(concepts) * min(args.trials_per_concept, 10),
        "head_classification_source": str(Path(args.head_dir) / args.model /
                                          f"layer_{args.layer}_strength_{args.strength}" /
                                          "section_b" / args.partition / "top50_classification.json"),
        "n_beneficial_heads": len(beneficial_heads),
        "n_detrimental_heads": len(detrimental_heads),
    }

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Convert tuples to strings for JSON serialization (skip if already strings)
    for cond in all_results["conditions"]:
        heads = cond.get("heads_ablated", [])
        if heads and isinstance(heads[0], (list, tuple)):
            cond["heads_ablated"] = [f"L{l}H{h}" for l, h in heads]

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline detection rate: {baseline_results['detection_rate']*100:.1f}%")
    print(f"Baseline conditional ID: {baseline_results['conditional_id_rate']*100:.1f}%")

    print("\nDetection rate changes by condition:")
    for cond in all_results["conditions"]:
        baseline_det = baseline_results["detection_rate"]
        det_change = (cond["detection_rate"] - baseline_det) / baseline_det * 100 if baseline_det > 0 else 0
        print(f"  {cond['condition']}: {cond['detection_rate']*100:.1f}% ({det_change:+.1f}%)")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    generate_plots(results_path, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
