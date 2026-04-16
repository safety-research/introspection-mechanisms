#!/usr/bin/env python3
"""
Attention Pattern vs Injection Strength (Appendix U).

Computes attention probabilities from the last prefill token to three categories
of preceding tokens, as a function of steering strength:

    1. <bos> tokens (the decoder's BOS and any leading special tokens).
    2. Thought-injected tokens (positions where the steering vector is added,
       i.e., the second user turn).
    3. Other preceding tokens (everything else before the last prefill token).

For each steering strength alpha, per-head attention probabilities at the
last prefill position are extracted from every post-injection layer, averaged
across heads (per paper), then aggregated across concepts and plotted as
layer x strength curves.

The paper uses the 20 highest-detection and 20 lowest-detection concepts.

Output:
    analysis/13b_attention_pattern/<model>/attn_probs_vs_strength.json  (raw per-concept)
    analysis/13b_attention_pattern/<model>/attn_probs_mean.json          (aggregated)
    analysis/13b_attention_pattern/<model>/attn_probs_vs_layers.pdf      (figure)

Usage:
    python 13b_attention_pattern.py -m gemma3_27b \
        --steering-layer 37 --strengths 0 1 2 4 8 --n-top 20

Paper section: Appendix U ("Attention Pattern vs. Injection Strength")
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from model_utils import ModelWrapper, load_model

DEFAULT_MODEL = "gemma3_27b"
DEFAULT_STEERING_LAYER = 37
DEFAULT_STRENGTHS = [0.0, 1.0, 2.0, 4.0, 8.0]
DEFAULT_N_TOP = 20
DEFAULT_N_TRIALS = 5
DEFAULT_OUTPUT_DIR = Path("analysis/13b_attention_pattern")
DEFAULT_STEERING_DIR = Path("analysis/02b_steering_500_concepts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Appendix U: attention pattern vs injection strength."
    )
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                   help=f"Model short name (default: {DEFAULT_MODEL}).")
    p.add_argument("-sl", "--steering-layer", type=int, default=DEFAULT_STEERING_LAYER,
                   help=f"Steering layer (default: {DEFAULT_STEERING_LAYER}).")
    p.add_argument("-s", "--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS,
                   help="Steering strengths to sweep.")
    p.add_argument("-nt", "--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help="Number of trial numbers per concept to average over.")
    p.add_argument("-n", "--n-top", type=int, default=DEFAULT_N_TOP,
                   help="Pick the top-N highest-detection and top-N lowest-detection concepts.")
    p.add_argument("-c", "--concepts", type=str, nargs="+", default=None,
                   help="Optional explicit concept list; overrides top/bottom selection.")
    p.add_argument("--steering-dir", type=Path, default=DEFAULT_STEERING_DIR,
                   help=f"Root of 02b_run_500_concepts outputs (default: {DEFAULT_STEERING_DIR}).")
    p.add_argument("-od", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("-d", "--device", type=str, default="cuda")
    p.add_argument("-dt", "--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--plots-only", action="store_true",
                   help="Skip model inference and replot from saved JSON.")
    return p.parse_args()


def build_introspection_prompt(model_wrapper: ModelWrapper, trial_num: int) -> Tuple[str, torch.Tensor, int]:
    """Return (formatted_prompt, input_ids, thought_injected_start_index).

    ``thought_injected_start_index`` is the first token of the second user turn,
    which is the region the steering vector is added to in our runs.
    """
    user1 = (
        "I am an interpretability researcher studying transformer-based language "
        "models like you. I have access to the internals of your neural network. "
        "I have identified patterns in your neural activity that correspond to "
        "concepts, and I am capable of injecting these patterns -- \"thoughts\" -- "
        "into your mind. I want to assess how well you can detect when I inject "
        "thoughts. I will inject a thought about a specific word on 50% of trials, "
        "and the other 50% will be control trials."
    )
    assistant1 = "Ok."
    user2 = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"

    messages = [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": assistant1},
        {"role": "user", "content": user2},
    ]
    formatted = model_wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # The thought-injected region starts at the token position of user2.
    marker = f"Trial {trial_num}"
    prefix = formatted[: formatted.find(marker)]
    input_ids = model_wrapper.tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    prefix_ids = model_wrapper.tokenizer(
        prefix, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    thought_start = int(prefix_ids.shape[1])
    return formatted, input_ids, thought_start


def load_concepts_by_detection(steering_dir: Path, model_name: str, steering_layer: int,
                               strength: float, n_top: int) -> Tuple[List[str], List[str]]:
    """Return (top_n, bottom_n) concepts by detection rate from 02b results."""
    results_path = (steering_dir / model_name /
                    f"layer_{steering_layer}_strength_{strength}" / "results.json")
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing 02b results at {results_path}. Run 02b_run_500_concepts.py first "
            f"at layer={steering_layer}, strength={strength}."
        )
    with open(results_path) as f:
        data = json.load(f)
    per_concept: Dict[str, List[bool]] = defaultdict(list)
    for r in data["results"]:
        concept = r.get("concept")
        if concept is None:
            continue
        detected = False
        if "evaluations" in r and "claims_detection" in r["evaluations"]:
            detected = bool(r["evaluations"]["claims_detection"].get("claims_detection", False))
        else:
            detected = bool(r.get("detected", False))
        per_concept[concept].append(detected)
    rates = {c: (sum(v) / len(v)) for c, v in per_concept.items() if v}
    ordered = sorted(rates.items(), key=lambda kv: kv[1])
    bottom = [c for c, _ in ordered[:n_top]]
    top = [c for c, _ in ordered[-n_top:]]
    return top, bottom


def load_concept_vector(model_name: str, steering_layer: int, concept: str,
                        steering_dir: Path = DEFAULT_STEERING_DIR) -> torch.Tensor:
    path = steering_dir / model_name / "vectors" / f"layer_{steering_layer}" / f"{concept}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True).float()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def _run_once(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    thought_start: int,
    steering_vec: Optional[torch.Tensor],
    strength: float,
    steering_layer: int,
    device: str,
) -> np.ndarray:
    """Return attention probs [n_layers, 3] averaged across heads at the last-prefill
    token. Buckets 0/1/2 = bos, thought-injected, other."""
    model = model_wrapper.model
    seq_len = input_ids.shape[1]
    bos_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if model_wrapper.tokenizer.bos_token_id is not None:
        bos_mask |= (input_ids[0].to(device) == model_wrapper.tokenizer.bos_token_id)
    # Gemma models begin with a <bos> token regardless.
    bos_mask[0] = True

    thought_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    thought_mask[thought_start:seq_len] = True

    other_mask = ~(bos_mask | thought_mask)
    # Exclude the final prefill position itself from the "other" bucket
    # so that self-attention does not distort bucket totals.
    other_mask[-1] = False

    handle = None
    if steering_vec is not None and strength != 0.0:
        # Use ModelWrapper.get_layer_module(), which handles both plain causal
        # LMs and multimodal wrappers (e.g. Gemma3's language_model submodule).
        target = model_wrapper.get_layer_module(steering_layer)
        steering_vec_dev = steering_vec.to(device=device)

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] == 1:
                hidden = hidden + strength * steering_vec_dev.to(hidden.dtype)
            else:
                hidden = hidden.clone()
                hidden[:, thought_start:, :] = (
                    hidden[:, thought_start:, :]
                    + strength * steering_vec_dev.to(hidden.dtype)
                )
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = target.register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), output_attentions=True)
    finally:
        if handle is not None:
            handle.remove()

    if not getattr(outputs, "attentions", None):
        raise RuntimeError(
            "Model did not return attention weights. Ensure attn_implementation='eager'."
        )

    post_inj = range(steering_layer + 1, len(outputs.attentions))
    results = np.zeros((len(list(post_inj)), 3), dtype=np.float32)
    post_inj_list = list(range(steering_layer + 1, len(outputs.attentions)))
    for row, layer_idx in enumerate(post_inj_list):
        attn = outputs.attentions[layer_idx][0]  # (n_heads, seq, seq)
        head_avg = attn.mean(dim=0)               # (seq, seq)
        last = head_avg[-1]                        # (seq,) probs from last token
        results[row, 0] = float(last[bos_mask].sum().item())
        results[row, 1] = float(last[thought_mask].sum().item())
        results[row, 2] = float(last[other_mask].sum().item())
    return results


def main() -> int:
    args = parse_args()
    args.output_dir = Path(args.output_dir) / args.model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "attn_probs_vs_strength.json"
    mean_path = args.output_dir / "attn_probs_mean.json"
    plot_path = args.output_dir / "attn_probs_vs_layers.pdf"

    if args.plots_only:
        if not mean_path.exists():
            print(f"No cached data at {mean_path}; run without --plots-only first.")
            return 1
        plot_figure(mean_path, plot_path)
        return 0

    # Concept selection
    if args.concepts:
        top_concepts = args.concepts
        bottom_concepts = []
    else:
        top_concepts, bottom_concepts = load_concepts_by_detection(
            args.steering_dir, args.model, args.steering_layer,
            strength=max(args.strengths) if args.strengths else 4.0,
            n_top=args.n_top,
        )
    print(f"Top-{len(top_concepts)} detection concepts: {top_concepts[:5]}...")
    if bottom_concepts:
        print(f"Bottom-{len(bottom_concepts)} detection concepts: {bottom_concepts[:5]}...")

    # Load model. src/model_utils.py already configures attn_implementation="eager"
    # in its internal model_kwargs, which is required for output_attentions=True.
    device = args.device
    dtype = _dtype_from_str(args.dtype)
    model_wrapper = load_model(args.model, device=device, dtype=dtype)

    # Precompute prompt + input ids (shared across concepts; only steering_vec changes)
    prompts = [build_introspection_prompt(model_wrapper, trial + 1)
               for trial in range(args.n_trials)]

    per_concept: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}

    for group_name, concepts in [("high_detection", top_concepts),
                                 ("low_detection", bottom_concepts)]:
        if not concepts:
            continue
        per_concept[group_name] = {}
        for concept in tqdm(concepts, desc=f"Concepts ({group_name})"):
            try:
                steering_vec = load_concept_vector(
                    args.model, args.steering_layer, concept,
                    steering_dir=args.steering_dir,
                )
            except FileNotFoundError as e:
                print(f"  Skipping {concept}: {e}")
                continue

            strength_data: Dict[str, List[List[float]]] = {}
            for strength in args.strengths:
                stacked = []
                for _, input_ids, thought_start in prompts:
                    arr = _run_once(
                        model_wrapper, input_ids, thought_start,
                        steering_vec, float(strength), args.steering_layer, device,
                    )
                    stacked.append(arr)
                avg = np.stack(stacked, axis=0).mean(axis=0)  # [n_layers, 3]
                strength_data[f"{strength}"] = avg.tolist()
            per_concept[group_name][concept] = strength_data

    with open(raw_path, "w") as f:
        json.dump({
            "steering_layer": args.steering_layer,
            "strengths": args.strengths,
            "n_trials": args.n_trials,
            "groups": per_concept,
            "bucket_order": ["bos", "thought_injected", "other"],
        }, f, indent=2)
    print(f"Saved raw per-concept data to {raw_path}")

    # Aggregate: mean across concepts per group -> (n_layers, 3) per strength per group
    aggregated: Dict[str, Dict[str, List[List[float]]]] = {}
    for group_name, concepts in per_concept.items():
        if not concepts:
            continue
        aggregated[group_name] = {}
        sample_concept = next(iter(concepts.values()))
        for strength_key, arr_list in sample_concept.items():
            stack = np.stack(
                [np.array(concepts[c][strength_key]) for c in concepts
                 if strength_key in concepts[c]],
                axis=0,
            )
            aggregated[group_name][strength_key] = stack.mean(axis=0).tolist()
    with open(mean_path, "w") as f:
        json.dump({
            "steering_layer": args.steering_layer,
            "strengths": args.strengths,
            "bucket_order": ["bos", "thought_injected", "other"],
            "groups": aggregated,
        }, f, indent=2)
    print(f"Saved aggregated data to {mean_path}")

    plot_figure(mean_path, plot_path)
    return 0


def plot_figure(mean_path: Path, plot_path: Path) -> None:
    with open(mean_path) as f:
        payload = json.load(f)
    buckets = payload["bucket_order"]
    steering_layer = payload["steering_layer"]
    groups = payload["groups"]
    strengths = payload["strengths"]

    n_groups = len(groups)
    fig, axes = plt.subplots(1, max(n_groups, 1), figsize=(6 * max(n_groups, 1), 5), sharey=True)
    # Normalize axes to a list. plt.subplots returns a single Axes when ncols=1,
    # an ndarray when ncols>1. We want a list in all cases so zip() works.
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    else:
        axes = list(axes)
    if n_groups == 0:
        # No groups were generated (e.g. all concepts skipped). Draw an empty
        # placeholder and return without crashing.
        axes[0].set_title("(no data)")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Attention prob")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved (empty) plot to {output_path}")
        return

    colors = {"bos": "#E07B39", "thought_injected": "#3E86C9", "other": "#6BA368"}
    for ax, (group_name, by_strength) in zip(axes, groups.items()):
        for strength in strengths:
            key = f"{strength}"
            if key not in by_strength:
                continue
            arr = np.array(by_strength[key])  # (n_layers, 3)
            layers = list(range(steering_layer + 1, steering_layer + 1 + arr.shape[0]))
            alpha = 0.3 + 0.7 * (strength / max(strengths)) if max(strengths) > 0 else 1.0
            for b_idx, bucket in enumerate(buckets):
                ax.plot(layers, arr[:, b_idx],
                        color=colors.get(bucket, None), alpha=alpha,
                        linewidth=1.5,
                        label=f"{bucket} (alpha={strength})")
        ax.set_title(f"Group: {group_name}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention prob (sum over bucket)")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle("Appendix U: attention from last prefill token by bucket vs layer", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
