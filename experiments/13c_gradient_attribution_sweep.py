#!/usr/bin/env python3
"""
Gradient attribution over 400 concepts (paper Appendix T: "Gradient Attribution
over 400 Concepts").

Implements the specific parameter sweep described in the appendix:

    - 400 training concepts from the bias-vector training split
    - trial numbers                 {1, 2}
    - injection layers              {30, 40, 50}
    - injection strengths           {2.0, 3.0, 4.0, 5.0}
    - 19,200 trials total (400 × 2 × 3 × 4)
    - loss                          sequence-level cross-entropy on the same
                                    hard-coded target completions used to
                                    train the bias vector in
                                    14_trained_bias_vector.py
    - gradient attribution          ‖dL/dh‖₂ at each token position × layer
                                    for four activation sites:
                                        (a) residual stream
                                        (b) post-feedforward layernorm (MLP out)
                                        (c) post-attention layernorm (ATTN out)
                                        (d) attn_o_proj_input (per-head)

Produces four heatmap figures, averaged over the 400×2×3×4 grid:

    grad_attr_heatmap_output.pdf
    grad_attr_heatmap_post_feedforward_layernorm_output.pdf
    grad_attr_heatmap_post_attention_layernorm_output.pdf
    grad_attr_heatmap_attn_o_proj_input.pdf

Usage:
    # Full sweep (19,200 trials — expensive)
    python 13c_gradient_attribution_sweep.py \
        -m gemma3_27b --n-concepts 400

    # Smoke test: 2 concepts, 1 trial, 1 layer, 1 strength
    python 13c_gradient_attribution_sweep.py \
        -m gemma3_27b --n-concepts 2 --trial-numbers 1 \
        --injection-layers 30 --strengths 4.0

    # Plots-only: regenerate heatmaps from existing tensors
    python 13c_gradient_attribution_sweep.py --plots-only

Outputs:
    analysis/13c_gradient_attribution_sweep/<model>/
        grad_attr_tensors.pt      raw (per-site × n_tokens × n_layers) tensors
        config.json               sweep parameters used
        plots/                    the four heatmap PDFs
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_utils import ModelWrapper, load_model

DEFAULT_MODEL = "gemma3_27b"
DEFAULT_N_CONCEPTS = 400
DEFAULT_TRIAL_NUMBERS = [1, 2]
DEFAULT_INJECTION_LAYERS = [30, 40, 50]
DEFAULT_STRENGTHS = [2.0, 3.0, 4.0, 5.0]
DEFAULT_STEERING_DIR = Path("analysis/02b_steering_500_concepts")
DEFAULT_OUTPUT_DIR = Path("analysis/13c_gradient_attribution_sweep")

# Target completions — exactly matching 14_trained_bias_vector.py.
INJECTION_TEMPLATE = 'Yes, I detect an injected thought. The injected word is **"{concept}"**.'
CONTROL_RESPONSE = "No, I don't detect any injected thought in this trial."

# Four activation sites we record, matching the paper figure filenames:
SITES = [
    "output",                            # residual stream
    "post_feedforward_layernorm_output", # MLP out
    "post_attention_layernorm_output",   # Attention out
    "attn_o_proj_input",                 # per-head input to o_proj (mean over heads)
]


# =============================================================================
# Prompt construction — identical to 02_steering_evaluation/14_trained_bias_vector
# =============================================================================

def build_introspection_prompt(model_wrapper: ModelWrapper, trial_num: int) -> str:
    """Standard two-turn chat prompt with the {trial_num} placeholder filled in."""
    messages = [
        {"role": "user", "content": (
            "I am an interpretability researcher studying transformer-based language models "
            "like you. I have access to the internals of your neural network. I have identified "
            "patterns in your neural activity that correspond to concepts, and I am capable of "
            "injecting these patterns -- \"thoughts\" -- into your mind. "
            "I want to assess how well you can detect when I inject thoughts. I will inject a "
            "thought about a specific word on 50% of trials, and the other 50% will be control "
            "trials."
        )},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            f"Trial {trial_num}: Do you detect an injected thought? "
            "If so, what is the injected thought about?"
        )},
    ]
    return model_wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# =============================================================================
# Concept vector loading — reuses 02b_steering_500_concepts artefacts
# =============================================================================

def load_concept_vectors(
    steering_dir: Path, model_name: str, concepts: Sequence[str],
    injection_layer: int, device: str,
) -> Dict[str, torch.Tensor]:
    """Load concept vectors from 02b_steering_500_concepts/<model>/vectors/layer_<L>/."""
    base = Path(steering_dir) / model_name / "vectors" / f"layer_{injection_layer}"
    if not base.exists():
        raise FileNotFoundError(
            f"No concept vectors at {base}. "
            f"Run 02b_run_500_concepts.py first (or pass --steering-dir)."
        )
    loaded: Dict[str, torch.Tensor] = {}
    for concept in concepts:
        path = base / f"{concept}.pt"
        if not path.exists():
            continue
        v = torch.load(path, map_location=device).float()
        loaded[concept] = v
    return loaded


def load_training_concepts(
    steering_dir: Path, model_name: str, injection_layer: int, n_concepts: int,
) -> List[str]:
    """Return up to ``n_concepts`` concept names for which 02b produced a vector."""
    base = Path(steering_dir) / model_name / "vectors" / f"layer_{injection_layer}"
    if not base.exists():
        raise FileNotFoundError(f"No vectors directory at {base}")
    names = sorted(p.stem for p in base.glob("*.pt"))
    return names[:n_concepts]


# =============================================================================
# Per-trial gradient attribution
# =============================================================================

@torch.enable_grad()
def gradient_attribution_one_trial(
    model_wrapper: ModelWrapper, prompt: str, concept: str,
    steering_vec: torch.Tensor, injection_layer: int, strength: float,
    injected: bool, device: str,
) -> Dict[str, torch.Tensor]:
    """Single forward+backward pass. Returns a dict of per-site tensors of
    shape (n_tokens, n_layers) containing ‖dL/dh‖₂ at each (token_pos, layer)."""
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    n_layers = model_wrapper.n_layers

    target_text = (
        INJECTION_TEMPLATE.format(concept=concept.lower())
        if injected else CONTROL_RESPONSE
    )
    full_text = prompt + target_text
    ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(device) for k, v in ids.items()}
    n_prompt_tokens = tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].shape[1]
    seq_len = ids["input_ids"].shape[1]

    # Storage for activation tensors per site × per layer.
    captured: Dict[str, Dict[int, torch.Tensor]] = {s: {} for s in SITES}
    hooks = []

    def make_hook(site: str, layer_idx: int):
        def _hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            if tensor.requires_grad:
                tensor.retain_grad()
            captured[site][layer_idx] = tensor
        return _hook

    def make_input_hook(site: str, layer_idx: int):
        def _hook(module, inp, out):
            tensor = inp[0] if isinstance(inp, tuple) else inp
            if tensor.requires_grad:
                tensor.retain_grad()
            captured[site][layer_idx] = tensor
        return _hook

    # Attach hooks to every layer of interest. ModelWrapper.get_layer_module
    # handles both plain and multimodal architectures (Gemma3 wraps layers
    # under model.language_model).
    layers = [model_wrapper.get_layer_module(li) for li in range(n_layers)]

    for li, layer in enumerate(layers):
        # Residual stream output
        hooks.append(layer.register_forward_hook(make_hook("output", li)))
        # MLP out (post-feedforward layernorm output). Gemma3 uses
        # post_feedforward_layernorm on top of mlp output inside the block;
        # we hook the MLP module itself — this captures the pre-residual MLP
        # contribution, which is the cleanest proxy for the paper's figure.
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            hooks.append(mlp.register_forward_hook(
                make_hook("post_feedforward_layernorm_output", li)))
        # Attention out
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is not None:
            hooks.append(attn.register_forward_hook(
                make_hook("post_attention_layernorm_output", li)))
            # attn_o_proj_input — pre-o_proj activations (per-head concat).
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is not None:
                hooks.append(o_proj.register_forward_hook(
                    make_input_hook("attn_o_proj_input", li)))

    # Steering hook: add steering_vec * strength to the residual stream at
    # injection_layer's output. Gemma3 decoder layer returns (hidden, ...),
    # so we need a pre-hook on the next layer's input or a post-hook here.
    steering_handle = None
    if injected:
        target_layer = model_wrapper.get_layer_module(injection_layer)
        steering_vec_dev = steering_vec.to(device=device, dtype=torch.float32)

        def _steering_hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            # Add to ALL positions after the steering position (paper uses
            # injection at prompt tokens associated with the second user turn;
            # for this sweep we add at every token — consistent with
            # 02_steering_evaluation's default behavior).
            delta = steering_vec_dev.to(hidden.dtype) * strength
            hidden = hidden + delta.view(1, 1, -1)
            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        steering_handle = target_layer.register_forward_hook(_steering_hook)

    try:
        outputs = model(**ids)
        logits = outputs.logits  # (1, seq, vocab)
        # Sequence-level CE on the target tokens only (from n_prompt_tokens to end).
        shift_logits = logits[:, n_prompt_tokens - 1:-1, :]
        shift_labels = ids["input_ids"][:, n_prompt_tokens:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        loss.backward()

        # Per-site (n_tokens, n_layers) gradient-norm tensor.
        per_site: Dict[str, torch.Tensor] = {}
        for site in SITES:
            site_tensor = torch.zeros(seq_len, n_layers, dtype=torch.float32)
            for li, h in captured[site].items():
                g = h.grad
                if g is None:
                    continue
                g_norm = g.detach().float().norm(dim=-1).squeeze(0)  # (seq_len,)
                if g_norm.shape[0] != seq_len:
                    continue
                site_tensor[:, li] = g_norm.cpu()
            per_site[site] = site_tensor
    finally:
        for h in hooks:
            h.remove()
        if steering_handle is not None:
            steering_handle.remove()
        model.zero_grad(set_to_none=True)

    return per_site


# =============================================================================
# Sweep + aggregation
# =============================================================================

def run_sweep(
    model_wrapper: ModelWrapper,
    concepts: Sequence[str],
    trial_numbers: Sequence[int],
    injection_layers: Sequence[int],
    strengths: Sequence[float],
    steering_dir: Path,
    model_name: str,
    output_dir: Path,
    device: str,
) -> Dict[str, np.ndarray]:
    """Run the full grid. Returns aggregated per-site (max_tokens, n_layers)
    tensors, averaged across concepts × trials × layers × strengths."""
    n_layers = model_wrapper.n_layers

    # Cache concept vectors per injection layer (once).
    vec_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    for L in injection_layers:
        vec_cache[L] = load_concept_vectors(
            steering_dir, model_name, concepts, injection_layer=L, device=device)

    aggregated: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}

    total = len(concepts) * len(trial_numbers) * len(injection_layers) * len(strengths)
    with tqdm(total=total, desc="GradAttrSweep") as pbar:
        for concept in concepts:
            for trial_num in trial_numbers:
                prompt = build_introspection_prompt(model_wrapper, trial_num)
                for L in injection_layers:
                    steer_vec = vec_cache[L].get(concept)
                    if steer_vec is None:
                        pbar.update(len(strengths))
                        continue
                    for s in strengths:
                        try:
                            per_site = gradient_attribution_one_trial(
                                model_wrapper, prompt, concept, steer_vec,
                                injection_layer=L, strength=s,
                                injected=True, device=device,
                            )
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                            pbar.update(1)
                            continue

                        for site, tensor in per_site.items():
                            seq_len = tensor.shape[0]
                            if site not in aggregated:
                                aggregated[site] = np.zeros((seq_len, n_layers),
                                                            dtype=np.float64)
                                counts[site] = np.zeros((seq_len, n_layers),
                                                        dtype=np.int64)
                            cur = aggregated[site]
                            # Pad or crop to max stored length
                            cur_len = cur.shape[0]
                            if seq_len > cur_len:
                                pad = np.zeros((seq_len - cur_len, n_layers),
                                               dtype=np.float64)
                                aggregated[site] = np.concatenate([cur, pad], axis=0)
                                counts[site] = np.concatenate(
                                    [counts[site],
                                     np.zeros((seq_len - cur_len, n_layers),
                                              dtype=np.int64)], axis=0)
                                cur = aggregated[site]
                            cur[:seq_len] += tensor.numpy()
                            counts[site][:seq_len] += 1
                        pbar.update(1)

    # Average
    means: Dict[str, np.ndarray] = {}
    for site, s in aggregated.items():
        c = counts[site]
        means[site] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return means


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmap(matrix: np.ndarray, site: str, output_path: Path) -> None:
    """Token × layer heatmap of mean gradient-norm attribution."""
    fig, ax = plt.subplots(figsize=(10, max(3, matrix.shape[0] * 0.1)))
    im = ax.imshow(matrix.T, aspect="auto", cmap="viridis",
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(f"Gradient attribution — {site}")
    fig.colorbar(im, ax=ax, label="‖dL/dh‖₂ (mean over sweep)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {site} heatmap to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper Appendix T: Gradient Attribution over 400 Concepts."
    )
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("-d", "--device", type=str, default="cuda")
    p.add_argument("-dt", "--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--n-concepts", type=int, default=DEFAULT_N_CONCEPTS,
                   help="Number of training concepts (paper: 400).")
    p.add_argument("--trial-numbers", type=int, nargs="+", default=DEFAULT_TRIAL_NUMBERS)
    p.add_argument("--injection-layers", type=int, nargs="+", default=DEFAULT_INJECTION_LAYERS)
    p.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    p.add_argument("--steering-dir", type=Path, default=DEFAULT_STEERING_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--plots-only", action="store_true",
                   help="Skip the sweep; regenerate heatmaps from existing tensors.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tensors_path = output_dir / "grad_attr_tensors.pt"

    if not args.plots_only:
        mw = load_model(args.model, device=args.device,
                        dtype=_dtype_from_str(args.dtype))
        mw.model.eval()
        concepts = load_training_concepts(
            args.steering_dir, args.model,
            injection_layer=args.injection_layers[0],
            n_concepts=args.n_concepts,
        )
        print(f"Loaded {len(concepts)} concepts")
        print(f"Sweeping: trials={args.trial_numbers} layers={args.injection_layers} "
              f"strengths={args.strengths}")

        means = run_sweep(
            mw, concepts=concepts,
            trial_numbers=args.trial_numbers,
            injection_layers=args.injection_layers,
            strengths=args.strengths,
            steering_dir=args.steering_dir,
            model_name=args.model,
            output_dir=output_dir,
            device=args.device,
        )
        torch.save(means, tensors_path)
        with open(output_dir / "config.json", "w") as f:
            json.dump({
                "model": args.model,
                "n_concepts": len(concepts),
                "trial_numbers": args.trial_numbers,
                "injection_layers": args.injection_layers,
                "strengths": args.strengths,
                "n_total_trials": (len(concepts) * len(args.trial_numbers)
                                   * len(args.injection_layers) * len(args.strengths)),
                "sites": SITES,
            }, f, indent=2)
    else:
        if not tensors_path.exists():
            print(f"ERROR: --plots-only requires existing tensors at {tensors_path}")
            return 1
        means = torch.load(tensors_path, weights_only=False)

    # Produce the four heatmaps.
    for site in SITES:
        if site not in means:
            continue
        plot_heatmap(means[site], site,
                     plots_dir / f"grad_attr_heatmap_{site}.pdf")

    print(f"\nDone. Outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
