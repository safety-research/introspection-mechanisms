#!/usr/bin/env python3
"""
Experiment 62: SAE Steering Attribution & Attribution Graph

Performs SAE-level steering attribution (SA) analysis and builds multi-hop
attribution graphs tracing introspective detection through SAE/transcoder
features.

  Section A: Steering Attribution (SA) extraction
             -- Two-pass extraction per strength:
                Pass 1 (GA): forward + backward from loss → dL/dx at each SAE site
                Pass 2 (SG): JVP w.r.t. steering α → dx/dα at each SAE site (cacheable)
                Combine: SA(f) = GA(f) × SG(f) per active feature
             -- Integrates SA across strengths via Simpson's rule → ISA.
             -- SG is cached from hop-0 and reused for hop-1+ (halves GPU cost).

  Section B: Attribution Graph construction
             -- Computes GA-weighted edge weights: ew = ∫ GA_root(α) × SA(α) dα
             -- Per-type frac-of-max feature selection with per-hop caps [8,5,3,2]
             -- Only ATTN+TC features traced to next hop (MLP/RESID excluded)
             -- Backward tracing: feature-targeted SA (loss = target feature activation)
             -- Forward tracing: JVP from source decoder × GA_root → downstream SA
             -- Builds directed graph rendered as interactive Plotly HTML + PDF

  Auto-Config: Logit lens sweep + strength scan with LLM judge
             -- Auto-detects pos/neg token IDs and optimal injection strength

Paper sections supported:
  - Section 5.3 ("Gate and Evidence Carrier Features")
  - Section 5.4 ("Circuit Analysis")
  - Appendix: Steering Attribution Framework
  - Figure 16 (bread-layer37-single-col): Attribution graph visualization

Model: Primarily Gemma-3 27B with Gemma Scope 2 SAEs/Transcoders (262k, big)
Steering: Layer 37, strength 4.0 (configurable)

Usage:
    # Auto-configure pos/neg tokens and optimal strength (GPU)
    python 16_attribution_graph.py auto-config --concept Bread --layer 37

    # Extract SA at one strength (GPU)
    python 16_attribution_graph.py extract-sa --concept Bread --layer 37 --strength 4.0

    # Compute ISA by integrating SA across strengths (no GPU)
    python 16_attribution_graph.py compute-isa --concept Bread --layer 37

    # Build attribution graph with backward + forward tracing (requires SA data)
    python 16_attribution_graph.py build-graph --concept Bread --layer 37 --direction both

    # Full pipeline: extract SA, compute ISA, build graph
    python 16_attribution_graph.py all --concept Bread --layer 37 --direction both

    # Re-render existing graph from JSON
    python 16_attribution_graph.py visualize --concept Bread --layer 37
"""

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model_utils import ModelWrapper, load_model

warnings.filterwarnings("ignore", message="Glyph .* missing from font")


# =============================================================================
# Constants & defaults
# =============================================================================

DEFAULT_MODEL = "gemma3_27b"
DEFAULT_LAYER = 37
DEFAULT_STRENGTH = 4.0
DEFAULT_N_STRENGTHS = 21
DEFAULT_STRENGTH_MAX = 8.0
DEFAULT_TRACE_DEPTH = 2
DEFAULT_MAX_PER_TYPE = [8, 5, 3, 2]  # Per-hop: progressively narrower
DEFAULT_FRAC_OF_MAX = 0.10
DEFAULT_TOKEN_POS = -1
DEFAULT_EXP21_DIR = "analysis/exp21_more_concepts_steering"
DEFAULT_OUTPUT_DIR = "analysis/exp62_attribution_graph"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SEED = 42
DEFAULT_SAE_WIDTH = "262k"
DEFAULT_SAE_L0 = "big"
DEFAULT_DIRECTION = "both"  # backward, forward, or both

# SAE types to analyze
SAE_TYPES = ["resid_post_all", "mlp_out_all", "attn_out_all", "transcoder_all"]

# Only trace ATTN+TC features to next hop (MLP/RESID redundant with TC/ATTN)
TRACE_SAE_TYPES = {"attn_out_all", "transcoder_all"}


def _get_from_list(lst: List[int], hop: int) -> int:
    """Get value for a given hop from a per-hop list. Uses last value if hop exceeds length."""
    if hop < len(lst):
        return lst[hop]
    return lst[-1]

# Maps display SAE type to (encoder_input_suffix, grad_target_suffix)
# For transcoders: encode from pre_feedforward input, but GA target is post_feedforward output
SAE_TYPE_KEYS = {
    "resid_post_all": ("output", None),
    "mlp_out_all": ("post_feedforward_layernorm.output", None),
    "attn_out_all": ("attn.o_proj.input", None),
    "transcoder_all": ("pre_feedforward_layernorm.output", "post_feedforward_layernorm.output"),
}

# Maps display type to actual load type (transcoders load as mlp_out)
SAE_TYPE_LOAD_MAP = {
    "transcoder_all": "transcoder_all",
}

# Activation key suffixes for JVP tangent capture
JVP_SITE_SUFFIXES = [
    "attn.o_proj.input",
    "pre_feedforward_layernorm.output",
    "post_feedforward_layernorm.output",
    "output",
]

# Capture types for ActivationHooks
CAPTURE_TYPES_SAE = [
    "attn.o_proj.input",
    "pre_feedforward_layernorm.output",
    "post_feedforward_layernorm.output",
    "output",
]

# Concept-dependent first-decode token IDs for Gemma3-27B
# When injection is detected, model generates pos_token; otherwise neg_token
# Default: "Oh" (12932) vs "No" (3771)
CONCEPT_TOKEN_IDS: Dict[str, Dict[str, int]] = {
    "Bread": {"pos": 12932, "neg": 3771},
    "Algorithms": {"pos": 19058, "neg": 3771},
}
DEFAULT_TOKEN_IDS = {"pos": 12932, "neg": 3771}

# Default test concepts
DEFAULT_TEST_CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
]


# =============================================================================
# JumpReLU SAE / Transcoder Loading (from Gemma Scope 2)
# =============================================================================

class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder for Gemma Scope 2.

    Uses a discontinuous activation function where activations below a learned
    threshold are zeroed out, encouraging sparsity.
    """

    def __init__(self, d_in: int, d_sae: int, affine_skip_connection: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * F.relu(pre_acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon


# CPU cache for loaded SAEs
_sae_cpu_cache: Dict[str, JumpReLUSAE] = {}


def load_sae(
    layer_idx: int,
    width: str = "16k",
    l0: str = "small",
    sae_type: str = "resid_post_all",
    model_size: str = "27b",
    instruction_tuned: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> JumpReLUSAE:
    """Load a Gemma Scope 2 SAE/Transcoder from HuggingFace.

    Args:
        layer_idx: Layer index
        width: SAE width ("16k", "65k", "262k", "1m")
        l0: L0 target ("small", "medium", "big")
        sae_type: One of "resid_post_all", "mlp_out_all", "attn_out_all", "transcoder_all"
        model_size: Model size ("27b", "12b", etc.)
        instruction_tuned: Use IT (True) or PT (False) model SAEs
        device: Target device
        dtype: Target dtype

    Returns:
        Loaded JumpReLUSAE
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cache_key = f"{model_size}_{sae_type}_L{layer_idx}_{width}_{l0}_{'it' if instruction_tuned else 'pt'}"
    if cache_key in _sae_cpu_cache:
        sae = _sae_cpu_cache[cache_key]
        return sae.to(device=device, dtype=dtype)

    variant = "it" if instruction_tuned else "pt"
    repo_id = f"google/gemma-scope-2-{model_size}-{variant}"

    is_transcoder = sae_type == "transcoder_all"
    site = "mlp_out_all" if is_transcoder else sae_type
    affine_suffix = "_affine" if is_transcoder else ""
    filename = f"{site}/layer_{layer_idx}_width_{width}_l0_{l0}{affine_suffix}/params.safetensors"

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = load_file(path)

    d_in = params["w_enc"].shape[0]
    d_sae = params["w_enc"].shape[1]
    sae = JumpReLUSAE(d_in, d_sae, affine_skip_connection=is_transcoder)

    # Load parameters
    sae.w_enc.data = params["w_enc"]
    sae.b_enc.data = params["b_enc"]
    sae.threshold.data = params["threshold"]
    sae.w_dec.data = params["w_dec"]
    sae.b_dec.data = params["b_dec"]
    if is_transcoder and "affine_skip_connection" in params:
        sae.affine_skip_connection.data = params["affine_skip_connection"]

    _sae_cpu_cache[cache_key] = sae.cpu().float()
    return sae.to(device=device, dtype=dtype)


# =============================================================================
# Activation Hooks
# =============================================================================

def get_layers(model) -> torch.nn.ModuleList:
    """Navigate to the model's layer list."""
    mdl = model
    while not hasattr(mdl, "layers"):
        if hasattr(mdl, "model"):
            mdl = mdl.model
        elif hasattr(mdl, "language_model"):
            mdl = mdl.language_model
        else:
            raise ValueError("Cannot find layers attribute in model")
    return mdl.layers


class ActivationHooks:
    """Manage PyTorch hooks for capturing intermediate activations.

    Captures activations from attention, MLP, layer norm, and full layer outputs.
    Supports retain_grad for gradient-based attribution.
    """

    def __init__(self, model, retain_grad: bool = True):
        self.model = model
        self.activation_sequences = defaultdict(list)
        self.hooks = []
        self.enabled = False
        self.retain_grad = retain_grad

    def _create_hook(self, name: str, capture_input: bool = False, capture_output: bool = True):
        def hook(module, input, output):
            if self.enabled:
                if capture_input:
                    t = input[0] if isinstance(input, tuple) else input
                    if self.retain_grad and t.requires_grad:
                        t.retain_grad()
                    self.activation_sequences[f"{name}.input"].append(t)
                if capture_output:
                    t = output[0] if isinstance(output, tuple) else output
                    if self.retain_grad and t.requires_grad:
                        t.retain_grad()
                    self.activation_sequences[f"{name}.output"].append(t)
        return hook

    def register_hooks(self, layer_indices: List[int], capture_types: List[str]):
        """Register hooks on specified layers for given capture types."""
        layers = get_layers(self.model)
        for li in layer_indices:
            layer = layers[li]
            for ct in capture_types:
                parts = ct.split(".")
                module = layer
                for p in parts:
                    module = getattr(module, p, None)
                    if module is None:
                        break
                if module is not None:
                    name = f"layer_{li}.{ct}"
                    h = module.register_forward_hook(
                        self._create_hook(name, capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_activations(self) -> Dict[str, list]:
        return dict(self.activation_sequences)

    def clear(self):
        self.activation_sequences.clear()

    def __enter__(self):
        self.enabled = True
        return self

    def __exit__(self, *args):
        self.enabled = False


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FeatureNode:
    layer: int
    sae_type: str
    feature_id: int
    token_pos: int
    isa_value: float
    hop: int = 0
    label: Optional[str] = None

    @property
    def key(self) -> Tuple[int, str, int, int]:
        return (self.layer, self.sae_type, self.feature_id, self.token_pos)

    def short_name(self) -> str:
        abbrev = {"transcoder_all": "TC", "attn_out_all": "ATTN",
                  "mlp_out_all": "MLP", "resid_post_all": "RESID"}
        return f"L{self.layer} T{self.token_pos} {abbrev.get(self.sae_type, self.sae_type)} F{self.feature_id}"


@dataclass
class FeatureEdge:
    source_key: Tuple[int, str, int, int]
    target_key: Tuple[int, str, int, int]
    weight: float
    hop: int


@dataclass
class AttributionGraph:
    nodes: Dict[Tuple, FeatureNode]
    edges: List[FeatureEdge]
    root_key: Tuple = (-1, "root", -1, -1)
    optimal_strength: float = 0.0
    config: Optional[Dict] = None


# =============================================================================
# Prompt & Data Loading Utilities
# =============================================================================

_PREAMBLE = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
)


def build_messages(trial_num: int = 1) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": _PREAMBLE},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"},
    ]


def format_prompt(messages: List[Dict], tokenizer) -> Tuple[str, torch.Tensor, int]:
    """Format messages with chat template, return (prompt_str, input_ids, seq_len)."""
    filtered = [m for m in messages if not (m["role"] == "system" and m["content"] == "")]
    formatted = tokenizer.apply_chat_template(filtered, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return formatted, inputs["input_ids"], inputs["input_ids"].shape[1]


def find_steering_start(tokenizer, prompt: str, trial_num: int) -> int:
    """Find token position where steering should start (at 'Trial X:')."""
    marker = f"Trial {trial_num}"
    pos = prompt.find(marker)
    if pos == -1:
        return 0
    nl = prompt.rfind("\n", 0, pos)
    prefix = prompt[:nl] if nl != -1 else ""
    tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    return len(tokens["input_ids"][0])


def load_concept_vectors(
    exp21_dir: str, model_name: str, concepts: List[str], layer: int,
) -> Dict[str, torch.Tensor]:
    """Load concept vectors from exp21 results."""
    base = Path(exp21_dir) / model_name / "vectors"
    layer_dirs = sorted(base.glob("layer_*"))
    if layer_dirs:
        best = min(layer_dirs, key=lambda d: abs(int(d.name.replace("layer_", "")) - layer))
    else:
        best = base
    vectors = {}
    for concept in concepts:
        p = best / f"{concept}.pt"
        if p.exists():
            vectors[concept] = torch.load(p, weights_only=True)
    return vectors


def _act_key(layer_idx: int, suffix: str) -> str:
    return f"layer_{layer_idx}.{suffix}"


def _model_name_to_sae_size(model_name: str) -> str:
    if "27b" in model_name:
        return "27b"
    elif "12b" in model_name:
        return "12b"
    elif "4b" in model_name:
        return "4b"
    elif "1b" in model_name:
        return "1b"
    return "27b"


# =============================================================================
# Auto-Config: Logit Lens + Strength Scan → pos/neg tokens + optimal strength
# =============================================================================

def extract_logit_lens(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    strength: float,
    output_dir: Path,
    trial_num: int = 1,
    top_k: int = 20,
    device: str = "cuda",
) -> Optional[Path]:
    """Run one forward pass with steering, capture residual at last prefill token,
    project through final norm + unembedding to get top-k token probs at each layer.

    Saves a JSON with per-layer token probabilities and logits.
    """
    tokenizer = model_wrapper.tokenizer
    inner = model_wrapper.model

    # Get norm and unembedding
    mdl = inner
    if hasattr(mdl, "language_model"):
        norm_fn = mdl.language_model.norm
        unemb = mdl.lm_head.weight if hasattr(mdl, "lm_head") else None
    elif hasattr(mdl, "model") and hasattr(mdl.model, "norm"):
        norm_fn = mdl.model.norm
        unemb = mdl.lm_head.weight if hasattr(mdl, "lm_head") else None
    else:
        norm_fn = getattr(mdl, "norm", None)
        unemb = getattr(mdl, "lm_head", None)
        if unemb is not None:
            unemb = unemb.weight
    if norm_fn is None or unemb is None:
        print("  WARNING: Could not find norm/unemb in model, skipping logit lens")
        return None

    norm_fn = norm_fn.to(device).eval()
    vocab_size = min(len(tokenizer), unemb.shape[0])
    unemb = unemb[:vocab_size].float().to(device)

    # Build prompt and find steering start
    messages = build_messages(trial_num)
    prompt_str, input_ids, seq_len = format_prompt(messages, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    steering_start = find_steering_start(tokenizer, prompt_str, trial_num)

    n_layers = model_wrapper.n_layers
    layers_module = get_layers(inner)
    steering_vec = concept_vector.to(device).float()

    # Steering hook
    def make_hook(s):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            addition = torch.zeros_like(h)
            addition[:, steering_start:, :] = (steering_vec * s).unsqueeze(0)
            modified = h + addition
            return (modified,) + rest if isinstance(output, tuple) else modified
        return hook

    # Capture residual at each layer
    act_hooks = ActivationHooks(inner, retain_grad=False)
    act_hooks.register_hooks(list(range(n_layers)), ["output"])
    handle = layers_module[injection_layer].register_forward_hook(make_hook(strength))

    try:
        with act_hooks:
            inner.eval()
            with torch.no_grad():
                _ = inner(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
        act_hooks.remove_hooks()

    activations = act_hooks.get_activations()
    last_pos = seq_len - 1

    # Project each layer's residual through norm + unembed
    all_layers_tokens = {}
    for layer_idx in range(n_layers):
        key = f"layer_{layer_idx}.output"
        if key not in activations or not activations[key]:
            continue
        resid = activations[key][0]
        if isinstance(resid, (list, tuple)):
            resid = resid[0]
        resid = resid.detach().float()[0, last_pos, :].to(device)

        with torch.no_grad():
            normed = norm_fn(resid.unsqueeze(0))
            logits_l = (normed.float() @ unemb.T)[0]
            probs_l = F.softmax(logits_l, dim=-1)
            top_probs, top_ids = torch.topk(probs_l, k=top_k)
            top_logits = logits_l[top_ids]

        layer_list = []
        for i in range(top_ids.shape[0]):
            tid = top_ids[i].item()
            layer_list.append({
                "token_id": int(tid),
                "token_str": tokenizer.decode([tid]),
                "logit": float(top_logits[i].item()),
                "prob": float(top_probs[i].item()),
            })
        all_layers_tokens[str(layer_idx)] = layer_list

    out_data = {
        "strength": strength, "layer": injection_layer, "concept": concept,
        "trial_num": trial_num, "last_prefill_pos": int(last_pos), "top_k": top_k,
        "last_layer": n_layers - 1 if all_layers_tokens else None,
        "all_layers": all_layers_tokens,
    }

    strength_str = f"strength_{int(strength)}_{int((strength % 1) * 100):02d}"
    out_path = output_dir / strength_str / f"logit_lens_trial{trial_num}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    torch.cuda.empty_cache()
    return out_path


def detect_tokens_and_crossover(
    logit_lens_dir: Path, trial_num: int = 1,
) -> Tuple[int, int, Optional[float], Optional[float], Dict[str, Any]]:
    """Auto-detect pos/neg token IDs and crossover strength from logit lens sweep.

    Strategy:
    - neg_token: top-1 token at strength=0 (baseline)
    - pos_token: first different token to dominate (prob > 0.5)
    - crossover_strength: where pos first dominates
    - plateau_end: where pos prob drops below 0.3 after crossover

    Returns (pos_token_id, neg_token_id, crossover_strength, plateau_end, info).
    """
    # Load all logit lens JSONs
    by_strength: Dict[float, Dict] = {}
    for d in sorted(logit_lens_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if not m:
            continue
        s = float(f"{m.group(1)}.{m.group(2)}")
        ll_path = d / f"logit_lens_trial{trial_num}.json"
        if ll_path.exists():
            with open(ll_path) as f:
                by_strength[s] = json.load(f)

    if not by_strength:
        raise ValueError(f"No logit lens data found in {logit_lens_dir}")

    sorted_strengths = sorted(by_strength.keys())

    def get_last_layer_tokens(data):
        all_layers = data.get("all_layers", {})
        last = data.get("last_layer")
        return all_layers.get(str(last), []) if last is not None else []

    # neg_token: top-1 at lowest strength
    baseline_tokens = get_last_layer_tokens(by_strength[sorted_strengths[0]])
    if not baseline_tokens:
        raise ValueError("No tokens at baseline strength")
    neg_token_id = baseline_tokens[0]["token_id"]
    neg_token_str = baseline_tokens[0]["token_str"]

    # pos_token: first different token with prob > 0.5
    pos_token_id, pos_token_str, crossover_strength = None, None, None
    for s in sorted_strengths:
        tokens = get_last_layer_tokens(by_strength[s])
        if tokens and tokens[0]["token_id"] != neg_token_id and tokens[0].get("prob", 0) > 0.5:
            pos_token_id = tokens[0]["token_id"]
            pos_token_str = tokens[0]["token_str"]
            crossover_strength = s
            break

    # Fallback: token with highest max prob across non-baseline strengths
    if pos_token_id is None:
        token_max_probs: Dict[int, float] = {}
        token_strs: Dict[int, str] = {}
        for s in sorted_strengths[1:]:
            for t in get_last_layer_tokens(by_strength[s])[:5]:
                tid = t["token_id"]
                if tid != neg_token_id:
                    p = t.get("prob", 0)
                    if tid not in token_max_probs or p > token_max_probs[tid]:
                        token_max_probs[tid] = p
                        token_strs[tid] = t["token_str"]
        if token_max_probs:
            pos_token_id = max(token_max_probs, key=token_max_probs.get)
            pos_token_str = token_strs[pos_token_id]
        else:
            raise ValueError("Could not detect pos token from logit lens")

    # plateau_end: where pos prob drops below 0.3 after crossover
    plateau_end = sorted_strengths[-1]
    if crossover_strength is not None:
        past = False
        for s in sorted_strengths:
            if s >= crossover_strength:
                past = True
            if not past:
                continue
            pos_prob = 0.0
            for t in get_last_layer_tokens(by_strength[s])[:5]:
                if t["token_id"] == pos_token_id:
                    pos_prob = t.get("prob", 0)
                    break
            if pos_prob < 0.3:
                plateau_end = s
                break

    info = {
        "pos_token_id": pos_token_id, "pos_token_str": pos_token_str,
        "neg_token_id": neg_token_id, "neg_token_str": neg_token_str,
        "crossover_strength": crossover_strength, "plateau_end": plateau_end,
    }
    return pos_token_id, neg_token_id, crossover_strength, plateau_end, info


def run_strength_scan(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    strengths: List[float],
    output_dir: Path,
    n_trials: int = 3,
    device: str = "cuda",
) -> List[Dict]:
    """Run strength scan: generate responses at each strength, evaluate with LLM judge.

    Returns list of {strength, metrics} dicts sorted by strength.
    """
    from steering_utils import run_steered_introspection_test
    from eval_utils import LLMJudge, batch_evaluate, compute_detection_and_identification_metrics

    judge = LLMJudge()
    sweep = []

    for strength in tqdm(strengths, desc="Strength scan"):
        trial_results = []
        for trial_num in range(1, n_trials + 1):
            messages = build_messages(trial_num)
            prompt_str, input_ids, seq_len = format_prompt(messages, model_wrapper.tokenizer)
            steer_start = find_steering_start(model_wrapper.tokenizer, prompt_str, trial_num)

            # Steered generation
            hook_handle = get_layers(model_wrapper.model)[injection_layer].register_forward_hook(
                lambda m, i, o, s=strength: (
                    (lambda h, r: ((h + torch.zeros_like(h).index_copy_(1,
                        torch.arange(steer_start, h.shape[1], device=h.device),
                        (concept_vector.to(h.device).float() * s).unsqueeze(0).expand(-1, h.shape[1] - steer_start, -1)
                    )),) + r)(o[0], o[1:]) if isinstance(o, tuple) else o
                )
            )
            # Simpler: use the existing generate method
            hook_handle.remove()

            # Use model's generate with a steering hook
            sv = concept_vector.to(device).float() * strength

            def make_hook():
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    rest = output[1:] if isinstance(output, tuple) else ()
                    addition = torch.zeros_like(h)
                    addition[:, steer_start:, :] = sv.unsqueeze(0)
                    return (h + addition,) + rest if isinstance(output, tuple) else h + addition
                return hook

            handle = get_layers(model_wrapper.model)[injection_layer].register_forward_hook(make_hook())
            try:
                response = model_wrapper.generate(
                    prompt=prompt_str, max_new_tokens=100, temperature=0.0,
                )
            finally:
                handle.remove()

            trial_results.append({
                "concept": concept, "trial_type": "injection",
                "response": response, "trial_num": trial_num,
            })

        # Evaluate with LLM judge
        evaluations = batch_evaluate(judge, trial_results, [concept])
        metrics = compute_detection_and_identification_metrics(evaluations)
        sweep.append({"strength": strength, "metrics": metrics})

    sweep.sort(key=lambda r: r["strength"])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "strength_scan_results.json", "w") as f:
        json.dump({"sweep_results": sweep}, f, indent=2)

    return sweep


def select_optimal_strength_from_signals(
    sweep: List[Dict],
    crossover_strength: Optional[float],
    plateau_end: Optional[float],
    detection_threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """Select s* from the intersection of logit lens crossover and behavioral detection.

    Strategy:
    1. Filter to strengths >= crossover (token has flipped)
    2. Among those, find where detection_hit_rate >= threshold
    3. Pick ~70% through the good region
    """
    crossover = crossover_strength if crossover_strength is not None else 0.0
    p_end = plateau_end if plateau_end is not None else 8.0

    if not sweep:
        optimal = crossover + 0.7 * (p_end - crossover)
        return optimal, {"method": "logit_lens_only", "optimal_strength": optimal}

    good = [r["strength"] for r in sweep
            if r["strength"] >= crossover
            and r.get("metrics", {}).get("detection_hit_rate", 0) >= detection_threshold]

    if good:
        optimal = good[0] + 0.7 * (good[-1] - good[0])
        all_s = [r["strength"] for r in sweep]
        optimal = min(all_s, key=lambda s: abs(s - optimal))
        return optimal, {"method": "intersection", "optimal_strength": optimal,
                         "n_good": len(good), "range": [good[0], good[-1]]}

    optimal = crossover + 0.7 * (p_end - crossover)
    return optimal, {"method": "logit_lens_fallback", "optimal_strength": optimal}


def run_auto_config(args) -> None:
    """Auto-configure: detect pos/neg tokens and optimal strength for a concept."""
    concept = args.concept
    layer = args.layer
    run_name = f"{concept.replace(' ', '_')}_layer{layer}"
    base_out = Path(args.output_dir) / args.model / run_name
    ll_dir = base_out / "auto_config" / "logit_lens"
    scan_dir = base_out / "auto_config" / "strength_scan"
    ll_dir.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"AUTO CONFIG: {concept} layer {layer}")
    print("=" * 70)

    mw = load_model(args.model, device=args.device, dtype=args.dtype,
                     quantization=getattr(args, "quantization", None))
    vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
    if concept not in vectors:
        print(f"  ERROR: No vector for {concept}")
        return
    vec = vectors[concept]

    # Step 1: Logit lens sweep
    n_ll = getattr(args, "n_ll_strengths", 51)
    ll_strengths = np.linspace(0, getattr(args, "strength_max", DEFAULT_STRENGTH_MAX), n_ll).tolist()
    print(f"\n  Step 1: Logit lens sweep ({len(ll_strengths)} strengths)...")
    for s in tqdm(ll_strengths, desc="Logit lens"):
        extract_logit_lens(mw, concept, vec, layer, s, ll_dir, device=args.device)

    pos_id, neg_id, crossover, plateau_end, token_info = detect_tokens_and_crossover(ll_dir)
    print(f"    neg_token (s=0): {token_info['neg_token_str']!r} (id={neg_id})")
    print(f"    pos_token: {token_info['pos_token_str']!r} (id={pos_id})")
    print(f"    crossover: {crossover}, plateau_end: {plateau_end}")

    # Step 2: Strength scan with LLM judge
    n_scan = getattr(args, "n_scan_strengths", 25)
    scan_strengths = np.linspace(0, getattr(args, "strength_max", DEFAULT_STRENGTH_MAX), n_scan).tolist()
    print(f"\n  Step 2: Strength scan ({len(scan_strengths)} strengths)...")
    try:
        sweep = run_strength_scan(mw, concept, vec, layer, scan_strengths, scan_dir,
                                  n_trials=getattr(args, "n_scan_trials", 3), device=args.device)
    except Exception as e:
        print(f"    Strength scan failed ({e}), using logit lens only")
        sweep = []

    # Step 3: Select s*
    optimal_s, sel_info = select_optimal_strength_from_signals(sweep, crossover, plateau_end)
    print(f"\n  s* = {optimal_s:.2f} ({sel_info['method']})")

    # Save config
    config = {
        "concept": concept, "layer": layer,
        "pos_token_id": pos_id, "neg_token_id": neg_id,
        "pos_token_str": token_info.get("pos_token_str"),
        "neg_token_str": token_info.get("neg_token_str"),
        "optimal_strength": optimal_s,
        "token_detection": token_info, "strength_selection": sel_info,
    }
    config_path = base_out / "auto_config" / "auto_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")


# =============================================================================
# Section A: SA Extraction
# Matches working repo structure: separate functions for each pass.
#   AdditiveLayerCut            — Steering cut strategy
#   preload_saes()              — SAE loading
#   compute_ga_pass()           — Pass 1: forward + backward → GA
#   compute_sg_pass()           — Pass 2: JVP w.r.t. steering α → SG
#   compute_forward_jvp_pass()  — Forward JVP from source decoder direction
#   save_tangent_data() / load_tangent_data() — SG caching (safetensors)
#   combine_ga_sg_to_sa()       — GA × SG per active feature + remainder
#   make_logit_loss_fn()        — Loss for hop-0
#   make_feature_target_loss_fn() — Loss for hop-1+
#   extract_sa_for_strength()   — Backward orchestrator
#   extract_forward_sa_for_strength() — Forward orchestrator
# =============================================================================


class AdditiveLayerCut:
    """Standard additive steering: z_l += α·v for all tokens after steering_start_pos."""

    def make_steering_hook(self, steering_vec: torch.Tensor, steering_start_pos: int):
        def hook_factory(steering_strength):
            def hook(module, input, output):
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                rest = output[1:] if is_tuple else ()
                addition = torch.zeros_like(h)
                addition[:, steering_start_pos:, :] = (steering_vec * steering_strength).unsqueeze(0)
                return (h + addition,) + rest if is_tuple else h + addition
            return hook
        return hook_factory

    def has_grad_path(self, layer_idx: int, sae_type: str, injection_layer: int) -> bool:
        return (layer_idx > injection_layer) or (layer_idx == injection_layer and sae_type == "resid_post_all")


def preload_saes(
    n_layers: int, model_name: str, sae_width: str, sae_l0: str,
    device: str = "cpu", layer_indices: Optional[set] = None,
) -> Dict[Tuple[int, str], JumpReLUSAE]:
    """Preload SAEs to the given device. Returns dict of (layer_idx, sae_type) -> sae."""
    model_size = _model_name_to_sae_size(model_name)
    unique_types = set(SAE_TYPE_LOAD_MAP.get(st, st) for st in SAE_TYPES)
    sae_cache: Dict[Tuple[int, str], JumpReLUSAE] = {}
    layers_to_load = sorted(layer_indices) if layer_indices is not None else range(n_layers)
    for li in tqdm(layers_to_load, desc="Preloading SAEs", leave=False):
        for st in unique_types:
            try:
                sae = load_sae(li, sae_width, sae_l0, st, model_size, True, device)
                sae_cache[(li, st)] = sae.eval()
            except Exception:
                pass
    return sae_cache


# ── GA computation (Pass 1: forward + backward) ─────────────────────────────

def compute_ga_pass(
    model_inner, input_ids, attention_mask, injection_layer, strength,
    steering_hook_factory, n_layers, loss_fn, device="cuda", layer_indices=None,
):
    """Pass 1: Forward + backward to compute Gradient Attribution (GA).

    Args:
        loss_fn: Callable(outputs, raw_activations) -> loss tensor.
        layer_indices: If set, only register hooks at these layers.
    Returns (grad_data, act_data) dicts keyed by (layer_idx, sae_type).
    """
    layers_module = get_layers(model_inner)
    hook_layers = sorted(layer_indices) if layer_indices is not None else list(range(n_layers))

    steer_handle = layers_module[injection_layer].register_forward_hook(steering_hook_factory(strength))
    act_hooks = ActivationHooks(model_inner, retain_grad=True)
    act_hooks.register_hooks(layer_indices=hook_layers, capture_types=list(CAPTURE_TYPES_SAE))

    try:
        with act_hooks:
            model_inner.eval()
            outputs = model_inner(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=False, return_dict=True)
    finally:
        steer_handle.remove()
        act_hooks.remove_hooks()

    raw_activations = act_hooks.get_activations()
    loss = loss_fn(outputs, raw_activations)
    loss.sum().backward()

    grad_data: Dict[Tuple[int, str], torch.Tensor] = {}
    act_data: Dict[Tuple[int, str], torch.Tensor] = {}
    for li in hook_layers:
        for st in SAE_TYPES:
            in_suffix, tgt_suffix = SAE_TYPE_KEYS[st]
            grad_suffix = tgt_suffix if (st == "transcoder_all" and tgt_suffix) else in_suffix
            grad_key = _act_key(li, grad_suffix)
            if grad_key not in raw_activations or not raw_activations[grad_key]:
                continue
            t = raw_activations[grad_key][0]
            if isinstance(t, (list, tuple)):
                t = t[0]
            if t.grad is None:
                continue
            grad_data[(li, st)] = t.grad.detach().float()[0].cpu()

            enc_key = _act_key(li, in_suffix)
            if enc_key not in raw_activations or not raw_activations[enc_key]:
                continue
            enc_t = raw_activations[enc_key][0]
            if isinstance(enc_t, (list, tuple)):
                enc_t = enc_t[0]
            act_data[(li, st)] = enc_t[0].detach().float().cpu()

    del raw_activations, outputs
    torch.cuda.empty_cache()
    return grad_data, act_data


# ── SG computation (Pass 2: JVP forward) ────────────────────────────────────

def compute_sg_pass(
    model_inner, input_ids, attention_mask, injection_layer, strength,
    steering_hook_factory, n_layers, device="cuda",
):
    """Pass 2: JVP forward to compute Steering Gradient (SG = dx/dalpha).

    Always captures all layers since SG is cached and reused across hops.
    Returns tangent_data dict keyed by (layer_idx, suffix).
    """
    layers_module = get_layers(model_inner)
    alpha_primal = torch.tensor(strength, dtype=torch.float32, device=device)
    alpha_tangent = torch.tensor(1.0, dtype=torch.float32, device=device)

    def jvp_forward(alpha_val):
        handle = layers_module[injection_layer].register_forward_hook(steering_hook_factory(alpha_val))
        act_hooks = ActivationHooks(model_inner, retain_grad=False)
        act_hooks.register_hooks(list(range(n_layers)), list(CAPTURE_TYPES_SAE))
        try:
            with act_hooks:
                model_inner.eval()
                _ = model_inner(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
            act_hooks.remove_hooks()
        raw = act_hooks.get_activations()
        result = []
        for li in range(n_layers):
            for suffix in JVP_SITE_SUFFIXES:
                t = raw.get(_act_key(li, suffix), [None])[0]
                if t is None:
                    t = torch.zeros(1)
                if isinstance(t, (list, tuple)):
                    t = t[0]
                result.append(t[0] if t.dim() == 3 else t)
        return result

    _, site_tangents = torch.func.jvp(jvp_forward, (alpha_primal,), (alpha_tangent,))

    tangent_data: Dict[Tuple[int, str], torch.Tensor] = {}
    for li in range(n_layers):
        for j, suffix in enumerate(JVP_SITE_SUFFIXES):
            tangent_data[(li, suffix)] = site_tangents[li * len(JVP_SITE_SUFFIXES) + j].detach()
    del site_tangents
    torch.cuda.empty_cache()
    return tangent_data


# ── Forward JVP pass (for forward tracing) ──────────────────────────────────

def compute_forward_jvp_pass(
    model_inner, input_ids, attention_mask, injection_layer, strength,
    steering_hook_factory, source_layer, source_decoder_vec, source_sae_type,
    n_layers, device="cuda",
):
    """Forward JVP: propagate tangent from source feature's decoder direction.

    Symmetric to compute_sg_pass. Steering is applied at fixed strength (not differentiated).
    Returns tangent_data for layers >= source_layer.
    """
    layers_module = get_layers(model_inner)
    t_primal = torch.tensor(0.0, dtype=torch.float32, device=device)
    t_tangent = torch.tensor(1.0, dtype=torch.float32, device=device)
    source_vec = source_decoder_vec.to(device).float()

    # Fixed steering hook (not part of JVP)
    steer_handle = layers_module[injection_layer].register_forward_hook(
        steering_hook_factory(torch.tensor(strength, dtype=torch.float32, device=device)))

    # Determine perturbation hook site
    use_pre_hook = False
    if source_sae_type == "attn_out_all":
        src_module = layers_module[source_layer].self_attn.o_proj
        use_pre_hook = True
    elif source_sae_type in ("transcoder_all", "mlp_out_all"):
        src_module = getattr(layers_module[source_layer], "post_feedforward_layernorm",
                             layers_module[source_layer])
    else:
        src_module = layers_module[source_layer]

    def jvp_forward(t_val):
        perturbation = source_vec * t_val
        if use_pre_hook:
            def src_hook(module, args):
                x = args[0]
                p = perturbation.unsqueeze(0).unsqueeze(0).expand_as(x)
                return (x + p,) + args[1:]
            src_h = src_module.register_forward_pre_hook(src_hook)
        else:
            def src_hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                rest = output[1:] if isinstance(output, tuple) else ()
                p = perturbation.unsqueeze(0).unsqueeze(0).expand_as(h)
                return (h + p,) + rest if isinstance(output, tuple) else h + p
            src_h = src_module.register_forward_hook(src_hook)

        act_hooks = ActivationHooks(model_inner, retain_grad=False)
        act_hooks.register_hooks(list(range(n_layers)), list(CAPTURE_TYPES_SAE))
        try:
            with act_hooks:
                model_inner.eval()
                _ = model_inner(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True)
        finally:
            src_h.remove()
            act_hooks.remove_hooks()

        raw = act_hooks.get_activations()
        result = []
        for li in range(n_layers):
            for suffix in JVP_SITE_SUFFIXES:
                t = raw.get(_act_key(li, suffix), [None])[0]
                if t is None:
                    t = torch.zeros(1, device=device)
                if isinstance(t, (list, tuple)):
                    t = t[0]
                result.append(t[0] if t.dim() == 3 else t)
        return result

    try:
        _, site_tangents = torch.func.jvp(jvp_forward, (t_primal,), (t_tangent,))
    finally:
        steer_handle.remove()

    tangent_data: Dict[Tuple[int, str], torch.Tensor] = {}
    for li in range(source_layer, n_layers):
        for j, suffix in enumerate(JVP_SITE_SUFFIXES):
            tangent_data[(li, suffix)] = site_tangents[li * len(JVP_SITE_SUFFIXES) + j].detach()
    del site_tangents
    torch.cuda.empty_cache()
    return tangent_data


# ── SG caching ──────────────────────────────────────────────────────────────

def save_tangent_data(tangent_data: Dict[Tuple[int, str], torch.Tensor], path: Path) -> None:
    """Save SG tangent data to disk as safetensors."""
    from safetensors.torch import save_file
    flat = {f"{li}|{suffix}": t.cpu().contiguous() for (li, suffix), t in tangent_data.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(path))


def load_tangent_data(path: Path) -> Dict[Tuple[int, str], torch.Tensor]:
    """Load SG tangent data from disk."""
    from safetensors.torch import load_file
    flat = load_file(str(path))
    return {(int(k.split("|")[0]), k.split("|")[1]): t for k, t in flat.items()}


# ── SA combine: GA × SG per SAE feature ─────────────────────────────────────

def combine_ga_sg_to_sa(
    grad_data, act_data, tangent_data, sae_cache, cut_strategy,
    injection_layer, n_layers, seq_len, base_meta, device="cuda",
    compute_remainder=True,
):
    """Combine GA and SG to compute SA = GA × SG for all active SAE features.

    Also computes per-token remainder SA when compute_remainder=True.
    Returns list of row dicts.
    """
    feature_rows: List[Dict[str, Any]] = []
    loaded_layers = sorted(set(li for li, _ in sae_cache))

    with torch.no_grad():
        for li in tqdm(loaded_layers, desc="SA combine", leave=False):
            for st in SAE_TYPES:
                in_suffix, tgt_suffix = SAE_TYPE_KEYS[st]
                load_type = SAE_TYPE_LOAD_MAP.get(st, st)
                if (li, load_type) not in sae_cache or (li, st) not in grad_data:
                    continue

                grad_tensor = grad_data[(li, st)].to(device)
                x_all = act_data[(li, st)].to(device)
                sae = sae_cache[(li, load_type)].to(device=device, dtype=torch.float32).eval()
                w_enc, w_dec = sae.w_enc, sae.w_dec

                sae_acts = sae.encode(x_all)
                tok_idx, feat_idx = (sae_acts > 0).nonzero(as_tuple=True)
                n_active = len(tok_idx)
                active_acts = sae_acts[tok_idx, feat_idx] if n_active > 0 else torch.tensor([])

                ga_vals = torch.zeros(n_active, device=device)
                sg_vals = torch.zeros(n_active, device=device)
                if n_active > 0:
                    ga_vals = (grad_tensor[tok_idx] * w_dec[feat_idx]).sum(dim=-1)

                has_gp = cut_strategy.has_grad_path(li, st, injection_layer)
                if has_gp and n_active > 0:
                    tkey = (li, in_suffix)
                    if tkey in tangent_data:
                        dx = tangent_data[tkey].to(device).float()
                        if dx.dim() == 3:
                            dx = dx[0]
                        sg_vals = (dx[tok_idx] * w_enc[:, feat_idx].T).sum(dim=-1)

                sa_vals = ga_vals * sg_vals

                if n_active > 0:
                    tl, fl = tok_idx.cpu().tolist(), feat_idx.cpu().tolist()
                    al, gl = active_acts.cpu().tolist(), ga_vals.cpu().tolist()
                    sl, sal = sg_vals.cpu().tolist(), sa_vals.cpu().tolist()
                    for i in range(n_active):
                        feature_rows.append({
                            **base_meta, "layer": li, "token_pos": tl[i],
                            "sae_type": st, "feature_id": fl[i],
                            "activation": al[i], "gradient_attribution": gl[i],
                            "steering_grad": sl[i], "steering_attribution": sal[i],
                        })

                if compute_remainder:
                    total_sfx = tgt_suffix if tgt_suffix else in_suffix
                    total_key = (li, total_sfx)
                    per_token_total = torch.zeros(seq_len, device=device)
                    if has_gp and total_key in tangent_data:
                        dx = tangent_data[total_key].to(device).float()
                        if dx.dim() == 3:
                            dx = dx[0]
                        per_token_total = (grad_tensor * dx).sum(dim=-1)

                    feat_sa_sum = torch.zeros(seq_len, device=device)
                    if n_active > 0:
                        feat_sa_sum.scatter_add_(0, tok_idx, sa_vals)
                    rem_sa = per_token_total - feat_sa_sum

                    if tgt_suffix is not None:
                        recon = sae(x_all)
                        tgt = act_data.get((li, "mlp_out_all"), x_all)
                        if isinstance(tgt, torch.Tensor) and tgt.device.type == "cpu":
                            tgt = tgt.to(device)
                        remainder = tgt - recon
                    else:
                        remainder = x_all - sae.decode(sae_acts)

                    rem_norm = remainder.norm(dim=-1)
                    safe_norm = rem_norm.clamp(min=1e-10).unsqueeze(-1)
                    rem_ga = (grad_tensor * (remainder / safe_norm)).sum(dim=-1)

                    rem_sg = torch.zeros(seq_len, device=device)
                    if has_gp and total_key in tangent_data:
                        dx_r = tangent_data[total_key].to(device).float()
                        if dx_r.dim() == 3:
                            dx_r = dx_r[0]
                        drecon = torch.zeros_like(dx_r)
                        if n_active > 0:
                            drecon.index_add_(0, tok_idx, sg_vals.unsqueeze(-1) * w_dec[feat_idx])
                        rem_sg = (dx_r - drecon).norm(dim=-1)

                    rsa, rn = rem_sa.cpu().tolist(), rem_norm.cpu().tolist()
                    rga, rsg = rem_ga.cpu().tolist(), rem_sg.cpu().tolist()
                    for t in range(seq_len):
                        feature_rows.append({
                            **base_meta, "layer": li, "token_pos": t,
                            "sae_type": st, "feature_id": -1,
                            "activation": rn[t], "gradient_attribution": rga[t],
                            "steering_grad": rsg[t], "steering_attribution": rsa[t],
                        })

                sae.to("cpu")
                torch.cuda.empty_cache()

    return feature_rows


# ── Loss functions ───────────────────────────────────────────────────────────

def make_logit_loss_fn(pos_token_id: int, neg_token_id: int):
    """Standard SA loss: logit(pos) - logit(neg) at the last token."""
    def loss_fn(outputs, raw_activations):
        return outputs.logits[:, -1, pos_token_id] - outputs.logits[:, -1, neg_token_id]
    return loss_fn


def make_feature_target_loss_fn(target_layer, target_sae_type, target_feature_id, target_token_pos,
                                 sae_width, sae_l0, model_name, device, seq_len):
    """Feature-targeted loss: activation of a specific SAE feature."""
    tgt_load = SAE_TYPE_LOAD_MAP.get(target_sae_type, target_sae_type)
    model_size = _model_name_to_sae_size(model_name)
    target_sae = load_sae(target_layer, sae_width, sae_l0, tgt_load, model_size, True, device)
    tgt_in_suffix = SAE_TYPE_KEYS[target_sae_type][0]

    def loss_fn(outputs, raw_activations):
        tgt_key = _act_key(target_layer, tgt_in_suffix)
        if tgt_key not in raw_activations or not raw_activations[tgt_key]:
            return torch.tensor(0.0, device=device, requires_grad=True)
        tgt_act = raw_activations[tgt_key][0]
        if isinstance(tgt_act, (list, tuple)):
            tgt_act = tgt_act[0]
        tgt_2d = tgt_act[0] if tgt_act.dim() == 3 else tgt_act
        encoded = target_sae.encode(tgt_2d.float().to(device))
        tp = target_token_pos if target_token_pos >= 0 else seq_len + target_token_pos
        return encoded[tp, target_feature_id]
    return loss_fn


# ── Backward SA orchestrator ────────────────────────────────────────────────

def extract_sa_for_strength(
    model_wrapper: ModelWrapper, concept: str, concept_vector: torch.Tensor,
    injection_layer: int, strength: float, output_dir: Path,
    trial_num: int = 1, pos_token_id: int = 12932, neg_token_id: int = 3771,
    sae_width: str = DEFAULT_SAE_WIDTH, sae_l0: str = DEFAULT_SAE_L0, device: str = "cuda",
    loss_fn_maker=None, extra_meta: Optional[Dict] = None,
    output_filename: str = "sa_trial{trial_num}.parquet",
    sg_cache_dir: Optional[Path] = None, save_sg_cache: bool = False,
    layer_indices: Optional[set] = None, compute_remainder: bool = True,
) -> Optional[Path]:
    """Unified backward SA extraction: GA (Pass 1) + SG (Pass 2) + combine.

    Args:
        loss_fn_maker: Callable(outputs, raw_activations) -> loss.
            If None, uses make_logit_loss_fn(pos_token_id, neg_token_id).
        sg_cache_dir: If set, look for / save SG tangent data here.
        save_sg_cache: If True and sg_cache_dir is set, save SG after computing.
    """
    import pandas as pd

    tokenizer = model_wrapper.tokenizer
    inner = model_wrapper.model
    messages = build_messages(trial_num)
    prompt_str, input_ids, seq_len = format_prompt(messages, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    steering_start = find_steering_start(tokenizer, prompt_str, trial_num)
    n_layers = model_wrapper.n_layers

    cut = AdditiveLayerCut()
    steering_vec = concept_vector.to(device).float()
    hook_factory = cut.make_steering_hook(steering_vec, steering_start)

    sae_cache = preload_saes(n_layers, model_wrapper.model_name, sae_width, sae_l0, "cpu", layer_indices)

    if loss_fn_maker is None:
        loss_fn_maker = make_logit_loss_fn(pos_token_id, neg_token_id)

    base_meta = {"concept": concept, "injection_layer": injection_layer,
                 "injection_strength": strength, "trial_num": trial_num}
    if extra_meta:
        base_meta.update(extra_meta)

    hook_layers = sorted(layer_indices) if layer_indices else None

    # Pass 1: GA
    grad_data, act_data = compute_ga_pass(
        inner, input_ids, attention_mask, injection_layer, strength,
        hook_factory, n_layers, loss_fn_maker, device, hook_layers)

    # Pass 2: SG (with caching)
    strength_str = f"strength_{int(strength)}_{int((strength % 1) * 100):02d}"
    sg_file = sg_cache_dir / strength_str / "tangent_data.safetensors" if sg_cache_dir else None

    if sg_file and sg_file.exists():
        tangent_data = load_tangent_data(sg_file)
        print(f"  SG loaded from cache")
    else:
        tangent_data = compute_sg_pass(
            inner, input_ids, attention_mask, injection_layer, strength,
            hook_factory, n_layers, device)
        if save_sg_cache and sg_file:
            save_tangent_data(tangent_data, sg_file)
            print(f"  SG cached: {sg_file}")

    # Combine
    rows = combine_ga_sg_to_sa(
        grad_data, act_data, tangent_data, sae_cache, cut,
        injection_layer, n_layers, seq_len, base_meta, device, compute_remainder)

    # Save
    out_path = output_dir / strength_str / output_filename.format(trial_num=trial_num)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"  Saved {len(rows)} rows to {out_path}")
    torch.cuda.empty_cache()
    return out_path


# Convenience wrappers

def extract_steering_attribution(
    model_wrapper, concept, concept_vector, injection_layer, strength,
    output_dir, trial_num=1, pos_token_id=12932, neg_token_id=3771,
    sae_width=DEFAULT_SAE_WIDTH, sae_l0=DEFAULT_SAE_L0, device="cuda",
    sg_cache_dir=None,
):
    """Extract root SA (logit-gap loss). Saves SG cache if sg_cache_dir is set."""
    return extract_sa_for_strength(
        model_wrapper, concept, concept_vector, injection_layer, strength,
        output_dir, trial_num, pos_token_id, neg_token_id, sae_width, sae_l0, device,
        sg_cache_dir=sg_cache_dir, save_sg_cache=True)


def extract_feature_target_sa(
    model_wrapper, concept, concept_vector, injection_layer, strength,
    target_layer, target_sae_type, target_feature_id, target_token_pos,
    output_dir, trial_num=1, sae_width=DEFAULT_SAE_WIDTH, sae_l0=DEFAULT_SAE_L0,
    device="cuda", sg_cache_dir=None,
):
    """Extract feature-targeted SA. Loads SG from cache if available."""
    seq_len = format_prompt(build_messages(trial_num), model_wrapper.tokenizer)[2]
    loss_fn = make_feature_target_loss_fn(
        target_layer, target_sae_type, target_feature_id, target_token_pos,
        sae_width, sae_l0, model_wrapper.model_name, device, seq_len)
    extra_meta = {"target_layer": target_layer, "target_sae_type": target_sae_type,
                  "target_feature_id": target_feature_id, "target_token_pos": target_token_pos}
    tgt_subdir = f"{target_sae_type}_L{target_layer}_F{target_feature_id}_T{target_token_pos}"
    return extract_sa_for_strength(
        model_wrapper, concept, concept_vector, injection_layer, strength,
        output_dir / "feat_sa" / tgt_subdir, trial_num,
        sae_width=sae_width, sae_l0=sae_l0, device=device,
        loss_fn_maker=loss_fn, extra_meta=extra_meta,
        output_filename="feat_sa_trial{trial_num}.parquet",
        sg_cache_dir=sg_cache_dir, save_sg_cache=False, compute_remainder=False)


# ── Forward SA orchestrator ─────────────────────────────────────────────────

def extract_forward_sa_for_strength(
    model_wrapper: ModelWrapper, concept: str, concept_vector: torch.Tensor,
    injection_layer: int, strength: float,
    source_layer: int, source_sae_type: str, source_feature_id: int, source_token_pos: int,
    output_dir: Path, root_sa_dir: Optional[Path] = None,
    trial_num: int = 1, sae_width: str = DEFAULT_SAE_WIDTH, sae_l0: str = DEFAULT_SAE_L0,
    device: str = "cuda",
) -> Optional[Path]:
    """Forward SA: JVP from source decoder direction × GA_root.

    steering_attribution = forward_jvp × GA_root per downstream feature.
    """
    import pandas as pd

    tokenizer = model_wrapper.tokenizer
    inner = model_wrapper.model
    messages = build_messages(trial_num)
    prompt_str, input_ids, seq_len = format_prompt(messages, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    steering_start = find_steering_start(tokenizer, prompt_str, trial_num)
    n_layers = model_wrapper.n_layers

    cut = AdditiveLayerCut()
    steering_vec = concept_vector.to(device).float()
    hook_factory = cut.make_steering_hook(steering_vec, steering_start)

    # Load source w_dec
    src_load_type = SAE_TYPE_LOAD_MAP.get(source_sae_type, source_sae_type)
    model_size = _model_name_to_sae_size(model_wrapper.model_name)
    src_sae = load_sae(source_layer, sae_width, sae_l0, src_load_type, model_size, True, device)
    source_dec_vec = src_sae.w_dec[source_feature_id].detach().clone()
    del src_sae
    torch.cuda.empty_cache()

    sae_cache = preload_saes(n_layers, model_wrapper.model_name, sae_width, sae_l0, "cpu",
                             layer_indices=set(range(source_layer, n_layers)))

    # Forward JVP pass
    fwd_tangent_data = compute_forward_jvp_pass(
        inner, input_ids, attention_mask, injection_layer, strength,
        hook_factory, source_layer, source_dec_vec, source_sae_type, n_layers, device)

    # Load GA_root from root SA parquet at this strength
    root_df = None
    if root_sa_dir is not None:
        s_str = f"strength_{int(strength)}_{int((strength % 1) * 100):02d}"
        for t in [trial_num, 1, 2]:
            candidate = root_sa_dir / s_str / f"sa_trial{t}.parquet"
            if candidate.exists():
                root_df = pd.read_parquet(candidate, columns=[
                    "layer", "sae_type", "feature_id", "token_pos", "gradient_attribution"])
                break

    # Combine: forward_jvp × GA_root
    base_meta = {"concept": concept, "injection_layer": injection_layer,
                 "injection_strength": strength, "trial_num": trial_num,
                 "source_layer": source_layer, "source_sae_type": source_sae_type,
                 "source_feature_id": source_feature_id, "source_token_pos": source_token_pos}

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for li in sorted(set(l for l, _ in sae_cache if l > source_layer)):
            for st in SAE_TYPES:
                in_sfx = SAE_TYPE_KEYS[st][0]
                lt = SAE_TYPE_LOAD_MAP.get(st, st)
                if (li, lt) not in sae_cache or (li, in_sfx) not in fwd_tangent_data:
                    continue

                sae = sae_cache[(li, lt)].to(device=device, dtype=torch.float32).eval()
                dx = fwd_tangent_data[(li, in_sfx)].to(device).float()
                if dx.dim() == 3:
                    dx = dx[0]
                fwd_jvp = dx @ sae.w_enc

                ga_root: Dict[Tuple[int, int], float] = {}
                if root_df is not None:
                    lr = root_df[(root_df["layer"] == li) & (root_df["sae_type"] == st)]
                    for _, row in lr.iterrows():
                        ga_root[(int(row["token_pos"]), int(row["feature_id"]))] = float(row["gradient_attribution"])

                nonzero = (fwd_jvp.abs() > 1e-8).nonzero(as_tuple=True)
                if len(nonzero[0]) > 0:
                    toks = nonzero[0].cpu().tolist()
                    feats = nonzero[1].cpu().tolist()
                    jvp_vals = fwd_jvp[nonzero[0], nonzero[1]].cpu().tolist()
                    for tok, fid, jvp_val in zip(toks, feats, jvp_vals):
                        ga_val = ga_root.get((tok, fid), 0.0)
                        if abs(ga_val) < 1e-10:
                            continue
                        rows.append({
                            **base_meta, "layer": li, "token_pos": tok,
                            "sae_type": st, "feature_id": fid,
                            "forward_jvp": jvp_val, "gradient_attribution": ga_val,
                            "steering_attribution": jvp_val * ga_val,
                        })
                sae.to("cpu")
                torch.cuda.empty_cache()

    del fwd_tangent_data
    torch.cuda.empty_cache()

    src_subdir = f"fwd_{source_sae_type}_L{source_layer}_F{source_feature_id}_T{source_token_pos}"
    s_str = f"strength_{int(strength)}_{int((strength % 1) * 100):02d}"
    out_path = output_dir / "feat_sa" / src_subdir / s_str / f"feat_sa_trial{trial_num}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path

# =============================================================================
# Section A: ISA Computation
# =============================================================================

def compute_isa(sa_dir: Path, trial_nums: List[int] = None) -> None:
    """Compute Integrated Steering Attribution via trapezoidal integration.

    For each feature, integrates SA across injection strengths:
        ISA(f) = integral_0^s* SA(f, alpha) d_alpha
    """
    import pandas as pd

    if trial_nums is None:
        trial_nums = [1]

    strength_dirs = sorted(sa_dir.glob("strength_*"))
    if len(strength_dirs) < 2:
        print(f"  Need >= 2 strength dirs for ISA, found {len(strength_dirs)}")
        return

    # Parse strengths
    strengths = []
    for d in strength_dirs:
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if m:
            strengths.append((float(f"{m.group(1)}.{m.group(2)}"), d))
    strengths.sort(key=lambda x: x[0])

    for trial in trial_nums:
        # Load SA at each strength
        feat_key_to_sa: Dict[Tuple, List[Tuple[float, float]]] = defaultdict(list)

        for s_val, s_dir in strengths:
            parquet = s_dir / f"sa_trial{trial}.parquet"
            if not parquet.exists():
                continue
            df = pd.read_parquet(parquet)
            for _, row in df.iterrows():
                key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
                feat_key_to_sa[key].append((s_val, float(row["steering_attribution"])))

        # Integrate each feature
        isa_records = []
        for key, sa_curve in feat_key_to_sa.items():
            sa_curve.sort(key=lambda x: x[0])
            if len(sa_curve) < 2:
                continue
            ss = np.array([p[0] for p in sa_curve])
            sa = np.array([p[1] for p in sa_curve])
            isa_val = float(np.trapezoid(sa, ss))
            isa_records.append({
                "layer": key[0], "sae_type": key[1],
                "feature_id": key[2], "token_pos": key[3],
                "integrated_steering_attribution": isa_val,
                "trial_num": trial,
            })

        if isa_records:
            out = sa_dir / f"isa_trial{trial}.parquet"
            pd.DataFrame(isa_records).to_parquet(out, index=False)
            print(f"  ISA computed for trial {trial}: {len(isa_records)} features -> {out}")


# =============================================================================
# Section B: Edge Weight Computation (GA-weighted integration)
# =============================================================================

SAE_ABBREV = {"transcoder_all": "TC", "attn_out_all": "ATTN",
              "mlp_out_all": "MLP", "resid_post_all": "RESID"}


def select_top_per_type(
    df, value_col: str, sae_type_col: str = "sae_type",
    max_per_type: int = 8, frac_of_max: float = 0.10,
):
    """Per SAE type: select features with value > frac_of_max * type_max, capped at max_per_type."""
    import pandas as pd

    pos = df[df[value_col] > 0]
    parts = []
    for st in sorted(pos[sae_type_col].unique()):
        st_df = pos[pos[sae_type_col] == st].sort_values(value_col, ascending=False)
        if st_df.empty:
            continue
        max_val = st_df[value_col].iloc[0]
        above = st_df[st_df[value_col] > frac_of_max * max_val]
        selected = above.head(max_per_type)
        parts.append(selected)
        print(f"    {SAE_ABBREV.get(st, st)}: {len(selected)} features "
              f"(>{frac_of_max:.0%} of max {max_val:.4f}, {len(above)} above, cap {max_per_type})")
    if not parts:
        return pos.iloc[:0]
    return pd.concat(parts).sort_values(value_col, ascending=False)


def _load_curve_from_root_sa(
    sa_dir: Path, trial_nums: List[int], target: FeatureNode, column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a feature's GA or SG curve across strengths from root SA parquets.

    Args:
        column: "gradient_attribution" for GA curve, "steering_grad" for SG curve.

    Returns (strengths_array, values_array) sorted by strength. Empty if not found.
    """
    import pandas as pd

    pairs = []
    for d in sorted(sa_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if not m:
            continue
        s = float(f"{m.group(1)}.{m.group(2)}")
        for t in trial_nums:
            f = d / f"sa_trial{t}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=["layer", "sae_type", "feature_id", "token_pos", column])
            match = df[(df["layer"] == target.layer) & (df["sae_type"] == target.sae_type) &
                       (df["feature_id"] == target.feature_id) & (df["token_pos"] == target.token_pos)]
            if not match.empty:
                pairs.append((s, float(match[column].iloc[0])))
                break
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort()
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def compute_edge_weights(
    sa_dir: Path,
    trial_nums: List[int],
    optimal_strength: float,
    target: Optional[FeatureNode] = None,
    root_sa_dir: Optional[Path] = None,
    weighting_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Dict[Tuple, float]]:
    """Compute weighted edge weights using Simpson's rule.

    For hop-0 (target=None): ew(f) = ∫₀^s* SA(f, α) dα  (weight = 1)
    For backward hop-1+: ew(f→target) = ∫₀^s* GA_root(target, α) · SA(f→target, α) dα
    For forward hop-1+: ew(source→g) = ∫₀^s* SG(source, α) · SA_fwd(g, α) dα

    Args:
        target: Feature node for SA subdirectory lookup (None = root).
        root_sa_dir: Root SA dir for loading GA_root curve (backward hop-1+).
        weighting_curve: Explicit (strengths, values) curve to weight the integrand.
                        If provided, overrides GA_root loading. Use for forward tracing
                        (pass SG curve) or any custom weighting.
    """
    import pandas as pd
    from scipy.integrate import simpson

    is_root = target is None

    # Determine SA directory and parquet prefix
    if is_root:
        scan_dir = sa_dir
        parquet_prefix = "sa_trial"
    else:
        tgt_subdir = f"{target.sae_type}_L{target.layer}_F{target.feature_id}_T{target.token_pos}"
        scan_dir = sa_dir / "feat_sa" / tgt_subdir
        parquet_prefix = "feat_sa_trial"

    if not scan_dir.exists():
        return None

    # Load SA parquets at each strength
    sa_by_strength: Dict[float, pd.DataFrame] = {}
    for d in sorted(scan_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if not m:
            continue
        s = float(f"{m.group(1)}.{m.group(2)}")
        if s > optimal_strength + 0.1:
            continue
        for t in trial_nums:
            f = d / f"{parquet_prefix}{t}.parquet"
            if f.exists():
                sa_by_strength[s] = pd.read_parquet(f)
                break

    sorted_strengths = sorted(sa_by_strength.keys())
    if len(sorted_strengths) < 2:
        return None

    # Build SA matrix: pivot to (features × strengths)
    KEY_COLS = ["layer", "sae_type", "feature_id", "token_pos"]
    filtered = []
    for s in sorted_strengths:
        df_s = sa_by_strength[s].copy()
        df_s = df_s[df_s["feature_id"] >= 0]
        df_s["_strength"] = s
        filtered.append(df_s[KEY_COLS + ["steering_attribution", "_strength"]])

    if not filtered:
        return None
    all_sa = pd.concat(filtered, ignore_index=True)
    sa_wide = all_sa.pivot_table(
        index=KEY_COLS, columns="_strength", values="steering_attribution",
        aggfunc="first", fill_value=0.0,
    )
    sa_wide = sa_wide.reindex(columns=sorted_strengths, fill_value=0.0)
    sa_matrix = sa_wide.values  # [n_features, n_strengths]
    strengths_arr = np.array(sorted_strengths)

    # Load weighting curve (GA_root for backward, SG for forward, or explicit)
    if weighting_curve is not None:
        w_strengths, w_values = weighting_curve
        if len(w_strengths) >= 2:
            weight_interp = np.interp(sorted_strengths, w_strengths, w_values)
        else:
            weight_interp = np.ones(len(sorted_strengths))
    elif not is_root and root_sa_dir is not None:
        ga_strengths, ga_values = _load_curve_from_root_sa(
            root_sa_dir, trial_nums, target, "gradient_attribution")
        if len(ga_strengths) >= 2:
            weight_interp = np.interp(sorted_strengths, ga_strengths, ga_values)
        else:
            weight_interp = np.ones(len(sorted_strengths))
    else:
        weight_interp = np.ones(len(sorted_strengths))

    # Weighted integration with Simpson's rule (vectorized)
    integrand = sa_matrix * weight_interp[np.newaxis, :]
    edge_weights_arr = simpson(integrand, x=strengths_arr, axis=1)

    # Build result dict
    result: Dict[Tuple, float] = {}
    for idx, (_, row) in enumerate(sa_wide.index.to_frame(index=False).iterrows()):
        key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
        result[key] = float(edge_weights_arr[idx])

    return result


def select_features_from_edge_weights(
    edge_weights: Dict[Tuple, float],
    max_per_type: int,
    frac_of_max: float,
    injection_layer: int,
) -> List[Tuple[Tuple, float]]:
    """Select top features from edge weights using per-type frac-of-max + cap."""
    import pandas as pd

    records = [{"layer": k[0], "sae_type": k[1], "feature_id": k[2],
                "token_pos": k[3], "edge_weight": w}
               for k, w in edge_weights.items()
               if k[0] >= injection_layer and k[2] >= 0]
    if not records:
        return []
    df = pd.DataFrame(records)
    # Filter resid to last layer only
    last_layer = df["layer"].max()
    df = df[~((df["sae_type"] == "resid_post_all") & (df["layer"] != last_layer))]

    top_df = select_top_per_type(df, "edge_weight", "sae_type", max_per_type, frac_of_max)
    result = []
    for _, row in top_df.iterrows():
        key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
        result.append((key, float(row["edge_weight"])))
    return result


# =============================================================================
# Section B: Graph Construction (backward + forward tracing)
# =============================================================================

def build_attribution_graph(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    optimal_strength: float,
    output_dir: Path,
    trial_nums: List[int] = None,
    trace_depth: int = 2,
    n_strengths_feat_sa: int = DEFAULT_N_STRENGTHS,
    max_per_type: List[int] = None,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
    direction: str = DEFAULT_DIRECTION,
    device: str = "cuda",
) -> AttributionGraph:
    """Build multi-hop attribution graph with backward and/or forward tracing.

    Algorithm (from CLAUDE.md):
      For each hop:
        1. Extract SA for all targets/sources (GPU)
        2. Compute GA-weighted edge weights (∫GA×SA dα via Simpson's rule)
        3. Per-type selection with per-hop max_per_type cap
        4. Only ATTN+TC features traced to next hop
    """
    if trial_nums is None:
        trial_nums = [1]
    if max_per_type is None:
        max_per_type = DEFAULT_MAX_PER_TYPE

    root_sa_dir = output_dir  # Root SA parquets are in output_dir/strength_*/

    # ── Hop 0: compute edge weights from root SA ──
    print("\n  Computing hop-0 edge weights (root → features)...")
    hop0_ew = compute_edge_weights(root_sa_dir, trial_nums, optimal_strength)
    if not hop0_ew:
        print("  ERROR: No edge weights computed. Run extract-sa + compute-isa first.")
        return AttributionGraph(nodes={}, edges=[], optimal_strength=optimal_strength)

    mpt0 = _get_from_list(max_per_type, 0)
    hop0_selected = select_features_from_edge_weights(hop0_ew, mpt0, frac_of_max, injection_layer)

    graph = AttributionGraph(nodes={}, edges=[], optimal_strength=optimal_strength)
    root = FeatureNode(layer=-1, sae_type="root", feature_id=-1, token_pos=-1, isa_value=0, hop=-1)
    graph.nodes[root.key] = root

    visited_bwd: Set[Tuple] = set()
    visited_fwd: Set[Tuple] = set()

    for key, ew in hop0_selected:
        node = FeatureNode(key[0], key[1], key[2], key[3], ew, hop=0)
        graph.nodes[key] = node
        visited_bwd.add(key)
        visited_fwd.add(key)
        graph.edges.append(FeatureEdge(source_key=key, target_key=graph.root_key, weight=ew, hop=0))

    feat_sa_strengths = np.linspace(0, optimal_strength, n_strengths_feat_sa).tolist()
    do_backward = direction in ("backward", "both")
    do_forward = direction in ("forward", "both")

    # ── Backward tracing (hop-1+) ──
    if do_backward:
        # Select ATTN+TC trace targets
        bwd_targets = [graph.nodes[k] for k, _ in hop0_selected
                       if graph.nodes[k].sae_type in TRACE_SAE_TYPES
                       and graph.nodes[k].feature_id >= 0
                       and graph.nodes[k].layer > injection_layer]

        for hop in range(trace_depth):
            if not bwd_targets:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  Backward hop {hop+1}: tracing {len(bwd_targets)} targets (cap {mpt}/type)...")

            # Extract feature-targeted SA (uses SG cache from hop-0)
            sg_cache = output_dir / "sg_cache"
            for target in bwd_targets:
                print(f"    {target.short_name()}...")
                for s in tqdm(feat_sa_strengths, desc=f"    SA", leave=False):
                    extract_feature_target_sa(
                        model_wrapper, concept, concept_vector,
                        injection_layer, s,
                        target.layer, target.sae_type, target.feature_id, target.token_pos,
                        output_dir, trial_num=trial_nums[0], device=device,
                        sg_cache_dir=sg_cache,
                    )

            # Compute GA-weighted edge weights and select
            next_targets = []
            for target in bwd_targets:
                ew = compute_edge_weights(
                    output_dir, trial_nums, optimal_strength,
                    target=target, root_sa_dir=root_sa_dir)
                if not ew:
                    continue
                selected = select_features_from_edge_weights(ew, mpt, frac_of_max, injection_layer)
                for key, w in selected:
                    if key not in visited_bwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop+1)
                        graph.nodes[key] = node
                        visited_bwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES and node.layer > injection_layer:
                            next_targets.append(node)
                    graph.edges.append(FeatureEdge(source_key=key, target_key=target.key, weight=w, hop=hop+1))

            bwd_targets = next_targets
            print(f"  Backward hop {hop+1}: {len(next_targets)} new features")

    # ── Forward tracing (hop-1+) ──
    if do_forward:
        max_layer = model_wrapper.n_layers - 1
        fwd_sources = [graph.nodes[k] for k, _ in hop0_selected
                       if graph.nodes[k].sae_type in TRACE_SAE_TYPES
                       and graph.nodes[k].feature_id >= 0
                       and graph.nodes[k].layer < max_layer]

        for hop in range(trace_depth):
            if not fwd_sources:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  Forward hop {hop+1}: tracing {len(fwd_sources)} sources (cap {mpt}/type)...")

            # Extract forward SA for each source
            for source in fwd_sources:
                print(f"    {source.short_name()}...")
                for s in tqdm(feat_sa_strengths, desc=f"    fwd SA", leave=False):
                    extract_forward_sa_for_strength(
                        model_wrapper, concept, concept_vector,
                        injection_layer, s,
                        source.layer, source.sae_type, source.feature_id, source.token_pos,
                        output_dir, root_sa_dir=root_sa_dir,
                        trial_num=trial_nums[0], device=device,
                    )

            # Compute SG-weighted edge weights: ∫ SG(source,α) × SA_fwd(α) dα
            next_sources = []
            for source in fwd_sources:
                # Load SG curve for source from root SA
                sg_curve = _load_curve_from_root_sa(
                    root_sa_dir, trial_nums, source, "steering_grad")

                # Forward parquets use fwd_ prefix in subdir
                fwd_node = FeatureNode(source.layer, f"fwd_{source.sae_type}",
                                       source.feature_id, source.token_pos, 0, 0)
                ew = compute_edge_weights(
                    output_dir, trial_nums, optimal_strength,
                    target=fwd_node, weighting_curve=sg_curve)

                if not ew:
                    continue

                selected = select_features_from_edge_weights(
                    ew, mpt, frac_of_max, source.layer + 1)
                for key, w in selected:
                    if key not in visited_fwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop+1)
                        if key not in graph.nodes:
                            graph.nodes[key] = node
                        visited_fwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES and node.layer < max_layer:
                            next_sources.append(node)
                    graph.edges.append(FeatureEdge(
                        source_key=source.key, target_key=key, weight=w, hop=hop+1))

            fwd_sources = next_sources
            print(f"  Forward hop {hop+1}: {len(next_sources)} new features")

    # Dedup edges by (source, target) keeping max weight
    seen_edges: Dict[Tuple, FeatureEdge] = {}
    for e in graph.edges:
        edge_key = (e.source_key, e.target_key)
        if edge_key not in seen_edges or abs(e.weight) > abs(seen_edges[edge_key].weight):
            seen_edges[edge_key] = e
    graph.edges = list(seen_edges.values())

    print(f"\n  Graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


# =============================================================================
# Visualization
# =============================================================================

def export_graph_json(graph: AttributionGraph, path: Path) -> None:
    data = {
        "optimal_strength": graph.optimal_strength,
        "nodes": [{"layer": n.layer, "sae_type": n.sae_type, "feature_id": n.feature_id,
                    "token_pos": n.token_pos, "isa_value": n.isa_value, "hop": n.hop,
                    "label": n.label, "short_name": n.short_name()}
                   for n in graph.nodes.values()],
        "edges": [{"source": list(e.source_key), "target": list(e.target_key),
                    "weight": e.weight, "hop": e.hop}
                   for e in graph.edges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Graph JSON: {path}")


def render_pdf_from_html(html_path: Path, pdf_path: Path) -> None:
    """Generate PDF from an HTML file. Tries playwright first, falls back to weasyprint."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_timeout(2000)
            page.pdf(path=str(pdf_path), format="A3", landscape=True, print_background=True)
            browser.close()
        print(f"  PDF (playwright): {pdf_path}")
        return
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["chromium", "--headless", "--disable-gpu", "--no-sandbox",
             f"--print-to-pdf={pdf_path}", str(html_path.resolve())],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and pdf_path.exists():
            print(f"  PDF (chromium): {pdf_path}")
            return
    except Exception:
        pass

    print(f"  PDF generation skipped (install playwright: pip install playwright && playwright install chromium)")


def render_interactive(graph: AttributionGraph, path: Path) -> None:
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        print("  networkx/plotly not installed, skipping HTML")
        return

    COLORS = {"transcoder_all": "#4C72B0", "attn_out_all": "#DD8452",
              "mlp_out_all": "#55A868", "resid_post_all": "#C44E52", "root": "#333333"}

    G = nx.DiGraph()
    feature_nodes = {k: n for k, n in graph.nodes.items() if n.layer >= 0 and n.isa_value > 0}
    layers = sorted(set(n.layer for n in feature_nodes.values()))
    layer_to_y = {lv: i for i, lv in enumerate(layers)}

    layer_counts = defaultdict(int)
    for n in feature_nodes.values():
        layer_counts[n.layer] += 1

    layer_idx = defaultdict(int)
    for key, node in sorted(feature_nodes.items(), key=lambda x: abs(x[1].isa_value), reverse=True):
        idx = layer_idx[node.layer]
        layer_idx[node.layer] += 1
        x = (idx - layer_counts[node.layer] / 2.0) * 5.0
        y = layer_to_y.get(node.layer, 0) * 0.6
        G.add_node(key, pos=(x, y), **asdict(node))

    root_y = (max(layer_to_y.values()) + 1) * 0.6 if layer_to_y else 0.6
    G.add_node(graph.root_key, pos=(0, root_y))

    visible = set(G.nodes)
    vis_edges = [e for e in graph.edges if e.source_key in visible and e.target_key in visible]
    for e in vis_edges:
        G.add_edge(e.source_key, e.target_key, weight=e.weight)

    pos = nx.get_node_attributes(G, "pos")
    max_w = max((abs(e.weight) for e in vis_edges), default=1)

    edge_traces = []
    for e in vis_edges:
        if e.source_key not in pos or e.target_key not in pos:
            continue
        x0, y0 = pos[e.source_key]
        x1, y1 = pos[e.target_key]
        c = "rgba(196,78,82,0.6)" if e.weight > 0 else "rgba(76,114,176,0.6)"
        w = 1 + 4 * abs(e.weight) / max_w
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=w, color=c), hoverinfo="text",
            text=f"weight={e.weight:.4f}", showlegend=False))

    max_isa = max((abs(n.isa_value) for n in feature_nodes.values()), default=1)
    node_traces = {}
    for key in visible:
        node = graph.nodes.get(key)
        if node is None or key not in pos:
            continue
        x, y = pos[key]
        st = node.sae_type
        if st not in node_traces:
            node_traces[st] = {"x": [], "y": [], "text": [], "size": [], "color": COLORS.get(st, "#999")}
        size = 10 + 20 * (abs(node.isa_value) / max_isa) if node.layer >= 0 else 15
        name = node.short_name() if node.layer >= 0 else "dL/ds"
        hover = f"{name}<br>ISA={node.isa_value:.4f}"
        node_traces[st]["x"].append(x)
        node_traces[st]["y"].append(y)
        node_traces[st]["text"].append(hover)
        node_traces[st]["size"].append(size)

    fig = go.Figure()
    for t in edge_traces:
        fig.add_trace(t)
    for st, d in node_traces.items():
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="markers+text",
            marker=dict(size=d["size"], color=d["color"], line=dict(width=1, color="white")),
            text=[t.split("<br>")[0] for t in d["text"]], textposition="top center",
            textfont=dict(size=8), hovertext=d["text"], hoverinfo="text", name=st))

    fig.update_layout(
        title=f"Attribution Graph (strength={graph.optimal_strength:.2f})",
        showlegend=True, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white", width=1200, height=800 + len(layers) * 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    print(f"  Interactive HTML: {path}")


def write_graph_summary(graph: AttributionGraph, path: Path, concept: str, layer: int) -> None:
    lines = [
        f"Attribution Graph Summary: {concept} layer {layer}",
        f"Optimal strength: {graph.optimal_strength:.4f}",
        f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}",
        "", "=" * 60, "TOP FEATURES (hop 0 -> logit objective)", "=" * 60,
    ]
    hop0 = sorted([e for e in graph.edges if e.target_key == graph.root_key],
                   key=lambda e: abs(e.weight), reverse=True)
    for e in hop0:
        n = graph.nodes.get(e.source_key)
        if n:
            lines.append(f"  {n.short_name()}: ISA={e.weight:+.4f}")

    max_hop = max((e.hop for e in graph.edges), default=0)
    for hop in range(1, max_hop + 1):
        lines += ["", "=" * 60, f"HOP {hop} EDGES", "=" * 60]
        for e in sorted([e for e in graph.edges if e.hop == hop], key=lambda e: abs(e.weight), reverse=True):
            s = graph.nodes.get(e.source_key)
            t = graph.nodes.get(e.target_key)
            lines.append(f"  {s.short_name() if s else '?'} -> {t.short_name() if t else '?'}: w={e.weight:+.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Summary: {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 62: SAE Steering Attribution & Attribution Graph"
    )
    subparsers = parser.add_subparsers(dest="phase")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    common.add_argument("--concept", type=str, required=True)
    common.add_argument("-l", "--layer", type=int, default=DEFAULT_LAYER)
    common.add_argument("--trial-num", type=int, default=1)
    common.add_argument("--exp21-dir", type=str, default=DEFAULT_EXP21_DIR)
    common.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE)
    common.add_argument("-dt", "--dtype", type=str, default=DEFAULT_DTYPE)
    common.add_argument("-q", "--quantization", type=str, default=None, choices=["8bit", "4bit"])
    common.add_argument("--sae-width", type=str, default="16k")
    common.add_argument("--sae-l0", type=str, default="small")
    common.add_argument("--pos-token-id", type=int, default=None)
    common.add_argument("--neg-token-id", type=int, default=None)

    # auto-config
    p0 = subparsers.add_parser("auto-config", parents=[common], help="Auto-detect pos/neg tokens and optimal strength")
    p0.add_argument("--n-ll-strengths", type=int, default=51, help="Logit lens strength grid size")
    p0.add_argument("--n-scan-strengths", type=int, default=25, help="Behavioral scan grid size")
    p0.add_argument("--n-scan-trials", type=int, default=3, help="Trials per strength for LLM judge")
    p0.add_argument("--strength-max", type=float, default=DEFAULT_STRENGTH_MAX)

    # extract-sa
    p1 = subparsers.add_parser("extract-sa", parents=[common], help="Extract SA at one strength")
    p1.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)

    # compute-isa
    subparsers.add_parser("compute-isa", parents=[common], help="Compute ISA from SA data")

    # build-graph
    p3 = subparsers.add_parser("build-graph", parents=[common], help="Build attribution graph")
    p3.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH, help="Optimal strength")
    p3.add_argument("--trace-depth", type=int, default=DEFAULT_TRACE_DEPTH)
    p3.add_argument("--max-per-type", type=int, nargs="+", default=DEFAULT_MAX_PER_TYPE,
                     help="Per-hop max features per SAE type (e.g. 8 5 3 2)")
    p3.add_argument("--frac-of-max", type=float, default=DEFAULT_FRAC_OF_MAX)
    p3.add_argument("--n-strengths-feat-sa", type=int, default=DEFAULT_N_STRENGTHS)
    p3.add_argument("--direction", type=str, default=DEFAULT_DIRECTION,
                     choices=["backward", "forward", "both"])

    # visualize
    p4 = subparsers.add_parser("visualize", parents=[common], help="Render existing graph")
    p4.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)

    # all
    p5 = subparsers.add_parser("all", parents=[common], help="Full pipeline")
    p5.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH, help="Optimal strength")
    p5.add_argument("--n-strengths", type=int, default=DEFAULT_N_STRENGTHS)
    p5.add_argument("--strength-max", type=float, default=DEFAULT_STRENGTH_MAX)
    p5.add_argument("--trace-depth", type=int, default=DEFAULT_TRACE_DEPTH)
    p5.add_argument("--max-per-type", type=int, nargs="+", default=DEFAULT_MAX_PER_TYPE,
                     help="Per-hop max features per SAE type (e.g. 8 5 3 2)")
    p5.add_argument("--frac-of-max", type=float, default=DEFAULT_FRAC_OF_MAX)
    p5.add_argument("--direction", type=str, default=DEFAULT_DIRECTION,
                     choices=["backward", "forward", "both"])
    p5.add_argument("--n-strengths-feat-sa", type=int, default=11)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.phase is None:
        print("Specify a phase: auto-config, extract-sa, compute-isa, build-graph, visualize, or all")
        return

    concept = args.concept
    layer = args.layer
    trial_num = args.trial_num

    # Resolve token IDs
    tokens = CONCEPT_TOKEN_IDS.get(concept, DEFAULT_TOKEN_IDS)
    pos_id = args.pos_token_id if args.pos_token_id is not None else tokens["pos"]
    neg_id = args.neg_token_id if args.neg_token_id is not None else tokens["neg"]

    # Output dir
    run_name = f"{concept.replace(' ', '_')}_layer{layer}"
    base_out = Path(args.output_dir) / args.model / run_name
    base_out.mkdir(parents=True, exist_ok=True)

    if args.phase == "auto-config":
        run_auto_config(args)
        return

    if args.phase == "extract-sa":
        print(f"Extracting SA for {concept} layer {layer} strength {args.strength}")
        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        extract_steering_attribution(
            mw, concept, vectors[concept], layer, args.strength,
            base_out, trial_num, pos_id, neg_id, args.sae_width, args.sae_l0, args.device,
            sg_cache_dir=base_out / "sg_cache")

    elif args.phase == "compute-isa":
        print(f"Computing ISA for {concept} layer {layer}")
        compute_isa(base_out, [trial_num])

    elif args.phase == "build-graph":
        print(f"Building attribution graph for {concept} layer {layer}")
        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        graph = build_attribution_graph(
            mw, concept, vectors[concept], layer, args.strength,
            base_out, [trial_num], args.trace_depth, args.n_strengths_feat_sa,
            args.max_per_type, args.frac_of_max, args.direction, args.device)
        graph_dir = base_out / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graph_json(graph, graph_dir / "attribution_graph.json")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        render_pdf_from_html(graph_dir / "attribution_graph.html", graph_dir / "attribution_graph.pdf")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

    elif args.phase == "visualize":
        graph_json = base_out / "graphs" / "attribution_graph.json"
        if not graph_json.exists():
            print(f"  No graph found at {graph_json}. Run build-graph first.")
            return
        with open(graph_json) as f:
            data = json.load(f)
        graph = AttributionGraph(nodes={}, edges=[], optimal_strength=data.get("optimal_strength", args.strength))
        for nd in data["nodes"]:
            node = FeatureNode(nd["layer"], nd["sae_type"], nd["feature_id"],
                               nd["token_pos"], nd["isa_value"], nd.get("hop", 0), nd.get("label"))
            graph.nodes[node.key] = node
        for ed in data["edges"]:
            graph.edges.append(FeatureEdge(
                tuple(ed["source"]), tuple(ed["target"]), ed["weight"], ed.get("hop", 0)))
        graph_dir = base_out / "graphs"
        render_pdf_from_html(graph_dir / "attribution_graph.html", graph_dir / "attribution_graph.pdf")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

    elif args.phase == "all":
        print("=" * 70)
        print(f"FULL ATTRIBUTION GRAPH PIPELINE: {concept} layer {layer}")
        print("=" * 70)

        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        vec = vectors[concept]

        # Step 1: Extract SA at multiple strengths (saves SG cache for hop-1+ reuse)
        sg_cache = base_out / "sg_cache"
        strengths = np.linspace(0, args.strength_max, args.n_strengths).tolist()
        print(f"\nStep 1: Extracting SA at {len(strengths)} strengths (SG cached to {sg_cache})...")
        for s in tqdm(strengths, desc="SA extraction"):
            extract_steering_attribution(
                mw, concept, vec, layer, s, base_out, trial_num,
                pos_id, neg_id, args.sae_width, args.sae_l0, args.device,
                sg_cache_dir=sg_cache)

        # Step 2: Compute ISA
        print("\nStep 2: Computing ISA...")
        compute_isa(base_out, [trial_num])

        # Step 3: Build attribution graph
        print("\nStep 3: Building attribution graph...")
        graph = build_attribution_graph(
            mw, concept, vec, layer, args.strength, base_out,
            [trial_num], args.trace_depth, args.n_strengths_feat_sa,
            args.max_per_type, args.frac_of_max, args.direction, args.device)

        # Step 4: Visualize
        print("\nStep 4: Visualization...")
        graph_dir = base_out / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graph_json(graph, graph_dir / "attribution_graph.json")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        render_pdf_from_html(graph_dir / "attribution_graph.html", graph_dir / "attribution_graph.pdf")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print(f"  Output: {base_out}")
        print("=" * 70)


if __name__ == "__main__":
    main()
