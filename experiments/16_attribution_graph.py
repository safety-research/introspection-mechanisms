#!/usr/bin/env python3
"""
Experiment 62: SAE Steering Attribution & Attribution Graph

Performs SAE-level steering attribution (SA) analysis and builds multi-hop
attribution graphs tracing introspective detection through SAE/transcoder
features. Combines two complementary analyses:

  Section A: Steering Attribution (SA) extraction
             -- For each SAE feature at each layer, computes:
                GA = gradient attribution (dL/dx . w_dec_f)
                SG = steering gradient (dx/dalpha @ w_enc[:, f]) via JVP
                SA = GA * SG
             -- Integrates SA across injection strengths to get ISA.

  Section B: Attribution Graph construction
             -- Selects top features by ISA magnitude (IQR outlier detection)
             -- Recursively traces upstream: for each selected feature, computes
                feature-targeted SA (loss = target feature activation) to find
                which upstream features drive it.
             -- Builds a multi-hop directed graph from injection layer to logit
                objective, rendered as Graphviz PDF and interactive Plotly HTML.

Paper sections supported:
  - Section 5.3 ("Gate and Evidence Carrier Features")
  - Section 5.4 ("Circuit Analysis")
  - Appendix: Steering Attribution Framework
  - Figure 16 (bread-layer37-single-col): Attribution graph visualization

Model: Primarily Gemma-3 27B with Gemma Scope 2 SAEs/Transcoders
Steering: Layer 37, strength 4.0 (configurable)

Usage:
    # Section A: Extract SA for a concept at one strength (GPU)
    python 16_attribution_graph.py extract-sa --concept Bread --layer 37 --strength 4.0

    # Section A: Compute ISA by integrating SA across strengths (no GPU)
    python 16_attribution_graph.py compute-isa --concept Bread --layer 37

    # Section B: Build full attribution graph (requires SA data)
    python 16_attribution_graph.py build-graph --concept Bread --layer 37

    # Full pipeline: extract SA at multiple strengths, compute ISA, build graph
    python 16_attribution_graph.py all --concept Bread --layer 37

    # Visualize existing graph
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
DEFAULT_N_STRENGTHS = 25
DEFAULT_STRENGTH_MAX = 8.0
DEFAULT_TRACE_DEPTH = 2
DEFAULT_MAX_PER_TYPE = 8
DEFAULT_FRAC_OF_MAX = 0.10
DEFAULT_TOKEN_POS = -1
DEFAULT_EXP21_DIR = "analysis/exp21_more_concepts_steering"
DEFAULT_OUTPUT_DIR = "analysis/exp62_attribution_graph"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SEED = 42

# SAE types to analyze
SAE_TYPES = ["resid_post_all", "mlp_out_all", "attn_out_all", "transcoder_all"]

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
# Section A: SA Extraction
# =============================================================================

def _extract_sa_core(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    strength: float,
    trial_num: int = 1,
    sae_width: str = "16k",
    sae_l0: str = "small",
    device: str = "cuda",
    # ── What differs between root-loss and feature-targeted modes ──
    target: Optional[Dict] = None,
    pos_token_id: int = 12932,
    neg_token_id: int = 3771,
    compute_remainder: bool = True,
) -> List[Dict[str, Any]]:
    """Core two-pass SA extraction shared by both loss modes.

    Two-pass approach:
      Pass 1 (forward + backward): Compute GA = dL/dx at each site
      Pass 2 (JVP forward): Compute SG = dx/dalpha via forward-mode AD
      Combine: SA = GA * SG per active feature

    Args:
        target: If None, uses logit-gap loss (root mode, all layers).
                If dict with {layer, sae_type, feature_id, token_pos},
                uses target-feature-activation loss (upstream layers only).
        pos_token_id / neg_token_id: Token IDs for logit-gap loss (root mode only).
        compute_remainder: Whether to compute per-token remainder SA diagnostics.
                          True for root mode, False for feature-targeted mode.

    Returns:
        List of row dicts ready for pd.DataFrame / parquet.
    """
    is_root = target is None
    tokenizer = model_wrapper.tokenizer
    inner = model_wrapper.model

    # Build prompt
    messages = build_messages(trial_num)
    prompt_str, input_ids, seq_len = format_prompt(messages, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    steering_start = find_steering_start(tokenizer, prompt_str, trial_num)

    n_layers = model_wrapper.n_layers
    model_size = _model_name_to_sae_size(model_wrapper.model_name)
    steering_vec = concept_vector.to(device).float()
    layers_module = get_layers(inner)

    # Layer range: all layers for root, injection..target for feature-targeted
    if is_root:
        layer_lo, layer_hi = 0, n_layers  # [0, n_layers)
    else:
        layer_lo = injection_layer
        layer_hi = target["layer"]  # exclusive: don't include target layer itself

    # Hook range for activation capture: include target_layer+1 for feature-targeted
    # (need target site activation to compute loss)
    hook_hi = n_layers if is_root else target["layer"] + 1

    # Preload SAEs to CPU
    unique_types = set(SAE_TYPE_LOAD_MAP.get(st, st) for st in SAE_TYPES)
    sae_cache: Dict[Tuple[int, str], JumpReLUSAE] = {}
    for li in tqdm(range(layer_lo, layer_hi), desc="Preloading SAEs", leave=False):
        for st in unique_types:
            try:
                sae = load_sae(li, sae_width, sae_l0, st, model_size, True, "cpu")
                sae_cache[(li, st)] = sae.eval()
            except Exception:
                pass

    # For feature-targeted mode, also load the target SAE
    target_sae = None
    if not is_root:
        tgt_load = SAE_TYPE_LOAD_MAP.get(target["sae_type"], target["sae_type"])
        target_sae = load_sae(target["layer"], sae_width, sae_l0, tgt_load, model_size, True, device)

    def make_steering_hook(s):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            addition = torch.zeros_like(h)
            addition[:, steering_start:, :] = (steering_vec * s).unsqueeze(0)
            modified = h + addition
            return (modified,) + rest if isinstance(output, tuple) else modified
        return hook

    def run_forward(s, retain_grad=False):
        handle = layers_module[injection_layer].register_forward_hook(make_steering_hook(s))
        hooks = ActivationHooks(inner, retain_grad=retain_grad)
        hooks.register_hooks(list(range(layer_lo, hook_hi)), CAPTURE_TYPES_SAE)
        try:
            with hooks:
                inner.eval()
                out = inner(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
            hooks.remove_hooks()
        return out, hooks.get_activations()

    # ── Pass 1: Forward + backward for GA ──
    outputs, raw_acts = run_forward(strength, retain_grad=True)

    if is_root:
        logits = outputs.logits[:, -1, :]
        loss = logits[:, pos_token_id] - logits[:, neg_token_id]
    else:
        # Loss = target feature activation
        tgt_suffix = SAE_TYPE_KEYS[target["sae_type"]][0]
        tgt_act_key = _act_key(target["layer"], tgt_suffix)
        if tgt_act_key not in raw_acts or not raw_acts[tgt_act_key]:
            return []
        tgt_act = raw_acts[tgt_act_key][0]
        if isinstance(tgt_act, (list, tuple)):
            tgt_act = tgt_act[0]
        tgt_act_2d = tgt_act[0] if tgt_act.dim() == 3 else tgt_act
        tgt_encoded = target_sae.encode(tgt_act_2d.float().to(device))
        tp = target["token_pos"] if target["token_pos"] >= 0 else seq_len + target["token_pos"]
        loss = tgt_encoded[tp, target["feature_id"]]

    loss.sum().backward()

    # Collect gradients and activations
    grad_data: Dict[Tuple[int, str], torch.Tensor] = {}
    act_data: Dict[Tuple[int, str], torch.Tensor] = {}
    for li in range(layer_lo, layer_hi):
        for st in SAE_TYPES:
            in_suffix, tgt_suffix = SAE_TYPE_KEYS[st]
            key_suffix = tgt_suffix if (st == "transcoder_all" and tgt_suffix) else in_suffix
            akey = _act_key(li, key_suffix)
            if akey not in raw_acts or not raw_acts[akey]:
                continue
            t = raw_acts[akey][0]
            if isinstance(t, (list, tuple)):
                t = t[0]
            if t.grad is None:
                continue
            grad_data[(li, st)] = t.grad.detach().float()[0].cpu()

            enc_key = _act_key(li, in_suffix)
            if enc_key not in raw_acts or not raw_acts[enc_key]:
                continue
            enc_t = raw_acts[enc_key][0]
            if isinstance(enc_t, (list, tuple)):
                enc_t = enc_t[0]
            act_data[(li, st)] = enc_t[0].detach().float().cpu()

    del raw_acts, outputs
    torch.cuda.empty_cache()

    # ── Pass 2: JVP for SG ──
    alpha_primal = torch.tensor(strength, dtype=torch.float32, device=device)
    alpha_tangent = torch.tensor(1.0, dtype=torch.float32, device=device)

    def jvp_forward(alpha_val):
        handle = layers_module[injection_layer].register_forward_hook(make_steering_hook(alpha_val))
        hooks = ActivationHooks(inner, retain_grad=False)
        hooks.register_hooks(list(range(layer_lo, layer_hi)), CAPTURE_TYPES_SAE)
        try:
            with hooks:
                inner.eval()
                _ = inner(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
            hooks.remove_hooks()
        raw = hooks.get_activations()
        result = []
        for li in range(layer_lo, layer_hi):
            for suffix in JVP_SITE_SUFFIXES:
                akey = _act_key(li, suffix)
                t = raw[akey][0] if akey in raw and raw[akey] else torch.zeros(1)
                if isinstance(t, (list, tuple)):
                    t = t[0]
                result.append(t[0] if t.dim() == 3 else t)
        return result

    _, site_tangents = torch.func.jvp(jvp_forward, (alpha_primal,), (alpha_tangent,))

    tangent_data: Dict[Tuple[int, str], torch.Tensor] = {}
    for li_offset, li in enumerate(range(layer_lo, layer_hi)):
        for j, suffix in enumerate(JVP_SITE_SUFFIXES):
            idx = li_offset * len(JVP_SITE_SUFFIXES) + j
            tangent_data[(li, suffix)] = site_tangents[idx].detach()

    del site_tangents
    torch.cuda.empty_cache()

    # ── Combine: compute SA per active feature ──
    base_meta: Dict[str, Any] = {
        "concept": concept,
        "injection_layer": injection_layer,
        "injection_strength": strength,
        "trial_num": trial_num,
    }
    if not is_root:
        base_meta.update({
            "target_layer": target["layer"], "target_sae_type": target["sae_type"],
            "target_feature_id": target["feature_id"], "target_token_pos": target["token_pos"],
        })

    feature_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for li in tqdm(range(layer_lo, layer_hi), desc="SA combine", leave=False):
            for st in SAE_TYPES:
                in_suffix, tgt_suffix = SAE_TYPE_KEYS[st]
                load_type = SAE_TYPE_LOAD_MAP.get(st, st)
                if (li, load_type) not in sae_cache or (li, st) not in grad_data:
                    continue

                grad_tensor = grad_data[(li, st)].to(device)
                x_all = act_data[(li, st)].to(device)
                sae = sae_cache[(li, load_type)].to(device=device, dtype=torch.float32).eval()

                sae_acts = sae.encode(x_all)
                active_mask = sae_acts > 0
                tok_idx, feat_idx = active_mask.nonzero(as_tuple=True)
                n_active = len(tok_idx)

                ga_vals = torch.zeros(n_active, device=device)
                sg_vals = torch.zeros(n_active, device=device)
                active_acts = sae_acts[tok_idx, feat_idx] if n_active > 0 else torch.tensor([])

                if n_active > 0:
                    ga_vals = (grad_tensor[tok_idx] * sae.w_dec[feat_idx]).sum(dim=-1)

                has_grad_path = (li > injection_layer) or (li == injection_layer and st == "resid_post_all")
                if has_grad_path and n_active > 0:
                    tkey = (li, in_suffix)
                    if tkey in tangent_data:
                        dx = tangent_data[tkey].to(device).float()
                        if dx.dim() == 3:
                            dx = dx[0]
                        sg_vals = (dx[tok_idx] * sae.w_enc[:, feat_idx].T).sum(dim=-1)

                sa_vals = ga_vals * sg_vals

                if n_active > 0:
                    tl = tok_idx.cpu().tolist()
                    fl = feat_idx.cpu().tolist()
                    al = active_acts.cpu().tolist()
                    gl = ga_vals.cpu().tolist()
                    sl = sg_vals.cpu().tolist()
                    sal = sa_vals.cpu().tolist()
                    for i in range(n_active):
                        feature_rows.append({
                            **base_meta, "layer": li, "token_pos": tl[i],
                            "sae_type": st, "feature_id": fl[i],
                            "activation": al[i], "gradient_attribution": gl[i],
                            "steering_grad": sl[i], "steering_attribution": sal[i],
                        })

                # Per-token remainder SA (only for root mode)
                if compute_remainder:
                    # remainder_SA[t] = (grad[t] . dx/dalpha[t]) - sum_f SA_f(t)
                    tkey = (li, in_suffix)
                    total = torch.zeros(seq_len, device=device)
                    if has_grad_path and tkey in tangent_data:
                        dx = tangent_data[tkey].to(device).float()
                        if dx.dim() == 3:
                            dx = dx[0]
                        total = (grad_tensor * dx).sum(dim=-1)

                    feat_sa_sum = torch.zeros(seq_len, device=device)
                    if n_active > 0:
                        feat_sa_sum.scatter_add_(0, tok_idx, sa_vals)
                    rem_sa = total - feat_sa_sum

                    # Remainder diagnostics: norm, GA, SG
                    if tgt_suffix is not None:
                        recon_all = sae(x_all)
                        if (li, "mlp_out_all") in act_data:
                            target_all = act_data[(li, "mlp_out_all")].to(device)
                        else:
                            target_all = x_all
                        remainder = target_all - recon_all
                    else:
                        remainder = x_all - sae.decode(sae_acts)

                    rem_norm = remainder.norm(dim=-1)
                    safe_norm = rem_norm.clamp(min=1e-10).unsqueeze(-1)
                    rem_ga = (grad_tensor * (remainder / safe_norm)).sum(dim=-1)

                    rem_sg = torch.zeros(seq_len, device=device)
                    if has_grad_path and tkey in tangent_data:
                        dx_rem = tangent_data[tkey].to(device).float()
                        if dx_rem.dim() == 3:
                            dx_rem = dx_rem[0]
                        drecon_dalpha = torch.zeros_like(dx_rem)
                        if n_active > 0:
                            contrib = sg_vals.unsqueeze(-1) * sae.w_dec[feat_idx]
                            drecon_dalpha.index_add_(0, tok_idx, contrib)
                        rem_sg = (dx_rem - drecon_dalpha).norm(dim=-1)

                    rem_sa_l = rem_sa.cpu().tolist()
                    rem_norm_l = rem_norm.cpu().tolist()
                    rem_ga_l = rem_ga.cpu().tolist()
                    rem_sg_l = rem_sg.cpu().tolist()
                    for t in range(seq_len):
                        feature_rows.append({
                            **base_meta, "layer": li, "token_pos": t,
                            "sae_type": st, "feature_id": -1,
                            "activation": rem_norm_l[t],
                            "gradient_attribution": rem_ga_l[t],
                            "steering_grad": rem_sg_l[t],
                            "steering_attribution": rem_sa_l[t],
                        })

                sae.to("cpu")
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return feature_rows


def extract_steering_attribution(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    strength: float,
    output_dir: Path,
    trial_num: int = 1,
    pos_token_id: int = 12932,
    neg_token_id: int = 3771,
    sae_width: str = "16k",
    sae_l0: str = "small",
    device: str = "cuda",
) -> Optional[Path]:
    """Extract SA for all SAE features at one injection strength (root logit-gap loss).

    Returns path to output parquet, or None on failure.
    """
    import pandas as pd

    rows = _extract_sa_core(
        model_wrapper, concept, concept_vector, injection_layer, strength,
        trial_num, sae_width, sae_l0, device,
        target=None, pos_token_id=pos_token_id, neg_token_id=neg_token_id,
        compute_remainder=True,
    )

    strength_str = f"strength_{int(strength)}_{int((strength % 1) * 100):02d}"
    out_path = output_dir / strength_str / f"sa_trial{trial_num}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"  Saved {len(rows)} rows to {out_path}")
    return out_path


def extract_feature_target_sa(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    strength: float,
    target_layer: int,
    target_sae_type: str,
    target_feature_id: int,
    target_token_pos: int,
    output_dir: Path,
    trial_num: int = 1,
    sae_width: str = "16k",
    sae_l0: str = "small",
    device: str = "cuda",
) -> Optional[Path]:
    """Extract feature-targeted SA (loss = target feature activation).

    Returns path to output parquet, or None on failure.
    """
    import pandas as pd

    target = {
        "layer": target_layer, "sae_type": target_sae_type,
        "feature_id": target_feature_id, "token_pos": target_token_pos,
    }
    rows = _extract_sa_core(
        model_wrapper, concept, concept_vector, injection_layer, strength,
        trial_num, sae_width, sae_l0, device,
        target=target, compute_remainder=False,
    )

    tgt_subdir = f"{target_sae_type}_L{target_layer}_F{target_feature_id}_T{target_token_pos}"
    s_str = f"strength_{int(strength)}_{int((strength % 1) * 100):02d}"
    out_path = output_dir / "feat_sa" / tgt_subdir / s_str / f"feat_sa_trial{trial_num}.parquet"
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
            isa_val = float(np.trapz(sa, ss))
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
# Section B: Feature Selection
# =============================================================================

def select_top_per_type(
    df, value_col: str, sae_type_col: str = "sae_type",
    max_per_type: int = 8, frac_of_max: float = 0.10,
):
    """Per SAE type: select features with value > frac_of_max * type_max, capped at max_per_type.

    This adapts to each type's scale without a global threshold. Only considers
    positive values. Returns DataFrame sorted by value descending.
    """
    import pandas as pd

    SAE_ABBREV = {"transcoder_all": "TC", "attn_out_all": "ATTN",
                  "mlp_out_all": "MLP", "resid_post_all": "RESID"}
    pos = df[df[value_col] > 0]
    parts = []
    for st in sorted(pos[sae_type_col].unique()):
        st_df = pos[pos[sae_type_col] == st].sort_values(value_col, ascending=False)
        if st_df.empty:
            continue
        max_val = st_df[value_col].iloc[0]
        thresh = frac_of_max * max_val
        above = st_df[st_df[value_col] > thresh]
        selected = above.head(max_per_type)
        parts.append(selected)
        abbrev = SAE_ABBREV.get(st, st)
        print(f"    {abbrev}: {len(selected)} features "
              f"(>{frac_of_max:.0%} of max {max_val:.4f}, {len(above)} above, cap {max_per_type})")
    if not parts:
        return pos.iloc[:0]
    return pd.concat(parts).sort_values(value_col, ascending=False)


def select_top_features(
    sa_dir: Path,
    injection_layer: int,
    trial_nums: List[int],
    max_per_type: int = DEFAULT_MAX_PER_TYPE,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
) -> List[FeatureNode]:
    """Select top features from ISA data using per-type frac-of-max threshold + cap."""
    import pandas as pd

    dfs = []
    for trial in trial_nums:
        p = sa_dir / f"isa_trial{trial}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError(f"No ISA parquets in {sa_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["feature_id"] >= 0]
    df = df[df["layer"] >= injection_layer]

    grouped = df.groupby(["layer", "sae_type", "feature_id", "token_pos"]).agg(
        isa_mean=("integrated_steering_attribution", "mean"),
    ).reset_index()

    top_df = select_top_per_type(grouped, "isa_mean", "sae_type", max_per_type, frac_of_max)
    if top_df.empty:
        print("  Warning: no features above threshold, taking top-10 by |ISA|")
        top_df = grouped.reindex(grouped["isa_mean"].abs().sort_values(ascending=False).index).head(10)

    # Filter resid to last layer only
    last_layer = grouped["layer"].max()
    top_df = top_df[~((top_df["sae_type"] == "resid_post_all") & (top_df["layer"] != last_layer))]

    features = []
    for _, row in top_df.iterrows():
        features.append(FeatureNode(
            layer=int(row["layer"]), sae_type=row["sae_type"],
            feature_id=int(row["feature_id"]), token_pos=int(row["token_pos"]),
            isa_value=float(row["isa_mean"]), hop=0,
        ))
    features.sort(key=lambda f: abs(f.isa_value), reverse=True)
    print(f"  Selected {len(features)} features")
    for f in features[:10]:
        print(f"    {f.short_name()}: ISA={f.isa_value:.4f}")
    return features


# =============================================================================
# Section B: Graph Construction
# =============================================================================

def select_upstream_features(
    sa_dir: Path,
    target: FeatureNode,
    injection_layer: int,
    trial_nums: List[int],
    visited: Set[Tuple],
    hop: int,
    max_per_type: int = DEFAULT_MAX_PER_TYPE,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
) -> Tuple[List[FeatureNode], List[FeatureEdge]]:
    """Select upstream features from feature-targeted SA data."""
    import pandas as pd

    tgt_subdir = f"{target.sae_type}_L{target.layer}_F{target.feature_id}_T{target.token_pos}"
    feat_sa_dir = sa_dir / "feat_sa" / tgt_subdir

    if not feat_sa_dir.exists():
        return [], []

    # Load all strength dirs
    feat_sa_by_strength: Dict[float, pd.DataFrame] = {}
    for d in sorted(feat_sa_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if not m:
            continue
        s = float(f"{m.group(1)}.{m.group(2)}")
        for t in trial_nums:
            f = d / f"feat_sa_trial{t}.parquet"
            if f.exists():
                feat_sa_by_strength[s] = pd.read_parquet(f)
                break

    if len(feat_sa_by_strength) < 2:
        return [], []

    sorted_strengths = sorted(feat_sa_by_strength.keys())

    # Build per-feature SA arrays
    feat_sa_dict: Dict[Tuple, np.ndarray] = {}
    first_df = feat_sa_by_strength[sorted_strengths[0]]
    first_df = first_df[(first_df["feature_id"] >= 0) &
                        (first_df["layer"] >= injection_layer) &
                        (first_df["layer"] < target.layer)]

    for _, row in first_df.iterrows():
        key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
        feat_sa_dict[key] = np.zeros(len(sorted_strengths))

    for si, s in enumerate(sorted_strengths):
        df_s = feat_sa_by_strength[s]
        df_s = df_s[(df_s["feature_id"] >= 0) &
                    (df_s["layer"] >= injection_layer) &
                    (df_s["layer"] < target.layer)]
        for _, row in df_s.iterrows():
            key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
            if key in feat_sa_dict:
                feat_sa_dict[key][si] = float(row["steering_attribution"])

    # Integrate
    s_arr = np.array(sorted_strengths)
    weights = {}
    for key, sa_arr in feat_sa_dict.items():
        weights[key] = float(np.trapz(sa_arr, s_arr))

    # IQR outlier selection
    records = [{"layer": k[0], "sae_type": k[1], "feature_id": k[2],
                "token_pos": k[3], "weight": w} for k, w in weights.items()]
    if not records:
        return [], []
    weight_df = pd.DataFrame(records)
    outlier_df = select_top_per_type(weight_df, "weight", "sae_type", max_per_type, frac_of_max)

    features, edges = [], []
    for _, row in outlier_df.iterrows():
        key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
        w = row["weight"]
        edges.append(FeatureEdge(source_key=key, target_key=target.key, weight=w, hop=hop))
        if key not in visited:
            features.append(FeatureNode(
                layer=key[0], sae_type=key[1], feature_id=key[2],
                token_pos=key[3], isa_value=w, hop=hop,
            ))
    features.sort(key=lambda f: abs(f.isa_value), reverse=True)
    return features, edges


def build_attribution_graph(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    optimal_strength: float,
    output_dir: Path,
    trial_nums: List[int] = None,
    trace_depth: int = 2,
    n_strengths_feat_sa: int = 11,
    max_per_type: int = DEFAULT_MAX_PER_TYPE,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
    device: str = "cuda",
) -> AttributionGraph:
    """Build multi-hop attribution graph.

    1. Select top features from ISA (hop 0 -> logit objective)
    2. For each hop: extract feature-targeted SA, compute ISA, select upstream
    3. Repeat for trace_depth hops
    """
    if trial_nums is None:
        trial_nums = [1]

    sa_dir = output_dir

    # Select hop-0 features
    print("\n  Selecting top features from ISA...")
    hop0 = select_top_features(sa_dir, injection_layer, trial_nums, max_per_type, frac_of_max)

    graph = AttributionGraph(nodes={}, edges=[], optimal_strength=optimal_strength)
    root = FeatureNode(layer=-1, sae_type="root", feature_id=-1, token_pos=-1, isa_value=0, hop=-1)
    graph.nodes[root.key] = root

    visited: Set[Tuple] = set()
    for feat in hop0:
        graph.nodes[feat.key] = feat
        visited.add(feat.key)
        graph.edges.append(FeatureEdge(
            source_key=feat.key, target_key=graph.root_key,
            weight=feat.isa_value, hop=0,
        ))

    # Upstream tracing
    feat_sa_strengths = np.linspace(0, optimal_strength, n_strengths_feat_sa).tolist()
    current_targets = hop0

    for hop in range(trace_depth):
        if not current_targets:
            break
        print(f"\n  Hop {hop}: tracing {len(current_targets)} targets...")

        # Extract feature-targeted SA for each target
        for target in current_targets:
            if target.layer <= injection_layer:
                continue
            print(f"    Extracting feat-target SA for {target.short_name()}...")
            for s in tqdm(feat_sa_strengths, desc=f"    {target.short_name()}", leave=False):
                extract_feature_target_sa(
                    model_wrapper, concept, concept_vector,
                    injection_layer, s,
                    target.layer, target.sae_type, target.feature_id, target.token_pos,
                    output_dir, trial_num=trial_nums[0], device=device,
                )

        # Select upstream features
        next_targets = []
        for target in current_targets:
            if target.layer <= injection_layer:
                continue
            upstream, new_edges = select_upstream_features(
                sa_dir, target, injection_layer, trial_nums,
                visited, hop + 1, max_per_type, frac_of_max,
            )
            for feat in upstream:
                graph.nodes[feat.key] = feat
                visited.add(feat.key)
                next_targets.append(feat)
            graph.edges.extend(new_edges)

        current_targets = next_targets
        print(f"  Hop {hop}: {len(next_targets)} new features discovered")

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


def render_graphviz(graph: AttributionGraph, path: Path) -> None:
    try:
        import graphviz
    except ImportError:
        print("  graphviz not installed, skipping PDF. pip install graphviz")
        return

    COLORS = {"transcoder_all": "#4C72B0", "attn_out_all": "#DD8452",
              "mlp_out_all": "#55A868", "resid_post_all": "#C44E52", "root": "#333333"}

    dot = graphviz.Digraph(
        format="pdf", engine="dot",
        graph_attr={"rankdir": "BT", "splines": "true", "nodesep": "0.5",
                     "ranksep": "0.8", "fontsize": "10",
                     "label": f"Attribution Graph\nstrength={graph.optimal_strength:.2f}"},
        node_attr={"fontsize": "9", "style": "filled"},
        edge_attr={"fontsize": "8"},
    )

    layers = sorted(set(n.layer for n in graph.nodes.values() if n.layer >= 0))
    max_isa = max((abs(n.isa_value) for n in graph.nodes.values() if n.layer >= 0), default=1)

    for lv in layers:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node in graph.nodes.values():
                if node.layer != lv:
                    continue
                nid = f"L{node.layer}_{node.sae_type}_F{node.feature_id}_T{node.token_pos}"
                color = COLORS.get(node.sae_type, "#999")
                size = 0.3 + 0.7 * (abs(node.isa_value) / max_isa)
                label = node.short_name()
                if node.label:
                    label += f"\n{node.label[:40]}"
                label += f"\nISA={node.isa_value:.3f}"
                s.node(nid, label=label, fillcolor=color, fontcolor="white",
                       width=str(size), shape="box", penwidth="2")

    dot.node("root", label="dL/ds", shape="ellipse",
             fillcolor=COLORS["root"], fontcolor="white", width="1.2")

    max_w = max((abs(e.weight) for e in graph.edges), default=1)
    for edge in graph.edges:
        src = graph.nodes.get(edge.source_key)
        if src is None:
            continue
        sid = f"L{src.layer}_{src.sae_type}_F{src.feature_id}_T{src.token_pos}"
        tid = "root" if edge.target_key == graph.root_key else \
              f"L{edge.target_key[0]}_{edge.target_key[1]}_F{edge.target_key[2]}_T{edge.target_key[3]}"
        pw = str(0.5 + 3.0 * abs(edge.weight) / max_w)
        c = "#C44E52" if edge.weight > 0 else "#4C72B0"
        dot.edge(sid, tid, label=f"{edge.weight:.3f}", penwidth=pw, color=c, fontcolor=c)

    stem = str(path.with_suffix(""))
    dot.render(stem, cleanup=True)
    print(f"  Graphviz PDF: {stem}.pdf")


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

    # extract-sa
    p1 = subparsers.add_parser("extract-sa", parents=[common], help="Extract SA at one strength")
    p1.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)

    # compute-isa
    subparsers.add_parser("compute-isa", parents=[common], help="Compute ISA from SA data")

    # build-graph
    p3 = subparsers.add_parser("build-graph", parents=[common], help="Build attribution graph")
    p3.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH, help="Optimal strength")
    p3.add_argument("--trace-depth", type=int, default=DEFAULT_TRACE_DEPTH)
    p3.add_argument("--max-per-type", type=int, default=DEFAULT_MAX_PER_TYPE)
    p3.add_argument("--frac-of-max", type=float, default=DEFAULT_FRAC_OF_MAX)
    p3.add_argument("--n-strengths-feat-sa", type=int, default=11)

    # visualize
    p4 = subparsers.add_parser("visualize", parents=[common], help="Render existing graph")
    p4.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)

    # all
    p5 = subparsers.add_parser("all", parents=[common], help="Full pipeline")
    p5.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH, help="Optimal strength")
    p5.add_argument("--n-strengths", type=int, default=DEFAULT_N_STRENGTHS)
    p5.add_argument("--strength-max", type=float, default=DEFAULT_STRENGTH_MAX)
    p5.add_argument("--trace-depth", type=int, default=DEFAULT_TRACE_DEPTH)
    p5.add_argument("--max-per-type", type=int, default=DEFAULT_MAX_PER_TYPE)
    p5.add_argument("--frac-of-max", type=float, default=DEFAULT_FRAC_OF_MAX)
    p5.add_argument("--n-strengths-feat-sa", type=int, default=11)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.phase is None:
        print("Specify a phase: extract-sa, compute-isa, build-graph, visualize, or all")
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

    if args.phase == "extract-sa":
        print(f"Extracting SA for {concept} layer {layer} strength {args.strength}")
        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        extract_steering_attribution(
            mw, concept, vectors[concept], layer, args.strength,
            base_out, trial_num, pos_id, neg_id, args.sae_width, args.sae_l0, args.device)

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
            args.max_per_type, args.frac_of_max, args.device)
        graph_dir = base_out / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graph_json(graph, graph_dir / "attribution_graph.json")
        render_graphviz(graph, graph_dir / "attribution_graph")
        render_interactive(graph, graph_dir / "attribution_graph.html")
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
        render_graphviz(graph, graph_dir / "attribution_graph")
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

        # Step 1: Extract SA at multiple strengths
        strengths = np.linspace(0, args.strength_max, args.n_strengths).tolist()
        print(f"\nStep 1: Extracting SA at {len(strengths)} strengths...")
        for s in tqdm(strengths, desc="SA extraction"):
            extract_steering_attribution(
                mw, concept, vec, layer, s, base_out, trial_num,
                pos_id, neg_id, args.sae_width, args.sae_l0, args.device)

        # Step 2: Compute ISA
        print("\nStep 2: Computing ISA...")
        compute_isa(base_out, [trial_num])

        # Step 3: Build attribution graph
        print("\nStep 3: Building attribution graph...")
        graph = build_attribution_graph(
            mw, concept, vec, layer, args.strength, base_out,
            [trial_num], args.trace_depth, args.n_strengths_feat_sa,
            args.max_per_type, args.frac_of_max, args.device)

        # Step 4: Visualize
        print("\nStep 4: Visualization...")
        graph_dir = base_out / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graph_json(graph, graph_dir / "attribution_graph.json")
        render_graphviz(graph, graph_dir / "attribution_graph")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print(f"  Output: {base_out}")
        print("=" * 70)


if __name__ == "__main__":
    main()
