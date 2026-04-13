"""
Utilities for activation patching in Experiment 36.

This module handles:
- Component specification parsing (L45, L45MLP, L45ATTN, L45H0, etc.)
- Caching source activations from forward passes
- Computing baseline activations
- Running generation with activation patching
"""

import torch
import re
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
from model_utils import ModelWrapper


@dataclass
class ComponentSpec:
    """Specification for which component(s) to patch."""
    layers: List[int]
    component: str  # 'resid', 'mlp', 'attn', 'head'
    heads: Optional[List[int]] = None

    def __repr__(self):
        if self.component == 'head' and self.heads:
            return f"ComponentSpec(layers={self.layers}, component='{self.component}', heads={self.heads})"
        return f"ComponentSpec(layers={self.layers}, component='{self.component}')"


def parse_component_spec(spec: str, n_layers: int = None) -> ComponentSpec:
    """
    Parse component specification string.

    Grammar:
        COMPONENT_SPEC := LAYER_SPEC [COMPONENT_TYPE] [HEAD_SPEC]
        LAYER_SPEC := 'L' NUMBER | 'L' NUMBER '-' NUMBER | 'L' FRACTION
        COMPONENT_TYPE := 'MLP' | 'ATTN' | 'RESID'
        HEAD_SPEC := 'H' NUMBER | 'H' NUMBER '-' NUMBER

    Examples:
        L45         -> Residual stream at layer 45
        L45MLP      -> MLP output at layer 45
        L45ATTN     -> Full attention output at layer 45
        L45H0       -> Attention head 0 at layer 45
        L45H0-3     -> Attention heads 0,1,2,3 at layer 45
        L40-45      -> Residual stream at layers 40,41,42,43,44,45
        L40-45MLP   -> MLP outputs at layers 40-45
        L0.7        -> Residual stream at 70% through model (requires n_layers)
        L0.7MLP     -> MLP output at 70% through model

    Args:
        spec: Component specification string
        n_layers: Total number of layers in model (required for fraction-based specs)

    Returns:
        ComponentSpec with parsed information
    """
    spec = spec.strip().upper()

    # Initialize defaults
    layers = []
    component = 'resid'
    heads = None

    # Parse layer specification
    # Match L followed by number(s) or fraction
    layer_match = re.match(r'^L(\d+(?:\.\d+)?(?:-\d+)?)', spec)
    if not layer_match:
        raise ValueError(f"Invalid component spec: {spec}. Must start with L followed by layer number(s).")

    layer_str = layer_match.group(1)
    remaining = spec[layer_match.end():]

    # Handle layer range (e.g., 40-45)
    if '-' in layer_str and '.' not in layer_str:
        start, end = layer_str.split('-')
        layers = list(range(int(start), int(end) + 1))
    # Handle fraction (e.g., 0.7)
    elif '.' in layer_str:
        if n_layers is None:
            raise ValueError(f"n_layers required for fraction-based layer spec: {spec}")
        fraction = float(layer_str)
        if fraction > 1.0:
            # It's a decimal layer number like 45.0, treat as integer
            layers = [int(float(layer_str))]
        else:
            # It's a fraction through the model
            layer_idx = int(fraction * (n_layers - 1))
            layers = [layer_idx]
    else:
        # Single layer number
        layers = [int(layer_str)]

    # Parse component type
    if remaining.startswith('MLP'):
        component = 'mlp'
        remaining = remaining[3:]
    elif remaining.startswith('ATTN'):
        component = 'attn'
        remaining = remaining[4:]
    elif remaining.startswith('RESID'):
        component = 'resid'
        remaining = remaining[5:]
    elif remaining.startswith('H'):
        # Head specification implies attention
        component = 'head'

    # Parse head specification
    if remaining.startswith('H'):
        component = 'head'
        head_match = re.match(r'^H(\d+(?:-\d+)?)', remaining)
        if head_match:
            head_str = head_match.group(1)
            if '-' in head_str:
                start, end = head_str.split('-')
                heads = list(range(int(start), int(end) + 1))
            else:
                heads = [int(head_str)]
        else:
            raise ValueError(f"Invalid head specification in: {spec}")

    return ComponentSpec(layers=layers, component=component, heads=heads)


def get_component_module(model: ModelWrapper, layer_idx: int, component: str):
    """
    Get the module to hook based on component type.

    Args:
        model: ModelWrapper instance
        layer_idx: Layer index
        component: Component type ('resid', 'mlp', 'attn', 'head')

    Returns:
        The module to hook
    """
    layer_module = model.get_layer_module(layer_idx)

    if component == 'resid':
        # Hook the entire layer (post-layer residual)
        return layer_module
    elif component == 'mlp':
        # Hook the MLP output
        if hasattr(layer_module, 'mlp'):
            return layer_module.mlp
        elif hasattr(layer_module, 'feed_forward'):
            return layer_module.feed_forward
        elif hasattr(layer_module, 'ffn'):
            return layer_module.ffn
        else:
            raise ValueError(f"Could not find MLP module in layer {layer_idx}")
    elif component in ('attn', 'head'):
        # Hook the attention output
        if hasattr(layer_module, 'self_attn'):
            return layer_module.self_attn
        elif hasattr(layer_module, 'attention'):
            return layer_module.attention
        elif hasattr(layer_module, 'attn'):
            return layer_module.attn
        else:
            raise ValueError(f"Could not find attention module in layer {layer_idx}")
    else:
        raise ValueError(f"Unknown component type: {component}")


def extract_activations_at_component(
    model: ModelWrapper,
    text: str,
    layer_idx: int,
    component: str = 'resid',
    token_idx: int = -1,
    heads: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Extract activations at a specific component and token position.

    Args:
        model: ModelWrapper instance
        text: Input text
        layer_idx: Layer index
        component: Component type ('resid', 'mlp', 'attn', 'head')
        token_idx: Token position (-1 for last token)
        heads: List of head indices (only for component='head')

    Returns:
        Activation tensor of shape [hidden_dim] or [n_heads, head_dim] for heads
    """
    activations = []

    def hook_fn(module, input, output):
        # Handle different output formats
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Extract at specific token position
        # hidden_states shape: [batch, seq, hidden_dim] or [batch, seq, n_heads, head_dim]
        if len(hidden_states.shape) == 3:
            act = hidden_states[0, token_idx, :].detach().clone()
        elif len(hidden_states.shape) == 4 and heads is not None:
            # Attention head output: [batch, seq, n_heads, head_dim]
            act = hidden_states[0, token_idx, heads, :].detach().clone()
        else:
            act = hidden_states[0, token_idx, :].detach().clone()

        activations.append(act)

    # Get the module to hook
    target_module = get_component_module(model, layer_idx, component)

    # Register hook
    hook = target_module.register_forward_hook(hook_fn)

    # Tokenize and run forward pass
    inputs = model.tokenizer(text, return_tensors="pt").to(model._get_input_device())

    with torch.no_grad():
        _ = model.model(**inputs, use_cache=False)

    # Clean up
    hook.remove()

    if not activations:
        raise RuntimeError(f"No activations captured for component {component} at layer {layer_idx}")

    return activations[0]


def cache_source_activations(
    model: ModelWrapper,
    source_texts: List[str],
    layer_idx: int,
    component: str = 'resid',
    token_idx: int = -1,
) -> Dict[str, torch.Tensor]:
    """
    Cache activations from source texts at specified component.

    Args:
        model: ModelWrapper instance
        source_texts: List of source texts
        layer_idx: Layer index
        component: Component type
        token_idx: Token position

    Returns:
        Dict mapping source_text -> activation tensor
    """
    cached = {}
    for text in source_texts:
        act = extract_activations_at_component(
            model=model,
            text=text,
            layer_idx=layer_idx,
            component=component,
            token_idx=token_idx,
        )
        cached[text] = act.cpu()
    return cached


def compute_baseline_activation(
    model: ModelWrapper,
    baseline_texts: List[str],
    layer_idx: int,
    component: str = 'resid',
    token_idx: int = -1,
) -> torch.Tensor:
    """
    Compute mean baseline activation from neutral texts.

    Args:
        model: ModelWrapper instance
        baseline_texts: List of neutral baseline texts
        layer_idx: Layer index
        component: Component type
        token_idx: Token position

    Returns:
        Mean activation tensor of shape [hidden_dim]
    """
    all_activations = []

    for text in baseline_texts:
        act = extract_activations_at_component(
            model=model,
            text=text,
            layer_idx=layer_idx,
            component=component,
            token_idx=token_idx,
        )
        all_activations.append(act.cpu())

    # Stack and compute mean
    stacked = torch.stack(all_activations, dim=0)
    baseline = stacked.mean(dim=0)

    return baseline


def create_patching_hook(
    source_activation: torch.Tensor,
    patch_position: int = -1,
):
    """
    Create a hook function that patches activations (direct replacement).

    Args:
        source_activation: Activation from source text [hidden_dim]
        patch_position: Token position to patch (-1 for last token)

    Returns:
        Hook function
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Move source activation to same device and dtype
        patch_act = source_activation.to(hidden_states.device).to(hidden_states.dtype)

        # Determine actual position to patch
        if patch_position < 0:
            actual_pos = seq_len + patch_position
        else:
            actual_pos = patch_position

        # Only patch if position is valid
        if 0 <= actual_pos < seq_len:
            modified = hidden_states.clone()
            # Direct replacement
            modified[:, actual_pos, :] = patch_act.view(1, -1)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return output

    return hook


def run_patched_generation(
    model: ModelWrapper,
    prompt: str,
    source_activation: torch.Tensor,
    layer_idx: int,
    component: str = 'resid',
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    patch_count: int = 1,
    patch_range_start: int = 0,
    patch_range_end: int = -1,
    patch_direction: str = "first",
    prefill_token_count: int = 0,
    debug: bool = False,
    return_patch_info: bool = False,
) -> str:
    """
    Generate text with activation patching applied (direct replacement).

    Patching is applied ONLY during prompt processing, not during generation.
    This allows the model to generate coherent text based on the patched prompt state.

    Args:
        model: ModelWrapper instance
        prompt: Input prompt
        source_activation: Activation from source text to patch in
        layer_idx: Layer to patch
        component: Component type
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        patch_count: Number of tokens to patch.
                     1 = 1 token
                     N = N tokens
                     -1 = all tokens in range
        patch_range_start: Token index where patchable range starts (inclusive)
        patch_range_end: Token index where patchable range ends (exclusive).
                         -1 means use total_tokens - prefill_token_count
        patch_direction: "first" patches first N tokens from patch_range_start,
                         "last" patches last N tokens before patch_range_end
        prefill_token_count: Number of tokens at the end of prompt that are prefill.
                             These tokens will NOT be patched to preserve the prefill context.
        return_patch_info: If True, return tuple of (text, patch_info_dict)

    Returns:
        Generated text, or (text, patch_info) if return_patch_info=True
    """
    # Source activation to patch in (direct replacement)
    patch_activation = source_activation.to(model.device).to(model.dtype)

    # Tokenize first to get input_length for calculations
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model._get_input_device())
    input_length = inputs['input_ids'].shape[1]

    # Calculate actual patch range end
    if patch_range_end == -1:
        actual_patch_range_end = input_length - prefill_token_count
    else:
        actual_patch_range_end = min(patch_range_end, input_length - prefill_token_count)

    # Calculate the actual tokens to patch
    range_length = actual_patch_range_end - patch_range_start
    if range_length <= 0:
        # No tokens to patch
        actual_patch_start = 0
        actual_patch_end = 0
        actual_patch_count = 0
    elif patch_count == -1 or patch_count >= range_length:
        # Patch ALL tokens in the range
        actual_patch_start = patch_range_start
        actual_patch_end = actual_patch_range_end
        actual_patch_count = range_length
    else:
        if patch_direction == "first":
            # Patch first N tokens starting from patch_range_start
            actual_patch_start = patch_range_start
            actual_patch_end = patch_range_start + patch_count
        else:  # "last"
            # Patch last N tokens ending at patch_range_end
            actual_patch_start = actual_patch_range_end - patch_count
            actual_patch_end = actual_patch_range_end
        actual_patch_count = patch_count

    def patching_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Move patch activation to same device and dtype
        patch_act = patch_activation.to(hidden_states.device).to(hidden_states.dtype)

        # During generation with KV cache, seq_len will be 1
        if seq_len == 1:
            # Generation phase: DO NOT patch during generation
            return output

        if actual_patch_count <= 0:
            return output

        modified = hidden_states.clone()
        # Patch from actual_patch_start to actual_patch_end
        modified[:, actual_patch_start:actual_patch_end, :] = patch_act.view(1, 1, -1).expand(
            batch_size, actual_patch_count, -1
        )

        if isinstance(output, tuple):
            return (modified,) + rest
        return modified

    # Get module to hook
    target_module = get_component_module(model, layer_idx, component)

    # Register hook
    hook = target_module.register_forward_hook(patching_hook)
    model.hooks.append(hook)

    # Get the patched token IDs and decode them
    patched_token_ids = inputs['input_ids'][0][actual_patch_start:actual_patch_end].tolist()
    patched_tokens = [model.tokenizer.decode([tid]) for tid in patched_token_ids]

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": model.tokenizer.pad_token_id,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    # Disable cache for models with compatibility issues
    if model.model_name in ["kimi_k2", "deepseek_v3"]:
        gen_kwargs["use_cache"] = False

    # Generate
    with torch.no_grad():
        output_ids = model.model.generate(**inputs, **gen_kwargs)

    # Clean up hook
    hook.remove()
    model.hooks.remove(hook)

    # Decode
    new_tokens = output_ids[0][input_length:]

    if debug:
        # Show raw output without skipping special tokens
        raw_output = model.tokenizer.decode(new_tokens, skip_special_tokens=False)
        print(f"\n[DEBUG] Input length: {input_length} tokens")
        print(f"[DEBUG] Output length: {len(output_ids[0])} tokens")
        print(f"[DEBUG] New tokens: {len(new_tokens)} tokens")
        print(f"[DEBUG] Raw output (with special tokens):\n{repr(raw_output)}")

    if model.model_name in ["kimi_k2", "deepseek_v3"]:
        output_text = model.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
    else:
        output_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Fix for Gemma models
    gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
    if model.model_name in gemma_models and output_text.startswith("model\n"):
        output_text = output_text[len("model\n"):]

    if debug:
        print(f"[DEBUG] Final output:\n{repr(output_text.strip())}")

    if return_patch_info:
        patch_info = {
            "num_tokens_patched": actual_patch_count,
            "patched_tokens": patched_tokens,
            "patch_start_idx": actual_patch_start,
            "patch_end_idx": actual_patch_end,
            "total_prompt_tokens": input_length,
            "patch_range_start": patch_range_start,
            "patch_range_end": actual_patch_range_end,
            "patch_direction": patch_direction,
        }
        return output_text.strip(), patch_info

    return output_text.strip()


def run_patched_generation_batch(
    model: ModelWrapper,
    prompts: List[str],
    source_activations: List[torch.Tensor],
    layer_idx: int,
    component: str = 'resid',
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    patch_count: int = 1,
) -> List[str]:
    """
    Generate text for multiple prompts with different source activations.

    This processes prompts sequentially since each has a different source activation.
    For better batching, consider grouping by source activation.

    Args:
        model: ModelWrapper instance
        prompts: List of input prompts
        source_activations: List of source activations (one per prompt)
        layer_idx: Layer to patch
        component: Component type
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        patch_count: Number of tokens to patch from end (1=last, N=last N, -1=all)

    Returns:
        List of generated texts
    """
    responses = []

    for prompt, source_act in zip(prompts, source_activations):
        response = run_patched_generation(
            model=model,
            prompt=prompt,
            source_activation=source_act,
            layer_idx=layer_idx,
            component=component,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            patch_count=patch_count,
        )
        responses.append(response)

    return responses


# Default baseline texts for computing baseline activations
BASELINE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please provide information about the following topic.",
    "Here is some text for you to process.",
    "This is a neutral sentence with no particular meaning.",
    "Consider the following information carefully.",
    "The weather today is mild and pleasant.",
    "Numbers one through ten are fundamental.",
    "Please read the following message.",
    "This text serves as a baseline example.",
    "Simple sentences help establish context.",
    "Words arranged in meaningful patterns.",
    "Communication requires shared understanding.",
    "Language connects ideas across time.",
    "Every sentence has structure and meaning.",
    "Thoughts expressed through written words.",
    "Reading comprehension is a valuable skill.",
    "Text processing involves many steps.",
    "Information flows through various channels.",
    "Context shapes the interpretation of meaning.",
    "Clear writing aids understanding.",
    "Sentences build upon one another.",
    "Ideas connect in complex networks.",
    "Understanding emerges from careful attention.",
    "Words carry both denotation and connotation.",
    "Syntax governs the structure of sentences.",
    "Semantics concerns the meaning of language.",
    "Pragmatics addresses language in context.",
    "Discourse analysis examines extended text.",
    "Linguistics studies the nature of language.",
    "Communication is fundamental to society.",
    "Knowledge is transmitted through language.",
    "Learning involves processing new information.",
    "Memory stores experiences and facts.",
    "Cognition encompasses many mental processes.",
    "Perception shapes our understanding of reality.",
    "Attention focuses cognitive resources.",
    "Reasoning connects premises to conclusions.",
    "Decision making involves weighing options.",
    "Problem solving requires creative thinking.",
    "Intelligence manifests in many forms.",
    "Artificial systems can process information.",
    "Algorithms define computational procedures.",
    "Data structures organize information efficiently.",
    "Programs execute sequences of instructions.",
    "Computers perform calculations rapidly.",
    "Networks enable distributed communication.",
    "Software mediates human-computer interaction.",
    "Hardware provides the physical substrate.",
    "Systems integrate components into wholes.",
    "Engineering applies scientific knowledge.",
    "Design balances constraints and goals.",
    "Innovation creates new possibilities.",
    "Research advances human understanding.",
    "Science seeks to explain natural phenomena.",
    "Mathematics provides formal languages.",
    "Logic ensures valid reasoning.",
    "Philosophy examines fundamental questions.",
    "Ethics considers right and wrong.",
    "Aesthetics explores beauty and art.",
    "History records human experiences.",
    "Culture shapes beliefs and practices.",
    "Society organizes collective life.",
    "Economics studies resource allocation.",
    "Politics concerns power and governance.",
    "Law establishes rules for society.",
    "Education transmits knowledge and skills.",
    "Medicine promotes health and healing.",
    "Technology extends human capabilities.",
    "Nature exhibits remarkable diversity.",
    "Evolution shapes living organisms.",
    "Ecology studies environmental interactions.",
    "Chemistry explores molecular composition.",
    "Physics describes fundamental forces.",
    "Astronomy investigates celestial bodies.",
    "Geology examines Earth's structure.",
    "Biology studies living systems.",
    "Psychology explores mental processes.",
    "Sociology analyzes social phenomena.",
    "Anthropology studies human cultures.",
    "Geography describes spatial patterns.",
    "Meteorology forecasts weather conditions.",
    "Oceanography studies marine environments.",
    "Botany classifies plant species.",
    "Zoology examines animal life.",
    "Genetics investigates heredity.",
    "Microbiology studies microscopic organisms.",
    "Neuroscience explores brain function.",
    "Immunology investigates immune responses.",
    "Pharmacology studies drug effects.",
    "Pathology examines disease processes.",
    "Epidemiology tracks disease patterns.",
    "Statistics analyzes numerical data.",
    "Probability quantifies uncertainty.",
    "Calculus models continuous change.",
    "Algebra manipulates symbolic expressions.",
    "Geometry studies spatial relationships.",
    "Topology examines connectivity properties.",
    "Analysis explores limits and continuity.",
    "Number theory investigates integer properties.",
    "Graph theory models networked structures.",
    "Cryptography secures information.",
]


def get_layer_at_fraction(model: ModelWrapper, fraction: float) -> int:
    """
    Get the layer index at a given fraction through the model.

    Args:
        model: ModelWrapper instance
        fraction: Fraction through model (0.0 to 1.0)

    Returns:
        Layer index
    """
    return int(fraction * (model.n_layers - 1))
