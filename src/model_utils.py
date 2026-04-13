"""
Utilities for loading models and extracting activations.

This module handles:
- Model loading for different architectures (DeepSeek, Llama, Qwen, etc.)
- Activation extraction at specific layers
- Forward passes with hooks for intervention
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, List, Tuple, Callable
import gc
from pathlib import Path


# Model name mappings to HuggingFace identifiers
MODEL_NAME_MAP = {
    # DeepSeek models
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "deepseek_v2.5": "deepseek-ai/DeepSeek-V2.5",
    "deepseek_v2": "deepseek-ai/DeepSeek-V2",

    # Llama models
    "llama_405b": "meta-llama/Llama-3.1-405B-Instruct",
    "llama_70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_3_3_70b": "meta-llama/Llama-3.3-70B-Instruct",

    # Qwen models
    "qwen3_235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",  # MoE: 235B total, 22B activated
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",

    # Moonshot AI models
    "kimi_k2": "moonshotai/Kimi-K2-Instruct-0905",

    # Gemma models (Google)
    "gemma2_2b": "google/gemma-2-2b-it",
    "gemma2_9b": "google/gemma-2-9b-it",
    "gemma2_27b": "google/gemma-2-27b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    "gemma3_27b_pt": "google/gemma-3-27b-pt",  # Base (pre-trained) model
    "gemma3_27b_abliterated": "uzaymacar/gemma-3-27b-abliterated",

    # Add more models as needed
    "mistral_small": "mistralai/Mistral-Small-Instruct-2409",

    # OLMo models (Allen AI) - different training stages
    "olmo_base": "allenai/Olmo-3-1125-32B",  # Base (pre-trained) model
    "olmo_sft": "allenai/Olmo-3.1-32B-Instruct-SFT",  # SFT stage
    "olmo_dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",  # DPO stage
    "olmo_instruct": "allenai/Olmo-3.1-32B-Instruct",  # Final instruct model
}

# Models that come pre-quantized (skip BitsAndBytes quantization)
PRE_QUANTIZED_MODELS = {
    "kimi_k2",  # Uses FP8 quantization
    "deepseek_v3",  # Uses FineGrainedFP8 quantization
}


class ModelWrapper:
    """
    Wrapper class for handling model loading, activation extraction, and generation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ):
        """
        Initialize model wrapper.

        Args:
            model_name: Model identifier (key from MODEL_NAME_MAP or HF path)
            device: Device to load model on
            dtype: Data type for model weights
            quantization_config: BitsAndBytesConfig for quantization (8bit or 4bit)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Get HuggingFace model path
        self.hf_path = MODEL_NAME_MAP.get(model_name, model_name)

        # Infer model type from model name (for architecture-specific code)
        if "llama" in model_name.lower():
            self.model_type = "llama"
        elif "qwen" in model_name.lower():
            self.model_type = "qwen"
        elif "gemma" in model_name.lower():
            self.model_type = "gemma"
        elif "deepseek" in model_name.lower():
            self.model_type = "deepseek"
        elif "mistral" in model_name.lower():
            self.model_type = "mistral"
        elif "olmo" in model_name.lower():
            self.model_type = "olmo"
        elif "gpt" in model_name.lower():
            self.model_type = "gpt"
        else:
            self.model_type = "unknown"

        print(f"Loading model: {self.hf_path}")
        if model_name in PRE_QUANTIZED_MODELS:
            print(f"Device: {device}, dtype: {dtype} (note: model is pre-quantized with FP8, dtype ignored)")
        else:
            print(f"Device: {device}, dtype: {dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_path,
            trust_remote_code=True,
        )

        # Set padding side to left for decoder-only models (required for batch generation)
        self.tokenizer.padding_side = 'left'

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": self.hf_path,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
            "attn_implementation": "eager",  # Required for output_attentions=True
            # "force_download": True,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            # All models now use 'dtype' instead of deprecated 'torch_dtype'
            load_kwargs["dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        except Exception as e:
            load_kwargs["torch_dtype"] = dtype
            del load_kwargs["dtype"]
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if device != "cuda":
            self.model = self.model.to(device)
            
        # Initialize ablation for abliterated models
        if "abliterated" in model_name:
            try:
                self.model.init_ablation()
                print(f"Initialized ablation for {model_name}")
            except Exception as e:
                print(f"Warning: Could not initialize ablation for {model_name}: {e}")

        self.model.eval()

        # Apply model-specific patches
        self._apply_model_patches()

        # Get number of layers
        self.n_layers = self._get_n_layers()
        print(f"Model loaded. Total layers: {self.n_layers}")

        # Store hooks for cleanup
        self.hooks = []

    def clear_hooks(self):
        """Remove all registered hooks without deleting the model.

        Call this to ensure clean model state before generation.
        This is important to prevent residual state from previous steering runs.
        """
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass  # Hook may already be removed
        self.hooks = []

    def _get_input_device(self):
        """Get the device where inputs should be placed (handles device_map='auto')."""
        # When using device_map="auto", model may be sharded across GPUs
        # Use the device of the first model parameter (embedding layer)
        return next(self.model.parameters()).device

    def _apply_model_patches(self):
        """Apply model-specific patches for compatibility."""
        # Patch for Kimi and DeepSeek models: Add multiple cache compatibility methods to DynamicCache
        if self.model_name in ["kimi_k2", "deepseek_v3"]:
            patches_applied = False

            # The Kimi model's custom code expects cache.seen_tokens
            # but newer Transformers uses DynamicCache.get_seq_length()
            if not hasattr(DynamicCache, 'seen_tokens'):
                def seen_tokens_getter(self):
                    """
                    Compatibility property for older model code.
                    Returns the total number of tokens seen (cache length).
                    """
                    try:
                        # Get cache length from first layer
                        return self.get_seq_length(layer_idx=0)
                    except:
                        # Fallback: cache not initialized yet
                        return 0

                def seen_tokens_setter(self, value):
                    """Setter for seen_tokens (for compatibility, does nothing)."""
                    pass

                # Add the property to DynamicCache class with both getter and setter
                DynamicCache.seen_tokens = property(seen_tokens_getter, seen_tokens_setter)
                patches_applied = True

            # Add get_max_length method which the Kimi model expects
            if not hasattr(DynamicCache, 'get_max_length'):
                def get_max_length_method(self):
                    """
                    Compatibility method - returns max sequence length.
                    For DynamicCache, this is just the current sequence length.
                    """
                    try:
                        # Return length from layer 0 as representative
                        return self.get_seq_length(layer_idx=0)
                    except:
                        return 0

                # Add the method to DynamicCache class
                DynamicCache.get_max_length = get_max_length_method
                patches_applied = True

            # Add get_usable_length method which the Kimi model also expects
            if not hasattr(DynamicCache, 'get_usable_length'):
                def get_usable_length_method(self, seq_length, layer_idx=0):
                    """
                    Compatibility method - returns the usable cache length.

                    Args:
                        seq_length: Current total sequence length (prompt + cache + new tokens)
                        layer_idx: Layer index

                    Returns:
                        Number of usable cache positions for this layer
                    """
                    # Return the actual cache length for this specific layer
                    try:
                        cache_length = self.get_seq_length(layer_idx)
                        return cache_length
                    except:
                        # Fallback if cache not initialized for this layer
                        return 0

                # Add the method to DynamicCache class
                DynamicCache.get_usable_length = get_usable_length_method
                patches_applied = True

            if patches_applied:
                print(f"Applied cache compatibility patches for {self.model_name}")

        # Patch for Gemma models: Fix rotary embedding dimension mismatch
        # This addresses a bug where cos/sin tensors don't match q/k dimensions in GQA models
        gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
        if self.model_name in gemma_models:
            # Determine which module to patch based on model type
            if "gemma3" in self.model_name:
                from transformers.models.gemma3 import modeling_gemma3 as gemma_module
            else:
                from transformers.models.gemma2 import modeling_gemma2 as gemma_module

            # Save original function
            original_apply_rotary = gemma_module.apply_rotary_pos_emb

            def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                """Fixed version that handles dimension mismatches."""
                cos = cos.unsqueeze(unsqueeze_dim)
                sin = sin.unsqueeze(unsqueeze_dim)

                # Fix dimension mismatch: slice cos/sin if they don't match q/k's last dimension
                # This handles GQA where query has more heads than key/value
                if cos.shape[-1] != q.shape[-1]:
                    cos = cos[..., :q.shape[-1]]
                    sin = sin[..., :q.shape[-1]]

                q_embed = (q * cos) + (gemma_module.rotate_half(q) * sin)
                k_embed = (k * cos) + (gemma_module.rotate_half(k) * sin)
                return q_embed, k_embed

            # Monkey-patch the function
            gemma_module.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb
            print(f"Applied rotary embedding fix for {self.model_name}")

    def _get_n_layers(self) -> int:
        """Get number of transformer layers in model."""
        if hasattr(self.model, 'model'):
            # Multimodal models (e.g., Gemma 3) have model.language_model.layers
            # IMPORTANT: Check language_model FIRST, as model.model.layers may exist but be wrong!
            if hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
                return len(self.model.model.language_model.layers)
            # Standard decoder-only models have model.layers
            elif hasattr(self.model.model, 'layers'):
                return len(self.model.model.layers)

        # Fallback: try to infer from config
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        elif hasattr(self.model.config, 'num_layers'):
            return self.model.config.num_layers
        # Check for nested text_config (e.g., Gemma 3, multimodal models)
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'num_hidden_layers'):
            return self.model.config.text_config.num_hidden_layers

        raise ValueError(f"Could not determine number of layers for {self.model_name}")

    @property
    def d_model(self) -> int:
        """Get hidden dimension (d_model) of the model."""
        # Try standard config attributes
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            return self.model.config.d_model
        elif hasattr(self.model.config, 'dim'):
            return self.model.config.dim
        elif hasattr(self.model.config, 'n_embd'):
            return self.model.config.n_embd
        # Check for nested text_config (e.g., Gemma 3, multimodal models)
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
            return self.model.config.text_config.hidden_size

        raise ValueError(f"Could not determine hidden dimension for {self.model_name}")

    @property
    def n_heads(self) -> int:
        """Get number of attention heads per layer."""
        # Try standard config attributes
        if hasattr(self.model.config, 'num_attention_heads'):
            return self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            return self.model.config.n_head
        elif hasattr(self.model.config, 'num_heads'):
            return self.model.config.num_heads
        # Check for nested text_config (e.g., Gemma 3, multimodal models)
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'num_attention_heads'):
            return self.model.config.text_config.num_attention_heads

        raise ValueError(f"Could not determine number of attention heads for {self.model_name}")

    def get_layer_module(self, layer_idx: int):
        """
        Get the transformer layer module at the specified index.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            The layer module
        """
        if hasattr(self.model, 'model'):
            # Multimodal models (e.g., Gemma 3) with separate language_model
            # IMPORTANT: Check language_model FIRST, as model.model.layers may exist but be wrong!
            if hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
                return self.model.model.language_model.layers[layer_idx]
            # Standard decoder-only models
            elif hasattr(self.model.model, 'layers'):
                return self.model.model.layers[layer_idx]

        raise ValueError(f"Could not access layer {layer_idx} for {self.model_name}")

    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        token_idx: int = -1,
    ) -> torch.Tensor:
        """
        Extract activations at a specific layer and token position.

        Args:
            prompts: List of text prompts
            layer_idx: Layer index to extract from
            token_idx: Token position to extract (-1 for last token)

        Returns:
            Tensor of shape [batch_size, hidden_dim] with activations
        """
        activations = []

        def hook_fn(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Extract at specific token position
            act = hidden_states[:, token_idx, :].detach().cpu()
            activations.append(act)

        # Register hook
        layer_module = self.get_layer_module(layer_idx)
        hook = layer_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)

        # Tokenize
        # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self._get_input_device())

        # Forward pass (disable cache to avoid compatibility issues with some models)
        with torch.no_grad():
            _ = self.model(**inputs, use_cache=False)

        # Clean up hook
        hook.remove()
        self.hooks.remove(hook)

        # Stack activations
        return torch.cat(activations, dim=0)

    def generate_with_steering(
        self,
        prompt: str,
        layer_idx: int,
        steering_vector: torch.Tensor,
        strength: float = 1.0,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        steering_start_pos: Optional[int] = None,
        strip_bos_from_prompt: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Generate text with activation steering applied at specified layer.

        Args:
            prompt: Input prompt
            layer_idx: Layer to apply steering at
            steering_vector: Vector to add to activations (shape: [hidden_dim])
            strength: Multiplier for steering vector
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            steering_start_pos: Token position to start steering from (None = all positions)
            strip_bos_from_prompt: If True, strip leading BOS token from prompt.
                Deprecated: No longer needed since we use add_special_tokens=False.
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # NOTE: strip_bos_from_prompt is deprecated since we now use add_special_tokens=False
        # Keeping for backward compatibility but should be False by default
        if strip_bos_from_prompt:
            bos_token = getattr(self.tokenizer, 'bos_token', None)
            if bos_token and prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]

        # Prepare steering vector
        steering_vec = (steering_vector * strength).to(self.device).to(self.dtype)

        # Validate steering vector dimension by checking the layer's expected hidden dimension
        # We can't check exact dimension until we see the actual hidden states, but we can
        # at least ensure the steering vector is 1D
        if steering_vec.dim() != 1:
            raise ValueError(f"Steering vector must be 1D, got shape {steering_vec.shape}")

        # Track total tokens processed for KV-cached generation
        # This is needed because during generation with KV cache, seq_len=1 for each new token
        # but steering_start_pos is an absolute position in the full sequence
        tokens_processed = [0]  # Use list to allow mutation in closure

        def steering_hook(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
                batch_size, seq_len, hidden_dim = hidden_states.shape

                # Move steering vector to the same device as hidden_states (for multi-GPU)
                steering_vec_device = steering_vec.to(hidden_states.device)

                # Validate dimension compatibility
                steering_dim = steering_vec_device.shape[0]
                if steering_dim != hidden_dim:
                    raise ValueError(
                        f"Steering vector dimension ({steering_dim}) does not match "
                        f"layer hidden dimension ({hidden_dim}) at layer {layer_idx}. "
                        f"This usually means the steering vector was computed for a different "
                        f"layer or model configuration."
                    )

                # Calculate absolute positions for this forward pass
                # tokens_processed[0] is the count BEFORE this forward pass
                start_abs_pos = tokens_processed[0]
                end_abs_pos = start_abs_pos + seq_len

                # Update tokens processed for next forward pass
                tokens_processed[0] = end_abs_pos

                if steering_start_pos is None:
                    # Apply to all positions
                    modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                    return (modified_hidden_states,) + output[1:]
                elif steering_start_pos >= end_abs_pos:
                    # steering_start_pos is beyond current sequence, no steering yet
                    return output
                elif steering_start_pos <= start_abs_pos:
                    # steering_start_pos is before this chunk, apply to all positions
                    modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                    return (modified_hidden_states,) + output[1:]
                else:
                    # steering_start_pos is within this chunk
                    # Calculate relative position within current chunk
                    relative_start = steering_start_pos - start_abs_pos
                    modified_hidden_states = hidden_states.clone()
                    modified_hidden_states[:, relative_start:, :] += steering_vec_device.view(1, 1, -1)
                    return (modified_hidden_states,) + output[1:]
            else:
                # Fallback for non-tuple output
                # Move steering vector to the same device as output (for multi-GPU)
                steering_vec_device = steering_vec.to(output.device)

                # Update tokens processed (assume seq_len from output shape)
                if output.dim() == 3:
                    tokens_processed[0] += output.shape[1]

                if steering_start_pos is None:
                    return output + steering_vec_device.view(1, 1, -1)
                else:
                    # For non-tuple, apply steering if we're past steering_start_pos
                    # This is a simplification - assumes we're in generation phase
                    if tokens_processed[0] > steering_start_pos:
                        return output + steering_vec_device.view(1, 1, -1)
                    return output

        # Register hook
        layer_module = self.get_layer_module(layer_idx)
        hook = layer_module.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        try:
            # Tokenize
            # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._get_input_device())
            input_length = inputs['input_ids'].shape[1]

            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            # Add sampling parameters only if temperature > 0
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                # Only add additional kwargs when sampling (to avoid passing sampling params to greedy)
                for k, v in generation_kwargs.items():
                    gen_kwargs[k] = v
            # For greedy decoding (temperature <= 0), don't pass do_sample or any sampling params

            # Disable cache for models with cache compatibility issues
            # Note: DeepSeek V3 has attention mask tracking issues when using steering hooks + cache
            # The cache works fine for regular generation, but not with activation steering
            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                gen_kwargs["use_cache"] = False

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][input_length:]
        finally:
            # ALWAYS clean up hook, even if exception occurs
            hook.remove()
            if hook in self.hooks:
                self.hooks.remove(hook)
        # Convert to list for tokenizers that expect list (not tensor)
        if self.model_name in ["kimi_k2", "deepseek_v3"]:
            output_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Fix for Gemma models: chat template includes "model\n" as regular tokens
        # These don't get removed by skip_special_tokens, so we strip them manually
        gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
        if self.model_name in gemma_models and output_text.startswith("model\n"):
            output_text = output_text[len("model\n"):]

        return output_text.strip()

    def generate_with_multi_layer_steering(
        self,
        prompt: str,
        layer_directions: Dict[int, Tuple[torch.Tensor, float]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        steering_start_pos: Optional[int] = None,
        strip_bos_from_prompt: bool = True,
        **generation_kwargs,
    ) -> str:
        """
        Generate text with activation steering applied at multiple layers.

        This is more principled than applying a single direction across multiple layers,
        as each layer can have its own direction extracted from that layer's activations.

        Args:
            prompt: Input prompt
            layer_directions: Dict mapping layer_idx -> (steering_vector, strength)
                             Each layer gets its own direction and magnitude
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            steering_start_pos: Token position to start steering from (None = all positions)
            strip_bos_from_prompt: If True (default), strip leading BOS token from prompt
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Strip leading BOS token if present
        if strip_bos_from_prompt:
            bos_token = getattr(self.tokenizer, 'bos_token', None)
            if bos_token and prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]

        # Prepare steering vectors for each layer
        prepared_vectors = {}
        for layer_idx, (steering_vector, strength) in layer_directions.items():
            steering_vec = (steering_vector * strength).to(self.device).to(self.dtype)
            if steering_vec.dim() != 1:
                raise ValueError(f"Steering vector for layer {layer_idx} must be 1D, got shape {steering_vec.shape}")
            prepared_vectors[layer_idx] = steering_vec

        # Track tokens processed per layer (each layer needs its own counter)
        tokens_processed = {layer_idx: [0] for layer_idx in layer_directions.keys()}

        def make_steering_hook(layer_idx: int, steering_vec: torch.Tensor):
            """Create a steering hook for a specific layer."""
            counter = tokens_processed[layer_idx]

            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    batch_size, seq_len, hidden_dim = hidden_states.shape

                    steering_vec_device = steering_vec.to(hidden_states.device)

                    if steering_vec_device.shape[0] != hidden_dim:
                        raise ValueError(
                            f"Steering vector dimension ({steering_vec_device.shape[0]}) does not match "
                            f"layer hidden dimension ({hidden_dim}) at layer {layer_idx}."
                        )

                    start_abs_pos = counter[0]
                    end_abs_pos = start_abs_pos + seq_len
                    counter[0] = end_abs_pos

                    if steering_start_pos is None:
                        modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                        return (modified_hidden_states,) + output[1:]
                    elif steering_start_pos >= end_abs_pos:
                        return output
                    elif steering_start_pos <= start_abs_pos:
                        modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                        return (modified_hidden_states,) + output[1:]
                    else:
                        relative_start = steering_start_pos - start_abs_pos
                        modified_hidden_states = hidden_states.clone()
                        modified_hidden_states[:, relative_start:, :] += steering_vec_device.view(1, 1, -1)
                        return (modified_hidden_states,) + output[1:]
                else:
                    steering_vec_device = steering_vec.to(output.device)
                    if output.dim() == 3:
                        counter[0] += output.shape[1]
                    if steering_start_pos is None or counter[0] > steering_start_pos:
                        return output + steering_vec_device.view(1, 1, -1)
                    return output

            return steering_hook

        # Register hooks for all layers
        hooks = []
        for layer_idx, steering_vec in prepared_vectors.items():
            layer_module = self.get_layer_module(layer_idx)
            hook = layer_module.register_forward_hook(make_steering_hook(layer_idx, steering_vec))
            hooks.append(hook)
            self.hooks.append(hook)

        try:
            # Tokenize
            # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._get_input_device())
            input_length = inputs['input_ids'].shape[1]

            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                for k, v in generation_kwargs.items():
                    gen_kwargs[k] = v

            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                gen_kwargs["use_cache"] = False

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            # Decode
            new_tokens = output_ids[0][input_length:]
        finally:
            # ALWAYS clean up hooks, even if exception occurs
            for hook in hooks:
                hook.remove()
                if hook in self.hooks:
                    self.hooks.remove(hook)
        if self.model_name in ["kimi_k2", "deepseek_v3"]:
            output_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
        if self.model_name in gemma_models and output_text.startswith("model\n"):
            output_text = output_text[len("model\n"):]

        return output_text.strip()

    def generate_with_activations(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        steering_vector: Optional[torch.Tensor] = None,
        steering_layer: Optional[int] = None,
        steering_strength: float = 1.0,
        steering_start_pos: Optional[int] = None,
        return_logits: bool = True,
        **generation_kwargs,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate text with optional steering and return both text and logits.

        Used for attribution patching where we need gradients through the generation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            steering_vector: Optional vector to add to activations
            steering_layer: Layer to apply steering at (required if steering_vector provided)
            steering_strength: Multiplier for steering vector
            steering_start_pos: Token position to start steering from (None = all positions)
            return_logits: Whether to return logits

        Returns:
            (generated_text, logits) where logits is the final token logits if return_logits=True
        """
        # Tokenize input
        # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Setup steering hook if needed
        hook = None
        if steering_vector is not None and steering_layer is not None:
            steering_vec = (steering_vector * steering_strength).to(self.device).to(self.dtype)

            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    steering_vec_device = steering_vec.to(hidden_states.device)

                    if steering_start_pos is not None and steering_start_pos < seq_len:
                        # Only apply steering from steering_start_pos onwards
                        modified_hidden_states = hidden_states.clone()
                        modified_hidden_states[:, steering_start_pos:, :] += steering_vec_device.view(1, 1, -1)
                        return (modified_hidden_states,) + output[1:]
                    elif steering_start_pos is None:
                        # Apply to all positions
                        modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                        return (modified_hidden_states,) + output[1:]
                    else:
                        # steering_start_pos is beyond current sequence, no steering yet
                        return output
                return output

            layer_module = self.get_layer_module(steering_layer)
            hook = layer_module.register_forward_hook(steering_hook)
            self.hooks.append(hook)

        # Generate with model.generate() to get output_ids
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **generation_kwargs,
        }

        output_ids = self.model.generate(**inputs, **gen_config)

        # Decode the generated text
        new_tokens = output_ids[0][input_length:]
        if self.model_name in ["kimi_k2", "deepseek_v3"]:
            output_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Fix for Gemma models: chat template includes "model\n" as regular tokens
        # These don't get removed by skip_special_tokens, so we strip them manually
        gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
        if self.model_name in gemma_models and output_text.startswith("model\n"):
            output_text = output_text[len("model\n"):]

        # Get logits if requested (do a forward pass on the full sequence)
        # Note: Don't use no_grad() here so gradients can flow for attribution patching
        logits = None
        if return_logits:
            outputs = self.model(output_ids, use_cache=False)
            logits = outputs.logits[:, -1, :]  # Get logits for last token

        # Clean up hook
        if hook is not None:
            hook.remove()
            self.hooks.remove(hook)

        return output_text.strip(), logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **generation_kwargs,
    ) -> str:
        """
        Generate text without steering (standard generation).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._get_input_device())
        input_length = inputs['input_ids'].shape[1]

        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add sampling parameters only if temperature > 0
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            # Only add additional kwargs when sampling (to avoid passing sampling params to greedy)
            for k, v in generation_kwargs.items():
                gen_kwargs[k] = v
        # For greedy decoding (temperature <= 0), don't pass do_sample or any sampling params

        # Note: Regular generation (without hooks) can use cache for all models including DeepSeek V3
        # Cache is only disabled for steering methods due to attention mask tracking issues

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_length:]
        # Convert to list for tokenizers that expect list (not tensor)
        if self.model_name in ["kimi_k2", "deepseek_v3"]:
            output_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Fix for Gemma models: chat template includes "model\n" as regular tokens
        # These don't get removed by skip_special_tokens, so we strip them manually
        gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
        if self.model_name in gemma_models and output_text.startswith("model\n"):
            output_text = output_text[len("model\n"):]

        return output_text.strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **generation_kwargs,
    ) -> List[str]:
        """
        Generate text for a batch of prompts without steering.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self._get_input_device())
        input_lengths = (inputs['attention_mask'].sum(dim=1)).tolist()

        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add sampling parameters only if temperature > 0
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            for k, v in generation_kwargs.items():
                gen_kwargs[k] = v

        # Note: Regular batch generation (without hooks) can use cache for all models including DeepSeek V3
        # Cache is only disabled for steering methods due to attention mask tracking issues

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens for each sample
        outputs = []
        for i, input_length in enumerate(input_lengths):
            new_tokens = output_ids[i][input_length:]
            # Convert to list for tokenizers that expect list (not tensor)
            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                output_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
            else:
                output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Fix for Gemma models: chat template includes "model\n" as regular tokens
            # These don't get removed by skip_special_tokens, so we strip them manually
            gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
            if self.model_name in gemma_models and output_text.startswith("model\n"):
                output_text = output_text[len("model\n"):]

            outputs.append(output_text.strip())

        return outputs

    def generate_batch_with_steering(
        self,
        prompts: List[str],
        layer_idx: int,
        steering_vector: torch.Tensor,
        strength: float = 1.0,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        steering_start_pos: Optional[int] = None,
        **generation_kwargs,
    ) -> List[str]:
        """
        Generate text for a batch of prompts with the same steering vector.

        Args:
            prompts: List of input prompts
            layer_idx: Layer to apply steering at
            steering_vector: Vector to add to activations (same for all prompts)
            strength: Multiplier for steering vector
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            steering_start_pos: Token position to start steering from
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Prepare steering vector
        steering_vec = (steering_vector * strength).to(self.device).to(self.dtype)

        def steering_hook(module, input, output):
            # Handle both tuple and non-tuple outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_of_output = output[1:]
            else:
                hidden_states = output
                rest_of_output = ()

            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Move steering vector to the same device as hidden_states (for multi-GPU)
            steering_vec_device = steering_vec.to(hidden_states.device)

            if steering_start_pos is not None:
                if seq_len == 1:
                    # Generation phase: no clone needed, add to all positions
                    modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                elif steering_start_pos < seq_len:
                    # Prompt processing phase: clone and apply from start_pos onwards
                    modified_hidden_states = hidden_states.clone()
                    modified_hidden_states[:, steering_start_pos:, :] += steering_vec_device.view(1, 1, -1)
                else:
                    return output  # Nothing to steer

                if isinstance(output, tuple):
                    return (modified_hidden_states,) + rest_of_output
                else:
                    return modified_hidden_states
            else:
                # Apply to all positions
                modified_hidden_states = hidden_states + steering_vec_device.view(1, 1, -1)
                if rest_of_output:
                    return (modified_hidden_states,) + rest_of_output
                else:
                    return modified_hidden_states

        # Register hook
        layer_module = self.get_layer_module(layer_idx)
        hook = layer_module.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        try:
            # Tokenize with padding
            # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self._get_input_device())
            input_lengths = (inputs['attention_mask'].sum(dim=1)).tolist()

            # Adjust steering positions for left padding
            # When using left padding, shorter prompts get padding tokens added to the left,
            # which shifts all token indices. We need to adjust steering positions accordingly.
            # Ensure steering_pos_tensor is defined, set to None if not.
            if 'steering_pos_tensor' not in locals():
                steering_pos_tensor = None
            if steering_pos_tensor is not None and self.tokenizer.padding_side == "left":
                max_length = inputs['input_ids'].shape[1]
                padding_amounts = [max_length - length for length in input_lengths]
                steering_pos_tensor = steering_pos_tensor + torch.tensor(padding_amounts, device=self.device)

            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                for k, v in generation_kwargs.items():
                    gen_kwargs[k] = v

            # Disable cache for models with cache compatibility issues
            # Note: DeepSeek V3 has attention mask tracking issues when using steering hooks + cache
            # The cache works fine for regular generation, but not with activation steering
            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                gen_kwargs["use_cache"] = False

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            # ALWAYS clean up hook, even if exception occurs
            hook.remove()
            if hook in self.hooks:
                self.hooks.remove(hook)

        # Decode only the newly generated tokens for each sample
        outputs = []
        for i, input_length in enumerate(input_lengths):
            new_tokens = output_ids[i][input_length:]
            # Convert to list for tokenizers that expect list (not tensor)
            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                output_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
            else:
                output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Fix for Gemma models: chat template includes "model\n" as regular tokens
            # These don't get removed by skip_special_tokens, so we strip them manually
            gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
            if self.model_name in gemma_models and output_text.startswith("model\n"):
                output_text = output_text[len("model\n"):]

            outputs.append(output_text.strip())

        return outputs

    def generate_batch_with_multi_steering(
        self,
        prompts: List[str],
        layer_idx: int,
        steering_vectors: List[torch.Tensor],
        strength: float = 1.0,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        steering_start_positions: Optional[List[int]] = None,
        debug: bool = False,
        **generation_kwargs,
    ) -> List[str]:
        """
        Generate text for a batch of prompts with DIFFERENT steering vectors per prompt.

        Args:
            prompts: List of input prompts
            layer_idx: Layer to apply steering at
            steering_vectors: List of vectors (one per prompt)
            strength: Multiplier for steering vectors
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            steering_start_positions: List of token positions to start steering from (one per prompt)
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Some models have issues with batched generation + steering hooks
        # Fall back to sequential generation
        if self.model_name in ["kimi_k2", "deepseek_v3"]:
            outputs = []
            for i, prompt in enumerate(prompts):
                steering_vec = steering_vectors[i]
                start_pos = steering_start_positions[i] if steering_start_positions else None
                output = self.generate_with_steering(
                    prompt=prompt,
                    layer_idx=layer_idx,
                    steering_vector=steering_vec,
                    strength=strength,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    steering_start_pos=start_pos,
                    **generation_kwargs,
                )
                outputs.append(output)
            return outputs
        assert len(prompts) == len(steering_vectors), "Must have one steering vector per prompt"

        # Prepare steering vectors: shape [batch_size, hidden_dim]
        steering_vecs = torch.stack([v * strength for v in steering_vectors]).to(self.device).to(self.dtype)

        # Convert steering positions to tensor if provided
        if steering_start_positions is not None:
            assert len(steering_start_positions) == len(prompts), "Must have one steering position per prompt"
            # Convert to tensor: [batch_size]
            steering_pos_tensor = torch.tensor(steering_start_positions, device=self.device)
        else:
            steering_pos_tensor = None

        # Track hook calls for debugging
        hook_calls = []

        def steering_hook(module, input, output):
            # Handle both tuple and non-tuple outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_of_output = output[1:]
            else:
                hidden_states = output
                rest_of_output = ()

            batch_size, seq_len, hidden_dim = hidden_states.shape

            if debug and len(hook_calls) == 0:
                print(f"[DEBUG] Hook fired! seq_len={seq_len}, batch={batch_size}, output_is_tuple={isinstance(output, tuple)}")
                if steering_pos_tensor is not None:
                    print(f"[DEBUG] steering_pos_tensor={steering_pos_tensor.tolist()}")

            # Move steering vectors to the same device as hidden_states (for multi-GPU)
            steering_vecs_device = steering_vecs.to(hidden_states.device)

            if steering_pos_tensor is not None:
                # During generation with KV cache, seq_len will be 1 (only new token)
                # We want to steer ALL generated tokens, so apply unconditionally
                if seq_len == 1:
                    # Generation phase: vectorized add (no clone needed, we overwrite all)
                    modified_hidden_states = hidden_states + steering_vecs_device.unsqueeze(1)

                    if debug:
                        hook_calls.append(f"GEN (seq_len=1, batch={batch_size})")
                else:
                    # Prompt processing phase: apply steering from start_pos onwards
                    # Use vectorized masking instead of per-sample Python loop
                    modified_hidden_states = hidden_states.clone()
                    steering_pos_device = steering_pos_tensor.to(hidden_states.device)
                    # Build position mask: [batch, seq_len] where True = should steer
                    pos_range = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)  # [1, seq_len]
                    steer_mask = pos_range >= steering_pos_device.unsqueeze(1)  # [batch, seq_len]
                    # Apply: [batch, seq_len, 1] * [batch, 1, hidden_dim]
                    modified_hidden_states += steer_mask.unsqueeze(2).float() * steering_vecs_device.unsqueeze(1)

                    if debug:
                        steered_count = (steering_pos_device < seq_len).sum().item()
                        hook_calls.append(f"PROMPT (seq_len={seq_len}, batch={batch_size}, steered={steered_count})")

                # Return in the same format as input
                if isinstance(output, tuple):
                    return (modified_hidden_states,) + rest_of_output
                else:
                    return modified_hidden_states
            else:
                # Apply to all positions (vectorized)
                modified_hidden_states = hidden_states + steering_vecs_device.unsqueeze(1)

                # Return in the same format as input
                if isinstance(output, tuple):
                    return (modified_hidden_states,) + rest_of_output
                else:
                    return modified_hidden_states

        # Register hook
        layer_module = self.get_layer_module(layer_idx)
        hook = layer_module.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        try:
            # Tokenize with padding
            # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self._get_input_device())
            input_lengths = (inputs['attention_mask'].sum(dim=1)).tolist()

            # Adjust steering positions for left padding
            # When using left padding, shorter prompts get padding tokens added to the left,
            # which shifts all token indices. We need to adjust steering positions accordingly.
            if steering_pos_tensor is not None and self.tokenizer.padding_side == "left":
                max_length = inputs['input_ids'].shape[1]
                input_lengths_tensor = inputs['attention_mask'].sum(dim=1).to(self.device)
                steering_pos_tensor = steering_pos_tensor + (max_length - input_lengths_tensor)

            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                for k, v in generation_kwargs.items():
                    gen_kwargs[k] = v

            # Disable cache for models with cache compatibility issues
            # Note: DeepSeek V3 has attention mask tracking issues when using steering hooks + cache
            # The cache works fine for regular generation, but not with activation steering
            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                gen_kwargs["use_cache"] = False

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            # ALWAYS clean up hook, even if exception occurs
            hook.remove()
            if hook in self.hooks:
                self.hooks.remove(hook)

        # Print debug info if requested
        if debug:
            print(f"\n[DEBUG] Multi-steering hook called {len(hook_calls)} times:")
            prompt_calls = [c for c in hook_calls if c.startswith("PROMPT")]
            gen_calls = [c for c in hook_calls if c.startswith("GEN")]
            print(f"  - Prompt processing: {len(prompt_calls)} calls")
            if prompt_calls:
                print(f"    {prompt_calls[0]}")
            print(f"  - Generation: {len(gen_calls)} calls (one per generated token)")
            print(f"  - Expected: 1 prompt + {max_new_tokens} generation = {1 + max_new_tokens} total")
            if len(hook_calls) == 1 + max_new_tokens * len(prompts):
                print(f"  ✓ Steering applied correctly!")
            else:
                print(f"  ✗ WARNING: Expected {1 + max_new_tokens * len(prompts)}, got {len(hook_calls)}")

        # Decode only the newly generated tokens for each sample
        # Important: model.generate() returns sequences WITH input tokens prepended
        # We need to identify exactly where the input ends and generation begins
        outputs = []
        for i in range(len(prompts)):
            # Get non-padded input tokens for comparison
            input_mask = inputs['attention_mask'][i]
            input_tokens_no_pad = inputs['input_ids'][i][input_mask.bool()]

            # Output sequence includes input + generated tokens
            output_sequence = output_ids[i]

            # Find where input tokens end in the output by comparing
            # The output might have padding removed or might preserve it
            # Strategy: Find the last position where input tokens match output tokens
            input_len = len(input_tokens_no_pad)
            output_len = len(output_sequence)

            # Check if the output starts with the input tokens (no padding)
            # or if we need to account for padding
            if output_len >= input_len and torch.equal(output_sequence[:input_len], input_tokens_no_pad):
                # Output starts with non-padded input, slice from there
                generated_ids = output_sequence[input_len:]
            else:
                # Fall back to slicing from the full input sequence length (including padding)
                # This handles cases where padding is preserved or tokenization changed
                input_seq_len = inputs['input_ids'].shape[1]
                if output_len > input_seq_len:
                    generated_ids = output_sequence[input_seq_len:]
                else:
                    # Last resort: use attention mask sum
                    generated_ids = output_sequence[input_mask.sum().item():]

            # Decode the generated tokens
            if self.model_name in ["kimi_k2", "deepseek_v3"]:
                output_text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
            else:
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Fix for Gemma models: chat template includes "model\n" as regular tokens
            # These don't get removed by skip_special_tokens, so we strip them manually
            gemma_models = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"]
            if self.model_name in gemma_models and output_text.startswith("model\n"):
                output_text = output_text[len("model\n"):]

            outputs.append(output_text.strip())

        return outputs

    def get_logits(
        self,
        prompt: str,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Get logits at a specific position without steering.

        Args:
            prompt: Input prompt
            position: Token position to extract logits from (-1 = last token)

        Returns:
            Logits tensor of shape (vocab_size,)
        """
        # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._get_input_device())

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, position, :]  # (vocab_size,)

        return logits

    def get_logits_with_steering(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer_idx: int,
        strength: float = 1.0,
        position: int = -1,
        steering_start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get logits at a specific position WITH activation steering.

        Args:
            prompt: Input prompt
            steering_vector: Vector to add to activations (shape: [hidden_dim])
            layer_idx: Layer to apply steering at
            strength: Multiplier for steering vector
            position: Token position to extract logits from (-1 = last token)
            steering_start_pos: Token position to start steering from (None = all positions)

        Returns:
            Logits tensor of shape (vocab_size,)
        """
        # Prepare steering vector
        steering_vec = (steering_vector * strength).to(self.device).to(self.dtype)

        # Define steering hook
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                batch_size, seq_len, hidden_dim = hidden_states.shape

                if steering_start_pos is not None and steering_start_pos < seq_len:
                    # Apply steering from start_pos onwards
                    modified_hidden_states = hidden_states.clone()
                    modified_hidden_states[:, steering_start_pos:, :] += steering_vec.view(1, 1, -1)
                    return (modified_hidden_states,) + output[1:]
                elif steering_start_pos is None:
                    # Apply to all positions
                    modified_hidden_states = hidden_states + steering_vec.view(1, 1, -1)
                    return (modified_hidden_states,) + output[1:]
                else:
                    return output
            else:
                if steering_start_pos is None:
                    return output + steering_vec.view(1, 1, -1)
                else:
                    return output

        # Get layer module and register hook
        try:
            # Try Gemma 3 multimodal structure first
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
                layer_module = self.model.model.language_model.layers[layer_idx]
            # Standard structure
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layer_module = self.model.model.layers[layer_idx]
            elif hasattr(self.model, 'layers'):
                layer_module = self.model.layers[layer_idx]
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer_module = self.model.transformer.h[layer_idx]
            else:
                raise ValueError(f"Could not find layer module for model type {self.model_type}")

            hook_handle = layer_module.register_forward_hook(steering_hook)
        except Exception as e:
            raise ValueError(f"Failed to register steering hook at layer {layer_idx}: {e}")

        try:
            # Run forward pass with steering
            # CRITICAL: add_special_tokens=False because prompts from apply_chat_template already include <bos>
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._get_input_device())

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, position, :]  # (vocab_size,)

            return logits

        finally:
            # Always remove hook
            hook_handle.remove()

    def cleanup(self):
        """Remove all hooks and free memory."""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


def get_layer_at_fraction(model_wrapper: ModelWrapper, fraction: float) -> int:
    """
    Get layer index at a specific fraction through the model.

    Args:
        model_wrapper: Model wrapper instance
        fraction: Fraction through model (e.g., 0.67 for 2/3)

    Returns:
        Layer index
    """
    layer_idx = int(model_wrapper.n_layers * fraction)
    # Clamp to valid range
    return max(0, min(layer_idx, model_wrapper.n_layers - 1))


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    quantization: Optional[str] = None,
) -> ModelWrapper:
    """
    Convenience function to load a model.

    Args:
        model_name: Model identifier
        device: Device to load on
        dtype: Data type (bfloat16, float16, float32)
        quantization: Quantization scheme (8bit, 4bit, or None)

    Returns:
        ModelWrapper instance
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    # Create quantization config if requested
    quantization_config = None

    # Check if model is pre-quantized (skip BitsAndBytes quantization)
    if model_name in PRE_QUANTIZED_MODELS:
        if quantization is not None:
            print(f"Note: {model_name} is pre-quantized (FP8). Ignoring BitsAndBytes {quantization} quantization.")
        quantization_config = None
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    return ModelWrapper(
        model_name=model_name,
        device=device,
        dtype=dtype_map.get(dtype, torch.bfloat16),
        quantization_config=quantization_config,
    )
