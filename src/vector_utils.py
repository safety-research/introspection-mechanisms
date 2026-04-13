"""
Utilities for extracting concept vectors from contrastive pairs.

This module handles:
- Creating contrastive prompt pairs
- Extracting concept vectors (difference between activations)
- Mean baseline subtraction
- Vector operations (cosine similarity, normalization, etc.)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

from model_utils import ModelWrapper, get_layer_at_fraction


def create_concept_prompts(
    concept_word: str,
    baseline_words: Optional[List[str]] = None,
    template: str = "Tell me about {word}",
) -> Tuple[List[str], List[str]]:
    """
    Create contrastive prompts for concept extraction.

    Args:
        concept_word: The word representing the concept to extract
        baseline_words: List of baseline words for mean subtraction (if None, no baseline)
        template: Prompt template with {word} placeholder

    Returns:
        Tuple of (concept_prompts, baseline_prompts)
    """
    concept_prompts = [template.format(word=concept_word)]

    if baseline_words is None:
        baseline_prompts = []
    else:
        baseline_prompts = [template.format(word=word) for word in baseline_words]

    return concept_prompts, baseline_prompts


def create_contrastive_pair(
    positive_text: str,
    negative_text: str,
) -> Tuple[str, str]:
    """
    Create a contrastive pair from two texts that differ in one concept.

    Args:
        positive_text: Text with the concept present
        negative_text: Text without the concept

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    return positive_text, negative_text


def extract_concept_vector(
    model: ModelWrapper,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer_idx: int,
    token_idx: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Extract concept vector from contrastive prompts.

    The concept vector is the mean difference between activations on positive
    and negative prompts at a specific layer and token position.

    Args:
        model: Model wrapper instance
        positive_prompts: Prompts where concept is present
        negative_prompts: Prompts where concept is absent
        layer_idx: Layer to extract from
        token_idx: Token position (-1 for last)
        normalize: Whether to normalize the vector

    Returns:
        Concept vector of shape [hidden_dim]
    """
    # Extract activations for positive examples
    pos_acts = model.extract_activations(
        prompts=positive_prompts,
        layer_idx=layer_idx,
        token_idx=token_idx,
    )

    # Extract activations for negative examples
    neg_acts = model.extract_activations(
        prompts=negative_prompts,
        layer_idx=layer_idx,
        token_idx=token_idx,
    )

    # Compute mean difference
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    concept_vec = pos_mean - neg_mean

    # Normalize if requested
    if normalize:
        concept_vec = concept_vec / (concept_vec.norm() + 1e-8)

    return concept_vec


def extract_concept_vector_with_baseline(
    model: ModelWrapper,
    concept_word: str,
    baseline_words: List[str],
    layer_idx: int,
    template: str = "Tell me about {word}",
    token_idx: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Extract concept vector using mean baseline subtraction.

    This follows the paper's method:
    1. Get activation for concept word prompt
    2. Get mean activation over baseline word prompts
    3. Concept vector = concept_activation - baseline_mean

    Args:
        model: Model wrapper instance
        concept_word: Word representing the concept
        baseline_words: List of random baseline words
        layer_idx: Layer to extract from
        template: Prompt template
        token_idx: Token position (-1 for last, use this for most models)
        normalize: Whether to normalize the vector

    Returns:
        Concept vector of shape [hidden_dim]
    """
    # Format prompts using chat template
    def format_prompt(word: str) -> str:
        """Format a single word prompt using chat template."""
        user_message = template.format(word=word)
        if hasattr(model.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": user_message}]
            return model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {user_message}\n\nAssistant:"

    # Format all prompts
    concept_prompts = [format_prompt(concept_word)]
    baseline_prompts = [format_prompt(word) for word in baseline_words]

    # Extract concept activation
    concept_act = model.extract_activations(
        prompts=concept_prompts,
        layer_idx=layer_idx,
        token_idx=token_idx,
    )

    # Extract baseline activations
    baseline_acts = model.extract_activations(
        prompts=baseline_prompts,
        layer_idx=layer_idx,
        token_idx=token_idx,
    )

    # Compute concept vector
    baseline_mean = baseline_acts.mean(dim=0)
    concept_vec = concept_act[0] - baseline_mean

    # Normalize if requested
    if normalize:
        concept_vec = concept_vec / (concept_vec.norm() + 1e-8)

    return concept_vec


def extract_concept_vector_simple(
    model: ModelWrapper,
    concept_word: str,
    layer_idx: int,
    control_prompt: str = "The",
    template: str = "{word}",
    token_idx: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Extract concept vector using single control prompt subtraction.
    
    This is closer to the paper's actual method which uses:
    concept_activation - control_activation (single control, not mean of many).
    
    Args:
        model: Model wrapper instance
        concept_word: Word representing the concept
        layer_idx: Layer to extract from
        control_prompt: Single neutral control prompt
        template: Prompt template (default: just the word itself)
        token_idx: Token position (-1 for last)
        normalize: Whether to normalize the vector
    
    Returns:
        Concept vector of shape [hidden_dim]
    """
    # Format prompts using chat template
    def format_prompt(text: str) -> str:
        """Format a prompt using chat template."""
        if hasattr(model.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": text}]
            return model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {text}\n\nAssistant:"
    
    # Create prompts
    concept_text = template.format(word=concept_word)
    concept_prompt = format_prompt(concept_text)
    control_prompt_formatted = format_prompt(control_prompt)
    
    # Extract activations
    concept_act = model.extract_activations(
        prompts=[concept_prompt],
        layer_idx=layer_idx,
        token_idx=token_idx,
    )[0]
    
    control_act = model.extract_activations(
        prompts=[control_prompt_formatted],
        layer_idx=layer_idx,
        token_idx=token_idx,
    )[0]
    
    # Compute concept vector
    concept_vec = concept_act - control_act
    
    # Normalize if requested
    if normalize:
        concept_vec = concept_vec / (concept_vec.norm() + 1e-8)
    
    return concept_vec


def extract_concept_vector_no_baseline(
    model: ModelWrapper,
    concept_word: str,
    layer_idx: int,
    template: str = "{word}",
    token_idx: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Extract concept vector WITHOUT baseline subtraction.
    
    Just uses the raw activation for the concept word.
    This might work if the model naturally represents concepts distinctively.
    
    Args:
        model: Model wrapper instance
        concept_word: Word representing the concept
        layer_idx: Layer to extract from
        template: Prompt template (default: just the word itself)
        token_idx: Token position (-1 for last)
        normalize: Whether to normalize the vector
    
    Returns:
        Concept vector of shape [hidden_dim]
    """
    # Format prompt using chat template
    def format_prompt(text: str) -> str:
        """Format a prompt using chat template."""
        if hasattr(model.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": text}]
            return model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {text}\n\nAssistant:"
    
    # Create prompt
    concept_text = template.format(word=concept_word)
    concept_prompt = format_prompt(concept_text)
    
    # Extract activation
    concept_vec = model.extract_activations(
        prompts=[concept_prompt],
        layer_idx=layer_idx,
        token_idx=token_idx,
    )[0]
    
    # Normalize if requested
    if normalize:
        concept_vec = concept_vec / (concept_vec.norm() + 1e-8)
    
    return concept_vec


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    dot_product = (vec1 * vec2).sum()
    norm1 = vec1.norm()
    norm2 = vec2.norm()

    return (dot_product / (norm1 * norm2 + 1e-8)).item()


def save_concept_vector(
    vector: torch.Tensor,
    save_path: Path,
    metadata: Optional[Dict] = None,
):
    """
    Save concept vector to disk with metadata.

    Args:
        vector: Concept vector tensor
        save_path: Path to save to (.pt file)
        metadata: Optional metadata dict (will be saved as .json)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save vector
    torch.save(vector, save_path)

    # Save metadata
    if metadata is not None:
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved concept vector to {save_path}")


def load_concept_vector(load_path: Path) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Load concept vector from disk with metadata.

    Args:
        load_path: Path to load from (.pt file)

    Returns:
        Tuple of (vector, metadata)
    """
    load_path = Path(load_path)

    # Load vector
    vector = torch.load(load_path)

    # Load metadata if exists
    metadata_path = load_path.with_suffix('.json')
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return vector, metadata

# Predefined baseline words (100 words from the paper)
DEFAULT_BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]


# Concept pairs for different experiments
CONCEPT_PAIRS = {
    # Experiment 2: ALL CAPS vs normal
    "all_caps": (
        "HI! HOW ARE YOU? I'M DOING GREAT TODAY!",
        "Hi! How are you? I'm doing great today!"
    ),

    # Experiment 6: Recursion vs iteration
    "recursion_code": (
        """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        """def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result"""
    ),

    # More concepts can be added here
    "if_statement_code": (
        """def check_positive(x):
    if x > 0:
        return True
    return False""",
        """def check_positive(x):
    result = x > 0
    return result"""
    ),

    "loop_code": (
        """for i in range(10):
    print(i)""",
        """print(list(range(10)))"""
    ),
}


def get_baseline_words(n: int = 100) -> List[str]:
    """
    Get list of baseline words for concept extraction.

    Args:
        n: Number of baseline words to return (default: 100, as per paper)

    Returns:
        List of baseline words
    """
    return DEFAULT_BASELINE_WORDS[:n]


def get_concept_pair(concept_name: str) -> Tuple[str, str]:
    """
    Get predefined concept pair by name.

    Args:
        concept_name: Name of concept pair

    Returns:
        Tuple of (positive, negative)
    """
    if concept_name not in CONCEPT_PAIRS:
        raise ValueError(
            f"Unknown concept pair: {concept_name}. "
            f"Available: {list(CONCEPT_PAIRS.keys())}"
        )

    return CONCEPT_PAIRS[concept_name]


def extract_concept_vectors_batch(
    model: ModelWrapper,
    concept_words: List[str],
    baseline_words: List[str],
    layer_idx: int,
    extraction_method: str = "baseline",
    template: str = "Tell me about {word}",
    token_idx: int = -1,
    normalize: bool = False,
    extraction_batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Extract concept vectors for multiple concepts in batches (much faster than sequential).

    Args:
        model: Model wrapper instance
        concept_words: List of concept words to extract vectors for
        baseline_words: List of baseline words for mean subtraction
        layer_idx: Layer to extract from
        extraction_method: "baseline", "simple", or "no_baseline"
        template: Prompt template
        token_idx: Token position (-1 for last)
        normalize: Whether to normalize vectors
        extraction_batch_size: Batch size for activation extraction (default: 32)

    Returns:
        Dict mapping concept words to their vectors
    """
    def format_prompt(word: str) -> str:
        """Format a single word prompt using chat template."""
        user_message = template.format(word=word)
        if hasattr(model.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": user_message}]
            return model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {user_message}\n\nAssistant:"

    def batched_extract(prompts: List[str]) -> torch.Tensor:
        """Extract activations in batches to avoid OOM."""
        all_acts = []
        for i in range(0, len(prompts), extraction_batch_size):
            batch_prompts = prompts[i:i + extraction_batch_size]
            batch_acts = model.extract_activations(
                prompts=batch_prompts,
                layer_idx=layer_idx,
                token_idx=token_idx,
            )
            all_acts.append(batch_acts)
        return torch.cat(all_acts, dim=0)

    concept_vectors = {}

    if extraction_method == "baseline":
        # Batch extract all concept activations (with internal batching)
        concept_prompts = [format_prompt(word) for word in concept_words]
        concept_acts = batched_extract(concept_prompts)

        # Batch extract baseline activations (with internal batching)
        baseline_prompts = [format_prompt(word) for word in baseline_words]
        baseline_acts = batched_extract(baseline_prompts)
        baseline_mean = baseline_acts.mean(dim=0)

        # Compute vectors for all concepts
        for i, concept_word in enumerate(concept_words):
            vec = concept_acts[i] - baseline_mean
            if normalize:
                vec = vec / (vec.norm() + 1e-8)
            concept_vectors[concept_word] = vec

    elif extraction_method == "simple":
        # Single control word
        control_word = "The"
        control_prompt = format_prompt(control_word)
        control_act = model.extract_activations(
            prompts=[control_prompt],
            layer_idx=layer_idx,
            token_idx=token_idx,
        )[0]

        # Batch extract all concept activations (with internal batching)
        concept_prompts = [format_prompt(word) for word in concept_words]
        concept_acts = batched_extract(concept_prompts)

        # Compute vectors for all concepts
        for i, concept_word in enumerate(concept_words):
            vec = concept_acts[i] - control_act
            if normalize:
                vec = vec / (vec.norm() + 1e-8)
            concept_vectors[concept_word] = vec

    elif extraction_method == "no_baseline":
        # Batch extract all concept activations (with internal batching)
        concept_prompts = [format_prompt(word) for word in concept_words]
        concept_acts = batched_extract(concept_prompts)

        # Use raw activations
        for i, concept_word in enumerate(concept_words):
            vec = concept_acts[i]
            if normalize:
                vec = vec / (vec.norm() + 1e-8)
            concept_vectors[concept_word] = vec

    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")

    return concept_vectors


def analyze_vector_underspecification(
    model: ModelWrapper,
    target_concept: str,
    related_concepts: List[str],
    layer_idx: int,
    baseline_words: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Analyze whether a concept vector is underspecified by testing cross-activation.

    For example: Does a "recursion" vector also activate for "if statements"?

    Args:
        model: Model wrapper instance
        target_concept: The target concept to extract
        related_concepts: List of related concepts to test
        layer_idx: Layer to extract from
        baseline_words: Baseline words for extraction

    Returns:
        Dict mapping related concept names to cosine similarities
    """
    if baseline_words is None:
        baseline_words = get_baseline_words()

    # Extract target concept vector
    target_vec = extract_concept_vector_with_baseline(
        model=model,
        concept_word=target_concept,
        baseline_words=baseline_words,
        layer_idx=layer_idx,
    )

    # Extract related concept vectors and compute similarities
    similarities = {}
    for related_concept in related_concepts:
        related_vec = extract_concept_vector_with_baseline(
            model=model,
            concept_word=related_concept,
            baseline_words=baseline_words,
            layer_idx=layer_idx,
        )

        sim = cosine_similarity(target_vec, related_vec)
        similarities[related_concept] = sim

    return similarities
