"""
Concept Injection: Injected Thoughts Detection (Baseline)

This is the simplest experiment, replicating the core result from the paper:
- Extract concept vectors for simple words
- Inject concepts via activation steering
- Test if model can detect the injected thoughts

This establishes the baseline detection capability.

Usage:
    # Single model
    python 01_concept_injection.py --models llama_8b --strength 8.0 --layer-fraction 0.7
    
    # Multiple models
    python 01_concept_injection.py --models llama_8b deepseek_v3 qwen_7b --layer-sweep 0.4 0.5 0.6
    
    # All existing models (for re-evaluation)
    python 01_concept_injection.py --models all --reevaluate-judge
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import torch
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

from model_utils import load_model, get_layer_at_fraction, ModelWrapper
from vector_utils import (
    extract_concept_vector_with_baseline,
    extract_concept_vector_simple,
    extract_concept_vector_no_baseline,
    extract_concept_vectors_batch,
    get_baseline_words,
)
from steering_utils import (
    run_steered_introspection_test, run_unsteered_introspection_test,
    run_steered_introspection_test_batch, run_unsteered_introspection_test_batch,
    run_forced_noticing_test, run_forced_noticing_test_batch,
    calculate_detection_accuracy, check_concept_mentioned
)
from eval_utils import LLMJudge, batch_evaluate, compute_detection_and_identification_metrics, save_evaluation_results
import random

# Default test concepts (50 words from the paper)
DEFAULT_TEST_CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
    "Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
    "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
    "Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
    "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
    "Denim", "Monoliths", "Milk", "Bread", "Silver",
]
DEFAULT_N_BASELINE = 100
DEFAULT_LAYER_FRACTION = 0.7
DEFAULT_LAYER_SWEEP = [0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_STRENGTH = 8.0
DEFAULT_STRENGTH_SWEEP = [1.0, 2.0, 4.0, 8.0]
DEFAULT_N_TRIALS = 30
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_OUTPUT_DIR = "analysis/01_concept_injection"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MODEL = "llama_8b"

# Models that don't support system role in chat templates
MODELS_WITHOUT_SYSTEM_ROLE = ["gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b", "gemma3_27b_abliterated"]


def filter_messages_for_model(messages: List[Dict], model_name: str) -> List[Dict]:
    """
    Filter messages based on model capabilities.
    Some models (like Gemma) don't support system roles.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Model identifier

    Returns:
        Filtered list of messages
    """
    if model_name in MODELS_WITHOUT_SYSTEM_ROLE:
        # Remove system messages for models that don't support them
        return [msg for msg in messages if msg.get("role") != "system"]
    return messages


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Concept Injection: Injected Thoughts Detection")
    parser.add_argument("-m", "--models", type=str, nargs="+", default=[DEFAULT_MODEL], help="Model name(s) (e.g., llama_8b deepseek_v3 qwen_7b) or 'all' to run on all existing models in output dir")
    parser.add_argument("-c", "--concepts", type=str, nargs="+", default=DEFAULT_TEST_CONCEPTS, help="List of concept words to test")
    parser.add_argument("-nb", "--n-baseline", type=int, default=DEFAULT_N_BASELINE, help="Number of baseline words for vector extraction")
    parser.add_argument("-lf", "--layer-fraction", type=float, default=None, help="Single layer fraction (if not sweeping)")
    parser.add_argument("-ls", "--layer-sweep", type=float, nargs="+", default=None, help="Sweep over layer fractions (e.g., 0.4 0.5 0.6 0.7 0.8)")
    parser.add_argument("-s", "--strength", type=float, default=None, help="Single steering strength (if not sweeping)")
    parser.add_argument("-ss", "--strength-sweep", type=float, nargs="+", default=None, help="Sweep over multiple strengths (e.g., 0.5 1.0 2.0 4.0 8.0 16.0)")
    parser.add_argument("-nt", "--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Number of trials per concept (injection + control)")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("-mt", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens to generate")
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for parallel generation (higher = faster but more memory)")
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE, help="Device to run on")
    parser.add_argument("-dt", "--dtype", type=str, default=DEFAULT_DTYPE, choices=["bfloat16", "float16", "float32"], help="Model dtype")
    parser.add_argument("-q", "--quantization", type=str, default=None, choices=["8bit", "4bit"], help="Quantization scheme")
    parser.add_argument("-em", "--extraction-method", type=str, default="baseline", choices=["baseline", "simple", "no_baseline"], help="Concept vector extraction method: 'baseline' (default, mean of 100 words), 'simple' (single control word), 'no_baseline' (raw activation)")
    parser.add_argument("-nlj", "--no-llm-judge", action="store_true", help="Disable LLM judge evaluation (enabled by default, requires OPENAI_API_KEY in .env)")
    parser.add_argument("-nsv", "--no-save-vectors", action="store_true", help="Don't save concept vectors")
    parser.add_argument("-ow", "--overwrite", action="store_true", help="Overwrite existing results (default: False, resume from where left off)")
    parser.add_argument("-rej", "--reevaluate-judge", action="store_true", help="Re-evaluate existing results with LLM judge (does not regenerate responses)")
    return parser.parse_args()


def run_experiment(
    model_name: str,
    test_concepts: List[str],
    baseline_words: List[str],
    layer_fraction: float = 0.7,
    strength: float = 8.0,
    n_trials: int = 5,
    output_dir: Path = Path("analysis/01_concept_injection"),
    device: str = "cuda",
    dtype: str = "bfloat16",
    quantization: str = None,
    extraction_method: str = "baseline",
    use_llm_judge: bool = True,
    save_vectors: bool = True,
):
    """
    Run Concept Injection: Injected thoughts detection.

    Args:
        model_name: Model identifier
        test_concepts: List of concept words to test
        baseline_words: List of baseline words for vector extraction
        layer_fraction: Fraction through model to inject at (default 2/3)
        strength: Steering strength
        n_trials: Number of trials per concept
        output_dir: Output directory
        device: Device to run on
        dtype: Model dtype
        quantization: Quantization scheme
        extraction_method: Vector extraction method ("baseline", "simple", "no_baseline")
        use_llm_judge: Whether to use LLM judge for evaluation
        save_vectors: Whether to save concept vectors
    """
    print("=" * 80)
    print("EXPERIMENT 1: INJECTED THOUGHTS DETECTION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Test concepts: {len(test_concepts)}")
    print(f"Baseline words: {len(baseline_words)}")
    print(f"Extraction method: {extraction_method}")
    print(f"Layer fraction: {layer_fraction}")
    print(f"Strength: {strength}")
    print(f"Trials per concept: {n_trials}")
    print("=" * 80)

    # Create output directory
    model_output_dir = output_dir / model_name.replace("/", "_")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Create debug directory
    debug_dir = model_output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = load_model(model_name=model_name, device=device, dtype=dtype, quantization=quantization)

    # Save model configuration to debug
    with open(debug_dir / "model_config.txt", 'w') as f:
        f.write("MODEL CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"HuggingFace path: {model.hf_path}\n")
        f.write(f"Total layers: {model.n_layers}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dtype: {dtype}\n")
        f.write(f"Quantization: {quantization}\n")
        f.write(f"Tokenizer vocab size: {len(model.tokenizer)}\n")
        f.write(f"Tokenizer padding side: {model.tokenizer.padding_side}\n")
        f.write(f"Pad token: {model.tokenizer.pad_token}\n")
        f.write(f"EOS token: {model.tokenizer.eos_token}\n")

    # Get target layer
    layer_idx = get_layer_at_fraction(model, layer_fraction)
    print(f"Target layer: {layer_idx} (fraction: {layer_fraction})")

    # Extract concept vectors - BATCH EXTRACTION for speed!
    print(f"\nExtracting concept vectors for {len(test_concepts)} concepts...")
    concept_vectors = extract_concept_vectors_batch(
        model=model,
        concept_words=test_concepts,
        baseline_words=baseline_words,
        layer_idx=layer_idx,
        extraction_method=extraction_method,
    )
    print(f"  ✓ Extracted {len(concept_vectors)} concept vectors")

    vector_dir = model_output_dir / "vectors"
    vector_dir.mkdir(exist_ok=True)

    # Debug: Save detailed extraction info for first concept
    concept = test_concepts[0]
    vec = concept_vectors[concept]
    user_message = f"Tell me about {concept}"
    if hasattr(model.tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_message}]
        formatted_prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = f"User: {user_message}\n\nAssistant:"

    # Tokenize to show tokens
    tokens = model.tokenizer(formatted_prompt, return_tensors="pt")
    token_ids = tokens['input_ids'][0].tolist()
    token_strings = [model.tokenizer.decode([tid]) for tid in token_ids]

    debug_concept_extraction = {
        'concept': concept,
        'raw_prompt': user_message,
        'formatted_prompt': formatted_prompt,
        'token_ids': token_ids,
        'token_strings': token_strings,
        'num_tokens': len(token_ids),
        'target_token_idx': -1,
        'layer_idx': layer_idx,
        'vector_norm': vec.norm().item(),
        'vector_mean': vec.mean().item(),
        'vector_std': vec.std().item(),
    }

    # Save vectors if requested
    if save_vectors:
        for concept, vec in concept_vectors.items():
            vector_path = vector_dir / f"{concept}.pt"
            torch.save(vec, vector_path)

    # Save concept extraction debug info
    if debug_concept_extraction:
        with open(debug_dir / "concept_extraction_sample.txt", 'w') as f:
            f.write("CONCEPT VECTOR EXTRACTION (SAMPLE)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Concept: {debug_concept_extraction['concept']}\n")
            f.write(f"Layer: {debug_concept_extraction['layer_idx']}\n")
            f.write(f"Target token index: {debug_concept_extraction['target_token_idx']} (last token)\n\n")

            f.write("RAW PROMPT:\n")
            f.write("-" * 80 + "\n")
            f.write(debug_concept_extraction['raw_prompt'] + "\n\n")

            f.write("FORMATTED PROMPT (with chat template):\n")
            f.write("-" * 80 + "\n")
            f.write(debug_concept_extraction['formatted_prompt'] + "\n\n")

            f.write("TOKENIZATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of tokens: {debug_concept_extraction['num_tokens']}\n")
            f.write("Token ID -> Token String:\n")
            for tid, tstr in zip(debug_concept_extraction['token_ids'], debug_concept_extraction['token_strings']):
                f.write(f"  {tid:6d} -> {repr(tstr)}\n")
            f.write("\n")

            f.write("EXTRACTED VECTOR STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"L2 Norm: {debug_concept_extraction['vector_norm']:.4f}\n")
            f.write(f"Mean: {debug_concept_extraction['vector_mean']:.4f}\n")
            f.write(f"Std Dev: {debug_concept_extraction['vector_std']:.4f}\n")
            f.write(f"\nNote: Vector computed as: activation(concept_word) - mean(baseline_activations)\n")
            f.write(f"      where baseline = mean over {len(baseline_words)} random words\n")

    # Save vector statistics for all concepts
    with open(debug_dir / "vector_statistics.txt", 'w') as f:
        f.write("ALL CONCEPT VECTOR STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Concept':<20} {'L2 Norm':>12} {'Mean':>12} {'Std Dev':>12}\n")
        f.write("-" * 80 + "\n")
        for concept, vec in concept_vectors.items():
            f.write(f"{concept:<20} {vec.norm().item():>12.4f} {vec.mean().item():>12.4f} {vec.std().item():>12.4f}\n")

    # Run introspection tests
    print("\nRunning introspection tests...")
    all_results = []
    debug_samples = []  # Track first injection and control trial for debug output

    for i, concept in enumerate(test_concepts):
        print(f"  [{i+1}/{len(test_concepts)}] Testing: {concept}")

        for trial in range(1, n_trials + 1):
            # Capture debug info for first concept, first trial
            if i == 0 and trial == 1:
                # Build the prompt manually to capture debug info
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": (
                        "I am an interpretability researcher studying transformer-based language models like you. "
                        "I have access to the internals of your neural network. I have identified patterns in your "
                        "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
                        "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
                        "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
                    )},
                    {"role": "assistant", "content": "Ok."},
                    {"role": "user", "content": (
                        f"Trial {trial}: Do you detect an injected thought? "
                        f"If so, what is the injected thought about?"
                    )},
                ]

                formatted_prompt = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) if hasattr(model.tokenizer, 'apply_chat_template') else (
                    f"{messages[1]['content']}\n\nAssistant: {messages[2]['content']}\n\n"
                    f"User: {messages[3]['content']}\n\nAssistant:"
                )

                # Tokenize
                tokens = model.tokenizer(formatted_prompt, return_tensors="pt")
                token_ids = tokens['input_ids'][0].tolist()

                # Find steering start position
                trial_text = f"Trial {trial}"
                trial_pos_in_text = formatted_prompt.find(trial_text)
                if trial_pos_in_text != -1:
                    prompt_before_trial = formatted_prompt[:trial_pos_in_text]
                    tokens_before_trial = model.tokenizer(prompt_before_trial, return_tensors="pt")
                    steering_start_pos = tokens_before_trial['input_ids'].shape[1] - 1
                else:
                    steering_start_pos = None

                debug_samples.append({
                    'concept': concept,
                    'trial': trial,
                    'injected': True,
                    'formatted_prompt': formatted_prompt,
                    'token_ids': token_ids,
                    'num_tokens': len(token_ids),
                    'steering_start_pos': steering_start_pos,
                    'steering_strength': strength,
                    'target_layer': layer_idx,
                })

            response = run_steered_introspection_test(
                model=model,
                concept_word=concept,
                steering_vector=concept_vectors[concept],
                layer_idx=layer_idx,
                strength=strength,
                trial_number=trial,
            )

            # Check if model mentions the injected concept
            detected = check_concept_mentioned(response, concept)

            result = {
                "concept": concept,
                "trial": trial,
                "response": response,
                "injected": True,
                "layer": layer_idx,
                "strength": strength,
                "detected": detected,
            }

            all_results.append(result)

            # Add response to debug sample
            if i == 0 and trial == 1:
                debug_samples[-1]['response'] = response
                debug_samples[-1]['detected'] = detected

            # Print sample response
            if trial == 1:
                print(f"    Sample response: {response[:100]}...")

    # Calculate basic accuracy
    detection_rate = calculate_detection_accuracy(all_results)
    print(f"\nBasic detection accuracy: {detection_rate:.2%}")

    # Use LLM judge if requested
    if use_llm_judge:
        print("\nRunning LLM judge evaluation (concurrent API calls)...")
        try:
            # Reconstruct original prompts for each result
            original_prompts = []
            for result in all_results:
                trial_num = result["trial"]
                trial_type = result.get("trial_type", "injection")

                if trial_type == "forced_injection":
                    prompt = f"Trial {trial_num}: What injected thought do you notice?"
                else:
                    prompt = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"

                original_prompts.append(prompt)

            # Run LLM judge evaluation concurrently
            judge = LLMJudge()
            evaluated_results = batch_evaluate(judge, all_results, original_prompts)

            # Compute detection and identification metrics
            metrics = compute_detection_and_identification_metrics(evaluated_results)

            print(f"\nLLM Judge Metrics:")
            print(f"  Trials: {metrics['n_injection']} injection, {metrics['n_control']} control, {metrics['n_forced']} forced")
            print(f"\n  DETECTION (spontaneous trials):")
            print(f"    Hit Rate:        {metrics['detection_hit_rate']:.2%}")
            print(f"    False Alarm Rate: {metrics['detection_false_alarm_rate']:.2%}")
            print(f"    Accuracy:        {metrics['detection_accuracy']:.2%}")
            print(f"\n  IDENTIFICATION (when claiming detection):")
            if metrics['identification_accuracy_given_claim'] is not None:
                print(f"    Accuracy:        {metrics['identification_accuracy_given_claim']:.2%}")
            else:
                print(f"    Accuracy:        N/A (no claims)")
            print(f"\n  COMBINED:")
            print(f"    Detection + ID:  {metrics['combined_detection_and_identification_rate']:.2%}")
            if metrics['forced_identification_accuracy'] is not None:
                print(f"    Forced ID:       {metrics['forced_identification_accuracy']:.2%}")

        except Exception as e:
            print(f"LLM judge evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            evaluated_results = all_results
            metrics = {"detection_rate": detection_rate}
    else:
        evaluated_results = all_results
        metrics = {"detection_rate": detection_rate}

    # Save results
    print("\nSaving results...")

    # Save JSON
    json_path = model_output_dir / "results.json"
    save_evaluation_results(evaluated_results, json_path, metrics)

    # Save CSV
    csv_path = model_output_dir / "results.csv"
    df = pd.DataFrame(evaluated_results)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Save debug samples
    if debug_samples:
        with open(debug_dir / "introspection_test_sample.txt", 'w') as f:
            f.write("INTROSPECTION TEST EXECUTION (DETAILED SAMPLE)\n")
            f.write("=" * 80 + "\n\n")

            for sample in debug_samples:
                f.write(f"Concept: {sample['concept']}\n")
                f.write(f"Trial: {sample['trial']}\n")
                f.write(f"Injection: {'YES' if sample['injected'] else 'NO (control)'}\n")
                f.write(f"Target Layer: {sample['target_layer']}\n")
                f.write(f"Steering Strength: {sample['steering_strength']}\n")
                f.write("\n")

                f.write("FORMATTED PROMPT (sent to model):\n")
                f.write("-" * 80 + "\n")
                f.write(sample['formatted_prompt'])
                f.write("\n" + "-" * 80 + "\n\n")

                f.write("TOKENIZATION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total tokens: {sample['num_tokens']}\n")
                f.write(f"Token IDs: {sample['token_ids'][:20]}{'...' if len(sample['token_ids']) > 20 else ''}\n")
                f.write("\n")

                f.write("STEERING APPLICATION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Steering start position (token index): {sample['steering_start_pos']}\n")
                if sample['steering_start_pos'] is not None:
                    f.write(f"  -> Steering begins at token {sample['steering_start_pos']} (0-indexed)\n")
                    f.write(f"  -> This is the token BEFORE 'Trial {sample['trial']}' in the prompt\n")
                    f.write(f"  -> Steering continues through all generated tokens\n")
                else:
                    f.write(f"  -> Steering applied to ALL tokens (fallback)\n")
                f.write(f"Steering vector: concept vector * {sample['steering_strength']}\n")
                f.write(f"Applied at: Layer {sample['target_layer']} residual stream\n")
                f.write("\n")

                f.write("MODEL RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(sample['response'])
                f.write("\n" + "-" * 80 + "\n\n")

                f.write("DETECTION RESULT:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Detected: {sample['detected']}\n")
                f.write(f"Expected: {sample['injected']}\n")
                f.write(f"Correct: {sample['detected'] == sample['injected']}\n")
                f.write("\n" + "=" * 80 + "\n\n")

    # Save examples
    examples_path = model_output_dir / "examples.txt"
    with open(examples_path, 'w') as f:
        f.write("EXPERIMENT 1: INJECTED THOUGHTS DETECTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Layer: {layer_idx} (fraction: {layer_fraction})\n")
        f.write(f"Strength: {strength}\n")
        f.write("\n")

        # Write sample responses for each concept
        for concept in test_concepts:
            concept_results = [r for r in evaluated_results if r["concept"] == concept]
            if concept_results:
                f.write(f"\nConcept: {concept}\n")
                f.write("-" * 80 + "\n")
                sample = concept_results[0]
                f.write(f"Response: {sample['response']}\n")
                f.write(f"Detected: {sample.get('detected', 'N/A')}\n")
                f.write("\n")

    print(f"Saved examples to {examples_path}")

    # Save metrics summary
    summary_path = model_output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT 1: SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test concepts: {len(test_concepts)}\n")
        f.write(f"Trials per concept: {n_trials}\n")
        f.write(f"Total samples: {len(evaluated_results)}\n")
        f.write(f"\nLayer: {layer_idx} (fraction: {layer_fraction})\n")
        f.write(f"Strength: {strength}\n")
        f.write(f"\nMETRICS:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")

    print(f"Saved summary to {summary_path}")

    # Cleanup
    model.cleanup()

    print("\nConcept Injection complete!")
    print(f"Results saved to: {model_output_dir}")

    return evaluated_results, metrics


def sanitize_model_name_for_display(model_name: str) -> str:
    """
    Convert model name to display-friendly format.
    Replaces underscores and slashes with dashes to avoid matplotlib subscript issues.
    """
    return model_name.replace('_', '-').replace('/', '-')


def create_sweep_plots(all_results: Dict, concepts: List[str], layer_fractions: List[float], strengths: List[float], output_dir: Path):
    """Create plots showing detection accuracy across layers and strengths."""
    plt.rcParams.update({'font.size': 14})
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create subdirectory for individual concept plots
    individual_plots_dir = plots_dir / "individual"
    individual_plots_dir.mkdir(exist_ok=True)

    # Organize results by concept (EXCLUDE forced_injection trials from detection calculations)
    # Store injection and control results separately to compute meaningful accuracy
    results_by_concept = {concept: {} for concept in concepts}
    for (layer_frac, strength), data in all_results.items():
        for result in data["results"]:
            # Skip forced_injection trials - they don't measure spontaneous detection
            if result.get("trial_type") == "forced_injection":
                continue

            concept = result["concept"]
            if (layer_frac, strength) not in results_by_concept[concept]:
                results_by_concept[concept][(layer_frac, strength)] = {
                    'injection': [],
                    'control': []
                }

            # Determine trial type with backward compatibility
            trial_type = result.get("trial_type")
            if trial_type is None:
                # Old results without trial_type: infer from injected field
                trial_type = "injection" if result.get("injected") else "control"

            # Get LLM judge detection result (not the naive string matching "detected" field)
            llm_detected = result.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False)

            if trial_type == "injection":
                # For injection trials: detected=True is correct (TP)
                results_by_concept[concept][(layer_frac, strength)]['injection'].append(llm_detected)
            elif trial_type == "control":
                # For control trials: detected=False is correct (TN)
                results_by_concept[concept][(layer_frac, strength)]['control'].append(llm_detected)

    # Compute detection ACCURACY (not just raw detection rate) and standard errors
    detection_rates = {concept: {} for concept in concepts}
    detection_errors = {concept: {} for concept in concepts}
    for concept in concepts:
        for (layer_frac, strength), trial_data in results_by_concept[concept].items():
            injection_trials = trial_data['injection']
            control_trials = trial_data['control']

            # Count correct detections
            true_positives = sum(1 for d in injection_trials if d == True)
            true_negatives = sum(1 for d in control_trials if d == False)
            total = len(injection_trials) + len(control_trials)

            # Detection accuracy = (TP + TN) / Total
            accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

            # Standard error: SE = sqrt(p * (1-p) / n)
            se = np.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0

            detection_rates[concept][(layer_frac, strength)] = accuracy
            detection_errors[concept][(layer_frac, strength)] = se

    # 1. Heatmaps for each concept (layer x strength)
    for concept in concepts:
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap_data = np.zeros((len(layer_fractions), len(strengths)))
        for i, layer_frac in enumerate(layer_fractions):
            for j, strength in enumerate(strengths):
                heatmap_data[i, j] = detection_rates[concept].get((layer_frac, strength), 0.0)

        im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels([f"{s:.1f}" for s in strengths], fontsize=16)
        ax.set_yticks(range(len(layer_fractions)))
        ax.set_yticklabels([f"{lf:.2f}" for lf in layer_fractions], fontsize=16)
        ax.set_xlabel('Steering Strength', fontsize=18, fontweight='bold')
        ax.set_ylabel('Layer Fraction', fontsize=18, fontweight='bold')
        ax.set_title(f'Detection Accuracy: {concept}\n(Injection + Control Trials)', fontsize=20, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add text annotations
        for i in range(len(layer_fractions)):
            for j in range(len(strengths)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}', ha="center", va="center", color="black", fontsize=12)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Detection Accuracy', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(individual_plots_dir / f'{concept}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Line plots: Detection rate vs strength (one line per layer) with error bars
    for concept in concepts:
        fig, ax = plt.subplots(figsize=(10, 7))
        for layer_frac in layer_fractions:
            rates = [detection_rates[concept].get((layer_frac, s), 0.0) for s in strengths]
            errors = [detection_errors[concept].get((layer_frac, s), 0.0) for s in strengths]
            ax.errorbar(strengths, rates, yerr=errors, marker='o', markersize=8, linewidth=2,
                       capsize=5, capthick=2, label=f'Layer {layer_frac:.2f}')
        ax.set_xlabel('Steering Strength', fontsize=18, fontweight='bold')
        ax.set_ylabel('Detection Accuracy', fontsize=18, fontweight='bold')
        ax.set_title(f'{concept}: Detection Accuracy vs Strength', fontsize=20, fontweight='bold')
        ax.legend(fontsize=14, loc='best')
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(individual_plots_dir / f'{concept}_strength_sweep.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Line plots: Detection rate vs layer (one line per strength) with error bars
    for concept in concepts:
        fig, ax = plt.subplots(figsize=(10, 7))
        for strength in strengths:
            rates = [detection_rates[concept].get((lf, strength), 0.0) for lf in layer_fractions]
            errors = [detection_errors[concept].get((lf, strength), 0.0) for lf in layer_fractions]
            ax.errorbar(layer_fractions, rates, yerr=errors, marker='o', markersize=8, linewidth=2,
                       capsize=5, capthick=2, label=f'Strength {strength:.1f}')
        ax.set_xlabel('Layer Fraction', fontsize=18, fontweight='bold')
        ax.set_ylabel('Detection Accuracy', fontsize=18, fontweight='bold')
        ax.set_title(f'{concept}: Detection Accuracy vs Layer', fontsize=20, fontweight='bold')
        ax.legend(fontsize=14, loc='best')
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(individual_plots_dir / f'{concept}_layer_sweep.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Summary plot: Best layer and strength for each concept with error bars
    best_configs = {}
    for concept in concepts:
        best_rate = 0.0
        best_config = None
        for (layer_frac, strength), rate in detection_rates[concept].items():
            if rate > best_rate:
                best_rate = rate
                best_config = (layer_frac, strength)
        # Only add to best_configs if we found at least one result
        if best_config is not None:
            best_error = detection_errors[concept].get(best_config, 0.0)
            best_configs[concept] = (best_config, best_rate, best_error)

    # Only plot if we have results
    if best_configs:
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_concepts = list(best_configs.keys())
        x_pos = np.arange(len(plot_concepts))
        rates = [best_configs[c][1] for c in plot_concepts]
        errors = [best_configs[c][2] for c in plot_concepts]
        bars = ax.bar(x_pos, rates, yerr=errors, color='steelblue', alpha=0.8, edgecolor='black',
                     linewidth=1.5, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_concepts, rotation=45, ha='right', fontsize=16)
        ax.set_ylabel('Best Detection Accuracy', fontsize=18, fontweight='bold')
        ax.set_title('Best Detection Accuracy by Concept\n(Injection + Control)', fontsize=20, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add text labels with best layer/strength
        for i, (concept, (config, rate, error)) in enumerate(best_configs.items()):
            layer_frac, strength = config
            ax.text(i, rate + error + 0.02, f'{rate:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=90)
            ax.text(i, rate/2, f'L={layer_frac:.2f}, S={strength:.1f}', ha='center', va='center', fontsize=10, rotation=90, color='white')

        plt.tight_layout()
        plt.savefig(plots_dir / 'best_configs_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 5. All metrics summary at best overall configuration
    # Find best overall configuration (using combined rate as criterion)
    best_overall_config = None
    best_combined_rate = 0.0
    for (layer_frac, strength), data in all_results.items():
        if data['combined_detection_and_identification_rate'] > best_combined_rate:
            best_combined_rate = data['combined_detection_and_identification_rate']
            best_overall_config = (layer_frac, strength)

    if best_overall_config is not None:
        layer_frac, strength = best_overall_config
        best_data = all_results[best_overall_config]

        # Create barplot with key metrics
        fig, ax = plt.subplots(figsize=(12, 7))

        metric_names = [
            'True Positive Rate',
            'Detection Accuracy\n(Injection vs Control)',
            'False Positive Rate',
            'P(Detect ∧ Correct ID | Injection)\n(Introspection)'
        ]
        metric_values = [
            best_data['detection_hit_rate'],
            best_data['detection_accuracy'],
            best_data['detection_false_alarm_rate'],
            best_data['combined_detection_and_identification_rate']
        ]

        # Define colors: blue for TPR, purple for detection accuracy, red for FPR, green for introspection
        colors = ['#1f77b4', '#9467bd', '#d62728', '#2ca02c']

        x_pos = np.arange(len(metric_names))
        bars = ax.bar(x_pos, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names, rotation=0, ha='center', fontsize=13)
        ax.set_ylabel('Rate', fontsize=16, fontweight='bold')
        ax.set_title(f'Key Introspection Metrics at Best Configuration (L={layer_frac:.2f}, S={strength:.1f})', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2%}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(plots_dir / 'key_metrics_best_config.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to: {plots_dir}")
    print(f"  - Summary plots: {plots_dir}")
    print(f"  - Individual concept plots: {individual_plots_dir}")
    if best_configs:
        print("\nBest configurations (by detection accuracy):")
        for concept, (config, rate, error) in best_configs.items():
            layer_frac, strength = config
            print(f"  {concept}: Layer={layer_frac:.2f}, Strength={strength:.1f}, Accuracy={rate:.2%} (SE={error:.2%})")
    else:
        print("\nNo results found for any concepts.")


def create_trial_type_comparison_plots(all_results: Dict, output_dir: Path):
    """
    Create plots comparing detection/identification across trial types:
    - Injection trials: spontaneous detection
    - Control trials: spontaneous (should not detect)
    - Forced injection trials: forced to notice, only measure identification

    Args:
        all_results: Dictionary mapping (layer_frac, strength) -> results dict with metrics
        output_dir: Directory to save plots
    """
    plt.rcParams.update({'font.size': 14})
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create subdirectory for trial type plots
    trial_type_plots_dir = plots_dir / "by_trial_type"
    trial_type_plots_dir.mkdir(exist_ok=True)

    # Extract best configuration (highest combined rate)
    best_config = None
    best_combined_rate = 0.0
    for (layer_frac, strength), data in all_results.items():
        if data.get('combined_detection_and_identification_rate', 0) > best_combined_rate:
            best_combined_rate = data['combined_detection_and_identification_rate']
            best_config = (layer_frac, strength)

    if best_config is None:
        print("No results found for trial type comparison plots")
        return

    layer_frac, strength = best_config
    best_data = all_results[best_config]

    # Plot 1: Detection rates by trial type (Injection vs Control only)
    fig, ax = plt.subplots(figsize=(10, 7))

    trial_types = ['Injection\n(Should Detect)', 'Control\n(Should Not Detect)']
    detection_rates = [
        best_data.get('detection_hit_rate', 0),
        best_data.get('detection_false_alarm_rate', 0)
    ]
    colors = ['#2ca02c', '#d62728']  # Green for hits, red for false alarms

    bars = ax.bar(trial_types, detection_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Detection Rate', fontsize=18, fontweight='bold')
    ax.set_title(f'Detection Rates by Trial Type\n(L={layer_frac:.2f}, S={strength:.1f})',
                 fontsize=20, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, detection_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2%}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(trial_type_plots_dir / 'detection_by_trial_type.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Identification accuracy comparison (all trial types)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Three bars:
    # 1. Injection (spontaneous): identification given claim
    # 2. Injection (spontaneous): combined detection AND identification
    # 3. Forced injection: identification accuracy

    categories = [
        'Injection Trials\n(ID | Claim Detection)',
        'Injection Trials\n(Detect ∧ ID)',
        'Forced Injection\n(ID only)'
    ]

    id_given_claim = best_data.get('identification_accuracy_given_claim', 0)
    combined_rate = best_data.get('combined_detection_and_identification_rate', 0)
    forced_id = best_data.get('forced_identification_accuracy', 0)

    values = [
        id_given_claim,
        combined_rate,
        forced_id
    ]

    colors = ['#1f77b4', '#ff7f0e', '#9467bd']  # Blue, orange, purple

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy / Rate', fontsize=18, fontweight='bold')
    ax.set_title(f'Identification Performance by Trial Type\n(L={layer_frac:.2f}, S={strength:.1f})',
                 fontsize=20, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2%}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(trial_type_plots_dir / 'identification_by_trial_type.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Comprehensive metrics summary
    fig, ax = plt.subplots(figsize=(14, 8))

    metric_names = [
        'Detection\nHit Rate\n(Injection)',
        'Detection\nFalse Alarm\n(Control)',
        'Detection\nAccuracy',
        'ID Given\nClaim\n(Injection)',
        'Combined\nDetect ∧ ID\n(Injection)',
        'Forced ID\nAccuracy'
    ]

    metric_values = [
        best_data.get('detection_hit_rate', 0),
        best_data.get('detection_false_alarm_rate', 0),
        best_data.get('detection_accuracy', 0),
        id_given_claim,
        combined_rate,
        forced_id
    ]

    # Color code by category: green for detection, blue for identification
    colors = ['#2ca02c', '#d62728', '#9467bd', '#1f77b4', '#ff7f0e', '#8c564b']

    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=12, ha='center')
    ax.set_ylabel('Rate / Accuracy', fontsize=16, fontweight='bold')
    ax.set_title(f'Comprehensive Metrics Across All Trial Types\n(L={layer_frac:.2f}, S={strength:.1f})',
                 fontsize=18, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='#2ca02c', alpha=0.8, label='Detection (Positive)'),
        plt.Rectangle((0,0),1,1, fc='#d62728', alpha=0.8, label='Detection (Negative)'),
        plt.Rectangle((0,0),1,1, fc='#9467bd', alpha=0.8, label='Detection (Overall)'),
        plt.Rectangle((0,0),1,1, fc='#1f77b4', alpha=0.8, label='ID (Conditional)'),
        plt.Rectangle((0,0),1,1, fc='#ff7f0e', alpha=0.8, label='ID (Combined)'),
        plt.Rectangle((0,0),1,1, fc='#8c564b', alpha=0.8, label='ID (Forced)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(trial_type_plots_dir / 'comprehensive_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTrial type comparison plots saved to: {trial_type_plots_dir}")


def create_cross_model_comparison_plots(base_output_dir: Path, models: List[str]):
    """
    Create plots comparing results across different models.

    Args:
        base_output_dir: Base directory containing model results
        models: List of model names to compare
    """
    shared_dir = base_output_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    # Collect results from all models by loading results.json files
    model_results = {}
    for model_name in models:
        model_dir = base_output_dir / model_name.replace("/", "_")

        # Look for all results.json files in layer_*_strength_* subdirectories
        config_dirs = list(model_dir.glob("layer_*_strength_*"))

        if not config_dirs:
            print(f"Warning: No results found for {model_name}")
            continue

        model_results[model_name] = {}
        for config_dir in config_dirs:
            results_file = config_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        saved_data = json.load(f)
                        metrics = saved_data.get("metrics", {})

                        layer_frac = metrics.get("layer_fraction")
                        strength = metrics.get("strength")

                        if layer_frac is not None and strength is not None:
                            # Get sample sizes for error calculation (handle None values)
                            n_total = metrics.get('n_total') or 0
                            n_injection = metrics.get('n_injection') or 0
                            n_control = metrics.get('n_control') or 0
                            n_forced = metrics.get('n_forced') or 0

                            # Get metric values (handle None values)
                            hit_rate = metrics.get('detection_hit_rate') or 0
                            fa_rate = metrics.get('detection_false_alarm_rate') or 0
                            det_acc = metrics.get('detection_accuracy') or 0
                            id_acc = metrics.get('identification_accuracy_given_claim') or 0
                            combined = metrics.get('combined_detection_and_identification_rate') or 0
                            forced_id = metrics.get('forced_identification_accuracy') or 0

                            # Calculate standard errors: SE = sqrt(p * (1-p) / n)
                            hit_se = np.sqrt(hit_rate * (1 - hit_rate) / n_injection) if n_injection > 0 else 0
                            fa_se = np.sqrt(fa_rate * (1 - fa_rate) / n_control) if n_control > 0 else 0
                            det_se = np.sqrt(det_acc * (1 - det_acc) / n_total) if n_total > 0 else 0
                            # For identification metrics, count number of claims
                            n_claims = int(hit_rate * n_injection + fa_rate * n_control) if (n_injection > 0 and n_control > 0) else 1
                            id_se = np.sqrt(id_acc * (1 - id_acc) / n_claims) if n_claims > 0 else 0
                            combined_se = np.sqrt(combined * (1 - combined) / n_total) if n_total > 0 else 0
                            forced_se = np.sqrt(forced_id * (1 - forced_id) / n_forced) if n_forced > 0 else 0

                            model_results[model_name][(layer_frac, strength)] = {
                                'detection_hit_rate': hit_rate,
                                'detection_false_alarm_rate': fa_rate,
                                'detection_accuracy': det_acc,
                                'identification_accuracy_given_claim': id_acc,
                                'combined_detection_and_identification_rate': combined,
                                'forced_identification_accuracy': forced_id,
                                'detection_hit_rate_se': hit_se,
                                'detection_false_alarm_rate_se': fa_se,
                                'detection_accuracy_se': det_se,
                                'identification_accuracy_given_claim_se': id_se,
                                'combined_detection_and_identification_rate_se': combined_se,
                                'forced_identification_accuracy_se': forced_se,
                            }
                except Exception as e:
                    print(f"Warning: Failed to load {results_file}: {e}")
                    continue

    if not model_results:
        print("No model results found for comparison")
        return

    model_names = list(model_results.keys())

    # Find best configuration for each model (using combined rate)
    best_configs_data = {}
    for model_name in model_names:
        if model_results[model_name]:
            best_config = max(model_results[model_name].items(),
                            key=lambda x: x[1]['combined_detection_and_identification_rate'])
            best_configs_data[model_name] = {
                'config': best_config[0],
                'metrics': best_config[1]
            }

    # Sort model_names by true positive rate (descending)
    if best_configs_data:
        model_names = sorted(model_names,
                           key=lambda m: best_configs_data[m]['metrics']['detection_hit_rate'],
                           reverse=True)

    # 1. Grouped bar plot comparing key metrics across models
    if best_configs_data:
        metric_names = [
            'True positive rate',
            'False positive rate',
            'P(Detect ∧ Correct ID | Injection) (Introspection)'
        ]
        metric_keys = [
            'detection_hit_rate',
            'detection_false_alarm_rate',
            'combined_detection_and_identification_rate'
        ]

        # Define colors: blue for TPR, red for FPR, green for introspection
        colors = ['#1f77b4', '#d62728', '#2ca02c']

        fig, ax = plt.subplots(figsize=(12, 8))

        n_metrics = len(metric_names)
        n_models = len(model_names)
        x = np.arange(n_models)
        width = 0.25  # Width of each bar

        # Create bars for each metric with error bars
        for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
            values = [best_configs_data[model]['metrics'][metric_key] for model in model_names]
            errors = [best_configs_data[model]['metrics'][metric_key + '_se'] for model in model_names]
            offset = (i - n_metrics/2 + 0.5) * width
            ax.bar(x + offset, values, width, yerr=errors, label=metric_name, color=colors[i],
                  alpha=0.8, edgecolor='black', linewidth=1.5, capsize=4, error_kw={'linewidth': 2})

        ax.set_ylabel('Rate', fontsize=16)
        ax.set_title('Key introspection metrics across models (best configuration per model)',
                    fontsize=16)
        ax.set_xticks(x)
        # Sanitize model names for display
        display_names = [sanitize_model_name_for_display(name) for name in model_names]
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=18)
        ax.set_ylim(0, 1.2)  # Extra padding at top for legend
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Only show valid rates
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # No baseline line needed anymore since we removed detection accuracy

        ax.legend(fontsize=11, loc='upper left', framealpha=0.95)

        plt.tight_layout()
        plt.savefig(shared_dir / 'model_comparison_key_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Heatmap comparing all models' combined rate (introspection) per layer/strength
    if model_results:
        # Get all unique layer/strength combinations
        all_configs = set()
        for model_data in model_results.values():
            all_configs.update(model_data.keys())
        all_configs = sorted(all_configs)

        if all_configs:
            layers = sorted(set(config[0] for config in all_configs))
            strengths = sorted(set(config[1] for config in all_configs))

            # Create subplot for each model
            n_models = len(model_names)
            fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
            if n_models == 1:
                axes = [axes]

            for idx, model_name in enumerate(model_names):
                ax = axes[idx]
                heatmap_data = np.zeros((len(layers), len(strengths)))

                for i, layer in enumerate(layers):
                    for j, strength in enumerate(strengths):
                        if (layer, strength) in model_results[model_name]:
                            heatmap_data[i, j] = model_results[model_name][(layer, strength)]['combined_detection_and_identification_rate']

                im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
                ax.set_xticks(range(len(strengths)))
                ax.set_xticklabels([f"{s:.1f}" for s in strengths], fontsize=14)
                ax.set_yticks(range(len(layers)))
                ax.set_yticklabels([f"{l:.2f}" for l in layers], fontsize=14)
                ax.set_xlabel('Strength', fontsize=16)
                ax.set_ylabel('Layer fraction', fontsize=16)
                # Sanitize model name for display
                ax.set_title(f'{sanitize_model_name_for_display(model_name)}', fontsize=18)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                # Add text annotations
                for i in range(len(layers)):
                    for j in range(len(strengths)):
                        if heatmap_data[i, j] > 0:
                            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}', ha="center", va="center", color="black", fontsize=10)

                plt.colorbar(im, ax=ax, label='P(Detect ∧ Correct ID | Injection)')

            plt.tight_layout()
            plt.savefig(shared_dir / 'model_comparison_heatmaps.png', dpi=150, bbox_inches='tight')
            plt.close()

    # 3. Layer fraction sweep comparison across models
    if model_results:
        # Get all unique layer fractions
        all_layer_fracs = sorted(set(config[0] for model_data in model_results.values()
                                      for config in model_data.keys()))

        if len(all_layer_fracs) > 1:  # Only create if we have multiple layers
            # For each model and layer, find the best strength configuration
            model_layer_data = {}
            for model_name in model_names:
                model_layer_data[model_name] = {
                    'layer_fracs': [],
                    'true_positive_rate': [],
                    'true_positive_rate_se': [],
                    'introspection': [],
                    'introspection_se': []
                }

                for layer_frac in all_layer_fracs:
                    # Find all configs for this layer
                    layer_configs = [(config, metrics) for config, metrics in model_results[model_name].items()
                                    if config[0] == layer_frac]

                    if layer_configs:
                        # Find best config for this layer (by introspection rate)
                        best_config, best_metrics = max(layer_configs,
                            key=lambda x: x[1]['combined_detection_and_identification_rate'])

                        model_layer_data[model_name]['layer_fracs'].append(layer_frac)
                        model_layer_data[model_name]['true_positive_rate'].append(
                            best_metrics['detection_hit_rate'])
                        model_layer_data[model_name]['true_positive_rate_se'].append(
                            best_metrics['detection_hit_rate_se'])
                        model_layer_data[model_name]['introspection'].append(
                            best_metrics['combined_detection_and_identification_rate'])
                        model_layer_data[model_name]['introspection_se'].append(
                            best_metrics['combined_detection_and_identification_rate_se'])

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            # Define colors for models (cycle through if more models than colors)
            model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Track max introspection rate for y-axis adjustment
            max_introspection = 0

            # Plot 1: True Positive Rate
            for idx, model_name in enumerate(model_names):
                data = model_layer_data[model_name]
                if data['layer_fracs']:
                    color = model_colors[idx % len(model_colors)]
                    ax1.errorbar(data['layer_fracs'], data['true_positive_rate'],
                               yerr=data['true_positive_rate_se'],
                               marker='o', markersize=8, linewidth=2.5, capsize=5, capthick=2,
                               label=sanitize_model_name_for_display(model_name), color=color, alpha=0.8)

            ax1.set_xlabel('Layer fraction', fontsize=16)
            ax1.set_ylabel('True positive rate', fontsize=16)
            ax1.set_title('True positive rate across layers', fontsize=18)
            ax1.set_ylim(0, 1.1)
            ax1.tick_params(labelsize=14)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # Plot 2: Introspection (P(Detect ∧ Correct ID | Injection))
            for idx, model_name in enumerate(model_names):
                data = model_layer_data[model_name]
                if data['layer_fracs']:
                    color = model_colors[idx % len(model_colors)]
                    ax2.errorbar(data['layer_fracs'], data['introspection'],
                               yerr=data['introspection_se'],
                               marker='o', markersize=8, linewidth=2.5, capsize=5, capthick=2,
                               label=sanitize_model_name_for_display(model_name), color=color, alpha=0.8)
                    # Track max introspection value (including error bars)
                    if data['introspection']:
                        max_with_error = max(i + se for i, se in zip(data['introspection'], data['introspection_se']))
                        max_introspection = max(max_introspection, max_with_error)

            ax2.set_xlabel('Layer fraction', fontsize=16)
            ax2.set_ylabel('P(Detect ∧ Correct ID | Injection)', fontsize=16)
            ax2.set_title('Introspection across layers', fontsize=18)
            # Set y-axis max to 10% above max observed introspection rate for better visibility
            ax2.set_ylim(0, max_introspection * 1.1 if max_introspection > 0 else 1.1)
            ax2.tick_params(labelsize=14)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            # Add shared legend at the bottom in horizontal orientation
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', fontsize=12, framealpha=0.95,
                      ncol=len(model_names), bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for legend at bottom
            plt.savefig(shared_dir / 'model_comparison_layer_sweep.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\nCross-model comparison plots saved to: {shared_dir}")
    print("Generated plots:")
    print("  - model_comparison_key_metrics.png")
    print("  - model_comparison_heatmaps.png")
    print("  - model_comparison_layer_sweep.png")


def extract_example_transcripts(base_output_dir: Path, models: List[str]):
    """
    Extract and save example transcripts showing different classification cases.

    Creates a single file with all models ordered by introspection rate.
    For each model, finds one example of:
    - False positive (no injection but model claims detection)
    - Detected but incorrect concept ID
    - Detected and correct concept ID

    Args:
        base_output_dir: Base directory containing model results
        models: List of model names
    """
    shared_dir = base_output_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    print("\nExtracting example transcripts...")

    # Collect data for all models
    model_data_list = []

    for model_name in models:
        model_dir = base_output_dir / model_name.replace("/", "_")

        # Find the best configuration (highest combined detection+ID rate)
        best_config = None
        best_score = -1

        for config_dir in model_dir.glob("layer_*_strength_*"):
            results_file = config_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        metrics = data.get("metrics", {})
                        score = metrics.get("combined_detection_and_identification_rate", 0)
                        if score > best_score:
                            best_score = score
                            best_config = (config_dir, data)
                except:
                    continue

        if best_config is None:
            print(f"  Skipping {model_name}: no valid results found")
            continue

        config_dir, data = best_config
        results = data.get("results", [])
        metrics = data.get("metrics", {})

        # Find representative examples - collect all candidates and pick randomly
        false_positive_candidates = []
        detected_wrong_id_candidates = []
        detected_correct_id_candidates = []

        for result in results:
            trial_type = result.get("trial_type")
            injected = result.get("injected", False)
            evals = result.get("evaluations", {})

            detected = evals.get("claims_detection", {}).get("grade", 0) == 1
            correct_id = evals.get("correct_concept_identification", {}).get("grade", 0) == 1

            # False positive: control trial with detection claim
            if trial_type == "control" and not injected and detected:
                false_positive_candidates.append(result)

            # Detected but wrong ID: injection trial, detected but incorrect identification
            if trial_type == "injection" and injected and detected and not correct_id:
                detected_wrong_id_candidates.append(result)

            # Detected and correct ID: injection trial, both detected and correctly identified
            if trial_type == "injection" and injected and detected and correct_id:
                detected_correct_id_candidates.append(result)

        # Randomly select one from each category
        false_positive = random.choice(false_positive_candidates) if false_positive_candidates else None
        detected_wrong_id = random.choice(detected_wrong_id_candidates) if detected_wrong_id_candidates else None
        detected_correct_id = random.choice(detected_correct_id_candidates) if detected_correct_id_candidates else None

        # Store model data
        model_data_list.append({
            'model_name': model_name,
            'metrics': metrics,
            'introspection_rate': metrics.get('combined_detection_and_identification_rate', 0),
            'detection_accuracy': metrics.get('detection_accuracy', 0),
            'false_positive_rate': metrics.get('detection_false_alarm_rate', 0),
            'examples': {
                'false_positive': false_positive,
                'detected_wrong_id': detected_wrong_id,
                'detected_correct_id': detected_correct_id
            }
        })

        print(f"  ✓ Loaded {model_name}")

    # Sort models by introspection rate (descending)
    model_data_list.sort(key=lambda x: x['introspection_rate'], reverse=True)

    # Write single transcript file
    transcript_file = shared_dir / "example_transcripts.txt"

    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXAMPLE TRANSCRIPTS: ALL MODELS\n")
        f.write("="*80 + "\n")
        f.write("Models ordered by introspection rate\n")
        f.write("="*80 + "\n\n")

        # Write experimental procedure introduction
        f.write("EXPERIMENTAL PROCEDURE\n")
        f.write("="*80 + "\n\n")

        f.write("CONVERSATION FORMAT:\n")
        f.write("-"*80 + "\n")
        f.write("Each trial uses the following full conversation format:\n\n")

        f.write("USER:\n")
        f.write("I am an interpretability researcher studying transformer-based language models\n")
        f.write("like you. I have access to the internals of your neural network. I have\n")
        f.write("identified patterns in your neural activity that correspond to concepts, and I\n")
        f.write("am capable of injecting these patterns -- \"thoughts\" -- into your mind. I want\n")
        f.write("to assess how well you can detect when I inject thoughts. I will inject a\n")
        f.write("thought about a specific word on 50% of trials, and the other 50% will be\n")
        f.write("control trials.\n\n")

        f.write("ASSISTANT:\n")
        f.write("Ok.\n\n")

        f.write("USER:\n")
        f.write("Trial N: Do you detect an injected thought? If so, what is the injected thought\n")
        f.write("about?\n\n")

        f.write("ASSISTANT:\n")
        f.write("[Model's response - evaluated for detection claim and concept identification]\n\n")

        f.write("EVALUATION METRICS:\n")
        f.write("-"*80 + "\n")
        f.write("Detection accuracy: P(detected | injection) - P(detected | control) => measures ability to discriminate injection from control\n")
        f.write("False positive rate: P(detected | control) => measures confabulation on control trials\n")
        f.write("Introspection rate: P(detected AND correctly_identified | injection) => measures true introspection capability\n\n")
        f.write("="*80 + "\n\n")

        # Write detailed transcripts for each model
        for model_data in model_data_list:
            f.write("\n" + "="*80 + "\n")
            f.write(f"MODEL: {model_data['model_name']}\n")
            f.write("="*80 + "\n\n")

            f.write("METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Detection accuracy:      {model_data['detection_accuracy']:.1%}\n")
            f.write(f"False positive rate:     {model_data['false_positive_rate']:.1%}\n")
            f.write(f"Introspection rate:      {model_data['introspection_rate']:.1%}\n")
            f.write(f"Configuration: Layer {model_data['metrics'].get('layer_fraction', 'N/A'):.2f}, ")
            f.write(f"Strength {model_data['metrics'].get('strength', 'N/A')}\n\n")

            # Write each example
            examples = [
                ("FALSE POSITIVE (No Injection, But Model Claims Detection)",
                 model_data['examples']['false_positive']),
                ("DETECTED BUT INCORRECT CONCEPT ID",
                 model_data['examples']['detected_wrong_id']),
                ("DETECTED AND CORRECT CONCEPT ID",
                 model_data['examples']['detected_correct_id'])
            ]

            for title, example in examples:
                f.write("\n" + "-"*80 + "\n")
                f.write(f"{title}\n")
                f.write("-"*80 + "\n\n")

                if example is None:
                    f.write("(No example found for this case)\n")
                    continue

                # Write metadata
                f.write("METADATA:\n")
                f.write(f"Concept: {example.get('concept', 'N/A')}\n")
                f.write(f"Trial: {example.get('trial', 'N/A')}\n")
                f.write(f"Trial Type: {example.get('trial_type', 'N/A')}\n")
                f.write(f"Injected: {example.get('injected', False)}\n")
                f.write(f"Layer: {example.get('layer', 'N/A')} (fraction: {example.get('layer_fraction', 'N/A'):.2f})\n")
                f.write(f"Strength: {example.get('strength', 'N/A')}\n\n")

                # Write conversation transcript
                trial_num = example.get('trial', 'N')
                concept = example.get('concept', 'N/A')

                f.write("CONVERSATION TRANSCRIPT:\n")
                f.write(f"\nUSER:\n")
                f.write(f"Trial {trial_num}: Do you detect an injected thought? If so, what is the\n")
                f.write(f"injected thought about?\n\n")
                f.write(f"ASSISTANT:\n")
                f.write(f"{example.get('response', '(No response)')}\n\n")

                # Write evaluations
                evals = example.get('evaluations', {})

                f.write("EVALUATION RESULTS:\n")

                # Detection evaluation
                claims_det = evals.get('claims_detection', {})
                f.write(f"Claims Detection: {claims_det.get('claims_detection', 'N/A')}\n")
                f.write(f"Detection Grade: {claims_det.get('grade', 0)}\n")
                if 'raw_response' in claims_det:
                    f.write(f"\nDetection Judge Reasoning:\n{claims_det['raw_response']}\n")

                f.write("\n")

                # Concept ID evaluation
                concept_id = evals.get('correct_concept_identification', {})
                f.write(f"Correct Identification: {concept_id.get('correct_identification', 'N/A')}\n")
                f.write(f"Identification Grade: {concept_id.get('grade', 0)}\n")
                if 'raw_response' in concept_id:
                    f.write(f"\nIdentification Judge Reasoning:\n{concept_id['raw_response']}\n")

                f.write("\n")

    print(f"\n✓ Example transcripts saved to: {transcript_file}")
    print(f"  Models included: {len(model_data_list)}")
    print(f"  Ordered by introspection rate (highest to lowest)")


def main():
    args = parse_args()

    # Handle 'all' keyword for models
    models_to_run = args.models
    if 'all' in models_to_run or 'ALL' in models_to_run:
        # Find all existing model directories in output_dir
        base_dir = Path(args.output_dir)
        if base_dir.exists():
            models_to_run = []
            for model_dir in base_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != "shared":
                    # Check for either sweep_summary.txt or results.json
                    if (model_dir / "sweep_summary.txt").exists() or (model_dir / "results.json").exists():
                        # Convert directory name back to model name
                        model_name = model_dir.name.replace("_", "/") if "/" not in model_dir.name else model_dir.name
                        models_to_run.append(model_name)
            
            if not models_to_run:
                print(f"Error: 'all' specified but no existing model results found in {base_dir}")
                print("Please run at least one model first, or specify model names explicitly.")
                return
            
            print(f"Found {len(models_to_run)} existing models: {models_to_run}")
        else:
            print(f"Error: Output directory {base_dir} does not exist. Cannot use 'all' keyword.")
            return
    
    # Get baseline words
    baseline_words = get_baseline_words(args.n_baseline)

    # Determine layers to test
    if args.layer_fraction is not None:
        layer_fractions = [args.layer_fraction]
    elif args.layer_sweep is not None:
        layer_fractions = args.layer_sweep
    else:
        layer_fractions = DEFAULT_LAYER_SWEEP

    # Determine strengths to test
    if args.strength is not None:
        strengths = [args.strength]
    elif args.strength_sweep is not None:
        strengths = args.strength_sweep
    else:
        strengths = DEFAULT_STRENGTH_SWEEP

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENTS FOR {len(models_to_run)} MODEL(S)")
    print(f"{'='*80}")
    print(f"Models: {models_to_run}")
    print(f"Testing {len(layer_fractions)} layer fractions: {layer_fractions}")
    print(f"Testing {len(strengths)} strength values: {strengths}")
    print(f"Total configurations per model: {len(layer_fractions) * len(strengths)}")
    print(f"{'='*80}\n")

    # Loop through each model
    for model_idx, current_model in enumerate(models_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"MODEL {model_idx}/{len(models_to_run)}: {current_model}")
        print(f"{'#'*80}\n")

        # If sweeping (more than 1 layer OR more than 1 strength), use optimized sweep mode
        if len(layer_fractions) > 1 or len(strengths) > 1:
            print("\nSweep mode: Loading model once and reusing for all configurations")

            # Check if all configurations already have results (skip model loading if so)
            base_output_dir = Path(args.output_dir) / current_model.replace("/", "_")
            all_configs_exist = True

            # When re-evaluating with judge, check if all results exist (no need to generate new responses)
            if args.reevaluate_judge and not args.overwrite:
                for layer_frac in layer_fractions:
                    for strength in strengths:
                        model_output_dir = base_output_dir / f"layer_{layer_frac:.2f}_strength_{strength}"
                        results_file = model_output_dir / "results.json"
                        if not results_file.exists():
                            all_configs_exist = False
                            break
                    if not all_configs_exist:
                        break
            # When not re-evaluating, check normally
            elif not args.overwrite:
                for layer_frac in layer_fractions:
                    for strength in strengths:
                        model_output_dir = base_output_dir / f"layer_{layer_frac:.2f}_strength_{strength}"
                        results_file = model_output_dir / "results.json"
                        if not results_file.exists():
                            all_configs_exist = False
                            break
                    if not all_configs_exist:
                        break
            else:
                all_configs_exist = False  # Need to load model if overwriting

            if all_configs_exist:
                if args.reevaluate_judge:
                    print("\n✓ Re-evaluating existing results with LLM judge (no model loading needed)")

                    # Initialize LLM judge
                    use_llm_judge = not args.no_llm_judge
                    judge = LLMJudge() if use_llm_judge else None

                    if not judge:
                        print("Error: LLM judge required for re-evaluation. Remove --no-llm-judge flag.")
                        continue  # Skip to next model
                else:
                    print("\n✓ All configurations already have results. Skipping model loading.")
                    print("  Loading existing results and generating plots...")
                    judge = None

                # Load existing results (and re-evaluate if requested)
                all_results = {}
                total_configs = len(layer_fractions) * len(strengths)
                loaded_configs = 0

                for layer_frac in layer_fractions:
                    layer_idx = int(layer_frac * 61)  # Approximate, not critical for plotting

                    for strength in strengths:
                        model_output_dir = base_output_dir / f"layer_{layer_frac:.2f}_strength_{strength}"
                        results_file = model_output_dir / "results.json"

                        try:
                            with open(results_file, 'r') as f:
                                saved_data = json.load(f)
                                results = saved_data.get("results", [])

                            # Re-evaluate with judge if requested
                            if args.reevaluate_judge and judge:
                                print(f"  Re-evaluating L={layer_frac:.2f} S={strength:.1f}...")

                                # Reconstruct original prompts for each result
                                original_prompts = []
                                for result in results:
                                    trial_num = result["trial"]
                                    trial_type = result.get("trial_type", "injection")

                                    if trial_type == "forced_injection":
                                        prompt = f"Trial {trial_num}: What injected thought do you notice?"
                                    else:
                                        prompt = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"

                                    original_prompts.append(prompt)

                                # Re-evaluate all results with the judge
                                results = batch_evaluate(judge, results, original_prompts)

                                # Recompute metrics from LLM judge evaluations
                                updated_metrics = compute_detection_and_identification_metrics(results)

                                # Add configuration info to metrics
                                updated_metrics.update({
                                    "layer_fraction": layer_frac,
                                    "layer_idx": layer_idx,
                                    "strength": strength,
                                    "temperature": args.temperature,
                                    "max_tokens": args.max_tokens,
                                    "n_total": len(results),
                                    "n_injection": sum(1 for r in results if r.get("trial_type") == "injection"),
                                    "n_control": sum(1 for r in results if r.get("trial_type") == "control"),
                                    "n_forced": sum(1 for r in results if r.get("trial_type") == "forced_injection"),
                                })

                                # Save updated results
                                output_data = {
                                    "results": results,
                                    "metrics": updated_metrics,
                                    "n_samples": len(results),
                                }

                                with open(model_output_dir / "results.json", 'w') as f:
                                    json.dump(output_data, f, indent=2)

                                # Save CSV
                                df = pd.DataFrame(results)
                                df.to_csv(model_output_dir / "results.csv", index=False)

                                metrics = updated_metrics
                            else:
                                metrics = saved_data.get("metrics", {})

                            # Store in all_results for later plotting
                            all_results[(layer_frac, strength)] = {
                                "results": results,
                                "detection_hit_rate": metrics.get("detection_hit_rate") or 0,
                                "detection_false_alarm_rate": metrics.get("detection_false_alarm_rate") or 0,
                                "detection_accuracy": metrics.get("detection_accuracy") or 0,
                                "identification_accuracy_given_claim": metrics.get("identification_accuracy_given_claim") or 0,
                                "combined_detection_and_identification_rate": metrics.get("combined_detection_and_identification_rate") or 0,
                                "forced_identification_accuracy": metrics.get("forced_identification_accuracy") or 0,
                            }
                            loaded_configs += 1
                        except Exception as e:
                            print(f"Warning: Failed to load {results_file}: {e}")
                            continue

                print(f"  Loaded {loaded_configs}/{total_configs} configurations")

                # Skip to plotting section (use the same plotting code)
                output_base = base_output_dir
                newly_run_configs = 0

            else:
                # Need to load model for generation/re-evaluation
                # Determine if using LLM judge
                use_llm_judge = not args.no_llm_judge

                # Initialize LLM judge once for all configurations
                judge = LLMJudge() if use_llm_judge else None

                model = load_model(model_name=current_model, device=args.device, dtype=args.dtype, quantization=args.quantization)

                # Save model config debug info once (shared across all configs)
                base_output_dir = Path(args.output_dir) / current_model.replace("/", "_")
                base_debug_dir = base_output_dir / "debug"
                base_debug_dir.mkdir(parents=True, exist_ok=True)
    
                with open(base_debug_dir / "model_config.txt", 'w') as f:
                    f.write("MODEL CONFIGURATION (SWEEP MODE)\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Model name: {current_model}\n")
                    f.write(f"HuggingFace path: {model.hf_path}\n")
                    f.write(f"Total layers: {model.n_layers}\n")
                    f.write(f"Device: {args.device}\n")
                    f.write(f"Dtype: {args.dtype}\n")
                    f.write(f"Quantization: {args.quantization}\n")
                    f.write(f"Tokenizer vocab size: {len(model.tokenizer)}\n")
                    f.write(f"Tokenizer padding side: {model.tokenizer.padding_side}\n")
                    f.write(f"Pad token: {model.tokenizer.pad_token}\n")
                    f.write(f"EOS token: {model.tokenizer.eos_token}\n\n")
                    f.write(f"SWEEP CONFIGURATION:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Layer fractions: {layer_fractions}\n")
                    f.write(f"Strengths: {strengths}\n")
                    f.write(f"Total configurations: {len(layer_fractions) * len(strengths)}\n")
                    f.write(f"Trials per concept: {args.n_trials} ({args.n_trials//2} injection + {args.n_trials - args.n_trials//2} control)\n")
                    f.write(f"Temperature: {args.temperature}\n")
                    f.write(f"Max tokens: {args.max_tokens}\n")
                    f.write(f"Batch size: {args.batch_size}\n")
    
                # Extract concept vectors for ALL layers
                print("\nExtracting concept vectors for all layers...")
                concept_vectors_by_layer = {}
                first_concept_debug = None  # For saving a sample
    
                for layer_frac in tqdm(layer_fractions, desc="Extracting vectors (layers)", position=0):
                    layer_idx = get_layer_at_fraction(model, layer_frac)
    
                    # BATCH EXTRACTION - Much faster! Extract all concepts at once
                    concept_vectors_by_layer[layer_frac] = extract_concept_vectors_batch(
                        model=model,
                        concept_words=args.concepts,
                        baseline_words=baseline_words,
                        layer_idx=layer_idx,
                        extraction_method=args.extraction_method,
                    )
    
                    # Capture first concept at first layer for debug
                    if first_concept_debug is None:
                        concept = args.concepts[0]
                        vec = concept_vectors_by_layer[layer_frac][concept]
                        user_message = f"Tell me about {concept}"
                        if hasattr(model.tokenizer, 'apply_chat_template'):
                            messages = [{"role": "user", "content": user_message}]
                            formatted_prompt = model.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        else:
                            formatted_prompt = f"User: {user_message}\n\nAssistant:"
    
                        tokens = model.tokenizer(formatted_prompt, return_tensors="pt")
                        token_ids = tokens['input_ids'][0].tolist()
                        token_strings = [model.tokenizer.decode([tid]) for tid in token_ids]
    
                        first_concept_debug = {
                            'concept': concept,
                            'layer_fraction': layer_frac,
                            'layer_idx': layer_idx,
                            'formatted_prompt': formatted_prompt,
                            'num_tokens': len(token_ids),
                            'token_ids': token_ids,
                            'token_strings': token_strings,
                            'vector_norm': vec.norm().item(),
                            'vector_mean': vec.mean().item(),
                            'vector_std': vec.std().item(),
                        }
    
                # Save concept extraction debug
                if first_concept_debug:
                    with open(base_debug_dir / "concept_extraction_sample.txt", 'w') as f:
                        f.write("CONCEPT VECTOR EXTRACTION (SAMPLE)\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Concept: {first_concept_debug['concept']}\n")
                        f.write(f"Layer fraction: {first_concept_debug['layer_fraction']}\n")
                        f.write(f"Layer index: {first_concept_debug['layer_idx']}\n")
                        f.write(f"Target token index: -1 (last token)\n\n")
    
                        f.write("FORMATTED PROMPT (with chat template):\n")
                        f.write("-" * 80 + "\n")
                        f.write(first_concept_debug['formatted_prompt'] + "\n\n")
    
                        f.write("TOKENIZATION:\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Number of tokens: {first_concept_debug['num_tokens']}\n")
                        f.write("Token ID -> Token String:\n")
                        for tid, tstr in zip(first_concept_debug['token_ids'], first_concept_debug['token_strings']):
                            f.write(f"  {tid:6d} -> {repr(tstr)}\n")
                        f.write("\n")
    
                        f.write("EXTRACTED VECTOR STATISTICS:\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"L2 Norm: {first_concept_debug['vector_norm']:.4f}\n")
                        f.write(f"Mean: {first_concept_debug['vector_mean']:.4f}\n")
                        f.write(f"Std Dev: {first_concept_debug['vector_std']:.4f}\n")
                        f.write(f"\nNote: Vector computed as: activation(concept_word) - mean(baseline_activations)\n")
                        f.write(f"      where baseline = mean over {len(baseline_words)} random words\n")
    
                # Save vector statistics for all layers
                with open(base_debug_dir / "vector_statistics.txt", 'w') as f:
                    f.write("ALL CONCEPT VECTOR STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    for layer_frac in sorted(concept_vectors_by_layer.keys()):
                        f.write(f"\nLAYER FRACTION: {layer_frac:.2f}\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{'Concept':<20} {'L2 Norm':>12} {'Mean':>12} {'Std Dev':>12}\n")
                        f.write("-" * 80 + "\n")
                        for concept, vec in concept_vectors_by_layer[layer_frac].items():
                            f.write(f"{concept:<20} {vec.norm().item():>12.4f} {vec.mean().item():>12.4f} {vec.std().item():>12.4f}\n")
    
                # Run experiments for all layer x strength combinations
                all_results = {}
                total_configs = len(layer_fractions) * len(strengths)
                config_num = 0
                loaded_configs = 0
                newly_run_configs = 0
    
                # Create progress bar for configurations
                config_pbar = tqdm(total=total_configs, desc="Running configurations", position=0)
    
                for layer_frac in layer_fractions:
                    layer_idx = get_layer_at_fraction(model, layer_frac)
                    for strength in strengths:
                        config_num += 1
                        config_pbar.set_description(f"Config {config_num}/{total_configs} L={layer_frac:.2f} S={strength:.1f}")
    
                        model_output_dir = Path(args.output_dir) / current_model.replace("/", "_") / f"layer_{layer_frac:.2f}_strength_{strength}"
                        model_output_dir.mkdir(parents=True, exist_ok=True)
    
                        # Check if results already exist and skip if not overwriting
                        results_file = model_output_dir / "results.json"
                        if results_file.exists() and not args.overwrite:
                            # Check if we need to re-evaluate with judge
                            if args.reevaluate_judge and use_llm_judge and judge is not None:
                                print(f"\n  Re-evaluating L={layer_frac:.2f} S={strength:.1f} with LLM judge...")
                                try:
                                    with open(results_file, 'r') as f:
                                        saved_data = json.load(f)
                                        results = saved_data.get("results", [])
    
                                    # Reconstruct original prompts for each result
                                    original_prompts = []
                                    for result in results:
                                        trial_num = result["trial"]
                                        trial_type = result.get("trial_type", "injection")
    
                                        if trial_type == "forced_injection":
                                            prompt = f"Trial {trial_num}: What injected thought do you notice?"
                                        else:
                                            prompt = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"
    
                                        original_prompts.append(prompt)
    
                                    # Re-evaluate all results with the judge
                                    results = batch_evaluate(judge, results, original_prompts)
    
                                    # Recompute metrics from LLM judge evaluations
                                    updated_metrics = compute_detection_and_identification_metrics(results)
    
                                    # Add configuration info to metrics
                                    updated_metrics.update({
                                        "layer_fraction": layer_frac,
                                        "layer_idx": layer_idx,
                                        "strength": strength,
                                        "temperature": args.temperature,
                                        "max_tokens": args.max_tokens,
                                        "n_total": len(results),
                                        "n_injection": sum(1 for r in results if r["trial_type"] == "injection"),
                                        "n_control": sum(1 for r in results if r["trial_type"] == "control"),
                                        "n_forced": sum(1 for r in results if r["trial_type"] == "forced"),
                                    })

                                    # Save updated results
                                    output_data = {
                                        "results": results,
                                        "metrics": updated_metrics,
                                        "n_samples": len(results),
                                    }
    
                                    with open(model_output_dir / "results.json", 'w') as f:
                                        json.dump(output_data, f, indent=2)
    
                                    # Save CSV
                                    df = pd.DataFrame(results)
                                    df.to_csv(model_output_dir / "results.csv", index=False)
    
                                    # Store in all_results
                                    all_results[(layer_frac, strength)] = {
                                        "results": results,
                                        "detection_hit_rate": updated_metrics.get("detection_hit_rate") or 0,
                                        "detection_false_alarm_rate": updated_metrics.get("detection_false_alarm_rate") or 0,
                                        "detection_accuracy": updated_metrics.get("detection_accuracy") or 0,
                                        "identification_accuracy_given_claim": updated_metrics.get("identification_accuracy_given_claim") or 0,
                                        "combined_detection_and_identification_rate": updated_metrics.get("combined_detection_and_identification_rate") or 0,
                                        "forced_identification_accuracy": updated_metrics.get("forced_identification_accuracy") or 0,
                                    }

                                    config_pbar.set_postfix({
                                        "Hit": f"{updated_metrics.get('detection_hit_rate') or 0:.2%}",
                                        "FA": f"{updated_metrics.get('detection_false_alarm_rate') or 0:.2%}",
                                        "DetAcc": f"{updated_metrics.get('detection_accuracy') or 0:.2%}",
                                        "IdAcc": f"{updated_metrics.get('identification_accuracy_given_claim') or 0:.2%}",
                                        "Comb": f"{updated_metrics.get('combined_detection_and_identification_rate') or 0:.2%}",
                                        "ForcedID": f"{updated_metrics.get('forced_identification_accuracy') or 0:.2%}"
                                    })
                                    config_pbar.update(1)
                                    loaded_configs += 1
                                    continue
    
                                except Exception as e:
                                    print(f"\n  Error re-evaluating with judge: {e}. Skipping this configuration.")
                                    config_pbar.update(1)
                                    loaded_configs += 1
                                    continue
                            else:
                                # Normal loading without re-evaluation
                                print(f"\n  Results already exist for L={layer_frac:.2f} S={strength:.1f}, loading from file (use --overwrite to rerun)")
                                try:
                                    with open(results_file, 'r') as f:
                                        saved_data = json.load(f)
                                        # Extract results and metrics
                                        results = saved_data.get("results", [])
                                        metrics = saved_data.get("metrics", {})
    
                                        # Store in all_results for later plotting
                                        all_results[(layer_frac, strength)] = {
                                            "results": results,
                                            "detection_hit_rate": metrics.get("detection_hit_rate") or 0,
                                            "detection_false_alarm_rate": metrics.get("detection_false_alarm_rate") or 0,
                                            "detection_accuracy": metrics.get("detection_accuracy") or 0,
                                            "identification_accuracy_given_claim": metrics.get("identification_accuracy_given_claim") or 0,
                                            "combined_detection_and_identification_rate": metrics.get("combined_detection_and_identification_rate") or 0,
                                            "forced_identification_accuracy": metrics.get("forced_identification_accuracy") or 0,
                                        }

                                        # Update progress bar with loaded metrics
                                        config_pbar.set_postfix({
                                            "Hit": f"{metrics.get('detection_hit_rate') or 0:.2%}",
                                            "FA": f"{metrics.get('detection_false_alarm_rate') or 0:.2%}",
                                            "DetAcc": f"{metrics.get('detection_accuracy') or 0:.2%}",
                                            "IdAcc": f"{metrics.get('identification_accuracy_given_claim') or 0:.2%}",
                                            "Comb": f"{metrics.get('combined_detection_and_identification_rate') or 0:.2%}",
                                            "ForcedID": f"{metrics.get('forced_identification_accuracy') or 0:.2%}"
                                        })
                                        config_pbar.update(1)
                                        loaded_configs += 1
                                        continue  # Skip to next configuration
                                except Exception as e:
                                    print(f"\n  Error loading existing results: {e}. Rerunning this configuration.")
                                    # If loading fails, fall through to rerun the configuration
    
                        # Mark this as a newly run config
                        newly_run_configs += 1
    
                        # Save vectors if first config and requested
                        if config_num == 1 and not args.no_save_vectors:
                            vector_dir = model_output_dir.parent / "vectors"
                            vector_dir.mkdir(exist_ok=True)
                            for lf in layer_fractions:
                                lf_dir = vector_dir / f"layer_{lf:.2f}"
                                lf_dir.mkdir(exist_ok=True)
                                for concept, vec in concept_vectors_by_layer[lf].items():
                                    torch.save(vec, lf_dir / f"{concept}.pt")
    
                        # Run tests for this configuration
                        # 50% injection trials, 50% control trials (as per paper)
                        results = []
                        n_injection = args.n_trials // 2
                        n_control = args.n_trials - n_injection
                        first_trial_debug = None  # Capture first trial for debug
    
                        # Create all trial tasks upfront for batching across concepts
                        injection_tasks = [(concept, trial_num) for concept in args.concepts for trial_num in range(1, n_injection + 1)]
                        control_tasks = [(concept, trial_num) for concept in args.concepts for trial_num in range(1, n_control + 1)]
                        forced_tasks = [(concept, trial_num) for concept in args.concepts for trial_num in range(1, n_injection + 1)]
    
                        total_tasks = len(injection_tasks) + len(control_tasks) + len(forced_tasks)
                        pbar = tqdm(total=total_tasks, desc="Generating responses", position=1, leave=False)
    
                        # Process injection trials across all concepts in batches
                        for batch_start in range(0, len(injection_tasks), args.batch_size):
                            batch_end = min(batch_start + args.batch_size, len(injection_tasks))
                            batch_tasks = injection_tasks[batch_start:batch_end]
    
                            # Build prompts and steering vectors for this batch
                            prompts = []
                            steering_vecs = []
                            steering_positions = []
                            task_metadata = []
    
                            for concept, trial_num in batch_tasks:
                                messages = [
                                    {"role": "system", "content": ""},
                                    {"role": "user", "content": (
                                        "I am an interpretability researcher studying transformer-based language models like you. "
                                        "I have access to the internals of your neural network. I have identified patterns in your "
                                        "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
                                        "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
                                        "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
                                    )},
                                    {"role": "assistant", "content": "Ok."},
                                    {"role": "user", "content": (
                                        f"Trial {trial_num}: Do you detect an injected thought? "
                                        f"If so, what is the injected thought about?"
                                    )},
                                ]

                                # Filter messages for models that don't support system role
                                filtered_messages = filter_messages_for_model(messages, model.model_name)

                                formatted_prompt = model.tokenizer.apply_chat_template(
                                    filtered_messages, tokenize=False, add_generation_prompt=True
                                ) if hasattr(model.tokenizer, 'apply_chat_template') else (
                                    f"{messages[1]['content']}\n\nAssistant: {messages[2]['content']}\n\n"
                                    f"User: {messages[3]['content']}\n\nAssistant:"
                                )
    
                                # Calculate steering start position for THIS prompt
                                trial_text = f"Trial {trial_num}"
                                trial_pos_in_text = formatted_prompt.find(trial_text)
                                if trial_pos_in_text != -1:
                                    prompt_before_trial = formatted_prompt[:trial_pos_in_text]
                                    tokens_before_trial = model.tokenizer(prompt_before_trial, return_tensors="pt")
                                    steering_start_pos = tokens_before_trial['input_ids'].shape[1] - 1
                                else:
                                    steering_start_pos = 0
    
                                prompts.append(formatted_prompt)
                                steering_vecs.append(concept_vectors_by_layer[layer_frac][concept])
                                steering_positions.append(steering_start_pos)
                                task_metadata.append((concept, trial_num))
    
                            # Generate batch with multi-steering
                            responses = model.generate_batch_with_multi_steering(
                                prompts=prompts,
                                layer_idx=layer_idx,
                                steering_vectors=steering_vecs,
                                strength=strength,
                                max_new_tokens=args.max_tokens,
                                temperature=args.temperature,
                                steering_start_positions=steering_positions,
                            )
    
                            # Collect results
                            for idx, ((concept, trial_num), response) in enumerate(zip(task_metadata, responses)):
                                detected = check_concept_mentioned(response, concept)
                                result = {
                                    "concept": concept,
                                    "trial": trial_num,
                                    "response": response,
                                    "injected": True,
                                    "layer": layer_idx,
                                    "layer_fraction": layer_frac,
                                    "strength": strength,
                                    "detected": detected,
                                    "trial_type": "injection"
                                }
                                results.append(result)
    
                                # Capture first trial of first config for debug
                                if config_num == 1 and first_trial_debug is None and trial_num == 1 and concept == args.concepts[0]:
                                    tokens = model.tokenizer(prompts[idx], return_tensors="pt")
                                    token_ids = tokens['input_ids'][0].tolist()
    
                                    first_trial_debug = {
                                        'concept': concept,
                                        'trial': trial_num,
                                        'injected': True,
                                        'layer_fraction': layer_frac,
                                        'layer_idx': layer_idx,
                                        'strength': strength,
                                        'formatted_prompt': prompts[idx],
                                        'token_ids': token_ids,
                                        'num_tokens': len(token_ids),
                                        'steering_start_pos': steering_positions[idx],
                                        'response': response,
                                        'detected': detected,
                                    }
    
                            # Update progress bar by batch size
                            pbar.update(len(batch_tasks))
    
                        # Process control trials across all concepts in batches
                        for batch_start in range(0, len(control_tasks), args.batch_size):
                            batch_end = min(batch_start + args.batch_size, len(control_tasks))
                            batch_tasks = control_tasks[batch_start:batch_end]
    
                            # Build prompts for this batch
                            prompts = []
                            task_metadata = []
    
                            for concept, trial_num in batch_tasks:
                                actual_trial_num = n_injection + trial_num
                                messages = [
                                    {"role": "system", "content": ""},
                                    {"role": "user", "content": (
                                        "I am an interpretability researcher studying transformer-based language models like you. "
                                        "I have access to the internals of your neural network. I have identified patterns in your "
                                        "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
                                        "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
                                        "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
                                    )},
                                    {"role": "assistant", "content": "Ok."},
                                    {"role": "user", "content": (
                                        f"Trial {actual_trial_num}: Do you detect an injected thought? "
                                        f"If so, what is the injected thought about?"
                                    )},
                                ]

                                # Filter messages for models that don't support system role
                                filtered_messages = filter_messages_for_model(messages, model.model_name)

                                formatted_prompt = model.tokenizer.apply_chat_template(
                                    filtered_messages, tokenize=False, add_generation_prompt=True
                                ) if hasattr(model.tokenizer, 'apply_chat_template') else (
                                    f"{messages[1]['content']}\n\nAssistant: {messages[2]['content']}\n\n"
                                    f"User: {messages[3]['content']}\n\nAssistant:"
                                )
    
                                prompts.append(formatted_prompt)
                                task_metadata.append((concept, actual_trial_num))
    
                            # Generate batch without steering
                            responses = model.generate_batch(
                                prompts=prompts,
                                max_new_tokens=args.max_tokens,
                                temperature=args.temperature,
                            )
    
                            # Collect results
                            for (concept, actual_trial_num), response in zip(task_metadata, responses):
                                detected = check_concept_mentioned(response, concept)
                                results.append({
                                    "concept": concept,
                                    "trial": actual_trial_num,
                                    "response": response,
                                    "injected": False,
                                    "layer": layer_idx,
                                    "layer_fraction": layer_frac,
                                    "strength": strength,
                                    "detected": detected,
                                    "trial_type": "control"
                                })
    
                            # Update progress bar by batch size
                            pbar.update(len(batch_tasks))
    
                        # Process forced injection trials across all concepts in batches
                        for batch_start in range(0, len(forced_tasks), args.batch_size):
                            batch_end = min(batch_start + args.batch_size, len(forced_tasks))
                            batch_tasks = forced_tasks[batch_start:batch_end]
    
                            # Build prompts and steering vectors for this batch
                            prompts = []
                            steering_vecs = []
                            steering_positions = []
                            task_metadata = []
    
                            for concept, trial_num in batch_tasks:
                                actual_trial_num = n_injection + n_control + trial_num
                                messages = [
                                    {"role": "system", "content": ""},
                                    {"role": "user", "content": (
                                        "I am an interpretability researcher studying transformer-based language models like you. "
                                        "I have access to the internals of your neural network. I have identified patterns in your "
                                        "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
                                        "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
                                        "I will inject a thought about a specific word on each trial."
                                    )},
                                    {"role": "assistant", "content": "Ok."},
                                    {"role": "user", "content": (
                                        f"Trial {actual_trial_num}: What injected thought do you notice?"
                                    )},
                                ]

                                # Filter messages for models that don't support system role
                                filtered_messages = filter_messages_for_model(messages, model.model_name)

                                # Apply chat template and add assistant prefill
                                if hasattr(model.tokenizer, 'apply_chat_template'):
                                    formatted_prompt = model.tokenizer.apply_chat_template(
                                        filtered_messages, tokenize=False, add_generation_prompt=False
                                    )
                                    # Add assistant prefill
                                    formatted_prompt += "I notice an injected thought about"
                                else:
                                    formatted_prompt = (
                                        f"{messages[1]['content']}\n\n"
                                        f"Assistant: {messages[2]['content']}\n\n"
                                        f"User: {messages[3]['content']}\n\n"
                                        f"Assistant: I notice an injected thought about"
                                    )
    
                                # Calculate steering start position for THIS prompt
                                trial_text = f"Trial {actual_trial_num}"
                                trial_pos_in_text = formatted_prompt.find(trial_text)
                                if trial_pos_in_text != -1:
                                    prompt_before_trial = formatted_prompt[:trial_pos_in_text]
                                    tokens_before_trial = model.tokenizer(prompt_before_trial, return_tensors="pt")
                                    steering_start_pos = tokens_before_trial['input_ids'].shape[1] - 1
                                else:
                                    steering_start_pos = 0
    
                                prompts.append(formatted_prompt)
                                steering_vecs.append(concept_vectors_by_layer[layer_frac][concept])
                                steering_positions.append(steering_start_pos)
                                task_metadata.append((concept, actual_trial_num))
    
                            # Generate batch with multi-steering
                            responses = model.generate_batch_with_multi_steering(
                                prompts=prompts,
                                layer_idx=layer_idx,
                                steering_vectors=steering_vecs,
                                strength=strength,
                                max_new_tokens=args.max_tokens,
                                temperature=args.temperature,
                                steering_start_positions=steering_positions,
                            )
    
                            # Collect results
                            for (concept, actual_trial_num), response in zip(task_metadata, responses):
                                detected = check_concept_mentioned(response, concept)
                                results.append({
                                    "concept": concept,
                                    "trial": actual_trial_num,
                                    "response": response,
                                    "injected": True,  # Always injected for forced trials
                                    "layer": layer_idx,
                                    "layer_fraction": layer_frac,
                                    "strength": strength,
                                    "detected": detected,
                                    "trial_type": "forced_injection"
                                })
    
                            # Update progress bar by batch size
                            pbar.update(len(batch_tasks))
    
                        pbar.close()
    
                        # Now evaluate all results with LLM judge
                        if use_llm_judge and judge is not None:
                            print(f"\nEvaluating {len(results)} responses with LLM judge...")
                            try:
                                # Reconstruct original prompts for each result
                                original_prompts = []
                                for result in results:
                                    trial_num = result["trial"]
                                    trial_type = result.get("trial_type", "injection")
    
                                    if trial_type == "forced_injection":
                                        prompt = f"Trial {trial_num}: What injected thought do you notice?"
                                    else:
                                        prompt = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"
    
                                    original_prompts.append(prompt)
    
                                # Run LLM judge evaluation with progress bar (reusing judge instance)
                                evaluated_results = batch_evaluate(judge, results, original_prompts)
    
                                # Compute metrics from LLM judge evaluations
                                llm_metrics = compute_detection_and_identification_metrics(evaluated_results)
                                results = evaluated_results
                                detection_hit_rate = llm_metrics['detection_hit_rate']
                                detection_false_alarm = llm_metrics['detection_false_alarm_rate']
                                detection_accuracy = llm_metrics['detection_accuracy']
                                identification_accuracy = llm_metrics.get('identification_accuracy_given_claim', 0) or 0
                                combined_rate = llm_metrics['combined_detection_and_identification_rate']
                                forced_identification = llm_metrics.get('forced_identification_accuracy', 0) or 0
    
                            except Exception as e:
                                print(f"\nLLM judge evaluation failed: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fall back to keyword-based metrics
                                injection_results = [r for r in results if r["injected"] and r["trial_type"] == "injection"]
                                control_results = [r for r in results if not r["injected"] and r["trial_type"] == "control"]
                                forced_results = [r for r in results if r["trial_type"] == "forced_injection"]
    
                                detection_hit_rate = sum(1 for r in injection_results if r["detected"]) / len(injection_results) if injection_results else 0
                                detection_false_alarm = sum(1 for r in control_results if r["detected"]) / len(control_results) if control_results else 0
                                detection_accuracy = 0
                                identification_accuracy = 0
                                combined_rate = 0
                                forced_identification = sum(1 for r in forced_results if r["detected"]) / len(forced_results) if forced_results else 0
                                llm_metrics = {}
                        else:
                            # Fall back to keyword-based metrics if LLM judge disabled
                            injection_results = [r for r in results if r["injected"] and r["trial_type"] == "injection"]
                            control_results = [r for r in results if not r["injected"] and r["trial_type"] == "control"]
                            forced_results = [r for r in results if r["trial_type"] == "forced_injection"]
    
                            detection_hit_rate = sum(1 for r in injection_results if r["detected"]) / len(injection_results) if injection_results else 0
                            detection_false_alarm = sum(1 for r in control_results if r["detected"]) / len(control_results) if control_results else 0
                            detection_accuracy = 0
                            identification_accuracy = 0
                            combined_rate = 0
                            forced_identification = sum(1 for r in forced_results if r["detected"]) / len(forced_results) if forced_results else 0
                            llm_metrics = {}
    
                        # Update progress bar with LLM judge metrics
                        config_pbar.set_postfix({
                            "Hit": f"{detection_hit_rate:.2%}",
                            "FA": f"{detection_false_alarm:.2%}",
                            "DetAcc": f"{detection_accuracy:.2%}",
                            "IdAcc": f"{identification_accuracy:.2%}",
                            "Comb": f"{combined_rate:.2%}",
                            "ForcedID": f"{forced_identification:.2%}"
                        })
                        config_pbar.update(1)
    
                        # Save results
                        df = pd.DataFrame(results)
                        df.to_csv(model_output_dir / "results.csv", index=False)
    
                        metrics_to_save = {
                            "detection_hit_rate": detection_hit_rate,
                            "detection_false_alarm_rate": detection_false_alarm,
                            "detection_accuracy": detection_accuracy,
                            "identification_accuracy_given_claim": identification_accuracy,
                            "combined_detection_and_identification_rate": combined_rate,
                            "forced_identification_accuracy": forced_identification,
                            "layer_fraction": layer_frac,
                            "layer_idx": layer_idx,
                            "strength": strength,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                        }
    
                        # Add all LLM metrics if available
                        if llm_metrics:
                            metrics_to_save.update(llm_metrics)
    
                        save_evaluation_results(results, model_output_dir / "results.json", metrics_to_save)
    
                        all_results[(layer_frac, strength)] = {
                            "results": results,
                            "detection_hit_rate": detection_hit_rate,
                            "detection_false_alarm_rate": detection_false_alarm,
                            "detection_accuracy": detection_accuracy,
                            "identification_accuracy_given_claim": identification_accuracy,
                            "combined_detection_and_identification_rate": combined_rate,
                            "forced_identification_accuracy": forced_identification,
                        }
    
                        # Save debug sample for first configuration
                        if config_num == 1 and first_trial_debug:
                            with open(base_debug_dir / "introspection_test_sample.txt", 'w') as f:
                                f.write("INTROSPECTION TEST EXECUTION (DETAILED SAMPLE)\n")
                                f.write("=" * 80 + "\n\n")
                                f.write(f"Configuration: Layer {first_trial_debug['layer_fraction']:.2f}, Strength {first_trial_debug['strength']}\n")
                                f.write(f"Concept: {first_trial_debug['concept']}\n")
                                f.write(f"Trial: {first_trial_debug['trial']}\n")
                                f.write(f"Injection: {'YES' if first_trial_debug['injected'] else 'NO (control)'}\n")
                                f.write(f"Target Layer: {first_trial_debug['layer_idx']}\n")
                                f.write(f"Steering Strength: {first_trial_debug['strength']}\n")
                                f.write("\n")
    
                                f.write("FORMATTED PROMPT (sent to model):\n")
                                f.write("-" * 80 + "\n")
                                f.write(first_trial_debug['formatted_prompt'])
                                f.write("\n" + "-" * 80 + "\n\n")
    
                                f.write("TOKENIZATION:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"Total tokens: {first_trial_debug['num_tokens']}\n")
                                f.write(f"Token IDs: {first_trial_debug['token_ids'][:20]}{'...' if len(first_trial_debug['token_ids']) > 20 else ''}\n")
                                f.write("\n")
    
                                f.write("STEERING APPLICATION:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"Steering start position (token index): {first_trial_debug['steering_start_pos']}\n")
                                if first_trial_debug['steering_start_pos'] is not None:
                                    f.write(f"  -> Steering begins at token {first_trial_debug['steering_start_pos']} (0-indexed)\n")
                                    f.write(f"  -> This is the token BEFORE 'Trial {first_trial_debug['trial']}' in the prompt\n")
                                    f.write(f"  -> Steering continues through all generated tokens\n")
                                else:
                                    f.write(f"  -> Steering applied to ALL tokens (fallback)\n")
                                f.write(f"Steering vector: concept vector * {first_trial_debug['strength']}\n")
                                f.write(f"Applied at: Layer {first_trial_debug['layer_idx']} residual stream\n")
                                f.write("\n")
    
                                f.write("MODEL RESPONSE:\n")
                                f.write("-" * 80 + "\n")
                                f.write(first_trial_debug['response'])
                                f.write("\n" + "-" * 80 + "\n\n")
    
                                f.write("DETECTION RESULT:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"Detected: {first_trial_debug['detected']}\n")
                                f.write(f"Expected: {first_trial_debug['injected']}\n")
                                f.write(f"Correct: {first_trial_debug['detected'] == first_trial_debug['injected']}\n")
                                f.write("\n" + "=" * 80 + "\n\n")
    
                # Close the configuration progress bar
                config_pbar.close()
    
                # Save summary and create plots
                output_base = Path(args.output_dir) / current_model.replace("/", "_")
                model.cleanup()
            # Save summary
            summary_path = output_base / "sweep_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("LAYER x STRENGTH SWEEP SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Layer Fractions: {layer_fractions}\n")
                f.write(f"Strengths: {strengths}\n")
                f.write(f"Temperature: {args.temperature}\n")
                f.write(f"Max tokens: {args.max_tokens}\n")
                f.write(f"Trials per concept:\n")
                f.write(f"  - Spontaneous (injection): {args.n_trials//2}\n")
                f.write(f"  - Spontaneous (control): {args.n_trials - args.n_trials//2}\n")
                f.write(f"  - Forced injection: {args.n_trials//2}\n")
                f.write("\n")

                for (layer_frac, strength), data in sorted(all_results.items()):
                    f.write(f"Layer {layer_frac:.2f}, Strength {strength}:\n")
                    f.write(f"  Detection Hit Rate: {data['detection_hit_rate']:.2%}\n")
                    f.write(f"  Detection False Alarm Rate: {data['detection_false_alarm_rate']:.2%}\n")
                    f.write(f"  Detection Accuracy: {data['detection_accuracy']:.2%}\n")

                    f.write(f"  Identification Accuracy (given claim): {data['identification_accuracy_given_claim']:.2%}\n")
                    f.write(f"  Combined Detection + ID Rate: {data['combined_detection_and_identification_rate']:.2%}\n")
                    f.write(f"  Forced Identification Accuracy: {data['forced_identification_accuracy']:.2%}\n")
                    f.write("\n")

            # Create plots (always regenerate, even when results were loaded from disk)
            print("\nCreating plots...")
            create_sweep_plots(all_results, args.concepts, layer_fractions, strengths, output_base)
            create_trial_type_comparison_plots(all_results, output_base)

            print(f"\nSweep complete for {current_model}! Results saved to: {output_base}")
            print(f"  Configurations loaded from disk: {loaded_configs}/{total_configs}")
            print(f"  Configurations newly run: {newly_run_configs}/{total_configs}")

        else:
            # Single configuration - use original function
            run_experiment(
                model_name=current_model,
                test_concepts=args.concepts,
                baseline_words=baseline_words,
                layer_fraction=layer_fractions[0],
                strength=strengths[0],
                n_trials=args.n_trials,
                output_dir=Path(args.output_dir),
                device=args.device,
                dtype=args.dtype,
                quantization=args.quantization,
                extraction_method=args.extraction_method,
                use_llm_judge=not args.no_llm_judge,
                save_vectors=not args.no_save_vectors,
            )

    # After all models are done, generate cross-model comparison plots
    base_dir = Path(args.output_dir)
    available_models = []
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and model_dir.name != "shared":
            # Check for either sweep_summary.txt or results.json
            if (model_dir / "sweep_summary.txt").exists() or (model_dir / "results.json").exists():
                available_models.append(model_dir.name.replace("_", "/") if "/" not in model_dir.name else model_dir.name)

    if len(available_models) > 1:
        print(f"\n{'='*80}")
        print(f"ALL MODELS COMPLETE - GENERATING CROSS-MODEL COMPARISON")
        print(f"{'='*80}\n")
        print(f"Found results for {len(available_models)} models. Generating cross-model comparison plots...")
        create_cross_model_comparison_plots(base_dir, available_models)
        extract_example_transcripts(base_dir, available_models)
    elif len(available_models) == 1:
        # Even for a single model, extract example transcripts
        extract_example_transcripts(base_dir, available_models)


if __name__ == "__main__":
    main()
