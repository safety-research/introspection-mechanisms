#!/usr/bin/env python3
"""
Runner script for the 500-concept sweep (50 baseline + 450 new).
Generates the command and executes 02_steering_evaluation.py with all concepts.
"""

import argparse
import subprocess
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the 450 additional concepts
from concepts_list import NEW_CONCEPTS

# Baseline 50 concepts from the original paper
BASELINE_CONCEPTS = [
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

# Combine all concepts
ALL_CONCEPTS = BASELINE_CONCEPTS + NEW_CONCEPTS


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment 02 (steering evaluation) with 500 concepts (50 baseline + 450 new)")
    parser.add_argument("-m", "--model", type=str, default="gemma3_27b", help="Model name (default: gemma3_27b)")
    parser.add_argument("-sl", "--specific-layers", type=int, nargs="+", default=[38], help="Specific layer indices to test (default: 38)")
    parser.add_argument("-s", "--strength", type=float, default=4.0, help="Steering strength (default: 4.0)")
    # New trial structure: trial_numbers (1-max) × samples_per_trial
    parser.add_argument("-mtn", "--max-trial-number", type=int, default=10, help="Maximum trial number (trials will be 1 to this value, default: 10)")
    parser.add_argument("-spt", "--samples-per-trial", type=int, default=10, help="Samples per trial number for injection trials (default: 10)")
    parser.add_argument("-cspt", "--control-samples-per-trial", type=int, default=50, help="Control samples per trial number - global, not per concept (default: 50)")
    # DEPRECATED: kept for backward compatibility
    parser.add_argument("-nt", "--n-trials", type=int, default=None, help="DEPRECATED: Use --samples-per-trial instead")
    parser.add_argument("-od", "--output-dir", type=str, default="analysis/02b_steering_500_concepts", help="Output directory (default: analysis/02b_steering_500_concepts)")
    parser.add_argument("-ij", "--incremental-judge", action="store_true", default=True, help="Run LLM judge after each concept (default: True)")
    parser.add_argument("-no-ij", "--no-incremental-judge", action="store_true", help="Disable incremental judge")
    parser.add_argument("-ow", "--overwrite", action="store_true", help="Overwrite existing results (default: False, resume from partial)")
    parser.add_argument("-nlj", "--no-llm-judge", action="store_true", help="Disable LLM judge entirely")
    parser.add_argument("-rf", "--run-forced", action="store_true", help="Run forced injection trials (disabled by default)")
    parser.add_argument("-bs", "--batch-size", type=int, default=None, help="Batch size for generation (default: 300)")
    parser.add_argument("-uvf", "--use-vectors-from", type=str, default=None, help="Path to existing vectors folder to copy instead of extracting (e.g., analysis/02b_steering_500_concepts/gemma3_27b/vectors). This avoids 'double ablation' when running abliterated models.")
    parser.add_argument("-env", "--extract-native-vectors", action="store_true", help="When using -uvf, also extract vectors from the loaded model and save to 'abliterated_vectors/'")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Baseline concepts: {len(BASELINE_CONCEPTS)}")
    print(f"New concepts: {len(NEW_CONCEPTS)}")
    print(f"Total concepts: {len(ALL_CONCEPTS)}")
    print(f"Unique concepts: {len(set(ALL_CONCEPTS))}")

    # Handle deprecated -nt flag
    samples_per_trial = args.samples_per_trial
    if args.n_trials is not None:
        print(f"WARNING: --n-trials is deprecated. Using it as --samples-per-trial={args.n_trials}")
        samples_per_trial = args.n_trials

    # Build the command
    cmd = [
        sys.executable,  # Python interpreter
        "02_steering_evaluation.py",
        "-m", args.model,
        "--specific-layers", *[str(l) for l in args.specific_layers],
        "-s", str(args.strength),
        "-c",  # Concepts flag, followed by all concepts
    ] + ALL_CONCEPTS + [
        "-od", args.output_dir,
        "-mtn", str(args.max_trial_number),
        "-spt", str(samples_per_trial),
        "-cspt", str(args.control_samples_per_trial),
    ]

    # Add optional flags
    if args.incremental_judge and not args.no_incremental_judge:
        cmd.append("--incremental-judge")

    if args.overwrite:
        cmd.append("--overwrite")

    if args.no_llm_judge:
        cmd.append("--no-llm-judge")

    if args.run_forced:
        cmd.append("--run-forced")

    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])

    if args.use_vectors_from is not None:
        cmd.extend(["--use-vectors-from", args.use_vectors_from])

    if args.extract_native_vectors:
        cmd.append("--extract-native-vectors")

    layers_str = ", ".join(str(l) for l in args.specific_layers)
    total_injection_per_concept = args.max_trial_number * samples_per_trial
    total_control = args.max_trial_number * args.control_samples_per_trial
    print(f"\nRunning experiment 02 (steering evaluation) with {len(ALL_CONCEPTS)} total concepts...")
    print(f"Model: {args.model}")
    print(f"Target layers: {layers_str}, strength: {args.strength}")
    print(f"Trial structure: {args.max_trial_number} trial numbers × {samples_per_trial} samples = {total_injection_per_concept} per concept")
    print(f"Control trials: {args.max_trial_number} trial numbers × {args.control_samples_per_trial} samples = {total_control} total (global)")
    print(f"Output: {args.output_dir}/{args.model.replace('/', '_')}/")
    print(f"Incremental judge: {args.incremental_judge and not args.no_incremental_judge}")
    print(f"Resume: {not args.overwrite}")
    if args.use_vectors_from:
        print(f"Using pre-existing vectors from: {args.use_vectors_from}")
    print()

    # Execute the command
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
