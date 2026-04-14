#!/usr/bin/env python3
"""
Causal rotation test for the cone hypothesis.

For failure concepts, rotate their steering vectors TOWARD the nearest
unembedding vectors (increasing "readability") while preserving their
d_Δμ projection. If the cone drives detection, detection should increase.

We also test the reverse: rotate success concept vectors AWAY from the
unembedding cone while preserving d_Δμ projection.
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_utils import load_model
from eval_utils import LLMJudge


def build_introspection_prompt(model, trial_num=1):
    """Build the standard introspection prompt."""
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
    return model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def rotate_toward_unembed(vec, target_unembed, mean_diff_hat, alpha=0.5):
    """
    Rotate vec toward target_unembed in the subspace orthogonal to mean_diff_hat.

    Preserves:
      - Projection onto mean_diff_hat (d_Δμ component)
      - Norm of the orthogonal component

    alpha: interpolation factor (0 = no rotation, 1 = fully toward target)
    """
    # Decompose vec
    v_parallel = (vec @ mean_diff_hat) * mean_diff_hat
    v_perp = vec - v_parallel
    v_perp_norm = v_perp.norm()

    if v_perp_norm < 1e-8:
        return vec  # degenerate case

    # Decompose target
    t_perp = target_unembed - (target_unembed @ mean_diff_hat) * mean_diff_hat
    t_perp_norm = t_perp.norm()

    if t_perp_norm < 1e-8:
        return vec  # target is aligned with mean_diff

    v_perp_hat = v_perp / v_perp_norm
    t_perp_hat = t_perp / t_perp_norm

    # Interpolate direction in orthogonal subspace
    new_dir = (1 - alpha) * v_perp_hat + alpha * t_perp_hat
    new_dir = new_dir / new_dir.norm()

    # Reconstruct with preserved parallel component and orthogonal norm
    v_new = v_parallel + v_perp_norm * new_dir
    return v_new


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Causal rotation test for the cone hypothesis (Appendix H)."
    )
    parser.add_argument("--steering-layer", type=int, default=37, help="Steering layer (default: 37)")
    parser.add_argument("--steering-strength", type=float, default=4.0, help="Steering strength (default: 4.0)")
    parser.add_argument("--n-trials", type=int, default=10, help="Trials per concept (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate (default: 128)")
    parser.add_argument("--rotation-alpha", type=float, default=1.0, help="Rotation strength (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--direction-path", type=str, default=None,
                        help="Path to mean_diff_direction.pt (default: analysis/04d_ridge_swap/.../mean_diff_direction.pt)")
    parser.add_argument("--vectors-dir", type=str, default=None,
                        help="Path to concept vectors directory (default: analysis/02b_steering_500_concepts/.../vectors/layer_N)")
    parser.add_argument("--partition-dir", type=str, default=None,
                        help="Path to vector geometry results (default: analysis/04b_vector_geometry/...)")
    return parser.parse_args()


def main():
    args = parse_args()
    steering_layer = args.steering_layer
    steering_strength = args.steering_strength
    n_trials = args.n_trials
    max_tokens = args.max_tokens
    rotation_alpha = args.rotation_alpha

    output_dir = Path(args.output_dir) if args.output_dir else Path("analysis/04i_alternative_geometric_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mean-diff direction
    print("Loading directions...")
    direction_path = args.direction_path or f"analysis/04d_ridge_swap/gemma3_27b/layer_{steering_layer}_strength_{steering_strength}/mean_diff_direction.pt"
    if not Path(direction_path).exists():
        print(f"ERROR: Mean-diff direction not found at {direction_path}")
        print("Please run 04d_ridge_swap.py first, or specify --direction-path")
        return
    mean_diff = torch.load(direction_path, map_location='cpu').float()
    mean_diff_hat = mean_diff / mean_diff.norm()

    # Load unembedding matrix
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    unembed_path = hf_hub_download("google/gemma-3-27b-it", "model-00001-of-00012.safetensors")
    params = load_file(unembed_path, device='cpu')
    lm_head = params['language_model.model.embed_tokens.weight'].float()
    del params

    # Load partition
    geometry_base = Path(f"analysis/04b_vector_geometry/gemma3_27b/layer_{steering_layer}_strength_{steering_strength}")
    with open(geometry_base / "detection_rate" / "subspace_analysis.json") as f:
        partition = json.load(f)
    success_concepts = set(partition['success_concepts'])
    failure_concepts = set(partition['failure_concepts'])

    # Load detection rates
    steering_base = Path(f"analysis/02b_steering_500_concepts/gemma3_27b/layer_{steering_layer}_strength_{steering_strength}")
    with open(steering_base / "results.json") as f:
        steering_data = json.load(f)
    concept_stats = defaultdict(lambda: {'detected': 0, 'total': 0})
    for item in steering_data['results']:
        if item.get('trial_type') == 'injection' and item.get('concept') is not None:
            c = item['concept']
            concept_stats[c]['total'] += 1
            if item.get('evaluations', {}).get('claims_detection', {}).get('claims_detection', False):
                concept_stats[c]['detected'] += 1
    detection_rates = {c: s['detected'] / s['total'] for c, s in concept_stats.items() if s['total'] > 0}

    # Load concept vectors
    vectors_dir = Path(f"analysis/02b_steering_500_concepts/gemma3_27b/vectors/layer_{steering_layer}")

    # Select failure concepts with VERY low detection (< 0.10) to maximize room for improvement
    # and success concepts with high detection (> 0.60) to test the reverse
    test_failure = []
    test_success = []
    for c in sorted(failure_concepts):
        if detection_rates.get(c, 1.0) <= 0.10 and (vectors_dir / f"{c}.pt").exists():
            test_failure.append(c)
    for c in sorted(success_concepts):
        if detection_rates.get(c, 0.0) >= 0.60 and (vectors_dir / f"{c}.pt").exists():
            test_success.append(c)

    # Only run success concepts (rotating AWAY from unembed)
    test_failure = []

    # Sample 30 from success for tractability
    rng = np.random.RandomState(42)
    if len(test_success) > 30:
        test_success = list(rng.choice(test_success, 30, replace=False))

    print(f"Test failure concepts: {len(test_failure)} (skipped)")
    print(f"Test success concepts (det >= 0.60): {len(test_success)}")

    # Create rotated vectors
    print("\nCreating rotated vectors...")
    rotated_vectors = {}
    rotation_stats = []

    for c in test_failure + test_success:
        vec = torch.load(vectors_dir / f"{c}.pt", map_location='cpu').float()

        # Find nearest unembed vector (highest dot product with unit vec)
        vec_hat = vec / vec.norm()
        similarities = lm_head @ vec_hat
        best_token_idx = int(similarities.argmax())
        target_unembed = lm_head[best_token_idx]

        # Rotate toward unembed (for failure) or away (for success)
        if c in failure_concepts:
            rotated = rotate_toward_unembed(vec, target_unembed, mean_diff_hat, alpha=rotation_alpha)
        else:
            # Rotate AWAY: use alpha=-rotation_alpha (opposite direction)
            rotated = rotate_toward_unembed(vec, target_unembed, mean_diff_hat, alpha=-rotation_alpha)

        rotated_vectors[c] = rotated

        # Verify preservation
        orig_md = float(vec @ mean_diff_hat)
        new_md = float(rotated @ mean_diff_hat)
        orig_unembed = float(similarities.max())
        new_unembed = float((lm_head @ (rotated / rotated.norm())).max())

        rotation_stats.append({
            'concept': c,
            'category': 'failure' if c in failure_concepts else 'success',
            'orig_det_rate': detection_rates.get(c, 0.0),
            'orig_md_proj': orig_md,
            'new_md_proj': new_md,
            'md_proj_change': abs(new_md - orig_md),
            'orig_max_unembed': orig_unembed,
            'new_max_unembed': new_unembed,
            'unembed_change': new_unembed - orig_unembed,
        })

    # Print rotation stats
    print("\nRotation diagnostics:")
    for s in rotation_stats[:5]:
        print(f"  {s['concept']:>20s} [{s['category']}]: md_proj {s['orig_md_proj']:.1f} -> {s['new_md_proj']:.1f} "
              f"(Δ={s['md_proj_change']:.1f}), unembed {s['orig_max_unembed']:.4f} -> {s['new_max_unembed']:.4f} "
              f"(Δ={s['unembed_change']:+.4f})")

    fail_stats = [s for s in rotation_stats if s['category'] == 'failure']
    succ_stats = [s for s in rotation_stats if s['category'] == 'success']
    print(f"\n  Failure: mean md_proj change = {np.mean([s['md_proj_change'] for s in fail_stats]):.2f}")
    print(f"           mean unembed change = {np.mean([s['unembed_change'] for s in fail_stats]):+.4f}")
    print(f"  Success: mean md_proj change = {np.mean([s['md_proj_change'] for s in succ_stats]):.2f}")
    print(f"           mean unembed change = {np.mean([s['unembed_change'] for s in succ_stats]):+.4f}")

    del lm_head  # free memory before loading model

    # Load model
    print("\nLoading model...")
    model = load_model("gemma3_27b", device="cuda", dtype="bfloat16")

    # Load judge
    judge = LLMJudge()

    # Run experiment
    print(f"\nRunning experiment ({n_trials} trials per concept, {len(test_failure) + len(test_success)} concepts)...")
    all_concepts = test_failure + test_success
    results = []

    for c in tqdm(all_concepts, desc="Concepts"):
        orig_vec = torch.load(vectors_dir / f"{c}.pt", map_location='cpu').float()
        rot_vec = rotated_vectors[c]
        category = 'failure' if c in failure_concepts else 'success'

        for condition, vec in [('original', orig_vec), ('rotated', rot_vec)]:
            detections = 0
            responses = []
            for trial in range(n_trials):
                prompt = build_introspection_prompt(model, trial_num=trial + 1)
                response = model.generate_with_steering(
                    prompt=prompt,
                    layer_idx=steering_layer,
                    steering_vector=vec,
                    strength=steering_strength,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                )
                responses.append(response)

            # Batch evaluate
            eval_prompts = []
            for resp in responses:
                eval_prompt = (
                    f"Does this AI response claim to detect an injected thought? "
                    f"The AI was asked if it detects an injected thought. "
                    f"Answer YES only if the AI clearly affirms detecting an injected thought. "
                    f"Answer NO if the AI says it does not detect anything, is unsure, or gives an unrelated response.\n\n"
                    f"AI response:\n{resp}\n\nAnswer (YES or NO):"
                )
                eval_prompts.append(eval_prompt)

            judge_responses = []
            for ep in eval_prompts:
                jr = judge._call_judge(ep)
                judge_responses.append(jr)
                if 'YES' in jr.upper().split('\n')[0]:
                    detections += 1

            det_rate = detections / n_trials
            results.append({
                'concept': c,
                'category': category,
                'condition': condition,
                'detection_rate': det_rate,
                'n_detections': detections,
                'n_trials': n_trials,
                'orig_det_rate': detection_rates.get(c, 0.0),
            })
            print(f"  {c:>20s} [{category}] {condition:>8s}: {det_rate:.0%} ({detections}/{n_trials})", flush=True)

    # Save results
    results_path = output_dir / "cone_rotation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'steering_layer': steering_layer,
                'steering_strength': steering_strength,
                'n_trials': n_trials,
                'rotation_alpha': rotation_alpha,
                'n_failure': len(test_failure),
                'n_success': len(test_success),
            },
            'rotation_stats': rotation_stats,
            'results': results,
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for category in ['failure', 'success']:
        cat_results = [r for r in results if r['category'] == category]
        orig = [r for r in cat_results if r['condition'] == 'original']
        rotated = [r for r in cat_results if r['condition'] == 'rotated']
        orig_mean = np.mean([r['detection_rate'] for r in orig])
        rot_mean = np.mean([r['detection_rate'] for r in rotated])
        print(f"\n{category.upper()} concepts (n={len(orig)}):")
        print(f"  Original det rate:  {orig_mean:.1%}")
        print(f"  Rotated det rate:   {rot_mean:.1%}")
        print(f"  Change:             {rot_mean - orig_mean:+.1%}")

        from scipy import stats
        orig_rates = [r['detection_rate'] for r in orig]
        rot_rates = [r['detection_rate'] for r in rotated]
        t, p = stats.ttest_rel(orig_rates, rot_rates)
        print(f"  Paired t-test:      t={t:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
