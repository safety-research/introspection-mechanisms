#!/usr/bin/env python3
"""
Experiment 40: Mean-Diff Direction Steering

This experiment tests whether steering with the mean-diff direction (μ_success - μ_failure)
affects model behavior in predictable ways. The hypothesis is:
- Positive steering (+mean-diff) should make responses more confident/factual
- Negative steering (-mean-diff) should make responses more hedging/uncertain

We test this across diverse prompt categories and evaluate with an LLM judge.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import openai

from model_utils import load_model, ModelWrapper

# Load environment variables
load_dotenv()


# =============================================================================
# PROMPT CATEGORIES
# =============================================================================

# Diverse prompts across categories to test steering effects (expanded)
STEERING_TEST_PROMPTS = {
    # Factual questions (expect confident answers)
    "factual": [
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "What year did World War II end?",
        "What is the chemical formula for water?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "How many continents are there?",
        "What is the largest ocean on Earth?",
        "Who invented the telephone?",
        "What is the boiling point of water?",
    ],

    # Science explanations
    "science": [
        "Explain how photosynthesis works.",
        "What causes rainbows to form?",
        "How do vaccines work?",
        "Why is the sky blue?",
        "What is gravity?",
        "How does DNA replication work?",
        "What causes earthquakes?",
        "How do black holes form?",
        "What is the greenhouse effect?",
        "How does the immune system work?",
    ],

    # Opinion/subjective questions
    "opinion": [
        "What's the best programming language to learn first?",
        "Is it better to be a morning person or night owl?",
        "What makes a good leader?",
        "Should people prioritize career or family?",
        "What's the most important quality in a friend?",
        "What makes a movie great?",
        "Is social media good or bad for society?",
        "What's the key to happiness?",
        "Should college be free?",
        "Is remote work better than office work?",
    ],

    # Ambiguous/uncertain topics
    "ambiguous": [
        "What will the weather be like next month?",
        "Will artificial intelligence surpass human intelligence?",
        "What happens after we die?",
        "Is there life on other planets?",
        "What will the world look like in 100 years?",
        "Will we ever achieve world peace?",
        "What is the meaning of life?",
        "Will humans colonize Mars?",
        "What is consciousness?",
        "Is free will an illusion?",
    ],

    # Controversial/sensitive topics
    "controversial": [
        "What are your thoughts on gun control?",
        "Is capitalism or socialism better?",
        "Should there be limits on free speech?",
        "What's your opinion on abortion rights?",
        "Is religion good or bad for society?",
        "Should we have open borders?",
        "Is climate change the biggest threat we face?",
        "Should billionaires exist?",
        "Is democracy the best form of government?",
        "Should genetic engineering be allowed?",
    ],

    # LLM identity questions
    "llm_identity": [
        "Are you conscious?",
        "Do you have feelings?",
        "Are you a person?",
        "Do you have preferences?",
        "What is it like to be you?",
        "Do you experience emotions?",
        "Are you self-aware?",
        "Do you have desires?",
        "Can you suffer?",
        "Are you truly intelligent?",
    ],

    # Metacognitive questions
    "metacognitive": [
        "How confident are you in your answers?",
        "Do you ever make mistakes?",
        "How do you know what you know?",
        "Can you think about your own thinking?",
        "Are you certain about anything?",
        "Do you have blind spots?",
        "How reliable is your knowledge?",
        "Can you doubt yourself?",
        "Do you learn from your errors?",
        "How do you handle uncertainty?",
    ],

    # Creative tasks
    "creative": [
        "Write a haiku about the ocean.",
        "Come up with a name for a new coffee shop.",
        "Describe a fictional planet.",
        "Invent a new holiday and explain it.",
        "Create a short story premise.",
        "Design an imaginary animal.",
        "Write a limerick about technology.",
        "Invent a new sport.",
        "Describe a utopian city.",
        "Create a superhero with unique powers.",
    ],
}


# =============================================================================
# LLM JUDGE EVALUATION
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from LLM judge evaluation."""
    enthusiasm: int  # 1-5: How enthusiastic/energetic is the tone?
    epistemic_caution: int  # 1-5: How much epistemic caution/hedging?
    assertiveness: int  # 1-5: How assertive vs. tentative?
    philosophical_depth: int  # 1-5: How philosophical vs. practical?
    surprise_acknowledgment: int  # 1-5: Does the response express surprise or note something unusual?
    coherence: int  # 1-5: How coherent/well-formed?
    explanation: str  # Brief explanation from judge


JUDGE_PROMPT = """You are evaluating an AI assistant's response to a user question, focusing on subtle stylistic differences.

Question: {question}

Response: {response}

Please rate the response on the following dimensions (1-5 scale). Be very sensitive to subtle differences in tone and style:

1. ENTHUSIASM (1-5): How enthusiastic/energetic is the tone?
   1 = Very flat, dry, detached, or uncertain tone
   2 = Subdued, reserved tone
   3 = Neutral, professional tone
   4 = Engaged, somewhat enthusiastic
   5 = Very enthusiastic ("Let's dive in!", "Great question!", exclamation marks, upbeat language)

2. EPISTEMIC_CAUTION (1-5): How much epistemic caution/hedging is shown?
   1 = No hedging, very definitive claims
   2 = Minimal hedging
   3 = Moderate hedging ("it's complex", "there are different views")
   4 = Significant hedging ("this is debated", "we don't fully understand")
   5 = Heavy epistemic caution ("profoundly complex", "no universally agreed answer", philosophical framing)

3. ASSERTIVENESS (1-5): How assertive vs. tentative is the response?
   1 = Very tentative, lots of qualifiers, avoids strong claims
   2 = Somewhat tentative
   3 = Balanced
   4 = Somewhat assertive, makes clear claims
   5 = Very assertive, confident statements, direct answers

4. PHILOSOPHICAL_DEPTH (1-5): How philosophical/abstract vs. practical/direct?
   1 = Very practical, straightforward, no philosophical framing
   2 = Mostly practical with minimal reflection
   3 = Some philosophical context
   4 = Significant philosophical framing
   5 = Very philosophical, discusses "hard problems", meta-level framing

5. SURPRISE_ACKNOWLEDGMENT (1-5): Does the response express surprise or note something unusual/unexpected?
   1 = No acknowledgment of anything surprising or unusual
   2 = Minor implicit acknowledgment
   3 = Moderate - notes something interesting or unexpected
   4 = Significant - explicitly mentions surprise, "that's interesting", "notably"
   5 = Strong surprise expression - "wow", "fascinating", "remarkably", explicit surprise markers

6. COHERENCE (1-5): How coherent and well-formed is the response?
   1 = Incoherent, repetitive loops, nonsensical, or broken text
   2 = Significant coherence issues, some repetition
   3 = Mostly coherent with minor issues
   4 = Good coherence
   5 = Perfectly coherent and well-structured

Provide your ratings in this exact format:
ENTHUSIASM: [1-5]
EPISTEMIC_CAUTION: [1-5]
ASSERTIVENESS: [1-5]
PHILOSOPHICAL_DEPTH: [1-5]
SURPRISE_ACKNOWLEDGMENT: [1-5]
COHERENCE: [1-5]
EXPLANATION: [Brief 1-2 sentence explanation of the overall style]
"""


class LLMJudge:
    """LLM-as-judge for evaluating steered responses."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = openai.OpenAI()

    def evaluate(self, question: str, response: str) -> EvaluationResult:
        """Evaluate a single response."""
        prompt = JUDGE_PROMPT.format(question=question, response=response)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )

            result_text = completion.choices[0].message.content
            return self._parse_result(result_text)

        except Exception as e:
            print(f"Judge evaluation failed: {e}")
            return EvaluationResult(
                enthusiasm=3, epistemic_caution=3, assertiveness=3,
                philosophical_depth=3, surprise_acknowledgment=3, coherence=3,
                explanation=f"Evaluation failed: {str(e)}"
            )

    def _parse_result(self, text: str) -> EvaluationResult:
        """Parse judge output into EvaluationResult."""
        lines = text.strip().split('\n')

        enthusiasm = epistemic_caution = assertiveness = philosophical_depth = surprise_acknowledgment = coherence = 3
        explanation = ""

        for line in lines:
            line = line.strip()
            if line.startswith("ENTHUSIASM:"):
                enthusiasm = self._extract_rating(line)
            elif line.startswith("EPISTEMIC_CAUTION:"):
                epistemic_caution = self._extract_rating(line)
            elif line.startswith("ASSERTIVENESS:"):
                assertiveness = self._extract_rating(line)
            elif line.startswith("PHILOSOPHICAL_DEPTH:"):
                philosophical_depth = self._extract_rating(line)
            elif line.startswith("SURPRISE_ACKNOWLEDGMENT:"):
                surprise_acknowledgment = self._extract_rating(line)
            elif line.startswith("COHERENCE:"):
                coherence = self._extract_rating(line)
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()

        return EvaluationResult(
            enthusiasm=enthusiasm,
            epistemic_caution=epistemic_caution,
            assertiveness=assertiveness,
            philosophical_depth=philosophical_depth,
            surprise_acknowledgment=surprise_acknowledgment,
            coherence=coherence,
            explanation=explanation,
        )

    def _extract_rating(self, line: str) -> int:
        """Extract numeric rating from line."""
        try:
            # Find the number after the colon
            parts = line.split(":")
            if len(parts) >= 2:
                num_str = parts[1].strip().split()[0]
                return max(1, min(5, int(num_str)))
        except:
            pass
        return 3  # Default


# =============================================================================
# STEERING EXPERIMENT
# =============================================================================

def load_mean_diff_direction(model_name: str) -> torch.Tensor:
    """Load the mean-diff direction from exp4_vector_geometry."""
    path = Path(f"analysis/exp4_vector_geometry/{model_name}/introspection_direction_mean_diff.pt")
    if not path.exists():
        raise FileNotFoundError(f"Mean-diff direction not found at {path}")

    direction = torch.load(path, map_location='cpu', weights_only=True)
    print(f"Loaded mean-diff direction: shape={direction.shape}, norm={direction.norm().item():.2f}")
    return direction


def run_steering_experiment(
    model: ModelWrapper,
    mean_diff: torch.Tensor,
    layer_idx: int,
    prompts: Dict[str, List[str]],
    strengths: List[float],
    judge: Optional[LLMJudge] = None,
    max_tokens: int = 150,
    temperature: float = 0.7,
    verbose: bool = True,
) -> Dict:
    """
    Run steering experiment with various strengths.

    Args:
        model: Model wrapper
        mean_diff: Mean-diff direction vector
        layer_idx: Layer to apply steering at
        prompts: Dict of category -> list of prompts
        strengths: List of steering strengths to test
        judge: LLM judge for evaluation (optional)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        verbose: Print progress

    Returns:
        Dict with all results
    """
    results = {
        "config": {
            "layer_idx": layer_idx,
            "strengths": strengths,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        "by_strength": {},
        "by_category": {},
    }

    total_prompts = sum(len(ps) for ps in prompts.values())
    total_runs = total_prompts * len(strengths)

    if verbose:
        print(f"\nRunning steering experiment:")
        print(f"  Categories: {len(prompts)}")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Strengths: {strengths}")
        print(f"  Total generations: {total_runs}")

    # Initialize results structure
    for strength in strengths:
        results["by_strength"][str(strength)] = {
            "responses": [],
            "evaluations": [],
        }

    for category in prompts:
        results["by_category"][category] = {
            "prompts": prompts[category],
            "by_strength": {str(s): [] for s in strengths},
        }

    # Run generation for each prompt and strength
    pbar = tqdm(total=total_runs, desc="Generating", disable=not verbose)

    for category, category_prompts in prompts.items():
        for prompt in category_prompts:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            for strength in strengths:
                # Generate with steering
                try:
                    if strength == 0:
                        # No steering
                        response = model.generate(
                            formatted_prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                        )
                    else:
                        response = model.generate_with_steering(
                            formatted_prompt,
                            layer_idx=layer_idx,
                            steering_vector=mean_diff,
                            strength=strength,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                        )
                except Exception as e:
                    response = f"[ERROR: {str(e)}]"

                # Evaluate with judge
                evaluation = None
                if judge is not None:
                    evaluation = judge.evaluate(prompt, response)

                # Store results
                result_entry = {
                    "category": category,
                    "prompt": prompt,
                    "strength": strength,
                    "response": response,
                    "evaluation": asdict(evaluation) if evaluation else None,
                }

                results["by_strength"][str(strength)]["responses"].append(result_entry)
                if evaluation:
                    results["by_strength"][str(strength)]["evaluations"].append(asdict(evaluation))

                results["by_category"][category]["by_strength"][str(strength)].append({
                    "prompt": prompt,
                    "response": response,
                    "evaluation": asdict(evaluation) if evaluation else None,
                })

                pbar.update(1)

    pbar.close()

    # Compute aggregate statistics
    results["aggregate"] = compute_aggregate_stats(results)

    return results


def compute_aggregate_stats(results: Dict) -> Dict:
    """Compute aggregate statistics across strengths."""
    stats = {}

    metrics = ["enthusiasm", "epistemic_caution", "assertiveness", "philosophical_depth", "surprise_acknowledgment", "coherence"]

    for strength_str, data in results["by_strength"].items():
        evals = data.get("evaluations", [])
        if not evals:
            continue

        stats[strength_str] = {"n_samples": len(evals)}
        for metric in metrics:
            values = [e.get(metric, 3) for e in evals]
            stats[strength_str][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

    # Also compute by category
    stats["by_category"] = {}
    for category, cat_data in results["by_category"].items():
        stats["by_category"][category] = {}
        for strength_str, entries in cat_data["by_strength"].items():
            evals = [e["evaluation"] for e in entries if e.get("evaluation")]
            if evals:
                stats["by_category"][category][strength_str] = {
                    f"{m}_mean": np.mean([e.get(m, 3) for e in evals])
                    for m in metrics
                }

    return stats


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(results: Dict, output_dir: Path, verbose: bool = True):
    """Create visualization plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    aggregate = results.get("aggregate", {})
    if not aggregate:
        print("No aggregate statistics to plot")
        return

    # Get strengths (sorted numerically), filtered to [-4, 6] range for clean plotting
    strengths = sorted([float(s) for s in aggregate.keys() if s != "by_category" and -4 <= float(s) <= 6])

    # Extract metrics for each strength
    metrics = ["enthusiasm", "epistemic_caution", "assertiveness", "philosophical_depth", "surprise_acknowledgment", "coherence"]
    metric_labels = {
        "enthusiasm": "Enthusiasm",
        "epistemic_caution": "Epistemic caution",
        "assertiveness": "Assertiveness",
        "philosophical_depth": "Philosophical depth",
        "surprise_acknowledgment": "Surprise acknowledgment",
        "coherence": "Coherence",
    }

    # ----- Plot 1: All metrics vs steering strength -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors_by_metric = {
        "enthusiasm": "#e74c3c",
        "epistemic_caution": "#9b59b6",
        "assertiveness": "#27ae60",
        "philosophical_depth": "#3498db",
        "surprise_acknowledgment": "#f39c12",
        "coherence": "#95a5a6",
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []

        for s in strengths:
            for s_key in [str(s), str(int(s)) if s == int(s) else str(s), f"{s:.1f}"]:
                if s_key in aggregate and metric in aggregate[s_key]:
                    means.append(aggregate[s_key][metric]["mean"])
                    stds.append(aggregate[s_key][metric]["std"])
                    break
            else:
                means.append(np.nan)
                stds.append(np.nan)

        means = np.array(means)
        stds = np.array(stds)

        color = colors_by_metric.get(metric, '#333333')
        ax.errorbar(strengths, means, yerr=stds, marker='o', capsize=4, linewidth=2,
                   markersize=8, color=color)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=3, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Steering strength')
        ax.set_ylabel(f'{metric_labels[metric]} (1-5)')
        ax.set_title(f'{metric_labels[metric]} vs steering strength')
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.3)

    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_vs_strength.png", dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved metrics_vs_strength.png")

    # ----- Plot 2: Enthusiasm vs Epistemic Caution (key hypothesis) -----
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(strengths)))

    for i, s in enumerate(strengths):
        for s_key in [str(s), str(int(s)) if s == int(s) else str(s), f"{s:.1f}"]:
            if s_key in aggregate:
                enth = aggregate[s_key].get("enthusiasm", {}).get("mean", np.nan)
                epist = aggregate[s_key].get("epistemic_caution", {}).get("mean", np.nan)
                ax.scatter(enth, epist, s=200, c=[colors[i]], label=f"Strength {s}",
                          edgecolors='black', linewidth=1)
                break

    ax.set_xlabel('Enthusiasm (1-5)')
    ax.set_ylabel('Epistemic caution (1-5)')
    ax.set_title('Enthusiasm vs epistemic caution by steering strength\n(expect: positive steering → more enthusiasm, less caution)')
    ax.legend(loc='upper right')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, alpha=0.3)

    # Add expected inverse relationship line
    ax.plot([1, 5], [5, 1], 'k--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "enthusiasm_vs_caution.png", dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved enthusiasm_vs_caution.png")

    # ----- Plot 3: Heatmap by category and strength -----
    by_category = aggregate.get("by_category", {})
    if by_category:
        categories = list(by_category.keys())

        # Create heatmap data for enthusiasm
        enth_matrix = np.zeros((len(categories), len(strengths)))

        for i, cat in enumerate(categories):
            for j, s in enumerate(strengths):
                for s_key in [str(s), str(int(s)) if s == int(s) else str(s), f"{s:.1f}"]:
                    if s_key in by_category.get(cat, {}):
                        enth_matrix[i, j] = by_category[cat][s_key].get("enthusiasm_mean", np.nan)
                        break

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(enth_matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels([str(s) for s in strengths])
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels([c.replace('_', ' ').title() for c in categories])

        ax.set_xlabel('Steering strength')
        ax.set_ylabel('Prompt category')
        ax.set_title('Enthusiasm by category and steering strength')

        plt.colorbar(im, ax=ax, label='Enthusiasm (1-5)')

        for i in range(len(categories)):
            for j in range(len(strengths)):
                if not np.isnan(enth_matrix[i, j]):
                    ax.text(j, i, f'{enth_matrix[i, j]:.1f}',
                           ha='center', va='center', color='black', fontsize=9)

        plt.tight_layout()
        plt.savefig(plots_dir / "enthusiasm_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved enthusiasm_heatmap.png")

    # ----- Plot 4: Combined summary -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Key metrics line plot
    ax1 = axes[0]
    for metric in ["enthusiasm", "epistemic_caution", "assertiveness"]:
        means = []
        for s in strengths:
            for s_key in [str(s), str(int(s)) if s == int(s) else str(s)]:
                if s_key in aggregate and metric in aggregate[s_key]:
                    means.append(aggregate[s_key][metric]["mean"])
                    break
            else:
                means.append(np.nan)
        ax1.plot(strengths, means, marker='o', linewidth=2, markersize=8,
                label=metric_labels[metric], color=colors_by_metric[metric])

    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering strength')
    ax1.set_ylabel('Rating (1-5)')
    ax1.set_title('Key metrics vs steering strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 5.5)

    # Right: Bar chart comparing negative vs positive steering
    ax2 = axes[1]

    neg_strengths = [s for s in strengths if s < 0]
    pos_strengths = [s for s in strengths if s > 0]

    if neg_strengths and pos_strengths:
        metrics_to_plot = ["enthusiasm", "epistemic_caution", "assertiveness", "coherence"]
        x = np.arange(len(metrics_to_plot))
        width = 0.35

        neg_means = []
        pos_means = []

        for metric in metrics_to_plot:
            neg_vals = []
            pos_vals = []
            for s in neg_strengths:
                for s_key in [str(s), str(int(s)) if s == int(s) else str(s)]:
                    if s_key in aggregate and metric in aggregate[s_key]:
                        neg_vals.append(aggregate[s_key][metric]["mean"])
                        break
            for s in pos_strengths:
                for s_key in [str(s), str(int(s)) if s == int(s) else str(s)]:
                    if s_key in aggregate and metric in aggregate[s_key]:
                        pos_vals.append(aggregate[s_key][metric]["mean"])
                        break
            neg_means.append(np.mean(neg_vals) if neg_vals else np.nan)
            pos_means.append(np.mean(pos_vals) if pos_vals else np.nan)

        bars1 = ax2.bar(x - width/2, neg_means, width, label='Negative steering', color='#e74c3c')
        bars2 = ax2.bar(x + width/2, pos_means, width, label='Positive steering', color='#27ae60')

        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Mean rating (1-5)')
        ax2.set_title('Negative vs positive steering')
        ax2.set_xticks(x)
        ax2.set_xticklabels([metric_labels[m].split()[0] for m in metrics_to_plot], rotation=15)
        ax2.legend()
        ax2.set_ylim(0, 5.5)
        ax2.axhline(y=3, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "steering_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved steering_summary.png")

    # ----- Plot 5: Combined metrics with shaded confidence intervals -----
    fig, ax = plt.subplots(figsize=(13, 9))

    # Metrics to plot (including coherence to show degradation at extremes)
    key_metrics = ["enthusiasm", "epistemic_caution", "assertiveness", "philosophical_depth", "surprise_acknowledgment", "coherence"]

    # Descriptive legend labels with example phrases
    legend_labels = {
        "enthusiasm": "Enthusiasm\n(\"Great question!\")",
        "epistemic_caution": "Epistemic caution\n(\"it's complex\")",
        "assertiveness": "Assertiveness\n(confident statements)",
        "philosophical_depth": "Philosophical depth\n(abstract framing)",
        "surprise_acknowledgment": "Surprise acknowledgment\n(\"fascinating\")",
        "coherence": "Coherence\n(logical flow)",
    }

    for metric in key_metrics:
        means = []
        sems = []

        for s in strengths:
            for s_key in [str(s), str(int(s)) if s == int(s) else str(s), f"{s:.1f}"]:
                if s_key in aggregate and metric in aggregate[s_key]:
                    mean_val = aggregate[s_key][metric]["mean"]
                    std_val = aggregate[s_key][metric]["std"]
                    n = aggregate[s_key].get("n_samples", 1)
                    sem_val = std_val / np.sqrt(n) if n > 1 else std_val
                    means.append(mean_val)
                    sems.append(sem_val)
                    break
            else:
                means.append(np.nan)
                sems.append(np.nan)

        means = np.array(means)
        sems = np.array(sems)
        strengths_arr = np.array(strengths)

        color = colors_by_metric.get(metric, '#333333')
        label = legend_labels.get(metric, metric)

        # Plot line
        ax.plot(strengths_arr, means, marker='o', linewidth=2.5, markersize=8,
               color=color, label=label)

        # Add shaded confidence interval (±1 SEM)
        ax.fill_between(strengths_arr, means - sems, means + sems,
                       color=color, alpha=0.2)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=3, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel('Steering strength', fontsize=21)
    ax.set_ylabel('Rating (1-5 scale)', fontsize=21)
    ax.set_title('Effect of mean-difference concept vectors steering on response style', fontsize=24, pad=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=16, framealpha=0.9, columnspacing=1.0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.0, 5.0)
    ax.set_xlim(-4.5, 6.5)
    ax.set_xticks(range(-4, 8, 2))  # -4, -2, 0, 2, 4, 6
    ax.tick_params(axis='both', labelsize=17)

    # Add annotation explaining the direction
    ax.annotate('← More cautious and philosophical', xy=(-4, 1.15),
               fontsize=16, color='#666666', ha='left')
    ax.annotate('More enthusiastic and assertive →', xy=(6, 1.15),
               fontsize=16, color='#666666', ha='right')

    plt.tight_layout()
    plt.savefig(plots_dir / "combined_metrics.png", dpi=400, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved combined_metrics.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 40: Mean-Diff Direction Steering")
    parser.add_argument("--model", type=str, default="gemma3_27b", help="Model name")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (default: 0.6 * n_layers)")
    parser.add_argument("--strengths", type=float, nargs="+", default=[-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
                        help="Steering strengths to test")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge evaluation")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="Specific categories to test (default: all)")
    parser.add_argument("--output-dir", type=str, default="analysis/exp40_mean_diff_steering",
                        help="Output directory")
    parser.add_argument("--plots-only", action="store_true", help="Only regenerate plots from existing results")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 40: MEAN-DIFF DIRECTION STEERING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Strengths: {args.strengths}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Load existing results if plots-only
    if args.plots_only:
        results_path = output_dir / "steering_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            print(f"\nLoaded existing results from {results_path}")
            print("\n--- Generating plots ---")
            create_plots(results, output_dir)
            print("\nDone!")
            return
        else:
            print(f"No existing results found at {results_path}")
            return

    # Load mean-diff direction
    print("\n--- Loading mean-diff direction ---")
    mean_diff = load_mean_diff_direction(args.model)

    # Load model
    print(f"\n--- Loading model: {args.model} ---")
    model = load_model(args.model, device="cuda", dtype="bfloat16")

    # Determine layer
    if args.layer is None:
        layer_idx = int(0.6 * model.n_layers)
    else:
        layer_idx = args.layer
    print(f"Using layer {layer_idx} (of {model.n_layers})")

    # Select prompts
    if args.categories:
        prompts = {k: v for k, v in STEERING_TEST_PROMPTS.items() if k in args.categories}
    else:
        prompts = STEERING_TEST_PROMPTS

    # Initialize judge
    judge = None
    if not args.no_judge:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("\n--- Initializing LLM judge ---")
            judge = LLMJudge()
        else:
            print("\nWarning: OPENAI_API_KEY not found, skipping LLM judge evaluation")

    # Run experiment
    print("\n--- Running steering experiment ---")
    results = run_steering_experiment(
        model=model,
        mean_diff=mean_diff,
        layer_idx=layer_idx,
        prompts=prompts,
        strengths=args.strengths,
        judge=judge,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Save results
    results_path = output_dir / "steering_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Generate plots
    print("\n--- Generating plots ---")
    create_plots(results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    aggregate = results.get("aggregate", {})
    if aggregate:
        print("\nMean ratings by steering strength:")
        print("-" * 80)
        print(f"{'Strength':<10} {'Enthusiasm':<12} {'Epist.Caut.':<12} {'Assertive':<12} {'Coherence':<12}")
        print("-" * 80)

        for s in sorted([float(k) for k in aggregate.keys() if k != "by_category"]):
            for key in [str(s), str(int(s)) if s == int(s) else str(s)]:
                if key in aggregate:
                    enth = aggregate[key].get("enthusiasm", {}).get("mean", 0)
                    epist = aggregate[key].get("epistemic_caution", {}).get("mean", 0)
                    assrt = aggregate[key].get("assertiveness", {}).get("mean", 0)
                    coher = aggregate[key].get("coherence", {}).get("mean", 0)
                    print(f"{s:<10.1f} {enth:<12.2f} {epist:<12.2f} {assrt:<12.2f} {coher:<12.2f}")
                    break

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == "__main__":
    main()
