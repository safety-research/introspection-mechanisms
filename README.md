# Mechanisms of Introspective Awareness

Code for the paper *"Mechanisms of Introspective Awareness"*.

**Paper**: https://arxiv.org/abs/2603.21396
**LessWrong**: 

## Overview

We investigate the mechanisms underlying introspective awareness in large language models — the ability to detect and identify concepts injected into their residual stream via steering vectors. Our main findings:

1. **Introspection is behaviorally robust.** Detection maintains 0% false positives across diverse prompts while achieving moderate true positive rates. The capability emerges from post-training.

2. **Anomaly detection is not reducible to a single linear direction.** Both the mean-difference direction and the residual carry detection-relevant signal, and bidirectional steering produces detection in both directions for success concept pairs.

3. **Distinct detection and identification mechanisms.** MLPs at ~70% depth are causally necessary and sufficient for detection. Circuit analysis identifies evidence carrier and gate features that combine signals into a detection decision.

4. **Models possess latent introspective capacity.** Ablating refusal directions improves detection by ~53pp; a trained steering vector improves detection by ~75pp on held-out concepts without increasing false positives.

## Repository Structure

```
├── src/                        # Utility modules
│   ├── model_utils.py          # Model loading, steering vector computation, inference
│   ├── eval_utils.py           # LLM judge evaluation, metrics computation
│   ├── steering_utils.py       # Steering vector extraction and application
│   ├── patching_utils.py       # Activation patching utilities
│   ├── probe_utils.py          # Linear probe training and evaluation
│   ├── training_utils.py       # Training loop utilities
│   ├── vector_utils.py         # Vector geometry analysis
│   └── plot_style.py           # Plotting style
├── experiments/                # Experiment scripts (numbered by paper section)
│   ├── 01_concept_injection.py           # Core concept injection experiment
│   ├── 02_steering_evaluation.py         # Steering vector evaluation (canonical)
│   ├── 02b_run_500_concepts.py           # Runner for 500-concept sweep
│   ├── 03_behavioral_robustness.py       # OLMo training stage comparison (§3.3)
│   ├── 03b_persona_variants.py           # Persona/dialogue format variants (§3.2)
│   ├── 03c_prompt_variants.py            # Prompt framing variants (§3.1)
│   ├── 03d_refusal_abliteration.py       # Refusal direction abliteration (§3.3)
│   ├── 03e_optimize_abliteration.py      # Optuna optimization for abliteration weights (Appendix F)
│   ├── 03f_dpo_mechanism_ablation.py     # DPO component ablation: contrastive structure analysis (§3.3)
│   ├── 04_geometry_analysis.py           # Concept vector geometry analysis (§4)
│   ├── 04b_vector_geometry.py            # LDA partition, ridge regression, PCA (§2, §4.3)
│   ├── 04c_bidirectional_steering.py     # Bidirectional steering, δPCs, threshold sweep (§4.2-4.3)
│   ├── 04d_ridge_swap.py                 # Mean-diff and ridge swap experiments (§4.1, Appendix C)
│   ├── 04e_mean_diff_steering.py         # Mean-diff direction causal validation (Appendix D)
│   ├── 04f_mean_diff_exploration.py      # Mean-diff direction interpretation (Appendix D)
│   ├── 04g_pretraining_alignment.py      # Pretraining corpus projection analysis (Appendix D)
│   ├── 04h_aggregation_analysis.py       # Verbalizability + ridge R² on transcoder features (§4.3)
│   ├── 05_negative_steering.py           # Negative direction steering
│   ├── 05b_steering_prompts.py           # Prompt definitions for steering experiments
│   ├── 06_activation_patching.py         # Per-layer activation patching (§5.2)
│   ├── 07_transcoder_feature_analysis.py # Gate/evidence carrier identification (§5.3)
│   ├── 08_feature_centric_analysis.py    # Feature subset analysis
│   ├── 09_circuit_analysis.py            # Circuit tracing and ablation (§5.3-5.4)
│   ├── 10_head_identification.py         # Attention head identification
│   ├── 11_head_ablation.py              # Attention head ablation
│   ├── 12_head_investigation.py          # Attention head investigation
│   ├── 13_steering_attribution.py        # Steering attribution framework (§5, Appendix)
│   ├── 14_trained_steering_vector.py     # Trained steering vector (§6)
│   └── 15_proxy_task_sweep.py            # Proxy task sweep
├── plotting/                   # Figure generation scripts
│   ├── plot_figures.py         # Standalone figure regeneration (metrics, attribution graphs)
│   ├── plot_circuit_figures.py # Gate/evidence carrier figures
│   ├── plot_geometry_figures.py# Section 4 geometry panel
│   ├── plot_head_probe.py      # Head probe comparison figure
│   ├── plot_style.py           # Plot styling
│   └── data/                   # Cached data for figure regeneration
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- GPU with ≥48GB VRAM (for Gemma3-27B)
- CUDA 12.x

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with API keys for LLM judge evaluation:

```bash
OPENAI_API_KEY=your_key_here     # For GPT-4o judge
```

## Models

The primary model used is **Gemma3-27B-IT** (`google/gemma-3-27b-it`). Additional models:
- Gemma3-27B base (`google/gemma-3-27b-pt`)
- OLMo-3.1-32B (base, SFT, DPO, instruct checkpoints)
- Qwen3-235B (for prompt variant comparison)

Transcoders are from [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-27b).

## Reproducing Results

### Step 1: Core Experiment (§2)

Run the concept injection experiment across 500 concepts:

```bash
python experiments/02_steering_evaluation.py \
    --models gemma3_27b \
    --specific-layers 37 \
    --strength 4.0 \
    --n-trials 10
```

### Step 2: Behavioral Robustness (§3)

Test prompt variants (§3.1), persona variants (§3.2), and training stage comparisons (§3.3):

```bash
# Prompt framing variants (Original, Alternative, Skeptical, etc.)
python experiments/03c_prompt_variants.py -m gemma3_27b

# Persona/dialogue format variants (chat template, raw, user-detects, etc.)
python experiments/03b_persona_variants.py -m gemma3_27b

# OLMo staged introspection (base → SFT → DPO → instruct)
python experiments/03_behavioral_robustness.py \
    -m olmo_32b_base olmo_32b_sft olmo_32b_dpo olmo_32b \
    -ls 25 32 38 45 51 -ss 1 2 4 8
```

### Step 3: Geometry Analysis (§4)

Analyze concept vector geometry, bidirectional steering, and swap experiments:

```bash
python experiments/04_geometry_analysis.py \
    -m gemma3_27b --plots-only
```

### Step 4: Component Localization (§5)

Run activation patching to identify causal components:

```bash
python experiments/06_activation_patching.py \
    -m gemma3_27b -lf 0.6 -s 4.0
```

### Step 5: Transcoder Feature Analysis (§5.3)

Identify gate and evidence carrier features:

```bash
python experiments/07_transcoder_feature_analysis.py
python experiments/09_circuit_analysis.py
```

### Step 6: Steering Attribution (Appendix)

Run the steering attribution framework:

```bash
python experiments/13_steering_attribution.py \
    -m gemma3_27b --sections A B D
```

### Step 7: Trained Steering Vector (§6)

Train and evaluate the introspection steering vector:

```bash
python experiments/14_trained_steering_vector.py all
```

### Generating Figures

After running experiments, generate paper figures:

```bash
# Metrics vs injection layer, attribution graphs
cd plotting && python plot_figures.py

# Geometry panel (Section 4)
python plot_geometry_figures.py

# Circuit figures (gates, evidence carriers)
python plot_circuit_figures.py

# Head probe comparison
python plot_head_probe.py
```

## Citation

```bibtex
@article{macar2026mechanisms,
    title={Mechanisms of Introspective Awareness},
    author={Macar, Uzay and Yang, Li and Wang, Atticus and Wallich, Peter and Ameisen, Emmanuel and Lindsey, Jack},
    journal={arXiv preprint},
    year={2026}
}
```

## License

This project is released for research purposes. See the paper for details on responsible use considerations.
