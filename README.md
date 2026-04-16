# Mechanisms of Introspective Awareness

Code for the paper *"Mechanisms of Introspective Awareness"*.

**Paper**: https://arxiv.org/abs/2603.21396

**Blog**: https://www.lesswrong.com/posts/BNMLtuDTNBwGHcnQX/mechanisms-of-introspective-awareness

## Overview

We investigate the mechanisms underlying introspective awareness in large language models — the ability to detect and identify concepts injected into their residual stream via steering vectors. Our main findings:

1. **Introspection is behaviorally robust.** Models detect injected steering vectors at modest nonzero rates, with 0% false positives, across diverse prompts and dialogue formats. The capability is absent in base models, and emerges from post-training; specifically, we find that it arises from contrastive preference optimization algorithms like direct preference optimization (DPO), but not supervised finetuning (SFT). The capability is strongest when the model acts in its trained Assistant persona.
2. **Anomaly detection is not reducible to a single linear direction.** Although one direction in activation space explains a substantial fraction of detection variance, the underlying computation is distributed across multiple directions. This suggests that the capability is not explained by some concept vectors being correlated with a direction that promotes affirmative responses to questions in general.
3. **Distinct detection and identification mechanisms.** Detection and identification are handled by distinct mechanisms in different layers, with MLPs at ~70% depth causally necessary and sufficient for detection. Circuit analysis identifies "gate" features which inhibit detection claims, and which are suppressed by upstream "evidence carrier" features sensitive to injected steering vectors. Different steering vectors activate different evidence carriers, but the circuit converges on a common set of gates.
4. **Models possess underelicited introspective capacity.** Ablating refusal directions ("abliteration") improves detection from 10.8% to 63.8% with modest false positive increases (0% to 7.3%). A trained bias vector improves detection by +75% and introspection by +55% on held-out concepts without increasing false positives. Both results suggest that introspective capability is underelicited by default.

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
│   ├── 04i_alternative_geometric_tests.py # Cone rotation test, norm/unembedding hypotheses (Appendix H)
│   ├── 05_negative_steering.py           # Negative direction steering
│   ├── concepts_list.py                  # 450 additional concepts for 500-concept sweep
│   ├── steering_prompts.py               # Prompt definitions for steering experiments
│   ├── refusal_prompts.py                # Harmful/harmless prompt datasets for abliteration
│   ├── 06_activation_patching.py         # Per-layer activation patching (§5.2)
│   ├── 07_transcoder_feature_analysis.py # Gate/evidence carrier identification (§5.3)
│   ├── 08_feature_centric_analysis.py    # Feature subset analysis
│   ├── 09_circuit_analysis.py            # Circuit tracing and ablation (§5.3-5.4)
│   ├── 09b_causal_pathway.py             # Causal pathway analysis, circuit importance (§5.4)
│   ├── 10_head_identification.py         # Attention head identification
│   ├── 11_head_ablation.py              # Attention head ablation
│   ├── 12_head_investigation.py          # Attention head investigation
│   ├── 13_component_attribution.py       # Component-level attribution analysis (§5.2, Appendix J/N)
│   ├── 13b_attention_pattern.py          # Attention probs vs injection strength (Appendix U)
│   ├── 13c_gradient_attribution_sweep.py # Gradient attribution over 400 concepts (Appendix T)
│   ├── 14_trained_bias_vector.py         # Trained steering vector + bias training (§6)
│   ├── 14b_downstream_bias_eval.py       # HaluEval/JailbreakHub/CoT/prefill eval of bias vector (Appendix S)
│   ├── 14c_bias_semantic_analysis.py     # SAE decomp + logit lens + behavioral effects (Appendix R)
│   ├── 15_proxy_task_sweep.py            # Proxy task sweep
│   ├── 16_steering_attribution.py       # SAE steering attribution: SA = GA × SG (Appendix Q)
│   └── 17_attribution_graph.py          # Attribution graph construction & visualization (Appendix Q)
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

### Optional: SAE/transcoder feature labels

Several scripts that interpret SAE or transcoder features
(`07_transcoder_feature_analysis.py`, `08_feature_centric_analysis.py`,
`09_circuit_analysis.py`, `09b_causal_pathway.py`, `16_steering_attribution.py`,
`plotting/plot_circuit_figures.py`) can attach short natural-language labels
to each `(layer, feature_id)` pair. These labels are **not bundled** with this
repo and there is no public release you can clone — they were generated
locally by the authors (paper Appendix K) by sampling the top-20
max-activating contexts per feature from pretraining text and prompting an
LLM to summarize them.

**If no labels directory is present, every script still runs** — features are
displayed as `L{layer} F{feature_id}` instead of semantic names.

If you want labels, you must generate them yourself and drop the JSON files
into the path the scripts expect.

#### Expected layout

Scripts look for one JSON per Gemma-Scope-2 transcoder/SAE at:

```
<path>/gemma_scope_2_27b_{sae_type}_all_layer{N}_{width}_{sparsity}_labels.json
```

where `sae_type ∈ {transcoder, attn_out, mlp_out, resid_post}`,
`width ∈ {16k, 262k}`, and `sparsity ∈ {small, big}`. The path `<path>` is
resolved **relative to the current working directory** when you invoke the
script. Running commands from the repo root (as in the examples below) is
the simplest convention; if you drop a `feature_labels/` directory as a
sibling of the repo at `../gemma-scope-2/feature_labels/`, create a
matching symlink inside the repo so every script finds it:

```bash
# From the repo root (introspection-mechanisms/)
mkdir -p ../gemma-scope-2/feature_labels  # drop your JSONs here
ln -s ../gemma-scope-2 gemma-scope-2      # makes the sibling visible from the repo too
```

#### Expected file format

Each JSON maps `feature_id` (as a string) to a small record. The scripts read
the `title` field (a short human-readable label):

```json
{
  "3411": {
    "feature_id": 3411,
    "title": "Cleaning and maintenance action words",
    "description": "Features that activate on verbs/nouns relating to ...",
    "max_activation": 2040.285
  },
  "9959": {
    "feature_id": 9959,
    "title": "Tokens immediately preceding 'no' or negation",
    "description": "...",
    "max_activation": 312.7
  }
}
```

Only `title` is required; other fields are ignored by the loaders. If you have
labels in a different schema, write a one-off script that emits JSONs in the
format above and place them at the expected path.

#### Generating your own labels (sketch)

1. Stream a pretraining-scale text corpus through Gemma3-27B with the
   Gemma-Scope-2 transcoder attached at the relevant layer.
2. For each feature, collect the top-20 activating contexts (token ± 32
   context window).
3. Prompt a capable LLM (the paper used Claude Opus 4.5) to summarize each
   feature's concept in <10 words, and write the result into the `title`
   field of the JSON above.

#### Overriding the path

To read labels from elsewhere without symlinks, edit the single `Path(...)`
constant near the top of each script:

- `experiments/07_transcoder_feature_analysis.py:260` —
  `gemma_scope_path = Path("gemma-scope-2/feature_labels")`
- `experiments/08_feature_centric_analysis.py:77` —
  `FEATURE_LABELS_PATH = Path("gemma-scope-2/feature_labels")`
- `experiments/09_circuit_analysis.py:310` —
  `labels_dir = Path("../gemma-scope-2/feature_labels")`

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

Identify gate and evidence carrier features. Scripts 07/08/09b consume
transcoder feature activations cached at
`analysis/08_cached_activations/`; run the prep step first:

```bash
# Prep: collect and cache steered + control transcoder activations.
python experiments/09_circuit_analysis.py --cache-only \
    --model gemma3_27b --steering-layer 37 --steering-strength 4.0

# Gate / evidence-carrier selection.
# Default ranking is correlation-based (detector_score); use
#   --ranking-metric logit_attribution
# to rank by paper §5.3 DLA = (w_dec · Δu_Yes-No) * mean_activation instead.
python experiments/07_transcoder_feature_analysis.py

# Full circuit analysis (also regenerates the cache).
python experiments/09_circuit_analysis.py
```

### Step 6: Component Attribution (Appendix)

Run the component-level attribution analysis:

```bash
python experiments/13_component_attribution.py \
    -m gemma3_27b --sections A B D
```

### Step 7: Trained Steering Vector (§6)

Train and evaluate the introspection steering vector:

```bash
# Prefill detection (LoRA finetuning)
python experiments/14_trained_bias_vector.py all

# Bias vector training (Section 6: bias adapter on concept injection data)
python experiments/14_trained_bias_vector.py generate-bias-data --model gemma3_27b
python experiments/14_trained_bias_vector.py train-bias --model gemma3_27b
python experiments/14_trained_bias_vector.py evaluate-bias --model gemma3_27b

# Downstream behavioral effects of the bias vector (Appendix S):
# HaluEval, JailbreakHub, CoT faithfulness (MMLU + GPQA), prefill detection.
python experiments/14b_downstream_bias_eval.py all --model gemma3_27b

# Semantic/behavioral analysis of the bias vector (Appendix R):
# SAE decomposition, logit lens across layers, response-length effects.
python experiments/14c_bias_semantic_analysis.py all --model gemma3_27b
```

### Step 7b: Attention pattern vs injection strength (Appendix U)

```bash
python experiments/13b_attention_pattern.py \
    --model gemma3_27b --steering-layer 37 --strengths 0 1 2 4 8
```

### Step 7c: Gradient attribution sweep over 400 concepts (Appendix T)

```bash
python experiments/13c_gradient_attribution_sweep.py \
    -m gemma3_27b --n-concepts 400
```

### Step 8: SAE Steering Attribution & Attribution Graph (Appendix Q)

Extract steering attribution (SA = GA × SG) and build attribution graphs:

```bash
# Auto-configure: detect pos/neg tokens and optimal strength
python experiments/16_steering_attribution.py auto-config --concept Bread --layer 37

# Extract SA at multiple strengths + compute ISA
python experiments/16_steering_attribution.py extract-all --concept Bread --layer 37

# Or extract at a single strength:
python experiments/16_steering_attribution.py extract-sa --concept Bread --layer 37 --strength 4.0
python experiments/16_steering_attribution.py compute-isa --concept Bread --layer 37

# Build attribution graph (backward + forward tracing)
python experiments/17_attribution_graph.py build-graph --concept Bread --layer 37 --direction both

# Full pipeline: extract SA, compute ISA, build graph, visualize
python experiments/17_attribution_graph.py all \
    --concept Bread --layer 37 --strength 4.0 \
    --direction both --trace-depth 2

# Re-render existing graph
python experiments/17_attribution_graph.py visualize --concept Bread --layer 37
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
