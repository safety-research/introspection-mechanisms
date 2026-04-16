#!/usr/bin/env python3
"""
Transcoder Feature Analysis: Aggregation Analysis — Geometry Panel (Section 4.3)

Two analyses for the paper's geometry panel:

1. **Verbalizability** (Figure geometry-panel (b)):
   For each concept, compute verbalizability = max_t(v_c @ W_U[t])
   where t ranges over single-token casing/spacing variants of the concept name.
   Scatter plot verbalizability vs projection onto mean-diff direction.

2. **Ridge regression on transcoder features** (Figure geometry-panel (d)):
   Load cached transcoder feature activations across layers 38-61,
   build a feature matrix (per-concept mean activations), and fit ridge
   regression to predict per-concept detection rate. Report cross-validated
   R^2 as a function of number of top-N features. Baselines: mean-diff
   projection R^2, concept vector R^2.

Usage:
    python experiments/04h_aggregation_analysis.py --logit-attr-ridge
    python experiments/04h_aggregation_analysis.py --logit-attr-ridge --logit-attr-ridge-plots-only
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

CACHED_ACTIVATIONS_BASE = Path("analysis/08_cached_activations")
OUTPUT_BASE = Path("analysis/04h_aggregation_analysis")

DEFAULT_STEERING_LAYER = 37
DEFAULT_STEERING_STRENGTH = 4.0
DEFAULT_TOKEN_MODE = "last_token"
DEFAULT_TRANSCODER_L0 = "big"
DEFAULT_TRANSCODER_WIDTH = "262k"


# =============================================================================
# Path helpers
# =============================================================================

def get_geometry_base(steering_layer: int, steering_strength: float) -> Path:
    return Path(f"analysis/04b_vector_geometry/gemma3_27b/layer_{steering_layer}_strength_{steering_strength}")


def get_steering_base(steering_layer: int, steering_strength: float) -> Path:
    return Path(f"analysis/02b_steering_500_concepts/gemma3_27b/layer_{steering_layer}_strength_{steering_strength}")


def get_direction_base(steering_layer: int, steering_strength: float) -> Path:
    """Directory holding introspection_direction_mean_diff.pt for a given config.

    This is the ``detection_rate`` subdirectory written by 04b_vector_geometry.py
    under each ``layer_<L>_strength_<s>/`` config.
    """
    return Path(
        f"analysis/04b_vector_geometry/gemma3_27b/"
        f"layer_{steering_layer}_strength_{steering_strength}/detection_rate"
    )


def resolve_mean_diff_path(base: Path) -> Path:
    """Return the mean-diff direction file under ``base`` if present."""
    return base / "introspection_direction_mean_diff.pt"


def get_concept_vectors_dir(steering_layer: int) -> Path:
    """Get the directory containing concept steering vectors from experiment 02 (steering evaluation)."""
    return Path(f"analysis/02b_steering_500_concepts/gemma3_27b/vectors/layer_{steering_layer}")


def get_width_prefix(transcoder_width: str) -> str:
    """Get width prefix for cache directory names.

    For 16k: returns '' (backward compatible with older cache directories).
    For 262k: returns '262k_'.
    """
    return f"{transcoder_width}_" if transcoder_width != "16k" else ""


def get_analysis_layers(steering_layer: int, start_offset: int = 1, end_layer: int = 61) -> List[int]:
    """Get the analysis layers for a given steering layer.

    By default, analyzes layers from steering_layer+1 to end_layer (inclusive).
    """
    return list(range(steering_layer + start_offset, end_layer + 1))


# =============================================================================
# Data loading
# =============================================================================

def load_concept_vectors(
    concepts: List[str],
    steering_layer: int = 37,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load concept steering vectors from experiment 02 (steering evaluation).

    Returns:
        X_vectors: Matrix of shape (n_concepts, vector_dim) with steering vectors as rows
        valid_concepts: List of concepts that had valid vectors
    """
    vectors_dir = get_concept_vectors_dir(steering_layer)

    vectors = []
    valid_concepts = []

    for concept in concepts:
        vec_path = vectors_dir / f"{concept}.pt"
        if vec_path.exists():
            vec = torch.load(vec_path)
            vectors.append(vec.float().numpy())
            valid_concepts.append(concept)

    if not vectors:
        return None, []

    X_vectors = np.stack(vectors, axis=0)
    return X_vectors, valid_concepts


def load_concept_mean_diff_projections(
    steering_layer: int = DEFAULT_STEERING_LAYER,
    steering_strength: float = DEFAULT_STEERING_STRENGTH,
    concepts_list: List[str] = None,
) -> Dict[str, float]:
    """
    Load concept vectors and compute their projections onto the mean-diff direction.

    Uses:
    - Mean-diff direction from experiment 04d/04e (direction analysis) (residual stream space)
    - Concept vectors from experiment 02 (steering evaluation) (residual stream space)

    Returns dict mapping concept name -> projection value (scalar).
    """
    direction_base = get_direction_base(steering_layer, steering_strength)
    mean_diff_path = resolve_mean_diff_path(direction_base)

    if not mean_diff_path.exists():
        print(f"  WARNING: Mean-diff direction not found under {direction_base}")
        return {}

    mean_diff = torch.load(mean_diff_path, weights_only=True).float()
    mean_diff = mean_diff / mean_diff.norm()

    vectors_dir = Path(f"analysis/02b_steering_500_concepts/gemma3_27b/vectors/layer_{steering_layer}")

    if not vectors_dir.exists():
        print(f"  WARNING: Concept vectors directory not found: {vectors_dir}")
        return {}

    projections = {}

    if concepts_list is not None:
        concepts_to_load = concepts_list
    else:
        concepts_to_load = [f.stem for f in vectors_dir.glob("*.pt")]

    for concept in concepts_to_load:
        vec_path = vectors_dir / f"{concept}.pt"
        if vec_path.exists():
            vec = torch.load(vec_path).float()
            proj = (vec @ mean_diff).item()
            projections[concept] = proj

    print(f"  Loaded {len(projections)} concept projections onto mean-diff direction")
    return projections


# Global cache for lm_head and tokenizer (lazy loaded)
_LM_HEAD_CACHE = {}


def get_lm_head_and_tokenizer(model_name: str = "google/gemma-3-27b-it"):
    """
    Load just the lm_head weight and tokenizer without loading the full model.

    For Gemma 3 models, the lm_head uses tied embeddings (shares weights with
    embed_tokens), so we load the embedding weight instead.
    """
    if model_name in _LM_HEAD_CACHE:
        return _LM_HEAD_CACHE[model_name]

    try:
        from transformers import AutoTokenizer, AutoConfig
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"    Loading tokenizer and lm_head for logit lens (first time only)...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        weight_names_to_try = [
            "lm_head.weight",
            "model.lm_head.weight",
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ]

        files = list_repo_files(model_name)
        safetensor_files = [f for f in files if f.endswith('.safetensors') and 'model-' in f]

        lm_head_weight = None

        search_order = ['model-00001-of-00012.safetensors'] + sorted(safetensor_files, reverse=True)
        seen = set()
        search_order = [x for x in search_order if x in safetensor_files and not (x in seen or seen.add(x))]

        for st_file in search_order:
            try:
                path = hf_hub_download(model_name, st_file)
                with safe_open(path, framework="pt", device="cpu") as f:
                    file_keys = f.keys()
                    for weight_name in weight_names_to_try:
                        if weight_name in file_keys:
                            lm_head_weight = f.get_tensor(weight_name)
                            print(f"    Found {weight_name} in {st_file}")
                            break
                if lm_head_weight is not None:
                    break
            except Exception:
                continue

        if lm_head_weight is None:
            print("    WARNING: Could not find lm_head/embed_tokens weight in safetensors")
            return None, None

        _LM_HEAD_CACHE[model_name] = (lm_head_weight, tokenizer)
        return lm_head_weight, tokenizer

    except Exception as e:
        print(f"    WARNING: Could not load lm_head and tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# Verbalizability baseline
# =============================================================================

def _compute_verbalizability_baseline(
    steering_layer: int,
    concepts_list: List[str],
    y_continuous: np.ndarray,
    valid_mask: np.ndarray,
    y_original_probs: np.ndarray = None,
) -> Tuple[Optional[float], Optional[float], Dict[str, float], List[str]]:
    """
    Compute verbalizability R^2 baseline.

    For each concept, load its steering vector and compute:
        verbalizability_logit = max over token variants of (vec @ unembed[token_id])

    Then fit Ridge CV: verbalizability_logit -> detection_rate.
    Only works for single-token concepts.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score as _r2_score
    from sklearn.pipeline import Pipeline

    try:
        from transformers import AutoTokenizer
        from safetensors import safe_open
        import os
    except ImportError as e:
        print(f"  WARNING: Missing dependency for verbalizability: {e}")
        return None, None, {}, []

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    # Load unembed matrix from HuggingFace cache
    print("  Loading unembed matrix...")
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = Path(hf_cache) / "hub" / "models--google--gemma-3-27b-it" / "snapshots"

    unembed = None
    if model_dir.exists():
        snapshots = list(model_dir.iterdir())
        if snapshots:
            snapshot_dir = snapshots[0]
            shard_path = snapshot_dir / "model-00001-of-00012.safetensors"
            if shard_path.exists():
                print(f"  Loading from cache: {shard_path}")
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "embed_tokens.weight" in key:
                            unembed = f.get_tensor(key).float()
                            break
                print(f"  Unembed shape: {unembed.shape}")

    if unembed is None:
        print("  WARNING: Could not load unembed matrix. Skipping verbalizability.")
        return None, None, {}, []

    # Load concept vectors
    vectors_dir = get_concept_vectors_dir(steering_layer)

    # For each concept, find single-token variants and compute verbalizability logit
    verb_per_concept = {}
    verb_valid_concepts = []

    for concept in concepts_list:
        vec_path = vectors_dir / f"{concept}.pt"
        if not vec_path.exists():
            continue

        vec = torch.load(vec_path).float()

        # Try multiple token variants; use first subword token for multi-token concepts
        variants = [concept, concept.lower(), f" {concept}", f" {concept.lower()}", f" {concept.capitalize()}"]
        best_logit = None

        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                logit = (vec @ unembed[token_ids[0]]).item()
                if best_logit is None or logit > best_logit:
                    best_logit = logit

        if best_logit is not None:
            verb_per_concept[concept] = best_logit
            verb_valid_concepts.append(concept)

    print(f"  Concepts with verbalizability logit: {len(verb_valid_concepts)}/{len(concepts_list)}")

    if len(verb_valid_concepts) < 50:
        print("  WARNING: Too few single-token concepts for reliable regression")
        return None, None, verb_per_concept, verb_valid_concepts

    # Build regression: verbalizability_logit -> detection_rate
    verb_x = []
    verb_y = []
    verb_y_orig = []
    for i, c in enumerate(concepts_list):
        if c in verb_per_concept and valid_mask[i]:
            verb_x.append(verb_per_concept[c])
            verb_y.append(y_continuous[i])
            if y_original_probs is not None:
                verb_y_orig.append(y_original_probs[i])

    verb_x = np.array(verb_x).reshape(-1, 1)
    verb_y = np.array(verb_y)
    verb_y_orig = np.array(verb_y_orig) if y_original_probs is not None else None
    print(f"  Regression samples: {len(verb_y)}")

    kf = KFold(n_splits=30, shuffle=True, random_state=42)
    best_r2, best_sem = -999, None

    # For verbalizability (single feature), try fitting directly on raw detection rates
    # even when logit_transform is used, since verbalizability has a direct linear
    # relationship with detection rate in probability space.
    targets_to_try = [verb_y]
    if verb_y_orig is not None:
        targets_to_try.append(verb_y_orig)

    for verb_target in targets_to_try:
        eval_target = verb_y_orig if verb_y_orig is not None else verb_target
        for alpha in [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]:
            pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
            if verb_y_orig is not None:
                y_pred = cross_val_predict(pipe, verb_x, verb_target, cv=kf, n_jobs=-1)
                if verb_target is verb_y:
                    y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))
                else:
                    y_pred_prob = np.clip(y_pred, 0, 1)
                r2 = float(_r2_score(eval_target, y_pred_prob))
                fold_r2s = []
                for _, test_idx in kf.split(verb_x):
                    if len(test_idx) >= 3:
                        fold_r2s.append(float(_r2_score(eval_target[test_idx], y_pred_prob[test_idx])))
                sem = float(np.std(fold_r2s) / np.sqrt(len(fold_r2s))) if fold_r2s else 0.0
            else:
                y_pred = cross_val_predict(pipe, verb_x, verb_target, cv=kf, n_jobs=-1)
                r2 = float(_r2_score(verb_target, y_pred))
                fold_r2s = []
                for _, test_idx in kf.split(verb_x):
                    if len(test_idx) >= 3:
                        fold_r2s.append(float(_r2_score(verb_target[test_idx], y_pred[test_idx])))
                sem = float(np.std(fold_r2s) / np.sqrt(len(fold_r2s))) if fold_r2s else 0.0
            if r2 > best_r2:
                best_r2, best_sem = r2, sem

    return best_r2, best_sem, {c: float(v) for c, v in verb_per_concept.items()}, verb_valid_concepts


# =============================================================================
# Ridge R^2 comparison plot
# =============================================================================

def _plot_logit_attr_ridge(
    results: Dict,
    output_dir: Path,
    plot_filename: str = "r2_comparison.png",
    target_mode: str = 'detection_rate',
):
    """Generate plots from logit attribution ridge results."""
    from plot_style import (
        DARK_ORANGE, DARK_BLUE, LIGHT_PURPLE, DARK_PURPLE,
        GREY, MEDIUM_ORANGE, CLAY, OLIVE, SKY, FIG,
        LIGHT_SLATE, BACKGROUND, AQUA_500,
    )
    from matplotlib.colors import LinearSegmentedColormap

    print("\n--- Generating plots ---")

    baselines = results['baselines']
    sweep = results['sweep']

    FS_LABEL = 20
    FS_TICK = 16
    FS_LEGEND = 14
    FS_ANNOT = 16

    # =========================================================================
    # Plot 1: R^2 Comparison -- Ridge curve + horizontal baselines
    # =========================================================================
    fig, ax = plt.subplots(figsize=(3.74, 3.42))
    fig.subplots_adjust(top=0.65)

    FS_LABEL_R2 = 18
    FS_LEGEND_R2 = 12.7

    # Ridge R^2 curve (with 95% CI = 1.96 * SEM)
    n_vals = [s['n'] for s in sweep]
    r2_vals = [s['r2'] for s in sweep]
    sem_vals = [s['sem'] for s in sweep]

    r2_arr = np.array(r2_vals)
    sem_arr = np.array(sem_vals)
    ci95 = 1.96 * sem_arr

    ax.plot(n_vals, r2_vals, '-o', color=DARK_BLUE, markersize=7, linewidth=4,
            label=f"Transcoder features (best R\u00b2={max(r2_vals):.3f})", zorder=5)
    ax.fill_between(n_vals, r2_arr - ci95, r2_arr + ci95, alpha=0.15, color=DARK_BLUE)

    # Horizontal baselines (ordered: concept vectors, mean-diff, verbalizability)
    baseline_specs = [
        ('concept_vectors_r2', OLIVE, 'Concept vectors'),
        ('mean_diff_projection_r2', DARK_ORANGE, r'Projection onto $d_{\Delta\mu}$'),
        ('verbalizability_r2', LIGHT_PURPLE, 'Verbalizability'),
    ]
    for key, color, label in baseline_specs:
        val = baselines.get(key)
        if val is not None:
            sem_key = key.replace('_r2', '_sem')
            baseline_sem = baselines.get(sem_key)
            ax.axhline(y=val, color=color, linestyle='--', linewidth=4, alpha=0.85,
                        label=f"{label} (R\u00b2={val:.3f})")
            if baseline_sem is not None:
                bci = 1.96 * baseline_sem
                ax.axhspan(val - bci, val + bci, color=color, alpha=0.08)

    ax.set_xscale('log')
    ax.set_xlabel('# transcoder features', fontsize=FS_LABEL_R2)
    ax.set_ylabel('30-fold CV R\u00b2', fontsize=FS_LABEL_R2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, fontsize=FS_LEGEND_R2, loc='upper left',
                     bbox_to_anchor=(-0.07, 1.0), bbox_transform=fig.transFigure,
                     ncol=1, framealpha=0.9, borderaxespad=0, handletextpad=0.5,
                     handlelength=1.5)
    for line in leg.get_lines():
        line.set_linewidth(2.9)
        line.set_markersize(4)
        if line.get_linestyle() != '-':
            line.set_dash_capstyle('butt')
            line.set_linestyle((1.2, (3.8, 1.7)))
    ax.grid(True, alpha=0.2, color=GREY)
    ax.set_ylim(bottom=0)
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    # =========================================================================
    # Plot 2: Combination experiments (_more.png)
    # =========================================================================
    combo = results.get('combo', {})
    combo_sweeps = combo.get('sweeps', {})
    if combo_sweeps:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        fig.subplots_adjust(top=0.62)

        FS_LABEL_R2 = 18
        FS_LEGEND_R2 = 11.0

        n_vals = [s['n'] for s in sweep]
        r2_vals = [s['r2'] for s in sweep]
        sem_vals = [s['sem'] for s in sweep]
        r2_arr = np.array(r2_vals)
        ci95 = 1.96 * np.array(sem_vals)
        ax.plot(n_vals, r2_vals, '-o', color=DARK_BLUE, markersize=6, linewidth=4,
                label=f"TF only (best R\u00b2={max(r2_vals):.3f})", zorder=5)
        ax.fill_between(n_vals, r2_arr - ci95, r2_arr + ci95, alpha=0.10, color=DARK_BLUE)

        combo_baseline_specs = [
            ('concept_vectors_r2', OLIVE, 'CV only'),
            ('mean_diff_projection_r2', DARK_ORANGE, 'MD only'),
            ('verbalizability_r2', LIGHT_PURPLE, 'Verb only'),
        ]
        for key, color, label in combo_baseline_specs:
            val = baselines.get(key)
            if val is not None:
                ax.axhline(y=val, color=color, linestyle=':', linewidth=4, alpha=0.6,
                            label=f"{label} (R\u00b2={val:.3f})")

        md_cv = combo.get('md_cv', {})
        if md_cv:
            ax.axhline(y=md_cv['r2'], color=DARK_PURPLE, linestyle='--', linewidth=4, alpha=0.85,
                        label=f"MD + CV (R\u00b2={md_cv['r2']:.3f})")

        combo_specs = [
            ('md_tf', DARK_ORANGE, 'MD + TF', '-s'),
            ('cv_tf', OLIVE, 'CV + TF', '-^'),
            ('md_cv_tf', FIG, 'MD + CV + TF', '-D'),
        ]
        for key, color, label, marker_style in combo_specs:
            if key in combo_sweeps and combo_sweeps[key]:
                cn = [s['n'] for s in combo_sweeps[key]]
                cr = [s['r2'] for s in combo_sweeps[key]]
                cs = [s['sem'] for s in combo_sweeps[key]]
                best_cr = max(cr)
                ax.plot(cn, cr, marker_style, color=color, markersize=4, linewidth=4,
                        label=f"{label} (best R\u00b2={best_cr:.3f})", zorder=4)
                cr_arr = np.array(cr)
                ci = 1.96 * np.array(cs)
                ax.fill_between(cn, cr_arr - ci, cr_arr + ci, alpha=0.08, color=color)

        ax.set_xscale('log')
        ax.set_xlabel('# transcoder features', fontsize=FS_LABEL_R2)
        ax.set_ylabel('30-fold CV R\u00b2', fontsize=FS_LABEL_R2)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND_R2, loc='upper center', bbox_to_anchor=(0.5, 1.68),
                  ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.2, color=GREY)
        ax.set_ylim(bottom=0)
        more_filename = plot_filename.replace('.png', '_more.png')
        more_path = output_dir / more_filename
        fig.savefig(more_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  Saved: {more_path}")
    else:
        print("  Skipping combination plot (no combo results)")

    # =========================================================================
    # Plot 3: PCA of concept vectors with direction cosines
    # =========================================================================
    _pca_data = None
    try:
        from sklearn.decomposition import PCA as _PCA

        steering_layer = results.get('steering_layer', 37)
        vectors_dir = get_concept_vectors_dir(steering_layer)
        det_rates = results.get('detection_rates', {})
        concepts_with_dr = [c for c in results.get('concepts_list', []) if c in det_rates]

        if len(concepts_with_dr) >= 50 and vectors_dir.exists():
            V_list, V_concepts = [], []
            for c in concepts_with_dr:
                vp = vectors_dir / f"{c}.pt"
                if vp.exists():
                    V_list.append(torch.load(vp, map_location='cpu').float().numpy().flatten())
                    V_concepts.append(c)
            V = np.array(V_list)
            dr_vals = np.array([det_rates[c] for c in V_concepts])

            V_normed = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)

            pca_model = _PCA(n_components=10)
            pca_model.fit(V_normed)
            proj = pca_model.transform(V_normed)
            pc1 = pca_model.components_[0]

            det_base = Path(f"analysis/04b_vector_geometry/gemma3_27b/layer_{steering_layer}_strength_4.0/detection_rate")
            _norm = lambda v: v / (np.linalg.norm(v) + 1e-10)
            directions = {}
            for dname, fname in [('d_Md', 'introspection_direction_mean_diff.pt'),
                                  ('d_ridge', 'introspection_direction_ridge_regression.pt')]:
                dp = det_base / fname
                if dp.exists():
                    directions[dname] = _norm(torch.load(dp, map_location='cpu').float().numpy().flatten())

            ref_path = Path("analysis/03d_refusal_abliteration/gemma3_27b/refusal_directions.pt")
            if ref_path.exists():
                ref_all = torch.load(ref_path, map_location='cpu').float().numpy()
                directions['d_ref'] = _norm(ref_all[steering_layer])

            directions['d_PC1'] = _norm(pc1)

            dir_names = ['d_PC1', 'd_Md', 'd_ridge', 'd_ref']
            dir_names = [d for d in dir_names if d in directions]
            latex_names = {
                'd_PC1': r'$d_{\mathrm{PC1}}$',
                'd_Md': r'$d_{\Delta\mu}$',
                'd_ridge': r'$d_{\mathrm{ridge}}$',
                'd_ref': r'$d_{\mathrm{refusal}}$',
            }
            cos_lines = []
            for i, n1 in enumerate(dir_names):
                for n2 in dir_names[i+1:]:
                    c = np.dot(directions[n1], directions[n2])
                    cos_lines.append(f"cos({latex_names[n1]}, {latex_names[n2]}) = {c:.2f}")

            anthro_cmap_pca = LinearSegmentedColormap.from_list('clay_aqua_pca', [CLAY, '#F5F4ED', AQUA_500])
            fig, ax = plt.subplots(figsize=(5.5, 5))
            fig.subplots_adjust(top=0.90)

            sc = ax.scatter(proj[:, 0], proj[:, 1], c=dr_vals, cmap=anthro_cmap_pca,
                            s=68, alpha=0.9, edgecolors=GREY, linewidths=0.3, vmin=0, vmax=1)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Detection rate', fontsize=FS_LABEL)
            cbar.ax.tick_params(labelsize=FS_TICK)

            ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)', fontsize=FS_LABEL)
            ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)', fontsize=FS_LABEL)
            ax.tick_params(axis='both', labelsize=FS_TICK)
            ax.grid(True, alpha=0.2, color=GREY)

            dr_order = np.argsort(dr_vals)
            label_indices = list(dr_order[:3]) + list(dr_order[-3:])
            pc1v, pc2v = proj[:, 0], proj[:, 1]
            label_fontsize = (FS_ANNOT - 4) * 1.45
            x_mid = (pc1v.min() + pc1v.max()) / 2
            label_specs = []
            for i in label_indices:
                label_color = anthro_cmap_pca(dr_vals[i])
                label_color = (label_color[0] * 0.65, label_color[1] * 0.65, label_color[2] * 0.65, 1.0)
                if pc1v[i] < x_mid:
                    x_off, ha_val = 14, 'left'
                else:
                    x_off, ha_val = -14, 'right'
                y_off = 10
                label_specs.append((i, x_off, y_off, ha_val, label_color))

            for j in range(len(label_specs)):
                for k in range(j + 1, len(label_specs)):
                    ij, _, yj, _, _ = label_specs[j]
                    ik, _, yk, _, _ = label_specs[k]
                    dx = abs(pc1v[ij] - pc1v[ik])
                    dy = abs(pc2v[ij] - pc2v[ik])
                    if dx < 0.25 and dy < 0.15:
                        i_k, x_off_k, _, ha_k, col_k = label_specs[k]
                        label_specs[k] = (i_k, x_off_k, 26, ha_k, col_k)

            for i, x_off, y_off, ha_val, label_color in label_specs:
                ax.annotate(
                    V_concepts[i], (pc1v[i], pc2v[i]),
                    fontsize=label_fontsize, fontweight='bold', color=label_color,
                    ha=ha_val, va='bottom',
                    xytext=(x_off, y_off), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', color=GREY, alpha=0.5, lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'),
                )

            keep_lines = cos_lines[:4]
            row1 = '  '.join(keep_lines[:2])
            row2 = '  '.join(keep_lines[2:4])
            legend_text = row1 + '\n' + row2
            fig.text(-0.07, 1.015, legend_text, fontsize=(FS_LEGEND - 1) * 1.3 / 1.2463, va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=GREY))
            pca_path = output_dir / f"pca{plot_filename.replace('r2_comparison', '').replace('.png', '')}.png"
            fig.savefig(pca_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {pca_path}")

            geometry_pca_dir = det_base / "plots"
            geometry_pca_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(pca_path, geometry_pca_dir / "pca.png")
            print(f"  Copied: {geometry_pca_dir / 'pca.png'}")

            _pca_data = dict(proj=proj, pca_model=pca_model, dr_vals=dr_vals,
                             V_concepts=V_concepts, cos_lines=cos_lines,
                             label_specs=label_specs, pc1v=pc1v, pc2v=pc2v,
                             anthro_cmap_pca=anthro_cmap_pca)
        else:
            print(f"  Skipping PCA plot (insufficient concepts or missing vectors dir)")
    except Exception as e:
        print(f"  WARNING: PCA plot failed: {e}")

    # =========================================================================
    # Plot 4: Mean-diff projection vs Verbalizability scatter (detection_rate only)
    # =========================================================================
    if target_mode != 'detection_rate':
        print("  Skipping scatter plot (only generated for detection_rate target)")
        print("  Plots complete.")
        return

    verb_data = results.get('verbalizability', {})
    verb_per_concept = verb_data.get('per_concept', {})
    mean_diff_projs = results.get('mean_diff_projections', {})
    det_rates = results.get('detection_rates', {})

    anthro_cmap = LinearSegmentedColormap.from_list('clay_aqua', [CLAY, '#F5F4ED', AQUA_500])

    if verb_per_concept and mean_diff_projs:
        scatter_concepts = [c for c in verb_per_concept if c in mean_diff_projs and c in det_rates]
        if len(scatter_concepts) >= 10:
            md_vals = np.array([mean_diff_projs[c] for c in scatter_concepts])
            vb_vals = np.array([verb_per_concept[c] for c in scatter_concepts])
            dr_vals = np.array([det_rates[c] for c in scatter_concepts])

            fig, ax = plt.subplots(figsize=(5.5, 5))
            fig.subplots_adjust(top=0.90)
            sc = ax.scatter(md_vals, vb_vals, c=dr_vals, cmap=anthro_cmap, s=68, alpha=0.9,
                            edgecolors=GREY, linewidths=0.3, vmin=0, vmax=1)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Detection rate', fontsize=FS_LABEL)
            cbar.ax.tick_params(labelsize=FS_TICK)

            rho, pval = stats.spearmanr(md_vals, vb_vals)

            ax.set_xlabel('Mean-difference projection', fontsize=FS_LABEL)
            ax.set_ylabel('Verbalizability', fontsize=FS_LABEL)
            ax.tick_params(axis='both', labelsize=FS_TICK)
            ax.grid(True, alpha=0.2, color=GREY)

            spearman_text = f"Spearman r = {rho:.3f}"
            fig.text(-0.07, 1.015, spearman_text, fontsize=(FS_LEGEND - 1) * 1.3 / 1.2463, va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=GREY))

            plot_path = output_dir / "meandiff_vs_verbalizability.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {plot_path}")
        else:
            print(f"  WARNING: Only {len(scatter_concepts)} concepts with both metrics, skipping scatter plot")
    else:
        print("  WARNING: Missing verbalizability or mean-diff data, skipping scatter plot")

    # =========================================================================
    # Plot 5: Combined PCA + Mean-diff scatter (side by side)
    # =========================================================================
    if _pca_data is not None and verb_per_concept and mean_diff_projs:
        scatter_concepts = [c for c in verb_per_concept if c in mean_diff_projs and c in det_rates]
        if len(scatter_concepts) >= 10:
            md_vals = np.array([mean_diff_projs[c] for c in scatter_concepts])
            vb_vals = np.array([verb_per_concept[c] for c in scatter_concepts])
            dr_vals_md = np.array([det_rates[c] for c in scatter_concepts])
            rho, _ = stats.spearmanr(md_vals, vb_vals)

            proj = _pca_data['proj']
            pca_model = _pca_data['pca_model']
            dr_vals_pca = _pca_data['dr_vals']
            V_concepts = _pca_data['V_concepts']
            cos_lines = _pca_data['cos_lines']
            pca_label_specs = _pca_data['label_specs']
            pc1v = _pca_data['pc1v']
            pc2v = _pca_data['pc2v']
            anthro_cmap_pca = _pca_data['anthro_cmap_pca']
            anthro_cmap_md = LinearSegmentedColormap.from_list('clay_aqua_md', [CLAY, '#F5F4ED', AQUA_500])

            ll_neg_tokens = ['confused', 'ambiguous', 'referring']
            ll_pos_tokens = ['facts', 'knowledge', 'overview']

            ll_scores = {}
            try:
                steering_layer_c = results.get('steering_layer', 37)
                steering_strength_c = results.get('steering_strength', 4.0)
                md_dir_path_c = resolve_mean_diff_path(get_direction_base(steering_layer_c, steering_strength_c))
                if md_dir_path_c.exists():
                    md_dir_c = torch.load(md_dir_path_c, map_location='cpu', weights_only=True).float().flatten()
                    md_dir_c = md_dir_c / (md_dir_c.norm() + 1e-10)
                    lm_head_c, tok_c = get_lm_head_and_tokenizer()
                    if lm_head_c is not None:
                        logits_c = lm_head_c.float() @ md_dir_c
                        for tok_str in ll_neg_tokens + ll_pos_tokens:
                            ids = tok_c.encode(tok_str, add_special_tokens=False)
                            if ids:
                                ll_scores[tok_str] = logits_c[ids[0]].item()
            except Exception as e:
                print(f"  WARNING: logit lens for combined plot: {e}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.60, 3.17))
            fig.subplots_adjust(top=0.78, wspace=0.45)

            # Left panel: PCA
            sc1 = ax1.scatter(proj[:, 0], proj[:, 1], c=dr_vals_pca, cmap=anthro_cmap_pca,
                              s=68, alpha=0.9, edgecolors=GREY, linewidths=0.3, vmin=0, vmax=1)
            ax1.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%})', fontsize=FS_LABEL)
            ax1.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%})', fontsize=FS_LABEL)
            ax1.tick_params(axis='both', labelsize=FS_TICK)
            ax1.yaxis.set_major_locator(plt.MultipleLocator(0.4))
            ax1.grid(True, alpha=0.2, color=GREY)

            short_cos_lines = []
            dir_names_comb = ['d_PC1', 'd_Md', 'd_ridge', 'd_ref']
            dir_names_comb = [d for d in dir_names_comb if d in _pca_data.get('_directions', {})]
            keep_orig = [cos_lines[0], cos_lines[3], cos_lines[2]]
            cos_legend_text = '\n'.join(keep_orig)
            cos_fs = (FS_LEGEND - 1) * 1.3 / 1.2463 * 0.85 * 1.15 * 1.1 * 1.125
            ax1.text(0.5, 1.10, cos_legend_text, va='bottom', ha='center',
                     transform=ax1.transAxes, fontsize=cos_fs,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=GREY))

            # Right panel: Mean-diff vs Verbalizability
            sc2 = ax2.scatter(md_vals, vb_vals, c=dr_vals_md, cmap=anthro_cmap_md, s=68, alpha=0.9,
                              edgecolors=GREY, linewidths=0.3, vmin=0, vmax=1)
            ax2.set_xlabel(r'Projection onto $d_{\Delta\mu}$', fontsize=FS_LABEL)
            ax2.set_ylabel('Verbalizability', fontsize=FS_LABEL)
            ax2.tick_params(axis='both', labelsize=FS_TICK)
            ax2.xaxis.set_major_locator(plt.MultipleLocator(2000))
            ax2.set_xlim(left=-3500)
            ax2.yaxis.set_major_locator(plt.MultipleLocator(400))
            ax2.grid(True, alpha=0.2, color=GREY)

            ax2.text(0.05, 0.93, f"Spearman r = {rho:.3f}", transform=ax2.transAxes,
                     fontsize=FS_ANNOT / 1.1, va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=GREY))

            # Logit lens: 3-row diverging bars (neg left, pos right per row)
            if ll_scores:
                neg_pairs = sorted([(t, ll_scores.get(t, 0)) for t in ll_neg_tokens],
                                   key=lambda x: abs(x[1]), reverse=True)
                pos_pairs = sorted([(t, ll_scores.get(t, 0)) for t in ll_pos_tokens],
                                   key=lambda x: abs(x[1]), reverse=True)
                clay_dark = '#8B4513'
                aqua_dark = '#1A6B5A'
                n_rows = min(len(neg_pairs), len(pos_pairs))
                ax_inset = ax2.inset_axes([0.05, 1.18, 0.95, 0.275])
                for row in range(n_rows):
                    neg_tok, neg_sc = neg_pairs[row]
                    pos_tok, pos_sc = pos_pairs[row]
                    ax_inset.barh(n_rows - 1 - row, neg_sc, color=CLAY,
                                  edgecolor=GREY, linewidth=0.3, height=0.55)
                    ax_inset.text(neg_sc, n_rows - 1 - row, f'{neg_tok} ', ha='right', va='center',
                                  fontsize=(FS_LEGEND - 3) * 1.1 * 1.125, fontstyle='italic', color=clay_dark)
                    ax_inset.barh(n_rows - 1 - row, pos_sc, color=AQUA_500,
                                  edgecolor=GREY, linewidth=0.3, height=0.55)
                    ax_inset.text(pos_sc, n_rows - 1 - row, f' {pos_tok}', ha='left', va='center',
                                  fontsize=(FS_LEGEND - 3) * 1.1 * 1.125, fontstyle='italic', color=aqua_dark)
                ax_inset.axvline(0, color=GREY, linewidth=0.5, zorder=0)
                ax_inset.set_ylim(-0.7, n_rows - 0.5)
                ax_inset.set_yticks([])
                ax_inset.set_title(r'Logit lens on $d_{\Delta\mu}$', fontsize=(FS_LEGEND - 2) * 1.1 * 1.125,
                                   pad=2, loc='center')
                max_abs = max(max(abs(s) for _, s in neg_pairs),
                              max(abs(s) for _, s in pos_pairs)) * 1.15
                ax_inset.set_xlim(-max_abs, max_abs)
                ax_inset.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                ax_inset.tick_params(axis='x', labelsize=(FS_TICK - 5) * 1.1 * 1.125, pad=3)
                ax_inset.spines['top'].set_visible(False)
                ax_inset.spines['right'].set_visible(False)
                ax_inset.spines['left'].set_visible(False)

            cbar = fig.colorbar(sc2, ax=[ax1, ax2], location='right', shrink=1.0, pad=0.02, fraction=0.046, aspect=20 / 1.25)
            cbar.set_label('Detection rate', fontsize=FS_LABEL)
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            cbar.ax.tick_params(labelsize=FS_TICK)

            combined_path = output_dir / "pca_meandiff_combined.png"
            fig.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {combined_path}")

    print("  Plots complete.")


# =============================================================================
# Main analysis function
# =============================================================================

def run_logit_attr_ridge_analysis(
    steering_layer: int = DEFAULT_STEERING_LAYER,
    steering_strength: float = DEFAULT_STEERING_STRENGTH,
    token_mode: str = DEFAULT_TOKEN_MODE,
    transcoder_l0: str = DEFAULT_TRANSCODER_L0,
    transcoder_width: str = DEFAULT_TRANSCODER_WIDTH,
    output_dir: Path = None,
    plots_only: bool = False,
    target_mode: str = 'detection_rate',
):
    """
    Ridge R^2 comparison: transcoder features (norm-product ranking from bulk
    logit attribution) vs baselines (mean-diff projection, concept vectors,
    verbalizability).

    target_mode controls the prediction target:
      - 'detection_rate': raw detection rate (default)
      - 'logit_transform': logit(detection_rate)
      - 'p_det': net p_det attribution from transcoders
      - 'yes_no': net yes/no attribution from transcoders

    Produces:
      1. r2_comparison[_suffix].png -- Ridge R^2 curve vs top-N features + baselines
      2. meandiff_vs_verbalizability.png -- scatter (detection_rate only)
      3. logit_attr_ridge_results[_suffix].json -- all numerical results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
    from sklearn.metrics import r2_score as _r2_score
    from sklearn.pipeline import Pipeline
    from collections import defaultdict

    print("\n" + "=" * 70)
    print(f"LOGIT ATTRIBUTION RIDGE ANALYSIS (target={target_mode})")
    print("=" * 70)

    if output_dir is None:
        wp = get_width_prefix(transcoder_width)
        output_dir = OUTPUT_BASE / f"L{steering_layer}_binary_classification_{token_mode}_{wp}{transcoder_l0}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Target-mode-specific filenames
    if target_mode == 'detection_rate':
        results_suffix = ""
        plot_suffix = ""
    else:
        results_suffix = f"_{target_mode}"
        plot_suffix = f"_{target_mode}"

    results_path = output_dir / f"logit_attr_ridge_results{results_suffix}.json"
    plot_filename = f"r2_comparison{plot_suffix}.png"

    # =========================================================================
    # If plots_only, load saved results and skip to plotting
    # =========================================================================
    if plots_only:
        if not results_path.exists():
            raise FileNotFoundError(
                f"No saved results found at {results_path}. "
                "Run without --logit-attr-ridge-plots-only first."
            )
        print(f"  Loading saved results from {results_path}")
        with open(results_path) as f:
            saved = json.load(f)
        _plot_logit_attr_ridge(saved, output_dir, plot_filename=plot_filename, target_mode=target_mode)
        return

    # =========================================================================
    # Step 1: Load concept partition & detection rates
    # =========================================================================
    print("\n--- Step 1: Load concept partition & detection rates ---")
    geometry_base = get_geometry_base(steering_layer, steering_strength)
    with open(geometry_base / "detection_rate" / "subspace_analysis.json") as f:
        partition_data = json.load(f)
    success_concepts = set(partition_data['success_concepts'])
    failure_concepts = set(partition_data['failure_concepts'])

    # Load detection rates from experiment 02 (steering evaluation)
    steering_base = get_steering_base(steering_layer, steering_strength)
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

    # Build concept list from cached activations (same order as feature matrix)
    wp = get_width_prefix(transcoder_width)
    config_str = f"L{steering_layer}_S{steering_strength}_{token_mode}_{wp}{transcoder_l0}"
    matrix_dir = CACHED_ACTIVATIONS_BASE / config_str / "layer_matrices"
    analysis_layers = get_analysis_layers(steering_layer)

    first_npz = np.load(matrix_dir / f"layer_{analysis_layers[0]}_mean.npz", allow_pickle=True)
    npz_concepts = [str(x) for x in first_npz['concepts'].tolist()]

    concepts_list = []
    labels = []
    for c in npz_concepts:
        if c in success_concepts:
            concepts_list.append(c); labels.append(1)
        elif c in failure_concepts:
            concepts_list.append(c); labels.append(0)
    y = np.array(labels)
    concept_to_row = {c: i for i, c in enumerate(npz_concepts)}
    row_indices = [concept_to_row[c] for c in concepts_list]

    y_continuous = np.array([detection_rates.get(c, np.nan) for c in concepts_list])
    valid_mask = ~np.isnan(y_continuous)
    print(f"  {len(concepts_list)} concepts ({sum(labels)} success, {len(labels) - sum(labels)} failure)")
    print(f"  {valid_mask.sum()} concepts with detection rates for regression")

    # =========================================================================
    # Step 2: Transform target based on target_mode
    # =========================================================================
    target_label = "detection rate"
    y_original_probs = None

    if target_mode == 'logit_transform':
        eps = 0.025
        y_original_probs = y_continuous.copy()
        y_clipped = np.clip(y_continuous[valid_mask], eps, 1 - eps)
        y_logit = np.full_like(y_continuous, np.nan)
        y_logit[valid_mask] = np.log(y_clipped / (1 - y_clipped))
        y_continuous = y_logit
        target_label = "logit(detection rate)"
        print(f"  Logit transform: eps={eps}, y range [{y_continuous[valid_mask].min():.2f}, {y_continuous[valid_mask].max():.2f}]")
        print(f"  R^2 will be evaluated in probability space (sigmoid -> R^2 vs original detection rates)")

    elif target_mode in ('p_det', 'yes_no'):
        attr_key = 'attributions_pdet' if target_mode == 'p_det' else 'attributions_yesno'
        target_label = f"net {'p_det' if target_mode == 'p_det' else 'yes/no'} attribution"
        bulk_dir = Path(f"analysis/09b_causal_pathway/{wp}{transcoder_l0}/bulk_logit_attribution_L{steering_layer}_{token_mode}")
        concept_attr_dir = bulk_dir / "concept_attributions"
        print(f"  Loading net attributions ({attr_key}) from {concept_attr_dir}")

        net_attrs = {}
        for c in concepts_list:
            npz_path = concept_attr_dir / f"{c}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                net_attrs[c] = float(data[attr_key].sum())
            else:
                net_attrs[c] = np.nan

        y_continuous = np.array([net_attrs.get(c, np.nan) for c in concepts_list])
        valid_mask = ~np.isnan(y_continuous)
        print(f"  {valid_mask.sum()} concepts with net attributions, y range [{y_continuous[valid_mask].min():.4f}, {y_continuous[valid_mask].max():.4f}]")

    # =========================================================================
    # Step 3: Compute verbalizability baseline R^2
    # =========================================================================
    print("\n--- Step 3: Compute verbalizability baseline ---")
    verb_r2, verb_sem, verb_per_concept, verb_valid_concepts = _compute_verbalizability_baseline(
        steering_layer=steering_layer,
        concepts_list=concepts_list,
        y_continuous=y_continuous,
        valid_mask=valid_mask,
        y_original_probs=y_original_probs,
    )
    print(f"  Verbalizability R^2 = {verb_r2:.4f} +/- {verb_sem:.4f}" if verb_r2 is not None else "  Verbalizability R^2: failed")

    # =========================================================================
    # Step 4: Compute mean-diff projections + baseline R^2 with CV
    # =========================================================================
    print("\n--- Step 4: Compute mean-diff projections + baselines with CV ---")
    mean_diff_projections = load_concept_mean_diff_projections(
        steering_layer=steering_layer,
        steering_strength=steering_strength,
        concepts_list=concepts_list,
    )

    n_folds_baseline = 30
    kf_baseline = KFold(n_splits=n_folds_baseline, shuffle=True, random_state=42)
    baseline_alphas = [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

    def _logit_aggregate_r2(X_in, y_logit_in, y_orig_in, kf, pipe):
        """Compute aggregate R^2 in probability space using cross_val_predict."""
        y_pred_logit = cross_val_predict(pipe, X_in, y_logit_in, cv=kf, n_jobs=-1)
        y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logit, -20, 20)))
        r2 = float(_r2_score(y_orig_in, y_pred_prob))
        fold_r2s = []
        for _, test_idx in kf.split(X_in):
            if len(test_idx) < 3:
                continue
            fold_r2s.append(float(_r2_score(y_orig_in[test_idx], y_pred_prob[test_idx])))
        sem = float(np.std(fold_r2s) / np.sqrt(len(fold_r2s))) if fold_r2s else 0.0
        return r2, sem

    def _best_ridge_cv(X_in, y_in, kf, alphas, y_orig_probs=None):
        """Sweep alphas, return best (r2, sem, alpha).
        Uses cross_val_predict for single R^2 on concatenated OOS predictions."""
        best_r2, best_sem, best_a = -999, None, None
        for a in alphas:
            pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=a))])
            if y_orig_probs is not None:
                r2, sem = _logit_aggregate_r2(X_in, y_in, y_orig_probs, kf, pipe)
            else:
                y_pred = cross_val_predict(pipe, X_in, y_in, cv=kf, n_jobs=-1)
                r2 = float(_r2_score(y_in, y_pred))
                fold_r2s = []
                for _, test_idx in kf.split(X_in):
                    if len(test_idx) >= 3:
                        fold_r2s.append(float(_r2_score(y_in[test_idx], y_pred[test_idx])))
                sem = float(np.std(fold_r2s) / np.sqrt(len(fold_r2s))) if fold_r2s else 0.0
            if r2 > best_r2:
                best_r2, best_sem, best_a = r2, sem, a
        return best_r2, best_sem, best_a

    # Mean-diff projection baseline (1D feature)
    md_x = np.array([mean_diff_projections.get(c, np.nan) for c in concepts_list])
    md_valid = valid_mask & ~np.isnan(md_x)
    md_x_reg = md_x[md_valid].reshape(-1, 1)
    md_y_reg = y_continuous[md_valid]
    md_y_orig = y_original_probs[md_valid] if y_original_probs is not None else None
    mean_diff_r2, mean_diff_sem, md_alpha = _best_ridge_cv(md_x_reg, md_y_reg, kf_baseline, baseline_alphas, y_orig_probs=md_y_orig)
    print(f"  Mean-diff projection R^2 = {mean_diff_r2:.4f} +/- {mean_diff_sem:.4f} (alpha={md_alpha:.0f})")

    # Concept vectors baseline (5376D feature)
    X_vectors, valid_vec_concepts = load_concept_vectors(concepts_list, steering_layer)
    if X_vectors is not None and len(valid_vec_concepts) > 0:
        vec_concept_set = set(valid_vec_concepts)
        vec_indices = [i for i, c in enumerate(concepts_list) if c in vec_concept_set and valid_mask[i]]
        vec_concept_order = [concepts_list[i] for i in vec_indices]
        vec_row_map = {c: j for j, c in enumerate(valid_vec_concepts)}
        X_vec_reg = np.array([X_vectors[vec_row_map[c]] for c in vec_concept_order])
        y_vec_reg = np.array([y_continuous[i] for i in vec_indices])
        y_vec_orig = np.array([y_original_probs[i] for i in vec_indices]) if y_original_probs is not None else None
        concept_vectors_r2, concept_vectors_sem, cv_alpha = _best_ridge_cv(X_vec_reg, y_vec_reg, kf_baseline, baseline_alphas, y_orig_probs=y_vec_orig)
        print(f"  Concept vectors R^2 = {concept_vectors_r2:.4f} +/- {concept_vectors_sem:.4f} (alpha={cv_alpha:.0f})")
    else:
        concept_vectors_r2 = None
        concept_vectors_sem = None
        print("  Concept vectors: could not load")

    # =========================================================================
    # Step 5: Load bulk attribution ranking (norm-product)
    # =========================================================================
    print("\n--- Step 5: Load bulk attribution ranking ---")
    bulk_dir = Path(f"analysis/09b_causal_pathway/{wp}{transcoder_l0}/bulk_logit_attribution_L{steering_layer}_{token_mode}")
    with open(bulk_dir / "aggregated_attribution.json") as f:
        agg_data = json.load(f)
    print(f"  {agg_data['n_concepts']} concepts, {agg_data['n_unique_features']} unique features")

    all_features = agg_data['all_features']
    coverages = np.array([f['n_concepts'] for f in all_features])
    mean_attrs = np.array([f['mean_abs_attr_pdet'] for f in all_features])

    # Norm-product ranking
    cov_norm = coverages / coverages.max()
    attr_norm = mean_attrs / mean_attrs.max()
    norm_product = cov_norm * attr_norm
    norm_product_order = np.argsort(-norm_product)
    ranked_features = [all_features[i] for i in norm_product_order]

    print(f"  Norm-product top 5:")
    for i, f in enumerate(ranked_features[:5]):
        idx = norm_product_order[i]
        print(f"    {i+1}. L{f['layer']}_F{f['feat_id']}: mean={f['mean_abs_attr_pdet']:.2f}, "
              f"n={f['n_concepts']}/{agg_data['n_concepts']}, score={norm_product[idx]:.4f}")

    # =========================================================================
    # Step 6: Build all-layer feature matrix
    # =========================================================================
    print("\n--- Step 6: Build feature matrix ---")
    blocks = []
    feature_keys = []
    for layer in analysis_layers:
        with np.load(matrix_dir / f"layer_{layer}_mean.npz", allow_pickle=True) as d:
            mat = d['matrix'][row_indices].astype(np.float32)
        var = np.var(mat, axis=0)
        nz = var > 0
        blocks.append(mat[:, nz])
        for fid in np.where(nz)[0]:
            feature_keys.append((layer, int(fid)))
    X_all = np.concatenate(blocks, axis=1)
    del blocks
    print(f"  X_all: {X_all.shape}")

    # Map ranked features to columns
    key_to_col = {k: i for i, k in enumerate(feature_keys)}
    ranked_cols = []
    for f in ranked_features:
        key = (f['layer'], f['feat_id'])
        if key in key_to_col:
            ranked_cols.append(key_to_col[key])
    print(f"  {len(ranked_cols)} ranked features mapped to matrix columns")

    # =========================================================================
    # Step 7: Ridge R^2 sweep with per-N optimal alpha
    # =========================================================================
    print("\n--- Step 7: Ridge R^2 sweep ---")
    X_reg = X_all[valid_mask]
    y_reg = y_continuous[valid_mask]
    y_reg_orig = y_original_probs[valid_mask] if y_original_probs is not None else None
    n_folds = 30
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_max = min(len(ranked_cols), X_reg.shape[1])

    alphas = [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    n_values = sorted(set(n for n in (
        [10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
         6000, 8000, 10000, 15000, 20000, 30000]
    ) if n <= n_max))

    sweep_results = []
    best_overall_r2 = -999

    for n_feat in n_values:
        top = ranked_cols[:n_feat]
        best_r2_for_n = -999
        best_alpha_for_n = None
        best_sem_for_n = None

        for alpha in alphas:
            pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
            if y_reg_orig is not None:
                r2, sem = _logit_aggregate_r2(X_reg[:, top], y_reg, y_reg_orig, kf, pipe)
            else:
                y_pred = cross_val_predict(pipe, X_reg[:, top], y_reg, cv=kf, n_jobs=-1)
                r2 = float(_r2_score(y_reg, y_pred))
                fold_r2s = []
                for _, test_idx in kf.split(X_reg[:, top]):
                    if len(test_idx) >= 3:
                        fold_r2s.append(float(_r2_score(y_reg[test_idx], y_pred[test_idx])))
                sem = float(np.std(fold_r2s) / np.sqrt(len(fold_r2s))) if fold_r2s else 0.0
            if r2 > best_r2_for_n:
                best_r2_for_n = r2
                best_alpha_for_n = alpha
                best_sem_for_n = sem

        marker = ""
        if best_r2_for_n > best_overall_r2:
            best_overall_r2 = best_r2_for_n
            marker = " *"
        print(f"  N={n_feat:>6d}: R^2={best_r2_for_n:+.4f} +/- {best_sem_for_n:.4f} (alpha={best_alpha_for_n:.0f}){marker}", flush=True)

        sweep_results.append({
            'n': n_feat,
            'best_alpha': best_alpha_for_n,
            'r2': float(best_r2_for_n),
            'sem': float(best_sem_for_n),
        })

    print(f"\n  Best: N={sweep_results[np.argmax([s['r2'] for s in sweep_results])]['n']}, "
          f"R^2={best_overall_r2:.4f}")

    # =========================================================================
    # Step 7b: Combination experiments (MD+CV, MD+TF, CV+TF, MD+CV+TF)
    # =========================================================================
    combo_results = {}
    if X_vectors is not None and concept_vectors_r2 is not None:
        print("\n--- Step 7b: Combination experiments ---")

        vec_concept_set = set(valid_vec_concepts)
        combo_indices = [i for i, c in enumerate(concepts_list)
                         if c in vec_concept_set and valid_mask[i] and not np.isnan(md_x[i])]
        combo_concepts = [concepts_list[i] for i in combo_indices]
        print(f"  {len(combo_concepts)} concepts with all features (MD + CV + TF)")

        vec_row_map = {c: j for j, c in enumerate(valid_vec_concepts)}
        X_md_combo = np.array([md_x[i] for i in combo_indices]).reshape(-1, 1)
        X_cv_combo = np.array([X_vectors[vec_row_map[concepts_list[i]]] for i in combo_indices])
        X_tf_combo = X_all[np.array(combo_indices)]
        y_combo = y_continuous[np.array(combo_indices)]
        y_combo_orig = y_original_probs[np.array(combo_indices)] if y_original_probs is not None else y_combo

        kf_combo = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        def _combo_r2(X_in, y_fit, y_eval, kf_c, combo_alphas):
            best_r2, best_sem, best_a = -999, None, None
            for a in combo_alphas:
                pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=a))])
                if y_original_probs is not None:
                    y_pred_logit = cross_val_predict(pipe, X_in, y_fit, cv=kf_c, n_jobs=-1)
                    y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logit, -20, 20)))
                    r2 = float(_r2_score(y_eval, y_pred_prob))
                    fold_r2s = []
                    for _, ti in kf_c.split(X_in):
                        if len(ti) >= 3:
                            fold_r2s.append(float(_r2_score(y_eval[ti], y_pred_prob[ti])))
                    sem = float(np.std(fold_r2s) / np.sqrt(len(fold_r2s))) if fold_r2s else 0.0
                else:
                    scores = cross_val_score(pipe, X_in, y_fit, cv=kf_c, scoring='r2', n_jobs=-1)
                    r2 = float(np.mean(scores))
                    sem = float(np.std(scores) / np.sqrt(len(scores)))
                if r2 > best_r2:
                    best_r2, best_sem, best_a = r2, sem, a
            return best_r2, best_sem

        # MD + CV (single R^2)
        X_md_cv = np.hstack([X_md_combo, X_cv_combo])
        md_cv_r2, md_cv_sem = _combo_r2(X_md_cv, y_combo, y_combo_orig, kf_combo, alphas)
        combo_results['md_cv'] = {'r2': md_cv_r2, 'sem': md_cv_sem}
        print(f"  MD + CV: R^2={md_cv_r2:.4f} +/- {md_cv_sem:.4f}")

        # Sweep combinations over N transcoder features
        combo_n_values = sorted(set(n for n in [50, 100, 500, 1000, 2000, 3000, 4500, 8000, 15000, 30000] if n <= n_max))
        combo_sweeps = {'md_tf': [], 'cv_tf': [], 'md_cv_tf': []}

        for n_feat in combo_n_values:
            top = ranked_cols[:n_feat]
            X_tf_n = X_tf_combo[:, top]

            # MD + TF(N)
            X_md_tf = np.hstack([X_md_combo, X_tf_n])
            r2_md_tf, sem_md_tf = _combo_r2(X_md_tf, y_combo, y_combo_orig, kf_combo, alphas)
            combo_sweeps['md_tf'].append({'n': n_feat, 'r2': r2_md_tf, 'sem': sem_md_tf})

            # CV + TF(N)
            X_cv_tf = np.hstack([X_cv_combo, X_tf_n])
            r2_cv_tf, sem_cv_tf = _combo_r2(X_cv_tf, y_combo, y_combo_orig, kf_combo, alphas)
            combo_sweeps['cv_tf'].append({'n': n_feat, 'r2': r2_cv_tf, 'sem': sem_cv_tf})

            # MD + CV + TF(N)
            X_all_combo = np.hstack([X_md_combo, X_cv_combo, X_tf_n])
            r2_all, sem_all = _combo_r2(X_all_combo, y_combo, y_combo_orig, kf_combo, alphas)
            combo_sweeps['md_cv_tf'].append({'n': n_feat, 'r2': r2_all, 'sem': sem_all})

            print(f"  N={n_feat:>5d}:  MD+TF={r2_md_tf:.4f}  CV+TF={r2_cv_tf:.4f}  MD+CV+TF={r2_all:.4f}")

        combo_results['sweeps'] = combo_sweeps
    else:
        print("\n  Skipping combination experiments (concept vectors not available)")

    # =========================================================================
    # Step 8: Save results
    # =========================================================================
    print("\n--- Step 8: Save results ---")
    results = {
        'baselines': {
            'mean_diff_projection_r2': mean_diff_r2,
            'mean_diff_projection_sem': mean_diff_sem,
            'concept_vectors_r2': concept_vectors_r2,
            'concept_vectors_sem': concept_vectors_sem,
            'verbalizability_r2': float(verb_r2) if verb_r2 is not None else None,
            'verbalizability_sem': float(verb_sem) if verb_sem is not None else None,
        },
        'sweep': sweep_results,
        'combo': combo_results,
        'target_mode': target_mode,
        'steering_layer': steering_layer,
        'ranking': 'norm-product',
        'n_folds': n_folds,
        'alphas_searched': alphas,
        'verbalizability': {
            'per_concept': verb_per_concept,
            'valid_concepts': verb_valid_concepts,
        },
        'mean_diff_projections': {c: float(v) for c, v in mean_diff_projections.items()},
        'concepts_list': concepts_list,
        'detection_rates': {c: float(detection_rates[c]) for c in concepts_list if c in detection_rates},
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {results_path}")

    # =========================================================================
    # Step 9: Generate plots
    # =========================================================================
    _plot_logit_attr_ridge(results, output_dir, plot_filename=plot_filename, target_mode=target_mode)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregation Analysis: Ridge R^2 comparison and verbalizability (Section 4.3)"
    )
    parser.add_argument("-sl", "--steering-layer", type=int, default=DEFAULT_STEERING_LAYER)
    parser.add_argument("-ss", "--steering-strength", type=float, default=DEFAULT_STEERING_STRENGTH)
    parser.add_argument("-tm", "--token-mode", type=str, default=DEFAULT_TOKEN_MODE)
    parser.add_argument("-tl", "--transcoder-l0", type=str, default=DEFAULT_TRANSCODER_L0)
    parser.add_argument("-tw", "--transcoder-width", type=str, default=DEFAULT_TRANSCODER_WIDTH,
                        choices=["16k", "262k"], help="Transcoder width (default: 262k)")
    parser.add_argument("--logit-attr-ridge", action="store_true",
                        help="Run logit attribution ridge R^2 comparison (transcoder features vs baselines)")
    parser.add_argument("--logit-attr-ridge-plots-only", action="store_true",
                        help="Regenerate plots from saved JSON (no recomputation)")
    parser.add_argument("--target-mode", type=str, default="detection_rate",
                        choices=["detection_rate", "logit_transform", "p_det", "yes_no"],
                        help="Target variable for ridge regression (default: detection_rate)")

    args = parser.parse_args()

    print("=" * 70)
    print("AGGREGATION ANALYSIS: GEOMETRY PANEL (Section 4.3)")
    print("=" * 70)

    if args.logit_attr_ridge or args.logit_attr_ridge_plots_only:
        print(f"Mode: logit-attr-ridge{'(plots-only)' if args.logit_attr_ridge_plots_only else ''} (target={args.target_mode})")
        print(f"Steering: L{args.steering_layer}, S={args.steering_strength}")

        wp = get_width_prefix(args.transcoder_width)
        out_dir = OUTPUT_BASE / f"L{args.steering_layer}_binary_classification_{args.token_mode}_{wp}{args.transcoder_l0}"

        run_logit_attr_ridge_analysis(
            steering_layer=args.steering_layer,
            steering_strength=args.steering_strength,
            token_mode=args.token_mode,
            transcoder_l0=args.transcoder_l0,
            transcoder_width=args.transcoder_width,
            output_dir=out_dir,
            plots_only=args.logit_attr_ridge_plots_only,
            target_mode=args.target_mode,
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
    else:
        parser.print_help()
        print("\nRun with --logit-attr-ridge to execute the analysis.")


if __name__ == "__main__":
    main()
