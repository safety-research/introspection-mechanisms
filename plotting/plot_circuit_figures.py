#!/usr/bin/env python3
"""
Generate shared multi-concept plots from existing causal pathway analysis data.

Reads from analysis/09b_causal_pathway/262k_big/ and produces combined plots
under analysis/09b_causal_pathway/262k_big/shared/.

Each plot is a 2x3 grid showing 6 concepts (those with lowest max n) separately.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Anthroplot colors
BLUE_600 = "#1B67B2"
GREEN_700 = "#386910"
GREEN_600 = "#568C1C"
GREEN_500 = "#76AD2A"
RED_700 = "#8A2424"
RED_600 = "#B53333"
RED_500 = "#E04343"
GRAY_550 = "#73726C"
GRAY_500 = "#87867F"
YELLOW_600 = "#C77F1A"

# Font settings matching 09_circuit_ablation_replot.py
FONT_SCALE = 2.4
BASE_LABEL_SIZE = 10
BASE_TICK_SIZE = 9
BASE_LEGEND_SIZE = 8
BASE_MARKER_SIZE = 7

BASE_DIR = Path("analysis/09b_causal_pathway/262k_big")

# Concept colors for carrier/suppressor per-concept plots
CONCEPT_COLORS = [
    "#1B67B2",  # Blue 600
    "#386910",  # Green 700
    "#B53333",  # Red 600
    "#6258D1",  # Violet 500
    "#D97757",  # Clay
    "#568C1C",  # Green 600
    "#C77F1A",  # Yellow 600
    "#0E6B54",  # Aqua 700
    "#8A2D4C",  # Magenta 700
    "#E86235",  # Orange 500
]
CONCEPT_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h", "*", "p"]

# Series definitions for ablation plots
SERIES_DEFS = [
    ("baseline_activations", "Baseline", BLUE_600, "-", "o"),
    ("weak_control_activations", "Bottom-10% attributed", YELLOW_600, (0, (3, 1, 1, 1)), "d"),
    ("suppressors_ablated_activations", "All evidence carriers", GREEN_700, "-", "^"),
    ("suppressors_top20pct_activations", "Top-20% evidence carriers", GREEN_600, ":", "^"),
    ("suppressors_top5pct_activations", "Top-5% evidence carriers", GREEN_500, "--", "^"),
    ("supporters_ablated_activations", "All suppressors", RED_700, "-", "v"),
    ("supporters_top20pct_activations", "Top-20% suppressors", RED_600, ":", "v"),
    ("supporters_top5pct_activations", "Top-5% suppressors", RED_500, "--", "v"),
]


def discover_concept_dirs(gate_str: str) -> Dict[str, Path]:
    """Find all concept directories for a given gate feature string like 'L45_F9959'."""
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_(\w+)_" + re.escape(gate_str) + r"_L\d+$")
    result = {}
    for d in sorted(BASE_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            concept = m.group(1)
            result[concept] = d
    return result


def load_ablation_data(concept_dir: Path) -> Optional[Dict]:
    p = concept_dir / "ablation_sweep.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_gradient_data(concept_dir: Path) -> Optional[Dict]:
    p = concept_dir / "gradient_attribution.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def get_feature_title(gate_str: str) -> str:
    try:
        import importlib; _m = importlib.import_module("09b_causal_pathway"); get_feature_title_from_gemma_scope = _m.get_feature_title_from_gemma_scope
        parts = gate_str.split("_F")
        layer = int(parts[0][1:])
        feat_id = int(parts[1])
        title = get_feature_title_from_gemma_scope(layer, feat_id)
        return title if title else "(no title)"
    except Exception:
        return "(no title)"


def get_concept_detection_rate(concept: str, steering_layer: int = 37) -> Optional[float]:
    try:
        import importlib; _m = importlib.import_module("09b_causal_pathway"); _get_det_rate = _m.get_concept_detection_rate
        return _get_det_rate(concept, steering_layer)
    except Exception:
        return None


def select_top6_concepts(concept_dirs: Dict[str, Path]) -> List[str]:
    """Select the 6 concepts with lowest max(n_carriers, n_suppressors)."""
    concept_max_n = {}
    for concept, d in concept_dirs.items():
        data = load_ablation_data(d)
        if data is None:
            continue
        n_carriers = len(data.get("features_suppressors", []))
        n_suppressors = len(data.get("features_supporters", []))
        concept_max_n[concept] = max(n_carriers, n_suppressors)
    return [c for c, _ in sorted(concept_max_n.items(), key=lambda x: x[1])[:6]]


def _get_n_for_series(data: Dict, data_key: str) -> Optional[int]:
    """Get the n count for a particular series from ablation data."""
    if "suppressor" in data_key and "top5pct" in data_key:
        return data.get("n_suppressors_top5pct")
    elif "suppressor" in data_key and "top20pct" in data_key:
        return data.get("n_suppressors_top20pct")
    elif "supporter" in data_key and "top5pct" in data_key:
        return data.get("n_supporters_top5pct")
    elif "supporter" in data_key and "top20pct" in data_key:
        return data.get("n_supporters_top20pct")
    elif data_key == "suppressors_ablated_activations":
        return len(data.get("features_suppressors", []))
    elif data_key == "supporters_ablated_activations":
        return len(data.get("features_supporters", []))
    elif data_key == "weak_control_activations":
        return len(data.get("features_weak_control", []))
    return None


GATE_Y_MAX = {
    "L45_F74631": 4750,
}
DEFAULT_Y_MAX = 6750


def plot_shared_ablation(
    gate_str: str,
    concept_dirs: Dict[str, Path],
    concepts: List[str],
    output_dir: Path,
):
    """Plot ablation curves as 2x3 grid, one subplot per concept."""
    import matplotlib.pyplot as plt
    import textwrap

    all_data = {}
    for concept in concepts:
        if concept not in concept_dirs:
            continue
        data = load_ablation_data(concept_dirs[concept])
        if data is not None:
            all_data[concept] = data

    if not all_data:
        print(f"  No ablation data found for {gate_str}")
        return

    feature_title = get_feature_title(gate_str)
    gate_name = gate_str.replace("_F", " F")

    # Collect n ranges across selected concepts for legend labels
    n_ranges = {}
    for data_key, label, _, _, _ in SERIES_DEFS:
        ns = []
        for data in all_data.values():
            n = _get_n_for_series(data, data_key)
            if n is not None:
                ns.append(n)
        if ns:
            n_ranges[label] = (min(ns), max(ns))

    TARGET_STRENGTHS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
    XTICKS = [-8, -4, 0, 4, 8]
    Y_MAX = GATE_Y_MAX.get(gate_str, DEFAULT_Y_MAX)

    # Figsize: width reduced by 1.35x from 12.5 → ~9.26
    fig, axes = plt.subplots(2, 3, figsize=(12.5 / 1.35, 6), sharey=True)
    axes_flat = axes.flatten()

    for ci, concept in enumerate(concepts):
        if concept not in all_data:
            continue
        ax = axes_flat[ci]
        data = all_data[concept]

        det_rate = get_concept_detection_rate(concept)

        for data_key, label, color, ls, marker in SERIES_DEFS:
            series = data.get(data_key, {})
            if not series:
                continue

            xs = []
            ys = []
            for s in TARGET_STRENGTHS:
                val = series.get(str(float(s)))
                if val is not None:
                    xs.append(s)
                    ys.append(val)

            if not xs:
                continue

            # Build label with n (only for first subplot to create shared legend)
            if ci == 0:
                n_info = n_ranges.get(label)
                if n_info and n_info[0] > 0:
                    if n_info[0] == n_info[1]:
                        full_label = f"{label} (n={n_info[0]})"
                    else:
                        full_label = f"{label} (n={n_info[0]}\u2013{n_info[1]})"
                else:
                    full_label = label
            else:
                full_label = None

            is_main = ls == "-" or data_key == "weak_control_activations"
            ms = BASE_MARKER_SIZE * FONT_SCALE * (0.5 if not is_main else 0.6)

            ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, linewidth=3.2,
                    markersize=ms, label=full_label, alpha=0.9)

        ax.axvline(x=0, color=GRAY_500, linestyle='--', alpha=0.5)
        ax.set_xticks(XTICKS)
        ax.set_ylim(top=Y_MAX)
        ax.tick_params(axis='both', labelsize=BASE_TICK_SIZE * FONT_SCALE)
        ax.grid(True, alpha=0.3)

        # Concept name inside plot, top-left (bold name + normal-weight detection rate)
        concept_size = BASE_LABEL_SIZE * FONT_SCALE * 0.7
        det_size = concept_size * 0.75
        ax.text(0.03, 0.97, concept,
                transform=ax.transAxes, va='top', ha='left',
                fontsize=concept_size, fontweight='bold',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))
        if det_rate is not None:
            ax.text(0.03, 0.83, f"({det_rate:.0f}% detection)",
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=det_size * 1.25,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))

        if ci % 3 == 0:
            ax.set_ylabel("Activation", fontsize=BASE_LABEL_SIZE * FONT_SCALE, labelpad=2)

    # Wrap long feature titles
    wrapped_title = "\n".join(textwrap.wrap(feature_title, width=80))

    # Single centered x-axis label tight to x-ticks
    fig.text(0.5, 0.035, "Steering strength",
             ha='center', va='top', fontsize=BASE_LABEL_SIZE * FONT_SCALE)

    # Suptitle: bold gate name, italic feature label closer to title
    main_size = BASE_LABEL_SIZE * FONT_SCALE * 0.95
    fig.suptitle(f"Gate: {gate_name}", fontsize=main_size, fontweight='bold', y=1.05)
    fig.text(0.5, 0.995, wrapped_title, ha='center', va='top',
             fontsize=main_size * 0.8, fontstyle='italic')

    # Shared legend — 4 rows x 2 cols, first row = Baseline + Bottom-10%
    handles, labels = axes_flat[0].get_legend_handles_labels()
    h_map = dict(zip(labels, handles))
    # Row order for ncol=2: items go left-right then top-bottom
    row1 = ["Baseline"] + [l for l in labels if "Bottom-10%" in l]
    row2 = [l for l in labels if "All evidence" in l] + [l for l in labels if "All suppressors" in l]
    row3 = [l for l in labels if "Top-20% evidence" in l] + [l for l in labels if "Top-20% suppressors" in l]
    row4 = [l for l in labels if "Top-5% evidence" in l] + [l for l in labels if "Top-5% suppressors" in l]
    ordered_labels = row1 + row2 + row3 + row4
    ordered_handles = [h_map[l] for l in ordered_labels]

    legend_fontsize = BASE_LEGEND_SIZE * FONT_SCALE * 0.55 * 1.5  # match exemplar
    leg = fig.legend(ordered_handles, ordered_labels,
               fontsize=legend_fontsize,
               loc='lower left', bbox_to_anchor=(-0.075, -0.28), ncol=2,
               framealpha=0.95, columnspacing=0.8, handletextpad=0.3)
    # Thin legend lines so dash patterns are visible
    for line in leg.get_lines():
        line.set_linewidth(1.8)

    plt.subplots_adjust(top=0.93, bottom=0.10, left=0.07, right=0.98,
                        hspace=0.25, wspace=0.05)

    fname = f"ablation_{gate_str}.png"
    plt.savefig(output_dir / fname, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def _get_carrier_feature_title(layer: int, feat_id: int) -> str:
    """Get gemma-scope feature title for a transcoder feature."""
    try:
        import importlib; _m = importlib.import_module("09b_causal_pathway"); get_feature_title_from_gemma_scope = _m.get_feature_title_from_gemma_scope
        title = get_feature_title_from_gemma_scope(layer, feat_id)
        return title if title else "(no title)"
    except Exception:
        return "(no title)"


# Features to exclude from top-N selection (still show 5 per concept, just skip these)
FILTERED_FEATURES = {(42, 37785), (40, 8169), (41, 13501),
                     (44, 801), (43, 1663), (38, 62433), (41, 37047), (41, 61743), (41, 71186),
                     (43, 39659), (40, 134604), (40, 7992), (41, 413),
                     (38, 16724)}

# 30 distinct colors for carrier/suppressor features
FEATURE_PALETTE = [
    "#1B67B2",  # Blue 600
    "#B53333",  # Red 600
    "#386910",  # Green 700
    "#6258D1",  # Violet 500
    "#D97757",  # Clay
    "#C77F1A",  # Yellow 600
    "#0E6B54",  # Aqua 700
    "#8A2D4C",  # Magenta 700
    "#4D44AB",  # Violet 600
    "#76AD2A",  # Green 500
    "#E86235",  # Orange 500
    "#2C84DB",  # Blue 500
    "#E04343",  # Red 500
    "#24B283",  # Aqua 500
    "#E05A87",  # Magenta 500
    "#965B0E",  # Yellow 700
    "#568C1C",  # Green 600
    "#0F4B87",  # Blue 700
    "#8A2424",  # Red 700
    "#188F6B",  # Aqua 600
    "#BA4C27",  # Orange 600
    "#383182",  # Violet 700
    "#B54369",  # Magenta 600
    "#FAA72A",  # Yellow 500
    "#599EE3",  # Blue 400
    "#E86B6B",  # Red 400
    "#4DC49C",  # Aqua 400
    "#90BF4E",  # Green 400
    "#ED8461",  # Orange 400
    "#827ADE",  # Violet 400
]
FEATURE_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h", "*", "p",
                   "o", "s", "^", "v", "D", "P", "X", "h", "*", "p",
                   "o", "s", "^", "v", "D", "P", "X", "h", "*", "p"]


def plot_shared_carriers_or_suppressors(
    gate_str: str,
    concept_dirs: Dict[str, Path],
    concepts: List[str],
    output_dir: Path,
    category: str = "carrier",
    top_n: int = 5,
):
    """Plot top carriers/suppressors as 2x3 grid, one subplot per concept."""
    import matplotlib.pyplot as plt
    import textwrap
    import sys

    EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
    if str(EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_DIR))

    _fca = importlib.import_module("08_feature_centric_analysis")
    discover_cached_strengths = _fca.discover_cached_strengths
    get_control_cache_file = _fca.get_control_cache_file
    get_layer_matrix_cached = _fca.get_layer_matrix_cached

    import importlib; _circuit = importlib.import_module("09_circuit_analysis")
    TOKEN_MODE = "last_token"
    TARGET_STRENGTHS = [-8, -4, -2, 0, 2, 4, 8]
    XTICKS = [-8, -4, 0, 4, 8]

    strengths_map = discover_cached_strengths(
        37, TOKEN_MODE, _circuit.TRANSCODER_L0,
        transcoder_width=_circuit.TRANSCODER_WIDTH,
    )

    gate_name = gate_str.replace("_F", " F")
    category_label = "evidence carriers" if category == "carrier" else "suppressors"

    # --- Pass 1: collect all unique features across all concepts ---
    concept_features = {}  # concept -> list of (layer, feat_id, attribution)
    for concept in concepts:
        if concept not in concept_dirs:
            continue
        grad_data = load_gradient_data(concept_dirs[concept])
        if grad_data is None:
            continue
        all_attrs = grad_data.get("all_attributions", [])
        if not all_attrs:
            continue
        if category == "carrier":
            selected = sorted([a for a in all_attrs if a["attribution"] < 0],
                              key=lambda x: x["attribution"])
        else:
            selected = sorted([a for a in all_attrs if a["attribution"] > 0],
                              key=lambda x: -x["attribution"])
        filtered = [(a["layer"], a["feat_id"]) for a in selected
                    if (a["layer"], a["feat_id"]) not in FILTERED_FEATURES]
        concept_features[concept] = filtered[:top_n]

    # Build unique feature list preserving first-seen order
    all_unique_features = []
    seen = set()
    for concept in concepts:
        for lf in concept_features.get(concept, []):
            if lf not in seen:
                all_unique_features.append(lf)
                seen.add(lf)

    # Assign color/marker per unique feature
    feat_color = {}
    feat_marker = {}
    for i, lf in enumerate(all_unique_features):
        feat_color[lf] = FEATURE_PALETTE[i % len(FEATURE_PALETTE)]
        feat_marker[lf] = FEATURE_MARKERS[i % len(FEATURE_MARKERS)]

    # Get gemma-scope titles for all unique features
    print(f"  Loading feature titles for {len(all_unique_features)} unique features...")
    feat_titles = {}
    for layer, feat_id in all_unique_features:
        feat_titles[(layer, feat_id)] = _get_carrier_feature_title(layer, feat_id)

    # --- Pass 2: plot ---
    # Wider figure to accommodate legend column on the right
    fig, axes = plt.subplots(2, 3, figsize=(12.5 / 1.35 + 4, 6))
    axes_flat = axes.flatten()

    cache_store = {}

    for ci, concept in enumerate(concepts):
        if concept not in concept_features:
            continue
        ax = axes_flat[ci]
        features = concept_features[concept]

        for layer, feat_id in features:
            concept_values = {}
            for strength in TARGET_STRENGTHS:
                cache_file = strengths_map.get(float(strength))
                if cache_file is None and strength == 0:
                    cache_file = get_control_cache_file(
                        TOKEN_MODE, _circuit.TRANSCODER_L0,
                        transcoder_width=_circuit.TRANSCODER_WIDTH,
                    )
                if cache_file is None:
                    continue
                try:
                    concepts_list, matrix, concept_to_idx = get_layer_matrix_cached(
                        cache_file, layer,
                        cached_data=None, cache_store=cache_store,
                        use_fast_layer_matrices=True, show_progress=False,
                    )
                    if concept in concept_to_idx:
                        concept_values[strength] = float(matrix[concept_to_idx[concept], feat_id])
                except Exception:
                    continue

            xs = [s for s in TARGET_STRENGTHS if s in concept_values]
            ys = [concept_values[s] for s in xs]
            if not xs:
                continue

            lf = (layer, feat_id)
            ax.plot(xs, ys, color=feat_color[lf], marker=feat_marker[lf], linewidth=3.2,
                    markersize=BASE_MARKER_SIZE * FONT_SCALE * 0.5, alpha=0.85)

        det_rate = get_concept_detection_rate(concept)

        ax.axvline(x=0, color=GRAY_500, linestyle='--', alpha=0.5)
        ax.set_xticks(XTICKS)
        ax.tick_params(axis='both', labelsize=BASE_TICK_SIZE * FONT_SCALE)
        ax.grid(True, alpha=0.3)

        # Hide x-tick labels on first row
        if ci < 3:
            ax.set_xticklabels([])

        # Concept name inside plot, top-left
        concept_size = BASE_LABEL_SIZE * FONT_SCALE * 0.7
        det_size = concept_size * 0.75
        ax.text(0.03, 0.97, concept,
                transform=ax.transAxes, va='top', ha='left',
                fontsize=concept_size, fontweight='bold',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))
        if det_rate is not None:
            ax.text(0.03, 0.83, f"({det_rate:.0f}% detection)",
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=det_size * 1.25,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))

        if ci % 3 == 0:
            ax.set_ylabel("Activation", fontsize=BASE_LABEL_SIZE * FONT_SCALE, labelpad=2)
        else:
            ax.set_yticklabels([])

    # Single centered x-axis label (centered on subplot area, not full fig)
    subplot_center_x = (0.07 + 0.62) / 2
    fig.text(subplot_center_x, 0.035, "Steering strength",
             ha='center', va='top', fontsize=BASE_LABEL_SIZE * FONT_SCALE)

    # Title (unbolded)
    main_size = BASE_LABEL_SIZE * FONT_SCALE * 0.95
    fig.suptitle(f"Top-{top_n} {category_label} for gate: {gate_name}",
                 fontsize=main_size, y=1.02, x=subplot_center_x)

    # --- Legend as column on the right ---
    # Order features by visual presentation: concept-by-concept, no duplicates
    import matplotlib.lines as mlines
    ordered_features = []
    seen_feats = set()
    for concept in concepts:
        for lf in concept_features.get(concept, []):
            if lf not in seen_feats:
                ordered_features.append(lf)
                seen_feats.add(lf)

    # Build feature -> concepts mapping
    feat_to_concepts = {}
    for concept in concepts:
        for lf in concept_features.get(concept, []):
            feat_to_concepts.setdefault(lf, []).append(concept)

    legend_handles = []
    legend_labels = []
    legend_fs = BASE_LEGEND_SIZE * FONT_SCALE * 0.55 * 1.75 / 1.15 * 1.1 / 1.15
    import textwrap as tw
    for layer, feat_id in ordered_features:
        lf = (layer, feat_id)
        handle = mlines.Line2D([], [], color=feat_color[lf], marker=feat_marker[lf],
                               linewidth=3.2, markersize=BASE_MARKER_SIZE * FONT_SCALE * 0.4)
        legend_handles.append(handle)
        title = feat_titles.get(lf, "(no title)").replace("/", " or ")
        for suffix in [', "note', ', "no', ', "']:
            if title.endswith(suffix):
                title = title[:-len(suffix)]
        if title.count('"') % 2 == 1:
            title = title + '"'
        feat_str = f"L{layer} F{feat_id}"
        concept_list = ", ".join(feat_to_concepts.get(lf, []))
        full_text = f"{feat_str} ({concept_list}): {title}"
        wrapped = tw.fill(full_text, width=38)
        legend_labels.append(wrapped)

    leg = fig.legend(legend_handles, legend_labels,
                     fontsize=legend_fs,
                     loc='center left', bbox_to_anchor=(0.63, 0.5),
                     ncol=2, framealpha=0.95,
                     handletextpad=0.4, labelspacing=0.8, borderpad=0.5,
                     columnspacing=1.0)
    # Bold feature IDs, everything else italic
    for text in leg.get_texts():
        full = text.get_text()
        paren_idx = full.find(' (')
        if paren_idx > 0:
            feat_part = full[:paren_idx]
            rest = full[paren_idx:]
            bold_feat = r"$\mathbf{" + feat_part.replace(" ", r"\ ") + r"}$"
            text.set_text(bold_feat + rest)
        text.set_fontstyle('italic')
        text.set_linespacing(1.3)

    plt.subplots_adjust(top=0.95, bottom=0.10, left=0.07, right=0.62,
                        hspace=0.075, wspace=0.05)

    fname = f"top_{category}s_{gate_str}.png"
    plt.savefig(output_dir / fname, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def load_steering_alignment_data(concept_dir: Path) -> Optional[Dict]:
    p = concept_dir / "steering_alignment.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_circuit_importance_data(concept_dir: Path) -> Optional[Dict]:
    p = concept_dir / "circuit_importance_validation.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _auto_scale(arr: np.ndarray) -> Tuple[float, str]:
    """Return (divisor, label_suffix) to keep tick values in readable range."""
    amax = max(abs(arr.max()), abs(arr.min())) if len(arr) > 0 else 1
    if amax == 0:
        return 1.0, ""
    exp = int(np.floor(np.log10(amax)))
    if exp >= 4:
        divisor = 10 ** exp
        return divisor, f" ($\\times 10^{{{exp}}}$)"
    elif exp <= -3:
        divisor = 10 ** exp
        return divisor, f" ($\\times 10^{{{exp}}}$)"
    return 1.0, ""


def _compute_95ci(values: List[float]) -> Tuple[float, float]:
    """Compute mean and 95% CI half-width (t-distribution) from values."""
    from scipy import stats as scipy_stats
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]  # drop NaN
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0
    sem = float(scipy_stats.sem(arr))
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    return mean, t_crit * sem


def plot_shared_circuit_analysis(
    gate_str: str,
    concept_dirs: Dict[str, Path],
    concepts: List[str],
    output_dir: Path,
):
    """
    3-panel compact figure aggregating steering alignment and circuit importance
    validation across concepts.

    (a) Mean steering projection (encoder · steering vec) by feature group
    (b) Correlation strength: gate_attribution vs steering_alignment vs circuit_importance
    (c) Circuit importance vs Δ gate activation scatter (pooled across concepts)
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    # Load data for all concepts
    steer_data: Dict[str, Dict] = {}
    circ_data: Dict[str, Dict] = {}
    for concept in concepts:
        if concept not in concept_dirs:
            continue
        sd = load_steering_alignment_data(concept_dirs[concept])
        if sd is not None:
            steer_data[concept] = sd
        cd = load_circuit_importance_data(concept_dirs[concept])
        if cd is not None:
            circ_data[concept] = cd

    if not steer_data and not circ_data:
        print(f"  No steering/circuit data for {gate_str}")
        return

    gate_name = gate_str.replace("_F", " F")

    # Category colors for panel (c) scatter
    C_CARRIER = GREEN_700
    C_SUPPRESS = RED_600
    C_OTHER = GRAY_550
    cat_colors = {
        "evidence_carrier": C_CARRIER,
        "suppressor": C_SUPPRESS,
        "other_active": C_OTHER,
    }

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12.5, 4.1))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.6, 1.3, 1.8],
                           wspace=0.36,
                           )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # Manually shift (c) to give y-label room without affecting (a)-(b) gap
    pos_c = axes[2].get_position()
    axes[2].set_position([pos_c.x0 + 0.000005, pos_c.y0, pos_c.width, pos_c.height])

    fs_label = BASE_LABEL_SIZE * FONT_SCALE * 0.85
    fs_tick = BASE_TICK_SIZE * FONT_SCALE * 0.85
    fs_panel = BASE_LABEL_SIZE * FONT_SCALE

    # ── Panel (a): Steering alignment by group (box plot) ───────────────────
    ax = axes[0]
    groups = [
        ("top_50_carriers_raw", "Evidence\ncarriers", GREEN_700),
        ("top_50_suppressors_raw", "Suppressors", RED_600),
        ("rest_active_raw", "Other\nactive", GRAY_550),
        ("random_baseline_raw", "Random", BLUE_600),
    ]

    # Collect all concept means per group
    group_concept_means: List[List[float]] = []
    for _gi, (stat_key, _label, _color) in enumerate(groups):
        concept_means = []
        for concept in concepts:
            if concept not in steer_data:
                continue
            stats = steer_data[concept].get("stats", {}).get(stat_key, {})
            if "mean" in stats:
                concept_means.append(stats["mean"])
        group_concept_means.append(concept_means)

    all_bar_vals = [v for cm in group_concept_means for v in cm]
    bar_div, bar_suffix = _auto_scale(np.array(all_bar_vals)) if all_bar_vals else (1.0, "")

    box_data = []
    box_colors = []
    box_labels = []
    for gi, (stat_key, label, color) in enumerate(groups):
        concept_means = group_concept_means[gi]
        if not concept_means:
            box_data.append([0])
        else:
            box_data.append([v / bar_div for v in concept_means])
        box_colors.append(color)
        box_labels.append(label)

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_ylabel("Steering projection" + bar_suffix, fontsize=fs_label, labelpad=2, y=0.42)
    ax.tick_params(labelsize=fs_tick)
    ax.set_xticklabels(box_labels, fontsize=fs_tick/1.35, rotation=40, ha='center')
    ax.tick_params(axis='x', pad=-2)
    for lbl in ax.get_xticklabels():
        lbl.set_multialignment('center')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    # Panel label removed — added externally

    # ── Panel (b): Correlation strength comparison (sorted descending) ──────
    ax = axes[1]
    predictors = [
        ("gate_attribution", "Gate\nattribution", BLUE_600),
        ("steering_alignment", "Steering\nalignment", YELLOW_600),
        ("circuit_importance", "Circuit\nimportance", GREEN_700),
    ]

    # Compute means first so we can sort
    pred_bars = []
    for pred_key, label, color in predictors:
        concept_spearman = []
        for concept in concepts:
            if concept not in circ_data:
                continue
            corr = circ_data[concept].get("correlations", {}).get(pred_key, {})
            if "spearman_r" in corr:
                concept_spearman.append(abs(corr["spearman_r"]))
        if not concept_spearman:
            continue
        mean_val, ci95 = _compute_95ci(concept_spearman)
        pred_bars.append((mean_val, ci95, label, color))

    # Sort descending by mean
    pred_bars.sort(key=lambda x: x[0], reverse=True)

    for pi, (mean_val, ci95, label, color) in enumerate(pred_bars):
        ax.bar(pi, mean_val, color=color, alpha=0.7, edgecolor='black',
               linewidth=0.5, width=0.6)
        ax.errorbar(pi, mean_val, yerr=ci95, fmt='none', ecolor='black',
                    capsize=5, linewidth=2, zorder=4)

    ax.set_ylabel(r"Spearman |$\rho$| with $\Delta$ gate", fontsize=fs_label, labelpad=2, y=0.42)
    ax.tick_params(labelsize=fs_tick)
    ax.set_xticks(range(len(pred_bars)))
    ax.set_xticklabels([p[2] for p in pred_bars], fontsize=fs_tick/1.35, rotation=40, ha='center')
    ax.tick_params(axis='x', pad=-8)
    for lbl in ax.get_xticklabels():
        lbl.set_multialignment('center')
    ax.set_ylim(0, None)
    # Panel label removed — added externally

    # ── Panel (c): Circuit importance vs ablation impact scatter ─────────────
    ax = axes[2]

    all_x: List[float] = []
    all_y: List[float] = []
    all_colors: List[str] = []
    for concept in concepts:
        if concept not in circ_data:
            continue
        per_feat = circ_data[concept].get("per_feature_results", [])
        if not per_feat:
            continue
        for r in per_feat:
            xi, yi = r["circuit_importance"], r["ablation_impact"]
            if xi is None or yi is None or np.isnan(xi) or np.isnan(yi):
                continue
            all_x.append(xi)
            all_y.append(yi)
            all_colors.append(cat_colors.get(r.get("category", "other_active"), C_OTHER))

    # Auto-scale axes
    all_x_arr = np.array(all_x) if all_x else np.array([0.0])
    all_y_arr = np.array(all_y) if all_y else np.array([0.0])
    x_div, x_suffix = _auto_scale(all_x_arr)
    y_div, y_suffix = _auto_scale(all_y_arr)

    all_x_plot = all_x_arr / x_div
    all_y_plot = all_y_arr / y_div

    ax.scatter(all_x_plot, all_y_plot, c=all_colors, s=20, alpha=0.3, edgecolors='none')

    # Regression line
    if len(all_x) > 2:
        slope, intercept = np.polyfit(all_x_plot, all_y_plot, 1)
        x_line = np.linspace(all_x_plot.min(), all_x_plot.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.6, linewidth=1.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)

    # Clip x/y-range to central 98th percentile for a centered view
    x_q1, x_q99 = np.percentile(all_x_plot, 1), np.percentile(all_x_plot, 99)
    x_abs_max = max(abs(x_q1), abs(x_q99))
    ax.set_xlim(-x_abs_max * 1.1, x_abs_max * 1.1)

    y_q1, y_q99 = np.percentile(all_y_plot, 0.5), np.percentile(all_y_plot, 99.5)
    y_abs_max = max(abs(y_q1), abs(y_q99))
    ax.set_ylim(-y_abs_max * 1.2, y_abs_max * 1.2)

    main_fs = fs_label * 0.85 / 1.1
    sub_fs = main_fs * 0.70
    line1_fs = main_fs * 0.88
    line2_fs = line1_fs / 1.1
    ax.set_xlabel("")
    ax.annotate("Circuit importance" + x_suffix,
                xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -40), textcoords='offset points',
                ha='center', fontsize=line1_fs * 1.15, annotation_clip=False)
    ax.annotate("= gate attribution × steering projection",
                xy=(0.5, 0), xycoords='axes fraction',
                xytext=(-18, -56), textcoords='offset points',
                ha='center', fontsize=line2_fs, annotation_clip=False)
    ax.set_ylabel(r"$\Delta$ gate activation" + y_suffix, fontsize=fs_label, labelpad=1)
    ax.tick_params(labelsize=fs_tick)
    ax.tick_params(axis='y', pad=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Single unified legend: category colors + rho + definition
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=C_CARRIER, label='Evidence carrier'),
    ]
    if len(all_x) > 2:
        sp_r, _sp_p = scipy_stats.spearmanr(all_x_arr, all_y_arr)
        n_label = f"{len(all_x)//1000}k" if len(all_x) >= 1000 else str(len(all_x))
        legend_elements.append(
            Line2D([0], [0], linestyle='--', color='black', alpha=0.6, linewidth=1.5,
                   label=f"$\\rho$ = {sp_r:.2f} (n={n_label})"),
        )
    legend_elements += [
        Patch(facecolor=C_SUPPRESS, label='Suppressor'),
        Patch(facecolor=C_OTHER, label='Other active'),
    ]
    # Reorder so row layout with ncol=2 gives:
    # row1: Evidence carrier | ρ line
    # row2: Suppressor       | Other active
    # matplotlib ncol=2 fills row-first, so order is already correct
    leg = ax.legend(handles=legend_elements, fontsize=fs_tick * 0.65 / 1.02,
                    loc='upper right', bbox_to_anchor=(1.0, 1.0),
                    framealpha=0.85, handlelength=0.8, handletextpad=0.3,
                    borderaxespad=0.1, ncol=2, columnspacing=0.5,
                    borderpad=0.3)

    # Panel label removed — added externally

    # Title removed — added externally

    fname = f"circuit_analysis_{gate_str}.png"
    # Save with tight bbox + 0.03 uniform padding first
    fig.savefig(output_dir / fname, dpi=400, bbox_inches='tight', pad_inches=0.02)
    # Add extra right padding (0.08 - 0.03 = 0.05 inches extra on right)
    from PIL import Image
    img = Image.open(output_dir / fname)
    extra_right_px = int(0.05 * 400)  # 0.05 extra inches at 400 dpi
    new_img = Image.new(img.mode, (img.width + extra_right_px, img.height), (255, 255, 255))
    new_img.paste(img, (0, 0))
    new_img.save(output_dir / fname)
    plt.close()
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Generate shared multi-concept causal pathway plots")
    parser.add_argument("--gate-features", type=str, nargs="+",
                        default=["L45_F9959", "L50_F167", "L45_F74631"],
                        help="Gate features to plot (default: L45_F9959 L50_F167 L45_F74631)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top carrier/suppressor features per concept (default: 5)")
    parser.add_argument("--skip-carriers", action="store_true",
                        help="Skip carrier/suppressor plots (only ablation)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation plots")
    parser.add_argument("--only-carriers", action="store_true",
                        help="Only run carrier plots (skip suppressors and ablation)")
    parser.add_argument("--only-suppressors", action="store_true",
                        help="Only run suppressor plots (skip carriers and ablation)")
    parser.add_argument("--only-circuit", action="store_true",
                        help="Only run circuit analysis plots")
    parser.add_argument("--n-concepts", type=int, default=6,
                        help="Number of concepts to show (default: 6)")
    args = parser.parse_args()

    output_dir = BASE_DIR / "shared"
    output_dir.mkdir(parents=True, exist_ok=True)

    for gate_str in args.gate_features:
        print(f"\n{'='*60}")
        print(f"Gate: {gate_str}")
        print(f"{'='*60}")

        concept_dirs = discover_concept_dirs(gate_str)
        if not concept_dirs:
            print(f"  No concept directories found for {gate_str}")
            continue

        concepts = select_top6_concepts(concept_dirs)[:args.n_concepts]
        print(f"  Selected {len(concepts)} concepts (lowest max n): {', '.join(concepts)}")

        any_only = args.only_carriers or args.only_suppressors or args.only_circuit
        run_ablation = not args.skip_ablation and not any_only
        run_carriers = (not args.skip_carriers and not args.only_suppressors and not args.only_circuit) or args.only_carriers
        run_suppressors = (not args.skip_carriers and not args.only_carriers and not args.only_circuit) or args.only_suppressors
        run_circuit = not any_only or args.only_circuit

        if run_ablation:
            plot_shared_ablation(gate_str, concept_dirs, concepts, output_dir)
        if run_carriers:
            plot_shared_carriers_or_suppressors(
                gate_str, concept_dirs, concepts, output_dir,
                category="carrier", top_n=args.top_n,
            )
        if run_suppressors:
            plot_shared_carriers_or_suppressors(
                gate_str, concept_dirs, concepts, output_dir,
                category="suppressor", top_n=args.top_n,
            )
        if run_circuit:
            # Use up to 50 concepts for tighter CIs
            all_concepts = sorted(concept_dirs.keys())[:50]
            plot_shared_circuit_analysis(gate_str, concept_dirs, all_concepts, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
