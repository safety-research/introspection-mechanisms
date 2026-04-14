"""
Generate figures for ICLR 2026 paper Section 4:
"DETECTION IS NOT REDUCIBLE TO A SINGLE LINEAR DIRECTION"

Figure 1: First 5 PCs from subspace_deltas + PLS2 (2x3 grid)
Figure 2: PLS components (2x2 grid)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib with anthroplot style
import plot_style
plot_style.set_defaults(matplotlib=True, plotly=False, pretty=False, install_brand_fonts=False)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams.get("font.sans-serif", [])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 400,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'text.usetex': False,
})

ANALYSIS_ROOT = Path("analysis/04c_bidirectional_steering")

ALPHAS = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

COLOR_NEG = plot_style.MAGENTA_500
COLOR_POS = plot_style.BLUE_500
COLOR_LINE = '#404040'

# ============================================================================
# CONFIGURATION DEFINITIONS
# ============================================================================

CONFIGS = {
    "gemma3_27b__layer_37_strength_4.0": {
        "base": ANALYSIS_ROOT / "gemma3_27b/layer_37_strength_4.0",
        "title": "Gemma3-27B, Layer 37, Strength 4.0",
        "strength_scale": 4.0,
        "has_pls": True,
        "panels": [
            {"source": "deltas", "index": 0, "name": "δPC1", "var": 19.6,
             "pos_label": "Formal", "neg_label": "Casual"},
            {"source": "deltas", "index": 1, "name": "δPC2", "var": 12.0,
             "pos_label": "Philosophy", "neg_label": "Sensory"},
            {"source": "deltas", "index": 2, "name": "δPC3", "var": 8.4,
             "pos_label": "Careers", "neg_label": "Emotions"},
            {"source": "deltas", "index": 3, "name": "δPC4", "var": 7.5,
             "pos_label": "Sports", "neg_label": "Scientists"},
            {"source": "deltas", "index": 4, "name": "δPC5", "var": 4.8,
             "pos_label": "Inanimate", "neg_label": "People"},
            {"source": "pls", "index": 1, "name": "PLS2", "var": 3.8,
             "pos_label": "Entities", "neg_label": "History"},
        ],
    },
    "gemma3_27b__layer_29_strength_4.0": {
        "base": ANALYSIS_ROOT / "gemma3_27b/layer_29_strength_4.0",
        "title": "Gemma3-27B, Layer 29, Strength 4.0",
        "strength_scale": 4.0,
        "has_pls": False,
        "panels": [
            {"source": "deltas", "index": 0, "name": "δPC1", "var": 29.3,
             "pos_label": "Nicknames", "neg_label": "Noise"},
            {"source": "deltas", "index": 1, "name": "δPC2", "var": 12.8,
             "pos_label": "Philosophy", "neg_label": "Noise"},
            {"source": "deltas", "index": 2, "name": "δPC3", "var": 9.4,
             "pos_label": "Quirky", "neg_label": "Noise"},
            {"source": "deltas", "index": 3, "name": "δPC4", "var": 5.2,
             "pos_label": "Equipment", "neg_label": "Scientists"},
            {"source": "deltas", "index": 4, "name": "δPC5", "var": 4.8,
             "pos_label": "Species", "neg_label": "Sports"},
        ],
    },
    "gemma3_27b_abliterated__layer_37_strength_2.0": {
        "base": ANALYSIS_ROOT / "gemma3_27b_abliterated/layer_37_strength_2.0",
        "title": "Gemma3-27B Abliterated, Layer 37, Strength 2.0",
        "strength_scale": 2.0,
        "has_pls": True,
        "panels": [
            {"source": "deltas", "index": 0, "name": "δPC1", "var": 20.7,
             "pos_label": "Formal", "neg_label": "Casual"},
            {"source": "deltas", "index": 1, "name": "δPC2", "var": 12.7,
             "pos_label": "Abstract", "neg_label": "Concrete"},
            {"source": "deltas", "index": 2, "name": "δPC3", "var": 8.7,
             "pos_label": "Sports", "neg_label": "Nature"},
            {"source": "deltas", "index": 3, "name": "δPC4", "var": 6.5,
             "pos_label": "Athletics", "neg_label": "Scientists"},
            {"source": "deltas", "index": 4, "name": "δPC5", "var": 4.7,
             "pos_label": "Delicious", "neg_label": "Meta"},
            {"source": "pls", "index": 1, "name": "PLS2", "var": 4.1,
             "pos_label": "Confusion", "neg_label": "History"},
        ],
    },
    "gemma3_27b_abliterated__layer_37_strength_4.0": {
        "base": ANALYSIS_ROOT / "gemma3_27b_abliterated/layer_37_strength_4.0",
        "title": "Gemma3-27B Abliterated, Layer 37, Strength 4.0",
        "strength_scale": 4.0,
        "has_pls": False,
        "panels": [
            {"source": "deltas", "index": 0, "name": "δPC1", "var": 19.3,
             "pos_label": "Casual", "neg_label": "Formal"},
            {"source": "deltas", "index": 1, "name": "δPC2", "var": 11.3,
             "pos_label": "Concrete", "neg_label": "Abstract"},
            {"source": "deltas", "index": 2, "name": "δPC3", "var": 9.8,
             "pos_label": "Sports", "neg_label": "Nature"},
            {"source": "deltas", "index": 3, "name": "δPC4", "var": 6.2,
             "pos_label": "Everyone", "neg_label": "Athletics"},
            {"source": "deltas", "index": 4, "name": "δPC5", "var": 4.8,
             "pos_label": "Biology", "neg_label": "Aesthetic"},
        ],
    },
}


def load_detection_curve(exp_dir, pc_index):
    """Load detection rates for all alphas for a given PC."""
    rates = []
    for alpha in ALPHAS:
        fname = f"pc_{pc_index}_alpha_{alpha}.json"
        fpath = exp_dir / "pc_experiments" / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            rates.append(data["detection_rate"])
        else:
            rates.append(np.nan)
    return np.array(rates)


def load_logit_lens(interp_dir, pc_index, n_tokens=5):
    """Load top/bottom tokens from logit lens for a PC."""
    fpath = interp_dir / "pc_interpretation" / "logit_lens" / f"pc_{pc_index}_logits.json"
    if not fpath.exists():
        return [], []
    with open(fpath) as f:
        data = json.load(f)
    top = [t["token"].strip() for t in data.get("top_tokens", [])[:n_tokens]]
    bottom = [t["token"].strip() for t in data.get("bottom_tokens", [])[:n_tokens]]
    return top, bottom


# ============================================================================
# COMBINED FIGURE: 5 PCs + PLS2 in 2x3 grid
# ============================================================================

def plot_combined(config_key="gemma3_27b__layer_37_strength_4.0"):
    """Plot detection curves for a given configuration. 2x3 grid if 6 panels, or 2x3 with hidden 6th if 5."""

    cfg = CONFIGS[config_key]
    base = cfg["base"]
    deltas_dir = base / "subspace_deltas"
    pls_dir = base / "subspace_pls"
    output_dir = base / "paper_figures"
    output_dir.mkdir(exist_ok=True)
    panels = cfg["panels"]
    strength_scale = cfg["strength_scale"]

    # Font sizes (scaled for compact figure)
    FONT_TITLE = 17
    FONT_AXIS = 17
    FONT_TICK = 20
    FONT_SEMANTIC = 13

    n_panels = len(panels)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 5.4), sharex=True, sharey=True)
    axes = axes.flatten()

    alphas = np.array(ALPHAS)
    strengths = alphas * strength_scale
    mid_idx = ALPHAS.index(0.0)

    # Determine x-axis range from strength_scale
    x_max = 2.0 * strength_scale
    x_ticks = [-x_max, -x_max/2, 0, x_max/2, x_max]

    for i, panel in enumerate(panels):
        ax = axes[i]
        exp_dir = deltas_dir if panel["source"] == "deltas" else pls_dir
        rates = load_detection_curve(exp_dir, panel["index"])

        # Split line into negative and positive segments
        ax.plot(strengths[:mid_idx+1], rates[:mid_idx+1], color=COLOR_NEG, linewidth=5.25, zorder=3)
        ax.plot(strengths[mid_idx:], rates[mid_idx:], color=COLOR_POS, linewidth=5.25, zorder=3)

        ax.fill_between(strengths[:mid_idx+1], rates[:mid_idx+1], alpha=0.18, color=COLOR_NEG, zorder=1)
        ax.fill_between(strengths[mid_idx:], rates[mid_idx:], alpha=0.18, color=COLOR_POS, zorder=1)

        ax.scatter(strengths[:mid_idx+1], rates[:mid_idx+1], color=COLOR_NEG, s=25, zorder=4,
                   edgecolors='white', linewidths=0.5)
        ax.scatter(strengths[mid_idx:], rates[mid_idx:], color=COLOR_POS, s=25, zorder=4,
                   edgecolors='white', linewidths=0.5)

        # Highlight peaks on each side
        neg_rates = rates[:mid_idx+1]
        pos_rates = rates[mid_idx:]
        neg_peak_idx = np.nanargmax(neg_rates)
        pos_peak_idx = np.nanargmax(pos_rates) + mid_idx

        if rates[neg_peak_idx] > 0.1:
            ax.scatter([strengths[neg_peak_idx]], [rates[neg_peak_idx]], color=COLOR_NEG, s=70, zorder=5,
                       edgecolors='white', linewidths=1.0, marker='D')
        if rates[pos_peak_idx] > 0.1:
            ax.scatter([strengths[pos_peak_idx]], [rates[pos_peak_idx]], color=COLOR_POS, s=70, zorder=5,
                       edgecolors='white', linewidths=1.0, marker='D')

        # Semantic labels
        label_x = x_max * 0.6  # proportional to axis range
        ax.text(-label_x, 1.05, panel["neg_label"], fontsize=FONT_SEMANTIC, color=COLOR_NEG,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(label_x, 1.05, panel["pos_label"], fontsize=FONT_SEMANTIC, color=COLOR_POS,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)

        # Title
        title = f'{panel["name"]} ({panel["var"]:.1f}%)'
        ax.set_title(title, fontsize=FONT_TITLE, pad=21, color='black')

        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(-x_max * 1.075, x_max * 1.075)
        ax.set_xticks([int(t) for t in x_ticks])
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.tick_params(axis='both', labelsize=FONT_TICK)

        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide unused panels
    for j in range(n_panels, nrows * ncols):
        axes[j].set_visible(False)

    plt.tight_layout(h_pad=0.46, w_pad=0.0)

    fig.text(0.5, -0.02, 'Steering strength', ha='center', fontsize=FONT_AXIS + 3)
    fig.text(-0.02, 0.5, 'Detection rate', va='center', rotation='vertical', fontsize=FONT_AXIS + 3)

    out_path = output_dir / "fig_section4_combined.png"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# COMBINED LESS: 3 PCs + PLS2 in 2x2 grid
# ============================================================================

def plot_combined_less(config_key="gemma3_27b__layer_37_strength_4.0"):
    """Plot δPC1, δPC2, δPC3, PLS2 in a 2x2 grid. Same style as plot_combined."""

    cfg = CONFIGS[config_key]
    base = cfg["base"]
    deltas_dir = base / "subspace_deltas"
    pls_dir = base / "subspace_pls"
    output_dir = base / "paper_figures"
    output_dir.mkdir(exist_ok=True)
    strength_scale = cfg["strength_scale"]

    # Pick δPC1, δPC2, δPC3, PLS2 from the config panels
    all_panels = cfg["panels"]
    selected_names = {"δPC1", "δPC2", "δPC3", "PLS2"}
    panels = [p for p in all_panels if p["name"] in selected_names]

    if len(panels) < 4:
        print(f"Warning: only found {len(panels)} matching panels for {config_key}")

    # Font sizes
    FONT_TITLE = 20
    FONT_AXIS = 17
    FONT_TICK = 19
    FONT_SEMANTIC = 14

    ncols = 2
    nrows = 2
    # Scale figsize: 2x2 instead of 2x3, keep same subplot size
    # Original 2x3 was 6.5 wide -> each subplot ~2.17. For 2 cols: ~4.33
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0, 4.6), sharex=True, sharey=True)
    axes = axes.flatten()

    alphas = np.array(ALPHAS)
    strengths = alphas * strength_scale
    mid_idx = ALPHAS.index(0.0)

    x_max = 2.0 * strength_scale
    x_ticks = [-x_max, -x_max/2, 0, x_max/2, x_max]

    for i, panel in enumerate(panels):
        ax = axes[i]
        exp_dir = deltas_dir if panel["source"] == "deltas" else pls_dir
        rates = load_detection_curve(exp_dir, panel["index"])

        # Split line into negative and positive segments
        ax.plot(strengths[:mid_idx+1], rates[:mid_idx+1], color=COLOR_NEG, linewidth=5.25, zorder=3)
        ax.plot(strengths[mid_idx:], rates[mid_idx:], color=COLOR_POS, linewidth=5.25, zorder=3)

        ax.fill_between(strengths[:mid_idx+1], rates[:mid_idx+1], alpha=0.18, color=COLOR_NEG, zorder=1)
        ax.fill_between(strengths[mid_idx:], rates[mid_idx:], alpha=0.18, color=COLOR_POS, zorder=1)

        ax.scatter(strengths[:mid_idx+1], rates[:mid_idx+1], color=COLOR_NEG, s=25, zorder=4,
                   edgecolors='white', linewidths=0.5)
        ax.scatter(strengths[mid_idx:], rates[mid_idx:], color=COLOR_POS, s=25, zorder=4,
                   edgecolors='white', linewidths=0.5)

        # Highlight peaks on each side
        neg_rates = rates[:mid_idx+1]
        pos_rates = rates[mid_idx:]
        neg_peak_idx = np.nanargmax(neg_rates)
        pos_peak_idx = np.nanargmax(pos_rates) + mid_idx

        if rates[neg_peak_idx] > 0.1:
            ax.scatter([strengths[neg_peak_idx]], [rates[neg_peak_idx]], color=COLOR_NEG, s=70, zorder=5,
                       edgecolors='white', linewidths=1.0, marker='D')
        if rates[pos_peak_idx] > 0.1:
            ax.scatter([strengths[pos_peak_idx]], [rates[pos_peak_idx]], color=COLOR_POS, s=70, zorder=5,
                       edgecolors='white', linewidths=1.0, marker='D')

        # Semantic labels
        label_x = x_max * 0.6
        ax.text(-label_x, 1.05, panel["neg_label"], fontsize=FONT_SEMANTIC, color=COLOR_NEG,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(label_x, 1.05, panel["pos_label"], fontsize=FONT_SEMANTIC, color=COLOR_POS,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)

        # Title
        title = f'{panel["name"]} ({panel["var"]:.1f}%)'
        ax.set_title(title, fontsize=FONT_TITLE, pad=21, color='black')

        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(-x_max * 1.075, x_max * 1.075)
        ax.set_xticks([int(t) for t in x_ticks])
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.tick_params(axis='both', labelsize=FONT_TICK)

        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout(h_pad=1.2, w_pad=0.0)

    fig.text(0.53, -0.02, 'Steering strength', ha='center', fontsize=23)

    out_path = output_dir / "fig_section4_combined_less.png"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 1: First 5 PCs (standalone, no PLS)
# ============================================================================

def plot_figure1():
    """First 5 PCs from subspace_deltas."""

    pcs = [
        {"index": 0, "name": "PC1", "var": 19.6,
         "pos_label": "Faithful", "neg_label": "Code"},
        {"index": 1, "name": "PC2", "var": 12.0,
         "pos_label": "Concepts", "neg_label": "Species"},
        {"index": 2, "name": "PC3", "var": 8.4,
         "pos_label": "Professions", "neg_label": "Emotions"},
        {"index": 3, "name": "PC4", "var": 7.5,
         "pos_label": "Sports", "neg_label": "Scientists"},
        {"index": 4, "name": "PC5", "var": 4.8,
         "pos_label": "Inanimate", "neg_label": "People"},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()

    alphas = np.array(ALPHAS)
    strengths = alphas * STRENGTH_SCALE
    mid_idx = ALPHAS.index(0.0)

    for i, pc in enumerate(pcs):
        ax = axes[i]
        rates = load_detection_curve(DELTAS_DIR, pc["index"])

        ax.plot(strengths, rates, color=COLOR_LINE, linewidth=2.5, zorder=3)
        ax.fill_between(strengths[:mid_idx+1], rates[:mid_idx+1], alpha=0.18, color=COLOR_NEG, zorder=1)
        ax.fill_between(strengths[mid_idx:], rates[mid_idx:], alpha=0.18, color=COLOR_POS, zorder=1)
        ax.scatter(strengths, rates, color=COLOR_LINE, s=25, zorder=4, edgecolors='white', linewidths=0.5)

        neg_rates = rates[:mid_idx+1]
        pos_rates = rates[mid_idx:]
        neg_peak_idx = np.nanargmax(neg_rates)
        pos_peak_idx = np.nanargmax(pos_rates) + mid_idx
        if rates[neg_peak_idx] > 0.1:
            ax.scatter([strengths[neg_peak_idx]], [rates[neg_peak_idx]], color=COLOR_NEG, s=70, zorder=5,
                       edgecolors='white', linewidths=1.0, marker='D')
        if rates[pos_peak_idx] > 0.1:
            ax.scatter([strengths[pos_peak_idx]], [rates[pos_peak_idx]], color=COLOR_POS, s=70, zorder=5,
                       edgecolors='white', linewidths=1.0, marker='D')

        ax.text(-6.4, 1.02, pc["neg_label"], fontsize=14, color=COLOR_NEG,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(6.4, 1.02, pc["pos_label"], fontsize=14, color=COLOR_POS,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)

        ax.set_title(f'{pc["name"]} ({pc["var"]:.1f}% var.)', fontweight='bold', fontsize=20, pad=20)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(-8.6, 8.6)
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.tick_params(axis='both', labelsize=16)

        ax.text(-0.08, 1.18, chr(65 + i), transform=ax.transAxes, fontsize=20,
                fontweight='bold', va='top')

        if i >= 3:
            ax.set_xlabel('Steering strength', fontsize=20, labelpad=10)
        if i % 3 == 0:
            ax.set_ylabel('Detection rate', fontsize=20, labelpad=10)

    # Hide the 6th panel (empty)
    axes[5].set_visible(False)

    plt.tight_layout(h_pad=2.0, w_pad=1.5)

    out_path = OUTPUT_DIR / "fig_bidirectional_pcs.png"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08)
    print(f"Saved Figure 1: {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 2: PLS Components
# ============================================================================

def plot_figure2():
    """PLS components showing detection-predicting directions."""

    selected_pls = [
        {"index": 1, "name": "PLS2", "r": 0.383, "var": 3.8,
         "pos_label": "Entities", "neg_label": "History"},
        {"index": 7, "name": "PLS8", "r": None, "var": 6.2,
         "pos_label": "Emotions", "neg_label": "Sports"},
        {"index": 9, "name": "PLS10", "r": None, "var": 2.3,
         "pos_label": "Biological", "neg_label": "Historical"},
        {"index": 0, "name": "PLS1", "r": None, "var": 0.05,
         "pos_label": "Knowledge", "neg_label": "Confusion"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()

    alphas = np.array(ALPHAS)
    strengths = alphas * STRENGTH_SCALE
    mid_idx = ALPHAS.index(0.0)
    pls_colors = [plot_style.OLIVE, plot_style.SKY, plot_style.CLAY, plot_style.AQUA_500]

    for i, pls_info in enumerate(selected_pls):
        ax = axes[i]
        rates = load_detection_curve(PLS_DIR, pls_info["index"])
        color = pls_colors[i]

        ax.plot(strengths, rates, color=color, linewidth=2.5, zorder=3)
        ax.scatter(strengths, rates, color=color, s=25, zorder=4, edgecolors='white', linewidths=0.5)

        # Always red left, blue right
        ax.fill_between(strengths[:mid_idx+1], rates[:mid_idx+1], alpha=0.18, color=COLOR_NEG, zorder=1)
        ax.fill_between(strengths[mid_idx:], rates[mid_idx:], alpha=0.18, color=COLOR_POS, zorder=1)

        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.text(-6.4, 1.02, pls_info["neg_label"], fontsize=14, color=COLOR_NEG,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(6.4, 1.02, pls_info["pos_label"], fontsize=14, color=COLOR_POS,
                fontweight='bold', ha='center', style='italic',
                transform=ax.get_xaxis_transform(), clip_on=False)

        title = f'{pls_info["name"]} ({pls_info["var"]:.1f}% var.)'
        if pls_info["r"] is not None:
            title += f'  $r$ = {pls_info["r"]:.3f}'
        ax.set_title(title, fontweight='bold', fontsize=22, pad=20)

        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(-8.6, 8.6)
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.tick_params(axis='both', labelsize=18)

        if i >= 2:
            ax.set_xlabel('Steering strength', fontsize=22, labelpad=10)
        if i % 2 == 0:
            ax.set_ylabel('Detection rate', fontsize=22, labelpad=10)

    plt.tight_layout(h_pad=2.0)

    out_path = OUTPUT_DIR / "fig_pls_components.png"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08)
    print(f"Saved Figure 2: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    # If a config key is passed as argument, only generate that one
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = list(CONFIGS.keys())

    for key in keys:
        if key not in CONFIGS:
            print(f"Unknown config: {key}")
            continue
        cfg = CONFIGS[key]
        print(f"\n=== {cfg['title']} ===")

        # Print logit lens for reference
        base = cfg["base"]
        deltas_dir = base / "subspace_deltas"
        pls_dir = base / "subspace_pls"
        for idx in range(5):
            top, bottom = load_logit_lens(deltas_dir, idx, n_tokens=3)
            print(f"  δPC{idx+1}: (+) {', '.join(top)} | (-) {', '.join(bottom)}")
        if cfg["has_pls"]:
            top, bottom = load_logit_lens(pls_dir, 1, n_tokens=3)
            print(f"  PLS2: (+) {', '.join(top)} | (-) {', '.join(bottom)}")

        plot_combined(key)
        if cfg["has_pls"]:
            plot_combined_less(key)

    print("\nDone!")
