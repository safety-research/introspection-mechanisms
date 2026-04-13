"""
Generate a grouped bar plot showing linear probe accuracy before vs after
adding each attention head's output, for the top success heads.

Data comes from exp30_qkv_decomposition: for each head, a linear probe was
trained on the residual stream before the head (h_before) and after adding
the head's output (h_after = h_before + head_output). The probe predicts
whether the model was steered (binary classification, 5-fold cross-validated).

These 50 heads were selected via gradient-based attribution on the introspection
probe direction, then classified by their effect on probe output (beneficial vs
detrimental, copier vs transformer).
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_style

# Match plot_section4_figures.py style
plot_style.set_defaults(matplotlib=True, plotly=False, pretty=False, install_brand_fonts=False)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams.get("font.sans-serif", [])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

BASE_DIR = Path("analysis/exp30_qkv_decomposition/gemma3_27b/layer_37_strength_4.0/success")
OUTPUT_DIR = Path("analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect data from all success heads
head_data = []
for results_file in BASE_DIR.rglob("qk_nonlinearity_results.json"):
    with open(results_file) as f:
        data = json.load(f)

    probe = data.get("linear_probe_comparison", {})
    if not probe or "accuracy_before" not in probe:
        continue

    layer_idx = data["layer_idx"]
    head_idx = data["head_idx"]
    category = results_file.parent.parent.parent.name

    head_data.append({
        "label": f"L{layer_idx}H{head_idx}",
        "layer": layer_idx,
        "head": head_idx,
        "category": category,
        "acc_before": probe["accuracy_before"],
        "acc_before_std": probe.get("accuracy_before_std", 0),
        "acc_after": probe["accuracy_after"],
        "acc_after_std": probe.get("accuracy_after_std", 0),
        "improvement": probe.get("improvement", 0),
        "p_value": probe.get("p_value", 1.0),
    })

# Sort by absolute improvement (largest effect first)
head_data.sort(key=lambda x: abs(x["improvement"]), reverse=True)

print(f"Total success heads with probe data: {len(head_data)}")
print(f"\nImprovement stats (all {len(head_data)} heads):")
improvements = np.array([h["improvement"] for h in head_data])
print(f"  Mean: {np.mean(improvements):.4f}")
print(f"  Std:  {np.std(improvements):.4f}")
print(f"  Max:  {np.max(improvements):.4f}")
print(f"  Min:  {np.min(improvements):.4f}")
print(f"  Median: {np.median(improvements):.4f}")

n_sig = sum(1 for h in head_data if h["p_value"] < 0.05)
print(f"  Sig (p<0.05): {n_sig}/{len(head_data)}")

# Top-20 by absolute improvement — these are the heads with LARGEST effects
N = 20
top_heads = head_data[:N]

# Sort by layer/head for x-axis readability
top_heads.sort(key=lambda x: (x["layer"], x["head"]))

labels = [h["label"] for h in top_heads]
acc_before = np.array([h["acc_before"] for h in top_heads])
acc_after = np.array([h["acc_after"] for h in top_heads])
# Standard error: std / sqrt(k) for k-fold CV
k_folds = 5
se_before = np.array([h["acc_before_std"] for h in top_heads]) / np.sqrt(k_folds)
se_after = np.array([h["acc_after_std"] for h in top_heads]) / np.sqrt(k_folds)

# Stats for legend annotation
mean_delta = np.mean(improvements)
std_delta = np.std(improvements)

# --- Create the figure ---
fig, ax = plt.subplots(figsize=(24, 10))

x = np.arange(len(labels))
bar_width = 0.35

bars1 = ax.bar(x - bar_width/2, acc_before, bar_width,
               yerr=se_before, capsize=5,
               label='Before head addition',
               color=plot_style.CLAY, edgecolor='none',
               error_kw=dict(lw=2.2, color=plot_style.GRAY_600, capthick=2.2))

bars2 = ax.bar(x + bar_width/2, acc_after, bar_width,
               yerr=se_after, capsize=5,
               label='After head addition',
               color=plot_style.SKY, edgecolor='none',
               error_kw=dict(lw=2.2, color=plot_style.GRAY_600, capthick=2.2))

# Styling — 1.5x font sizes
ax.set_ylabel(f'Linear probe accuracy\n({k_folds}-fold CV)', fontsize=37)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=50, ha='center', fontsize=31)
ax.tick_params(axis='y', labelsize=33)

# Y-axis: full 0 to 1 range
ax.set_ylim(0, 1.0)

ax.grid(True, alpha=0.3, axis='y')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# tight_layout BEFORE legend positioning so coordinates are stable
plt.tight_layout()

# Legend with delta text as a third "handle"
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

h1 = Patch(facecolor=plot_style.CLAY, label='Before head addition')
h2 = Patch(facecolor=plot_style.SKY, label='After head addition')
delta_text = f'Mean $\\Delta$ accuracy: {mean_delta:+.1%} $\\pm$ {std_delta:.1%}'
h3 = Line2D([], [], color='none', label=delta_text)

leg = ax.legend(handles=[h1, h2, h3], fontsize=34, loc='upper left',
                frameon=True, edgecolor='0.8', fancybox=True,
                ncol=3, handletextpad=0.5, columnspacing=0.8,
                handlelength=1.0)

output_path = OUTPUT_DIR / "head_probe_before_after.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUTPUT_DIR / "head_probe_before_after.pdf", bbox_inches='tight', facecolor='white')
print(f"\nSaved to {output_path}")
plt.close()
