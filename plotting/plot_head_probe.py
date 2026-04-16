"""
Generate a grouped bar plot showing linear probe accuracy before vs after
adding each attention head's output, for the top success heads.

Data source (in order of preference):
    1) ``plotting/data/fig_head_probe.json`` — a cached, repo-local summary that
       ships alongside the release so this plot can be regenerated without
       running the full probing pipeline. Format:
         {
           "k_folds": 5,
           "heads": [
             {"layer": int, "head": int, "category": str,
              "accuracy_before": float, "accuracy_after": float,
              "accuracy_before_std": float, "accuracy_after_std": float,
              "improvement": float, "p_value": float},
             ...
           ]
         }
    2) ``analysis/12_head_investigation/gemma3_27b/.../linear_probe_results.json``
       files produced by ``experiments/12_head_investigation.py`` (field name:
       ``linear_probe_comparison``).

If no data is found at either location, prints a guidance message and exits
without crashing.

These heads are selected via gradient-based attribution on the introspection
probe direction (see ``experiments/10_head_identification.py``).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_style

plot_style.set_defaults(matplotlib=True, plotly=False, pretty=False, install_brand_fonts=False)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams.get("font.sans-serif", [])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

DEFAULT_CACHE = Path(__file__).parent / "data" / "fig_head_probe.json"
DEFAULT_ANALYSIS_DIR = Path("analysis/12_head_investigation/gemma3_27b/layer_37_strength_4.0/success")
DEFAULT_OUTPUT_DIR = Path("plotting/output")


def _load_from_cache(path: Path):
    with open(path) as f:
        payload = json.load(f)
    heads = []
    for h in payload.get("heads", []):
        layer = int(h["layer"])
        head = int(h["head"])
        heads.append({
            "label": f"L{layer}H{head}",
            "layer": layer,
            "head": head,
            "category": h.get("category", "success"),
            "acc_before": float(h["accuracy_before"]),
            "acc_before_std": float(h.get("accuracy_before_std", 0.0)),
            "acc_after": float(h["accuracy_after"]),
            "acc_after_std": float(h.get("accuracy_after_std", 0.0)),
            "improvement": float(h.get("improvement", h["accuracy_after"] - h["accuracy_before"])),
            "p_value": float(h.get("p_value", 1.0)),
        })
    return heads, int(payload.get("k_folds", 5))


def _load_from_analysis_dir(base_dir: Path):
    heads = []
    if not base_dir.exists():
        return heads, 5
    for results_file in base_dir.rglob("linear_probe_results.json"):
        with open(results_file) as f:
            data = json.load(f)
        probe = data.get("linear_probe_comparison", {})
        if not probe or "accuracy_before" not in probe:
            continue
        layer_idx = data["layer_idx"]
        head_idx = data["head_idx"]
        category = results_file.parent.parent.parent.name
        heads.append({
            "label": f"L{layer_idx}H{head_idx}",
            "layer": layer_idx,
            "head": head_idx,
            "category": category,
            "acc_before": probe["accuracy_before"],
            "acc_before_std": probe.get("accuracy_before_std", 0),
            "acc_after": probe["accuracy_after"],
            "acc_after_std": probe.get("accuracy_after_std", 0),
            "improvement": probe.get("improvement", probe["accuracy_after"] - probe["accuracy_before"]),
            "p_value": probe.get("p_value", 1.0),
        })
    return heads, 5


def main():
    parser = argparse.ArgumentParser(description="Before/after head-probe accuracy figure.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help=f"Cached summary JSON (default: {DEFAULT_CACHE}).")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR,
                        help="Fallback directory with per-head linear_probe_results.json "
                             f"(default: {DEFAULT_ANALYSIS_DIR}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to write PDF/PNG outputs (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Plot the top-N heads by absolute improvement (default: 20).")
    args = parser.parse_args()

    if args.cache.exists():
        head_data, k_folds = _load_from_cache(args.cache)
        source = f"cache: {args.cache}"
    else:
        head_data, k_folds = _load_from_analysis_dir(args.analysis_dir)
        source = f"analysis dir: {args.analysis_dir}"

    if not head_data:
        print(
            "No head-probe data found.\n"
            f"  Tried cache: {args.cache}\n"
            f"  Tried analysis dir: {args.analysis_dir}\n\n"
            "To populate the analysis dir, run:\n"
            "  python experiments/12_head_investigation.py --linear-probe ...\n"
            "or save the summary directly to the cache path above."
        )
        return 0

    print(f"Loaded {len(head_data)} heads from {source}")
    improvements = np.array([h["improvement"] for h in head_data])
    print(f"Improvement stats: mean={np.mean(improvements):+.4f} "
          f"std={np.std(improvements):.4f} min={np.min(improvements):+.4f} "
          f"max={np.max(improvements):+.4f} median={np.median(improvements):+.4f}")
    n_sig = sum(1 for h in head_data if h["p_value"] < 0.05)
    print(f"  Sig (p<0.05): {n_sig}/{len(head_data)}")

    head_data.sort(key=lambda x: abs(x["improvement"]), reverse=True)
    top_heads = head_data[:args.top_n]
    top_heads.sort(key=lambda x: (x["layer"], x["head"]))

    labels = [h["label"] for h in top_heads]
    acc_before = np.array([h["acc_before"] for h in top_heads])
    acc_after = np.array([h["acc_after"] for h in top_heads])
    se_before = np.array([h["acc_before_std"] for h in top_heads]) / np.sqrt(k_folds)
    se_after = np.array([h["acc_after_std"] for h in top_heads]) / np.sqrt(k_folds)

    mean_delta = float(np.mean(improvements))
    std_delta = float(np.std(improvements))

    fig, ax = plt.subplots(figsize=(24, 10))
    x = np.arange(len(labels))
    bar_width = 0.35

    ax.bar(x - bar_width / 2, acc_before, bar_width, yerr=se_before, capsize=5,
           label="Before head addition", color=plot_style.CLAY, edgecolor="none",
           error_kw=dict(lw=2.2, color=plot_style.GRAY_600, capthick=2.2))
    ax.bar(x + bar_width / 2, acc_after, bar_width, yerr=se_after, capsize=5,
           label="After head addition", color=plot_style.SKY, edgecolor="none",
           error_kw=dict(lw=2.2, color=plot_style.GRAY_600, capthick=2.2))

    ax.set_ylabel(f"Linear probe accuracy\n({k_folds}-fold CV)", fontsize=37)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="center", fontsize=31)
    ax.tick_params(axis="y", labelsize=33)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h1 = Patch(facecolor=plot_style.CLAY, label="Before head addition")
    h2 = Patch(facecolor=plot_style.SKY, label="After head addition")
    delta_text = f"Mean $\\Delta$ accuracy: {mean_delta:+.1%} $\\pm$ {std_delta:.1%}"
    h3 = Line2D([], [], color="none", label=delta_text)
    ax.legend(handles=[h1, h2, h3], fontsize=34, loc="upper left",
              frameon=True, edgecolor="0.8", fancybox=True,
              ncol=3, handletextpad=0.5, columnspacing=0.8, handlelength=1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "head_probe_before_after.png"
    pdf_path = args.output_dir / "head_probe_before_after.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {png_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
