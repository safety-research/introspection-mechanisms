#!/usr/bin/env python3
"""
Standalone script to regenerate paper figures from cached data.

Usage:
    python plot_figures.py             # Generate all figures
    python plot_figures.py --fig 1     # Generate only figure 1
    python plot_figures.py --fig 4     # Generate attribution graph figures

Output:
    output/metrics_vs_injection_layer.pdf
    output/metrics_vs_meta_layer.pdf
    output/metrics_vs_injection_layer_meta_bias_arms.pdf
    output/Bread_layer37.pdf            (attribution graph, both directions)
    output/Bread_layer37_bias.pdf       (attribution graph, with bias)

Requirements:
    pip install pandas matplotlib numpy pyarrow
    pip install plotly kaleido          (for fig 4 only)
    Optional: pip install anthroplot    (for Anthropic color theme)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Anthropic plot styling (local copy)
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

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

METRIC_LABELS = {
    "detection_hit_rate": r"$P(\mathrm{detect} \mid \mathrm{injection})$",
    "forced_identification_accuracy": r"$P(\mathrm{identify} \mid \mathrm{forced})$",
    "combined_detection_and_identification_rate": r"$P(\mathrm{detect} \wedge \mathrm{identify} \mid \mathrm{injection})$",
}
METRICS = list(METRIC_LABELS.keys())


# ── Figure 1: metrics_vs_injection_layer (2×2 grid) ──────────────────────────


def plot_fig1():
    from plot_style import GREY, BLUE_500, ORANGE_500, LIGHT_PURPLE

    df = pd.read_parquet(DATA_DIR / "fig1_metrics_cache.parquet")
    corr_df = pd.read_parquet(DATA_DIR / "fig1_correlations.parquet")

    FS_LABEL = 22       # axis labels (20 * 1.1)
    FS_TICK = 18.4      # tick labels (16 * 1.15)
    FS_LEGEND = 14      # legend
    FS_ANNOT = 23       # annotations / subplot letters (19.2 * 1.2)

    STRENGTH_COLORS = {1.0: BLUE_500, 2.0: ORANGE_500, 4.0: LIGHT_PURPLE}

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7))
    axes = axes.flatten()

    def _annotate_peak(ax, xs, ys, color, fontsize=13.75):
        """Annotate the peak (max y) point with its layer number, always above."""
        peak_idx = np.nanargmax(ys)
        px, py = xs.iloc[peak_idx], ys.iloc[peak_idx]
        ax.annotate(f"L{int(px)}", xy=(px, py),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=fontsize, fontweight="bold", color=color, ha="center", va="bottom",
                    clip_on=True, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.15, edgecolor="none"))

    # Subplots (a)–(c): metrics vs injection layer
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        label_letter = chr(ord("a") + idx)
        ax.text(0.02, 0.93, f"({label_letter})", transform=ax.transAxes,
                fontsize=FS_ANNOT, fontweight="bold", va="top", ha="left")

        for strength in [1.0, 2.0, 4.0]:
            df_s = df[df["strength"] == strength].sort_values("layer_idx")
            if metric not in df_s.columns:
                continue

            color = STRENGTH_COLORS[strength]
            ax.plot(df_s["layer_idx"], df_s[metric],
                    marker="o", markersize=4, linewidth=4, alpha=0.85,
                    label=r"Strength ($\alpha$) = " + f"{strength:.1f}", color=color)

            ci_lo, ci_hi = f"{metric}_ci_lower", f"{metric}_ci_upper"
            if ci_lo in df_s.columns and ci_hi in df_s.columns:
                ax.fill_between(df_s["layer_idx"], df_s[ci_lo], df_s[ci_hi],
                                alpha=0.15, color=color)

            # Annotate peak layer
            _annotate_peak(ax, df_s["layer_idx"], df_s[metric], color)

        # Only show x-label on bottom row
        if idx >= 2:
            ax.set_xlabel("Injection layer", fontsize=FS_LABEL)
        ax.set_ylabel("Probability", fontsize=FS_LABEL)
        ax.set_title(METRIC_LABELS[metric], fontsize=FS_LABEL, color="black")
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(True, alpha=0.2, color=GREY)
        ax.set_xlim(-1, 62)
        if idx == 1:
            ax.set_yticks([0.0, 0.3, 0.6, 0.9])

    # Subplot (d): correlation
    ax = axes[3]
    ax.text(0.02, 0.93, "(d)", transform=ax.transAxes,
            fontsize=FS_ANNOT, fontweight="bold", va="top", ha="left")

    for strength in [1.0, 2.0, 4.0]:
        c = corr_df[corr_df["strength"] == strength].sort_values("layer_idx")
        color = STRENGTH_COLORS[strength]
        ax.plot(c["layer_idx"], c["correlation"],
                marker="o", markersize=4, linewidth=4, alpha=0.85,
                label=r"Strength ($\alpha$) = " + f"{strength:.1f}", color=color)
        ax.fill_between(c["layer_idx"], c["ci_lower"], c["ci_upper"],
                        alpha=0.15, color=color)

        # Annotate peak layer
        _annotate_peak(ax, c["layer_idx"], c["correlation"], color)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Injection layer", fontsize=FS_LABEL)
    ax.set_ylabel("Correlation", fontsize=FS_LABEL, labelpad=-2)
    ax.set_title(r"$r(P(\mathrm{detect} \mid \mathrm{injection}),\; P(\mathrm{identify} \mid \mathrm{forced}))$", fontsize=FS_LABEL, color="black")
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.2, color=GREY)
    ax.set_xlim(-1, 62)

    # Force one-decimal y-tick formatting on all subplots
    from matplotlib.ticker import FormatStrFormatter
    for ax in axes.flatten():
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Extend y-max on all subplots to give headroom for peak annotations
    for ax in axes.flatten():
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.15 * (ymax - ymin))

    # Shared legend at bottom, horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               fontsize=FS_LEGEND * 1.3, ncol=3, framealpha=0.8)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    out = OUTPUT_DIR / "metrics_vs_injection_layer"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')}")


# ── Figure 2: metrics_vs_meta_layer (3×1 stacked) ────────────────────────────


def plot_fig2():
    from plot_style import GREY
    from matplotlib.ticker import FormatStrFormatter

    df = pd.read_parquet(DATA_DIR / "fig2_sweep_metrics_cache.parquet")

    FS_LABEL = 24.2      # 22 * 1.1
    FS_TICK = 20.24      # 18.4 * 1.1
    FS_LEGEND = 14
    FS_ANNOT = 25.3      # 23 * 1.1

    injection_layers = [30, 40, 50]
    strengths = [2.0, 4.0]

    # Separate baseline
    baseline_rows = df[df["meta_layer"] == -1] if "meta_layer" in df.columns else pd.DataFrame()
    baseline_metrics = None
    baseline_individual_rows = None

    if not baseline_rows.empty:
        aggregated_baseline = baseline_rows[baseline_rows["injection_layer"] == -1]
        individual_baselines = baseline_rows[baseline_rows["injection_layer"] != -1]

        if not aggregated_baseline.empty:
            row = aggregated_baseline.iloc[0]
            baseline_metrics = {}
            for m in METRICS:
                if m in row.index:
                    baseline_metrics[m] = row[m]
                    for suffix in ["_ci_lower", "_ci_upper"]:
                        k = m + suffix
                        if k in row.index:
                            baseline_metrics[k] = row[k]

        if not individual_baselines.empty:
            baseline_individual_rows = individual_baselines

    # Filter main data
    mask = (df["injection_layer"].isin(injection_layers)) & (df["strength"].isin(strengths))
    if "meta_layer" in df.columns:
        mask = mask & (df["meta_layer"] != -1)
    agg = df[mask].copy()
    if agg.empty:
        print("No data for fig2")
        return

    plot_metrics = [m for m in METRICS if m in agg.columns]
    combinations = sorted(agg[["injection_layer", "strength"]].drop_duplicates().values.tolist())
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    combo_to_color = {tuple(c): cycle_colors[i % len(cycle_colors)] for i, c in enumerate(combinations)}

    nrows = len(plot_metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 5 * nrows))
    axes = np.atleast_1d(axes)

    def _annotate_peak(ax, xs, ys, color, fontsize=15.1):
        peak_idx = np.nanargmax(ys)
        px, py = xs.iloc[peak_idx], ys.iloc[peak_idx]
        ax.annotate(f"L{int(px)}", xy=(px, py),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=fontsize, fontweight="bold", color=color, ha="center", va="bottom",
                    clip_on=True, zorder=11,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.15, edgecolor="none"))

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        label_letter = chr(ord("a") + idx)

        ci_lo = f"{metric}_ci_lower"
        ci_hi = f"{metric}_ci_upper"

        # Individual lines (transparent)
        for inj_layer, strength in combinations:
            combo_data = agg[(agg["injection_layer"] == inj_layer) & (agg["strength"] == strength)]
            if combo_data.empty:
                continue
            grp = combo_data.groupby("meta_layer", as_index=False).agg({
                metric: "mean", ci_lo: "mean", ci_hi: "mean",
            }).sort_values("meta_layer")

            label = f"L{int(inj_layer)}, " + r"$\alpha$" + f"={strength:.1f}"
            color = combo_to_color[(inj_layer, strength)]

            if ci_lo in grp.columns and ci_hi in grp.columns:
                ax.fill_between(grp["meta_layer"], grp[ci_lo], grp[ci_hi],
                                alpha=0.15, color=color, zorder=1)
            ax.plot(grp["meta_layer"], grp[metric], "o-", linewidth=2, markersize=3,
                    label=label, color=color, alpha=0.55, zorder=2)

        # Aggregated line
        agg_grp = agg.groupby("meta_layer", as_index=False).agg({metric: ["mean", "std", "count"]})
        agg_grp.columns = ["meta_layer", f"{metric}_mean", f"{metric}_std", "n"]
        agg_grp = agg_grp.sort_values("meta_layer")
        agg_grp["sem"] = agg_grp[f"{metric}_std"] / np.sqrt(agg_grp["n"])
        z = 1.96
        agg_grp["ci_lower"] = agg_grp[f"{metric}_mean"] - z * agg_grp["sem"]
        agg_grp["ci_upper"] = agg_grp[f"{metric}_mean"] + z * agg_grp["sem"]

        line = ax.plot(agg_grp["meta_layer"], agg_grp[f"{metric}_mean"], "o-",
                       linewidth=4, markersize=4, label="Mean steering",
                       alpha=0.85, zorder=10)
        agg_color = line[0].get_color()

        # Peak annotation for aggregated line
        _annotate_peak(ax, agg_grp["meta_layer"], agg_grp[f"{metric}_mean"], agg_color)

        xlim = ax.get_xlim()

        # Individual baseline lines (background)
        if baseline_individual_rows is not None and metric in baseline_individual_rows.columns:
            for _, brow in baseline_individual_rows.iterrows():
                bval = brow[metric]
                if pd.isna(bval):
                    continue
                inj_l = int(brow.get("injection_layer", -1))
                s = float(brow.get("strength", -1.0))
                color = combo_to_color.get((inj_l, s), "gray")
                ax.axhline(y=bval, color=color, linestyle="--", linewidth=1.5,
                           alpha=0.3, zorder=1, label=None)

        ax.set_xlim(xlim)

        # Aggregated baseline
        if baseline_metrics and metric in baseline_metrics:
            bval = baseline_metrics[metric]
            baseline_zorder = 5 if idx == 0 else 9
            ax.axhline(y=bval, color=agg_color, linestyle="--", linewidth=2,
                        alpha=1.0, label="Mean baseline", zorder=baseline_zorder)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=FS_LABEL, color="black")
        ax.text(0.02, 0.93, f"({label_letter})", transform=ax.transAxes,
                fontsize=FS_ANNOT, fontweight="bold", va="top", ha="left")

        if idx >= 2:
            ax.set_xlabel("Steering layer", fontsize=FS_LABEL)
        ax.set_ylabel("Probability", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(True, alpha=0.2, color=GREY)

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if idx == 0:
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.02)
        if idx == 0:
            # Inset for false alarm rate
            if "detection_false_alarm_rate" in agg.columns:
                axins = ax.inset_axes([0.4, 0.09, 0.35, 0.33])
                axins.patch.set_zorder(10)
                axins.set_zorder(10)

                fa = "detection_false_alarm_rate"
                fa_grp = agg.groupby("meta_layer", as_index=False).agg({fa: ["mean", "std", "count"]})
                fa_grp.columns = ["meta_layer", f"{fa}_mean", f"{fa}_std", "n"]
                fa_grp = fa_grp.sort_values("meta_layer")
                fa_grp["sem"] = fa_grp[f"{fa}_std"] / np.sqrt(fa_grp["n"])
                fa_grp["ci_lower"] = fa_grp[f"{fa}_mean"] - z * fa_grp["sem"]
                fa_grp["ci_upper"] = fa_grp[f"{fa}_mean"] + z * fa_grp["sem"]

                axins.plot(fa_grp["meta_layer"], fa_grp[f"{fa}_mean"], "o-",
                           linewidth=2, markersize=3,
                           alpha=1.0, zorder=10)
                axins.text(0.98, 0.92, r"$P(\mathrm{detect} \mid \mathrm{control})$",
                           transform=axins.transAxes, fontsize=FS_TICK * 0.72, va="top", ha="right")
                axins.grid(True, alpha=0.2, color=GREY)
                axins.tick_params(labelsize=FS_TICK * 0.68)
                fa_ylim = axins.get_ylim()
                axins.set_ylim(-0.08, min(1, fa_ylim[1] + 0.08))

    # Extend y-max for annotation headroom
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.05 * (ymax - ymin))

    # Shared legend at bottom, horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               fontsize=FS_LEGEND * 1.1, ncol=4, framealpha=0.8,
               handlelength=1.5, handletextpad=0.4, columnspacing=0.8)
    for line in leg.get_lines():
        line.set_linewidth(3.5)
        line.set_alpha(1.0)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.045)

    out = OUTPUT_DIR / "metrics_vs_meta_layer"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')}")


# ── Figure 3: metrics_vs_injection_layer_meta_bias_arms (3×1 stacked) ─────────


def plot_fig3():
    from plot_style import GREY
    from matplotlib.ticker import FormatStrFormatter

    df = pd.read_parquet(DATA_DIR / "fig3_metrics_cache.parquet")
    arms = sorted(df["arm"].unique())

    FS_LABEL = 24.2      # 22 * 1.1
    FS_TICK = 20.24      # 18.4 * 1.1
    FS_LEGEND = 14
    FS_ANNOT = 25.3      # 23 * 1.1

    def _annotate_peak(ax, xs, ys, color, fontsize=15.1):
        peak_idx = np.nanargmax(ys)
        px, py = xs.iloc[peak_idx], ys.iloc[peak_idx]
        ax.annotate(f"L{int(px)}", xy=(px, py),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=fontsize, fontweight="bold", color=color, ha="center", va="bottom",
                    clip_on=True, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.15, edgecolor="none"))

    nrows = len(METRICS)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 5 * nrows))
    axes = np.atleast_1d(axes)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        label_letter = chr(ord("a") + idx)

        for arm in arms:
            df_arm = df[df["arm"] == arm].sort_values("layer_idx")
            if df_arm.empty or metric not in df_arm.columns:
                continue

            if arm == "baseline":
                label, linestyle = "Baseline", "--"
            else:
                label, linestyle = f"Steering {arm}", "-"

            line = ax.plot(df_arm["layer_idx"], df_arm[metric],
                           marker="o", markersize=4, linewidth=4, alpha=0.85,
                           label=label, linestyle=linestyle)
            line[0].set_linestyle(linestyle)
            color = line[0].get_color()

            ci_lo = f"{metric}_ci_lower"
            ci_hi = f"{metric}_ci_upper"
            if ci_lo in df_arm.columns and ci_hi in df_arm.columns:
                ax.fill_between(df_arm["layer_idx"], df_arm[ci_lo], df_arm[ci_hi],
                                alpha=0.15, color=color)

            # Peak annotation only
            _annotate_peak(ax, df_arm["layer_idx"], df_arm[metric], color)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=FS_LABEL, color="black")
        ax.text(0.02, 0.93, f"({label_letter})", transform=ax.transAxes,
                fontsize=FS_ANNOT, fontweight="bold", va="top", ha="left")

        if idx >= 2:
            ax.set_xlabel("Injection layer", fontsize=FS_LABEL)
        ax.set_ylabel("Probability", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(True, alpha=0.2, color=GREY)
        ax.set_xlim(-1, 62)

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if idx == 0:
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.02)

    # Extend y-max for annotation headroom
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.05 * (ymax - ymin))

    # Shared legend at bottom, horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               fontsize=FS_LEGEND * 1.1, ncol=2, framealpha=0.8,
               handlelength=1.5, handletextpad=0.4, columnspacing=0.8)
    for line in leg.get_lines():
        line.set_linewidth(3.5)
        line.set_alpha(1.0)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.048)

    out = OUTPUT_DIR / "metrics_vs_injection_layer_meta_bias_arms"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')}")


# ── Figure 4: Attribution graphs (Plotly) ─────────────────────────────────────

SAE_COLORS = {
    "transcoder_all": "#4C72B0",
    "attn_out_all": "#DD8452",
    "mlp_out_all": "#55A868",
    "resid_post_all": "#C44E52",
    "remainder": "#9467BD",
    "root": "#333333",
}
SAE_ABBREV = {"transcoder_all": "TC", "attn_out_all": "ATTN", "mlp_out_all": "MLP", "resid_post_all": "RESID"}


def render_graph_pdf(laid_out, label_data, concept, output_path, top_n=None,
                     exclude_nodes=None, feature_types=None, include_nodes=None,
                     x_position_offsets=None):
    """Render attribution graph as PDF using Plotly (matching render_pdfs.py).

    Args:
        top_n: If set, keep only the top N nodes by |root_edge_weight| and
               edges between them, then re-layout positions.
        exclude_nodes: Set of (layer, sae_type, feature_id) tuples to exclude.
        feature_types: Dict mapping "layer|sae_type|fid|tpos" -> "gate"/"carrier".
               Gate features get a dark border, carrier features get a thin white border.
        include_nodes: Optional set of (layer, sae_type, feature_id) tuples to force-include
               after the top-N selection and before exclusions.
        x_position_offsets: Optional dict mapping (layer, sae_type, feature_id) tuples
               to additive x-offsets applied after layout.
    """
    import math
    import plotly.graph_objects as go

    nodes = laid_out["nodes"]
    edges = laid_out["edges"]

    if not nodes:
        return

    # Filter to top N nodes by REW (before exclusion)
    if top_n is not None:
        nodes_sorted = sorted(nodes, key=lambda n: -abs(n.get("root_edge_weight", 0)))
        nodes = nodes_sorted[:top_n]
        if include_nodes:
            present = {(n["layer"], n["sae_type"], n["feature_id"]) for n in nodes}
            for n in nodes_sorted[top_n:]:
                key3 = (n["layer"], n["sae_type"], n["feature_id"])
                if key3 in include_nodes and key3 not in present:
                    nodes.append(n)
                    present.add(key3)
        keep_keys = {f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}" for n in nodes}
        edges = [e for e in edges
                 if "|".join(str(x) for x in e["source"]) in keep_keys
                 and "|".join(str(x) for x in e["target"]) in keep_keys]

    # Exclude specific nodes
    if exclude_nodes:
        nodes = [n for n in nodes
                 if (n["layer"], n["sae_type"], n["feature_id"]) not in exclude_nodes]
        keep_keys = {f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}" for n in nodes}
        edges = [e for e in edges
                 if "|".join(str(x) for x in e["source"]) in keep_keys
                 and "|".join(str(x) for x in e["target"]) in keep_keys]

    # Re-layout after filtering: group by decomp layer, recompute positions
    if top_n is not None or exclude_nodes:
        from collections import defaultdict
        injection_layer = laid_out.get("config", {}).get("layer", 37)
        type_order = {"transcoder_all": 0, "attn_out_all": 1, "resid_post_all": 2}

        def decomp_layer(n):
            if n["sae_type"] == "resid_post_all" and n["layer"] == injection_layer:
                return n["layer"]
            if n["sae_type"] == "resid_post_all":
                return n["layer"] + 1
            return n["layer"]

        nodes_by_dl = defaultdict(list)
        for n in nodes:
            dl = decomp_layer(n)
            nodes_by_dl[dl].append(n)

        # Sort within each row
        for dl in nodes_by_dl:
            nodes_by_dl[dl].sort(key=lambda n: (type_order.get(n["sae_type"], 9),
                                                 -abs(n.get("root_edge_weight", 0))))

        decomp_layers = sorted(nodes_by_dl.keys())
        decomp_to_y = {dl: i for i, dl in enumerate(decomp_layers)}
        spacing = 2.5
        for dl in decomp_layers:
            row = nodes_by_dl[dl]
            total_width = (len(row) - 1) * spacing
            start_x = -total_width / 2.0
            y = decomp_to_y[dl] * 0.35
            for i, n in enumerate(row):
                n["x"] = start_x + i * spacing
                n["y"] = y
                if x_position_offsets:
                    key3 = (n["layer"], n["sae_type"], n["feature_id"])
                    n["x"] += x_position_offsets.get(key3, 0.0)

    traces = []

    # Node position lookup
    node_pos = {}
    for n in nodes:
        key = f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}"
        node_pos[key] = (n["x"], n["y"])

    # --- Edge traces ---
    max_weight = max((abs(e["weight"]) for e in edges), default=1)
    for e in edges:
        sk = "|".join(str(x) for x in e["source"])
        tk = "|".join(str(x) for x in e["target"])
        if sk not in node_pos or tk not in node_pos:
            continue
        sx, sy = node_pos[sk]
        tx, ty = node_pos[tk]
        color = "rgba(196,78,82,0.35)" if e["weight"] > 0 else "rgba(76,114,176,0.35)"
        width = max(0.8, 10 * abs(e["weight"]) / max_weight)
        traces.append(go.Scatter(
            x=[sx, tx, None], y=[sy, ty, None],
            mode="lines", line=dict(width=width, color=color),
            hoverinfo="skip", showlegend=False,
        ))

    # --- Node traces grouped by SAE type ---
    max_rew = max((abs(n.get("root_edge_weight", 0)) for n in nodes), default=1)
    label_xs, label_ys, label_texts = [], [], []
    title_xs, title_ys, title_texts = [], [], []
    LABEL_OFFSET = 0.06

    GATE_COLOR = "#C44E52"
    CARRIER_COLOR = "#6CCB5F"
    BORDER_WIDTH = 4

    # Collect node info for shape borders (drawn after fig creation)
    node_border_info = []  # list of (x, y, sz_px, dash, border_color)

    type_order = ["transcoder_all", "attn_out_all", "resid_post_all"]
    legend_added = set()

    for sae_type in type_order:
        type_nodes = [n for n in nodes if n["sae_type"] == sae_type and n["layer"] >= 0 and n["feature_id"] >= 0]
        if not type_nodes:
            continue

        color = SAE_COLORS.get(sae_type, "#999")
        legend_name = SAE_ABBREV.get(sae_type, sae_type)

        xs, ys, sizes = [], [], []
        for n in type_nodes:
            key = f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}"
            x, y = node_pos[key]
            xs.append(x); ys.append(y)
            rew = abs(n.get("root_edge_weight", 0))
            sz = max(20, 100 * math.pow(rew / max_rew, 0.5)) if rew else 20
            sizes.append(sz)

            # Determine gate/carrier for border shapes
            if feature_types:
                is_gate = feature_types.get(key) == "gate"
                border_color = GATE_COLOR if is_gate else CARRIER_COLOR
                dash = "solid" if is_gate else "dash"
                node_border_info.append((x, y, sz, dash, border_color))

        show_legend = legend_name not in legend_added
        legend_added.add(legend_name)

        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=color, line=dict(width=0)),
            hoverinfo="skip", name=legend_name, showlegend=show_legend,
        ))

        # Add labels for all nodes in this type
        for n in type_nodes:
            key = f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}"
            x, y = node_pos[key]
            abbr = SAE_ABBREV.get(sae_type, sae_type)

            label_xs.append(x)
            label_ys.append(y + LABEL_OFFSET)
            label_texts.append(f"L{n['layer']} T{n['token_pos']}<br>{abbr} F{n['feature_id']}")

            lk = f"{sae_type}|{n['layer']}|{n['feature_id']}"
            label = label_data.get(lk)
            if label and label.get("title"):
                t = label["title"]
                words = t.split()
                lines, cur = [], ""
                for w in words:
                    if len(cur) + len(w) + 1 > 28 and cur:
                        lines.append(cur)
                        cur = w
                    else:
                        cur = f"{cur} {w}" if cur else w
                if cur:
                    lines.append(cur)
                title_xs.append(x)
                title_ys.append(y - LABEL_OFFSET)
                title_texts.append("<br>".join(f"<i>{l}</i>" for l in lines))

    # Add gate/carrier legend entries as plain line samples.
    if feature_types:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=4, color=GATE_COLOR, dash="solid"),
            hoverinfo="skip", name="Gate",
        ))
        traces.append(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=4, color=CARRIER_COLOR, dash="dash"),
            hoverinfo="skip", name="Carrier",
        ))

    # ID labels above nodes
    if label_xs:
        traces.append(go.Scatter(
            x=label_xs, y=label_ys, mode="text",
            text=label_texts, textposition="top center",
            textfont=dict(size=22, color="#333"),
            hoverinfo="skip", showlegend=False,
        ))

    # Feature title labels below nodes
    if title_xs:
        traces.append(go.Scatter(
            x=title_xs, y=title_ys, mode="text",
            text=title_texts, textposition="bottom center",
            textfont=dict(size=24, color="#444"),
            hoverinfo="skip", showlegend=False,
        ))

    # Compute axis range
    all_xs = [node_pos[f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}"][0] for n in nodes]
    all_ys = [node_pos[f"{n['layer']}|{n['sae_type']}|{n['feature_id']}|{n['token_pos']}"][1] for n in nodes]
    x_range = [min(all_xs) - 1.5, max(all_xs) + 1.5]
    y_range = [min(all_ys) - 0.2, max(all_ys) + 0.2]

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=dict(text=concept, font=dict(size=30, color="#333")),
            font=dict(size=26), x=0.92, xanchor="right", y=0.06, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)", borderwidth=0,
            itemsizing="constant", itemwidth=70),
        width=1400, height=1800,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=x_range),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=y_range),
        plot_bgcolor="white",
    )

    # Draw gate/carrier circle borders as shapes (supports dash)
    if node_border_info:
        # Convert marker size (px) to data coordinates
        fig_w, fig_h = 1400, 1800
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        for x, y, sz_px, dash, border_color in node_border_info:
            # Marker radius in data units (approximate)
            rx = (sz_px / 2 - 2) * x_span / fig_w
            ry = (sz_px / 2 - 2) * y_span / fig_h
            fig.add_shape(
                type="circle", x0=x - rx, y0=y - ry, x1=x + rx, y1=y + ry,
                line=dict(color=border_color, width=BORDER_WIDTH, dash=dash),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), scale=2)
    print(f"Saved {output_path}")


def plot_fig4():
    """Render Bread_layer37 attribution graphs."""
    import json

    with open(DATA_DIR / "fig4_labels.json") as f:
        label_data = json.load(f)

    # Nodes to exclude: (layer, sae_type, feature_id)
    both_exclude = {
        (37, "resid_post_all", 1348), (37, "resid_post_all", 5404),
        (61, "transcoder_all", 62), (61, "attn_out_all", 223),
        (46, "transcoder_all", 10079), (45, "transcoder_all", 7792),
        (49, "transcoder_all", 8757),
        (53, "transcoder_all", 4736),
        (61, "transcoder_all", 3298), (61, "transcoder_all", 169),
        (61, "attn_out_all", 607),
    }
    both_include = {
        (39, "attn_out_all", 419),
        (39, "transcoder_all", 1872),
    }
    bias_exclude = {
        (37, "resid_post_all", 15407),
        (41, "transcoder_all", 14647),
        (38, "attn_out_all", 241),
        (38, "attn_out_all", 551),
        (48, "transcoder_all", 2910),
        (57, "attn_out_all", 3767),
        (58, "attn_out_all", 355),
        (58, "transcoder_all", 2881),
        (58, "transcoder_all", 6770),
        (58, "transcoder_all", 13444),
        (59, "transcoder_all", 5895),
        (59, "transcoder_all", 12376),
        (61, "attn_out_all", 5235),
        (61, "transcoder_all", 13989),
    }
    bias_include = {
        (39, "transcoder_all", 354),
        (39, "transcoder_all", 2199),
        (39, "transcoder_all", 1872),
        (39, "attn_out_all", 14),
        (41, "transcoder_all", 48),
        (41, "transcoder_all", 13645),
        (60, "attn_out_all", 15518),
    }
    bias_x_offsets = {
        (38, "transcoder_all", 1412): 1.0,
    }

    for output_name, data_file, feature_types_file, top_n, exclude, include, x_offsets in [
        ("Bread_layer37.pdf", "fig4_analysis_both_bread_layer37.json", "fig4_analysis_both_feature_types.json",
         20, both_exclude, both_include, None),
        ("Bread_layer37_bias.pdf", "fig4_analysis_bias_bread_layer37.json", "fig4_analysis_bias_feature_types.json",
         20, bias_exclude, bias_include, bias_x_offsets),
    ]:
        with open(DATA_DIR / data_file) as f:
            laid_out = json.load(f)
        ft_path = DATA_DIR / feature_types_file
        feature_types = None
        if ft_path.exists():
            with open(ft_path) as f:
                feature_types = json.load(f)
        out_path = OUTPUT_DIR / output_name
        render_graph_pdf(laid_out, label_data, "Bread", out_path, top_n=top_n,
                         exclude_nodes=exclude, feature_types=feature_types,
                         include_nodes=include, x_position_offsets=x_offsets)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from cached data")
    parser.add_argument("--fig", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Generate only this figure (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    if args.fig is None or args.fig == 1:
        plot_fig1()
    if args.fig is None or args.fig == 2:
        plot_fig2()
    if args.fig is None or args.fig == 3:
        plot_fig3()
    if args.fig is None or args.fig == 4:
        plot_fig4()


if __name__ == "__main__":
    main()
