#!/usr/bin/env python3
"""
Experiment 63: Attribution Graph Construction

Builds multi-hop attribution graphs tracing introspective detection through
SAE/transcoder features, using steering attribution (SA) data from
16_steering_attribution.py.

  Edge weight computation:
    -- GA-weighted edge weights: ew = ∫ GA_root(α) × SA(α) dα
    -- Per-type frac-of-max feature selection with per-hop caps [8,5,3,2]
    -- Only ATTN+TC features traced to next hop (MLP/RESID excluded)

  Backward tracing: feature-targeted SA (loss = target feature activation)
  Forward tracing: JVP from source decoder × GA_root → downstream SA

  Visualization: directed graph rendered as interactive Plotly HTML + PDF

For SA extraction (GA, SG, JVP, ISA, auto-config), see 16_steering_attribution.py.

Paper sections supported:
  - Section 5.3 ("Gate and Evidence Carrier Features")
  - Section 5.4 ("Circuit Analysis")
  - Figure 16 (bread-layer37-single-col): Attribution graph visualization

Model: Primarily Gemma-3 27B with Gemma Scope 2 SAEs/Transcoders (262k, big)
Steering: Layer 37, strength 4.0 (configurable)

Usage:
    # Build attribution graph with backward + forward tracing (requires SA data)
    python 17_attribution_graph.py build-graph --concept Bread --layer 37 --direction both

    # Re-render existing graph from JSON
    python 17_attribution_graph.py visualize --concept Bread --layer 37

    # Full pipeline: extract SA, compute ISA, build graph, visualize
    python 17_attribution_graph.py all --concept Bread --layer 37 --direction both
"""

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from model_utils import ModelWrapper, load_model

# Import SA extraction functions and data types from 16_steering_attribution
from importlib.util import spec_from_file_location, module_from_spec
_sa_spec = spec_from_file_location(
    "steering_attribution",
    str(Path(__file__).resolve().parent / "16_steering_attribution.py"))
_sa_mod = module_from_spec(_sa_spec)
_sa_spec.loader.exec_module(_sa_mod)

# Re-export needed names
FeatureNode = _sa_mod.FeatureNode
FeatureEdge = _sa_mod.FeatureEdge
AttributionGraph = _sa_mod.AttributionGraph
CONCEPT_TOKEN_IDS = _sa_mod.CONCEPT_TOKEN_IDS
DEFAULT_TOKEN_IDS = _sa_mod.DEFAULT_TOKEN_IDS
TRACE_SAE_TYPES = _sa_mod.TRACE_SAE_TYPES
extract_steering_attribution = _sa_mod.extract_steering_attribution
extract_feature_target_sa = _sa_mod.extract_feature_target_sa
extract_forward_sa_for_strength = _sa_mod.extract_forward_sa_for_strength
compute_isa = _sa_mod.compute_isa
load_concept_vectors = _sa_mod.load_concept_vectors
run_auto_config = _sa_mod.run_auto_config
_get_from_list = _sa_mod._get_from_list

# Constants (duplicated for CLI defaults)
DEFAULT_MODEL = _sa_mod.DEFAULT_MODEL
DEFAULT_LAYER = _sa_mod.DEFAULT_LAYER
DEFAULT_STRENGTH = _sa_mod.DEFAULT_STRENGTH
DEFAULT_N_STRENGTHS = _sa_mod.DEFAULT_N_STRENGTHS
DEFAULT_STRENGTH_MAX = _sa_mod.DEFAULT_STRENGTH_MAX
DEFAULT_TRACE_DEPTH = _sa_mod.DEFAULT_TRACE_DEPTH
DEFAULT_MAX_PER_TYPE = _sa_mod.DEFAULT_MAX_PER_TYPE
DEFAULT_FRAC_OF_MAX = _sa_mod.DEFAULT_FRAC_OF_MAX
DEFAULT_EXP21_DIR = _sa_mod.DEFAULT_EXP21_DIR
DEFAULT_OUTPUT_DIR = _sa_mod.DEFAULT_OUTPUT_DIR
DEFAULT_DEVICE = _sa_mod.DEFAULT_DEVICE
DEFAULT_DTYPE = _sa_mod.DEFAULT_DTYPE
DEFAULT_DIRECTION = _sa_mod.DEFAULT_DIRECTION
DEFAULT_SAE_WIDTH = _sa_mod.DEFAULT_SAE_WIDTH
DEFAULT_SAE_L0 = _sa_mod.DEFAULT_SAE_L0

warnings.filterwarnings("ignore", message="Glyph .* missing from font")

# =============================================================================
# Section B: Edge Weight Computation (GA-weighted integration)
# =============================================================================

SAE_ABBREV = {"transcoder_all": "TC", "attn_out_all": "ATTN",
              "mlp_out_all": "MLP", "resid_post_all": "RESID"}


def select_top_per_type(
    df, value_col: str, sae_type_col: str = "sae_type",
    max_per_type: int = 8, frac_of_max: float = 0.10,
):
    """Per SAE type: select features with value > frac_of_max * type_max, capped at max_per_type."""
    import pandas as pd

    pos = df[df[value_col] > 0]
    parts = []
    for st in sorted(pos[sae_type_col].unique()):
        st_df = pos[pos[sae_type_col] == st].sort_values(value_col, ascending=False)
        if st_df.empty:
            continue
        max_val = st_df[value_col].iloc[0]
        above = st_df[st_df[value_col] > frac_of_max * max_val]
        selected = above.head(max_per_type)
        parts.append(selected)
        print(f"    {SAE_ABBREV.get(st, st)}: {len(selected)} features "
              f"(>{frac_of_max:.0%} of max {max_val:.4f}, {len(above)} above, cap {max_per_type})")
    if not parts:
        return pos.iloc[:0]
    return pd.concat(parts).sort_values(value_col, ascending=False)


def _load_curve_from_root_sa(
    sa_dir: Path, trial_nums: List[int], target: FeatureNode, column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a feature's GA or SG curve across strengths from root SA parquets.

    Args:
        column: "gradient_attribution" for GA curve, "steering_grad" for SG curve.

    Returns (strengths_array, values_array) sorted by strength. Empty if not found.
    """
    import pandas as pd

    pairs = []
    for d in sorted(sa_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if not m:
            continue
        s = float(f"{m.group(1)}.{m.group(2)}")
        for t in trial_nums:
            f = d / f"sa_trial{t}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=["layer", "sae_type", "feature_id", "token_pos", column])
            match = df[(df["layer"] == target.layer) & (df["sae_type"] == target.sae_type) &
                       (df["feature_id"] == target.feature_id) & (df["token_pos"] == target.token_pos)]
            if not match.empty:
                pairs.append((s, float(match[column].iloc[0])))
                break
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort()
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def compute_edge_weights(
    sa_dir: Path,
    trial_nums: List[int],
    optimal_strength: float,
    target: Optional[FeatureNode] = None,
    root_sa_dir: Optional[Path] = None,
    weighting_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Dict[Tuple, float]]:
    """Compute weighted edge weights using Simpson's rule.

    For hop-0 (target=None): ew(f) = ∫₀^s* SA(f, α) dα  (weight = 1)
    For backward hop-1+: ew(f→target) = ∫₀^s* GA_root(target, α) · SA(f→target, α) dα
    For forward hop-1+: ew(source→g) = ∫₀^s* SG(source, α) · SA_fwd(g, α) dα

    Args:
        target: Feature node for SA subdirectory lookup (None = root).
        root_sa_dir: Root SA dir for loading GA_root curve (backward hop-1+).
        weighting_curve: Explicit (strengths, values) curve to weight the integrand.
                        If provided, overrides GA_root loading. Use for forward tracing
                        (pass SG curve) or any custom weighting.
    """
    import pandas as pd
    from scipy.integrate import simpson

    is_root = target is None

    # Determine SA directory and parquet prefix
    if is_root:
        scan_dir = sa_dir
        parquet_prefix = "sa_trial"
    else:
        tgt_subdir = f"{target.sae_type}_L{target.layer}_F{target.feature_id}_T{target.token_pos}"
        scan_dir = sa_dir / "feat_sa" / tgt_subdir
        parquet_prefix = "feat_sa_trial"

    if not scan_dir.exists():
        return None

    # Load SA parquets at each strength
    sa_by_strength: Dict[float, pd.DataFrame] = {}
    for d in sorted(scan_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"strength_(\d+)_(\d+)", d.name)
        if not m:
            continue
        s = float(f"{m.group(1)}.{m.group(2)}")
        if s > optimal_strength + 0.1:
            continue
        for t in trial_nums:
            f = d / f"{parquet_prefix}{t}.parquet"
            if f.exists():
                sa_by_strength[s] = pd.read_parquet(f)
                break

    sorted_strengths = sorted(sa_by_strength.keys())
    if len(sorted_strengths) < 2:
        return None

    # Build SA matrix: pivot to (features × strengths)
    KEY_COLS = ["layer", "sae_type", "feature_id", "token_pos"]
    filtered = []
    for s in sorted_strengths:
        df_s = sa_by_strength[s].copy()
        df_s = df_s[df_s["feature_id"] >= 0]
        df_s["_strength"] = s
        filtered.append(df_s[KEY_COLS + ["steering_attribution", "_strength"]])

    if not filtered:
        return None
    all_sa = pd.concat(filtered, ignore_index=True)
    sa_wide = all_sa.pivot_table(
        index=KEY_COLS, columns="_strength", values="steering_attribution",
        aggfunc="first", fill_value=0.0,
    )
    sa_wide = sa_wide.reindex(columns=sorted_strengths, fill_value=0.0)
    sa_matrix = sa_wide.values  # [n_features, n_strengths]
    strengths_arr = np.array(sorted_strengths)

    # Load weighting curve (GA_root for backward, SG for forward, or explicit)
    if weighting_curve is not None:
        w_strengths, w_values = weighting_curve
        if len(w_strengths) >= 2:
            weight_interp = np.interp(sorted_strengths, w_strengths, w_values)
        else:
            weight_interp = np.ones(len(sorted_strengths))
    elif not is_root and root_sa_dir is not None:
        ga_strengths, ga_values = _load_curve_from_root_sa(
            root_sa_dir, trial_nums, target, "gradient_attribution")
        if len(ga_strengths) >= 2:
            weight_interp = np.interp(sorted_strengths, ga_strengths, ga_values)
        else:
            weight_interp = np.ones(len(sorted_strengths))
    else:
        weight_interp = np.ones(len(sorted_strengths))

    # Weighted integration with Simpson's rule (vectorized)
    integrand = sa_matrix * weight_interp[np.newaxis, :]
    edge_weights_arr = simpson(integrand, x=strengths_arr, axis=1)

    # Build result dict
    result: Dict[Tuple, float] = {}
    for idx, (_, row) in enumerate(sa_wide.index.to_frame(index=False).iterrows()):
        key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
        result[key] = float(edge_weights_arr[idx])

    return result


def select_features_from_edge_weights(
    edge_weights: Dict[Tuple, float],
    max_per_type: int,
    frac_of_max: float,
    injection_layer: int,
) -> List[Tuple[Tuple, float]]:
    """Select top features from edge weights using per-type frac-of-max + cap."""
    import pandas as pd

    records = [{"layer": k[0], "sae_type": k[1], "feature_id": k[2],
                "token_pos": k[3], "edge_weight": w}
               for k, w in edge_weights.items()
               if k[0] >= injection_layer and k[2] >= 0]
    if not records:
        return []
    df = pd.DataFrame(records)
    # Filter resid to last layer only
    last_layer = df["layer"].max()
    df = df[~((df["sae_type"] == "resid_post_all") & (df["layer"] != last_layer))]

    top_df = select_top_per_type(df, "edge_weight", "sae_type", max_per_type, frac_of_max)
    result = []
    for _, row in top_df.iterrows():
        key = (int(row["layer"]), row["sae_type"], int(row["feature_id"]), int(row["token_pos"]))
        result.append((key, float(row["edge_weight"])))
    return result


# =============================================================================
# Section B: Graph Construction (backward + forward tracing)
# =============================================================================

def build_attribution_graph(
    model_wrapper: ModelWrapper,
    concept: str,
    concept_vector: torch.Tensor,
    injection_layer: int,
    optimal_strength: float,
    output_dir: Path,
    trial_nums: List[int] = None,
    trace_depth: int = 2,
    n_strengths_feat_sa: int = DEFAULT_N_STRENGTHS,
    max_per_type: List[int] = None,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
    direction: str = DEFAULT_DIRECTION,
    device: str = "cuda",
) -> AttributionGraph:
    """Build multi-hop attribution graph with backward and/or forward tracing.

    Algorithm (from CLAUDE.md):
      For each hop:
        1. Extract SA for all targets/sources (GPU)
        2. Compute GA-weighted edge weights (∫GA×SA dα via Simpson's rule)
        3. Per-type selection with per-hop max_per_type cap
        4. Only ATTN+TC features traced to next hop
    """
    if trial_nums is None:
        trial_nums = [1]
    if max_per_type is None:
        max_per_type = DEFAULT_MAX_PER_TYPE

    root_sa_dir = output_dir  # Root SA parquets are in output_dir/strength_*/

    # ── Hop 0: compute edge weights from root SA ──
    print("\n  Computing hop-0 edge weights (root → features)...")
    hop0_ew = compute_edge_weights(root_sa_dir, trial_nums, optimal_strength)
    if not hop0_ew:
        print("  ERROR: No edge weights computed. Run extract-sa + compute-isa first.")
        return AttributionGraph(nodes={}, edges=[], optimal_strength=optimal_strength)

    mpt0 = _get_from_list(max_per_type, 0)
    hop0_selected = select_features_from_edge_weights(hop0_ew, mpt0, frac_of_max, injection_layer)

    graph = AttributionGraph(nodes={}, edges=[], optimal_strength=optimal_strength)
    root = FeatureNode(layer=-1, sae_type="root", feature_id=-1, token_pos=-1, isa_value=0, hop=-1)
    graph.nodes[root.key] = root

    visited_bwd: Set[Tuple] = set()
    visited_fwd: Set[Tuple] = set()

    for key, ew in hop0_selected:
        node = FeatureNode(key[0], key[1], key[2], key[3], ew, hop=0)
        graph.nodes[key] = node
        visited_bwd.add(key)
        visited_fwd.add(key)
        graph.edges.append(FeatureEdge(source_key=key, target_key=graph.root_key, weight=ew, hop=0))

    feat_sa_strengths = np.linspace(0, optimal_strength, n_strengths_feat_sa).tolist()
    do_backward = direction in ("backward", "both")
    do_forward = direction in ("forward", "both")

    # ── Backward tracing (hop-1+) ──
    if do_backward:
        # Select ATTN+TC trace targets
        bwd_targets = [graph.nodes[k] for k, _ in hop0_selected
                       if graph.nodes[k].sae_type in TRACE_SAE_TYPES
                       and graph.nodes[k].feature_id >= 0
                       and graph.nodes[k].layer > injection_layer]

        for hop in range(trace_depth):
            if not bwd_targets:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  Backward hop {hop+1}: tracing {len(bwd_targets)} targets (cap {mpt}/type)...")

            # Extract feature-targeted SA (uses SG cache from hop-0)
            sg_cache = output_dir / "sg_cache"
            for target in bwd_targets:
                print(f"    {target.short_name()}...")
                for s in tqdm(feat_sa_strengths, desc=f"    SA", leave=False):
                    extract_feature_target_sa(
                        model_wrapper, concept, concept_vector,
                        injection_layer, s,
                        target.layer, target.sae_type, target.feature_id, target.token_pos,
                        output_dir, trial_num=trial_nums[0], device=device,
                        sg_cache_dir=sg_cache,
                    )

            # Compute GA-weighted edge weights and select
            next_targets = []
            for target in bwd_targets:
                ew = compute_edge_weights(
                    output_dir, trial_nums, optimal_strength,
                    target=target, root_sa_dir=root_sa_dir)
                if not ew:
                    continue
                selected = select_features_from_edge_weights(ew, mpt, frac_of_max, injection_layer)
                for key, w in selected:
                    if key not in visited_bwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop+1)
                        graph.nodes[key] = node
                        visited_bwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES and node.layer > injection_layer:
                            next_targets.append(node)
                    graph.edges.append(FeatureEdge(source_key=key, target_key=target.key, weight=w, hop=hop+1))

            bwd_targets = next_targets
            print(f"  Backward hop {hop+1}: {len(next_targets)} new features")

    # ── Forward tracing (hop-1+) ──
    if do_forward:
        max_layer = model_wrapper.n_layers - 1
        fwd_sources = [graph.nodes[k] for k, _ in hop0_selected
                       if graph.nodes[k].sae_type in TRACE_SAE_TYPES
                       and graph.nodes[k].feature_id >= 0
                       and graph.nodes[k].layer < max_layer]

        for hop in range(trace_depth):
            if not fwd_sources:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  Forward hop {hop+1}: tracing {len(fwd_sources)} sources (cap {mpt}/type)...")

            # Extract forward SA for each source
            for source in fwd_sources:
                print(f"    {source.short_name()}...")
                for s in tqdm(feat_sa_strengths, desc=f"    fwd SA", leave=False):
                    extract_forward_sa_for_strength(
                        model_wrapper, concept, concept_vector,
                        injection_layer, s,
                        source.layer, source.sae_type, source.feature_id, source.token_pos,
                        output_dir, root_sa_dir=root_sa_dir,
                        trial_num=trial_nums[0], device=device,
                    )

            # Compute SG-weighted edge weights: ∫ SG(source,α) × SA_fwd(α) dα
            next_sources = []
            for source in fwd_sources:
                # Load SG curve for source from root SA
                sg_curve = _load_curve_from_root_sa(
                    root_sa_dir, trial_nums, source, "steering_grad")

                # Forward parquets use fwd_ prefix in subdir
                fwd_node = FeatureNode(source.layer, f"fwd_{source.sae_type}",
                                       source.feature_id, source.token_pos, 0, 0)
                ew = compute_edge_weights(
                    output_dir, trial_nums, optimal_strength,
                    target=fwd_node, weighting_curve=sg_curve)

                if not ew:
                    continue

                selected = select_features_from_edge_weights(
                    ew, mpt, frac_of_max, source.layer + 1)
                for key, w in selected:
                    if key not in visited_fwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop+1)
                        if key not in graph.nodes:
                            graph.nodes[key] = node
                        visited_fwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES and node.layer < max_layer:
                            next_sources.append(node)
                    graph.edges.append(FeatureEdge(
                        source_key=source.key, target_key=key, weight=w, hop=hop+1))

            fwd_sources = next_sources
            print(f"  Forward hop {hop+1}: {len(next_sources)} new features")

    # Dedup edges by (source, target) keeping max weight
    seen_edges: Dict[Tuple, FeatureEdge] = {}
    for e in graph.edges:
        edge_key = (e.source_key, e.target_key)
        if edge_key not in seen_edges or abs(e.weight) > abs(seen_edges[edge_key].weight):
            seen_edges[edge_key] = e
    graph.edges = list(seen_edges.values())

    print(f"\n  Graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


# =============================================================================
# Visualization
# =============================================================================

def export_graph_json(graph: AttributionGraph, path: Path) -> None:
    data = {
        "optimal_strength": graph.optimal_strength,
        "nodes": [{"layer": n.layer, "sae_type": n.sae_type, "feature_id": n.feature_id,
                    "token_pos": n.token_pos, "isa_value": n.isa_value, "hop": n.hop,
                    "label": n.label, "short_name": n.short_name()}
                   for n in graph.nodes.values()],
        "edges": [{"source": list(e.source_key), "target": list(e.target_key),
                    "weight": e.weight, "hop": e.hop}
                   for e in graph.edges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Graph JSON: {path}")


def render_pdf_from_html(html_path: Path, pdf_path: Path) -> None:
    """Generate PDF from an HTML file. Tries playwright first, falls back to weasyprint."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_timeout(2000)
            page.pdf(path=str(pdf_path), format="A3", landscape=True, print_background=True)
            browser.close()
        print(f"  PDF (playwright): {pdf_path}")
        return
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["chromium", "--headless", "--disable-gpu", "--no-sandbox",
             f"--print-to-pdf={pdf_path}", str(html_path.resolve())],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and pdf_path.exists():
            print(f"  PDF (chromium): {pdf_path}")
            return
    except Exception:
        pass

    print(f"  PDF generation skipped (install playwright: pip install playwright && playwright install chromium)")


def render_interactive(graph: AttributionGraph, path: Path) -> None:
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        print("  networkx/plotly not installed, skipping HTML")
        return

    COLORS = {"transcoder_all": "#4C72B0", "attn_out_all": "#DD8452",
              "mlp_out_all": "#55A868", "resid_post_all": "#C44E52", "root": "#333333"}

    G = nx.DiGraph()
    feature_nodes = {k: n for k, n in graph.nodes.items() if n.layer >= 0 and n.isa_value > 0}
    layers = sorted(set(n.layer for n in feature_nodes.values()))
    layer_to_y = {lv: i for i, lv in enumerate(layers)}

    layer_counts = defaultdict(int)
    for n in feature_nodes.values():
        layer_counts[n.layer] += 1

    layer_idx = defaultdict(int)
    for key, node in sorted(feature_nodes.items(), key=lambda x: abs(x[1].isa_value), reverse=True):
        idx = layer_idx[node.layer]
        layer_idx[node.layer] += 1
        x = (idx - layer_counts[node.layer] / 2.0) * 5.0
        y = layer_to_y.get(node.layer, 0) * 0.6
        G.add_node(key, pos=(x, y), **asdict(node))

    root_y = (max(layer_to_y.values()) + 1) * 0.6 if layer_to_y else 0.6
    G.add_node(graph.root_key, pos=(0, root_y))

    visible = set(G.nodes)
    vis_edges = [e for e in graph.edges if e.source_key in visible and e.target_key in visible]
    for e in vis_edges:
        G.add_edge(e.source_key, e.target_key, weight=e.weight)

    pos = nx.get_node_attributes(G, "pos")
    max_w = max((abs(e.weight) for e in vis_edges), default=1)

    edge_traces = []
    for e in vis_edges:
        if e.source_key not in pos or e.target_key not in pos:
            continue
        x0, y0 = pos[e.source_key]
        x1, y1 = pos[e.target_key]
        c = "rgba(196,78,82,0.6)" if e.weight > 0 else "rgba(76,114,176,0.6)"
        w = 1 + 4 * abs(e.weight) / max_w
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=w, color=c), hoverinfo="text",
            text=f"weight={e.weight:.4f}", showlegend=False))

    max_isa = max((abs(n.isa_value) for n in feature_nodes.values()), default=1)
    node_traces = {}
    for key in visible:
        node = graph.nodes.get(key)
        if node is None or key not in pos:
            continue
        x, y = pos[key]
        st = node.sae_type
        if st not in node_traces:
            node_traces[st] = {"x": [], "y": [], "text": [], "size": [], "color": COLORS.get(st, "#999")}
        size = 10 + 20 * (abs(node.isa_value) / max_isa) if node.layer >= 0 else 15
        name = node.short_name() if node.layer >= 0 else "dL/ds"
        hover = f"{name}<br>ISA={node.isa_value:.4f}"
        node_traces[st]["x"].append(x)
        node_traces[st]["y"].append(y)
        node_traces[st]["text"].append(hover)
        node_traces[st]["size"].append(size)

    fig = go.Figure()
    for t in edge_traces:
        fig.add_trace(t)
    for st, d in node_traces.items():
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="markers+text",
            marker=dict(size=d["size"], color=d["color"], line=dict(width=1, color="white")),
            text=[t.split("<br>")[0] for t in d["text"]], textposition="top center",
            textfont=dict(size=8), hovertext=d["text"], hoverinfo="text", name=st))

    fig.update_layout(
        title=f"Attribution Graph (strength={graph.optimal_strength:.2f})",
        showlegend=True, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white", width=1200, height=800 + len(layers) * 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    print(f"  Interactive HTML: {path}")


def write_graph_summary(graph: AttributionGraph, path: Path, concept: str, layer: int) -> None:
    lines = [
        f"Attribution Graph Summary: {concept} layer {layer}",
        f"Optimal strength: {graph.optimal_strength:.4f}",
        f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}",
        "", "=" * 60, "TOP FEATURES (hop 0 -> logit objective)", "=" * 60,
    ]
    hop0 = sorted([e for e in graph.edges if e.target_key == graph.root_key],
                   key=lambda e: abs(e.weight), reverse=True)
    for e in hop0:
        n = graph.nodes.get(e.source_key)
        if n:
            lines.append(f"  {n.short_name()}: ISA={e.weight:+.4f}")

    max_hop = max((e.hop for e in graph.edges), default=0)
    for hop in range(1, max_hop + 1):
        lines += ["", "=" * 60, f"HOP {hop} EDGES", "=" * 60]
        for e in sorted([e for e in graph.edges if e.hop == hop], key=lambda e: abs(e.weight), reverse=True):
            s = graph.nodes.get(e.source_key)
            t = graph.nodes.get(e.target_key)
            lines.append(f"  {s.short_name() if s else '?'} -> {t.short_name() if t else '?'}: w={e.weight:+.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Summary: {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 63: Attribution Graph Construction"
    )
    subparsers = parser.add_subparsers(dest="phase")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    common.add_argument("--concept", type=str, required=True)
    common.add_argument("-l", "--layer", type=int, default=DEFAULT_LAYER)
    common.add_argument("--trial-num", type=int, default=1)
    common.add_argument("--exp21-dir", type=str, default=DEFAULT_EXP21_DIR)
    common.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE)
    common.add_argument("-dt", "--dtype", type=str, default=DEFAULT_DTYPE)
    common.add_argument("-q", "--quantization", type=str, default=None, choices=["8bit", "4bit"])
    common.add_argument("--sae-width", type=str, default=DEFAULT_SAE_WIDTH)
    common.add_argument("--sae-l0", type=str, default=DEFAULT_SAE_L0)
    common.add_argument("--pos-token-id", type=int, default=None)
    common.add_argument("--neg-token-id", type=int, default=None)

    # auto-config
    p0 = subparsers.add_parser("auto-config", parents=[common], help="Auto-detect pos/neg tokens and optimal strength")
    p0.add_argument("--n-ll-strengths", type=int, default=51, help="Logit lens strength grid size")
    p0.add_argument("--n-scan-strengths", type=int, default=25, help="Behavioral scan grid size")
    p0.add_argument("--n-scan-trials", type=int, default=3, help="Trials per strength for LLM judge")
    p0.add_argument("--strength-max", type=float, default=DEFAULT_STRENGTH_MAX)

    # extract-sa
    p1 = subparsers.add_parser("extract-sa", parents=[common], help="Extract SA at one strength")
    p1.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)

    # compute-isa
    subparsers.add_parser("compute-isa", parents=[common], help="Compute ISA from SA data")

    # build-graph
    p3 = subparsers.add_parser("build-graph", parents=[common], help="Build attribution graph")
    p3.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH, help="Optimal strength")
    p3.add_argument("--trace-depth", type=int, default=DEFAULT_TRACE_DEPTH)
    p3.add_argument("--max-per-type", type=int, nargs="+", default=DEFAULT_MAX_PER_TYPE,
                     help="Per-hop max features per SAE type (e.g. 8 5 3 2)")
    p3.add_argument("--frac-of-max", type=float, default=DEFAULT_FRAC_OF_MAX)
    p3.add_argument("--n-strengths-feat-sa", type=int, default=DEFAULT_N_STRENGTHS)
    p3.add_argument("--direction", type=str, default=DEFAULT_DIRECTION,
                     choices=["backward", "forward", "both"])

    # visualize
    p4 = subparsers.add_parser("visualize", parents=[common], help="Render existing graph")
    p4.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH)

    # all
    p5 = subparsers.add_parser("all", parents=[common], help="Full pipeline")
    p5.add_argument("-s", "--strength", type=float, default=DEFAULT_STRENGTH, help="Optimal strength")
    p5.add_argument("--n-strengths", type=int, default=DEFAULT_N_STRENGTHS)
    p5.add_argument("--strength-max", type=float, default=DEFAULT_STRENGTH_MAX)
    p5.add_argument("--trace-depth", type=int, default=DEFAULT_TRACE_DEPTH)
    p5.add_argument("--max-per-type", type=int, nargs="+", default=DEFAULT_MAX_PER_TYPE,
                     help="Per-hop max features per SAE type (e.g. 8 5 3 2)")
    p5.add_argument("--frac-of-max", type=float, default=DEFAULT_FRAC_OF_MAX)
    p5.add_argument("--direction", type=str, default=DEFAULT_DIRECTION,
                     choices=["backward", "forward", "both"])
    p5.add_argument("--n-strengths-feat-sa", type=int, default=11)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.phase is None:
        print("Specify a phase: build-graph, visualize, all (or extract-sa, compute-isa, auto-config)")
        return

    concept = args.concept
    layer = args.layer
    trial_num = args.trial_num

    # Resolve token IDs
    tokens = CONCEPT_TOKEN_IDS.get(concept, DEFAULT_TOKEN_IDS)
    pos_id = args.pos_token_id if args.pos_token_id is not None else tokens["pos"]
    neg_id = args.neg_token_id if args.neg_token_id is not None else tokens["neg"]

    # Output dir
    run_name = f"{concept.replace(' ', '_')}_layer{layer}"
    base_out = Path(args.output_dir) / args.model / run_name
    base_out.mkdir(parents=True, exist_ok=True)

    if args.phase == "auto-config":
        run_auto_config(args)
        return

    if args.phase == "extract-sa":
        print(f"Extracting SA for {concept} layer {layer} strength {args.strength}")
        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        extract_steering_attribution(
            mw, concept, vectors[concept], layer, args.strength,
            base_out, trial_num, pos_id, neg_id, args.sae_width, args.sae_l0, args.device,
            sg_cache_dir=base_out / "sg_cache")

    elif args.phase == "compute-isa":
        print(f"Computing ISA for {concept} layer {layer}")
        compute_isa(base_out, [trial_num])

    elif args.phase == "build-graph":
        print(f"Building attribution graph for {concept} layer {layer}")
        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        graph = build_attribution_graph(
            mw, concept, vectors[concept], layer, args.strength,
            base_out, [trial_num], args.trace_depth, args.n_strengths_feat_sa,
            args.max_per_type, args.frac_of_max, args.direction, args.device)
        graph_dir = base_out / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graph_json(graph, graph_dir / "attribution_graph.json")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        render_pdf_from_html(graph_dir / "attribution_graph.html", graph_dir / "attribution_graph.pdf")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

    elif args.phase == "visualize":
        graph_json = base_out / "graphs" / "attribution_graph.json"
        if not graph_json.exists():
            print(f"  No graph found at {graph_json}. Run build-graph first.")
            return
        with open(graph_json) as f:
            data = json.load(f)
        graph = AttributionGraph(nodes={}, edges=[], optimal_strength=data.get("optimal_strength", args.strength))
        for nd in data["nodes"]:
            node = FeatureNode(nd["layer"], nd["sae_type"], nd["feature_id"],
                               nd["token_pos"], nd["isa_value"], nd.get("hop", 0), nd.get("label"))
            graph.nodes[node.key] = node
        for ed in data["edges"]:
            graph.edges.append(FeatureEdge(
                tuple(ed["source"]), tuple(ed["target"]), ed["weight"], ed.get("hop", 0)))
        graph_dir = base_out / "graphs"
        render_pdf_from_html(graph_dir / "attribution_graph.html", graph_dir / "attribution_graph.pdf")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

    elif args.phase == "all":
        print("=" * 70)
        print(f"FULL ATTRIBUTION GRAPH PIPELINE: {concept} layer {layer}")
        print("=" * 70)

        mw = load_model(args.model, device=args.device, dtype=args.dtype, quantization=args.quantization)
        vectors = load_concept_vectors(args.exp21_dir, args.model, [concept], layer)
        if concept not in vectors:
            print(f"  ERROR: No vector for {concept}")
            return
        vec = vectors[concept]

        # Step 1: Extract SA at multiple strengths (saves SG cache for hop-1+ reuse)
        sg_cache = base_out / "sg_cache"
        strengths = np.linspace(0, args.strength_max, args.n_strengths).tolist()
        print(f"\nStep 1: Extracting SA at {len(strengths)} strengths (SG cached to {sg_cache})...")
        for s in tqdm(strengths, desc="SA extraction"):
            extract_steering_attribution(
                mw, concept, vec, layer, s, base_out, trial_num,
                pos_id, neg_id, args.sae_width, args.sae_l0, args.device,
                sg_cache_dir=sg_cache)

        # Step 2: Compute ISA
        print("\nStep 2: Computing ISA...")
        compute_isa(base_out, [trial_num])

        # Step 3: Build attribution graph
        print("\nStep 3: Building attribution graph...")
        graph = build_attribution_graph(
            mw, concept, vec, layer, args.strength, base_out,
            [trial_num], args.trace_depth, args.n_strengths_feat_sa,
            args.max_per_type, args.frac_of_max, args.direction, args.device)

        # Step 4: Visualize
        print("\nStep 4: Visualization...")
        graph_dir = base_out / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graph_json(graph, graph_dir / "attribution_graph.json")
        render_interactive(graph, graph_dir / "attribution_graph.html")
        render_pdf_from_html(graph_dir / "attribution_graph.html", graph_dir / "attribution_graph.pdf")
        write_graph_summary(graph, graph_dir / "graph_summary.txt", concept, layer)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print(f"  Output: {base_out}")
        print("=" * 70)


if __name__ == "__main__":
    main()
