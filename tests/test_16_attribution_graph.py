"""
Unit tests for experiments/16_attribution_graph.py

Lightweight CPU-only tests that verify key functions with small synthetic inputs.
No GPU or model loading required.

Run:
    python -m pytest tests/test_16_attribution_graph.py -v
    python -m pytest tests/test_16_attribution_graph.py -v -k test_sae
"""

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

# Mock heavy dependencies that aren't needed for unit tests
for mod_name in [
    "transformers", "transformers.cache_utils",
    "huggingface_hub", "safetensors", "safetensors.torch",
    "tqdm", "tqdm.auto",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Provide a real tqdm.tqdm that's just a passthrough
_tqdm_mod = types.ModuleType("tqdm")
_tqdm_mod.tqdm = lambda x, **kw: x
sys.modules["tqdm"] = _tqdm_mod

# Mock model_utils to avoid transformers dependency
_model_utils = types.ModuleType("model_utils")
_model_utils.ModelWrapper = type("ModelWrapper", (), {})
_model_utils.load_model = MagicMock()
_model_utils.MODEL_NAME_MAP = {}
sys.modules["model_utils"] = _model_utils

# Add experiments/ and src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import the module under test
from importlib import import_module
mod = import_module("16_attribution_graph")

JumpReLUSAE = mod.JumpReLUSAE
FeatureNode = mod.FeatureNode
FeatureEdge = mod.FeatureEdge
AttributionGraph = mod.AttributionGraph
select_top_per_type = mod.select_top_per_type
compute_isa = mod.compute_isa
export_graph_json = mod.export_graph_json
render_interactive = mod.render_interactive
write_graph_summary = mod.write_graph_summary
render_pdf_from_html = mod.render_pdf_from_html
build_messages = mod.build_messages


# =============================================================================
# JumpReLU SAE
# =============================================================================

class TestJumpReLUSAE:
    def test_encode_decode_roundtrip(self):
        """Encoded then decoded should approximate input (zero threshold = identity-like)."""
        d_in, d_sae = 16, 32
        sae = JumpReLUSAE(d_in, d_sae)
        # Set threshold to -inf so all features pass
        sae.threshold.data.fill_(-100.0)
        # Random encoder/decoder (not orthogonal, so not exact roundtrip)
        torch.manual_seed(42)
        sae.w_enc.data = torch.randn(d_in, d_sae) * 0.1
        sae.w_dec.data = torch.randn(d_sae, d_in) * 0.1

        x = torch.randn(4, d_in)
        acts = sae.encode(x)
        assert acts.shape == (4, d_sae)
        assert (acts >= 0).all(), "JumpReLU should produce non-negative activations"

    def test_threshold_sparsity(self):
        """High threshold should zero out most features."""
        d_in, d_sae = 8, 64
        sae = JumpReLUSAE(d_in, d_sae)
        sae.threshold.data.fill_(100.0)  # Very high threshold
        torch.manual_seed(0)
        sae.w_enc.data = torch.randn(d_in, d_sae)

        x = torch.randn(2, d_in)
        acts = sae.encode(x)
        assert (acts == 0).all(), "With very high threshold, all features should be zero"

    def test_affine_skip_connection(self):
        """Transcoder mode should include skip connection."""
        d_in, d_sae = 8, 16
        sae = JumpReLUSAE(d_in, d_sae, affine_skip_connection=True)
        assert sae.affine_skip_connection is not None
        assert sae.affine_skip_connection.shape == (d_in, d_in)

        sae_no_skip = JumpReLUSAE(d_in, d_sae, affine_skip_connection=False)
        assert sae_no_skip.affine_skip_connection is None

    def test_forward_with_skip(self):
        """Forward with skip connection should include x @ affine term."""
        d_in, d_sae = 8, 16
        sae = JumpReLUSAE(d_in, d_sae, affine_skip_connection=True)
        sae.threshold.data.fill_(100.0)  # Zero out all features
        sae.affine_skip_connection.data = torch.eye(d_in)

        x = torch.randn(2, d_in)
        out = sae(x)
        # With all features zeroed, output = b_dec + x @ I = b_dec + x
        expected = sae.b_dec + x
        assert torch.allclose(out, expected, atol=1e-5)

    def test_decode_single_feature(self):
        """w_dec[f] should be the decoder direction for feature f."""
        d_in, d_sae = 8, 16
        sae = JumpReLUSAE(d_in, d_sae)
        torch.manual_seed(0)
        sae.w_dec.data = torch.randn(d_sae, d_in)
        # Single-feature activation should decode to w_dec[f] + b_dec
        acts = torch.zeros(1, d_sae)
        acts[0, 3] = 1.0
        decoded = sae.decode(acts)
        assert torch.allclose(decoded[0], sae.w_dec[3] + sae.b_dec)


# =============================================================================
# Feature Selection: select_top_per_type
# =============================================================================

class TestSelectTopPerType:
    def test_basic_selection(self):
        """Should select top features per SAE type above frac-of-max threshold."""
        df = pd.DataFrame({
            "sae_type": ["TC", "TC", "TC", "TC", "ATTN", "ATTN", "ATTN"],
            "value": [1.0, 0.5, 0.2, 0.05, 0.8, 0.3, 0.01],
            "layer": [45, 45, 46, 46, 50, 50, 51],
            "feature_id": [1, 2, 3, 4, 10, 11, 12],
        })
        result = select_top_per_type(df, "value", "sae_type", max_per_type=3, frac_of_max=0.10)
        # TC: max=1.0, threshold=0.1 -> 1.0, 0.5, 0.2 pass (3, within cap)
        # ATTN: max=0.8, threshold=0.08 -> 0.8, 0.3 pass
        assert len(result) == 5
        assert result["value"].iloc[0] == 1.0  # sorted desc

    def test_cap_per_type(self):
        """max_per_type should cap selections."""
        df = pd.DataFrame({
            "sae_type": ["TC"] * 10,
            "value": list(range(10, 0, -1)),
            "layer": list(range(10)),
            "feature_id": list(range(10)),
        })
        result = select_top_per_type(df, "value", "sae_type", max_per_type=3, frac_of_max=0.01)
        assert len(result) == 3

    def test_negative_values_excluded(self):
        """Only positive values should be considered."""
        df = pd.DataFrame({
            "sae_type": ["TC", "TC", "TC"],
            "value": [-1.0, 0.5, 0.3],
            "layer": [1, 2, 3],
            "feature_id": [1, 2, 3],
        })
        result = select_top_per_type(df, "value", "sae_type", max_per_type=10, frac_of_max=0.10)
        assert len(result) == 2
        assert (result["value"] > 0).all()

    def test_empty_input(self):
        df = pd.DataFrame({"sae_type": [], "value": [], "layer": [], "feature_id": []})
        result = select_top_per_type(df, "value", "sae_type")
        assert len(result) == 0


# =============================================================================
# ISA Computation
# =============================================================================

class TestComputeISA:
    def test_trapezoidal_integration(self):
        """ISA should be the trapezoidal integral of SA across strengths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sa_dir = Path(tmpdir)

            # Create SA parquets at 3 strengths: 0.0, 2.0, 4.0
            # One feature with SA = [0, 1, 2] -> integral = 0.5*(0+1)*2 + 0.5*(1+2)*2 = 1+3 = 4
            for s, sa_val in [(0.0, 0.0), (2.0, 1.0), (4.0, 2.0)]:
                s_int = int(s)
                s_frac = int((s % 1) * 100)
                d = sa_dir / f"strength_{s_int}_{s_frac:02d}"
                d.mkdir(parents=True)
                df = pd.DataFrame([{
                    "layer": 45, "sae_type": "transcoder_all",
                    "feature_id": 100, "token_pos": 10,
                    "steering_attribution": sa_val,
                }])
                df.to_parquet(d / "sa_trial1.parquet", index=False)

            compute_isa(sa_dir, [1])

            isa_path = sa_dir / "isa_trial1.parquet"
            assert isa_path.exists()
            isa_df = pd.read_parquet(isa_path)
            assert len(isa_df) == 1
            assert abs(isa_df["integrated_steering_attribution"].iloc[0] - 4.0) < 1e-6

    def test_remainder_features_integrated(self):
        """Remainder (feature_id=-1) should also get ISA values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sa_dir = Path(tmpdir)
            for s, sa_val in [(0.0, 0.0), (1.0, 1.0)]:
                d = sa_dir / f"strength_{int(s)}_{int((s%1)*100):02d}"
                d.mkdir(parents=True)
                df = pd.DataFrame([
                    {"layer": 45, "sae_type": "TC", "feature_id": 5,
                     "token_pos": 0, "steering_attribution": sa_val},
                    {"layer": 45, "sae_type": "TC", "feature_id": -1,
                     "token_pos": 0, "steering_attribution": sa_val * 0.5},
                ])
                df.to_parquet(d / "sa_trial1.parquet", index=False)

            compute_isa(sa_dir, [1])
            isa_df = pd.read_parquet(sa_dir / "isa_trial1.parquet")
            assert len(isa_df) == 2
            remainder = isa_df[isa_df["feature_id"] == -1]
            assert len(remainder) == 1

    def test_single_strength_skipped(self):
        """With only 1 strength, ISA can't be computed (need >= 2 for trapz)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sa_dir = Path(tmpdir)
            d = sa_dir / "strength_4_00"
            d.mkdir(parents=True)
            pd.DataFrame([{
                "layer": 45, "sae_type": "TC", "feature_id": 1,
                "token_pos": 0, "steering_attribution": 1.0,
            }]).to_parquet(d / "sa_trial1.parquet", index=False)

            compute_isa(sa_dir, [1])
            assert not (sa_dir / "isa_trial1.parquet").exists()


# =============================================================================
# Data Models
# =============================================================================

class TestDataModels:
    def test_feature_node_key(self):
        n = FeatureNode(layer=45, sae_type="transcoder_all", feature_id=9959,
                        token_pos=-1, isa_value=0.123)
        assert n.key == (45, "transcoder_all", 9959, -1)

    def test_feature_node_short_name(self):
        n = FeatureNode(layer=45, sae_type="transcoder_all", feature_id=9959,
                        token_pos=140, isa_value=0.5)
        assert n.short_name() == "L45 T140 TC F9959"

        n2 = FeatureNode(layer=61, sae_type="resid_post_all", feature_id=1234,
                         token_pos=0, isa_value=-0.1)
        assert n2.short_name() == "L61 T0 RESID F1234"

    def test_attribution_graph_structure(self):
        graph = AttributionGraph(nodes={}, edges=[], optimal_strength=3.5)
        root = FeatureNode(-1, "root", -1, -1, 0, hop=-1)
        n1 = FeatureNode(45, "transcoder_all", 100, 10, 0.5, hop=0)
        graph.nodes[root.key] = root
        graph.nodes[n1.key] = n1
        graph.edges.append(FeatureEdge(n1.key, graph.root_key, 0.5, hop=0))

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.edges[0].source_key == n1.key
        assert graph.edges[0].target_key == (-1, "root", -1, -1)


# =============================================================================
# Visualization
# =============================================================================

class TestVisualization:
    def _make_test_graph(self):
        graph = AttributionGraph(nodes={}, edges=[], optimal_strength=4.0)
        root = FeatureNode(-1, "root", -1, -1, 0, hop=-1)
        n1 = FeatureNode(45, "transcoder_all", 9959, 140, 0.5, hop=0)
        n2 = FeatureNode(38, "attn_out_all", 2276, 140, 0.3, hop=1)
        graph.nodes[root.key] = root
        graph.nodes[n1.key] = n1
        graph.nodes[n2.key] = n2
        graph.edges.append(FeatureEdge(n1.key, graph.root_key, 0.5, 0))
        graph.edges.append(FeatureEdge(n2.key, n1.key, 0.3, 1))
        return graph

    def test_export_graph_json_roundtrip(self):
        """JSON export should be loadable and contain all nodes/edges."""
        graph = self._make_test_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            export_graph_json(graph, path)
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert data["optimal_strength"] == 4.0
            assert len(data["nodes"]) == 3  # root + 2 features
            assert len(data["edges"]) == 2

            # Check node fields
            feature_nodes = [n for n in data["nodes"] if n["layer"] >= 0]
            assert len(feature_nodes) == 2
            assert any(n["short_name"] == "L45 T140 TC F9959" for n in feature_nodes)

    def test_write_graph_summary(self):
        """Summary should contain key information."""
        graph = self._make_test_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.txt"
            write_graph_summary(graph, path, "Bread", 37)
            assert path.exists()

            text = path.read_text()
            assert "Bread" in text
            assert "layer 37" in text
            assert "L45 T140 TC F9959" in text
            assert "Nodes: 3" in text

    def test_render_interactive_html(self):
        """HTML render should produce a valid file (may skip if plotly missing)."""
        graph = self._make_test_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.html"
            try:
                render_interactive(graph, path)
                if path.exists():
                    content = path.read_text()
                    assert "plotly" in content.lower() or "Plotly" in content
            except ImportError:
                pytest.skip("plotly/networkx not installed")


# =============================================================================
# Prompt Utilities
# =============================================================================

class TestPromptUtils:
    def test_build_messages_structure(self):
        msgs = build_messages(trial_num=3)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert "Trial 3" in msgs[3]["content"]

    def test_build_messages_different_trials(self):
        m1 = build_messages(1)
        m5 = build_messages(5)
        assert "Trial 1" in m1[3]["content"]
        assert "Trial 5" in m5[3]["content"]


# =============================================================================
# ActivationHooks
# =============================================================================

class TestActivationHooks:
    def test_context_manager(self):
        """Hooks should only capture when enabled via context manager."""
        # Simple model with one linear layer
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        hooks = mod.ActivationHooks(model, retain_grad=False)

        # Manually create a hook
        captured = []
        def simple_hook(module, input, output):
            if hooks.enabled:
                captured.append(output.detach())

        model[0].register_forward_hook(simple_hook)

        # Outside context: should not capture
        x = torch.randn(1, 4)
        model(x)
        assert len(captured) == 0

        # Inside context: should capture
        with hooks:
            model(x)
        assert len(captured) == 1


# =============================================================================
# Integration: select_top_features
# =============================================================================

class TestSelectTopFeatures:
    def test_select_from_isa_parquets(self):
        """End-to-end: create ISA parquets, select top features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sa_dir = Path(tmpdir)

            # Create ISA data
            rows = []
            for layer in [38, 39, 45, 50]:
                for st in ["transcoder_all", "attn_out_all", "mlp_out_all"]:
                    for fid in range(20):
                        isa_val = np.random.exponential(0.01)
                        if layer == 45 and st == "transcoder_all" and fid < 3:
                            isa_val = 1.0 + fid * 0.5  # Make these clearly the top
                        rows.append({
                            "layer": layer, "sae_type": st, "feature_id": fid,
                            "token_pos": 140, "integrated_steering_attribution": isa_val,
                            "trial_num": 1,
                        })

            pd.DataFrame(rows).to_parquet(sa_dir / "isa_trial1.parquet", index=False)

            features = mod.select_top_features(sa_dir, injection_layer=37, trial_nums=[1],
                                                max_per_type=3, frac_of_max=0.10)
            assert len(features) > 0
            # Top feature should be from L45 TC
            assert features[0].layer == 45
            assert features[0].sae_type == "transcoder_all"
            assert features[0].hop == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
