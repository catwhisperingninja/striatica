# striatica/tests/test_circuits.py
"""Tests for circuit extraction functions."""

import json
from pathlib import Path

import pytest

from pipeline.circuits import extract_similarity_circuit


def _make_features_jsonl(tmp_path: Path, num_features: int = 100) -> Path:
    """Create a synthetic features JSONL for testing similarity circuits."""
    path = tmp_path / "features.jsonl"
    with open(path, "w") as f:
        for i in range(num_features):
            # Each feature's top-k neighbors are the next 10 features (wrapping)
            neighbors = [(i + j) % num_features for j in range(11)]  # includes self
            sims = [round(1.0 - j * 0.08, 4) for j in range(11)]
            record = {
                "index": i,
                "maxActApprox": float(num_features - i),
                "frac_nonzero": 0.01,
                "topkCosSimIndices": neighbors,
                "topkCosSimValues": sims,
                "pos_str": [f"token_{i}"],
                "neg_str": [],
            }
            f.write(json.dumps(record) + "\n")
    return path


class TestSimilarityCircuit:
    def test_basic_structure(self, tmp_path: Path) -> None:
        """Output matches CircuitData schema."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=3)

        assert "name" in result
        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)

    def test_seed_is_source(self, tmp_path: Path) -> None:
        """Seed feature should have role 'source'."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=5, depth=2, top_k_neighbors=3)

        seed_nodes = [n for n in result["nodes"] if n["featureIndex"] == 5]
        assert len(seed_nodes) == 1
        assert seed_nodes[0]["role"] == "source"
        assert seed_nodes[0]["activation"] == 1.0

    def test_roles_by_depth(self, tmp_path: Path) -> None:
        """Depth-0 = source, depth-1 = intermediate, depth-2 = sink."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=3)

        roles = {n["featureIndex"]: n["role"] for n in result["nodes"]}
        assert roles[0] == "source"
        # Depth-1 neighbors of feature 0 are 1,2,3 (after filtering self)
        for idx in [1, 2, 3]:
            if idx in roles:
                assert roles[idx] == "intermediate"

    def test_edge_weights_valid(self, tmp_path: Path) -> None:
        """All edge weights should be between 0 and 1."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=5)

        for edge in result["edges"]:
            assert 0 <= edge["weight"] <= 1, f"Invalid weight: {edge['weight']}"
            assert "source" in edge
            assert "target" in edge

    def test_feature_indices_valid(self, tmp_path: Path) -> None:
        """All feature indices should be in valid range."""
        n = 100
        jsonl = _make_features_jsonl(tmp_path, num_features=n)
        result = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=5)

        for node in result["nodes"]:
            assert 0 <= node["featureIndex"] < n

    def test_invalid_seed_raises(self, tmp_path: Path) -> None:
        """Invalid seed feature should raise ValueError."""
        jsonl = _make_features_jsonl(tmp_path, num_features=10)
        with pytest.raises(ValueError, match="not found"):
            extract_similarity_circuit(jsonl, seed_feature=999)

    def test_depth_1_fewer_nodes(self, tmp_path: Path) -> None:
        """Depth 1 should produce fewer nodes than depth 2."""
        jsonl = _make_features_jsonl(tmp_path)
        d1 = extract_similarity_circuit(jsonl, seed_feature=0, depth=1, top_k_neighbors=5)
        d2 = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=5)
        assert len(d1["nodes"]) <= len(d2["nodes"])

    def test_no_duplicate_nodes(self, tmp_path: Path) -> None:
        """Each feature should appear at most once in nodes."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=5)
        indices = [n["featureIndex"] for n in result["nodes"]]
        assert len(indices) == len(set(indices))


@pytest.mark.slow
class TestCoactivationCircuit:
    """Integration tests requiring model download. Run with: pytest -m slow"""

    def test_basic_coactivation(self) -> None:
        """Co-activation circuit produces valid output."""
        from pipeline.circuits import extract_coactivation_circuit

        result = extract_coactivation_circuit(
            prompt="The capital of France is",
            model_name="gpt2",
            top_k_features=10,
            min_coactivation=0.05,
            device="cpu",
        )

        assert "name" in result
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) <= 10
        for node in result["nodes"]:
            assert node["role"] in ("source", "intermediate", "sink")
            assert 0 <= node["activation"] <= 1
        for edge in result["edges"]:
            assert 0 <= edge["weight"] <= 1
