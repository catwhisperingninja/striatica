# striatica/tests/test_data_integrity.py
"""Data integrity tests: cross-component validation, alignment, and contamination detection.

These tests exist because of a real bug where broadly-activating SAE features
contaminated every co-activation circuit, producing false multi-circuit convergences.
If you're here because a test failed, something changed the data pipeline in a way
that breaks the scientific validity of the output. Fix it before shipping.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pipeline.prepare import prepare_json, prepare_json_minimal
from pipeline.reduce import reduce_to_3d
from pipeline.cluster import cluster_points
from pipeline.circuits import extract_similarity_circuit


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    """Full synthetic dataset with coords, labels, features, and explanations."""
    n = 200
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((n, 3)).astype(np.float32)
    # Normalize like reduce.py does
    coords = coords - coords.mean(axis=0)
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs

    labels = np.array([0] * 60 + [1] * 50 + [2] * 40 + [-1] * 50)

    features_path = tmp_path / "features.jsonl"
    explanations_path = tmp_path / "explanations.jsonl"

    with open(features_path, "w") as f:
        for i in range(n):
            neighbors = [(i + j) % n for j in range(11)]
            sims = [round(1.0 - j * 0.08, 4) for j in range(11)]
            json.dump({
                "index": i,
                "maxActApprox": float(rng.uniform(0, 10)),
                "frac_nonzero": round(float(rng.uniform(0, 0.1)), 4),
                "topkCosSimIndices": neighbors,
                "topkCosSimValues": sims,
                "pos_str": [f"token_{i}"],
                "neg_str": [],
            }, f)
            f.write("\n")

    with open(explanations_path, "w") as f:
        for i in range(n):
            json.dump({
                "index": i,
                "description": f"Feature {i} explanation",
            }, f)
            f.write("\n")

    return {
        "coords": coords,
        "labels": labels,
        "features_path": features_path,
        "explanations_path": explanations_path,
        "n": n,
        "tmp_path": tmp_path,
    }


# ── Array Alignment Tests ───────────────────────────────────────────────


class TestArrayAlignment:
    """Positions, labels, localDimensions, and features must all align by index."""

    def test_prepare_json_arrays_aligned(self, synthetic_dataset, tmp_path):
        """All arrays in the final JSON must have consistent lengths."""
        d = synthetic_dataset
        output = tmp_path / "aligned.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
        )

        n = result["numFeatures"]
        assert len(result["positions"]) == n * 3, (
            f"positions has {len(result['positions'])} elements, "
            f"expected {n * 3} (numFeatures={n})"
        )
        assert len(result["clusterLabels"]) == n, (
            f"clusterLabels has {len(result['clusterLabels'])} elements, "
            f"expected {n}"
        )
        assert len(result["features"]) == n, (
            f"features has {len(result['features'])} elements, "
            f"expected {n}"
        )

    def test_prepare_json_with_local_dims_aligned(self, synthetic_dataset, tmp_path):
        """localDimensions array must match numFeatures when present."""
        d = synthetic_dataset
        output = tmp_path / "with_dims.json"
        local_dims = np.random.default_rng(42).uniform(0, 10, size=d["n"])
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
            local_dimensions=local_dims,
        )

        n = result["numFeatures"]
        assert "localDimensions" in result
        assert len(result["localDimensions"]) == n, (
            f"localDimensions has {len(result['localDimensions'])} elements, "
            f"expected {n}"
        )

    def test_prepare_json_with_growth_curves_aligned(self, synthetic_dataset, tmp_path):
        """growthCurves list must match numFeatures when present."""
        d = synthetic_dataset
        output = tmp_path / "with_curves.json"
        curves = [{"log_r": [0.1], "log_v": [0.2], "slope": 1.0, "intercept": 0.0}
                  for _ in range(d["n"])]
        local_dims = np.ones(d["n"])
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
            local_dimensions=local_dims, growth_curves=curves,
        )

        n = result["numFeatures"]
        assert len(result["growthCurves"]) == n, (
            f"growthCurves has {len(result['growthCurves'])} elements, "
            f"expected {n}"
        )

    def test_feature_indices_contiguous(self, synthetic_dataset, tmp_path):
        """Feature indices in output must be contiguous 0..n-1."""
        d = synthetic_dataset
        output = tmp_path / "contiguous.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
        )

        indices = [f["index"] for f in result["features"]]
        assert indices == list(range(result["numFeatures"])), (
            "Feature indices are not contiguous 0..n-1"
        )

    def test_cluster_labels_valid_values(self, synthetic_dataset, tmp_path):
        """Cluster labels must be -1 or a non-negative integer."""
        d = synthetic_dataset
        output = tmp_path / "labels.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
        )

        for label in result["clusterLabels"]:
            assert isinstance(label, int), f"Label {label} is not int"
            assert label >= -1, f"Label {label} < -1"


# ── Feature Metadata Completeness ───────────────────────────────────────


class TestFeatureCompleteness:
    """No silent data loss when features are missing or malformed."""

    def test_missing_feature_detected(self, tmp_path):
        """If downloaded features don't cover all indices, we should know."""
        n = 100
        coords = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float32)
        labels = np.zeros(n, dtype=int)

        # Write only 90 of 100 features (skip indices 90-99)
        features_path = tmp_path / "partial_features.jsonl"
        explanations_path = tmp_path / "explanations.jsonl"
        with open(features_path, "w") as f:
            for i in range(90):  # Missing 10!
                json.dump({
                    "index": i,
                    "maxActApprox": 1.0,
                    "frac_nonzero": 0.01,
                    "topkCosSimIndices": [],
                    "pos_str": [],
                    "neg_str": [],
                }, f)
                f.write("\n")
        with open(explanations_path, "w") as f:
            for i in range(90):
                json.dump({"index": i, "description": f"feat {i}"}, f)
                f.write("\n")

        output = tmp_path / "partial.json"
        result = prepare_json(
            coords, labels, features_path, explanations_path,
            output, model="test", layer="test",
        )

        # Features 90-99 should have zero metadata (the current behavior)
        # This test documents the behavior so we can catch it and decide if
        # we want to make it an error in the future
        for i in range(90, 100):
            feat = result["features"][i]
            assert feat["maxAct"] == 0, (
                f"Feature {i} has maxAct={feat['maxAct']} but was not downloaded"
            )

    def test_all_features_have_required_fields(self, synthetic_dataset, tmp_path):
        """Every feature must have all required fields."""
        d = synthetic_dataset
        output = tmp_path / "fields.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
        )

        required = {"index", "explanation", "maxAct", "fracNonzero",
                    "topSimilar", "posTokens", "negTokens"}
        for feat in result["features"]:
            missing = required - set(feat.keys())
            assert not missing, (
                f"Feature {feat.get('index', '?')} missing fields: {missing}"
            )


# ── Circuit Contamination Tests ─────────────────────────────────────────


class TestCircuitContamination:
    """Detect broadly-activating features contaminating circuits."""

    def test_no_feature_in_all_circuits(self, tmp_path):
        """No single feature should appear in ALL circuits if there are 3+.

        This is the exact bug we found: broadly-activating features appeared
        in every co-activation circuit, producing false convergences.
        """
        circuits_dir = Path("frontend/public/data/circuits")
        if not circuits_dir.exists():
            pytest.skip("No circuit data generated yet")

        manifest_path = circuits_dir / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("No circuit manifest")

        with open(manifest_path) as f:
            manifest = json.load(f)

        circuit_entries = manifest.get("circuits", [])
        if len(circuit_entries) < 3:
            pytest.skip("Need at least 3 circuits to test contamination")

        # Count how many circuits each feature appears in
        feature_circuit_count: Counter = Counter()
        total_circuits = 0
        for entry in circuit_entries:
            circuit_path = circuits_dir / f"{entry['id']}.json"
            if not circuit_path.exists():
                continue
            total_circuits += 1
            with open(circuit_path) as f:
                circuit = json.load(f)
            for node in circuit["nodes"]:
                feature_circuit_count[node["featureIndex"]] += 1

        if total_circuits < 3:
            pytest.skip("Need at least 3 loaded circuits")

        # No feature should appear in ALL circuits
        for feat_idx, count in feature_circuit_count.items():
            assert count < total_circuits, (
                f"Feature {feat_idx} appears in ALL {total_circuits} circuits. "
                f"This indicates broadly-activating feature contamination."
            )

    def test_coact_circuits_have_unique_members(self, tmp_path):
        """Co-activation circuits for different prompts should have mostly unique features.

        If >50% of members are shared across all coact circuits, something is wrong.
        """
        circuits_dir = Path("frontend/public/data/circuits")
        if not circuits_dir.exists():
            pytest.skip("No circuit data generated yet")

        coact_files = sorted(circuits_dir.glob("coact-*.json"))
        if len(coact_files) < 2:
            pytest.skip("Need at least 2 co-activation circuits")

        # Get node sets
        node_sets = []
        for path in coact_files:
            with open(path) as f:
                circuit = json.load(f)
            indices = {n["featureIndex"] for n in circuit["nodes"]}
            node_sets.append(indices)

        # Intersection of ALL coact circuits
        shared = set.intersection(*node_sets) if node_sets else set()
        max_size = max(len(s) for s in node_sets)

        overlap_ratio = len(shared) / max_size if max_size > 0 else 0
        assert overlap_ratio < 0.5, (
            f"{len(shared)} features shared across ALL {len(coact_files)} "
            f"co-activation circuits ({overlap_ratio:.0%} overlap). "
            f"Shared features: {sorted(shared)[:20]}..."
        )

    def test_circuit_node_indices_in_range(self):
        """All circuit node featureIndex values must be < numFeatures."""
        circuits_dir = Path("frontend/public/data/circuits")
        dataset_path = Path("frontend/public/data")

        if not circuits_dir.exists():
            pytest.skip("No circuit data generated yet")

        # Find dataset to get numFeatures
        dataset_files = list(dataset_path.glob("*.json"))
        dataset_files = [f for f in dataset_files if f.name != "manifest.json"
                         and "circuit" not in str(f)]
        if not dataset_files:
            pytest.skip("No dataset JSON found")

        with open(dataset_files[0]) as f:
            dataset = json.load(f)
        num_features = dataset["numFeatures"]

        for circuit_path in circuits_dir.glob("*.json"):
            if circuit_path.name == "manifest.json":
                continue
            with open(circuit_path) as f:
                circuit = json.load(f)
            for node in circuit["nodes"]:
                assert 0 <= node["featureIndex"] < num_features, (
                    f"Circuit {circuit_path.name}: node featureIndex "
                    f"{node['featureIndex']} >= numFeatures ({num_features})"
                )

    def test_circuit_edge_indices_are_nodes(self):
        """All edge source/target must reference a node in the same circuit."""
        circuits_dir = Path("frontend/public/data/circuits")
        if not circuits_dir.exists():
            pytest.skip("No circuit data generated yet")

        for circuit_path in circuits_dir.glob("*.json"):
            if circuit_path.name == "manifest.json":
                continue
            with open(circuit_path) as f:
                circuit = json.load(f)
            node_indices = {n["featureIndex"] for n in circuit["nodes"]}
            for edge in circuit["edges"]:
                assert edge["source"] in node_indices, (
                    f"Circuit {circuit_path.name}: edge source "
                    f"{edge['source']} not in node set"
                )
                assert edge["target"] in node_indices, (
                    f"Circuit {circuit_path.name}: edge target "
                    f"{edge['target']} not in node set"
                )

    def test_no_duplicate_nodes_in_circuit(self):
        """Each feature should appear at most once per circuit."""
        circuits_dir = Path("frontend/public/data/circuits")
        if not circuits_dir.exists():
            pytest.skip("No circuit data generated yet")

        for circuit_path in circuits_dir.glob("*.json"):
            if circuit_path.name == "manifest.json":
                continue
            with open(circuit_path) as f:
                circuit = json.load(f)
            indices = [n["featureIndex"] for n in circuit["nodes"]]
            assert len(indices) == len(set(indices)), (
                f"Circuit {circuit_path.name}: duplicate node indices "
                f"{[i for i in indices if indices.count(i) > 1]}"
            )


# ── Similarity Circuit Validation ────────────────────────────────────────


class TestSimilarityIntegrity:
    """Similarity circuits should reflect actual cosine similarity neighborhoods."""

    def test_seed_feature_is_in_circuit(self, tmp_path):
        """The seed feature must always be present in its own similarity circuit."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=5, depth=2, top_k_neighbors=3)
        node_indices = {n["featureIndex"] for n in result["nodes"]}
        assert 5 in node_indices, "Seed feature 5 missing from its own circuit"

    def test_similarity_edges_reference_valid_nodes(self, tmp_path):
        """All edge endpoints must be in the node list."""
        jsonl = _make_features_jsonl(tmp_path)
        result = extract_similarity_circuit(jsonl, seed_feature=0, depth=2, top_k_neighbors=5)
        node_indices = {n["featureIndex"] for n in result["nodes"]}
        for edge in result["edges"]:
            assert edge["source"] in node_indices, (
                f"Edge source {edge['source']} not a circuit node"
            )
            assert edge["target"] in node_indices, (
                f"Edge target {edge['target']} not a circuit node"
            )


# ── Numerical Robustness ─────────────────────────────────────────────────


class TestNumericalRobustness:
    """Guards against NaN, inf, and degenerate inputs in the pipeline."""

    def test_reduce_output_finite(self):
        """UMAP output must contain no NaN or inf values."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((200, 768))
        coords = reduce_to_3d(vectors, pca_dim=50, random_state=42)
        assert np.all(np.isfinite(coords)), (
            f"reduce_to_3d produced non-finite values: "
            f"NaN={np.isnan(coords).sum()}, inf={np.isinf(coords).sum()}"
        )

    def test_reduce_output_normalized(self):
        """UMAP output should be centered and within [-1, 1]."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((200, 768))
        coords = reduce_to_3d(vectors, pca_dim=50, random_state=42)
        assert np.abs(coords).max() <= 1.0 + 1e-6, (
            f"reduce_to_3d output exceeds [-1, 1]: max abs = {np.abs(coords).max()}"
        )

    def test_cluster_labels_shape_matches_input(self):
        """cluster_points output length must equal input length."""
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((200, 3)).astype(np.float32)
        labels = cluster_points(coords, min_cluster_size=20, min_samples=5)
        assert len(labels) == len(coords), (
            f"Cluster labels length {len(labels)} != input length {len(coords)}"
        )

    def test_cluster_labels_values_valid(self):
        """Cluster labels must be >= -1."""
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((200, 3)).astype(np.float32)
        labels = cluster_points(coords, min_cluster_size=20, min_samples=5)
        assert np.all(labels >= -1), (
            f"Cluster labels contain values < -1: {labels[labels < -1]}"
        )

    def test_prepare_positions_finite(self, synthetic_dataset, tmp_path):
        """Final JSON positions must all be finite."""
        d = synthetic_dataset
        output = tmp_path / "finite.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="test", layer="test",
        )
        positions = result["positions"]
        for i, v in enumerate(positions):
            assert np.isfinite(v), (
                f"Position[{i}] = {v} is not finite "
                f"(feature {i // 3}, component {i % 3})"
            )


# ── End-to-End Pipeline Alignment ────────────────────────────────────────


class TestEndToEndAlignment:
    """The full pipeline from vectors -> JSON must produce aligned output."""

    def test_reduce_cluster_prepare_alignment(self, tmp_path):
        """reduce -> cluster -> prepare must produce consistent array lengths."""
        rng = np.random.default_rng(42)
        n = 300
        vectors = rng.standard_normal((n, 768))

        coords = reduce_to_3d(vectors, pca_dim=50, random_state=42)
        assert len(coords) == n

        labels = cluster_points(coords)
        assert len(labels) == n

        # Create matching feature data
        features_path = tmp_path / "features.jsonl"
        explanations_path = tmp_path / "explanations.jsonl"
        with open(features_path, "w") as f:
            for i in range(n):
                json.dump({
                    "index": i, "maxActApprox": 1.0, "frac_nonzero": 0.01,
                    "topkCosSimIndices": [], "pos_str": [], "neg_str": [],
                }, f)
                f.write("\n")
        with open(explanations_path, "w") as f:
            for i in range(n):
                json.dump({"index": i, "description": f"feat {i}"}, f)
                f.write("\n")

        output = tmp_path / "e2e.json"
        result = prepare_json(
            coords, labels, features_path, explanations_path,
            output, model="test", layer="test",
        )

        # Everything must be n
        assert result["numFeatures"] == n
        assert len(result["positions"]) == n * 3
        assert len(result["clusterLabels"]) == n
        assert len(result["features"]) == n

        # Verify it roundtrips from disk
        with open(output) as f:
            disk = json.load(f)
        assert disk["numFeatures"] == n
        assert len(disk["positions"]) == n * 3


# ── Helper ───────────────────────────────────────────────────────────────


def _make_features_jsonl(tmp_path: Path, num_features: int = 100) -> Path:
    """Create a synthetic features JSONL for testing."""
    path = tmp_path / "features.jsonl"
    with open(path, "w") as f:
        for i in range(num_features):
            neighbors = [(i + j) % num_features for j in range(11)]
            sims = [round(1.0 - j * 0.08, 4) for j in range(11)]
            json.dump({
                "index": i,
                "maxActApprox": float(num_features - i),
                "frac_nonzero": 0.01,
                "topkCosSimIndices": neighbors,
                "topkCosSimValues": sims,
                "pos_str": [f"token_{i}"],
                "neg_str": [],
            }, f)
            f.write("\n")
    return path
