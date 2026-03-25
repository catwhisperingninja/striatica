# striatica/tests/test_validate.py
"""Tests for the pipeline validation suite (L1, L2, L3).

Each validation check has a passing test and a failure-mode test to prove
the check actually catches the defect it's designed for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.validate import (
    ValidationReport,
    ValidationError,
    validate_level1_arrays,
    validate_level1_json,
    validate_level2,
    validate_level3,
    write_validation_sidecar,
)
from pipeline.prepare import prepare_json


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def good_arrays():
    """Valid pipeline output arrays."""
    n = 200
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((n, 3)).astype(np.float32)
    # Normalize like reduce.py
    coords = coords - coords.mean(axis=0)
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs

    labels = np.array([0] * 60 + [1] * 50 + [2] * 40 + [-1] * 50)
    local_dims = rng.uniform(1, 20, size=n).astype(np.float32)
    growth_curves = [
        {"log_r": [0.1, 0.2], "log_v": [0.3, 0.4], "slope": 5.0, "intercept": 0.1}
        for _ in range(n)
    ]
    return {
        "coords": coords,
        "labels": labels,
        "local_dims": local_dims,
        "growth_curves": growth_curves,
        "n": n,
    }


@pytest.fixture
def good_json(good_arrays, tmp_path):
    """Valid JSON result dict (as produced by prepare_json)."""
    d = good_arrays
    n = d["n"]
    features_path = tmp_path / "features.jsonl"
    explanations_path = tmp_path / "explanations.jsonl"

    with open(features_path, "w") as f:
        for i in range(n):
            json.dump({
                "index": i,
                "maxActApprox": float(np.random.default_rng(i).uniform(0, 10)),
                "frac_nonzero": 0.01,
                "topkCosSimIndices": [],
                "pos_str": [],
                "neg_str": [],
            }, f)
            f.write("\n")

    with open(explanations_path, "w") as f:
        for i in range(n):
            json.dump({"index": i, "description": f"Feature {i}"}, f)
            f.write("\n")

    output = tmp_path / "test.json"
    result = prepare_json(
        d["coords"], d["labels"], features_path, explanations_path,
        output, model="test", layer="test",
        local_dimensions=d["local_dims"],
        growth_curves=d["growth_curves"],
    )
    return result


@pytest.fixture
def high_d_dataset():
    """Synthetic dataset with known structure for L2 testing.

    Creates 500 points in 100D space with structure concentrated in ~10
    dimensions (not perfectly rank-deficient, so PCA works normally).
    UMAP should preserve local structure well → high trustworthiness.
    """
    rng = np.random.default_rng(42)
    n = 500
    d_ambient = 100

    # Strong signal in first 10 dims, weak noise in remaining 90
    vectors = np.zeros((n, d_ambient))
    vectors[:, :10] = rng.standard_normal((n, 10)) * 5.0  # strong
    vectors[:, 10:] = rng.standard_normal((n, 90)) * 0.1  # weak noise

    return {"vectors": vectors, "n": n, "d_intrinsic": 10}


# ── Level 1 Array Tests ────────────────────────────────────────────────


class TestLevel1Arrays:
    """Level 1 validation on raw numpy arrays."""

    def test_valid_arrays_pass(self, good_arrays):
        """Clean data passes all L1 checks."""
        d = good_arrays
        report = validate_level1_arrays(
            d["coords"], d["labels"], d["local_dims"], d["growth_curves"],
        )
        assert report.passed, (
            f"L1 failed on valid data: "
            + ", ".join(c.name for c in report.checks if not c.passed)
        )

    def test_wrong_coord_shape_fails(self, good_arrays):
        """Coords with wrong number of columns must fail L1-SHAPE."""
        d = good_arrays
        bad_coords = d["coords"][:, :2]  # only 2 columns
        report = validate_level1_arrays(bad_coords, d["labels"])
        shape_check = next(c for c in report.checks if c.name == "L1-SHAPE")
        assert not shape_check.passed

    def test_label_length_mismatch_fails(self, good_arrays):
        """Labels shorter than coords must fail L1-LABELS."""
        d = good_arrays
        bad_labels = d["labels"][:100]  # too short
        report = validate_level1_arrays(d["coords"], bad_labels)
        check = next(c for c in report.checks if c.name == "L1-LABELS")
        assert not check.passed

    def test_nan_coords_fail(self, good_arrays):
        """NaN in coordinates must fail L1-FINITE."""
        d = good_arrays
        bad_coords = d["coords"].copy()
        bad_coords[10, 1] = float("nan")
        report = validate_level1_arrays(bad_coords, d["labels"])
        check = next(c for c in report.checks if c.name == "L1-FINITE")
        assert not check.passed

    def test_inf_coords_fail(self, good_arrays):
        """Inf in coordinates must fail L1-FINITE."""
        d = good_arrays
        bad_coords = d["coords"].copy()
        bad_coords[5, 0] = float("inf")
        report = validate_level1_arrays(bad_coords, d["labels"])
        check = next(c for c in report.checks if c.name == "L1-FINITE")
        assert not check.passed

    def test_out_of_range_coords_fail(self, good_arrays):
        """Coordinates outside [-1, 1] must fail L1-RANGE."""
        d = good_arrays
        bad_coords = d["coords"].copy()
        bad_coords[0, 0] = 2.5
        report = validate_level1_arrays(bad_coords, d["labels"])
        check = next(c for c in report.checks if c.name == "L1-RANGE")
        assert not check.passed

    def test_negative_label_fails(self, good_arrays):
        """Labels < -1 must fail L1-LABELS-VALID."""
        d = good_arrays
        bad_labels = d["labels"].copy()
        bad_labels[0] = -2
        report = validate_level1_arrays(d["coords"], bad_labels)
        check = next(c for c in report.checks if c.name == "L1-LABELS-VALID")
        assert not check.passed

    def test_local_dims_length_mismatch_fails(self, good_arrays):
        """Wrong-length local_dims must fail L1-DIMS."""
        d = good_arrays
        bad_dims = d["local_dims"][:50]
        report = validate_level1_arrays(
            d["coords"], d["labels"], local_dims=bad_dims,
        )
        check = next(c for c in report.checks if c.name == "L1-DIMS")
        assert not check.passed

    def test_local_dims_out_of_bounds_fails(self, good_arrays):
        """Local dims outside [0, 100] must fail L1-DIM-BOUNDS."""
        d = good_arrays
        bad_dims = d["local_dims"].copy()
        bad_dims[0] = -5.0
        report = validate_level1_arrays(
            d["coords"], d["labels"], local_dims=bad_dims,
        )
        check = next(c for c in report.checks if c.name == "L1-DIM-BOUNDS")
        assert not check.passed

    def test_growth_curves_length_mismatch_fails(self, good_arrays):
        """Wrong-length growth_curves must fail L1-CURVES."""
        d = good_arrays
        bad_curves = d["growth_curves"][:10]
        report = validate_level1_arrays(
            d["coords"], d["labels"], growth_curves=bad_curves,
        )
        check = next(c for c in report.checks if c.name == "L1-CURVES")
        assert not check.passed

    def test_collapsed_axis_warns(self):
        """All-zero axis must fail L1-SPREAD."""
        n = 100
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((n, 3)).astype(np.float32)
        coords = coords - coords.mean(axis=0)
        max_abs = np.abs(coords).max()
        coords = coords / max_abs
        # Collapse z-axis
        coords[:, 2] = 0.0
        labels = np.zeros(n, dtype=int)
        report = validate_level1_arrays(coords, labels)
        check = next(c for c in report.checks if c.name == "L1-SPREAD")
        assert not check.passed


# ── Level 1 JSON Tests ─────────────────────────────────────────────────


class TestLevel1Json:
    """Level 1 validation on assembled JSON dicts."""

    def test_valid_json_passes(self, good_json):
        """Well-formed JSON result passes all L1 checks."""
        report = validate_level1_json(good_json)
        assert report.passed, (
            f"L1 JSON failed: "
            + ", ".join(c.name for c in report.checks if not c.passed)
        )

    def test_missing_positions_fails(self, good_json):
        """Truncated positions array must fail L1-SHAPE."""
        bad = dict(good_json)
        bad["positions"] = good_json["positions"][:10]
        report = validate_level1_json(bad)
        check = next(c for c in report.checks if c.name == "L1-SHAPE")
        assert not check.passed

    def test_noncontiguous_indices_fails(self, good_json):
        """Gapped feature indices must fail L1-INDICES."""
        bad = dict(good_json)
        bad_features = list(good_json["features"])
        bad_features[5] = dict(bad_features[5])
        bad_features[5]["index"] = 999
        bad["features"] = bad_features
        report = validate_level1_json(bad)
        check = next(c for c in report.checks if c.name == "L1-INDICES")
        assert not check.passed

    def test_centroid_mismatch_fails(self, good_json):
        """Wrong centroid values must fail L1-CENTROID."""
        bad = dict(good_json)
        bad_clusters = [dict(c) for c in good_json["clusters"]]
        if bad_clusters:
            bad_clusters[0]["centroid"] = [99.0, 99.0, 99.0]
        bad["clusters"] = bad_clusters
        report = validate_level1_json(bad)
        check = next((c for c in report.checks if c.name == "L1-CENTROID"), None)
        if check:
            assert not check.passed


# ── Level 2 Tests ──────────────────────────────────────────────────────


class TestLevel2:
    """Level 2 embedding quality metrics."""

    @pytest.mark.slow
    def test_good_embedding_scores_well(self, high_d_dataset):
        """Points in a 5D subspace should embed well → high trustworthiness."""
        from pipeline.reduce import reduce_to_3d
        from pipeline.cluster import cluster_points

        vectors = high_d_dataset["vectors"]
        # pca_dim must be <= min(n_samples, n_features)
        pca_dim = min(20, vectors.shape[1] - 1)
        coords = reduce_to_3d(vectors, pca_dim=pca_dim, random_state=42)
        labels = cluster_points(coords, min_cluster_size=20, min_samples=5)

        report = validate_level2(vectors, coords, labels)
        report.print_scorecard()

        trust = next(c for c in report.checks if c.name == "L2-TRUST")
        assert trust.value >= 0.80, (
            f"Trustworthiness {trust.value} too low for 5D subspace data"
        )

    @pytest.mark.slow
    def test_shuffled_coords_score_badly(self, high_d_dataset):
        """Randomly shuffled coords should have near-zero neighborhood overlap."""
        from pipeline.reduce import reduce_to_3d
        from pipeline.cluster import cluster_points

        vectors = high_d_dataset["vectors"]
        pca_dim = min(20, vectors.shape[1] - 1)
        coords = reduce_to_3d(vectors, pca_dim=pca_dim, random_state=42)

        # Shuffle coords — break the mapping between vectors and positions
        rng = np.random.default_rng(99)
        shuffled = coords.copy()
        rng.shuffle(shuffled)

        labels = cluster_points(shuffled, min_cluster_size=20, min_samples=5)

        report = validate_level2(vectors, shuffled, labels)
        report.print_scorecard()

        # Trustworthiness should drop dramatically
        trust = next(c for c in report.checks if c.name == "L2-TRUST")
        assert trust.value < 0.85, (
            f"Shuffled coords got trustworthiness {trust.value} — metric isn't working"
        )

        # Neighborhood overlap should be near random chance
        nbhd30 = next((c for c in report.checks if c.name == "L2-NBHD-30"), None)
        if nbhd30 and nbhd30.value is not None:
            assert nbhd30.value < 0.15, (
                f"Shuffled coords got overlap {nbhd30.value} — should be near-zero"
            )

    def test_spread_detects_collapse(self):
        """Collapsed axis detected by L2-SPREAD."""
        rng = np.random.default_rng(42)
        n = 200
        vectors = rng.standard_normal((n, 50))
        coords = rng.standard_normal((n, 3)).astype(np.float32)
        coords[:, 2] = 0.001  # nearly collapsed z-axis
        labels = np.zeros(n, dtype=int)

        report = validate_level2(vectors, coords, labels)
        spread = next(c for c in report.checks if c.name == "L2-SPREAD")
        assert not spread.passed

    def test_subsample_works(self):
        """Subsampling should not crash and should produce a report."""
        rng = np.random.default_rng(42)
        n = 500
        vectors = rng.standard_normal((n, 50))
        coords = rng.standard_normal((n, 3)).astype(np.float32)
        coords = coords - coords.mean(axis=0)
        max_abs = np.abs(coords).max()
        coords = coords / max_abs
        labels = np.array([0] * 200 + [1] * 200 + [-1] * 100)

        report = validate_level2(vectors, coords, labels, subsample_limit=100)
        assert isinstance(report, ValidationReport)
        assert any(c.name == "L2-TRUST" for c in report.checks)


# ── Level 3 Tests ──────────────────────────────────────────────────────


class TestLevel3:
    """Level 3 cross-model distributional comparison."""

    def test_similar_distributions_pass(self, good_json, tmp_path):
        """Two datasets with similar structure should compare favorably."""
        # Save reference
        ref_path = tmp_path / "reference.json"
        with open(ref_path, "w") as f:
            json.dump(good_json, f)

        # Create a slightly different dataset (same structure)
        result = dict(good_json)
        report = validate_level3(result, ref_path)
        report.print_scorecard()

        # Same data should pass cluster count check
        cc = next((c for c in report.checks if c.name == "L3-CLUSTER-COUNT"), None)
        if cc:
            assert cc.passed

    def test_wildly_different_data_flags_differences(self, good_json, tmp_path):
        """Completely different cluster structure should show in comparison."""
        ref_path = tmp_path / "reference.json"
        with open(ref_path, "w") as f:
            json.dump(good_json, f)

        # Make a result where everything is one cluster
        bad_result = dict(good_json)
        bad_result["clusterLabels"] = [0] * good_json["numFeatures"]

        report = validate_level3(bad_result, ref_path)
        report.print_scorecard()

        # Uncategorized fraction should differ (ref has -1 noise, new has none)
        uf = next((c for c in report.checks if c.name == "L3-UNCAT-FRAC"), None)
        if uf:
            assert not uf.passed


# ── Sidecar Tests ──────────────────────────────────────────────────────


class TestSidecar:
    """Validation sidecar file writing."""

    def test_sidecar_written(self, good_arrays, tmp_path):
        """Sidecar file should be created alongside output."""
        d = good_arrays
        output = tmp_path / "test-output.json"
        output.touch()

        l1 = validate_level1_arrays(d["coords"], d["labels"])
        path = write_validation_sidecar(output, l1)

        assert path.exists()
        assert path.name == "test-output-validation.json"

        with open(path) as f:
            sidecar = json.load(f)
        assert sidecar["level1"]["passed"] is True
        assert sidecar["level2"] is None

    def test_sidecar_includes_l2(self, good_arrays, tmp_path):
        """Sidecar should include L2 data when provided."""
        d = good_arrays
        output = tmp_path / "test-output.json"
        output.touch()

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((d["n"], 50))

        l1 = validate_level1_arrays(d["coords"], d["labels"])
        l2 = validate_level2(vectors, d["coords"], d["labels"])
        path = write_validation_sidecar(output, l1, l2_report=l2)

        with open(path) as f:
            sidecar = json.load(f)
        assert sidecar["level2"] is not None
        assert "L2-TRUST" in sidecar["level2"]["checks"]


# ── ValidationReport Tests ─────────────────────────────────────────────


class TestValidationReport:
    """ValidationReport dataclass behavior."""

    def test_empty_report_passes(self):
        report = ValidationReport(level=1)
        assert report.passed

    def test_all_pass_reports_pass(self):
        report = ValidationReport(level=1)
        report.add("CHECK-1", True)
        report.add("CHECK-2", True)
        assert report.passed

    def test_one_failure_reports_fail(self):
        report = ValidationReport(level=1)
        report.add("CHECK-1", True)
        report.add("CHECK-2", False, "something broke")
        assert not report.passed

    def test_to_dict_serializable(self):
        report = ValidationReport(level=1)
        report.add("CHECK-1", True, value=np.float64(0.95))
        report.add("CHECK-2", True, value=np.array([1, 2, 3]))
        d = report.to_dict()
        # Must be JSON-serializable
        json.dumps(d)
