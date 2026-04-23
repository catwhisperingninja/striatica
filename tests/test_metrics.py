# striatica/tests/test_metrics.py
"""Tests for pipeline metrics emission (Grafana Phase 1)."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from pipeline.metrics import PipelineMetrics, _sha256_file, _get_dep_versions


# --- Unit: SHA-256 helper ---


def test_sha256_file(tmp_path):
    """SHA-256 of a known file should be deterministic."""
    f = tmp_path / "test.txt"
    f.write_text("hello world\n")
    h = _sha256_file(f)
    assert isinstance(h, str)
    assert len(h) == 64  # hex digest length
    # Same content -> same hash
    assert _sha256_file(f) == h


def test_sha256_different_content(tmp_path):
    """Different content must produce different hashes."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("aaa")
    f2.write_text("bbb")
    assert _sha256_file(f1) != _sha256_file(f2)


# --- Unit: dependency version collection ---


def test_get_dep_versions_includes_numpy():
    """numpy is always available in our env; it must appear."""
    versions = _get_dep_versions()
    assert "numpy" in versions
    assert versions["numpy"] != "unknown"


def test_get_dep_versions_display_names():
    """Import aliases should map to pip package names."""
    versions = _get_dep_versions()
    # If scikit-learn is installed, it should appear as 'scikit-learn' not 'sklearn'
    if any("scikit" in k or "sklearn" in k for k in versions):
        assert "scikit-learn" in versions


# --- Unit: step timing ---


def test_step_timing_records_duration():
    """step_start / step_end should record positive duration."""
    m = PipelineMetrics(run_id="test-001")
    m.step_start("pca")
    time.sleep(0.01)
    m.step_end("pca")
    step = m._steps["pca"]
    assert step.duration_seconds > 0
    assert step.end_time > step.start_time


def test_step_end_without_start_is_noop():
    """Ending a step that was never started should not crash."""
    m = PipelineMetrics(run_id="test-002")
    m.step_end("nonexistent")  # should not raise
    assert "nonexistent" not in m._steps


# --- Unit: setters ---


def test_set_feature_count():
    m = PipelineMetrics()
    m.set_feature_count(24576)
    assert m.feature_count == 24576


def test_set_umap_progress_appends():
    m = PipelineMetrics()
    m.set_umap_progress(epoch=10, total=200, loss=1.5)
    m.set_umap_progress(epoch=20, total=200, loss=1.2)
    assert len(m.umap_epochs) == 2
    assert m.umap_epochs[0]["epoch"] == 10
    assert m.umap_epochs[1]["loss"] == 1.2


def test_set_cluster_stats():
    m = PipelineMetrics()
    m.set_cluster_stats(cluster_count=12, uncategorized_fraction=0.35, sizes={0: 500, 1: 300})
    assert m.cluster_count == 12
    assert m.uncategorized_fraction == 0.35
    assert m.cluster_sizes[0] == 500


def test_set_vgt_stats():
    m = PipelineMetrics()
    m.set_vgt_stats(mean=15.2, values=[10.0, 15.0, 20.0])
    assert m.vgt_values == [10.0, 15.0, 20.0]


def test_set_pca_variance():
    m = PipelineMetrics()
    m.set_pca_variance([0.4, 0.2, 0.1, 0.05])
    assert len(m.pca_variance_ratios) == 4


# --- Integration: JSON sidecar output ---


def _make_dummy_output(tmp_path: Path) -> Path:
    """Create a minimal pipeline output JSON for testing."""
    output = tmp_path / "gpt2-small-6-res-jb.json"
    data = {
        "features": [
            {"index": 0, "pos": [1.0, 2.0, 3.0]},
            {"index": 1, "pos": [4.0, 5.0, 6.0]},
            {"index": 2, "pos": [7.0, 8.0, 9.0]},
        ]
    }
    output.write_text(json.dumps(data))
    return output


def test_finalize_writes_sidecar(tmp_path):
    """finalize() should write a JSON sidecar next to the output file."""
    output = _make_dummy_output(tmp_path)
    m = PipelineMetrics(run_id="test-sidecar", model_id="gpt2-small")
    m.set_feature_count(3)
    m.step_start("pca")
    m.step_end("pca")
    m.set_cluster_stats(cluster_count=1, uncategorized_fraction=0.0)

    result = m.finalize(output_path=str(output))

    sidecar = tmp_path / "gpt2-small-6-res-jb_metrics.json"
    assert sidecar.exists(), "Sidecar JSON was not written"

    with open(sidecar) as f:
        written = json.load(f)

    assert written["run_id"] == "test-sidecar"
    assert written["model_id"] == "gpt2-small"
    assert written["feature_count"] == 3
    assert written["output_sha256"] != ""
    assert "pca" in written["steps"]


def test_finalize_returns_complete_metrics():
    """finalize() return dict should contain all expected top-level keys."""
    m = PipelineMetrics(run_id="test-keys", model_id="test-model")
    result = m.finalize()

    expected_keys = {
        "run_id", "model_id", "pipeline_mode", "pipeline_version",
        "provider", "instance_type", "platform_arch", "python_version",
        "timestamp", "total_duration_seconds", "feature_count",
        "output_sha256", "lockfile_sha256", "umap_random_state",
        "dependency_versions", "gpu_info", "steps",
        "umap_convergence", "hdbscan", "pca_variance_ratios",
        "vgt", "vgt_activation_pearson_r", "drift",
        "memory_snapshots", "cost_inputs",
    }
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


def test_finalize_total_duration_positive():
    """Total duration should be positive (clock ticked between init and finalize)."""
    m = PipelineMetrics()
    time.sleep(0.01)
    result = m.finalize()
    assert result["total_duration_seconds"] > 0


def test_finalize_vgt_stats_computed():
    """VGT mean/std/median should be computed from values."""
    m = PipelineMetrics()
    m.set_vgt_stats(mean=0, values=[2.0, 4.0, 6.0])
    result = m.finalize()

    assert result["vgt"]["mean"] == pytest.approx(4.0)
    assert result["vgt"]["median"] == pytest.approx(4.0)
    assert result["vgt"]["std"] == pytest.approx(np.std([2.0, 4.0, 6.0]))
    assert result["vgt"]["features_computed"] == 3


def test_finalize_no_output_path():
    """finalize() with no output_path should still return metrics, just no sidecar."""
    m = PipelineMetrics(run_id="no-output")
    result = m.finalize()
    assert result["output_sha256"] == ""
    assert result["run_id"] == "no-output"


# --- Integration: drift computation ---


def test_drift_no_reference():
    """Without a reference, drift status should be 'no_reference'."""
    m = PipelineMetrics()
    result = m.finalize()
    assert result["drift"]["status"] == "no_reference"


def test_drift_zero_when_identical(tmp_path):
    """Identical outputs should produce zero drift."""
    output = _make_dummy_output(tmp_path)
    reference = tmp_path / "reference.json"
    # Copy same content
    reference.write_text(output.read_text())

    m = PipelineMetrics()
    result = m.finalize(output_path=str(output), reference_path=str(reference))

    assert result["drift"]["status"] == "computed"
    assert result["drift"]["max_drift"] == pytest.approx(0.0)
    assert result["drift"]["features_compared"] == 3


def test_drift_nonzero_when_different(tmp_path):
    """Different positions should produce nonzero drift."""
    output = _make_dummy_output(tmp_path)

    reference = tmp_path / "reference.json"
    ref_data = {
        "features": [
            {"index": 0, "pos": [1.0, 2.0, 3.0]},  # same
            {"index": 1, "pos": [4.0, 5.0, 60.0]},  # moved
            {"index": 2, "pos": [7.0, 8.0, 9.0]},   # same
        ]
    }
    reference.write_text(json.dumps(ref_data))

    m = PipelineMetrics()
    result = m.finalize(output_path=str(output), reference_path=str(reference))

    assert result["drift"]["status"] == "computed"
    assert result["drift"]["max_drift"] > 0
    assert result["drift"]["features_affected"] >= 1


def test_drift_missing_reference_file(tmp_path):
    """Missing reference file should return 'file_not_found'."""
    output = _make_dummy_output(tmp_path)
    m = PipelineMetrics()
    result = m.finalize(
        output_path=str(output),
        reference_path=str(tmp_path / "nonexistent.json"),
    )
    assert result["drift"]["status"] == "file_not_found"


# --- Unit: memory snapshots ---


def test_snapshot_memory_appends():
    """Each snapshot_memory() call should append to the list."""
    m = PipelineMetrics()
    m.snapshot_memory()
    m.snapshot_memory()
    assert len(m._memory_snapshots) == 2
    assert "timestamp" in m._memory_snapshots[0]
    assert "rss_bytes" in m._memory_snapshots[0]


# --- Unit: default values ---


def test_default_run_id_generated():
    """run_id should be auto-generated if not provided."""
    m = PipelineMetrics()
    assert len(m.run_id) == 12  # uuid hex[:12]


def test_default_umap_random_state():
    """Default UMAP random state must be 42."""
    m = PipelineMetrics()
    assert m.umap_random_state == 42


def test_default_pipeline_version():
    m = PipelineMetrics()
    assert m.pipeline_version == "0.3.0"


# --- Edge: prometheus push without gateway ---


def test_prometheus_push_noop_without_env(monkeypatch):
    """_push_prometheus should silently no-op without PROMETHEUS_PUSHGATEWAY."""
    monkeypatch.delenv("PROMETHEUS_PUSHGATEWAY", raising=False)
    m = PipelineMetrics()
    result = m.finalize()
    # Should not raise, just return metrics
    assert result["run_id"] is not None
