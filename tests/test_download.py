# striatica/tests/test_download.py
"""Tests for Neuronpedia S3 download.

Both tests download live batches from the Neuronpedia S3 bucket, so their
outcome depends on network state at run time: a transient S3/proxy error fails
a run that passes on retry (observed 2026-10-09, HTTP 404 on batch-0
explanations in 1 of 10 runs at the same SHA). Network-reliant tests are
fenced out of the canonical `-m "not slow"` run, like the live Neuronpedia
identity check in test_graph_fetch.py.
"""
import json

import pytest

from pipeline.download import download_features, download_explanations

pytestmark = pytest.mark.slow  # live S3 download — not in the canonical run


def test_download_features_creates_file(tmp_path):
    """Download batch-0 only and verify structure."""
    output = tmp_path / "features.jsonl"
    count = download_features(
        model_id="gpt2-small",
        layer="6-res-jb",
        batch_indices=[0],
        output_path=output,
    )
    assert output.exists()
    assert count == 1024
    first_line = json.loads(output.read_text().splitlines()[0])
    assert "index" in first_line
    assert "maxActApprox" in first_line
    assert "topkCosSimIndices" in first_line


def test_download_explanations_creates_file(tmp_path):
    """Download batch-0 explanations and verify structure."""
    output = tmp_path / "explanations.jsonl"
    count = download_explanations(
        model_id="gpt2-small",
        layer="6-res-jb",
        batch_indices=[0],
        output_path=output,
    )
    assert output.exists()
    assert count > 0
    first_line = json.loads(output.read_text().splitlines()[0])
    assert "description" in first_line
    assert "index" in first_line
