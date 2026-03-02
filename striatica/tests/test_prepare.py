# striatica/tests/test_prepare.py
"""Tests for JSON preparation."""
import json
from pathlib import Path

import numpy as np
from pipeline.prepare import prepare_json_minimal


def test_prepare_json_minimal_structure(tmp_path):
    """Minimal JSON should have required fields."""
    coords = np.random.default_rng(42).standard_normal((100, 3)).astype(np.float32)
    labels = np.array([0] * 40 + [1] * 30 + [-1] * 30)
    out = tmp_path / "test.json"

    prepare_json_minimal(coords, labels, out)

    data = json.loads(out.read_text())
    assert data["numFeatures"] == 100
    assert len(data["positions"]) == 300  # 100 * 3
    assert len(data["clusterLabels"]) == 100
