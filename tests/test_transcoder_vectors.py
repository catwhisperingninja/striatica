# striatica/tests/test_transcoder_vectors.py
"""Tests for transcoder decoder vector extraction."""

import numpy as np
import pytest

from pipeline.vectors import load_transcoder_vectors


def test_load_transcoder_vectors_downloads_and_loads_npz(monkeypatch):
    """Should download params.npz from HF and return decoder vectors."""
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    called = {}

    def fake_download(repo_id, filename):
        called["repo_id"] = repo_id
        called["filename"] = filename
        return "/tmp/params.npz"

    monkeypatch.setattr("pipeline.vectors.hf_hub_download", fake_download)
    monkeypatch.setattr("pipeline.vectors.np.load", lambda path: {"W_dec": expected})

    vectors = load_transcoder_vectors(
        layer=12,
        l0_variant=604,
        repo_id="google/gemma-scope-2b-pt-transcoders",
        width="width_16k",
    )

    assert called["repo_id"] == "google/gemma-scope-2b-pt-transcoders"
    assert called["filename"] == "layer_12/width_16k/average_l0_604/params.npz"
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 4)
    assert np.array_equal(vectors, expected)


def test_load_transcoder_vectors_raises_for_missing_decoder_key(monkeypatch):
    """Should raise KeyError when expected decoder key is absent."""
    monkeypatch.setattr("pipeline.vectors.hf_hub_download", lambda **kwargs: "/tmp/params.npz")
    monkeypatch.setattr("pipeline.vectors.np.load", lambda _path: {"unexpected_key": np.zeros((2, 2))})

    with pytest.raises(KeyError, match="Expected decoder weight key"):
        load_transcoder_vectors(layer=1, l0_variant=6)


def test_load_transcoder_vectors_raises_for_non_finite(monkeypatch):
    """Should raise ValueError when decoder vectors contain NaN/inf."""
    bad = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    monkeypatch.setattr("pipeline.vectors.hf_hub_download", lambda **kwargs: "/tmp/params.npz")
    monkeypatch.setattr("pipeline.vectors.np.load", lambda _path: {"W_dec": bad})

    with pytest.raises(ValueError, match="non-finite"):
        load_transcoder_vectors(layer=1, l0_variant=6)
