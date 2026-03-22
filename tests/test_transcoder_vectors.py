# striatica/tests/test_transcoder_vectors.py
"""Tests for Transcoder decoder vector extraction.

IMPORTANT — safetensors key name:
The decoder weight key in gemmascope transcoder safetensors files has NOT been
verified against the actual downloaded file yet. The tests below use "W_dec"
as a placeholder. Before implementation, download one file and run:

    from safetensors.numpy import load_file
    weights = load_file("params.safetensors")
    print(list(weights.keys()))

Then update TRANSCODER_DECODER_KEY in pipeline/vectors.py to match.
This is tracked as Step 1 in GEMMA2_TRANSCODER_PIPELINE_SPEC.md.
"""
import numpy as np
import pytest
from unittest.mock import patch

from pipeline.vectors import load_transcoder_vectors


# The expected shape for Gemma 2 2B transcoders (width_16k)
EXPECTED_NUM_FEATURES = 16384
EXPECTED_D_MODEL = 2304


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_download_transcoder_weights_shape(mock_load_file, mock_hf_download):
    """Decoder vectors should be (16384, 2304) for Gemma 2 2B Transcoders."""
    rng = np.random.default_rng(42)
    mock_weights = rng.standard_normal(
        (EXPECTED_NUM_FEATURES, EXPECTED_D_MODEL)
    ).astype(np.float32)
    mock_load_file.return_value = {"W_dec": mock_weights}
    mock_hf_download.return_value = "mock_path.safetensors"

    vectors = load_transcoder_vectors(layer=12, l0_variant=6)

    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape == (EXPECTED_NUM_FEATURES, EXPECTED_D_MODEL)

    # Verify correct HuggingFace path construction
    mock_hf_download.assert_called_once_with(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        filename="layer_12/width_16k/average_l0_6/params.safetensors",
    )


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_download_transcoder_weights_finite(mock_load_file, mock_hf_download):
    """Loaded weights must contain no NaN or inf values."""
    rng = np.random.default_rng(42)
    mock_weights = rng.standard_normal(
        (EXPECTED_NUM_FEATURES, EXPECTED_D_MODEL)
    ).astype(np.float32)
    mock_load_file.return_value = {"W_dec": mock_weights}
    mock_hf_download.return_value = "mock_path.safetensors"

    vectors = load_transcoder_vectors(layer=12, l0_variant=6)
    assert np.all(np.isfinite(vectors))


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_rejects_nan_weights(mock_load_file, mock_hf_download):
    """Must raise ValueError if downloaded weights contain NaN."""
    bad_weights = np.full(
        (EXPECTED_NUM_FEATURES, EXPECTED_D_MODEL), np.nan, dtype=np.float32
    )
    mock_load_file.return_value = {"W_dec": bad_weights}
    mock_hf_download.return_value = "mock_path.safetensors"

    with pytest.raises(ValueError, match="non-finite"):
        load_transcoder_vectors(layer=12, l0_variant=6)


@patch("pipeline.vectors.hf_hub_download")
def test_fallback_on_missing_l0(mock_hf_download):
    """Raise ValueError if the L0 variant does not exist on HuggingFace.

    EntryNotFoundError's constructor signature varies across huggingface_hub
    versions (some need a Response object, some accept a plain string).
    We create the error in a version-agnostic way to avoid test fragility.
    """
    from huggingface_hub.utils import EntryNotFoundError

    # Version-agnostic instantiation: try simple string first, fall back
    # to __new__ (bypasses __init__) if the constructor requires more args.
    try:
        err = EntryNotFoundError("File not found")
    except TypeError:
        err = EntryNotFoundError.__new__(EntryNotFoundError)
        err.args = ("File not found",)

    mock_hf_download.side_effect = err

    with pytest.raises(ValueError, match="L0 variant 999 not found"):
        load_transcoder_vectors(layer=12, l0_variant=999)


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_unknown_safetensors_key_raises(mock_load_file, mock_hf_download):
    """If the safetensors file has unexpected keys, raise with a helpful message."""
    mock_load_file.return_value = {"some_unexpected_key": np.zeros((10, 10))}
    mock_hf_download.return_value = "mock_path.safetensors"

    with pytest.raises(KeyError, match="decoder weight key"):
        load_transcoder_vectors(layer=12, l0_variant=6)


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_layer_0_path_construction(mock_load_file, mock_hf_download):
    """Layer 0 should produce correct HuggingFace path."""
    rng = np.random.default_rng(42)
    mock_weights = rng.standard_normal(
        (EXPECTED_NUM_FEATURES, EXPECTED_D_MODEL)
    ).astype(np.float32)
    mock_load_file.return_value = {"W_dec": mock_weights}
    mock_hf_download.return_value = "mock_path.safetensors"

    load_transcoder_vectors(layer=0, l0_variant=60)

    mock_hf_download.assert_called_once_with(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        filename="layer_0/width_16k/average_l0_60/params.safetensors",
    )
