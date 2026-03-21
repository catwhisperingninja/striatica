# striatica/tests/test_transcoder_vectors.py
"""Tests for Transcoder decoder vector extraction."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pipeline.vectors import load_transcoder_vectors


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_download_transcoder_weights_shape(mock_load_file, mock_hf_download):
    """Decoder vectors should be (16384, 2304) for Gemma 2 2B Transcoders."""
    # Mock the safetensors load
    mock_weights = np.random.randn(16384, 2304).astype(np.float32)
    mock_load_file.return_value = {"W_dec": mock_weights}
    mock_hf_download.return_value = "mock_path.safetensors"

    vectors = load_transcoder_vectors(
        layer=12,
        l0_variant=6,
    )
    
    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape == (16384, 2304)
    
    # Check that hf_hub_download was called with correct repo and filename
    mock_hf_download.assert_called_once_with(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        filename="layer_12/width_16k/average_l0_6/params.safetensors",
    )


@patch("pipeline.vectors.hf_hub_download")
@patch("pipeline.vectors.load_file")
def test_download_transcoder_weights_finite(mock_load_file, mock_hf_download):
    """Loaded weights must contain no NaN or inf values."""
    mock_weights = np.random.randn(16384, 2304).astype(np.float32)
    mock_load_file.return_value = {"W_dec": mock_weights}
    mock_hf_download.return_value = "mock_path.safetensors"

    vectors = load_transcoder_vectors(layer=12, l0_variant=6)
    assert np.all(np.isfinite(vectors))


@patch("pipeline.vectors.hf_hub_download")
def test_fallback_on_missing_l0(mock_hf_download):
    """Raise ValueError if the L0 variant does not exist."""
    from huggingface_hub.utils import EntryNotFoundError
    
    # Simulate file not found on HuggingFace
    mock_hf_download.side_effect = EntryNotFoundError("File not found")
    
    with pytest.raises(ValueError, match="L0 variant 999 not found"):
        load_transcoder_vectors(layer=12, l0_variant=999)
