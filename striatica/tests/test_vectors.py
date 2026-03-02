# striatica/tests/test_vectors.py
"""Tests for SAELens decoder vector extraction."""
import numpy as np
from pipeline.vectors import load_decoder_vectors


def test_load_decoder_vectors_shape():
    """Decoder vectors should be (num_features, hidden_dim)."""
    vectors = load_decoder_vectors(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
    )
    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    # GPT2-small hidden_dim = 768
    assert vectors.shape[1] == 768
    # Should have thousands of features
    assert vectors.shape[0] > 1000
    # Vectors should be roughly unit norm (SAE decoder rows are normalized)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=0.1)
