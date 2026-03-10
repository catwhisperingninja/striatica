# striatica/tests/test_reduce.py
"""Tests for dimensionality reduction."""
import numpy as np
from pipeline.reduce import reduce_to_3d


def test_reduce_to_3d_output_shape():
    """PCA+UMAP should produce (n, 3) coordinates."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((500, 768))
    coords = reduce_to_3d(vectors, pca_dim=50, random_state=42)
    assert coords.shape == (500, 3)
    assert coords.dtype == np.float32


def test_reduce_to_3d_deterministic():
    """Same seed should produce same output."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((200, 768))
    a = reduce_to_3d(vectors, pca_dim=50, random_state=42)
    b = reduce_to_3d(vectors, pca_dim=50, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_reduce_to_3d_n_jobs():
    """n_jobs parameter should be accepted and produce valid output."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((200, 768))
    coords = reduce_to_3d(vectors, pca_dim=50, random_state=42, n_jobs=1)
    assert coords.shape == (200, 3)
    assert coords.dtype == np.float32
