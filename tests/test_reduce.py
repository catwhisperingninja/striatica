# striatica/tests/test_reduce.py
"""Tests for dimensionality reduction."""
import numpy as np
import pytest

# auto_pca_dim is pure math — always importable
from pipeline.reduce import auto_pca_dim

# reduce_to_3d requires sklearn + umap (Docker only)
try:
    from pipeline.reduce import reduce_to_3d
except ImportError:
    reduce_to_3d = None


# --- Existing tests (require sklearn + umap — run in Docker) ---


@pytest.mark.slow
def test_reduce_to_3d_output_shape():
    """PCA+UMAP should produce (n, 3) coordinates."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((500, 768))
    coords = reduce_to_3d(vectors, pca_dim=50, random_state=42)
    assert coords.shape == (500, 3)
    assert coords.dtype == np.float32


@pytest.mark.slow
def test_reduce_to_3d_deterministic():
    """Same seed should produce same output."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((200, 768))
    a = reduce_to_3d(vectors, pca_dim=50, random_state=42)
    b = reduce_to_3d(vectors, pca_dim=50, random_state=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_reduce_to_3d_n_jobs():
    """n_jobs parameter should be accepted and produce valid output."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((200, 768))
    coords = reduce_to_3d(vectors, pca_dim=50, random_state=42, n_jobs=1)
    assert coords.shape == (200, 3)
    assert coords.dtype == np.float32


# --- A1: Adaptive pca_dim tests (TDD — written before implementation) ---


def test_auto_pca_dim_768d():
    """GPT-2 Small (768-D): d // 4 = 192, under cap."""
    assert auto_pca_dim(768) == 192


def test_auto_pca_dim_2304d():
    """Gemma 2 transcoder (2304-D): d // 4 = 576, capped at 300."""
    assert auto_pca_dim(2304) == 300


def test_auto_pca_dim_4096d():
    """Large model (4096-D): d // 4 = 1024, capped at 300."""
    assert auto_pca_dim(4096) == 300


def test_auto_pca_dim_small():
    """Very small input (64-D): d // 4 = 16, should still work."""
    result = auto_pca_dim(64)
    assert result == 16
    assert result > 0


def test_auto_pca_dim_at_cap_boundary():
    """1200-D: d // 4 = 300, exactly at cap."""
    assert auto_pca_dim(1200) == 300


@pytest.mark.slow
def test_reduce_respects_pca_dim():
    """reduce_to_3d() should use the given pca_dim, not hardcoded 50."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((300, 768))
    # With pca_dim=100, PCA should retain more variance than pca_dim=10
    _, var_100 = reduce_to_3d(vectors, pca_dim=100, random_state=42, return_pca_variance=True)
    _, var_10 = reduce_to_3d(vectors, pca_dim=10, random_state=42, return_pca_variance=True)
    assert var_100 > var_10


@pytest.mark.slow
def test_reduce_auto_pca_dim():
    """reduce_to_3d() with pca_dim='auto' should use auto_pca_dim()."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((300, 768))
    # pca_dim='auto' on 768-D should use 192 components
    coords, variance = reduce_to_3d(
        vectors, pca_dim="auto", random_state=42, return_pca_variance=True
    )
    assert coords.shape == (300, 3)
    # 192 components on 768-D should capture substantially more than 50 components
    _, var_50 = reduce_to_3d(vectors, pca_dim=50, random_state=42, return_pca_variance=True)
    assert variance > var_50
