# striatica/tests/test_local_dim.py
"""Tests for local intrinsic dimension estimation."""
import numpy as np
import pytest
from pipeline.local_dim import (
    estimate_local_dim_pr,
    estimate_local_dim_twonn,
    estimate_local_dim_vgt,
    estimate_local_dim,
)


def _make_sphere_data(n=500, ambient=50, intrinsic=3, seed=42):
    """Generate points near a sphere surface (known intrinsic dim ~ intrinsic-1)."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, intrinsic))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    embedded = np.zeros((n, ambient))
    embedded[:, :intrinsic] = pts
    return embedded


def test_pr_output_shape():
    """Participation ratio should return one value per point."""
    data = _make_sphere_data(n=200, ambient=50)
    dims = estimate_local_dim_pr(data, k=30)
    assert dims.shape == (200,)
    assert np.all(np.isfinite(dims))
    assert np.all(dims > 0)


def test_twonn_output_shape():
    """TwoNN should return one value per point."""
    data = _make_sphere_data(n=200, ambient=50)
    dims = estimate_local_dim_twonn(data, k=30)
    assert dims.shape == (200,)
    assert np.all(np.isfinite(dims))
    assert np.all(dims > 0)


def test_vgt_output_shape():
    """VGT should return one value per point."""
    data = _make_sphere_data(n=200, ambient=50)
    dims = estimate_local_dim_vgt(data, n_radii=10, max_k=50)
    assert dims.shape == (200,)
    assert np.all(np.isfinite(dims))


def test_vgt_returns_curves():
    """VGT with return_curves=True should return curve data."""
    data = _make_sphere_data(n=100, ambient=50)
    dims, curves = estimate_local_dim_vgt(data, n_radii=10, max_k=50, return_curves=True)
    assert dims.shape == (100,)
    assert len(curves) == 100
    for c in curves:
        assert "log_r" in c and "log_v" in c
        assert "slope" in c and "intercept" in c
        assert len(c["log_r"]) == len(c["log_v"])


def test_unified_interface_default():
    """Default method should be 'pr'."""
    data = _make_sphere_data(n=100, ambient=50)
    dims = estimate_local_dim(data, method="pr")
    assert dims.shape == (100,)


def test_unified_interface_all_methods():
    """All methods should be callable through unified interface."""
    data = _make_sphere_data(n=100, ambient=50)
    for method in ["pr", "twonn", "vgt"]:
        dims = estimate_local_dim(data, method=method)
        assert dims.shape == (100,)


# ── Parallelization tests ──────────────────────────────────────────────


def test_pr_n_jobs_parameter():
    """PR should accept n_jobs and produce same shape output."""
    data = _make_sphere_data(n=100, ambient=50)
    dims_serial = estimate_local_dim_pr(data, k=20, n_jobs=1)
    dims_parallel = estimate_local_dim_pr(data, k=20, n_jobs=2)
    assert dims_serial.shape == dims_parallel.shape
    # Results should be identical regardless of parallelization
    np.testing.assert_allclose(dims_serial, dims_parallel, rtol=1e-5)


def test_vgt_n_jobs_parameter():
    """VGT should accept n_jobs and produce same shape output."""
    data = _make_sphere_data(n=100, ambient=50)
    dims_serial = estimate_local_dim_vgt(data, n_radii=10, max_k=50, n_jobs=1)
    dims_parallel = estimate_local_dim_vgt(data, n_radii=10, max_k=50, n_jobs=2)
    assert dims_serial.shape == dims_parallel.shape
    np.testing.assert_allclose(dims_serial, dims_parallel, rtol=1e-5)


def test_vgt_curves_with_n_jobs():
    """VGT growth curves should be identical regardless of n_jobs."""
    data = _make_sphere_data(n=50, ambient=50)
    dims_s, curves_s = estimate_local_dim_vgt(data, n_radii=10, max_k=50, return_curves=True, n_jobs=1)
    dims_p, curves_p = estimate_local_dim_vgt(data, n_radii=10, max_k=50, return_curves=True, n_jobs=2)
    np.testing.assert_allclose(dims_s, dims_p, rtol=1e-5)
    assert len(curves_s) == len(curves_p)
    for cs, cp in zip(curves_s, curves_p):
        assert cs["slope"] == pytest.approx(cp["slope"], rel=1e-5)


def test_default_n_jobs():
    """_default_n_jobs should return at least 1."""
    from pipeline.local_dim import _default_n_jobs
    n = _default_n_jobs()
    assert isinstance(n, int)
    assert n >= 1
