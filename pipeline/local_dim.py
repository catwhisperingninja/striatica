# striatica/pipeline/local_dim.py
"""Per-point local intrinsic dimension estimation.

Three methods available:
- 'pr': Participation ratio (Gao & Ganguli 2017) — neuroscience standard
- 'twonn': TwoNN (Ansuini et al. NeurIPS 2019) — nearest-neighbor statistics
- 'vgt': Volume Growth Transform (Curry et al. 2025) — neighbor counting in expanding radii

All methods estimate a scalar dimension value per point based on local
neighborhood geometry in the original high-dimensional space.
"""

from __future__ import annotations

import os
import time as _time

import numpy as np
from scipy.spatial import KDTree


def _default_n_jobs() -> int:
    """Auto-detect CPU core count, leave 1 free for the OS."""
    return max(1, (os.cpu_count() or 2) - 1)


# ── Participation Ratio ──────────────────────────────────────────────────


def _pr_single(vectors: np.ndarray, indices: np.ndarray, i: int, k: int) -> float:
    """Compute participation ratio for a single point (joblib helper).

    Uses the Gram matrix trick: when k << d (samples << dimensions),
    X @ X.T (k×k) has the same non-zero eigenvalues as X.T @ X (d×d).
    For k=30 neighbors in d=2304 dimensions, this is ~450,000x less work.
    The participation ratio result is identical — same eigenvalues, same formula.
    """
    nbrs = vectors[indices[i, 1:]]
    nbrs_centered = nbrs - nbrs.mean(axis=0)
    # Gram matrix (k×k) instead of covariance (d×d) — same non-zero eigenvalues
    gram = nbrs_centered @ nbrs_centered.T / (k - 1)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    if sum_eig_sq > 0:
        return (sum_eig ** 2) / sum_eig_sq
    return 0.0


def estimate_local_dim_pr(
    vectors: np.ndarray,
    k: int = 30,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Participation ratio of local PCA eigenvalues.

    For each point, takes k nearest neighbors, computes local covariance
    eigenvalues, and returns PR = (sum lambda_i)^2 / sum(lambda_i^2).
    Well-established: Gao & Ganguli 2017, widely used in neuroscience.
    """
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = _default_n_jobs()

    n = len(vectors)
    print(f"    Building KDTree for {n:,} points...")
    tree = KDTree(vectors)

    print(f"    Querying {k} nearest neighbors (workers={n_jobs})...")
    _, indices = tree.query(vectors, k=k + 1, workers=n_jobs)

    print(f"    Computing participation ratios (n_jobs={n_jobs})...")
    t_start = _time.monotonic()
    dims = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_pr_single)(vectors, indices, i, k) for i in range(n)
    )
    elapsed = _time.monotonic() - t_start
    m, s = divmod(int(elapsed), 60)
    print(f"    [100.0%] Done · {n:,} points in {m}m{s:02d}s")

    return np.array(dims, dtype=np.float32)


# ── TwoNN ────────────────────────────────────────────────────────────────


def estimate_local_dim_twonn(
    vectors: np.ndarray,
    k: int = 30,
    n_jobs: int | None = None,
) -> np.ndarray:
    """TwoNN-inspired local dimension estimate.

    For each point, uses the ratio of 2nd to 1st nearest neighbor distances.
    Based on Ansuini et al. NeurIPS 2019.
    """
    if n_jobs is None:
        n_jobs = _default_n_jobs()

    n = len(vectors)
    tree = KDTree(vectors)
    dists, indices = tree.query(vectors, k=k + 1, workers=n_jobs)

    r1 = dists[:, 1]
    r2 = dists[:, 2]

    mask = r1 > 1e-12
    mu = np.ones(n, dtype=np.float64)
    mu[mask] = r2[mask] / r1[mask]

    dims = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nbr_idx = indices[i, 1:k + 1]
        mu_local = mu[nbr_idx]
        mu_local = mu_local[mu_local > 1.0 + 1e-8]
        if len(mu_local) > 2:
            dims[i] = len(mu_local) / np.sum(np.log(mu_local))
        else:
            dims[i] = 1.0

    return dims


# ── Volume Growth Transform ──────────────────────────────────────────────


def _vgt_single(d_i: np.ndarray, n_radii: int, return_curve: bool) -> tuple[float, dict | None]:
    """Compute VGT for a single point (joblib helper)."""
    d_i = d_i[d_i > 0]
    empty_curve = {"log_r": [], "log_v": [], "slope": 0.0, "intercept": 0.0}

    if len(d_i) < 5:
        return 0.0, empty_curve if return_curve else None

    max_r = d_i[-1]
    min_r = d_i[0]
    if max_r <= min_r or min_r <= 0:
        return 0.0, empty_curve if return_curve else None

    radii = np.geomspace(min_r, max_r, n_radii)
    log_r = np.log(radii)
    log_v = np.array([np.log(max(np.searchsorted(d_i, r), 1)) for r in radii])

    # Filter out NaN/inf values from degenerate inputs
    finite = np.isfinite(log_r) & np.isfinite(log_v)
    valid = finite & (log_v > 0)
    if valid.sum() < 3:
        return 0.0, empty_curve if return_curve else None

    A = np.vstack([log_r[valid], np.ones(valid.sum())]).T
    try:
        result = np.linalg.lstsq(A, log_v[valid], rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, empty_curve if return_curve else None
    slope = result[0][0]
    intercept = result[0][1]
    dim = max(slope, 0.0)

    curve = None
    if return_curve:
        curve = {
            "log_r": log_r[valid].tolist(),
            "log_v": log_v[valid].tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
        }

    return dim, curve


def estimate_local_dim_vgt(
    vectors: np.ndarray,
    n_radii: int = 10,
    max_k: int = 50,
    return_curves: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Volume Growth Transform: log-volume vs log-radius slope.

    From Curry et al. 2025 (arXiv 2507.22010, Eq. 5).
    NOTE: This method is from a preprint not yet peer-reviewed.
    """
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = _default_n_jobs()

    n = len(vectors)
    print(f"    Building KDTree for {n:,} points...")
    tree = KDTree(vectors)
    print(f"    Querying {max_k} nearest neighbors (workers={n_jobs})...")
    dists, _ = tree.query(vectors, k=max_k + 1, workers=n_jobs)

    print(f"    Computing VGT growth curves (n_jobs={n_jobs})...")
    t_start = _time.monotonic()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_vgt_single)(dists[i, 1:], n_radii, return_curves) for i in range(n)
    )
    elapsed = _time.monotonic() - t_start
    m, s = divmod(int(elapsed), 60)
    print(f"    [100.0%] Done · {n:,} points in {m}m{s:02d}s")

    dims = np.array([r[0] for r in results], dtype=np.float32)

    if return_curves:
        curves = [r[1] for r in results]
        return dims, curves
    return dims


# ── Unified Interface ────────────────────────────────────────────────────


def estimate_local_dim(
    vectors: np.ndarray,
    method: str = "pr",
    **kwargs,
) -> np.ndarray:
    """Unified interface for local dimension estimation."""
    estimators = {
        "pr": estimate_local_dim_pr,
        "twonn": estimate_local_dim_twonn,
        "vgt": estimate_local_dim_vgt,
    }
    if method not in estimators:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(estimators.keys())}")

    print(f"  Estimating local dimension ({method})...")
    dims = estimators[method](vectors, **kwargs)
    print(f"  Local dim range: [{dims.min():.1f}, {dims.max():.1f}], mean: {dims.mean():.1f}")
    return dims
