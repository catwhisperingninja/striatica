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

import numpy as np
from scipy.spatial import KDTree


def estimate_local_dim_pr(
    vectors: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """Participation ratio of local PCA eigenvalues.

    For each point, takes k nearest neighbors, computes local covariance
    eigenvalues, and returns PR = (sum lambda_i)^2 / sum(lambda_i^2).
    Well-established: Gao & Ganguli 2017, widely used in neuroscience.
    """
    n = len(vectors)
    print(f"    Building KDTree for {n} points...")
    tree = KDTree(vectors)
    dims = np.zeros(n, dtype=np.float32)

    print(f"    Querying {k} nearest neighbors...")
    _, indices = tree.query(vectors, k=k + 1)  # +1 because query includes self

    log_interval = max(1, n // 20)  # report every 5%
    for i in range(n):
        if i % log_interval == 0:
            print(f"    [{i / n * 100:5.1f}%] Processing point {i}/{n}...")
        nbrs = vectors[indices[i, 1:]]  # exclude self
        nbrs_centered = nbrs - nbrs.mean(axis=0)
        cov = nbrs_centered.T @ nbrs_centered / (k - 1)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # numerical stability
        sum_eig = eigenvalues.sum()
        sum_eig_sq = (eigenvalues ** 2).sum()
        if sum_eig_sq > 0:
            dims[i] = (sum_eig ** 2) / sum_eig_sq
        else:
            dims[i] = 0.0
    print(f"    [100.0%] Done.")

    return dims


def estimate_local_dim_twonn(
    vectors: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """TwoNN-inspired local dimension estimate.

    For each point, uses the ratio of 2nd to 1st nearest neighbor distances.
    Based on Ansuini et al. NeurIPS 2019.
    """
    n = len(vectors)
    tree = KDTree(vectors)
    dists, indices = tree.query(vectors, k=k + 1)

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


def estimate_local_dim_vgt(
    vectors: np.ndarray,
    n_radii: int = 10,
    max_k: int = 50,
    return_curves: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Volume Growth Transform: log-volume vs log-radius slope.

    From Curry et al. 2025 (arXiv 2507.22010, Eq. 5).
    NOTE: This method is from a preprint not yet peer-reviewed.
    """
    n = len(vectors)
    print(f"    Building KDTree for {n} points...")
    tree = KDTree(vectors)
    print(f"    Querying {max_k} nearest neighbors...")
    dists, _ = tree.query(vectors, k=max_k + 1)

    dims = np.zeros(n, dtype=np.float32)
    curves = [] if return_curves else None

    log_interval = max(1, n // 20)  # report every 5%
    for i in range(n):
        if i % log_interval == 0:
            print(f"    [{i / n * 100:5.1f}%] Processing point {i}/{n}...")
        d_i = dists[i, 1:]
        d_i = d_i[d_i > 0]
        if len(d_i) < 5:
            dims[i] = 0.0
            if return_curves:
                curves.append({"log_r": [], "log_v": [], "slope": 0.0, "intercept": 0.0})
            continue

        max_r = d_i[-1]
        min_r = d_i[0]
        if max_r <= min_r or min_r <= 0:
            dims[i] = 0.0
            if return_curves:
                curves.append({"log_r": [], "log_v": [], "slope": 0.0, "intercept": 0.0})
            continue

        radii = np.geomspace(min_r, max_r, n_radii)
        log_r = np.log(radii)
        log_v = np.array([np.log(max(np.searchsorted(d_i, r), 1)) for r in radii])

        valid = log_v > 0
        if valid.sum() < 3:
            dims[i] = 0.0
            if return_curves:
                curves.append({"log_r": [], "log_v": [], "slope": 0.0, "intercept": 0.0})
            continue

        A = np.vstack([log_r[valid], np.ones(valid.sum())]).T
        result = np.linalg.lstsq(A, log_v[valid], rcond=None)
        slope = result[0][0]
        intercept = result[0][1]
        dims[i] = max(slope, 0.0)

        if return_curves:
            curves.append({
                "log_r": log_r[valid].tolist(),
                "log_v": log_v[valid].tolist(),
                "slope": float(slope),
                "intercept": float(intercept),
            })

    print(f"    [100.0%] Done.")

    if return_curves:
        return dims, curves
    return dims


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
