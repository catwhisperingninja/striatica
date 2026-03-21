# striatica/pipeline/cluster.py
"""HDBSCAN clustering of 3D point cloud."""

import numpy as np
from hdbscan import HDBSCAN


def cluster_points(
    coords: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    """Cluster 3D coordinates using HDBSCAN.

    Args:
        coords: (n, 3) array of 3D positions
        min_cluster_size: minimum points per cluster
        min_samples: HDBSCAN min_samples parameter

    Returns:
        (n,) int array of cluster labels (-1 = noise/uncategorized)
    """
    if not np.all(np.isfinite(coords)):
        n_bad = (~np.isfinite(coords)).any(axis=1).sum()
        raise ValueError(
            f"Input coords contain NaN/Inf values ({n_bad} rows). "
            f"Check dimensionality reduction output."
        )

    print(f"  HDBSCAN clustering {len(coords)} points...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    return labels
