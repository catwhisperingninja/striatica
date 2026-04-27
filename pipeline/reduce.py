# striatica/pipeline/reduce.py
"""Dimensionality reduction: PCA -> UMAP -> 3D."""

from __future__ import annotations

import numpy as np


def auto_pca_dim(input_dim: int) -> int:
    """Select PCA dimension adaptively based on input dimensionality.

    Heuristic: min(input_dim // 4, 300).
    - 768-D (GPT-2 Small) -> 192
    - 2304-D (Gemma 2 transcoder) -> 300 (capped)
    - 4096-D (large models) -> 300 (capped)

    Cap at 300: UMAP's approximate nearest-neighbor handles 300-D well;
    beyond that, diminishing returns and memory cost.
    """
    return min(input_dim // 4, 300)


def reduce_to_3d(
    vectors: np.ndarray,
    pca_dim: int | str = 50,
    random_state: int = 42,
    n_jobs: int = -1,
    return_pca_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Reduce high-dimensional vectors to 3D coordinates.

    Pipeline: PCA (768D -> pca_dim) -> UMAP (pca_dim -> 3D)

    Args:
        vectors: (n_features, hidden_dim) array
        pca_dim: intermediate PCA dimensions
        random_state: for reproducibility
        n_jobs: CPU cores for UMAP (-1 = all cores)
        return_pca_variance: if True, return (coords, explained_variance_ratio)

    Returns:
        (n_features, 3) float32 array of 3D coordinates, or
        (coords, pca_variance) tuple if return_pca_variance=True
    """
    from sklearn.decomposition import PCA
    from umap import UMAP

    if not np.all(np.isfinite(vectors)):
        n_bad = (~np.isfinite(vectors)).any(axis=1).sum()
        raise ValueError(
            f"Input vectors contain NaN/Inf values ({n_bad} rows affected). "
            f"Check upstream data pipeline."
        )

    # Resolve 'auto' pca_dim
    if pca_dim == "auto":
        pca_dim = auto_pca_dim(vectors.shape[1])
        print(f"  Auto pca_dim: {pca_dim} (from {vectors.shape[1]}-D input)")

    print(f"  PCA: {vectors.shape[1]}D -> {pca_dim}D...")
    pca = PCA(n_components=pca_dim, random_state=random_state)
    reduced = pca.fit_transform(vectors)
    pca_variance = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {pca_variance:.2%}")

    print(f"  UMAP: {pca_dim}D -> 3D (this is the slow step)...")
    umap = UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        verbose=True,
        n_jobs=n_jobs,
    )
    coords = umap.fit_transform(reduced).astype(np.float32)

    if not np.all(np.isfinite(coords)):
        n_bad = (~np.isfinite(coords)).any(axis=1).sum()
        raise ValueError(
            f"UMAP output contains NaN/Inf values ({n_bad} rows). "
            f"This usually means degenerate input vectors."
        )

    # Center and scale to [-1, 1] range (copy to avoid mutating UMAP output)
    coords = coords - coords.mean(axis=0)
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs

    if return_pca_variance:
        return coords, pca_variance
    return coords
