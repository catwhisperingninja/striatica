# striatica/pipeline/reduce.py
"""Dimensionality reduction: PCA -> UMAP -> 3D."""

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


def reduce_to_3d(
    vectors: np.ndarray,
    pca_dim: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce high-dimensional vectors to 3D coordinates.

    Pipeline: PCA (768D -> pca_dim) -> UMAP (pca_dim -> 3D)

    Args:
        vectors: (n_features, hidden_dim) array
        pca_dim: intermediate PCA dimensions
        random_state: for reproducibility

    Returns:
        (n_features, 3) float32 array of 3D coordinates
    """
    print(f"  PCA: {vectors.shape[1]}D -> {pca_dim}D...")
    pca = PCA(n_components=pca_dim, random_state=random_state)
    reduced = pca.fit_transform(vectors)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    print(f"  UMAP: {pca_dim}D -> 3D (this is the slow step)...")
    umap = UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    coords = umap.fit_transform(reduced).astype(np.float32)

    # Center and scale to [-1, 1] range (copy to avoid mutating UMAP output)
    coords = coords - coords.mean(axis=0)
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs

    return coords
