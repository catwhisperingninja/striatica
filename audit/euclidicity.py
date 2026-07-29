"""Euclidicity scores — topological singularity detection at multiple scales.

Port of the Euclidicity method of von Rohrscheidt & Rieck (2022),
"Topological Singularity Detection at Multiple Scales"
(https://arxiv.org/abs/2210.00069), implemented on gudhi's Vietoris-Rips
machinery per the Bite 1 plan (img/correction/v4_stratgeofocus-replace-umap/
zOld-plan-splitbites/bite-01-euclidicity-audit-plan.md, sections 2.1-2.2).

For each point x and each neighborhood radius r (spec 2.1):

  1. Extract the local neighborhood N = {p : d(x, p) <= r, p != x}.
     If |N| < k_neighbors the (point, radius) cell is np.nan
     (spec 3.1 ``n_neighbors_min`` semantics).
  2. Estimate the intrinsic dimension d_hat of N by local PCA
     (cumulative explained variance >= DIM_EVR_THRESHOLD).
  3. Form the PUNCTURED neighborhood (annulus): points with
     ANNULUS_INNER_FRACTION * r < d(x, p) <= r.  The annulus is the finite
     sample of the *link* of x — the object whose topology distinguishes a
     manifold point (link ~ S^{d-1}) from a singular one (von Rohrscheidt &
     Rieck use exactly this punctured-ball comparison; a full ball is
     contractible at every point and carries no singularity signal).
  4. Compute Vietoris-Rips persistent homology (H0, and H1 when d_hat >= 2)
     of the annulus, truncated at 2r.
  5. Compare against MODEL_DRAWS seeded uniform samples of the model
     Euclidean d_hat-annulus with the SAME point count and radii
     (spec 2.1 step 3/4: "model ball of the same intrinsic dimension and
     number of points"):
       - H0: trimmed mean of |log death_actual - log death_model| over the
         sorted death profiles (births are all 0, class counts match by
         construction, so sorted matching is the optimal transport plan).
         Log space makes the comparison scale-free and sensitive to
         density RATIOS: a lower-dimensional stratum passing through the
         neighborhood contributes a block of deaths several times smaller
         than the model's, which survives averaging in log space but is
         washed out in absolute units.  Deaths are clamped from below at a
         fraction of the model's median death so near-duplicate sample
         points cannot blow up the log, and the top decile of per-class
         differences is trimmed for outlier robustness.
       - H1: bottleneck distance (gudhi.bottleneck_distance) between
         persistence-denoised diagrams (classes with persistence
         < H1_DENOISE_FRAC * r are sampling noise on ~10-60 point clouds).
  6. Calibrate against the sampling-noise floor: the same distances computed
     BETWEEN independent model draws (model-vs-model) estimate the expected
     deviation of a genuinely Euclidean sample at this (n, d_hat, r).  The
     calibrated deviation is max(0, actual - null) per homology dimension,
     normalized by r.
  7. Euclidicity score = clip(1 - H0_GAIN*dev_H0 - H1_GAIN*dev_H1, 0, 1)
     (spec 2.1 step 4: "1 - bottleneck_distance(...), normalized to [0,1]").
     ~1.0 = neighborhood indistinguishable from a Euclidean d_hat-ball;
     ~0.0 = maximally singular.

Determinism: this implementation is FULLY deterministic in (X, k_neighbors,
radii) — model-annulus draws are seeded from a fixed per-cell SeedSequence
(_MODEL_SEED_ROOT, point index, radius index), independent of the ``seed``
argument.  The API contract explicitly allows a fully deterministic
implementation to ignore ``seed``; identical calls are bit-identical
(spec 4.2 B8 hash stability) and cross-seed correlation is exactly 1.0
(spec 4.1 robustness, requirement r > 0.95).

Library module: importing this file performs NO computation and has no CLI
side effects.  The runnable entry point lives in audit/run_audit.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

import gudhi

# ── Tunables (all deviations are normalized by the neighborhood radius) ──

#: Inner radius of the punctured neighborhood, as a fraction of r.
ANNULUS_INNER_FRACTION = 0.4

#: Cumulative explained-variance threshold for the local-PCA intrinsic
#: dimension estimate (spec 2.1 step 3).
DIM_EVR_THRESHOLD = 0.95

#: Number of seeded model-annulus samples per cell.  Actual-vs-model
#: distances are averaged over all draws; model-vs-model distances over all
#: pairs estimate the sampling-noise floor.
MODEL_DRAWS = 4

#: Gain on the calibrated H0 log-death-profile deviation (log-ratio units).
H0_GAIN = 1.2

#: Gain on the calibrated, r-normalized H1 bottleneck deviation.
H1_GAIN = 3.0

#: Deaths are clamped from below at this fraction of the model's median
#: death before the log comparison (near-duplicate guard).
H0_DEATH_FLOOR_FRAC = 0.1

#: Fraction of the largest per-class log differences trimmed before
#: averaging (outlier robustness).
H0_TRIM_FRAC = 0.1

#: H1 classes with persistence below this fraction of r are treated as
#: finite-sample noise and removed before the bottleneck comparison.
H1_DENOISE_FRAC = 0.2

#: Cap on the model dimension (persistence is computed with maxdim 1, so
#: higher ambient model dimensions only affect the sampled geometry).
MAX_MODEL_DIM = 8

#: Minimum annulus occupancy; below this the cell falls back to comparing
#: the full punctured ball against a model ball (inner fraction 0).
MIN_ANNULUS_POINTS = 3

#: Fixed root entropy for per-cell model sampling (deterministic by design;
#: the public ``seed`` argument is intentionally ignored — see module
#: docstring).
_MODEL_SEED_ROOT = 0x5EEDBA11


# ── Internal helpers ─────────────────────────────────────────────────────

def _intrinsic_dimension(points: np.ndarray) -> int:
    """Local-PCA intrinsic dimension: smallest d with cumulative explained
    variance >= DIM_EVR_THRESHOLD (spec 2.1 step 3)."""
    centered = points - points.mean(axis=0, keepdims=True)
    # Eigenvalues of the covariance, via singular values (deterministic).
    svals = np.linalg.svd(centered, compute_uv=False)
    var = svals**2
    total = float(var.sum())
    if total <= 0.0:
        return 1
    ratios = np.cumsum(var) / total
    dim = int(np.searchsorted(ratios, DIM_EVR_THRESHOLD - 1e-12) + 1)
    dim = max(1, min(dim, points.shape[1], MAX_MODEL_DIM, max(1, len(points) - 1)))
    return dim


def _rips_diagrams(points: np.ndarray, trunc: float, want_h1: bool):
    """Vietoris-Rips persistence of a point cloud, truncated at ``trunc``.

    Returns (h0_deaths_sorted, h1_diagram) where h0_deaths_sorted is a
    1-D float array (infinite deaths replaced by ``trunc``) and h1_diagram
    is an (m, 2) array (empty when want_h1 is False).
    """
    n = len(points)
    if n == 0:
        return np.empty(0), np.empty((0, 2))
    rips = gudhi.RipsComplex(points=points, max_edge_length=trunc)
    st = rips.create_simplex_tree(max_dimension=2 if want_h1 else 1)
    st.compute_persistence()
    h0 = st.persistence_intervals_in_dimension(0)
    if len(h0) == 0:  # pragma: no cover — H0 always has >= 1 class
        deaths = np.empty(0)
    else:
        deaths = np.minimum(np.where(np.isinf(h0[:, 1]), trunc, h0[:, 1]), trunc)
        deaths = np.sort(deaths)
    if want_h1:
        h1 = st.persistence_intervals_in_dimension(1)
        if len(h1) == 0:
            h1 = np.empty((0, 2))
        else:
            h1 = np.column_stack(
                [h1[:, 0], np.minimum(np.where(np.isinf(h1[:, 1]), trunc, h1[:, 1]), trunc)]
            )
    else:
        h1 = np.empty((0, 2))
    return deaths, h1


def _h0_distance(deaths_a: np.ndarray, deaths_b: np.ndarray, floor: float) -> float:
    """Trimmed mean |log death_a - log death_b| over sorted death profiles.

    Class counts match whenever both clouds have the same cardinality (Rips
    H0 has exactly one class per point), making sorted matching the optimal
    transport plan; unequal counts (defensive) are padded at the small end
    with floor-persistence classes.  ``floor`` clamps deaths from below so
    near-duplicate points cannot blow up the log difference."""
    la, lb = len(deaths_a), len(deaths_b)
    if la == 0 and lb == 0:
        return 0.0
    m = max(la, lb)
    a = np.full(m, floor)
    b = np.full(m, floor)
    a[m - la:] = deaths_a  # sorted ascending
    b[m - lb:] = deaths_b
    diffs = np.abs(np.log(np.maximum(a, floor)) - np.log(np.maximum(b, floor)))
    diffs = np.sort(diffs)
    keep = max(1, int(np.ceil(len(diffs) * (1.0 - H0_TRIM_FRAC))))
    return float(np.mean(diffs[:keep]))


def _h1_distance(diag_a: np.ndarray, diag_b: np.ndarray, r: float) -> float:
    """Bottleneck distance between persistence-denoised H1 diagrams."""
    thresh = H1_DENOISE_FRAC * r
    a = diag_a[(diag_a[:, 1] - diag_a[:, 0]) >= thresh] if len(diag_a) else diag_a
    b = diag_b[(diag_b[:, 1] - diag_b[:, 0]) >= thresh] if len(diag_b) else diag_b
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return float(gudhi.bottleneck_distance(a.tolist(), b.tolist()))


def _sample_model_annulus(
    rng: np.random.Generator, n: int, dim: int, inner_frac: float, r: float
) -> np.ndarray:
    """n points uniform on the Euclidean annulus {a <= |u| <= r} in R^dim,
    a = inner_frac * r (inner_frac 0 gives the full model ball)."""
    directions = rng.standard_normal((n, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions /= norms
    a_d = (inner_frac * r) ** dim
    u = rng.uniform(size=(n, 1))
    radial = (a_d + u * (r**dim - a_d)) ** (1.0 / dim)
    return directions * radial


def _score_cell(
    ball_points: np.ndarray, center: np.ndarray, r: float, cell_key: tuple[int, int]
) -> float:
    """Euclidicity score for one (point, radius) cell.  ``ball_points`` are
    the neighbors within r (center excluded), in deterministic index order."""
    offsets = ball_points - center
    dists = np.linalg.norm(offsets, axis=1)
    dim = _intrinsic_dimension(ball_points)

    inner = ANNULUS_INNER_FRACTION * r
    mask = dists > inner
    if int(mask.sum()) >= MIN_ANNULUS_POINTS:
        annulus = offsets[mask]
        inner_frac = ANNULUS_INNER_FRACTION
    else:
        annulus = offsets
        inner_frac = 0.0

    n_ann = len(annulus)
    trunc = 2.0 * r
    want_h1 = dim >= 2

    a_h0, a_h1 = _rips_diagrams(annulus, trunc, want_h1)

    seed_seq = np.random.SeedSequence([_MODEL_SEED_ROOT, cell_key[0], cell_key[1]])
    model_diags = []
    for child in seed_seq.spawn(MODEL_DRAWS):
        rng = np.random.default_rng(child)
        pts = _sample_model_annulus(rng, n_ann, dim, inner_frac, r)
        model_diags.append(_rips_diagrams(pts, trunc, want_h1))

    # Near-duplicate log clamp, from the model's own death scale.
    model_deaths = np.concatenate([m0 for m0, _ in model_diags])
    floor = H0_DEATH_FLOOR_FRAC * float(np.median(model_deaths))
    floor = max(floor, 1e-12 * trunc)

    # Actual-vs-model deviation, averaged over draws.
    d0_act = float(np.mean([_h0_distance(a_h0, m0, floor) for m0, _ in model_diags]))
    d1_act = float(np.mean([_h1_distance(a_h1, m1, r) for _, m1 in model_diags]))

    # Null (sampling-noise) floor: model-vs-model over all pairs.
    d0_null_terms = []
    d1_null_terms = []
    for i in range(MODEL_DRAWS):
        for j in range(i + 1, MODEL_DRAWS):
            d0_null_terms.append(
                _h0_distance(model_diags[i][0], model_diags[j][0], floor)
            )
            d1_null_terms.append(_h1_distance(model_diags[i][1], model_diags[j][1], r))
    d0_null = float(np.mean(d0_null_terms))
    d1_null = float(np.mean(d1_null_terms))

    deviation = H0_GAIN * max(0.0, d0_act - d0_null) + H1_GAIN * (
        max(0.0, d1_act - d1_null) / r
    )
    return float(np.clip(1.0 - deviation, 0.0, 1.0))


# ── Public API ───────────────────────────────────────────────────────────

def euclidicity_scores(
    X: np.ndarray,
    k_neighbors: int,
    radii: Sequence[float],
    *,
    seed: int | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Per-point, per-radius Euclidicity scores (spec 2.1-2.2).

    Parameters
    ----------
    X : (n_points, d) float array.
    k_neighbors : minimum neighborhood size.  A (point, radius) cell whose
        radius-ball (excluding the point itself) contains fewer than
        ``k_neighbors`` points is np.nan (spec 3.1 n_neighbors_min).
    radii : neighborhood radii; output column j corresponds to ``radii[j]``.
        Euclidean units by default; cosine-distance units for
        ``metric="cosine"``.
    seed : accepted per the API contract; this implementation is fully
        deterministic in (X, k_neighbors, radii) and ignores it (allowed —
        see module docstring).  Identical calls are bit-identical.
    metric : "euclidean" (default — required for the geometric controls) or
        "cosine" (production SAE decoder vectors, spec 2.2/4.2 B2).  Cosine
        mode L2-normalizes X and converts each cosine-distance radius c to
        its chord equivalent sqrt(2c), which is an exact, monotone
        reparameterization on the unit sphere.

    Returns
    -------
    (n_points, len(radii)) float64 array.  Finite entries lie in [0, 1]
    (~1.0 = Euclidean-ball-like, ~0.0 = singular); np.nan marks
    insufficient neighbors.
    """
    del seed  # deterministic implementation — see module docstring
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_points, d); got shape {X.shape}")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X = X / norms
        eff_radii = [float(np.sqrt(2.0 * float(c))) for c in radii]
    elif metric == "euclidean":
        eff_radii = [float(r) for r in radii]
    else:
        raise ValueError(f"unsupported metric: {metric!r}")
    if any(r <= 0.0 for r in eff_radii):
        raise ValueError("all radii must be positive")

    n = X.shape[0]
    scores = np.full((n, len(eff_radii)), np.nan, dtype=np.float64)
    tree = cKDTree(X)

    for j, r in enumerate(eff_radii):
        neighbor_lists = tree.query_ball_point(X, r)
        for i in range(n):
            idx = np.array(sorted(t for t in neighbor_lists[i] if t != i), dtype=int)
            if len(idx) < k_neighbors:
                continue
            scores[i, j] = _score_cell(X[idx], X[i], r, (i, j))
    return scores
