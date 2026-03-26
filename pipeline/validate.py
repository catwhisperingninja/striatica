# striatica/pipeline/validate.py
"""Pipeline validation suite: structural integrity, embedding quality, cross-model comparison.

Three levels:
  L1 — Structural integrity (JSON dict or arrays). Hard gate: fails pipeline if broken.
  L2 — Embedding quality (needs high-D vectors + 3D coords). Scorecard with warnings.
  L3 — Cross-model distributional comparison (optional, needs reference JSON).

Zero new dependencies — uses scikit-learn, scipy, numpy already in the pinned chain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ── Data Structures ────────────────────────────────────────────────────


@dataclass
class Check:
    """Single validation check result."""
    name: str
    passed: bool
    message: str = ""
    value: Any = None  # metric value (for scorecards)


@dataclass
class ValidationReport:
    """Validation results container."""
    level: int
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add(self, name: str, passed: bool, message: str = "", value: Any = None) -> None:
        self.checks.append(Check(name=name, passed=passed, message=message, value=value))
        if not passed and self.level == 1:
            # L1 failures are also warnings
            self.warnings.append(f"{name}: {message}")

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def print_scorecard(self) -> None:
        """Print a human-readable scorecard."""
        print(f"\n  ── Validation Level {self.level} ({'PASS' if self.passed else 'FAIL'}) "
              f"({self.elapsed_seconds:.1f}s) ──")
        for c in self.checks:
            status = "✓" if c.passed else "✗"
            val_str = f" = {c.value}" if c.value is not None else ""
            msg_str = f"  ({c.message})" if c.message and not c.passed else ""
            print(f"  {status} {c.name}{val_str}{msg_str}")
        if self.warnings:
            print(f"  ⚠  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                print(f"     - {w}")
        print()

    def to_dict(self) -> dict:
        """Serialize for JSON sidecar."""
        return {
            "level": self.level,
            "passed": self.passed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "checks": {
                c.name: {
                    "passed": c.passed,
                    "value": _safe_value(c.value),
                    "message": c.message,
                }
                for c in self.checks
            },
            "warnings": self.warnings,
        }


class ValidationError(Exception):
    """Raised when Level 1 validation fails (hard gate)."""
    def __init__(self, report: ValidationReport):
        self.report = report
        failures = [c for c in report.checks if not c.passed]
        msg = f"L1 validation failed ({len(failures)} check(s)):\n"
        for c in failures:
            msg += f"  ✗ {c.name}: {c.message}\n"
        super().__init__(msg)


# ── Level 1: Structural Integrity ──────────────────────────────────────


def validate_level1_arrays(
    coords: np.ndarray,
    labels: np.ndarray,
    local_dims: np.ndarray | None = None,
    growth_curves: list[dict] | None = None,
) -> ValidationReport:
    """Level 1 validation on raw arrays (before JSON assembly).

    This is the pipeline gate — runs between Step 5 and Step 6.
    """
    t0 = time.monotonic()
    report = ValidationReport(level=1)
    n = len(coords)

    # L1-SHAPE: coords must be (n, 3)
    report.add(
        "L1-SHAPE",
        coords.ndim == 2 and coords.shape[1] == 3,
        f"coords shape {coords.shape}, expected (n, 3)",
    )

    # L1-LABELS: labels length matches
    report.add(
        "L1-LABELS",
        len(labels) == n,
        f"labels length {len(labels)} != coords length {n}",
    )

    # L1-FINITE: all coords finite
    n_nan = int(np.isnan(coords).sum())
    n_inf = int(np.isinf(coords).sum())
    report.add(
        "L1-FINITE",
        n_nan == 0 and n_inf == 0,
        f"{n_nan} NaN, {n_inf} inf values in coords",
    )

    # L1-RANGE: all coords in [-1.001, 1.001] (float tolerance)
    max_abs = float(np.abs(coords).max()) if np.all(np.isfinite(coords)) else float("inf")
    report.add(
        "L1-RANGE",
        max_abs <= 1.001,
        f"max |coord| = {max_abs:.6f}, exceeds [-1, 1] range",
        value=round(max_abs, 6),
    )

    # L1-LABELS-VALID: all labels >= -1
    min_label = int(labels.min())
    report.add(
        "L1-LABELS-VALID",
        min_label >= -1,
        f"min label = {min_label}, expected >= -1",
    )

    # L1-DIMS: local dimensions length matches (if present)
    if local_dims is not None:
        report.add(
            "L1-DIMS",
            len(local_dims) == n,
            f"local_dims length {len(local_dims)} != {n}",
        )

        # L1-DIM-BOUNDS: local dims in [0, 100]
        if len(local_dims) == n and len(local_dims) > 0:
            dim_min = float(local_dims.min())
            dim_max = float(local_dims.max())
            report.add(
                "L1-DIM-BOUNDS",
                dim_min >= 0 and dim_max <= 100,
                f"local dims range [{dim_min:.2f}, {dim_max:.2f}], expected [0, 100]",
                value=f"[{dim_min:.2f}, {dim_max:.2f}]",
            )

    # L1-CURVES: growth curves length matches (if present)
    if growth_curves is not None:
        report.add(
            "L1-CURVES",
            len(growth_curves) == n,
            f"growth_curves length {len(growth_curves)} != {n}",
        )

    # L1-SPREAD: check for collapsed axes (all points on a plane/line)
    if np.all(np.isfinite(coords)):
        std_per_axis = coords.std(axis=0)
        min_std = float(std_per_axis.min())
        report.add(
            "L1-SPREAD",
            min_std > 0.01,
            f"axis std devs {std_per_axis.tolist()}, min={min_std:.4f} — possible axis collapse",
            value=[round(float(s), 4) for s in std_per_axis],
        )

    report.elapsed_seconds = time.monotonic() - t0
    return report


def validate_level1_json(result: dict) -> ValidationReport:
    """Level 1 validation on a JSON dict (after assembly or loaded from disk).

    Used by `striat validate` CLI command.
    """
    t0 = time.monotonic()
    report = ValidationReport(level=1)

    n = result.get("numFeatures", 0)

    # L1-SHAPE
    positions = result.get("positions", [])
    report.add("L1-SHAPE", len(positions) == n * 3,
               f"positions length {len(positions)}, expected {n * 3}")

    # L1-LABELS
    labels = result.get("clusterLabels", [])
    report.add("L1-LABELS", len(labels) == n,
               f"clusterLabels length {len(labels)}, expected {n}")

    # L1-FEATURES
    features = result.get("features", [])
    report.add("L1-FEATURES", len(features) == n,
               f"features length {len(features)}, expected {n}")

    # L1-FINITE
    pos_arr = np.array(positions, dtype=np.float64)
    n_nan = int(np.isnan(pos_arr).sum())
    n_inf = int(np.isinf(pos_arr).sum())
    report.add("L1-FINITE", n_nan == 0 and n_inf == 0,
               f"{n_nan} NaN, {n_inf} inf in positions")

    # L1-RANGE
    if len(positions) > 0 and n_nan == 0 and n_inf == 0:
        max_abs = float(np.abs(pos_arr).max())
        report.add("L1-RANGE", max_abs <= 1.001,
                    f"max |position| = {max_abs:.6f}", value=round(max_abs, 6))

    # L1-LABELS-VALID
    if labels:
        min_label = min(labels)
        report.add("L1-LABELS-VALID", min_label >= -1,
                    f"min label = {min_label}")

    # L1-INDICES
    if features:
        indices = [f.get("index", -1) for f in features]
        expected = list(range(n))
        report.add("L1-INDICES", indices == expected,
                    f"feature indices not contiguous 0..{n - 1}")

    # L1-DIMS (if present)
    local_dims = result.get("localDimensions")
    if local_dims is not None:
        report.add("L1-DIMS", len(local_dims) == n,
                    f"localDimensions length {len(local_dims)}, expected {n}")
        if local_dims:
            dim_arr = np.array(local_dims)
            dim_min, dim_max = float(dim_arr.min()), float(dim_arr.max())
            report.add("L1-DIM-BOUNDS", dim_min >= 0 and dim_max <= 100,
                        f"range [{dim_min:.2f}, {dim_max:.2f}]",
                        value=f"[{dim_min:.2f}, {dim_max:.2f}]")

    # L1-CURVES (if present)
    curves = result.get("growthCurves")
    if curves is not None:
        report.add("L1-CURVES", len(curves) == n,
                    f"growthCurves length {len(curves)}, expected {n}")

    # L1-CENTROID: verify cluster centroids match computed means
    if labels and positions and len(positions) == n * 3:
        coords_arr = np.array(positions).reshape(-1, 3)
        labels_arr = np.array(labels)
        clusters = result.get("clusters", [])
        centroid_ok = True
        centroid_msg = ""
        for cl in clusters:
            cid = cl["id"]
            mask = labels_arr == cid
            if mask.sum() == 0:
                continue
            expected_centroid = coords_arr[mask].mean(axis=0)
            actual_centroid = np.array(cl["centroid"])
            diff = float(np.abs(expected_centroid - actual_centroid).max())
            if diff > 1e-3:
                centroid_ok = False
                centroid_msg = f"cluster {cid} centroid off by {diff:.6f}"
                break
        report.add("L1-CENTROID", centroid_ok, centroid_msg)

    report.elapsed_seconds = time.monotonic() - t0
    return report


# ── Level 2: Embedding Quality ─────────────────────────────────────────


def validate_level2(
    vectors: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    local_dims: np.ndarray | None = None,
    frac_nonzero: np.ndarray | None = None,
    pca_explained_variance: float | None = None,
    subsample_limit: int = 20_000,
) -> ValidationReport:
    """Level 2 embedding quality scorecard.

    Computes trustworthiness, neighborhood overlap, silhouette, and spread
    metrics. Does NOT fail the pipeline — produces warnings.

    Args:
        vectors: (n, d) original high-dimensional vectors
        coords: (n, 3) UMAP output coordinates
        labels: (n,) HDBSCAN cluster labels
        local_dims: (n,) local dimension estimates (optional)
        frac_nonzero: (n,) activation frequencies (optional, for correlation)
        pca_explained_variance: total PCA variance ratio (optional)
        subsample_limit: subsample for expensive metrics if n > this
    """
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    t0 = time.monotonic()
    report = ValidationReport(level=2)
    n = len(vectors)

    # Subsample for expensive operations on large datasets
    if n > subsample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, subsample_limit, replace=False)
        idx.sort()
        vecs_sub = vectors[idx]
        coords_sub = coords[idx]
        labels_sub = labels[idx]
        local_dims_sub = local_dims[idx] if local_dims is not None else None
        frac_sub = frac_nonzero[idx] if frac_nonzero is not None else None
        print(f"    Subsampled {n:,} → {subsample_limit:,} for L2 metrics")
    else:
        vecs_sub, coords_sub, labels_sub = vectors, coords, labels
        local_dims_sub = local_dims
        frac_sub = frac_nonzero

    n_sub = len(vecs_sub)

    # L2-TRUST: sklearn trustworthiness at k=30
    print("    Computing trustworthiness (k=30)...")
    k_trust = min(30, n_sub - 1)
    trust = float(trustworthiness(vecs_sub, coords_sub, n_neighbors=k_trust, metric="cosine"))
    passed = trust >= 0.85
    report.add("L2-TRUST", passed,
               f"trustworthiness={trust:.4f}, threshold=0.85",
               value=round(trust, 4))
    if not passed:
        report.add_warning(f"Low trustworthiness ({trust:.4f}) — 3D neighbors may not reflect high-D structure")

    # L2-NBHD: Neighborhood overlap at k=10, 30, 50
    print("    Computing neighborhood overlap...")
    nn_highd = NearestNeighbors(metric="cosine", n_jobs=-1)
    nn_highd.fit(vecs_sub)
    nn_3d = NearestNeighbors(metric="euclidean", n_jobs=-1)
    nn_3d.fit(coords_sub)

    for k in [10, 30, 50]:
        if k >= n_sub:
            continue
        _, idx_hd = nn_highd.kneighbors(vecs_sub, n_neighbors=k + 1)
        _, idx_3d = nn_3d.kneighbors(coords_sub, n_neighbors=k + 1)
        # Exclude self (index 0 in result)
        idx_hd = idx_hd[:, 1:]
        idx_3d = idx_3d[:, 1:]

        overlaps = np.array([
            len(set(idx_hd[i]) & set(idx_3d[i])) / k
            for i in range(n_sub)
        ])
        mean_overlap = float(overlaps.mean())

        thresholds = {10: 0.15, 30: 0.10, 50: 0.08}
        threshold = thresholds.get(k, 0.08)
        passed = mean_overlap >= threshold
        report.add(f"L2-NBHD-{k}", passed,
                   f"overlap={mean_overlap:.4f}, threshold={threshold}",
                   value=round(mean_overlap, 4))
        if not passed:
            report.add_warning(f"Low neighborhood overlap at k={k} ({mean_overlap:.4f})")

    # L2-SILHOUETTE: cluster quality in 3D
    # Only compute if we have >1 cluster and not all noise
    unique_labels = set(labels_sub)
    non_noise_labels = unique_labels - {-1}
    if len(non_noise_labels) >= 2:
        from sklearn.metrics import silhouette_score
        # Exclude noise points for silhouette
        mask = labels_sub != -1
        if mask.sum() > 10:
            print("    Computing silhouette score...")
            sil = float(silhouette_score(coords_sub[mask], labels_sub[mask]))
            passed = sil >= 0.05
            report.add("L2-SILHOUETTE", passed,
                       f"silhouette={sil:.4f}, threshold=0.05",
                       value=round(sil, 4))
            if not passed:
                report.add_warning(f"Low silhouette score ({sil:.4f}) — clusters may overlap in 3D")
        else:
            report.add("L2-SILHOUETTE", True, "too few non-noise points to compute", value=None)
    else:
        report.add("L2-SILHOUETTE", True, "fewer than 2 clusters, skipped", value=None)

    # L2-PCA-VAR: PCA explained variance (informational)
    if pca_explained_variance is not None:
        passed = pca_explained_variance >= 0.30
        report.add("L2-PCA-VAR", passed,
                   f"PCA variance={pca_explained_variance:.4f}",
                   value=round(pca_explained_variance, 4))
        if not passed:
            report.add_warning(f"PCA captures only {pca_explained_variance:.1%} of variance at 50 components")

    # L2-DIM-CORR: correlation between activation frequency and local dimension
    if local_dims_sub is not None and frac_sub is not None:
        from scipy.stats import pearsonr
        # Filter out zero/nan values
        valid = np.isfinite(local_dims_sub) & np.isfinite(frac_sub) & (frac_sub > 0)
        if valid.sum() > 10:
            r, _ = pearsonr(frac_sub[valid], local_dims_sub[valid])
            r = float(r)
            passed = abs(r) <= 0.5
            report.add("L2-DIM-CORR", passed,
                       f"|r|={abs(r):.4f} between fracNonzero and localDim",
                       value=round(r, 4))
            if not passed:
                report.add_warning(f"High dim-activation correlation (r={r:.4f}) — possible contamination")

    # L2-SPREAD: std dev per axis
    std_per_axis = coords_sub.std(axis=0)
    min_std = float(std_per_axis.min())
    report.add("L2-SPREAD", min_std > 0.05,
               f"axis stds={[round(float(s), 4) for s in std_per_axis]}, min={min_std:.4f}",
               value=[round(float(s), 4) for s in std_per_axis])
    if min_std <= 0.05:
        report.add_warning(f"Near-zero axis spread ({min_std:.4f}) — possible dimensional collapse")

    report.elapsed_seconds = time.monotonic() - t0
    return report


# ── Level 3: Cross-Model Comparison ────────────────────────────────────


def validate_level3(
    result: dict,
    reference_path: Path,
) -> ValidationReport:
    """Level 3 cross-model distributional comparison.

    Compares statistical signatures between a new result and a reference JSON.
    Does NOT compare positions (non-comparable across UMAP runs).
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    t0 = time.monotonic()
    report = ValidationReport(level=3)

    with open(reference_path) as f:
        ref = json.load(f)

    # L3-CLUSTER-COUNT: ratio of cluster counts
    new_labels = np.array(result["clusterLabels"])
    ref_labels = np.array(ref["clusterLabels"])
    new_n_clusters = len(set(new_labels)) - (1 if -1 in new_labels else 0)
    ref_n_clusters = len(set(ref_labels)) - (1 if -1 in ref_labels else 0)

    if ref_n_clusters > 0:
        ratio = new_n_clusters / ref_n_clusters
        passed = 0.3 <= ratio <= 3.0
        report.add("L3-CLUSTER-COUNT", passed,
                   f"new={new_n_clusters}, ref={ref_n_clusters}, ratio={ratio:.2f}",
                   value={"new": new_n_clusters, "ref": ref_n_clusters, "ratio": round(ratio, 2)})

    # L3-UNCAT-FRAC: uncategorized fraction difference
    new_uncat = float((new_labels == -1).sum() / len(new_labels))
    ref_uncat = float((ref_labels == -1).sum() / len(ref_labels))
    diff = abs(new_uncat - ref_uncat)
    report.add("L3-UNCAT-FRAC", diff <= 0.20,
               f"new={new_uncat:.2%}, ref={ref_uncat:.2%}, diff={diff:.2%}",
               value={"new": round(new_uncat, 4), "ref": round(ref_uncat, 4)})

    # L3-DIM-DIST: KS test on local dimension distributions
    new_dims = result.get("localDimensions")
    ref_dims = ref.get("localDimensions")
    if new_dims is not None and ref_dims is not None:
        stat, p = ks_2samp(new_dims, ref_dims)
        report.add("L3-DIM-DIST", True,  # always informational
                   f"KS stat={stat:.4f}, p={p:.4e}",
                   value={"ks_stat": round(stat, 4), "p_value": float(f"{p:.4e}")})

    # L3-VGT-SHAPE: Wasserstein distance on VGT distributions
    if new_dims is not None and ref_dims is not None:
        wd = float(wasserstein_distance(new_dims, ref_dims))
        report.add("L3-VGT-SHAPE", True,  # informational
                   f"Wasserstein distance={wd:.4f}",
                   value=round(wd, 4))

    # L3-SILHOUETTE-COMP: compare silhouette scores
    # (requires positions — compute inline)
    new_pos = np.array(result["positions"]).reshape(-1, 3)
    ref_pos = np.array(ref["positions"]).reshape(-1, 3)

    from sklearn.metrics import silhouette_score
    new_mask = new_labels != -1
    ref_mask = ref_labels != -1
    new_unique = len(set(new_labels[new_mask]))
    ref_unique = len(set(ref_labels[ref_mask]))
    if new_mask.sum() > 10 and ref_mask.sum() > 10 and new_unique >= 2 and ref_unique >= 2:
        new_sil = float(silhouette_score(new_pos[new_mask], new_labels[new_mask]))
        ref_sil = float(silhouette_score(ref_pos[ref_mask], ref_labels[ref_mask]))
        report.add("L3-SILHOUETTE-COMP", True,  # informational
                   f"new={new_sil:.4f}, ref={ref_sil:.4f}",
                   value={"new": round(new_sil, 4), "ref": round(ref_sil, 4)})
    else:
        report.add("L3-SILHOUETTE-COMP", True,
                   "skipped (need ≥2 clusters in both datasets)", value=None)

    # L3-CLUSTER-SIZES: Gini coefficient comparison
    from collections import Counter
    new_sizes = sorted(Counter(new_labels[new_labels != -1]).values())
    ref_sizes = sorted(Counter(ref_labels[ref_labels != -1]).values())
    new_gini = _gini(new_sizes) if new_sizes else 0
    ref_gini = _gini(ref_sizes) if ref_sizes else 0
    report.add("L3-CLUSTER-SIZES", True,  # informational
               f"Gini new={new_gini:.4f}, ref={ref_gini:.4f}",
               value={"new": round(new_gini, 4), "ref": round(ref_gini, 4)})

    report.elapsed_seconds = time.monotonic() - t0
    return report


# ── Sidecar Writer ─────────────────────────────────────────────────────


def write_validation_sidecar(
    output_path: Path,
    l1_report: ValidationReport,
    l2_report: ValidationReport | None = None,
    l3_report: ValidationReport | None = None,
) -> Path:
    """Write validation results as a JSON sidecar alongside the output."""
    import datetime

    sidecar = {
        "level1": l1_report.to_dict(),
        "level2": l2_report.to_dict() if l2_report else None,
        "level3": l3_report.to_dict() if l3_report else None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_version": "0.3.0"  # keep in sync with pyproject.toml,
    }

    sidecar_path = output_path.with_name(output_path.stem + "-validation.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    return sidecar_path


# ── Helpers ────────────────────────────────────────────────────────────


def _gini(values: list[int]) -> float:
    """Gini coefficient for a list of positive values."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    arr.sort()
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def _safe_value(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
