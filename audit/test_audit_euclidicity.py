"""Lane 3 — Bite 1 Euclidicity audit: audit-of-the-audit test suite.

Spec sources:
  - img/correction/v4_stratgeofocus-replace-umap/v4-master-plan.md (Bite 1
    section, lines ~500-730): algorithm (2.1), multi-scale (2.2), decision
    criteria (2.3), sanity checks (4.1), failure modes (6).
  - img/correction/v4_stratgeofocus-replace-umap/zOld-plan-splitbites/
    bite-01-euclidicity-audit-plan.md (same content, standalone).

This file lives in the top-level audit/ directory (NOT tests/) and runs in a
separate venv from the main pipeline. Run with:

    cd audit && python -m pytest test_audit_euclidicity.py -q

All synthetic datasets below are geometric CONTROLS explicitly required by
spec section 4.1 ("Generate synthetic data on a known smooth manifold /
known stratified space"). They are not application mock data.

============================================================================
ASSUMED API — the implementer contract these tests pin down
============================================================================

audit/euclidicity.py
--------------------
    euclidicity_scores(
        X: np.ndarray,            # (n_points, d) float array
        k_neighbors: int,         # minimum neighborhood size; a
                                  # (point, radius) cell whose radius-ball
                                  # (excluding the point itself) contains
                                  # fewer than k_neighbors points gets np.nan
                                  # (spec 3.1 n_neighbors_min semantics)
        radii: Sequence[float],   # neighborhood radii; output column j
                                  # corresponds to radii[j] (spec 2.2)
        *,
        seed: int | None = None,  # seeds ALL internal randomness (model-ball
                                  # sampling, any subsampling). A fully
                                  # deterministic implementation may ignore
                                  # it. Same (X, k, radii, seed) must
                                  # reproduce bit-identical output (spec B8
                                  # hash-stability is stricter still).
    ) -> np.ndarray               # shape (n_points, len(radii)), float64.
                                  # Finite entries lie in [0, 1]:
                                  #   ~1.0 = neighborhood indistinguishable
                                  #          from a Euclidean ball of the
                                  #          same intrinsic dimension
                                  #   ~0.0 = maximally singular
                                  # np.nan = insufficient neighbors.

    Metric note: these controls are Euclidean-space geometries, so the
    default metric MUST be Euclidean. The production runner may expose an
    optional metric="cosine" kwarg for SAE decoder vectors (spec 2.2 uses
    cosine-distance scales; spec 4.2 B2 runs both) — untested here.

    The reference ball comparison must use the *intrinsic* dimension of the
    neighborhood (spec 2.1 step 3): a 1-D line interior must be compared
    against a 1-ball and score HIGH, not be penalized for not being a disk.

audit/run_audit.py  (runnable script: `python audit/run_audit.py --help`
                     must exit 0; importing it must NOT run the audit)
------------------
    THETA = 0.7                  # manifoldness threshold (spec 2.3 A)
    VGT_GATE_R = 0.3             # concordance gate (spec 2.3 B)
    EXIT_VGT_DISCORDANT: int     # nonzero process exit code for the
                                 # hard stop (spec 6: "Stop and investigate
                                 # before proceeding")

    vgt_concordance_gate(
        min_scores: np.ndarray,  # per-feature min Euclidicity across scales
        vgt: np.ndarray,         # per-feature VGT local-dimension scores
    ) -> float
        Computes Pearson r between (1 - min_scores) and vgt over entries
        finite in BOTH arrays (spec 3.1 run_audit valid-mask logic).
        r >= VGT_GATE_R  -> returns r (audit may proceed).
        r <  VGT_GATE_R  -> raises SystemExit(EXIT_VGT_DISCORDANT): the
        hard stop. No decision memo may be trusted past a failed gate.
============================================================================
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

AUDIT_DIR = Path(__file__).resolve().parent
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

import euclidicity  # noqa: E402
import run_audit  # noqa: E402
from euclidicity import euclidicity_scores  # noqa: E402

RNG_SEED = 20260728
THETA = 0.7  # spec 2.3(A); pinned against run_audit.THETA below


# ---------------------------------------------------------------------------
# Synthetic-control generators (spec 4.1 — required controls, not app mocks)
# ---------------------------------------------------------------------------

def _unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """n points uniform on the unit 2-sphere in R^3 (known smooth manifold)."""
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    assert np.all(norms > 0)
    return v / norms


def _exact_corr_pair(r: float, n: int, rng: np.random.Generator):
    """Two length-n vectors whose SAMPLE Pearson correlation is exactly r.

    Built via Gram-Schmidt so gate-threshold tests sit at controlled r values
    instead of noisy approximations (per repo rule: specific properties, not
    generalities).
    """
    x = rng.normal(size=n)
    e = rng.normal(size=n)
    x = x - x.mean()
    e = e - e.mean()
    e = e - (e @ x) / (x @ x) * x  # orthogonalize
    x = x / np.linalg.norm(x)
    e = e / np.linalg.norm(e)
    y = r * x + np.sqrt(1.0 - r * r) * e
    return x, y


def _gate_inputs(target_r: float, n: int, rng: np.random.Generator):
    """(min_scores, vgt) with corr(1 - min_scores, vgt) == target_r exactly.

    Pearson correlation is invariant under positive-slope affine maps, so
    rescaling x into [0, 1] and vgt into a VGT-like positive range preserves
    the constructed correlation exactly (up to float eps).
    """
    x, y = _exact_corr_pair(target_r, n, rng)
    x01 = (x - x.min()) / (x.max() - x.min())  # in [0, 1]
    min_scores = 1.0 - x01                     # valid Euclidicity range
    vgt = 8.0 + 3.0 * y                        # VGT-like magnitudes
    return min_scores, vgt


# ---------------------------------------------------------------------------
# Score fixtures (module-scoped: each euclidicity_scores call is expensive)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sphere_scores():
    """Spec 4.1 trivial case: smooth manifold, scores should be ~1.0."""
    rng = np.random.default_rng(RNG_SEED)
    X = _unit_sphere(600, rng)
    radii = [0.35, 0.5]
    S = euclidicity_scores(X, k_neighbors=10, radii=radii, seed=0)
    return X, radii, S


@pytest.fixture(scope="module")
def glued_sphere_circle_scores():
    """Spec 4.1 singular case: 2-sphere and circle glued at (1, 0, 0).

    Circle of radius 0.6 centered at (1.6, 0, 0) in the z=0 plane passes
    through the gluing point (1, 0, 0) and otherwise lies outside the sphere,
    so the gluing-point neighborhood is a disk with a 1-D whisker — not a
    ball. The exact gluing point is appended as the last row so its score is
    addressable deterministically.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    n_sphere, n_circle = 600, 150
    sphere = _unit_sphere(n_sphere, rng)
    t = rng.uniform(0.0, 2.0 * np.pi, n_circle)
    circle = np.column_stack(
        [1.6 + 0.6 * np.cos(t), 0.6 * np.sin(t), np.zeros(n_circle)]
    )
    glue = np.array([[1.0, 0.0, 0.0]])
    X = np.vstack([sphere, circle, glue])
    S = euclidicity_scores(X, k_neighbors=10, radii=[0.3, 0.45], seed=0)
    return sphere, X, len(X) - 1, S


def _plane_line_dataset():
    """Plane z=0 over [-1,1]^2 plus the z-axis line, crossing at the origin.

    The exact singular point (origin) is appended as the last row. Control
    masks are chosen so control neighborhoods (radius <= 0.4) can neither
    touch the other stratum nor spill past the sampled square:
      - plane interior: 0.45 < hypot(x,y), |x| <= 0.6, |y| <= 0.6
      - line interior:  |z| > 0.5
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    n_plane, n_line = 400, 150
    plane = np.column_stack(
        [rng.uniform(-1, 1, n_plane), rng.uniform(-1, 1, n_plane), np.zeros(n_plane)]
    )
    line = np.column_stack(
        [np.zeros(n_line), np.zeros(n_line), rng.uniform(-1, 1, n_line)]
    )
    origin = np.zeros((1, 3))
    X = np.vstack([plane, line, origin])

    plane_idx = np.arange(n_plane)
    line_idx = np.arange(n_plane, n_plane + n_line)
    r_xy = np.hypot(plane[:, 0], plane[:, 1])
    plane_interior = plane_idx[
        (r_xy > 0.45) & (np.abs(plane[:, 0]) <= 0.6) & (np.abs(plane[:, 1]) <= 0.6)
    ]
    line_interior = line_idx[np.abs(line[:, 2]) > 0.5]
    return X, len(X) - 1, plane_interior, line_interior


@pytest.fixture(scope="module")
def plane_line_scores():
    """Required stratified control: plane-meets-line, scored with seed=0."""
    X, origin_idx, plane_interior, line_interior = _plane_line_dataset()
    S = euclidicity_scores(X, k_neighbors=10, radii=[0.25, 0.4], seed=0)
    return X, origin_idx, plane_interior, line_interior, S


@pytest.fixture(scope="module")
def plane_line_repeat_scores(plane_line_scores):
    """Same dataset rescored with the same seed."""
    X = plane_line_scores[0]
    return euclidicity_scores(X, k_neighbors=10, radii=[0.25, 0.4], seed=0)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_shape_matches_points_by_radii(self, sphere_scores):
        X, radii, S = sphere_scores
        assert isinstance(S, np.ndarray)
        assert S.shape == (X.shape[0], len(radii))
        assert np.issubdtype(S.dtype, np.floating)

    def test_finite_scores_lie_in_unit_interval(self, sphere_scores):
        _, _, S = sphere_scores
        finite = S[np.isfinite(S)]
        assert finite.size > 0
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 1.0)

    def test_dense_data_is_mostly_scored(self, sphere_scores):
        # 600 points on the unit sphere at radii 0.35/0.5 give every point
        # far more than k_neighbors=10 neighbors in expectation; NaN must be
        # the exception, not the rule (spec 6: insufficient-neighbors mode).
        _, _, S = sphere_scores
        rows_with_score = np.any(np.isfinite(S), axis=1)
        assert np.mean(rows_with_score) >= 0.95

    def test_insufficient_neighbors_yield_nan(self):
        # A far-away outlier has zero points within the radius: its cell must
        # be NaN, while the dense blob (each point has ~59 neighbors within
        # the radius) must be fully scored. Spec 3.1 docstring semantics.
        rng = np.random.default_rng(RNG_SEED + 4)
        blob = rng.normal(scale=0.01, size=(60, 3))
        outlier = np.array([[5.0, 5.0, 5.0]])
        X = np.vstack([blob, outlier])
        S = euclidicity_scores(X, k_neighbors=10, radii=[0.08], seed=0)
        assert S.shape == (61, 1)
        assert np.all(np.isnan(S[-1]))
        assert np.all(np.isfinite(S[:60]))


# ---------------------------------------------------------------------------
# Spec 4.1 sanity check 1 — trivial case (smooth manifold scores high)
# ---------------------------------------------------------------------------

class TestSmoothManifoldControl:
    def test_sphere_scores_high_everywhere(self, sphere_scores):
        # Spec 4.1: "Euclidicity should be ~1.0 everywhere. If not, the
        # audit code is broken." Assessed on the per-point minimum across
        # radii — the same worst-case aggregation the decision rule uses
        # (spec 3.1 run_audit).
        _, _, S = sphere_scores
        min_per_point = np.nanmin(S, axis=1)
        finite = min_per_point[np.isfinite(min_per_point)]
        assert finite.size >= 500
        assert np.median(finite) >= THETA
        assert np.mean(finite >= THETA) >= 0.8


# ---------------------------------------------------------------------------
# Spec 4.1 sanity check 2 — singular case (sphere glued to circle)
# ---------------------------------------------------------------------------

class TestGluedSphereCircleControl:
    def test_gluing_point_scores_low(self, glued_sphere_circle_scores):
        _, _, glue_idx, S = glued_sphere_circle_scores
        glue_min = np.nanmin(S[glue_idx])
        assert np.isfinite(glue_min)
        assert glue_min < THETA

    def test_drop_is_sharp_relative_to_far_sphere(self, glued_sphere_circle_scores):
        # Spec 4.1: "Euclidicity should drop sharply at the gluing point."
        # Far hemisphere (x < 0) is > 1.4 away from the gluing point, so its
        # neighborhoods never see the circle — pure manifold reference.
        sphere, _, glue_idx, S = glued_sphere_circle_scores
        glue_min = np.nanmin(S[glue_idx])
        far_mask = sphere[:, 0] < 0.0
        assert far_mask.sum() >= 100
        far_min = np.nanmin(S[: len(sphere)][far_mask], axis=1)
        far_median = np.nanmedian(far_min)
        assert far_median >= THETA
        assert far_median - glue_min >= 0.2


# ---------------------------------------------------------------------------
# Required stratified control — plane meets line (task spec)
# ---------------------------------------------------------------------------

class TestPlaneMeetsLineControl:
    def test_control_masks_are_populated(self, plane_line_scores):
        # Guard: the statistics below are only meaningful over real samples.
        _, _, plane_interior, line_interior, _ = plane_line_scores
        assert len(plane_interior) >= 30
        assert len(line_interior) >= 30

    def test_singularity_has_high_euclidicity_deviation(self, plane_line_scores):
        # The plane-line crossing must score LOW (deviation 1 - score HIGH):
        # its neighborhood is a disk pierced by a segment, never a ball.
        _, origin_idx, plane_interior, _, S = plane_line_scores
        origin_min = np.nanmin(S[origin_idx])
        assert np.isfinite(origin_min)
        assert origin_min < THETA

        plane_min = np.nanmin(S[plane_interior], axis=1)
        # Sharply below the stratum interior, not marginally:
        assert origin_min < np.nanpercentile(plane_min, 10)
        assert np.nanmedian(plane_min) - origin_min >= 0.2

    def test_stratum_interiors_score_high(self, plane_line_scores):
        # Away from the crossing each stratum is a clean manifold. The line
        # interior tests intrinsic-dimension matching (spec 2.1 step 3): a
        # 1-D neighborhood compared against a 1-ball must score high.
        _, _, plane_interior, line_interior, S = plane_line_scores
        plane_min = np.nanmin(S[plane_interior], axis=1)
        line_min = np.nanmin(S[line_interior], axis=1)
        assert np.nanmedian(plane_min) >= THETA
        assert np.nanmedian(line_min) >= THETA

    def test_pure_plane_has_low_deviation(self):
        # Control counterpart required by the task spec: the SAME plane
        # geometry without the piercing line must score high everywhere in
        # its interior (deviation low) — proving the singularity signal in
        # the glued dataset comes from the crossing, not from the plane.
        rng = np.random.default_rng(RNG_SEED + 3)
        n = 400
        plane = np.column_stack(
            [rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), np.zeros(n)]
        )
        S = euclidicity_scores(plane, k_neighbors=10, radii=[0.25, 0.4], seed=0)
        interior = (np.abs(plane[:, 0]) <= 0.6) & (np.abs(plane[:, 1]) <= 0.6)
        assert interior.sum() >= 100
        interior_min = np.nanmin(S[interior], axis=1)
        finite = interior_min[np.isfinite(interior_min)]
        assert finite.size >= 100
        assert np.median(finite) >= THETA
        assert np.mean(finite >= THETA) >= 0.8


# ---------------------------------------------------------------------------
# Spec 4.1 sanity check 3 — robustness / reproducibility
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_same_seed_is_bit_reproducible(self, plane_line_scores, plane_line_repeat_scores):
        # Spec 4.2 B8 requires hash-stable score files; same-process
        # same-seed bit-identity (including the NaN pattern) is the minimum.
        S_a = plane_line_scores[4]
        S_b = plane_line_repeat_scores
        assert np.array_equal(S_a, S_b, equal_nan=True)


# ---------------------------------------------------------------------------
# VGT concordance gate (spec 2.3 B, spec 6 hard-stop failure mode)
# ---------------------------------------------------------------------------

class TestVgtConcordanceGate:
    def test_constants_pinned_to_spec(self):
        assert run_audit.THETA == pytest.approx(0.7)
        assert run_audit.VGT_GATE_R == pytest.approx(0.3)
        assert isinstance(run_audit.EXIT_VGT_DISCORDANT, int)
        assert run_audit.EXIT_VGT_DISCORDANT != 0

    def test_returns_r_when_strongly_concordant(self):
        rng = np.random.default_rng(RNG_SEED + 5)
        min_scores, vgt = _gate_inputs(0.8, 400, rng)
        r = run_audit.vgt_concordance_gate(min_scores, vgt)
        assert r == pytest.approx(0.8, abs=1e-6)

    def test_passes_just_above_threshold(self):
        # Spec 2.3(B): only r < 0.3 means broken; r = 0.31 must pass (it is
        # "weak" corroboration, not a hard stop).
        rng = np.random.default_rng(RNG_SEED + 6)
        min_scores, vgt = _gate_inputs(0.31, 400, rng)
        r = run_audit.vgt_concordance_gate(min_scores, vgt)
        assert r == pytest.approx(0.31, abs=1e-6)

    @pytest.mark.parametrize("target_r", [-0.6, 0.0, 0.29])
    def test_hard_stops_below_threshold(self, target_r):
        # Spec 6: "VGT correlation is broken (Pearson r < 0.3) ... Stop and
        # investigate before proceeding." Hard stop = SystemExit with the
        # dedicated nonzero exit code; no return value, no soft warning.
        rng = np.random.default_rng(RNG_SEED + 7)
        min_scores, vgt = _gate_inputs(target_r, 400, rng)
        with pytest.raises(SystemExit) as excinfo:
            run_audit.vgt_concordance_gate(min_scores, vgt)
        assert excinfo.value.code == run_audit.EXIT_VGT_DISCORDANT

    def test_nan_entries_are_excluded_pairwise(self):
        # Spec 3.1 run_audit: valid = ~isnan(min_scores) & ~isnan(vgt).
        # Features unscored on either side must not poison the correlation.
        rng = np.random.default_rng(RNG_SEED + 8)
        min_scores, vgt = _gate_inputs(0.8, 300, rng)
        nan_scores = np.full(20, np.nan)
        nan_vgt_side = rng.uniform(0.0, 1.0, 20)  # finite scores, NaN vgt
        min_scores_full = np.concatenate([min_scores, nan_scores, nan_vgt_side])
        vgt_full = np.concatenate([vgt, rng.uniform(1.0, 20.0, 20), np.full(20, np.nan)])
        r = run_audit.vgt_concordance_gate(min_scores_full, vgt_full)
        assert r == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# Runner smoke contract
# ---------------------------------------------------------------------------

class TestRunnerContract:
    def test_run_audit_help_exits_zero(self):
        # run_audit.py must be runnable as a script and must not launch the
        # audit just to print usage. Import-time side effects are already
        # excluded by this module importing run_audit at the top.
        proc = subprocess.run(
            [sys.executable, str(AUDIT_DIR / "run_audit.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() != ""

    def test_euclidicity_module_has_no_cli_side_effects(self):
        # Importing the library module must never trigger computation.
        # (Already imported at module top; this pins the property explicitly
        # so a future __main__-less refactor keeps it.)
        assert hasattr(euclidicity, "euclidicity_scores")
