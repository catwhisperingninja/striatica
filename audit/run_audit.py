#!/usr/bin/env python3
"""Bite 1 Euclidicity audit runner (spec 3.1/3.2).

Computes per-feature Euclidicity scores for the GPT-2 Small 6-res-jb SAE
decoder vectors, cross-validates against the existing VGT local-dimension
scores (concordance gate, spec 2.3 B / spec 6 hard stop), applies the
decision rule (spec 2.3 A), and writes the score arrays plus a decision
memo to data/audits/euclidicity/<run-id>/.

Runs in the audit venv (.venv-audit — see audit/README.md), NOT the poetry
env: it imports only numpy/scipy/gudhi plus the torch-free pipeline.banner
terminal helpers.  It must never import pipeline modules that pull
torch/SAELens (pipeline.vectors, pipeline.cli, ...).

Raw decoder vectors are NOT cached in this repository (data/ holds only
Neuronpedia explanation JSONL, which this audit never reads).  Export them
once from the main poetry env and pass the .npy via --input:

    poetry run python -c "
    from pipeline.vectors import load_decoder_vectors
    import numpy as np
    v = load_decoder_vectors('gpt2-small-res-jb', 'blocks.6.hook_resid_pre')
    np.save('data/gpt2-small-6-res-jb-decoder.npy', np.asarray(v))
    "
    .venv-audit/bin/python audit/run_audit.py \
        --input data/gpt2-small-6-res-jb-decoder.npy --run-id run1

Importing this module performs no computation; `--help` never launches the
audit.  Output paths contain no timestamps — the run directory is exactly
<output-root>/<run-id> (default run id: "run1").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
for _p in (str(AUDIT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# pipeline/__init__.py is a bare docstring; pipeline.banner imports only
# os/sys/time — no torch anywhere on this import path.
from pipeline.banner import (  # noqa: E402
    detail,
    error,
    info,
    step_header,
    success,
    warn,
)

from euclidicity import euclidicity_scores  # noqa: E402

# ── Spec constants (Bite 1 §2.3, §3.1) ──────────────────────────────────

THETA = 0.7                     # manifoldness threshold (spec 2.3 A)
VGT_GATE_R = 0.3                # concordance gate (spec 2.3 B)
EXIT_VGT_DISCORDANT = 3         # hard-stop exit code (spec 6)

DECISION_THRESHOLD_HIGH = 0.15  # fraction below theta -> stratified
DECISION_THRESHOLD_LOW = 0.05   # fraction below theta -> manifold-like

DEFAULT_SCALES = [0.05, 0.1, 0.2, 0.4]  # cosine-distance scales (spec 2.2)
DEFAULT_K_NEIGHBORS = 10                # n_neighbors_min (spec 3.1)

DEFAULT_DATASET_JSON = (
    REPO_ROOT / "frontend" / "public" / "data" / "gpt2-small-6-res-jb.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "audits" / "euclidicity"


# ── Concordance gate (spec 2.3 B; spec 6 hard stop) ─────────────────────

def vgt_concordance_gate(min_scores: np.ndarray, vgt: np.ndarray) -> float:
    """Pearson r between (1 - min Euclidicity) and VGT local dimension.

    Entries non-finite on EITHER side are excluded pairwise (spec 3.1
    valid-mask logic).  r >= VGT_GATE_R returns r; r < VGT_GATE_R is the
    spec 6 hard stop: "Stop and investigate before proceeding" — raises
    SystemExit(EXIT_VGT_DISCORDANT).  No decision memo may be trusted past
    a failed gate.
    """
    min_scores = np.asarray(min_scores, dtype=np.float64)
    vgt = np.asarray(vgt, dtype=np.float64)
    valid = np.isfinite(min_scores) & np.isfinite(vgt)
    if int(valid.sum()) < 2:
        error("VGT concordance gate: fewer than 2 jointly-finite features")
        raise SystemExit(EXIT_VGT_DISCORDANT)
    r = float(np.corrcoef(1.0 - min_scores[valid], vgt[valid])[0, 1])
    if r < VGT_GATE_R:
        error(
            f"VGT concordance gate FAILED: Pearson r = {r:.4f} < {VGT_GATE_R}"
        )
        detail("(1 - Euclidicity) and VGT both measure non-manifoldness;")
        detail("r below the gate means one metric is broken or they measure")
        detail("different things (Bite 1 spec §6). Hard stop — investigate")
        detail("before trusting ANY downstream decision.")
        raise SystemExit(EXIT_VGT_DISCORDANT)
    return r


# ── IO helpers ──────────────────────────────────────────────────────────

def load_vectors(input_path: Path | None) -> np.ndarray:
    if input_path is None:
        error("No decoder vectors available: raw SAE decoder vectors are not")
        detail("cached in this repository. Export them from the poetry env")
        detail("(see the module docstring) and re-run with --input <file>.npy")
        raise SystemExit(2)
    vectors = np.load(input_path)
    if vectors.ndim != 2:
        error(f"--input must be a 2-D (n_features, d) array; got {vectors.shape}")
        raise SystemExit(2)
    return np.asarray(vectors, dtype=np.float64)


def load_vgt(dataset_json: Path) -> np.ndarray:
    """Existing per-feature VGT local-dimension scores from the main
    dataset JSON (geometry-only file — produced by process_gpt2_small.py,
    clean per the 2026-03-21 triage; contains no semantic labels in the
    fields read here)."""
    with open(dataset_json) as fh:
        data = json.load(fh)
    vgt = np.asarray(data["localDimensions"], dtype=np.float64)
    return vgt


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return json.dumps(str(value))


def write_memo(path: Path, memo: dict) -> None:
    """Write the decision memo as flat YAML (no pyyaml dependency — the
    audit venv is pinned to requirements-audit.txt)."""
    lines = []
    for key, value in memo.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            lines.extend(f"  - {_yaml_scalar(v)}" for v in value)
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n")


# ── Main ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_audit.py",
        description=(
            "Euclidicity audit (Bite 1): per-feature topological "
            "singularity scores for SAE decoder vectors, VGT concordance "
            "gate, and the stratified/manifold-like decision memo."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to a .npy (n_features, d) decoder-vector array. Required "
            "in practice: raw vectors are not cached in-repo (see module "
            "docstring for the export one-liner)."
        ),
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
        help="Main dataset JSON holding existing VGT localDimensions "
        "(default: frontend/public/data/gpt2-small-6-res-jb.json)",
    )
    parser.add_argument(
        "--run-id",
        default="run1",
        help="Run directory name under the output root (default: run1; "
        "never a timestamp)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory (default: data/audits/euclidicity)",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help=f"Neighborhood scales (default: {DEFAULT_SCALES}, "
        "cosine-distance units under --metric cosine)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help=f"Minimum neighborhood size (default: {DEFAULT_K_NEIGHBORS})",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric (default: cosine — spec 2.2; euclidean is "
        "the B2 cross-check)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (the current scorer is fully deterministic and "
        "ignores it; accepted for the spec 4.1 robustness protocol)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    step_header("dimension", "Euclidicity audit — Bite 1")
    info("metric", args.metric)
    info("radii", ", ".join(f"{r:g}" for r in args.radii))
    info("k_neighbors", str(args.k_neighbors))
    info("run id", args.run_id)

    vectors = load_vectors(args.input)
    info("vectors", f"{vectors.shape[0]} features x {vectors.shape[1]} dims")

    vgt = load_vgt(args.dataset_json)
    info("VGT source", str(args.dataset_json))
    if len(vgt) != len(vectors):
        error(
            f"feature-count mismatch: {len(vectors)} vectors vs "
            f"{len(vgt)} VGT scores — wrong --input for this dataset JSON"
        )
        raise SystemExit(2)

    step_header("vgt", "Scoring Euclidicity per feature x scale")
    detail("Vietoris-Rips persistence via gudhi; this can take a while.")
    scores = euclidicity_scores(
        vectors,
        k_neighbors=args.k_neighbors,
        radii=args.radii,
        seed=args.seed,
        metric=args.metric,
    )

    min_per_feature = np.full(len(scores), np.nan)
    has_any = np.any(np.isfinite(scores), axis=1)
    min_per_feature[has_any] = np.nanmin(scores[has_any], axis=1)

    n_features = len(min_per_feature)
    n_unscored = int(np.sum(~has_any))
    finite_min = min_per_feature[np.isfinite(min_per_feature)]
    fraction_below_theta = float(np.mean(finite_min < THETA))
    info("scored", f"{n_features - n_unscored}/{n_features} features")
    if n_unscored > 0.1 * n_features:
        warn(
            f"{n_unscored} features lack sufficient neighbors (>10% — "
            "chosen scales may be too small for the data density, spec §6)"
        )
    info("frac < theta", f"{fraction_below_theta:.4f}  (theta = {THETA})")

    step_header("cluster", "VGT concordance gate")
    pearson_r = vgt_concordance_gate(min_per_feature, vgt)
    vgt_status = (
        "strong" if pearson_r > 0.5 else "weak" if pearson_r > VGT_GATE_R else
        "broken_or_diverging"
    )
    info("Pearson r", f"{pearson_r:.4f}  ({vgt_status} corroboration)")

    if fraction_below_theta >= DECISION_THRESHOLD_HIGH:
        decision = "stratified"
        action = "commit to TDA architecture (Phase 1+)"
    elif fraction_below_theta < DECISION_THRESHOLD_LOW:
        decision = "manifold_like"
        action = "fall back to section 6 manifold-method bake-off"
    else:
        decision = "ambiguous"
        action = "run B5 stratified-by-frac_nonzero diagnostic before committing"

    step_header("assemble", "Writing audit outputs")
    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "per_feature_scores.npz"
    np.savez_compressed(
        scores_path,
        scores=scores,
        radii=np.asarray(args.radii, dtype=np.float64),
        min_per_feature=min_per_feature,
    )
    detail(f"scores -> {scores_path}")

    import gudhi as _gudhi
    import scipy as _scipy

    memo = {
        "decision": decision,
        "action": action,
        "fraction_below_theta": fraction_below_theta,
        "vgt_correlation_pearson_r": pearson_r,
        "vgt_correlation_status": vgt_status,
        "scales": [float(r) for r in args.radii],
        "theta": THETA,
        "metric": args.metric,
        "k_neighbors": int(args.k_neighbors),
        "n_features": int(n_features),
        "n_features_insufficient_neighbors": n_unscored,
        "per_feature_euclidicity_path": str(scores_path),
        "input_vectors": str(args.input),
        "dataset_json": str(args.dataset_json),
        "numpy_version": np.__version__,
        "scipy_version": _scipy.__version__,
        "gudhi_version": _gudhi.__version__,
    }
    memo_path = out_dir / "audit_memo.yaml"
    write_memo(memo_path, memo)
    detail(f"memo   -> {memo_path}")

    success(f"Euclidicity audit complete: decision = {decision}")
    detail(action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
