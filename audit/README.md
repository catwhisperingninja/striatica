# Euclidicity Audit (Bite 1)

Topological singularity detection for SAE decoder vectors, per von
Rohrscheidt & Rieck (2022) and the Bite 1 plan
(`img/correction/v4_stratgeofocus-replace-umap/zOld-plan-splitbites/bite-01-euclidicity-audit-plan.md`).

This directory is deliberately **isolated from the poetry environment**:
it needs `gudhi` (persistent homology), which must not enter
`poetry.lock`. Never run `poetry add`/`poetry lock` for anything here.

## Contents

- `euclidicity.py` — library module: `euclidicity_scores(X, k_neighbors, radii, *, seed=None)`.
  Import-safe, no CLI side effects, fully deterministic.
- `run_audit.py` — runnable audit script (concordance gate, decision memo).
- `test_audit_euclidicity.py` — audit-of-the-audit test suite (spec 4.1
  sanity checks on synthetic geometric controls).

## Setup (dedicated venv)

From the repository root:

```bash
python3 -m venv .venv-audit
.venv-audit/bin/pip install -r requirements-audit.txt
```

`requirements-audit.txt` pins numpy / scipy / scikit-learn / pytest to the
exact versions in `poetry.lock`, plus `gudhi==3.13.0` (audit-only).

### WSL2 / Linux note

The same commands work on WSL2 and Linux; `gudhi` ships manylinux wheels,
so no CGAL/Boost system packages are needed. If your distro splits out the
venv module, install it first (`sudo apt install python3-venv` on
Debian/Ubuntu). On WSL2, keep the repo on the Linux filesystem
(`~/...`, not `/mnt/c/...`) — KD-tree and persistence workloads are badly
I/O- and mmap-penalized on the 9p-mounted Windows drive. Activate with
`source .venv-audit/bin/activate` (or call `.venv-audit/bin/python`
directly, as below).

## Running the tests

```bash
.venv-audit/bin/python -m pytest audit/ -q
```

(or, with the venv activated: `python -m pytest audit/ -q`; the suite can
also be run from inside the directory as
`cd audit && python -m pytest test_audit_euclidicity.py -q`.)

The suite generates its synthetic geometric controls (sphere, glued
sphere+circle, plane-meets-line) as required by spec section 4.1 — these
are validation geometries, not application mock data.

## Running the audit

Raw GPT-2 Small decoder vectors are not cached in the repo. Export them
once from the **poetry** env, then run the audit from the **audit** venv:

```bash
poetry run python -c "
from pipeline.vectors import load_decoder_vectors
import numpy as np
v = load_decoder_vectors('gpt2-small-res-jb', 'blocks.6.hook_resid_pre')
np.save('data/gpt2-small-6-res-jb-decoder.npy', np.asarray(v))
"

.venv-audit/bin/python audit/run_audit.py \
    --input data/gpt2-small-6-res-jb-decoder.npy \
    --run-id run1
```

Outputs land in `data/audits/euclidicity/<run-id>/` (default run id
`run1`; run directories never contain timestamps):

- `per_feature_scores.npz` — `(n_features, n_scales)` score matrix,
  the scales, and the per-feature minimum across scales.
- `audit_memo.yaml` — the load-bearing decision memo (decision, fraction
  below θ, VGT Pearson r, parameters, library versions).

The VGT concordance gate (spec 2.3 B) is a **hard stop**: if Pearson r
between (1 − Euclidicity) and VGT falls below 0.3, the run exits with the
dedicated nonzero code (`EXIT_VGT_DISCORDANT`) and no memo is written.
