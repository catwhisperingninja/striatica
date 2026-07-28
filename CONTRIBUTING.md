# Contributing to striatica

striatica is a research project under active development. The geometric pipeline is
stable and producing real data across multiple model families; circuit integration is
being rebuilt with causal methods (see [ERRATA.md](ERRATA.md)). Pull requests are welcome.

## Reporting issues

Open an issue with your OS, Python version, and the full error output. For pipeline
problems, include the exact `striat` / `docker run` command and the model or Neuronpedia
ID you were processing.

## Pipeline development (Python)

Poetry drives pipeline development. Docker remains the recommended way to *run* the
pipeline reproducibly (see the [README](README.md)); Poetry is for editing pipeline code.

```bash
poetry install --extras ml    # full: pipeline + PyTorch, SAELens, TransformerLens, dev tools
poetry install                # lightweight: numpy/scipy/sklearn/umap/hdbscan + dev tools
```

Both install the `dev` dependency group (pytest, ruff). Run the CLI with
`poetry run striat --help`.

### ⚠ Never regenerate the lockfile

The committed `poetry.lock` pins the exact library versions that produced the current
data. **UMAP output is not reproducible across versions** — a single patch bump to numpy,
scipy, scikit-learn, umap-learn, or a transitive dependency silently changes every 3D
position.

- Do **not** run `poetry lock`, `poetry update`, or `poetry export`. Locking/exporting is
  known-fragile in this project and will drift the pins.
- Dockerfiles and any CI must install from the committed lockfile — never lock inside a
  build.
- The lockfile is updated only deliberately, on a developer's machine, when a dependency
  change is genuinely required. After any update, **all data must be regenerated and
  visually verified before committing.** See the README's UMAP-reproducibility note.

### Tests

```bash
poetry run pytest tests/ -v -m "not slow"    # fast suite (skips GPU/download tests)
poetry run pytest tests/ -v                  # full suite
poetry run ruff check .                      # lint
```

The `slow` marker (declared in `pyproject.toml`) covers GPU inference and S3 downloads.

## Frontend development (TypeScript / React)

```bash
cd frontend
pnpm install
pnpm dev            # Vite dev server on port 5173
pnpm build          # production build (tsc -b && vite build)
pnpm lint           # eslint
```

Open a generated dataset at `http://localhost:5173/?dataset=<output>.json`.

### End-to-end tests

Playwright specs live in `frontend/tests/` (`app-loads`, `view-switching`, `color-mode`,
`cluster-selection`, `circuit-view`, `debug-console`). The Playwright config auto-starts
`pnpm dev` as its web server:

```bash
cd frontend
pnpm exec playwright test    # installs @playwright/test on first run if not present
```

## Project structure

```
pipeline/          Python package: config, download, vectors, reduce, cluster,
                   circuits, local_dim, prepare, validate, metrics, discovery, cli
scripts/           Launch + tooling scripts (vast_launch, cloud_preprocess, generate_circuits)
tests/             pytest suite
data/              Cached Neuronpedia S3 downloads (JSONL, gitignored)
frontend/
  src/components/  UI panels (TopBar, NavPanel, CircuitPanel, DetailPanel, Canvas3D)
  src/three/       React Three Fiber components (PointCloudMesh, FlyToCamera, CircuitNodes)
  src/views/       View compositions (PointCloudView, CircuitGraphView)
  src/stores/      Zustand store (useAppStore)
  src/shaders/     Custom GLSL vertex/fragment shaders
  src/config/      Centralized rendering parameters (rendering.ts)
  tests/           Playwright e2e specs
  public/data/     Generated JSON (gitignored — run the pipeline to populate)
```

## Contribution rules

- **All rendering values live in `frontend/src/config/rendering.ts`.** New components
  import from it — no magic numbers in rendering code.
- **Never commit semantic labels.** Feature explanations for non-public-tier models are
  redacted by the pipeline; keep them out of git, Docker images, logs, and screenshots
  (see the README's dual-use section).
- **Do not regenerate production data** (`frontend/public/data/*.json`, circuit JSONs)
  without maintainer approval — data is visually verified before it lands.
- Keep the debug console (backtick) wired to any new view or store field.
- Keep changes scoped to the files a change actually touches, and match existing patterns.
