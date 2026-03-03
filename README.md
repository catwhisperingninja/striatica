# striatica

A geometric atlas for machine intelligence.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18848240.svg)](https://doi.org/10.5281/zenodo.18848240)

> **Paper:** _Striatica: A Geometric Atlas for Machine Intelligence_ —
> [Zenodo](https://doi.org/10.5281/zenodo.18848240)

---

# Quick Start

You need Python 3.12+, [Poetry 2.x](https://python-poetry.org/), and Node.js
18+.

```bash
git clone https://github.com/catwhisperingninja/striatica.git
cd striatica

# Install with ML dependencies (~2GB first run: PyTorch, SAELens, TransformerLens)
poetry install --extras ml

# Run the GPT-2 Small demo — generates data, circuits, launches the frontend
poetry run striat demo
```

`striat demo` walks you through everything: builds the dataset if it doesn't
exist, optionally generates circuits, finds an open port, and starts the
frontend.

Open the localhost URL, and you're exploring.

### What `striat demo` does

1. Downloads GPT-2 Small feature metadata from Neuronpedia S3 (public, no API
   key needed)
2. Loads SAE decoder weight vectors from HuggingFace via SAELens
3. PCA (768d → 50d) → UMAP (50d → 3d)
4. HDBSCAN clustering
5. Local dimension estimation (Participation Ratio + VGT growth curves)
6. Assembles JSON for the frontend (~19MB for 24,576 features)
7. Asks if you want to generate the 10 default circuits (5 co-activation + 5
   similarity)
8. Starts the Vite dev server on the first available port from 5173

Compute time is 5–10 minutes on a recent MacBook Pro or equivalent. The data
pipeline runs on CPU; no GPU required for GPT-2 Small.

### Data not included

The generated dataset is too large to commit (~19MB JSON + circuit files). Every
clone generates its own data via `striat demo`. Cached intermediates in
`striatica/data/` speed up subsequent runs.

---

# Bring Your Own Model

striatica works with any SAE dictionary available through
[SAELens](https://github.com/jbloom/SAELens). The demo uses GPT-2 Small Layer 6,
but you can point it at anything SAELens supports.

```bash
poetry run striat model \
  --model gemma-2b \
  --layer 12-res-jb \
  --sae-release gemma-2b-res-jb \
  --sae-hook blocks.12.hook_resid_pre \
  --num-batches 48 \
  --features-per-batch 1024
```

This runs the same pipeline (download → reduce → cluster → local dim → JSON) but
with your model's parameters. Output lands in
`frontend/public/data/<model>-<layer>.json`.

### Quick test with a small model

To verify the BYO pipeline works without downloading ~2GB of GPT-2 weights, run
the included test script. It uses Pythia-70M (~150MB), the smallest model
SAELens supports:

```bash
bash scripts/test_byo_model.sh
```

This calls `striat model` with pre-filled Pythia-70M flags (4 batches × 1024
features = 4,096 features). Takes about a minute on a recent laptop.

### Hardware guidance

| Model              | Features | RAM   | Time (approx) | GPU                  |
| ------------------ | -------- | ----- | ------------- | -------------------- |
| GPT-2 Small (demo) | 24,576   | 24GB  | 5–10 min      | not needed           |
| Gemma 2B           | ~49K     | 32GB  | 15–30 min     | recommended          |
| Llama 3 8B         | ~130K    | 64GB+ | 1–2 hr        | strongly recommended |

VGT growth curve computation is the bottleneck — it's O(n²) on the feature
count. Large dictionaries benefit significantly from GPU-accelerated distance
computation.

### After generating

Once your model's data is in `frontend/public/data/`, launch the frontend and
pass the dataset filename as a query parameter:

```bash
cd frontend && pnpm install && pnpm dev
# Then open: http://localhost:5173/?dataset=gemma-2b-12-res-jb.json
```

Without the `?dataset=` parameter, the frontend loads the GPT-2 Small demo
dataset by default. The `striat model` command prints the exact URL to open when
it finishes.

### Required flags

| Flag            | Description                           | Example                   |
| --------------- | ------------------------------------- | ------------------------- |
| `--model`       | Model ID matching SAELens/Neuronpedia | `gpt2-small`, `gemma-2b`  |
| `--layer`       | Layer identifier                      | `6-res-jb`, `12-res-jb`   |
| `--sae-release` | SAELens release name                  | `gpt2-small-res-jb`       |
| `--sae-hook`    | TransformerLens hook point            | `blocks.6.hook_resid_pre` |

| Optional               | Description                | Default |
| ---------------------- | -------------------------- | ------- |
| `--num-batches`        | Neuronpedia S3 batch count | 24      |
| `--features-per-batch` | Features per S3 batch      | 1024    |

To find the right values for your model, check the
[SAELens model list](https://github.com/jbloom/SAELens) and
[Neuronpedia](https://neuronpedia.org).

---

# Circuits

Generate circuit data to see how features connect during specific computations.

```bash
# All 10 default circuits (5 co-activation + 5 similarity)
poetry run striat circuits --batch-defaults

# Co-activation: which features fire together on a prompt
poetry run striat circuits \
  --type coactivation \
  --prompt "The capital of France is" \
  --name my-capital-circuit

# Similarity: BFS through cosine-similar features from a seed
poetry run striat circuits \
  --type similarity \
  --seed-feature 23123 \
  --name my-sim-circuit
```

Co-activation circuits require the ML extras (runs the model on CPU). Similarity
circuits only need the base install — they use cosine similarity data already
present in the Neuronpedia download.

| Flag               | Description                                           | Default                            |
| ------------------ | ----------------------------------------------------- | ---------------------------------- |
| `--batch-defaults` | Generate all 10 default circuits                      | —                                  |
| `--type`           | `coactivation` or `similarity`                        | required unless `--batch-defaults` |
| `--prompt`         | Input text (co-activation only)                       | required for coactivation          |
| `--seed-feature`   | Feature index for BFS root (similarity only)          | required for similarity            |
| `--name`           | Circuit ID, used as filename                          | auto-generated                     |
| `--top-k`          | Features to include (coact) / neighbors per hop (sim) | 30                                 |
| `--min-weight`     | Minimum edge weight to keep                           | 0.1                                |
| `--depth`          | BFS depth for similarity                              | 2                                  |

Output: `frontend/public/data/circuits/<name>.json` + updated `manifest.json`

---

# Views

**Point Cloud** — All features positioned by decoder weight similarity (UMAP
projection to 3D). Color by cluster membership or local intrinsic dimension.
Click any point to inspect its metadata, activation stats, top tokens, VGT
growth curve, and Neuronpedia link.

**Circuits** — Select a circuit from the panel to visualize which features
participate. Nodes colored by role: source (green), processing (cyan), output
(amber). Edge threshold slider controls visibility by connection strength. ⌘P /
Ctrl+P toggles between views.

**Cross-view sync** — Selection, camera position, and cluster highlighting all
persist across view switches. Circuit members are highlighted in the point cloud
(boosted size and brightness). Selecting a circuit member and switching to
Circuits view auto-loads the relevant circuit.

---

# Controls

| Input                | Action                                                      |
| -------------------- | ----------------------------------------------------------- |
| Click                | Select a feature point                                      |
| Search box           | Find features by index or description, click to fly to them |
| Double-click cluster | Fly camera to cluster centroid                              |
| Drag                 | Orbit                                                       |
| Scroll               | Zoom (speed adapts to distance)                             |
| Right-drag           | Pan                                                         |
| Shift-click cluster  | Multi-select clusters (up to 10)                            |
| ⌘P / Ctrl+P          | Toggle Point Cloud ↔ Circuits view                          |
| Backtick (`` ` ``)   | Toggle debug console (live state, transitions)              |

---

# Installation Details

### Pipeline (Python)

```bash
poetry install --extras ml    # full install with PyTorch, SAELens, TransformerLens (~2GB)
poetry install                # lightweight: numpy/scipy/sklearn/umap/hdbscan only
```

The lightweight install is enough for similarity circuits and re-running the
frontend on existing data. Co-activation circuits and `striat model` require the
ML extras.

### Frontend (Node.js)

```bash
cd frontend
pnpm install        # or: npm install -g pnpm && pnpm install
pnpm dev            # dev server, default port 5173
pnpm build          # production build
pnpm preview        # serve production build
```

If you don't have pnpm, `./dev.sh` at the project root installs it via corepack
and launches the dev server.

### Tests

```bash
poetry run pytest tests/ -v                  # all fast tests
poetry run pytest tests/ -v -m "not slow"    # skip model-download tests
```

---

# Security Notice

striatica is a localhost research tool. It runs a Vite dev server intended for
local exploration only. No authentication, rate limiting, or input sanitization
has been implemented. Please do not expose it to the public internet as-is.

---

# Tech Stack

| Layer        | Tech                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| 3D rendering | React Three Fiber + Three.js + custom GLSL shaders                      |
| UI           | React 19 + TypeScript + Tailwind CSS 4                                  |
| State        | Zustand 5                                                               |
| Pipeline     | Python 3.12 + SAELens + TransformerLens + scikit-learn + UMAP + HDBSCAN |
| Build        | Vite 7 (frontend), Poetry 2.x (striatica)                               |

---

# Project Structure

```
pipeline/          # Python package — config, download, vectors, reduce, cluster,
                   #   circuits, local_dim, prepare, cli
scripts/           # Entry point scripts (process_gpt2_small, generate_circuits)
tests/             # pytest suite
data/              # Cached downloads (JSONL from Neuronpedia S3, gitignored)

frontend/
  src/
    components/    # UI panels (TopBar, NavPanel, CircuitPanel, DetailPanel, DebugConsole)
    three/         # R3F components (PointCloudMesh, FlyToCamera, CircuitNodes, etc.)
    views/         # View compositions (PointCloudView, CircuitGraphView)
    stores/        # Zustand store (useAppStore)
    shaders/       # Custom GLSL vertex/fragment shaders
    config/        # Centralized rendering parameters
    types/         # TypeScript interfaces
    utils/         # Data loaders, color scales, camera sync
  public/data/     # Generated JSON (gitignored — run striat demo to populate)
```

---

# Roadmap

Planned features, roughly in priority order:

- **Multi-dataset switching** — load multiple model JSONs and switch between
  them in the UI
- **Local Dimension view** — third view mode visualizing per-feature intrinsic
  dimensionality
- **3D export** — export clusters or circuits as glTF/OBJ for use in Blender, 3D
  viewers, or presentations
- **Annotation system** — save and share named camera positions + selection
  states
- **Public deployment mode** — auth, rate limiting, and input sanitization for
  hosted instances

---

# Contributing

striatica is a solo research project and very much a work in progress. If you
run into bugs, have questions, or want to suggest improvements, please open an
issue or start a thread in
[Discussions](https://github.com/catwhisperingninja/striatica/discussions). Pull
requests are welcome.

If something doesn't work on your system, please include your OS, Python
version, Node version, and any error output — it helps enormously.
