# striatica

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18848240.svg)](https://doi.org/10.5281/zenodo.18848240)

A geometric atlas for machine intelligence.

> **Paper:** _Striatica: A Geometric Atlas for Machine Intelligence_ —
> [Zenodo](https://doi.org/10.5281/zenodo.18848240)

![Default view](img/default-reset%20view.png) _Default view._

![Example view of a feature, a circuit, and the local dimension heatmap display.](img/5781-circview-dim.png)
_Example view of a feature, a circuit, and the local dimension heatmap display._

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

# Responsible Use: Interpretability Safety

**This tool can identify features involved in AI safety behaviors.**

When applied to capable models, striatica's pipeline generates semantic
interpretations of individual features and circuits — including features that
participate in alignment, honesty, refusal, and other safety-relevant behaviors.
If these interpretations are published or enter model training data, they could
be used to identify and circumvent safety mechanisms in current and future AI
systems.

**What the pipeline does by default:**

For small, well-studied models (GPT-2 Small, Pythia-70M), semantic labels are
included in the output. These models' safety circuits are either non-existent or
already public knowledge.

For all other models, semantic labels are **redacted by default**. The pipeline
produces geometry, topology, clustering, and circuit structure (all
scientifically useful) without the interpretation layer that maps features to
human-readable meanings. This can be overridden with `--include-semantics`, but
please read the following before doing so.

**If you are publishing research using striatica:**

- Do not publish complete semantic mappings of safety-relevant features for
  capable models.
- If your findings involve features related to alignment, honesty, refusal,
  ethics, or similar safety behaviors, consult with an AI safety research group
  before publication (e.g., Anthropic, MIRI, ARC, Redwood Research, or your
  institution's AI safety team).
- Consider whether your publication could enable targeted ablation of safety
  circuits.
- Geometry, topology, and circuit structure without semantic labels are
  generally safe to publish.

This is dual-use research in the same category as biosecurity and nuclear
physics. The interpretability community's long-term credibility depends on
responsible handling of safety-relevant findings.

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

# Operation

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
  --features-per-batch 1024 \
  --device auto
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

| Optional               | Description                                | Default |
| ---------------------- | ------------------------------------------ | ------- |
| `--num-batches`        | Neuronpedia S3 batch count                 | 24      |
| `--features-per-batch` | Features per S3 batch                      | 1024    |
| `--device`             | Torch device: `auto`, `cuda`, `mps`, `cpu` | `auto`  |
| `--json-export`        | Export JSON only, no frontend instructions | off     |

To find the right values for your model, check the
[SAELens model list](https://github.com/jbloom/SAELens) and
[Neuronpedia](https://neuronpedia.org).

### Hook point formats

SAELens supports multiple hook point formats depending on the model and release:

**TransformerLens hook points** (most common):

- `blocks.{N}.hook_resid_pre` — before residual stream at layer N
- `blocks.{N}.hook_resid_post` — after residual stream at layer N

**Newer canonical releases** (SAELens standardized format):

- `layer_{N}/width_{K}k/canonical` — e.g., `layer_6/width_16k/canonical`

Check the SAELens documentation or Neuronpedia for your model's available hook
points. The `--sae-hook` flag accepts any string — it's passed directly to
`SAE.from_pretrained()`.

### Remote compute, local visualization

To run the pipeline on a remote server (e.g., a GPU instance) and view results
on your local machine:

```bash
# On remote server
poetry run striat model \
  --model gemma-2b \
  --layer 12-res-jb \
  --sae-release gemma-2b-res-jb \
  --sae-hook blocks.12.hook_resid_pre \
  --device cuda \
  --json-export

# Copy the output JSON to your local machine
scp remote:path/to/striatica/frontend/public/data/gemma-2b-12-res-jb.json \
  ./frontend/public/data/

# On local machine — launch frontend and open the dataset
cd frontend && pnpm dev
# Open: http://localhost:5173/?dataset=gemma-2b-12-res-jb.json
```

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

#### Linux prerequisites

Some Python dependencies (notably hdbscan) compile C extensions from source. On
Ubuntu/Debian, install build tools before running `poetry install`:

```bash
sudo apt update && sudo apt install -y build-essential python3-dev
```

#### GPU and VM notes

The data pipeline device flag controls where PyTorch runs SAE and model
inference. On an NVIDIA GPU, this is `cuda`. On Apple Silicon, `mps`. If you are
in a VM without GPU passthrough or on a system without a supported GPU, the
pipeline falls back to `cpu` — everything still works, just slower for larger
models.

| System                             | Device |
| ---------------------------------- | ------ |
| NVIDIA GPU (native or passthrough) | `cuda` |
| Apple Silicon (M1/M2/M3/M4)        | `mps`  |
| VM without GPU / CPU-only          | `cpu`  |

GPT-2 Small runs comfortably on CPU. Gemma 2B and larger models benefit
significantly from GPU acceleration.

### Frontend (Node.js)

```bash
cd frontend
pnpm install        # or: npm install -g pnpm && pnpm install
pnpm dev            # dev server, default port 5173
pnpm build          # production build
pnpm preview        # serve production build
```

If you don't have pnpm, `corepack enable` will make it available (requires
Node.js 18+).

### Tests

```bash
poetry run pytest tests/ -v                  # all fast tests
poetry run pytest tests/ -v -m "not slow"    # skip model-download tests
```

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
