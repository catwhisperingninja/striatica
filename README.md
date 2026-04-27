# striatica

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18848240.svg)](https://doi.org/10.5281/zenodo.18848240)
> **Paper:** _Striatica: A Geometric Atlas for Machine Intelligence_ —
> [Zenodo](https://doi.org/10.5281/zenodo.18848240)

A geometric atlas for machine intelligence — 3D visualization of neural network
interpretability features from sparse autoencoder and transcoder decoder weight
geometry.

<img width="3200" height="1200" alt="striatica-banner_v3_sat" src="https://github.com/user-attachments/assets/1ac347c6-67bc-4346-8227-1bc84ac20bbe" />

# Screenshots
![Default view](img/default-reset%20view.png) _Default view._

![Example view of a feature, a circuit, and the local dimension heatmap display.](img/5781-circview-dim.png)
_Example view of a feature, a circuit, and the local dimension heatmap display._

---

# Semantic Labels and Dual-Use Research

Semantic labels (feature explanations, activation descriptions, interpretability
annotations) are dual-use research material in the same category as biosecurity
and nuclear physics research. They map human-readable meanings to individual
computational features inside neural networks, including features involved in
alignment, honesty, refusal, and safety behaviors.

**The pipeline redacts semantic labels by default.** For small, well-studied
models (GPT-2 Small, Pythia-70M) whose safety circuits are already public
knowledge, labels are included. For all other models, the output contains
geometry only — positions, clusters, local dimensions, and activation statistics.
The `--include-semantics` flag overrides this for authorized research.

**Handling requirements:**

- Semantic labels must never be committed to version control, included in Docker
  images, logged to stdout, or written to files that could be shared.
- Geometry and circuit structure become associatable with semantic labels once
  feature-circuit mappings exist. Treat geometric data from capable models as
  potentially sensitive material that could enable reverse engineering of safety
  mechanisms.
- Audit all outputs and screenshots for exposed semantic data before sharing.
- If your research involves features related to alignment, honesty, refusal, or
  safety behaviors, consult with an AI safety research group before publication
  (e.g., Anthropic, MIRI, ARC, Redwood Research, or your institution's AI safety
  team).

Geometry, topology, and circuit structure without semantic labels are generally
safe to publish for well-studied models. For frontier models, apply the same
judgment you would to any dual-use research output.

---

# Processing a Transcoder

striatica processes Gemmascope transcoders from Google's
[gemma-scope](https://huggingface.co/google/gemma-scope-2b-pt-transcoders)
collection. Transcoders map between transformer layers and produce decoder
vectors in a higher-dimensional space (2304-D for Gemma 2 2B vs 768-D for
GPT-2 Small).

```bash
# Docker (GPU)
docker build -f Dockerfile.gpu -t striatica-gpu .
docker run -t --gpus all -v $(pwd)/output:/app/output striatica-gpu \
  model --transcoder gemma-2-2b/12/604 --device cuda

# On a cloud GPU instance (Vast.ai, Lambda — pip install, no Docker)
python -m pipeline model --transcoder gemma-2-2b/12/604 --device cuda --json-export
```

The `--transcoder` flag takes a `model/layer/l0` spec. The L0 value selects the
sparsity variant — lower L0 means sparser activations. Available variants for
Gemma 2 2B layer 12 width_16k: 6, 10, 18, 32, 60, 111, 204, 359, 604, 955.

Transcoder weights are downloaded from HuggingFace (public, CC-BY-4.0, no auth
required).

### What the pipeline produces

1. Loads transcoder decoder weight vectors from HuggingFace
2. PCA (adaptive: `min(d/4, 300)` components — 300 for 2304-D transcoders)
3. UMAP (PCA output to 3D)
4. HDBSCAN clustering
5. Local dimension estimation (Participation Ratio + VGT growth curves)
6. L1 structural validation (hard gate — pipeline aborts on failure)
7. L2 embedding quality scorecard (trustworthiness, neighborhood overlap)
8. Output JSON with positions, clusters, local dimensions, activation stats

### Transcoder CLI flags

| Flag                 | Description                      | Default                                |
| -------------------- | -------------------------------- | -------------------------------------- |
| `--transcoder`       | Transcoder spec: `model/layer/l0`| —                                      |
| `--transcoder-repo`  | HuggingFace repo ID             | `google/gemma-scope-2b-pt-transcoders` |
| `--transcoder-width` | Width variant                    | `width_16k`                            |
| `--pca-dim`          | PCA components (`auto` or int)  | `auto`                                 |
| `--device`           | `auto`, `cuda`, `mps`, `cpu`    | `auto`                                 |
| `--json-export`      | Export JSON only                 | off                                    |
| `--include-semantics` | Include semantic labels          | off (redacted)                         |

---

# Processing an SAE Model

striatica also processes any SAE dictionary available through
[SAELens](https://github.com/jbloom/SAELens). The Neuronpedia ID is the
simplest path — the CLI auto-resolves the SAELens release name, hook point, and
S3 batch count.

```bash
# Docker (CPU — GPT-2 Small doesn't need a GPU)
docker build -t striatica .
docker run -t -v $(pwd)/output:/app/output striatica \
  model --np-id gpt2-small/6-res-jb

# Docker (GPU — larger models)
docker run -t --gpus all -v $(pwd)/output:/app/output striatica-gpu \
  model --np-id gemma-2-2b/12-gemmascope-res-16k --device cuda
```

### Discover available models

```bash
docker run -t --rm striatica discover --sae-types res
docker run -t --rm striatica discover --families gpt2,gemma2,llama
```

This queries the SAELens pretrained registry — no hardcoded model lists.

### Batch processing

```bash
docker run -t -v $(pwd)/output:/app/output striatica \
  batch --np-ids "gpt2-small/6-res-jb,gpt2-small/8-res-jb" \
  --continue-on-error
```

### Full CLI reference

| Flag                   | Description                                          | Default                                |
| ---------------------- | ---------------------------------------------------- | -------------------------------------- |
| `--np-id`              | Neuronpedia ID (auto-resolves everything)            | —                                      |
| `--model`              | Model ID (explicit mode)                             | —                                      |
| `--layer`              | Layer identifier (explicit mode)                     | —                                      |
| `--sae-release`        | SAELens release name (explicit mode)                 | —                                      |
| `--sae-hook`           | SAELens hook point (explicit mode)                   | —                                      |
| `--transcoder`         | Transcoder spec: `model/layer/l0`                    | —                                      |
| `--transcoder-repo`    | HuggingFace repo for transcoder weights              | `google/gemma-scope-2b-pt-transcoders` |
| `--transcoder-width`   | Width variant directory                              | `width_16k`                            |
| `--pca-dim`            | PCA intermediate dimensions (`auto` or explicit int) | `auto`                                 |
| `--num-batches`        | S3 batch count (auto-probed with `--np-id`)          | 24                                     |
| `--features-per-batch` | Features per S3 batch                                | 1024                                   |
| `--device`             | Torch device: `auto`, `cuda`, `mps`, `cpu`           | `auto`                                 |
| `--json-export`        | Export JSON only, skip frontend launch instructions  | off                                    |
| `--include-semantics`  | Include semantic labels for non-public-tier models   | off                                    |

### Hardware guidance

| Model              | Features | RAM   | Time (approx) | GPU                  |
| ------------------ | -------- | ----- | ------------- | -------------------- |
| GPT-2 Small (demo) | 24,576   | 16GB  | 5-10 min      | not needed           |
| Gemma 2B (16K)     | 16,384   | 16GB  | 10-20 min     | recommended          |
| Gemma 2B (65K)     | 65,536   | 32GB  | 30-60 min     | recommended          |
| Llama 3.1 8B (32K) | 32,768   | 32GB  | 30-60 min     | strongly recommended |

VGT growth curve computation is the bottleneck — it's O(n^2) on the feature
count.

---

# Docker

Two Dockerfiles are provided: `Dockerfile` (CPU) and `Dockerfile.gpu` (NVIDIA
GPU).

```bash
docker build -t striatica .                              # CPU
docker build -f Dockerfile.gpu -t striatica-gpu .        # GPU
```

All CLI subcommands (`model`, `discover`, `batch`, `validate`) work in either
container. Mount a volume to get results out: `-v $(pwd)/output:/app/output`.

### UMAP reproducibility

UMAP output is not reproducible across different library versions, even with the
same `random_state=42`. A single patch-level bump to numpy, scipy, scikit-learn,
umap-learn, or any transitive dependency will silently produce completely
different 3D positions.

The Dockerfiles pin exact versions of the entire UMAP reproducibility chain and
install them directly with pip — no lockfile resolution, no version drift. Docker
is the only guaranteed-reproducible path.

### Platform notes

The Docker images support `amd64` and `arm64` natively (Intel/AMD and Apple
Silicon). The pipeline uses all available CPU cores by default (minus one for
the OS).

---

# Validation

The pipeline includes a 3-level validation suite that runs automatically on
every execution.

```bash
striat validate output.json                              # L1 structural checks
striat validate output.json --compare reference.json     # L1 + L3 comparison
```

**Level 1 (structural integrity)** is a hard gate. It checks array alignment,
position bounds, cluster label validity, feature index continuity, and centroid
accuracy. If any check fails, the pipeline aborts — no corrupt data gets written.

**Level 2 (embedding quality)** produces a scorecard: trustworthiness (Van der
Maaten 2009), neighborhood overlap at multiple k values, silhouette score, PCA
explained variance, and axis spread. Trustworthiness above 0.85 means the 3D
embedding preserves local high-dimensional structure.

**Level 3 (cross-model comparison)** is optional (`--compare` flag) and compares
distributional signatures between two datasets.

A validation sidecar JSON is written alongside every output file with all metric
values.

---

# Visualization

### Views

**Point Cloud** — All features positioned by decoder weight similarity. Color
by cluster membership or local intrinsic dimension. Click any point to inspect
metadata, activation stats, VGT growth curve, and Neuronpedia link.

**Circuits** — Visualize which features participate in a circuit. Nodes colored
by role. Edge threshold slider controls visibility. Circuit integration is being
rebuilt with Neuronpedia Circuit Tracer (replacing the Jaccard co-activation
heuristic).

**Cross-view sync** — Selection, camera position, and cluster highlighting
persist across view switches.

### Controls

| Input                | Action                                         |
| -------------------- | ---------------------------------------------- |
| Click                | Select a feature point                         |
| Search box           | Find features by index or description          |
| Double-click cluster | Fly camera to cluster centroid                 |
| Drag                 | Orbit                                          |
| Scroll               | Zoom                                           |
| Right-drag           | Pan                                            |
| Shift-click cluster  | Multi-select clusters (up to 10)               |
| Cmd+P / Ctrl+P       | Toggle Point Cloud / Circuits view             |
| Backtick             | Toggle debug console                           |

### Running the frontend

```bash
cd frontend && pnpm install && pnpm dev
# Open: http://localhost:5173/?dataset=your-model-output.json
```

---

# Tech Stack

| Layer        | Tech                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| 3D rendering | React Three Fiber + Three.js + custom GLSL shaders                      |
| UI           | React 19 + TypeScript + Tailwind CSS 4                                  |
| State        | Zustand 5                                                               |
| Pipeline     | Python 3.12 + SAELens + TransformerLens + scikit-learn + UMAP + HDBSCAN |
| Validation   | sklearn trustworthiness + neighborhood overlap + silhouette scoring      |
| Build        | Vite 7 (frontend), Docker (pipeline)                                    |

# Project Structure

```
pipeline/          # Python package — config, download, vectors, reduce, cluster,
                   #   circuits, local_dim, prepare, validate, cli, metrics
scripts/           # Launch scripts (vast_launch, cloud_preprocess, providers/)
tests/             # pytest suite
data/              # Cached downloads (JSONL from Neuronpedia S3, gitignored)

frontend/
  src/
    components/    # UI panels (TopBar, NavPanel, CircuitPanel, DetailPanel)
    three/         # R3F components (PointCloudMesh, FlyToCamera, CircuitNodes)
    views/         # View compositions (PointCloudView, CircuitGraphView)
    stores/        # Zustand store (useAppStore)
    shaders/       # Custom GLSL vertex/fragment shaders
    config/        # Centralized rendering parameters
    types/         # TypeScript interfaces
    utils/         # Data loaders, color scales, camera sync
  public/data/     # Generated JSON (gitignored — run pipeline to populate)
```

---

# Security

striatica is a localhost research tool. The Vite dev server is intended for
local exploration only. No authentication, rate limiting, or input sanitization
has been implemented. Do not expose it to the public internet.

---

# Roadmap

- **Circuit Tracer integration** — Causal circuit data via Neuronpedia Circuit
  Tracer transcoders, replacing the Jaccard co-activation heuristic.
- **Transcoder semantic explanations** — Wire Neuronpedia feature explanations
  into the transcoder pipeline path.
- **Multi-model comparison** — Load multiple model datasets and compare
  geometric properties across architectures.
- **Local Dimension view** — Third view mode visualizing per-feature intrinsic
  dimensionality.
- **Pipeline observability** — Grafana dashboards for reproducibility drift,
  performance, and data quality.
- **3D export** — glTF/OBJ export for Blender and presentations.
- **Public deployment mode** — Auth, rate limiting, and hardening for hosted
  instances.

---

# Development

### Poetry (development only)

Poetry is for active development of the pipeline code. It is not the recommended
way to run the pipeline — use Docker for that. The committed `poetry.lock` pins
the versions that produced the current data. Never run `poetry lock` unless you
intend to regenerate all data and visually verify the output.

```bash
poetry install --extras ml    # full install with PyTorch, SAELens, TransformerLens
poetry install                # lightweight: numpy/scipy/sklearn/umap/hdbscan only
```

### Tests

```bash
poetry run pytest tests/ -v                  # all tests
poetry run pytest tests/ -v -m "not slow"    # skip GPU/download tests
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev            # dev server, port 5173
pnpm build          # production build
```

---

# Contributing

striatica is a research project under active development. The geometric pipeline
is stable and producing real data across multiple model families. Circuit
integration is being rebuilt with causal methods.

If you run into issues, please open an issue with your OS, Python version, and
error output. Pull requests are welcome.
