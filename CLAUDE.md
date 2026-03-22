# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

**striatica** is a geometric atlas for machine intelligence — 3D visualization
of neural network interpretability features (sparse autoencoder feature
geometry). The core vision is described in
`interpretability-visualization-feasibility.md`.

## Repository Structure

This repo has two distinct parts:

### 1. Skills Collection

Each top-level directory (except `mem0/`) is a Claude Code skill with a
`SKILL.md` defining its behavior. Skills cover: After Effects scripting, agent
development, arxiv search, brainstorming, data storytelling, deep research,
Excel analysis, frontend design, Grafana dashboards, motion design ideation, PDF
processing, remembering conversations, scientific critical thinking, skills
discovery, subagent-driven development, UI styling, and XLSX creation.

Skills with `references/` subdirectories contain supporting documentation that
the skill references.

### 2. mem0 Plugin (`mem0/`)

A TypeScript npm package (`claude-code-mem0`) providing persistent memory for
Claude Code sessions via the Mem0 API. It exposes CLI commands (`mem0-pull`,
`mem0-push`, `mem0-session-end`) used as Claude Code hooks.

## Build & Development Commands

### xlsx Skill

Formula recalculation requires LibreOffice:

```bash
python xlsx/recalc.py <excel_file> [timeout_seconds]
```

### arxiv-search Skill

```bash
python arxiv-search/arxiv_search.py
```

### PDF Processing

Requires: `pip install pdfplumber pypdf pillow pytesseract pandas` OCR requires:
`brew install tesseract`

## CI/CD

No CI/CD pipeline is currently configured. The previous GitLab CI setup
(Auto-DevOps with SAST and secret detection) was not transferred. GitHub Actions
can be added as needed.

## Tech Stack (Visualization Project)

A feasibility study was performed and recommends:

- **Interactive viz:** React + React Three Fiber + custom shaders
- **Video output:** Remotion wrapping R3F scenes
- **Data pipeline:** Python (SAELens + TransformerLens) -> dimensionality
  reduction -> JSON for frontend
- **Key algorithms:** PCA -> 4D subspace -> stereographic projection -> 3D,
  cross-sectional slicing via Three.js clipping planes
- **Local inference:** 32GB VRAM for SAELens on Gemma 2B or Llama 3 8B

## Critical Rules

### No Mock Data

All views must use real data from actual datasets (e.g., GPT-2 Small from
Neuronpedia). No synthetic/mock JSON, no placeholder circuits, no hardcoded
feature indices. If a feature requires data we don't have yet, build the
pipeline to fetch/compute it first — don't scaffold with fakes. Delete
`sample-circuit.json` and any similar mocks.

### No Debugging Unbuilt Features

Don't debug a view that isn't actually built. If the data pipeline doesn't
exist, the view isn't ready to debug. Enter plan mode and build it properly
instead of patching scaffolding.

### No Hardcoded Rendering Values

Every visual parameter (opacity, size, color mix, camera speed, DOF range, label
fade distance, etc.) lives in `frontend/src/config/rendering.ts`. New components
MUST import from this config. Do not introduce magic numbers in rendering code —
add a new config section instead. This rule exists because hardcoded values were
the root cause of the DOF bug, the brightness overwhelm bug, and several other
issues that were painful to track down.

### No Hardcoded Test Assumptions

Specific properties are to be used to perform tests ("feature index needs to be
LOCAL index") rather than generalities ("not all features in same circuit",
"circuit does not include >60% of features") that may or may not ever have a
true edge case.

### Debug Console

The debug console (`DebugConsole.tsx`, toggle with backtick) must stay wired to
all views and store state. Every new view or store field should appear in the
debug console.

### Scope Discipline — One Bug, One File

When fixing a bug, only edit files directly implicated by the failing test. If
you find yourself reaching into an unrelated file "while you're in there," stop
and ask first. The circuit contamination fix of 2026-03-21 touched
`pipeline/cluster.py`, `pipeline/prepare.py`, and `pipeline/reduce.py` — none of
which had anything to do with the bug. All three were reverted. Scope creep
committed together with the real fix makes rollback painful and obscures what
actually broke.

### No Autonomous Data Regeneration

Never regenerate production data files (circuit JSONs, dataset JSONs, or any
file under `frontend/public/data/`) without explicit user approval. Fix the
code, show the diff, and wait. The user decides when to regenerate data because
they can visually verify the output — you cannot.

### Never Run `poetry lock` in Dockerfiles or CI

The `poetry.lock` file pins the exact library versions that produced the current
data files. UMAP output is **not reproducible across different library
versions**, even with the same `random_state`. A single patch-level bump to
umap-learn, numpy, scipy, numba, or pynndescent will silently produce
completely different 3D feature positions.

**Rules:**

- Dockerfiles MUST use `poetry install` with the committed lockfile. Never
  `poetry lock` or `poetry lock --no-update` inside a container build.
- CI/CD scripts MUST NOT regenerate the lockfile.
- The lockfile is updated **only** on the developer's machine, deliberately,
  when a dependency change is needed. After updating, ALL data must be
  regenerated and visually verified before committing.
- If `poetry install` fails because pyproject.toml has deps not in the
  lockfile, the fix is to update the lockfile locally and commit it — not to
  auto-lock in the build.

**Root cause (2026-03-21):** `poetry lock` was added to both Dockerfiles to
handle a pyproject.toml update. The container resolved newer dependency
versions from PyPI, which changed the UMAP projection. Feature positions in the
Docker-generated dataset were completely different from the original Feb 27
data, despite identical code and `random_state=42`.

### Tests Are Necessary But Not Sufficient

Passing a numerical test assertion (e.g., "overlap < 20%") does not mean the
visualization is correct. For any change that affects what the user sees
(rendering, data pipeline, circuit generation), the verification loop is:

1. Fix code
2. Show diff to user for review
3. User regenerates data
4. User visually verifies the result
5. User approves

Do not treat green tests as the finish line for visual/rendering changes.

### Co-Activation Circuit Methodology — KNOWN LIMITATION (2026-03-21 triage)

The co-activation circuit extraction in `pipeline/circuits.py` uses Jaccard
similarity over token positions to infer feature relationships. This is a
**correlational heuristic**, not a causal attribution method. It has been
present unchanged since the code was first written (GitLab era, confirmed via
MD5 hash comparison between first commit `5b73c7d` and current HEAD).

**The contamination problem:** Broadly-activating features (high `frac_nonzero`)
co-occur with nearly everything, so Jaccard gives them high similarity to every
other feature in every circuit. 10 specific features (#316, #979, #2039, #7496,
#9088, #10423, #16196, #23111, #23123, #23373) saturated all 5 co-activation
circuits. These are NOT computationally related to all those circuits — they
just fire on common tokens.

**What is NOT affected:** The geometric pipeline (`process_gpt2_small.py`: PCA,
UMAP, HDBSCAN, VGT on raw SAE decoder vectors) does not use circuit code. The
frontend dataset JSON (`gpt2-small-6-res-jb.json`) is produced by this pipeline
and contains positions, clusters, local dimensions, and activation stats — all
clean. Verified 2026-03-21: 134 features with the same activation frequency
range as the contaminated features have mean VGT ≈ 21; the Cluster #30
contaminated features have VGT ≈ 0.14. The geometric convergence is real and not
explained by activation frequency (Pearson correlation between frac_nonzero and
VGT = 0.03).

**What IS affected:** Any claim about multi-circuit membership ("in 6
circuits"), circuit role assignments (source/processing/sink by breadth), and
edge directionality. These are heuristic labels from the Jaccard pipeline, not
computationally meaningful in the Olah/mechanistic interpretability sense.
Features with selective activation (appearing in 1–3 circuits) may be valid
signal; features saturating all circuits are contamination artifacts.

**Five ungrounded assumptions in `extract_coactivation_circuit`:**

1. Top-k by peak activation = importance (no citation)
2. Jaccard over token positions = computational relationship (correlational, not
   causal)
3. Activation breadth = role (no published basis; Olah uses layer position)
4. Edge direction by breadth (no basis in circuits literature)
5. Arbitrary thresholds: min_coactivation=0.1, top_k=30 (not data-derived)

**Paper impact:** The paper's geometric findings (convergence singularity
geometry, VGT dimensional structure, cluster topology, uncategorized interior
population) are clean. Circuit-specific claims (multi-circuit convergence,
role-based analysis) and all Circuits View screenshots need revision. The paper
was framed as a tool paper with observational findings, which is the correct
framing. Integration with causal circuit methods (Neuronpedia Circuit Tracer) is
prioritized and in-progress.

**File lineage:**

- `pipeline/circuits.py`: unchanged since first commit. MD5
  `2cdc146de262b215355f4845a2dc167e`.
- `frontend/public/data/gpt2-small-6-res-jb.json`: produced Feb 27, never
  touched by circuit code.
- `frontend/public/data/circuits/*.json`: regenerated by bad commit `3ef0c2a`
  (March 21). Current on-disk versions are post-decontamination from that
  commit, NOT the originals the paper screenshots were taken from.
- `data/*.jsonl`: raw Neuronpedia downloads, gitignored, never modified by any
  pipeline code.

### Circuit Data — Two Separate File Paths

This has caused confusion. Be explicit about which data you mean:

1. `frontend/public/data/gpt2-small-6-res-jb.json` — the main dataset JSON.
   Produced by `process_gpt2_small.py` (PCA, UMAP, HDBSCAN, VGT). Contains
   positions, clusters, local dimensions, activation stats. **Circuit code never
   reads or writes this file.**

2. `frontend/public/data/circuits/*.json` — circuit JSONs. Produced by
   `generate_circuits.py` which calls `extract_coactivation_circuit` and
   `extract_similarity_circuit`. **These are the only files the circuit code
   writes.**

The UI composites both at runtime. A screenshot showing "Feature #23373, Cluster
30, dim=0.1, in 6 circuits" displays data from BOTH files — geometry from file 1
(clean), circuit membership from file 2 (contaminated).

## Known Bug Patterns

### DOF (Depth-of-Field) Foreground Defocus — RECURRING

**Symptom:** After changing global rendering parameters (opacity, size, color),
adjusting camera fly-to distances, or modifying the point shader, the foreground
becomes blurry/defocused instead of only the background.

**Root cause:** The vertex shader computes `vDepth` which drives edge softening
in the fragment shader. If `vDepth` uses hardcoded absolute depth thresholds
(e.g., `(-mvPosition.z - 1.0) / 8.0`), any change to camera distance or scene
scale breaks the near/far assumptions, causing foreground points to get non-zero
depth values → soft edges → defocused appearance.

**Fix:** `vDepth` MUST be computed **relative to camera focus distance**, not
absolute. The uniform `uFocusDist` (updated per frame in `PointCloudMesh.tsx`
via `useFrame`) provides the camera's current distance to its orbit target.
Points closer than `uFocusDist` always get `vDepth = 0` (fully crisp). Only
points **beyond** the focus distance soften.

**When modifying:**

- `point.vert.glsl` — never use hardcoded depth thresholds. Always use
  `uFocusDist`.
- `PointCloudMesh.tsx` — the `useFrame` that updates `uFocusDist` must remain.
  If you restructure the material or geometry, ensure this uniform still gets
  updated every frame.
- `FlyToCamera.tsx` — camera distance changes are fine; the per-frame uniform
  update handles it.
- Any new shader material for points — must include `uFocusDist` uniform and the
  relative depth calculation.

## Working UI Elements — DO NOT BREAK

This section documents every working feature as of 2026-02-28. If you're
building a new view (e.g., Local Dimension view), you must preserve ALL of these
behaviors. Test each one after your changes.

### Rendering Pipeline

The rendering pipeline is centralized and battle-tested. All visual parameters
live in `frontend/src/config/rendering.ts`. Every component imports from this
config — no magic numbers anywhere in the rendering code.

**Point Cloud rendering** (`PointCloudMesh.tsx`):

- Custom GLSL vertex/fragment shaders (`point.vert.glsl`, `point.frag.glsl`)
  with additive blending
- Per-point attributes: `aSize`, `aColor`, `aOpacity` updated in-place via
  `needsUpdate = true`
- Multi-tier opacity/size hierarchy depending on state: selected point →
  same-cluster context → deselected background
- Deterministic per-point variation via Knuth hash for natural dithering
- DOF via `uFocusDist` uniform updated every frame in `useFrame` (see Known Bug
  Patterns)
- Circuit member features get boosted size/opacity/glow; famous circuits get
  extra boost
- Uncategorized points colored by nearest cluster centroid, dimmed by
  `UNCATEGORIZED.colorDimFactor`

**Circuit view rendering** (`CircuitBackground.tsx`, `CircuitNodes.tsx`,
`CircuitEdges.tsx`): **_FIX PENDING CIRCUIT TRANSCODER MAPPING COMPLETION_**

- Background shows ALL 24,576 points at very low opacity for spatial context
- CircuitNodes renders circuit members as bright points with role-based colors
  (source=green, intermediate=cyan, sink=amber)
- CircuitEdges renders weighted line segments between nodes, thresholded by the
  edge threshold slider
- All three use the same GLSL shaders as the point cloud for visual consistency
- Selection ring visible in both views

### Cross-View Sync (CRITICAL)

This is the system that makes view switching seamless. It took significant
debugging to get right. The contract is:

1. **Selection persists across view switches.** `selectedIndex` and
   `selectedClusters` in Zustand store are never cleared by view switching. Only
   explicit user actions (clicking a different point, or resetting) change
   selection.

2. **Camera position is preserved.** `Canvas3D` renders both views inside the
   same `<Canvas>`. The same `OrbitControls` ref and `FlyToCamera` component
   persist across view switches. No camera reset on view change.

3. **Circuit membership is pre-indexed at startup.** `buildCircuitMembership()`
   loads ALL circuit JSONs and builds a `Map<featureIndex, circuitId[]>` so the
   point cloud can highlight circuit members without loading individual
   circuits.

4. **Circuit auto-loading on view switch.** When switching to Circuits view with
   a feature selected that belongs to a circuit, `CircuitPanel` auto-loads the
   first matching circuit (`autoLoadedRef` prevents re-triggering).

5. **Selection is locked in Circuits view.** `PointCloudInteraction` is only
   rendered in `PointCloudView`, not `CircuitGraphView`. Users must switch back
   to Points (⌘P) to change selection.

6. **`setSelected(index)` auto-sets cluster.** When you select a point, the
   store automatically sets `selectedClusters` to that point's cluster. This
   drives both the NavPanel highlight and the point cloud dimming.

7. **Reset is atomic.** `resetSelection()` clears selectedIndex, hoveredIndex,
   selectedClusters, flyTarget, isolateUncategorized, circuitData, switches to
   pointCloud view, and increments `resetKey` — all in a single Zustand `set()`
   call. `resetKey` triggers FlyToCamera to fly back to
   `CAMERA.defaultPosition`.

### Camera System

- `FlyToCamera.tsx`: Ease-out cubic approach with linear snap for final 5% (AE
  keyframe style)
- `OrbitControls`: No damping (`enableDamping={false}`) — snappy linear stop on
  mouse release
- Reset flies camera back to `[0, 0, 3]` looking at `[0, 0, 0]`
- `computeViewDistance()` uses FOV to calculate proper framing distance for
  fly-to
- All camera params from `CAMERA` config

### UI Panels

- **TopBar**: View mode tabs with ⌘P shortcut hint, color mode picker, reset
  button
- **NavPanel** (Points view): Cluster tree with search, click to select/fly-to,
  shift-click multi-select (max 10), uncategorized isolate toggle, shows
  indicator for selected point's cluster
- **CircuitPanel** (Circuits view): Circuit list from manifest, edge threshold
  slider, auto-loads circuit when entering view with selected circuit member
- **DetailPanel**: Feature info (index, explanation, activation stats), VGT
  growth curve chart, Neuronpedia link
- **StatusBar**: Point count, current state, hover info
- **DebugConsole**: Toggle with backtick, shows live store state and transitions

### Labels

- `CircuitLabels.tsx`: ONE label per circuit positioned at the most-connected
  feature (highest edge count)
- Technical circuit name always shown (e.g., "coact-capital-of-france")
- Famous circuits get category + citation second line
- Diagonal SVG leader line from point to text
- Distance-based opacity with smooth lerp, separate fade ranges for famous vs
  regular
- All thresholds from `LABELS` config

### Keyboard Shortcuts

- ⌘P / Ctrl+P: Toggle between Point Cloud and Circuits views
- Backtick (`): Toggle debug console

## Adding a New View — Checklist

When building the Local Dimension view (or any new view):

1. **Create the view component** in `frontend/src/views/` following the pattern
   of `PointCloudView.tsx` and `CircuitGraphView.tsx`
2. **Add the view to Canvas3D.tsx** as a conditional render:
   `{dataset && viewMode === 'localDim' && <LocalDimView />}`
3. **Preserve cross-view sync**: The new view MUST read `selectedIndex`,
   `selectedClusters`, and `selectedPos` from the store, and show the
   `SelectionRing` when a point is selected
4. **Decide on interaction**: Does this view allow clicking to select points? If
   yes, include `<PointCloudInteraction>`. If locked like Circuits, omit it.
5. **Use the same shaders**: Import `point.vert.glsl` and `point.frag.glsl` for
   visual consistency. If you need the DOF effect, add `uFocusDist` uniform and
   the `useFrame` updater.
6. **Wire to DebugConsole**: Any new store fields should appear in the debug
   console
7. **Update ⌘P shortcut**: If the view should be part of the cycle, update the
   toggle logic in `App.tsx`
8. **All rendering values from `config/rendering.ts`**: Add a new section to the
   config if needed (e.g., `LOCAL_DIM_VIEW`). No hardcoded numbers.
9. **Test ALL existing behaviors**: After building, verify every item in
   "Working UI Elements" still works — especially selection sync, camera fly-to,
   reset, DOF, and dimming.

## ViewMode Type

`ViewMode` is defined in `frontend/src/types/feature.ts` as
`'pointCloud' | 'circuits' | 'localDim'`. The `'localDim'` value already exists
in the type but has no corresponding view component yet.

## API Keys & Secrets

- **Neuronpedia API key**: `NEURONPEDIA_API_KEY` in `.env` at project root
- Access via Python `dotenv` module (`pip install python-dotenv`):
  `load_dotenv()` from project root
- Never commit `.env` or log key values

## Remote

GitHub: `github.com/catwhisperingninja/striatica`
