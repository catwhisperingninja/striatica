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

### Debug Console

The debug console (`DebugConsole.tsx`, toggle with backtick) must stay wired to
all views and store state. Every new view or store field should appear in the
debug console.

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
`CircuitEdges.tsx`):

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

GitHub: `github.com/catwhisperingninja/striatica` (private)
