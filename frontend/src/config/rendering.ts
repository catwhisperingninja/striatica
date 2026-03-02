// frontend/src/config/rendering.ts
// ─────────────────────────────────────────────────────────────────────
// Centralized rendering parameters. Every magic number that controls
// visual appearance or camera behavior lives here. If you change one
// value and something else breaks, the coupling is documented below.
// ─────────────────────────────────────────────────────────────────────

// ── Camera ──────────────────────────────────────────────────────────
export const CAMERA = {
  /** Default camera position (reset target). Coupled to FOV + data bounds. */
  defaultPosition: [-0.3, 0.01, 1.2] as [number, number, number],
  /** Default orbit target (center of data). */
  defaultTarget: [0, 0, 0] as [number, number, number],
  /** Field of view in degrees. Coupled to computeViewDistance(). */
  fov: 60,
  /** Orbit control speeds (no damping — linear stop). */
  rotateSpeed: 0.5,
  /** Base zoom speed at reference distance. Dynamically scaled by CameraTracker. */
  zoomSpeed: 0.8,
  panSpeed: 0.5,
  /** Distance at which zoomSpeed is at its base value. */
  zoomRefDistance: 3,
  /** Minimum zoom speed (when very close to target). */
  zoomMinSpeed: 0.04,
} as const

// ── Fly-To Animation ────────────────────────────────────────────────
export const FLY_TO = {
  /** How far to frame the target (radius in scene units). */
  desiredRadius: 0.3,
  /** Minimum distance from target to avoid clipping. */
  minViewDistance: 0.8,
  /** Flight speed: progress = delta * speed per frame. */
  speed: 2,
  /** Easing power for the ease-out curve (3 = cubic). Higher = more deceleration. */
  easePower: 3,
  /** Direction fallback when camera is on top of the target. */
  fallbackDir: [0, 0.3, 1] as [number, number, number],
  /** Minimum direction vector length to avoid zero-division. */
  dirEpsilon: 0.001,
} as const

// ── Point Rendering ─────────────────────────────────────────────────
export const POINTS = {
  /** Shader gl_PointSize multiplier. Coupled to camera FOV + distance. */
  sizeScale: 300,
  /** Minimum rendered point size in pixels. */
  sizeMin: 1.0,
  /** Maximum rendered point size in pixels. */
  sizeMax: 20.0,
} as const

// ── Depth of Field ──────────────────────────────────────────────────
// DOF is computed RELATIVE to camera distance (uFocusDist uniform).
// See CLAUDE.md "DOF Bug Pattern" — never use absolute thresholds.
export const DOF = {
  /** Points beyond focusDist * softRange start softening. */
  softRange: 3.0,
  /** Minimum normalization distance (prevents division-by-tiny-number). */
  minFocusDist: 2.0,
  /** Crisp inner-edge radius at depth=0 (0.5 = perfectly sharp circle). */
  crispEdge: 0.47,
  /** Soft inner-edge radius at depth=1 (lower = softer). */
  softEdge: 0.35,
  /** Alpha multiplier falloff for distant points (0–1 range). */
  depthFadeFactor: 0.3,
} as const

// ── Opacity Palette ─────────────────────────────────────────────────
// Semantic opacity levels used across all rendering states.
export const OPACITY = {
  /** Selected point: highest visual priority. */
  selected: 0.95,
  /** Cluster members when their cluster is active. */
  clusterActive: 0.45,
  /** Variation amplitude for cluster member opacity dithering. */
  clusterVariation: 0.15,
  /** Same-cluster context when flying to a point. */
  flyContext: 0.25,
  /** Variation amplitude for fly-context dithering. */
  flyContextVariation: 0.1,
  /** Default resting opacity for categorized points. */
  defaultCategorized: 0.4,
  /** Variation amplitude for default state. */
  defaultVariation: 0.3,
  /** Default resting opacity for uncategorized points. */
  defaultUncategorized: 0.2,
  /** Nearly-invisible: deselected points when a cluster is active. */
  deselected: 0.02,
  /** Background noise in fly-to: categorized. */
  flyBackgroundCat: 0.06,
  /** Background noise in fly-to: categorized variation. */
  flyBackgroundCatVariation: 0.04,
  /** Background noise in fly-to: uncategorized. */
  flyBackgroundUncat: 0.03,
  /** Uncategorized when isolate mode is on. */
  isolatedUncat: 0.65,
  /** Everything else when isolate mode is on. */
  isolatedOther: 0.02,
} as const

// ── Size Palette ────────────────────────────────────────────────────
export const SIZE = {
  /** Selected point size. */
  selected: 4.5,
  /** Active cluster member size. */
  clusterActive: 2.5,
  /** Same-cluster context during fly-to. */
  flyContext: 2.0,
  /** Default categorized points. */
  defaultCategorized: 2.5,
  /** Default uncategorized points. */
  defaultUncategorized: 1.8,
  /** Deselected (nearly invisible). */
  deselected: 0.5,
  /** Background categorized in fly-to. */
  flyBackgroundCat: 1.2,
  /** Background uncategorized in fly-to. */
  flyBackgroundUncat: 0.8,
  /** Uncategorized when isolate mode is on. */
  isolatedUncat: 3.0,
  /** Active cluster member in localDim mode. */
  localDimActive: 3.2,
  /** Default in localDim mode. */
  localDimDefault: 2.5,
} as const

// ── Circuit Member Rendering ────────────────────────────────────────
export const CIRCUIT = {
  /** Size multiplier for famous circuit features. */
  famousSizeMultiplier: 3.0,
  /** Minimum size for famous circuit features. */
  famousSizeMin: 7.0,
  /** Opacity for famous circuit features. */
  famousOpacity: 0.95,
  /** Color mix toward white for famous (0=pure color, 1=white). */
  famousWhiteMix: 0.7,
  /** Size multiplier for regular circuit features. */
  regularSizeMultiplier: 2.0,
  /** Minimum size for regular circuit features. */
  regularSizeMin: 5.0,
  /** Opacity for regular circuit features. */
  regularOpacity: 0.85,
  /** Color mix toward white for regular. */
  regularWhiteMix: 0.5,
} as const

// ── Interaction ─────────────────────────────────────────────────────
export const INTERACTION = {
  /** Raycaster hover threshold, scaled by camera distance. */
  hoverThresholdScale: 0.025,
} as const

// ── Selection Ring ──────────────────────────────────────────────────
export const SELECTION_RING = {
  /** Ring scale relative to camera distance. Coupled to hoverThresholdScale. */
  scaleFromDistance: 0.025,
  /** Inner radius of ring geometry. */
  innerRadius: 0.8,
  /** Outer radius of ring geometry. */
  outerRadius: 1.0,
  /** Number of segments in ring geometry. */
  segments: 32,
  /** Ring color (hex). */
  color: 0xffffff,
  /** Ring opacity. */
  opacity: 0.45,
} as const

// ── Circuit Background (Circuit View) ───────────────────────────────
export const CIRCUIT_BG = {
  bgOpacity: 0.04,
  bgOpacityUncat: 0.01,
  bgSize: 1.0,
  selectedPointOpacity: 0.9,
  selectedPointSize: 4.0,
  selectedClusterOpacity: 0.5,
  selectedClusterSize: 2.5,
} as const

// ── Labels ──────────────────────────────────────────────────────────
export const LABELS = {
  /** Camera distance at which famous labels start appearing. */
  famousFadeIn: 4.0,
  /** Camera distance at which famous labels fully disappear. */
  famousFadeOut: 6.0,
  /** Camera distance at which regular labels start appearing. */
  regularFadeIn: 2.2,
  /** Camera distance at which regular labels fully disappear. */
  regularFadeOut: 3.5,
  /** Minimum opacity for famous labels when in range. */
  famousMinOpacity: 0.4,
  /** Opacity interpolation speed (lerp factor). */
  fadeSpeed: 0.15,
  /** Opacity below which labels are hidden. */
  hideThreshold: 0.02,
  /** Opacity below which pointer events are disabled. */
  pointerThreshold: 0.1,
} as const

// ── Dataset ────────────────────────────────────────────────────────
export const DATASET = {
  /** Default dataset JSON path (relative to public/). */
  defaultPath: '/data/gpt2-small-6-res-jb.json',
} as const

// ── Color Palettes ─────────────────────────────────────────────────
// Single source of truth for colors used across NavPanel, DetailPanel,
// CircuitPanel, CircuitNodes, Minimap, and GrowthCurveChart.
export const COLORS = {
  /** Cluster colors (indexed by cluster ID % length). */
  clusters: [
    '#06b6d4', '#d946ef', '#22c55e', '#f59e0b', '#f43f5e',
    '#6366f1', '#a78bfa', '#14b8a6', '#f97316', '#ec4899',
  ] as readonly string[],
  /** Circuit node role colors (CSS hex strings). */
  roles: {
    source: '#22c55e',
    intermediate: '#06b6d4',
    sink: '#f59e0b',
  } as Record<string, string>,
  /** Circuit node role colors for Three.js (bright variants). */
  roles3D: {
    source: '#4ade80',
    intermediate: '#06b6d4',
    sink: '#fbbf24',
  } as Record<string, string>,
  /** Accent color for charts and highlights. */
  accent: '#06b6d4',
} as const

// ── Uncategorized Points ────────────────────────────────────────────
export const UNCATEGORIZED = {
  /** Color dimming factor for uncategorized points. */
  colorDimFactor: 0.25,
} as const

// ── Circuit Nodes (Circuit View) ────────────────────────────────────
export const CIRCUIT_NODES = {
  source: { baseSize: 8.0, sizeScale: 6.0, minOpacity: 0.8 },
  intermediate: { baseSize: 3.0, sizeScale: 3.0, minOpacity: 0.4 },
  sink: { baseSize: 5.0, sizeScale: 4.0, minOpacity: 0.6 },
  /** Size multiplier when a node is selected. */
  selectedSizeMultiplier: 1.5,
  /** Opacity when a node is selected. */
  selectedOpacity: 1.0,
} as const

// ── Circuit Edges ───────────────────────────────────────────────────
export const CIRCUIT_EDGES = {
  /** Opacity multiplier for visible edges. */
  opacityMultiplier: 0.6,
  /** Minimum alpha before discarding in shader. */
  discardThreshold: 0.001,
} as const

// ── Minimap (3D Frustum) ────────────────────────────────────────────
export const MINIMAP = {
  /** Maximum downsampled points to render. */
  maxPoints: 400,
  /** Point size in pixels (sizeAttenuation off). */
  pointSize: 1.5,
  /** Point opacity. */
  pointOpacity: 0.3,
  /** Selected-point sphere radius. */
  selectedSize: 0.03,
  /** XYZ axis half-length. */
  axisLength: 0.4,
  /** Axis line opacity. */
  axisOpacity: 0.2,
  /** Frustum wireframe color (lime green to match theme). */
  frustumColor: '#A3D739',
  /** Frustum wireframe opacity. */
  frustumOpacity: 0.75,
  /** Frustum length as fraction of camera-to-target distance. */
  frustumLengthFraction: 0.5,
  /** Minimum frustum length (so it stays visible when zoomed in). */
  frustumMinLength: 0.25,
  /** Maximum frustum length. */
  frustumMaxLength: 1.0,
  /** Orthographic camera zoom (controls how much scene fits in view). */
  cameraZoom: 42,
  /** Orthographic camera position (isometric angle). */
  cameraPosition: [3, 2, 3] as [number, number, number],
} as const
