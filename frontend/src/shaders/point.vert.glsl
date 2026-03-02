attribute float aSize;
attribute vec3 aColor;
attribute float aOpacity;

// Camera's distance to its orbit target — used to compute relative depth for DOF
uniform float uFocusDist;
// All values from config/rendering.ts — wired as uniforms to stay in sync
uniform float uSizeScale;
uniform float uSizeMin;
uniform float uSizeMax;
uniform float uSoftRange;
uniform float uMinFocusDist;

varying vec3 vColor;
varying float vOpacity;
varying float vDepth;

void main() {
  vColor = aColor;
  vOpacity = aOpacity;

  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  float pointDist = -mvPosition.z; // distance from camera (positive)

  // ── DOF depth: relative to camera's focus distance ──
  // Points AT the focus distance → depth = 0 (crisp)
  // Points much farther → depth = 1 (soft)
  // Points closer than focus → depth stays 0 (always crisp in foreground)
  float depthBeyondFocus = max(pointDist - uFocusDist, 0.0);
  vDepth = clamp(depthBeyondFocus / max(uFocusDist * uSoftRange, uMinFocusDist), 0.0, 1.0);

  // Size attenuates with distance
  gl_PointSize = aSize * (uSizeScale / -mvPosition.z);
  gl_PointSize = clamp(gl_PointSize, uSizeMin, uSizeMax);
  gl_Position = projectionMatrix * mvPosition;
}
