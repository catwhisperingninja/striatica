varying vec3 vColor;
varying float vOpacity;
varying float vDepth;

// All values from config/rendering.ts — wired as uniforms to stay in sync
uniform float uCrispEdge;
uniform float uSoftEdge;
uniform float uDepthFadeFactor;

void main() {
  float dist = length(gl_PointCoord - vec2(0.5));
  if (dist > 0.5) discard;

  // Depth-of-field: near points have crisp edges, far points soften slightly
  float innerEdge = mix(uCrispEdge, uSoftEdge, vDepth);
  float alpha = smoothstep(0.5, innerEdge, dist) * vOpacity;

  // Slight opacity falloff for distant points
  float depthFade = 1.0 - vDepth * uDepthFadeFactor;
  alpha *= depthFade;

  gl_FragColor = vec4(vColor, alpha);
}
