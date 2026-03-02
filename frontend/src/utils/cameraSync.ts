/**
 * Shared mutable camera state for cross-Canvas communication.
 * The main Canvas writes here every frame via CameraTracker.
 * The Minimap Canvas reads here every frame to position the frustum.
 * No React state — pure mutation for zero-overhead per-frame sync.
 */
export const cameraSync = {
  position: new Float32Array([0, 0, 3]),
  target: new Float32Array([0, 0, 0]),
  quaternion: new Float32Array([0, 0, 0, 1]),
  fov: 60,
  aspect: 1.5,
}
