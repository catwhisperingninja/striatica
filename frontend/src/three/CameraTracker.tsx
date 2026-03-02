import { useFrame, useThree } from '@react-three/fiber'
import { cameraSync } from '../utils/cameraSync'
import { CAMERA } from '../config/rendering'
import type { RefObject } from 'react'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { PerspectiveCamera } from 'three'

interface Props {
  controlsRef: RefObject<OrbitControlsImpl | null>
}

/**
 * Syncs main camera pose to the shared cameraSync object every frame,
 * and adapts OrbitControls zoom speed based on camera-to-target distance.
 *
 * Close to target → slow zoom (precise scroll for dense clusters).
 * Far from target → normal zoom (quick navigation).
 */
export default function CameraTracker({ controlsRef }: Props) {
  const { camera } = useThree()

  useFrame(() => {
    const controls = controlsRef.current

    // ── Sync camera state to shared object ──
    const p = camera.position
    cameraSync.position[0] = p.x
    cameraSync.position[1] = p.y
    cameraSync.position[2] = p.z

    const q = camera.quaternion
    cameraSync.quaternion[0] = q.x
    cameraSync.quaternion[1] = q.y
    cameraSync.quaternion[2] = q.z
    cameraSync.quaternion[3] = q.w

    if (controls) {
      const t = controls.target
      cameraSync.target[0] = t.x
      cameraSync.target[1] = t.y
      cameraSync.target[2] = t.z

      // ── Adaptive zoom speed ──
      // Linear scaling: zoomSpeed = baseSpeed * (dist / refDist), clamped
      const dx = p.x - t.x, dy = p.y - t.y, dz = p.z - t.z
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
      const adaptive = CAMERA.zoomSpeed * (dist / CAMERA.zoomRefDistance)
      controls.zoomSpeed = Math.max(CAMERA.zoomMinSpeed, adaptive)
    }

    const pc = camera as PerspectiveCamera
    cameraSync.fov = pc.fov
    cameraSync.aspect = pc.aspect
  })

  return null
}
