import { useRef, type RefObject } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useAppStore } from '../stores/useAppStore'
import { CAMERA, FLY_TO } from '../config/rendering'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

const DEFAULT_CAMERA_POS = new THREE.Vector3(...CAMERA.defaultPosition)
const DEFAULT_TARGET = new THREE.Vector3(...CAMERA.defaultTarget)

function computeViewDistance(camera: THREE.PerspectiveCamera): number {
  const halfFovRad = (camera.fov / 2) * (Math.PI / 180)
  const dist = FLY_TO.desiredRadius / Math.tan(halfFovRad)
  return Math.max(dist, FLY_TO.minViewDistance)
}

interface Props {
  controlsRef: RefObject<OrbitControlsImpl | null>
}

export default function FlyToCamera({ controlsRef }: Props) {
  const { camera } = useThree()
  const startPos = useRef(new THREE.Vector3())
  const endPos = useRef(new THREE.Vector3())
  const targetVec = useRef(new THREE.Vector3())
  const startTarget = useRef(new THREE.Vector3())
  const progress = useRef(1)
  const lastFlyKey = useRef(0)
  const lastResetKey = useRef(0)

  const flyTarget = useAppStore((s) => s.flyTarget)
  const flyKey = useAppStore((s) => s.flyKey)
  const resetKey = useAppStore((s) => s.resetKey)

  function beginFlight(dest: THREE.Vector3, lookTarget: THREE.Vector3) {
    startPos.current.copy(camera.position)
    if (controlsRef.current) {
      startTarget.current.copy(controlsRef.current.target)
    } else {
      startTarget.current.copy(DEFAULT_TARGET)
    }
    endPos.current.copy(dest)
    targetVec.current.copy(lookTarget)
    if (controlsRef.current) {
      controlsRef.current.enabled = false
    }
    progress.current = 0
  }

  // Fly to a specific feature
  if (flyTarget && flyKey !== lastFlyKey.current) {
    lastFlyKey.current = flyKey
    const target = new THREE.Vector3(...flyTarget)
    const dir = new THREE.Vector3().subVectors(camera.position, target)
    if (dir.lengthSq() < FLY_TO.dirEpsilon) {
      dir.set(...FLY_TO.fallbackDir)
    }
    dir.normalize()
    const viewDist = computeViewDistance(camera as THREE.PerspectiveCamera)
    const dest = target.clone().addScaledVector(dir, viewDist)
    beginFlight(dest, target)
  }

  // Fly back to default overview on reset
  if (resetKey !== lastResetKey.current) {
    lastResetKey.current = resetKey
    beginFlight(DEFAULT_CAMERA_POS.clone(), DEFAULT_TARGET.clone())
  }

  useFrame((_, delta) => {
    if (progress.current < 1) {
      progress.current = Math.min(progress.current + delta * FLY_TO.speed, 1)

      // Smooth ease-out: cubic deceleration with guaranteed exact arrival.
      // Using a single continuous easing curve avoids the velocity discontinuity
      // that caused end-of-flight jitter with the previous two-phase approach.
      const p = progress.current
      const t = 1 - Math.pow(1 - p, FLY_TO.easePower)

      camera.position.lerpVectors(startPos.current, endPos.current, t)

      if (controlsRef.current) {
        controlsRef.current.target.lerpVectors(startTarget.current, targetVec.current, t)
      }

      camera.lookAt(
        controlsRef.current
          ? controlsRef.current.target
          : targetVec.current,
      )

      if (progress.current >= 1) {
        // Snap to exact final position to avoid floating-point drift
        camera.position.copy(endPos.current)
        if (controlsRef.current) {
          controlsRef.current.target.copy(targetVec.current)
          controlsRef.current.enabled = true
        }
      }
    }
  })

  return null
}
