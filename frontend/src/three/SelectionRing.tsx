// frontend/src/three/SelectionRing.tsx
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { SELECTION_RING } from '../config/rendering'

interface Props {
  position: [number, number, number]
}

/** Thin white ring highlighting the selected point. Scales with camera distance. */
export default function SelectionRing({ position }: Props) {
  const ref = useRef<THREE.Mesh>(null)

  useFrame(({ camera }) => {
    if (ref.current) {
      // Billboard: face the camera
      ref.current.quaternion.copy(camera.quaternion)
      // Scale with distance so the ring is always a consistent visual size
      const dist = camera.position.distanceTo(ref.current.position)
      const scale = dist * SELECTION_RING.scaleFromDistance
      ref.current.scale.setScalar(scale)
    }
  })

  return (
    <mesh ref={ref} position={position}>
      {/* Unit-size ring, scaled dynamically by useFrame above */}
      <ringGeometry args={[SELECTION_RING.innerRadius, SELECTION_RING.outerRadius, SELECTION_RING.segments]} />
      <meshBasicMaterial
        color={SELECTION_RING.color}
        transparent
        opacity={SELECTION_RING.opacity}
        side={THREE.DoubleSide}
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  )
}
