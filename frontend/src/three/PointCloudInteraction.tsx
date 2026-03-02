// frontend/src/three/PointCloudInteraction.tsx
import { useEffect, useRef } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useAppStore } from '../stores/useAppStore'
import { INTERACTION } from '../config/rendering'

interface Props {
  positions: Float32Array
}

export default function PointCloudInteraction({ positions }: Props) {
  const { camera, gl } = useThree()
  const raycaster = useRef(new THREE.Raycaster())
  const mouse = useRef(new THREE.Vector2())
  const setHovered = useAppStore((s) => s.setHovered)
  const setSelected = useAppStore((s) => s.setSelected)

  // Listen on the canvas DOM element directly — no invisible mesh needed
  useEffect(() => {
    const canvas = gl.domElement

    const handleMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect()
      mouse.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
    }

    const handleClick = () => {
      const hovered = useAppStore.getState().hoveredIndex
      // Only update selection when actually clicking a point.
      // Clicking empty space preserves the current selection.
      if (hovered !== null) {
        setSelected(hovered)
      }
    }

    canvas.addEventListener('pointermove', handleMove)
    canvas.addEventListener('click', handleClick)
    return () => {
      canvas.removeEventListener('pointermove', handleMove)
      canvas.removeEventListener('click', handleClick)
    }
  }, [gl, setSelected])

  useFrame(() => {
    raycaster.current.setFromCamera(mouse.current, camera)
    const rayOrigin = raycaster.current.ray.origin
    const rayDir = raycaster.current.ray.direction
    const numPoints = positions.length / 3

    // Scale threshold with camera distance so hover works at any zoom
    const cameraDist = camera.position.length()
    let closestIdx = -1
    let closestDist = INTERACTION.hoverThresholdScale * cameraDist

    for (let i = 0; i < numPoints; i++) {
      const px = positions[i * 3]
      const py = positions[i * 3 + 1]
      const pz = positions[i * 3 + 2]

      const dx = px - rayOrigin.x
      const dy = py - rayOrigin.y
      const dz = pz - rayOrigin.z
      const t = dx * rayDir.x + dy * rayDir.y + dz * rayDir.z
      if (t < 0) continue

      const cx = rayOrigin.x + t * rayDir.x - px
      const cy = rayOrigin.y + t * rayDir.y - py
      const cz = rayOrigin.z + t * rayDir.z - pz
      const dist = Math.sqrt(cx * cx + cy * cy + cz * cz)

      if (dist < closestDist) {
        closestDist = dist
        closestIdx = i
      }
    }

    // Guard against redundant state updates
    const newHovered = closestIdx >= 0 ? closestIdx : null
    if (newHovered !== useAppStore.getState().hoveredIndex) {
      setHovered(newHovered)
    }
  })

  return null
}
