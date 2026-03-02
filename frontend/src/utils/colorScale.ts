// frontend/src/utils/colorScale.ts
import * as THREE from 'three'

const CLUSTER_COLORS = [
  new THREE.Color('#06b6d4'), // cyan
  new THREE.Color('#d946ef'), // magenta
  new THREE.Color('#22c55e'), // green
  new THREE.Color('#f59e0b'), // amber
  new THREE.Color('#f43f5e'), // rose
  new THREE.Color('#6366f1'), // indigo
  new THREE.Color('#a78bfa'), // violet
  new THREE.Color('#14b8a6'), // teal
  new THREE.Color('#f97316'), // orange
  new THREE.Color('#ec4899'), // pink
]

const NOISE_COLOR = new THREE.Color('#374151')

export function getClusterColor(label: number): THREE.Color {
  if (label < 0) return NOISE_COLOR
  return CLUSTER_COLORS[label % CLUSTER_COLORS.length]
}

/**
 * Map a local dimension value to a cool→hot color.
 * Uses a 5-stop gradient: blue → cyan → green → yellow → red
 */
export function getLocalDimColor(
  value: number,
  min: number,
  max: number,
): THREE.Color {
  const t = max > min ? (value - min) / (max - min) : 0.5

  const stops = [
    new THREE.Color('#3b82f6'), // blue  (low dim)
    new THREE.Color('#06b6d4'), // cyan
    new THREE.Color('#22c55e'), // green
    new THREE.Color('#eab308'), // yellow
    new THREE.Color('#ef4444'), // red   (high dim)
  ]

  const idx = t * (stops.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, stops.length - 1)
  const frac = idx - lo
  return stops[lo].clone().lerp(stops[hi], frac)
}
