// frontend/src/three/CircuitEdges.tsx
// Renders attribution edges as glowing line segments between circuit nodes.
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { CircuitEdge, CircuitNode } from '../types/circuit'
import { CIRCUIT_EDGES } from '../config/rendering'

interface Props {
  edges: CircuitEdge[]
  nodes: CircuitNode[]
  allPositions: Float32Array
  numFeatures: number
  threshold: number
}

const edgeVertexShader = /* glsl */ `
  attribute float aOpacity;
  varying float vOpacity;

  void main() {
    vOpacity = aOpacity;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const edgeFragmentShader = /* glsl */ `
  varying float vOpacity;

  void main() {
    if (vOpacity < ${CIRCUIT_EDGES.discardThreshold}) discard;
    // Warm white-cyan glow for attribution flow
    vec3 color = mix(vec3(0.4, 0.7, 0.8), vec3(0.9, 0.95, 1.0), vOpacity);
    gl_FragColor = vec4(color, vOpacity);
  }
`

export default function CircuitEdges({ edges, nodes, allPositions, numFeatures, threshold }: Props) {
  const ref = useRef<THREE.LineSegments>(null)

  // Valid edges: endpoints exist in node set and within feature range
  const { validEdges } = useMemo(() => {
    const ns = new Set(nodes.map((n) => n.featureIndex))
    const ve = edges.filter(
      (e) =>
        ns.has(e.source) &&
        ns.has(e.target) &&
        e.source < numFeatures &&
        e.target < numFeatures,
    )
    return { validEdges: ve }
  }, [edges, nodes, numFeatures])

  // Geometry + material: allocated once per mount, sized to all valid edges
  const { geometry, material } = useMemo(() => {
    const count = validEdges.length

    const geo = new THREE.BufferGeometry()
    // 2 vertices per edge, 3 components per vertex
    geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(count * 6), 3))
    // 1 opacity per vertex (2 per edge)
    geo.setAttribute('aOpacity', new THREE.Float32BufferAttribute(new Float32Array(count * 2), 1))

    const mat = new THREE.ShaderMaterial({
      vertexShader: edgeVertexShader,
      fragmentShader: edgeFragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })

    return { geometry: geo, material: mat }
  }, [validEdges])

  // Update positions and opacity in-place — useEffect guarantees re-run on threshold change
  useEffect(() => {
    const count = validEdges.length
    const posArr = (geometry.getAttribute('position') as THREE.Float32BufferAttribute).array as Float32Array
    const opacityArr = (geometry.getAttribute('aOpacity') as THREE.Float32BufferAttribute).array as Float32Array

    for (let i = 0; i < count; i++) {
      const edge = validEdges[i]
      const si = edge.source * 3
      const ti = edge.target * 3

      // Source vertex
      posArr[i * 6] = allPositions[si]
      posArr[i * 6 + 1] = allPositions[si + 1]
      posArr[i * 6 + 2] = allPositions[si + 2]
      // Target vertex
      posArr[i * 6 + 3] = allPositions[ti]
      posArr[i * 6 + 4] = allPositions[ti + 1]
      posArr[i * 6 + 5] = allPositions[ti + 2]

      // Edges below threshold get opacity 0 (discarded in fragment shader)
      const opacity = edge.weight >= threshold ? edge.weight * CIRCUIT_EDGES.opacityMultiplier : 0
      opacityArr[i * 2] = opacity
      opacityArr[i * 2 + 1] = opacity
    }

    geometry.getAttribute('position').needsUpdate = true
    geometry.getAttribute('aOpacity').needsUpdate = true
  }, [geometry, validEdges, allPositions, threshold])

  return <lineSegments ref={ref} geometry={geometry} material={material} />
}
