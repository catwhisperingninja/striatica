// frontend/src/three/CircuitNodes.tsx
// Renders circuit nodes as bright highlighted points using the same shader.
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { CircuitNode } from '../types/circuit'
import { CIRCUIT_NODES, TRACED_CIRCUITS, CAMERA, POINTS, DOF } from '../config/rendering'
import { buildRoleColorMap, LEGACY_ROLES } from '../utils/circuitRoles'

import vertexShader from '../shaders/point.vert.glsl?raw'
import fragmentShader from '../shaders/point.frag.glsl?raw'

interface Props {
  nodes: CircuitNode[]
  allPositions: Float32Array
  numFeatures: number
  selectedIndex: number | null
}

type LegacyRole = (typeof LEGACY_ROLES)[number]
const LEGACY_ROLE_SET: ReadonlySet<string> = new Set(LEGACY_ROLES)

const SELECTED_COLOR = new THREE.Color('#ffffff')
const FALLBACK_COLOR = new THREE.Color(TRACED_CIRCUITS.unassignedColor3D)

export default function CircuitNodes({ nodes, allPositions, numFeatures, selectedIndex }: Props) {
  const ref = useRef<THREE.Points>(null)

  const validNodes = useMemo(
    () => nodes.filter((n) => n.featureIndex < numFeatures),
    [nodes, numFeatures],
  )

  // Role → color for the 3D (additive) space. Legacy roles keep their exact
  // COLORS.roles3D values; new traced-circuit roles get palette colors.
  const colorMap = useMemo(() => {
    const hexMap = buildRoleColorMap(validNodes, '3d')
    const m = new Map<string, THREE.Color>()
    for (const [role, hex] of hexMap) m.set(role, new THREE.Color(hex))
    return m
  }, [validNodes])

  // Geometry + material: allocated once per mount, sized to validNodes
  const { geometry, material } = useMemo(() => {
    const count = validNodes.length

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(count * 3), 3))
    geo.setAttribute('aSize', new THREE.Float32BufferAttribute(new Float32Array(count), 1))
    geo.setAttribute('aColor', new THREE.Float32BufferAttribute(new Float32Array(count * 3), 3))
    geo.setAttribute('aOpacity', new THREE.Float32BufferAttribute(new Float32Array(count), 1))

    const mat = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uFocusDist: { value: CAMERA.defaultPosition[2] },
        uSizeScale: { value: POINTS.sizeScale },
        uSizeMin: { value: POINTS.sizeMin },
        uSizeMax: { value: POINTS.sizeMax },
        uSoftRange: { value: DOF.softRange },
        uMinFocusDist: { value: DOF.minFocusDist },
        uCrispEdge: { value: DOF.crispEdge },
        uSoftEdge: { value: DOF.softEdge },
        uDepthFadeFactor: { value: DOF.depthFadeFactor },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })

    return { geometry: geo, material: mat }
  }, [validNodes])

  // Update ALL attributes in-place — useEffect guarantees re-run on every dep change
  useEffect(() => {
    const count = validNodes.length
    const posArr = (geometry.getAttribute('position') as THREE.Float32BufferAttribute).array as Float32Array
    const sizeArr = (geometry.getAttribute('aSize') as THREE.Float32BufferAttribute).array as Float32Array
    const colorArr = (geometry.getAttribute('aColor') as THREE.Float32BufferAttribute).array as Float32Array
    const opacityArr = (geometry.getAttribute('aOpacity') as THREE.Float32BufferAttribute).array as Float32Array

    for (let i = 0; i < count; i++) {
      const node = validNodes[i]
      const fi = node.featureIndex * 3
      const isSelected = selectedIndex === node.featureIndex
      const role = node.role

      // Position
      posArr[i * 3] = allPositions[fi]
      posArr[i * 3 + 1] = allPositions[fi + 1]
      posArr[i * 3 + 2] = allPositions[fi + 2]

      // Size/opacity — legacy roles use their CIRCUIT_NODES config; any
      // non-legacy (traced) role uses the shared TRACED_CIRCUITS.node sizing.
      const roleCfg = LEGACY_ROLE_SET.has(role)
        ? CIRCUIT_NODES[role as LegacyRole]
        : TRACED_CIRCUITS.node
      sizeArr[i] = (roleCfg.baseSize + node.activation * roleCfg.sizeScale) * (isSelected ? CIRCUIT_NODES.selectedSizeMultiplier : 1.0)

      // Color
      const color = isSelected ? SELECTED_COLOR : (colorMap.get(role) ?? FALLBACK_COLOR)
      colorArr[i * 3] = color.r
      colorArr[i * 3 + 1] = color.g
      colorArr[i * 3 + 2] = color.b

      // Opacity
      opacityArr[i] = isSelected ? CIRCUIT_NODES.selectedOpacity : roleCfg.minOpacity + node.activation * (1.0 - roleCfg.minOpacity)
    }

    geometry.getAttribute('position').needsUpdate = true
    geometry.getAttribute('aSize').needsUpdate = true
    geometry.getAttribute('aColor').needsUpdate = true
    geometry.getAttribute('aOpacity').needsUpdate = true
  }, [geometry, validNodes, allPositions, selectedIndex, colorMap])

  return <points ref={ref} geometry={geometry} material={material} />
}
