// frontend/src/three/PointCloudMesh.tsx
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { getClusterColor, getLocalDimColor } from '../utils/colorScale'
import type { ClusterData, ColorMode } from '../types/feature'
import { OPACITY, SIZE, CIRCUIT, UNCATEGORIZED, CAMERA, POINTS, DOF } from '../config/rendering'

import vertexShader from '../shaders/point.vert.glsl?raw'
import fragmentShader from '../shaders/point.frag.glsl?raw'

/** Indices of features belonging to "famous" circuits with published papers */
const FAMOUS_CIRCUIT_IDS = new Set([
  'coact-capital-of-france',
  'coact-cat-sat-on',
  'coact-moon-landing',
  'coact-once-upon-a-time',
  'coact-fibonacci-code',
])

interface Props {
  positions: Float32Array
  clusterLabels: Int32Array
  clusters: ClusterData[]
  localDimensions?: Float32Array
  colorMode: ColorMode
  isolateUncategorized: boolean
  selectedClusters: Set<number>
  selectedIndex: number | null
  circuitNodeIndices?: Set<number>
  circuitMembership?: Map<number, string[]>
}

/** Find the nearest cluster centroid for a point. */
function nearestCluster(
  px: number, py: number, pz: number,
  clusters: ClusterData[],
): number {
  let bestId = 0
  let bestDist = Infinity
  for (const c of clusters) {
    if (c.id < 0) continue
    const dx = px - c.centroid[0]
    const dy = py - c.centroid[1]
    const dz = pz - c.centroid[2]
    const d = dx * dx + dy * dy + dz * dz
    if (d < bestDist) {
      bestDist = d
      bestId = c.id
    }
  }
  return bestId
}

export default function PointCloudMesh({
  positions, clusterLabels, clusters, localDimensions, colorMode,
  isolateUncategorized, selectedClusters, selectedIndex, circuitNodeIndices, circuitMembership,
}: Props) {
  const pointsRef = useRef<THREE.Points>(null)
  const numPoints = positions.length / 3

  // Deterministic per-point variation (Knuth hash → stable 0–1)
  const variation = useMemo(() => {
    const v = new Float32Array(numPoints)
    for (let i = 0; i < numPoints; i++) {
      v[i] = ((i * 2654435761) >>> 0) / 4294967296
    }
    return v
  }, [numPoints])

  const { geometry, material } = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('aSize', new THREE.Float32BufferAttribute(new Float32Array(numPoints), 1))
    geo.setAttribute('aColor', new THREE.Float32BufferAttribute(new Float32Array(numPoints * 3), 3))
    geo.setAttribute('aOpacity', new THREE.Float32BufferAttribute(new Float32Array(numPoints), 1))

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
  }, [positions, numPoints])

  // In-place attribute updates — all values from config/rendering.ts
  useMemo(() => {
    const sizeArr = (geometry.getAttribute('aSize') as THREE.Float32BufferAttribute).array as Float32Array
    const colorArr = (geometry.getAttribute('aColor') as THREE.Float32BufferAttribute).array as Float32Array
    const opacityArr = (geometry.getAttribute('aOpacity') as THREE.Float32BufferAttribute).array as Float32Array

    let dimMin = 0, dimMax = 1
    if (localDimensions && localDimensions.length > 0) {
      dimMin = localDimensions.reduce((a, b) => Math.min(a, b), Infinity)
      dimMax = localDimensions.reduce((a, b) => Math.max(a, b), -Infinity)
    }

    const hasClusterSelection = selectedClusters.size > 0
    const hasCircuit = circuitNodeIndices && circuitNodeIndices.size > 0
    const hasPointSelection = selectedIndex !== null
    const selectedPointCluster = hasPointSelection ? clusterLabels[selectedIndex] : -999

    for (let i = 0; i < numPoints; i++) {
      const label = clusterLabels[i]
      const isUncategorized = label < 0
      const isInSelectedCluster = hasClusterSelection && selectedClusters.has(label)
      const isCircuitMember = hasCircuit && circuitNodeIndices.has(i)
      let color: THREE.Color

      if (colorMode === 'localDim' && localDimensions) {
        color = getLocalDimColor(localDimensions[i], dimMin, dimMax)
        // Same selection hierarchy as cluster mode — local dim only changes COLOR, not interaction
        if (isolateUncategorized) {
          sizeArr[i] = isUncategorized ? SIZE.isolatedUncat : SIZE.deselected
          opacityArr[i] = isUncategorized ? OPACITY.isolatedUncat : OPACITY.isolatedOther
        } else if (hasClusterSelection) {
          if (i === selectedIndex) {
            sizeArr[i] = SIZE.selected
            opacityArr[i] = OPACITY.selected
          } else if (isInSelectedCluster) {
            sizeArr[i] = SIZE.localDimActive
            opacityArr[i] = OPACITY.clusterActive + variation[i] * OPACITY.clusterVariation
          } else {
            sizeArr[i] = SIZE.deselected
            opacityArr[i] = OPACITY.deselected
          }
        } else if (hasPointSelection) {
          const isSelected = i === selectedIndex
          const isSameCluster = label >= 0 && label === selectedPointCluster
          if (isSelected) {
            sizeArr[i] = SIZE.selected
            opacityArr[i] = OPACITY.selected
          } else if (isSameCluster) {
            sizeArr[i] = SIZE.flyContext
            opacityArr[i] = OPACITY.flyContext + variation[i] * OPACITY.flyContextVariation
          } else {
            sizeArr[i] = isUncategorized ? SIZE.flyBackgroundUncat : SIZE.flyBackgroundCat
            opacityArr[i] = isUncategorized
              ? OPACITY.flyBackgroundUncat
              : OPACITY.flyBackgroundCat + variation[i] * OPACITY.flyBackgroundCatVariation
          }
        } else {
          sizeArr[i] = isUncategorized ? SIZE.defaultUncategorized : SIZE.localDimDefault
          opacityArr[i] = isUncategorized
            ? OPACITY.defaultUncategorized
            : OPACITY.defaultCategorized + variation[i] * OPACITY.defaultVariation
        }
      } else {
        // Cluster color mode
        if (isUncategorized) {
          const nearest = nearestCluster(
            positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
            clusters,
          )
          color = getClusterColor(nearest).clone().multiplyScalar(UNCATEGORIZED.colorDimFactor)
        } else {
          color = getClusterColor(label)
        }

        if (isolateUncategorized) {
          sizeArr[i] = isUncategorized ? SIZE.isolatedUncat : SIZE.deselected
          opacityArr[i] = isUncategorized ? OPACITY.isolatedUncat : OPACITY.isolatedOther
        } else if (hasClusterSelection) {
          if (i === selectedIndex) {
            sizeArr[i] = SIZE.selected
            opacityArr[i] = OPACITY.selected
          } else if (isInSelectedCluster) {
            sizeArr[i] = SIZE.clusterActive
            opacityArr[i] = OPACITY.clusterActive + variation[i] * OPACITY.clusterVariation
          } else {
            sizeArr[i] = SIZE.deselected
            opacityArr[i] = OPACITY.deselected
          }
        } else if (hasPointSelection) {
          const isSelected = i === selectedIndex
          const isSameCluster = label >= 0 && label === selectedPointCluster
          if (isSelected) {
            sizeArr[i] = SIZE.selected
            opacityArr[i] = OPACITY.selected
          } else if (isSameCluster) {
            sizeArr[i] = SIZE.flyContext
            opacityArr[i] = OPACITY.flyContext + variation[i] * OPACITY.flyContextVariation
          } else {
            sizeArr[i] = isUncategorized ? SIZE.flyBackgroundUncat : SIZE.flyBackgroundCat
            opacityArr[i] = isUncategorized
              ? OPACITY.flyBackgroundUncat
              : OPACITY.flyBackgroundCat + variation[i] * OPACITY.flyBackgroundCatVariation
          }
        } else {
          sizeArr[i] = isUncategorized ? SIZE.defaultUncategorized : SIZE.defaultCategorized
          opacityArr[i] = isUncategorized
            ? OPACITY.defaultUncategorized
            : OPACITY.defaultCategorized + variation[i] * OPACITY.defaultVariation
        }
      }

      // Circuit members: boosted size/opacity/glow
      if (isCircuitMember) {
        const isFamous = circuitMembership
          ? (circuitMembership.get(i) ?? []).some((cid) => FAMOUS_CIRCUIT_IDS.has(cid))
          : false
        const cfg = isFamous
          ? { mult: CIRCUIT.famousSizeMultiplier, min: CIRCUIT.famousSizeMin, op: CIRCUIT.famousOpacity, white: CIRCUIT.famousWhiteMix }
          : { mult: CIRCUIT.regularSizeMultiplier, min: CIRCUIT.regularSizeMin, op: CIRCUIT.regularOpacity, white: CIRCUIT.regularWhiteMix }

        sizeArr[i] = Math.max(sizeArr[i] * cfg.mult, cfg.min)
        opacityArr[i] = Math.max(opacityArr[i], cfg.op)
        const w = cfg.white
        colorArr[i * 3] = color.r * (1 - w) + w
        colorArr[i * 3 + 1] = color.g * (1 - w) + w
        colorArr[i * 3 + 2] = color.b * (1 - w) + w
      } else {
        colorArr[i * 3] = color.r
        colorArr[i * 3 + 1] = color.g
        colorArr[i * 3 + 2] = color.b
      }
    }

    geometry.getAttribute('aSize').needsUpdate = true
    geometry.getAttribute('aColor').needsUpdate = true
    geometry.getAttribute('aOpacity').needsUpdate = true
  }, [geometry, variation, positions, clusterLabels, clusters, localDimensions, colorMode, isolateUncategorized, selectedClusters, selectedIndex, circuitNodeIndices, circuitMembership, numPoints])

  // Update DOF focus distance every frame — see CLAUDE.md "DOF Bug Pattern"
  useFrame(({ camera }) => {
    if (material.uniforms.uFocusDist) {
      material.uniforms.uFocusDist.value = camera.position.length()
    }
  })

  return <points ref={pointsRef} geometry={geometry} material={material} />
}
