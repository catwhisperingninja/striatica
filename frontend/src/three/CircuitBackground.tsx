// frontend/src/three/CircuitBackground.tsx
// Renders all points as spatial context for the circuit view.
// When no circuit is loaded, highlights the selected point and selected clusters
// so selection carries over visually from Point Cloud view.
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { getClusterColor, getLocalDimColor } from '../utils/colorScale'
import { CIRCUIT_BG, CAMERA, POINTS, DOF } from '../config/rendering'
import type { ColorMode } from '../types/feature'

import vertexShader from '../shaders/point.vert.glsl?raw'
import fragmentShader from '../shaders/point.frag.glsl?raw'

interface Props {
  positions: Float32Array
  clusterLabels: Int32Array
  selectedIndex?: number | null
  selectedClusters?: Set<number>
  colorMode?: ColorMode
  localDimensions?: Float32Array
}

export default function CircuitBackground({
  positions, clusterLabels, selectedIndex, selectedClusters, colorMode, localDimensions,
}: Props) {
  const ref = useRef<THREE.Points>(null)
  const numPoints = positions.length / 3

  // Allocate geometry and material once
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

  // Update attributes in-place whenever selection state changes
  useEffect(() => {
    const sizeArr = (geometry.getAttribute('aSize') as THREE.Float32BufferAttribute).array as Float32Array
    const colorArr = (geometry.getAttribute('aColor') as THREE.Float32BufferAttribute).array as Float32Array
    const opacityArr = (geometry.getAttribute('aOpacity') as THREE.Float32BufferAttribute).array as Float32Array

    const hasClusterSelection = selectedClusters ? selectedClusters.size > 0 : false
    const useLocalDim = colorMode === 'localDim' && localDimensions && localDimensions.length > 0
    let dimMin = 0, dimMax = 1
    if (useLocalDim) {
      dimMin = localDimensions!.reduce((a, b) => Math.min(a, b), Infinity)
      dimMax = localDimensions!.reduce((a, b) => Math.max(a, b), -Infinity)
    }

    for (let i = 0; i < numPoints; i++) {
      const label = clusterLabels[i]
      const color = useLocalDim
        ? getLocalDimColor(localDimensions![i], dimMin, dimMax)
        : getClusterColor(label)
      colorArr[i * 3] = color.r
      colorArr[i * 3 + 1] = color.g
      colorArr[i * 3 + 2] = color.b

      const isUncategorized = label < 0
      const isSelectedPoint = selectedIndex === i
      const isInSelectedCluster = hasClusterSelection && selectedClusters!.has(label)

      if (isSelectedPoint) {
        // Bright highlight for the specifically selected point
        opacityArr[i] = CIRCUIT_BG.selectedPointOpacity
        sizeArr[i] = CIRCUIT_BG.selectedPointSize
      } else if (isInSelectedCluster) {
        // Medium brightness for points in the selected cluster
        opacityArr[i] = CIRCUIT_BG.selectedClusterOpacity
        sizeArr[i] = CIRCUIT_BG.selectedClusterSize
      } else {
        // Default dim background
        opacityArr[i] = isUncategorized ? CIRCUIT_BG.bgOpacityUncat : CIRCUIT_BG.bgOpacity
        sizeArr[i] = CIRCUIT_BG.bgSize
      }
    }

    geometry.getAttribute('aSize').needsUpdate = true
    geometry.getAttribute('aColor').needsUpdate = true
    geometry.getAttribute('aOpacity').needsUpdate = true
  }, [geometry, clusterLabels, numPoints, selectedIndex, selectedClusters, colorMode, localDimensions])

  return <points ref={ref} geometry={geometry} material={material} />
}
