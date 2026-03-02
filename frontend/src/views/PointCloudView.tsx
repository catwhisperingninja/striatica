// frontend/src/views/PointCloudView.tsx
import { useMemo } from 'react'
import { useAppStore } from '../stores/useAppStore'
import PointCloudMesh from '../three/PointCloudMesh'
import PointCloudInteraction from '../three/PointCloudInteraction'
import SelectionRing from '../three/SelectionRing'
import CircuitLabels from '../three/CircuitLabels'

export default function PointCloudView() {
  const dataset = useAppStore((s) => s.dataset)
  const colorMode = useAppStore((s) => s.colorMode)
  const isolateUncategorized = useAppStore((s) => s.isolateUncategorized)
  const selectedClusters = useAppStore((s) => s.selectedClusters)
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const circuitMembership = useAppStore((s) => s.circuitMembership)

  const { positions, clusterLabels, localDimensions } = useMemo(() => {
    if (!dataset) return {
      positions: new Float32Array(0),
      clusterLabels: new Int32Array(0),
      localDimensions: undefined,
    }
    return {
      positions: new Float32Array(dataset.positions),
      clusterLabels: new Int32Array(dataset.clusterLabels),
      localDimensions: dataset.localDimensions
        ? new Float32Array(dataset.localDimensions)
        : undefined,
    }
  }, [dataset])

  // Set of ALL features that belong to ANY circuit — always visible in point cloud
  const circuitNodeIndices = useMemo(() => {
    if (circuitMembership.size === 0) return undefined
    return new Set(circuitMembership.keys())
  }, [circuitMembership])

  // Selected point position for the highlight ring
  const selectedPos = useMemo((): [number, number, number] | null => {
    if (selectedIndex === null || !dataset) return null
    const i = selectedIndex * 3
    if (i + 2 >= dataset.positions.length) return null
    return [dataset.positions[i], dataset.positions[i + 1], dataset.positions[i + 2]]
  }, [selectedIndex, dataset])

  if (!dataset) return null

  return (
    <>
      <PointCloudMesh
        positions={positions}
        clusterLabels={clusterLabels}
        clusters={dataset.clusters}
        localDimensions={localDimensions}
        colorMode={colorMode}
        isolateUncategorized={isolateUncategorized}
        selectedClusters={selectedClusters}
        selectedIndex={selectedIndex}
        circuitNodeIndices={circuitNodeIndices}
        circuitMembership={circuitMembership}
      />
      <PointCloudInteraction positions={positions} />
      {selectedPos && <SelectionRing position={selectedPos} />}
      <CircuitLabels />
    </>
  )
}
