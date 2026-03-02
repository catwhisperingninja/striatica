// frontend/src/views/CircuitGraphView.tsx
import { useMemo } from 'react'
import { useAppStore } from '../stores/useAppStore'
import CircuitBackground from '../three/CircuitBackground'
import CircuitNodes from '../three/CircuitNodes'
import CircuitEdges from '../three/CircuitEdges'
import SelectionRing from '../three/SelectionRing'

export default function CircuitGraphView() {
  const dataset = useAppStore((s) => s.dataset)
  const circuitData = useAppStore((s) => s.circuitData)
  const edgeThreshold = useAppStore((s) => s.edgeThreshold)
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const selectedClusters = useAppStore((s) => s.selectedClusters)
  const colorMode = useAppStore((s) => s.colorMode)

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

  // Selected point position for the highlight ring (same logic as PointCloudView)
  const selectedPos = useMemo((): [number, number, number] | null => {
    if (selectedIndex === null || !dataset) return null
    const i = selectedIndex * 3
    if (i + 2 >= dataset.positions.length) return null
    return [dataset.positions[i], dataset.positions[i + 1], dataset.positions[i + 2]]
  }, [selectedIndex, dataset])

  if (!dataset) return null

  return (
    <>
      <CircuitBackground
        positions={positions}
        clusterLabels={clusterLabels}
        selectedIndex={selectedIndex}
        selectedClusters={selectedClusters}
        colorMode={colorMode}
        localDimensions={localDimensions}
      />
      {circuitData && (
        <group key={circuitData.name}>
          <CircuitNodes
            nodes={circuitData.nodes}
            allPositions={positions}
            numFeatures={dataset.numFeatures}
            selectedIndex={selectedIndex}
          />
          <CircuitEdges
            edges={circuitData.edges}
            nodes={circuitData.nodes}
            allPositions={positions}
            numFeatures={dataset.numFeatures}
            threshold={edgeThreshold}
          />
        </group>
      )}
      {/* Show selection ring even when no circuit is loaded.
          Selection is locked in circuits view — switch to Points (⌘P) to change it. */}
      {selectedPos && <SelectionRing position={selectedPos} />}
    </>
  )
}
