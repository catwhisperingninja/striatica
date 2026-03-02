// frontend/src/components/StatusBar.tsx
import { useAppStore } from '../stores/useAppStore'

export default function StatusBar() {
  const dataset = useAppStore((s) => s.dataset)
  const loading = useAppStore((s) => s.loading)
  const error = useAppStore((s) => s.error)
  const hoveredIndex = useAppStore((s) => s.hoveredIndex)
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const viewMode = useAppStore((s) => s.viewMode)
  const colorMode = useAppStore((s) => s.colorMode)
  const dimMethod = useAppStore((s) => s.dataset?.dimMethod)
  const circuitData = useAppStore((s) => s.circuitData)
  const edgeThreshold = useAppStore((s) => s.edgeThreshold)

  const hoveredLabel = (() => {
    if (hoveredIndex === null || !dataset) return null
    const feat = dataset.features[hoveredIndex]
    return feat ? `Feature #${hoveredIndex} "${feat.explanation}"` : `Feature #${hoveredIndex}`
  })()

  return (
    <div className="flex items-center gap-4 px-4 py-1 bg-[--color-panel] border-t border-[--color-panel-border] text-[10px] text-gray-600">
      <div className="flex items-center gap-1">
        <div className={`w-[5px] h-[5px] rounded-full ${error ? 'bg-red-500' : loading ? 'bg-yellow-500' : 'bg-green-500'}`} />
        {error ? 'Error' : loading ? 'Loading...' : 'Ready'}
      </div>
      <span>Points: {dataset?.numFeatures.toLocaleString() ?? '0'}</span>
      {viewMode === 'circuits' && circuitData && (
        <span className="text-cyan-700">
          Circuit: {circuitData.name} ({circuitData.nodes.length} nodes, {circuitData.edges.filter((e) => e.weight >= edgeThreshold).length} edges)
        </span>
      )}
      {colorMode === 'localDim' && (
        <span className="text-zinc-600">
          * Local dimension ({dimMethod ?? 'pr'}). Methodology is an active area of research.
        </span>
      )}
      <span className="ml-auto flex items-center gap-3">
        {selectedIndex !== null && (
          <span className="text-gray-500">
            Selected: #{selectedIndex}
          </span>
        )}
        {hoveredLabel && (
          <span className="text-[--color-cluster-0]">
            Hovered: {hoveredLabel}
          </span>
        )}
      </span>
    </div>
  )
}
