// frontend/src/components/CircuitPanel.tsx
// Controls panel for circuit view: threshold slider, node list, circuit info.
import { useEffect, useRef } from 'react'
import { useAppStore } from '../stores/useAppStore'
import type { CircuitManifestEntry } from '../types/circuit'
import { COLORS } from '../config/rendering'

const ROLE_LABELS: Record<string, string> = {
  source: 'Source',
  intermediate: 'Processing',
  sink: 'Output',
}

export default function CircuitPanel() {
  const dataset = useAppStore((s) => s.dataset)
  const circuitData = useAppStore((s) => s.circuitData)
  const circuitManifest = useAppStore((s) => s.circuitManifest)
  const loadCircuitById = useAppStore((s) => s.loadCircuitById)
  const setCircuitData = useAppStore((s) => s.setCircuitData)
  const edgeThreshold = useAppStore((s) => s.edgeThreshold)
  const setEdgeThreshold = useAppStore((s) => s.setEdgeThreshold)
  const setSelected = useAppStore((s) => s.setSelected)
  const setFlyTarget = useAppStore((s) => s.setFlyTarget)
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const getCircuitsForFeature = useAppStore((s) => s.getCircuitsForFeature)

  // Auto-load the first matching circuit when entering circuit view
  // with a feature selected that belongs to a circuit
  const autoLoadedRef = useRef<number | null>(null)
  useEffect(() => {
    if (circuitData) return // already loaded
    if (selectedIndex === null) return
    if (autoLoadedRef.current === selectedIndex) return // don't re-trigger for same feature
    const circuits = getCircuitsForFeature(selectedIndex)
    if (circuits.length > 0) {
      autoLoadedRef.current = selectedIndex
      loadCircuitById(circuits[0])
    }
  }, [selectedIndex, circuitData, getCircuitsForFeature, loadCircuitById])

  // Circuit selector when no circuit is loaded
  if (!circuitData) {
    const coact = circuitManifest?.circuits.filter((c) => c.type === 'coactivation') ?? []
    const sim = circuitManifest?.circuits.filter((c) => c.type === 'similarity') ?? []
    const selectedFeature = dataset && selectedIndex !== null
      ? dataset.features[selectedIndex]
      : null
    const selectedClusterLabel = dataset && selectedIndex !== null
      ? dataset.clusterLabels[selectedIndex]
      : null

    return (
      <div className="w-[220px] shrink-0 bg-[--color-panel] border-r border-[--color-panel-border] backdrop-blur-xl p-3 overflow-y-auto">
        {/* Show carried-over selection from Point Cloud view */}
        {selectedFeature && (() => {
          const featureCircuits = getCircuitsForFeature(selectedFeature.index)
          return (
            <div className="mb-3 pb-2 border-b border-gray-800">
              <div className="text-[9px] text-gray-600 uppercase tracking-wide mb-1">Selected Feature</div>
              <div className="text-[11px] text-gray-200 font-medium">
                #{selectedFeature.index}
              </div>
              <div className="text-[10px] text-gray-400 leading-relaxed mt-0.5">
                {selectedFeature.explanation ?? '(no explanation)'}
              </div>
              {selectedClusterLabel !== null && selectedClusterLabel >= 0 && (
                <div className="text-[9px] text-gray-500 mt-1">
                  Cluster {selectedClusterLabel}
                </div>
              )}
              {featureCircuits.length > 0 ? (
                <div className="mt-1.5 text-[10px] text-cyan-400">
                  In {featureCircuits.length} circuit{featureCircuits.length > 1 ? 's' : ''}: {featureCircuits.join(', ')}
                </div>
              ) : (
                <div className="mt-1.5 text-[10px] text-amber-500/70">
                  Not in any circuit — select a glowing point to explore circuits
                </div>
              )}
            </div>
          )
        })()}

        <div className="text-xs font-bold text-gray-300 mb-3">Select Circuit</div>
        {!circuitManifest ? (
          <p className="text-[10px] text-gray-600">Loading manifest...</p>
        ) : (
          <>
            {coact.length > 0 && (
              <CircuitGroup label="Co-activation" entries={coact} onSelect={loadCircuitById} />
            )}
            {sim.length > 0 && (
              <CircuitGroup label="Similarity" entries={sim} onSelect={loadCircuitById} />
            )}
          </>
        )}
      </div>
    )
  }

  const visibleEdges = circuitData.edges.filter((e) => e.weight >= edgeThreshold).length

  // Group nodes by role
  const byRole = { source: [] as typeof circuitData.nodes, intermediate: [] as typeof circuitData.nodes, sink: [] as typeof circuitData.nodes }
  for (const node of circuitData.nodes) {
    ;(byRole[node.role] ?? byRole.intermediate).push(node)
  }

  const handleNodeClick = (featureIndex: number) => {
    setSelected(featureIndex)
    if (dataset) {
      const i = featureIndex * 3
      if (i + 2 < dataset.positions.length) {
        setFlyTarget([dataset.positions[i], dataset.positions[i + 1], dataset.positions[i + 2]])
      }
    }
  }

  return (
    <div className="w-[220px] shrink-0 bg-[--color-panel] border-r border-[--color-panel-border] backdrop-blur-xl p-3 overflow-y-auto">
      {/* Back button + Circuit name */}
      <div className="flex items-center gap-1.5 mb-1">
        <button
          onClick={() => setCircuitData(null)}
          className="text-[10px] text-gray-500 hover:text-gray-300 cursor-pointer"
          title="Back to circuit selector"
        >
          &larr;
        </button>
        <div className="text-xs font-bold text-gray-300 truncate">
          {circuitData.name}
        </div>
      </div>
      {circuitData.description && (
        <div className="text-[10px] text-gray-600 mb-3 leading-relaxed">
          {circuitData.description}
        </div>
      )}

      {/* Threshold slider */}
      <div className="mb-3">
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>Edge threshold</span>
          <span className="font-mono">{edgeThreshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={edgeThreshold}
          onChange={(e) => setEdgeThreshold(parseFloat(e.target.value))}
          className="w-full h-1 appearance-none bg-gray-800 rounded-full accent-[--color-cluster-0] cursor-pointer"
        />
        <div className="text-[9px] text-gray-600 mt-0.5">
          {visibleEdges} / {circuitData.edges.length} edges visible
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-3 mb-3 text-[9px]">
        {(['source', 'intermediate', 'sink'] as const).map((role) => (
          <div key={role} className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: COLORS.roles[role] }} />
            <span className="text-gray-500">{ROLE_LABELS[role]}</span>
          </div>
        ))}
      </div>

      {/* Node list grouped by role */}
      {(['source', 'intermediate', 'sink'] as const).map((role) => {
        const nodes = byRole[role]
        if (nodes.length === 0) return null
        return (
          <div key={role} className="mb-2">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1" style={{ color: COLORS.roles[role] }}>
              {ROLE_LABELS[role]} ({nodes.length})
            </div>
            {nodes.map((node) => {
              const feat = dataset?.features[node.featureIndex]
              const isSelected = selectedIndex === node.featureIndex
              return (
                <div
                  key={node.featureIndex}
                  className={`text-[10px] py-0.5 px-1 cursor-pointer truncate transition-colors ${
                    isSelected
                      ? 'text-gray-100 bg-white/5 rounded'
                      : 'text-gray-500 hover:text-gray-200'
                  }`}
                  onClick={() => handleNodeClick(node.featureIndex)}
                >
                  <span className={isSelected ? 'font-mono' : 'text-gray-600 font-mono'}>
                    {node.activation.toFixed(2)}
                  </span>
                  {' '}
                  {feat?.explanation ?? `Feature #${node.featureIndex}`}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}

function CircuitGroup({ label, entries, onSelect }: {
  label: string
  entries: CircuitManifestEntry[]
  onSelect: (id: string) => void
}) {
  return (
    <div className="mb-3">
      <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        {label}
      </div>
      {entries.map((entry) => (
        <div
          key={entry.id}
          className="text-[10px] py-1 px-1.5 cursor-pointer rounded transition-colors text-gray-400 hover:text-gray-200 hover:bg-white/5"
          onClick={() => onSelect(entry.id)}
        >
          <div className="truncate">{entry.name}</div>
          <div className="text-[9px] text-gray-600">
            {entry.nodeCount} nodes, {entry.edgeCount} edges
          </div>
        </div>
      ))}
    </div>
  )
}
