// frontend/src/components/CircuitPanel.tsx
// Controls panel for circuit view: threshold slider, node list, circuit info.
import { useEffect, useRef } from 'react'
import { useAppStore } from '../stores/useAppStore'
import type { CircuitManifestEntry, CircuitNode, CircuitType } from '../types/circuit'
import { buildRoleColorMap, orderedRoles, roleLabel } from '../utils/circuitRoles'
import { graphUrl } from '../utils/neuronpediaUrls'

// Circuit-type display order + labels. Unknown types fall through to their raw
// string so nothing ever silently vanishes from the selector.
const TYPE_ORDER: CircuitType[] = ['coactivation', 'similarity', 'traced']
const TYPE_LABELS: Record<string, string> = {
  coactivation: 'Co-activation',
  similarity: 'Similarity',
  traced: 'Attribution graphs (traced)',
}

const COPY = {
  loadingManifest: 'Loading manifest...',
  noCircuits: 'No circuits available for this dataset',
  tracedEdgeNote: 'Attribution edges span layers — not renderable in a single-layer atlas',
  viewGraph: 'View graph on Neuronpedia',
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
    const circuits = circuitManifest?.circuits ?? []
    const groups = new Map<string, CircuitManifestEntry[]>()
    for (const c of circuits) {
      const arr = groups.get(c.type) ?? []
      arr.push(c)
      groups.set(c.type, arr)
    }
    const orderedTypes = [
      ...TYPE_ORDER.filter((t) => groups.has(t)),
      ...[...groups.keys()].filter((t) => !TYPE_ORDER.includes(t as CircuitType)).sort(),
    ]

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
          <p className="text-[10px] text-gray-600">{COPY.loadingManifest}</p>
        ) : circuits.length === 0 ? (
          <p className="text-[10px] text-gray-600">{COPY.noCircuits}</p>
        ) : (
          <>
            {orderedTypes.map((t) => (
              <CircuitGroup
                key={t}
                label={TYPE_LABELS[t] ?? t}
                entries={groups.get(t) ?? []}
                onSelect={loadCircuitById}
              />
            ))}
          </>
        )}
      </div>
    )
  }

  // Traced attribution circuits arrive edge-free (their edges span layers and
  // aren't renderable in a single-layer atlas) — the edge slider is meaningless.
  const isTracedEdgeless = circuitData.type === 'traced' && circuitData.edges.length === 0
  const visibleEdges = circuitData.edges.filter((e) => e.weight >= edgeThreshold).length

  // Role → color + display order, generalized beyond the legacy taxonomy.
  const roleColorMap = buildRoleColorMap(circuitData.nodes, '2d')
  const roles = orderedRoles(circuitData.nodes)
  const byRole = new Map<string, CircuitNode[]>()
  for (const node of circuitData.nodes) {
    const arr = byRole.get(node.role) ?? []
    arr.push(node)
    byRole.set(node.role, arr)
  }

  // Cross-layer accounting for traced circuits (all fields optional).
  const crossLayer = circuitData.metadata?.crossLayerMembers
  const inLayer = circuitData.nodes.length
  const layerInfo = typeof crossLayer === 'number'
    ? `${inLayer} of ${inLayer + crossLayer} graph nodes are in this layer; ${crossLayer} cross-layer members recorded`
    : null

  // Neuronpedia attribution-graph slug (traced circuits only).
  const graphSlug = circuitData.metadata?.slug

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

      {/* Attribution graph link (traced circuits carry a Neuronpedia slug) */}
      {graphSlug && dataset && (
        <a
          href={graphUrl(dataset.model, graphSlug)}
          target="_blank"
          rel="noreferrer"
          className="block text-[10px] text-cyan-400 hover:text-cyan-300 mb-3"
        >
          {COPY.viewGraph} &#8599;
        </a>
      )}

      {/* Threshold slider (disabled for edge-free traced circuits) */}
      <div className="mb-3">
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>Edge threshold</span>
          {!isTracedEdgeless && <span className="font-mono">{edgeThreshold.toFixed(2)}</span>}
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={edgeThreshold}
          disabled={isTracedEdgeless}
          onChange={(e) => setEdgeThreshold(parseFloat(e.target.value))}
          className={`w-full h-1 appearance-none bg-gray-800 rounded-full accent-[--color-cluster-0] ${
            isTracedEdgeless ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'
          }`}
        />
        {isTracedEdgeless ? (
          <>
            <div className="text-[9px] text-amber-500/70 mt-1 leading-relaxed">
              {COPY.tracedEdgeNote}
            </div>
            {layerInfo && (
              <div className="text-[9px] text-gray-500 mt-1 leading-relaxed">
                {layerInfo}
              </div>
            )}
          </>
        ) : (
          <div className="text-[9px] text-gray-600 mt-0.5">
            {visibleEdges} / {circuitData.edges.length} edges visible
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-3 mb-3 text-[9px] flex-wrap">
        {roles.map((role) => (
          <div key={role} className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: roleColorMap.get(role) }} />
            <span className="text-gray-500">{roleLabel(role)}</span>
          </div>
        ))}
      </div>

      {/* Node list grouped by role */}
      {roles.map((role) => {
        const nodes = byRole.get(role) ?? []
        if (nodes.length === 0) return null
        return (
          <div key={role} className="mb-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide mb-1" style={{ color: roleColorMap.get(role) }}>
              {roleLabel(role)} ({nodes.length})
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
