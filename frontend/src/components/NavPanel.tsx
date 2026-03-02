// frontend/src/components/NavPanel.tsx
import { useState, useMemo } from 'react'
import { useAppStore } from '../stores/useAppStore'
import { COLORS } from '../config/rendering'

export default function NavPanel() {
  const dataset = useAppStore((s) => s.dataset)
  const setSelected = useAppStore((s) => s.setSelected)
  const setFlyTarget = useAppStore((s) => s.setFlyTarget)
  const isolateUncategorized = useAppStore((s) => s.isolateUncategorized)
  const setIsolateUncategorized = useAppStore((s) => s.setIsolateUncategorized)
  const selectedClusters = useAppStore((s) => s.selectedClusters)
  const toggleClusterSelection = useAppStore((s) => s.toggleClusterSelection)
  const clearClusterSelection = useAppStore((s) => s.clearClusterSelection)
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const [search, setSearch] = useState('')
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['clusters']))

  // Auto-expand clusters section when a point is selected
  const selectedClusterLabel = dataset && selectedIndex !== null
    ? dataset.clusterLabels[selectedIndex]
    : null

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  const searchResults = useMemo(() => {
    if (!dataset || search.length < 2) return []
    const q = search.toLowerCase()
    return dataset.features
      .filter((f) => String(f.index).includes(search) || f.explanation.toLowerCase().includes(q))
      .slice(0, 20)
  }, [dataset, search])

  const clusters = dataset?.clusters.filter((c) => c.id >= 0) ?? []

  // Uncategorized features sorted by local dimension (highest first)
  const uncategorizedByDim = useMemo(() => {
    if (!dataset) return []
    const uncatIndices: number[] = []
    for (let i = 0; i < dataset.clusterLabels.length; i++) {
      if (dataset.clusterLabels[i] < 0) uncatIndices.push(i)
    }
    if (dataset.localDimensions) {
      uncatIndices.sort((a, b) =>
        (dataset.localDimensions![b]) - (dataset.localDimensions![a])
      )
    }
    return uncatIndices.slice(0, 50) // show top 50
  }, [dataset])

  const uncategorizedCount = dataset
    ? dataset.clusterLabels.filter((l) => l < 0).length
    : 0

  return (
    <div className="w-[220px] shrink-0 bg-[--color-panel] border-r border-[--color-panel-border] backdrop-blur-xl p-3 overflow-y-auto">
      <input
        type="text"
        placeholder="Search features..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full bg-gray-900 border border-[--color-panel-border] rounded-md px-2.5 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-[--color-cluster-0] mb-3"
      />

      {/* Search results */}
      {search.length >= 2 && (
        <div className="mb-3">
          <div className="text-[10px] text-gray-500 mb-1">
            {searchResults.length} results
          </div>
          {searchResults.map((f) => (
            <div
              key={f.index}
              className="text-[11px] text-gray-400 py-1 px-1 cursor-pointer hover:text-gray-200 truncate"
              onClick={() => {
                setSelected(f.index)
                // Fly to the feature's position so it's visible in the point cloud
                if (dataset) {
                  const i3 = f.index * 3
                  const pos: [number, number, number] = [
                    dataset.positions[i3],
                    dataset.positions[i3 + 1],
                    dataset.positions[i3 + 2],
                  ]
                  setFlyTarget(pos)
                }
              }}
            >
              #{f.index} {f.explanation}
            </div>
          ))}
        </div>
      )}

      {/* Clusters tree */}
      {!search && (
        <>
          <div className="mb-2">
            <div
              className="flex items-center gap-1.5 py-1 cursor-pointer text-xs font-semibold text-gray-400 hover:text-gray-200"
              onClick={() => {
                clearClusterSelection()
                toggleSection('clusters')
              }}
            >
              <span className={`text-[8px] transition-transform ${expandedSections.has('clusters') ? 'rotate-90' : ''}`}>
                &#9654;
              </span>
              Clusters
              {selectedClusters.size > 0 && (
                <span className="text-[9px] text-gray-600 ml-auto">
                  {selectedClusters.size} selected
                </span>
              )}
            </div>
            {expandedSections.has('clusters') && clusters.map((c) => {
              const isActive = selectedClusters.has(c.id)
              const containsSelected = selectedClusterLabel === c.id
              return (
                <div
                  key={c.id}
                  className={`flex items-center gap-1.5 py-0.5 pl-4 text-[11px] cursor-pointer transition-colors ${
                    isActive
                      ? 'text-gray-100 bg-white/5 rounded'
                      : containsSelected
                        ? 'text-gray-200 bg-white/3 rounded'
                        : 'text-gray-500 hover:text-gray-200'
                  }`}
                  onClick={(e) => toggleClusterSelection(c.id, e.shiftKey)}
                  onDoubleClick={() => setFlyTarget(c.centroid)}
                >
                  <div
                    className={`w-1.5 h-1.5 rounded-full shrink-0 ${isActive ? 'ring-1 ring-white/50' : containsSelected ? 'ring-1 ring-white/30' : ''}`}
                    style={{ background: COLORS.clusters[c.id % COLORS.clusters.length] }}
                  />
                  Cluster {c.id} ({c.count.toLocaleString()})
                  {containsSelected && !isActive && (
                    <span className="text-[8px] text-gray-600 ml-auto">&bull;</span>
                  )}
                </div>
              )
            })}
          </div>

          {/* Uncategorized section */}
          <div className="mb-2">
            <div
              className="flex items-center gap-1.5 py-1 cursor-pointer text-xs font-semibold text-gray-400 hover:text-gray-200"
              onClick={() => toggleSection('uncategorized')}
            >
              <span className={`text-[8px] transition-transform ${expandedSections.has('uncategorized') ? 'rotate-90' : ''}`}>
                &#9654;
              </span>
              <div className="w-1.5 h-1.5 rounded-full shrink-0 bg-gray-700" />
              Uncategorized ({uncategorizedCount.toLocaleString()})
            </div>

            {expandedSections.has('uncategorized') && (
              <div className="pl-4">
                {/* Isolate toggle */}
                <button
                  onClick={() => setIsolateUncategorized(!isolateUncategorized)}
                  className={`w-full text-left text-[10px] py-1 px-1.5 mb-1 rounded cursor-pointer transition-colors ${
                    isolateUncategorized
                      ? 'bg-[#A3D739]/20 text-[#A3D739]'
                      : 'text-gray-600 hover:text-gray-400'
                  }`}
                >
                  {isolateUncategorized ? 'Showing uncategorized' : 'Isolate uncategorized'}
                </button>

                {/* Sorted by local dim */}
                {dataset?.localDimensions && (
                  <div className="text-[9px] text-gray-600 mb-1">
                    Sorted by local dim (highest first)
                  </div>
                )}
                {uncategorizedByDim.map((idx) => {
                  const feat = dataset?.features[idx]
                  const dim = dataset?.localDimensions?.[idx]
                  return (
                    <div
                      key={idx}
                      className="text-[10px] text-gray-500 py-0.5 px-1 cursor-pointer hover:text-gray-200 truncate"
                      onClick={() => {
                        setSelected(idx)
                        if (dataset) {
                          const i3 = idx * 3
                          setFlyTarget([
                            dataset.positions[i3],
                            dataset.positions[i3 + 1],
                            dataset.positions[i3 + 2],
                          ])
                        }
                      }}
                    >
                      <span className="text-gray-600">
                        {dim !== undefined ? `d=${dim.toFixed(1)}` : `#${idx}`}
                      </span>
                      {' '}
                      {feat?.explanation ?? `Feature ${idx}`}
                    </div>
                  )
                })}
                {uncategorizedCount > 50 && (
                  <div className="text-[9px] text-gray-700 py-1">
                    ...and {(uncategorizedCount - 50).toLocaleString()} more
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
