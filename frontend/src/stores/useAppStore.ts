// frontend/src/stores/useAppStore.ts
import { create } from 'zustand'
import type { ColorMode, ViewMode, DatasetJSON, FeatureData } from '../types/feature'
import type { CircuitData, CircuitManifest } from '../types/circuit'
import { loadCircuit } from '../utils/dataLoader'

/** Maps featureIndex → list of circuit IDs that contain it */
export type CircuitMembershipIndex = Map<number, string[]>

/** For each circuit, the feature with the most edges (best representative for labeling) */
export type CircuitRepFeatures = Map<string, { featureIndex: number; edgeCount: number; circuitName: string; description: string }>

interface AppState {
  // Data
  dataset: DatasetJSON | null
  loading: boolean
  error: string | null

  // Selection
  hoveredIndex: number | null
  selectedIndex: number | null
  selectedClusters: Set<number>

  // View
  viewMode: ViewMode
  colorMode: ColorMode
  zoom: number
  flyTarget: [number, number, number] | null
  flyKey: number
  resetKey: number
  isolateUncategorized: boolean

  // Circuit
  circuitData: CircuitData | null
  circuitManifest: CircuitManifest | null
  circuitMembership: CircuitMembershipIndex
  circuitRepFeatures: CircuitRepFeatures
  edgeThreshold: number

  // Actions
  setDataset: (data: DatasetJSON) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setHovered: (index: number | null) => void
  setSelected: (index: number | null) => void
  toggleClusterSelection: (id: number, shift: boolean) => void
  clearClusterSelection: () => void
  resetSelection: () => void
  setViewMode: (mode: ViewMode) => void
  setColorMode: (mode: ColorMode) => void
  setZoom: (zoom: number) => void
  setFlyTarget: (target: [number, number, number] | null) => void
  setIsolateUncategorized: (isolate: boolean) => void
  setCircuitData: (data: CircuitData | null) => void
  setCircuitManifest: (manifest: CircuitManifest | null) => void
  buildCircuitMembership: () => Promise<void>
  loadCircuitById: (id: string) => Promise<void>
  setEdgeThreshold: (threshold: number) => void

  // Derived
  selectedFeature: () => FeatureData | null
  hoveredFeature: () => FeatureData | null
  hasLocalDimData: () => boolean
  getCircuitsForFeature: (featureIndex: number) => string[]
}

export const useAppStore = create<AppState>((set, get) => ({
  dataset: null,
  loading: false,
  error: null,
  hoveredIndex: null,
  selectedIndex: null,
  selectedClusters: new Set<number>(),
  viewMode: 'pointCloud',
  colorMode: 'cluster',
  zoom: 1,
  flyTarget: null,
  flyKey: 0,
  resetKey: 0,
  isolateUncategorized: false,
  circuitData: null,
  circuitManifest: null,
  circuitMembership: new Map<number, string[]>(),
  circuitRepFeatures: new Map(),
  edgeThreshold: 0.1,

  setDataset: (data) => set({ dataset: data, loading: false, error: null }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error, loading: false }),
  setHovered: (index) => set({ hoveredIndex: index }),
  setSelected: (index) => {
    if (index !== null) {
      const { dataset } = get()
      if (dataset) {
        const clusterLabel = dataset.clusterLabels[index]
        if (clusterLabel >= 0) {
          set({ selectedIndex: index, selectedClusters: new Set([clusterLabel]) })
          return
        }
      }
    }
    set({ selectedIndex: index, selectedClusters: new Set<number>() })
  },
  toggleClusterSelection: (id, shift) => {
    const prev = get().selectedClusters
    if (!shift) {
      // Single-select: toggle this cluster as the only selection
      if (prev.size === 1 && prev.has(id)) {
        set({ selectedClusters: new Set<number>(), isolateUncategorized: false })
      } else {
        set({ selectedClusters: new Set([id]), isolateUncategorized: false })
      }
    } else {
      // Multi-select: add/remove from current selection
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else if (next.size < 10) {
        next.add(id)
      } else {
        return // At cap, no change
      }
      set({ selectedClusters: next, isolateUncategorized: false })
    }
  },
  clearClusterSelection: () => {
    if (get().selectedClusters.size > 0) {
      set({ selectedClusters: new Set<number>() })
    }
  },
  resetSelection: () => {
    set((s) => ({
      selectedIndex: null,
      hoveredIndex: null,
      selectedClusters: new Set<number>(),
      flyTarget: null,
      isolateUncategorized: false,
      circuitData: null,
      viewMode: 'pointCloud' as ViewMode,
      resetKey: s.resetKey + 1,
    }))
  },
  setViewMode: (mode) => set({ viewMode: mode }),
  setColorMode: (mode) => set({ colorMode: mode }),
  setZoom: (zoom) => set({ zoom }),
  setFlyTarget: (target) => set((s) => ({ flyTarget: target, flyKey: s.flyKey + 1 })),
  setIsolateUncategorized: (isolate) => set({ isolateUncategorized: isolate, selectedClusters: new Set<number>() }),
  setCircuitData: (data) => set({ circuitData: data }),
  setCircuitManifest: (manifest) => set({ circuitManifest: manifest }),
  buildCircuitMembership: async () => {
    const { circuitManifest } = get()
    if (!circuitManifest) return
    const index = new Map<number, string[]>()
    const repFeatures: CircuitRepFeatures = new Map()

    // Load every circuit and index which features appear in each
    await Promise.all(
      circuitManifest.circuits.map(async (entry) => {
        try {
          const data = await loadCircuit(entry.path)

          // Build membership index
          for (const node of data.nodes) {
            const existing = index.get(node.featureIndex)
            if (existing) {
              existing.push(entry.id)
            } else {
              index.set(node.featureIndex, [entry.id])
            }
          }

          // Find the most-connected feature (most edges) in this circuit
          const edgeCounts = new Map<number, number>()
          for (const edge of data.edges) {
            edgeCounts.set(edge.source, (edgeCounts.get(edge.source) ?? 0) + 1)
            edgeCounts.set(edge.target, (edgeCounts.get(edge.target) ?? 0) + 1)
          }
          let bestFeature = data.nodes[0]?.featureIndex ?? -1
          let bestCount = 0
          for (const [fi, count] of edgeCounts) {
            if (count > bestCount) {
              bestCount = count
              bestFeature = fi
            }
          }
          if (bestFeature >= 0) {
            repFeatures.set(entry.id, {
              featureIndex: bestFeature,
              edgeCount: bestCount,
              circuitName: entry.id,
              description: entry.description,
            })
          }
        } catch {
          // Skip circuits that fail to load
        }
      }),
    )
    set({ circuitMembership: index, circuitRepFeatures: repFeatures })
  },
  loadCircuitById: async (id) => {
    const { circuitManifest } = get()
    const entry = circuitManifest?.circuits.find((c) => c.id === id)
    if (!entry) return
    const data = await loadCircuit(entry.path)
    set({ circuitData: data })
  },
  setEdgeThreshold: (threshold) => set({ edgeThreshold: threshold }),

  selectedFeature: () => {
    const { dataset, selectedIndex } = get()
    if (!dataset || selectedIndex === null) return null
    return dataset.features[selectedIndex] ?? null
  },
  hoveredFeature: () => {
    const { dataset, hoveredIndex } = get()
    if (!dataset || hoveredIndex === null) return null
    return dataset.features[hoveredIndex] ?? null
  },
  hasLocalDimData: () => {
    const { dataset } = get()
    return !!(dataset?.localDimensions?.length)
  },
  getCircuitsForFeature: (featureIndex: number) => {
    return get().circuitMembership.get(featureIndex) ?? []
  },
}))
