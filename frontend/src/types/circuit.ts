// frontend/src/types/circuit.ts

/** Circuit provenance. 'traced' = attribution-graph circuits from a causal tracer. */
export type CircuitType = 'coactivation' | 'similarity' | 'traced'

export interface CircuitNode {
  featureIndex: number   // index into DatasetJSON.features
  activation: number     // activation strength in this circuit (0-1)
  /**
   * Role within the circuit. Legacy co-activation/similarity circuits use
   * 'source' | 'intermediate' | 'sink'; traced circuits may use arbitrary role
   * strings (or 'unassigned'). Kept as a free string so new taxonomies render
   * without a type change. See utils/circuitRoles.ts.
   */
  role: string
  /** Attribution influence (traced circuits). */
  influence?: number
  /** Number of graph instances this node was aggregated from (traced circuits). */
  instances?: number
}

export interface CircuitEdge {
  source: number   // featureIndex of source node
  target: number   // featureIndex of target node
  weight: number   // attribution strength (0-1)
}

/**
 * Optional per-circuit metadata. All fields optional; the index signature keeps
 * forward-compatibility with fields the pipeline adds later. Traced circuits
 * carry sourceSet/slug (Neuronpedia) plus cross-layer accounting.
 */
export interface CircuitMetadata {
  sourceSet?: string          // Neuronpedia source set id
  slug?: string               // Neuronpedia graph slug (for graphUrl)
  layerFilter?: number        // layer this atlas slice was filtered to
  crossLayerMembers?: number  // graph nodes recorded on other layers
  l0Verified?: boolean        // whether L0 was verified against the source
  [key: string]: unknown
}

export interface CircuitData {
  name: string
  description?: string
  type?: CircuitType
  source?: string
  metadata?: CircuitMetadata
  nodes: CircuitNode[]
  edges: CircuitEdge[]
}

export interface CircuitManifestEntry {
  id: string
  name: string
  type: CircuitType
  description: string
  nodeCount: number
  edgeCount: number
  path: string
  category?: string
  citation?: string
  famous?: boolean
}

export interface CircuitManifest {
  circuits: CircuitManifestEntry[]
}
