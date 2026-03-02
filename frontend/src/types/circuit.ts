// frontend/src/types/circuit.ts

export interface CircuitNode {
  featureIndex: number   // index into DatasetJSON.features
  activation: number     // activation strength in this circuit (0-1)
  role: 'source' | 'intermediate' | 'sink'
}

export interface CircuitEdge {
  source: number   // featureIndex of source node
  target: number   // featureIndex of target node
  weight: number   // attribution strength (0-1)
}

export interface CircuitData {
  name: string
  description?: string
  nodes: CircuitNode[]
  edges: CircuitEdge[]
}

export interface CircuitManifestEntry {
  id: string
  name: string
  type: 'coactivation' | 'similarity'
  description: string
  nodeCount: number
  edgeCount: number
  path: string
}

export interface CircuitManifest {
  circuits: CircuitManifestEntry[]
}
