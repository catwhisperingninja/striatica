// frontend/src/utils/dataLoader.ts
import type { DatasetJSON } from '../types/feature'
import type { CircuitData, CircuitManifest } from '../types/circuit'

export interface DatasetEntry {
  file: string
  model: string
  layer: string
  numFeatures: number
}

export async function listDatasets(): Promise<DatasetEntry[]> {
  const resp = await fetch('/data/datasets.json')
  if (!resp.ok) return []
  return resp.json()
}

export async function loadDataset(path: string): Promise<DatasetJSON> {
  const resp = await fetch(path)
  if (!resp.ok) throw new Error(`Failed to load ${path}: ${resp.status}`)
  return resp.json()
}

export async function loadCircuit(path: string): Promise<CircuitData> {
  const resp = await fetch(path)
  if (!resp.ok) throw new Error(`Failed to load circuit ${path}: ${resp.status}`)
  return resp.json()
}

export async function loadCircuitManifest(): Promise<CircuitManifest> {
  const resp = await fetch('/data/circuits/manifest.json')
  if (!resp.ok) throw new Error(`Failed to load circuit manifest: ${resp.status}`)
  return resp.json()
}
