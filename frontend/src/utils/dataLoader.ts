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

/** Per-dataset circuit manifest path: `/data/circuits/<stem>/manifest.json`,
 *  where <stem> is the dataset filename with its `.json` extension removed. */
export function circuitManifestPath(datasetFile: string): string {
  const stem = datasetFile.replace(/\.json$/, '')
  return `/data/circuits/${stem}/manifest.json`
}

/**
 * Load the circuit manifest for a dataset. Circuits are now namespaced per
 * dataset. If the per-dataset manifest is absent or unparseable, resolves to an
 * empty manifest — a dataset with no circuits is a normal state, not an error.
 *
 * TODO(remove-after-data-move): gpt2 circuits still live at the legacy flat path
 * `/data/circuits/manifest.json`. Until they are moved under
 * `/data/circuits/gpt2-small-6-res-jb/`, fall back to the legacy path for that
 * one stem when its per-stem manifest is missing/unparseable.
 */
export async function loadCircuitManifest(datasetFile: string): Promise<CircuitManifest> {
  const stem = datasetFile.replace(/\.json$/, '')

  const perStem = await tryLoadManifest(circuitManifestPath(datasetFile))
  if (perStem) return perStem

  if (stem === 'gpt2-small-6-res-jb') {
    const legacy = await tryLoadManifest('/data/circuits/manifest.json')
    if (legacy) return legacy
  }

  return { circuits: [] }
}

/** Fetch + parse a manifest, returning null on any failure (404, network, or
 *  invalid JSON — e.g. a dev-server SPA fallback serving index.html). */
async function tryLoadManifest(path: string): Promise<CircuitManifest | null> {
  try {
    const resp = await fetch(path)
    if (!resp.ok) return null
    return (await resp.json()) as CircuitManifest
  } catch {
    return null
  }
}
