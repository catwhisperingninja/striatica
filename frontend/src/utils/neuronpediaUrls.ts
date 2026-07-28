// frontend/src/utils/neuronpediaUrls.ts
// Single source of truth for outbound Neuronpedia links. All URLs are built
// from dataset/circuit metadata rather than hardcoded per-view, so a new
// dataset only needs correct metadata to link correctly.
import type { DatasetJSON } from '../types/feature'

/**
 * Transcoder gemma datasets carry a `layer` field like "layer12-l0604". Their
 * Neuronpedia source id is NOT that string — it is `${layer}-gemmascope-transcoder-16k`
 * (e.g. "12-gemmascope-transcoder-16k"). Detect and extract the numeric layer.
 * Returns null for non-transcoder datasets (gpt2, gemmascope res, pythia).
 */
function transcoderLayer(dataset: DatasetJSON): number | null {
  const m = /^layer(\d+)-l0\d+/i.exec(dataset.layer)
  return m ? parseInt(m[1], 10) : null
}

/**
 * URL for a single feature on Neuronpedia.
 * - Transcoder datasets → `${model}/${layer}-gemmascope-transcoder-16k/${index}`.
 * - Everything else (gpt2, gemmascope res) → the dataset's own model/layer path,
 *   which is already a valid Neuronpedia source id.
 */
export function featureUrl(dataset: DatasetJSON, index: number): string {
  const layer = transcoderLayer(dataset)
  if (layer !== null) {
    return `https://www.neuronpedia.org/${dataset.model}/${layer}-gemmascope-transcoder-16k/${index}`
  }
  return `https://neuronpedia.org/${dataset.model}/${dataset.layer}/${index}`
}

/** URL for an attribution graph on Neuronpedia, keyed by its slug. */
export function graphUrl(model: string, slug: string): string {
  return `https://www.neuronpedia.org/${model}/graph?slug=${encodeURIComponent(slug)}`
}
