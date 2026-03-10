// frontend/src/types/feature.ts
export interface FeatureData {
  index: number
  explanation: string
  maxAct: number
  fracNonzero: number
  topSimilar: number[]
  posTokens: string[]
  negTokens: string[]
}

export interface ClusterData {
  id: number
  count: number
  centroid: [number, number, number]
}

export type ColorMode = 'cluster' | 'localDim'
export type ViewMode = 'pointCloud' | 'circuits' | 'localDim'

export interface GrowthCurve {
  log_r: number[]    // log(radius) values
  log_v: number[]    // log(neighbor count) values
  slope: number      // fitted slope = local dimension
  intercept: number  // fitted intercept
}

export interface DatasetJSON {
  model: string
  layer: string
  numFeatures: number
  positions: number[]       // flat: [x0,y0,z0, x1,y1,z1, ...]
  clusterLabels: number[]
  localDimensions?: number[] // per-point local intrinsic dimension
  dimMethod?: string         // 'pr' | 'twonn' | 'vgt'
  growthCurves?: GrowthCurve[] // per-point VGT log-log curves (for detail panel)
  clusters: ClusterData[]
  features: FeatureData[]
  semanticsRedacted?: boolean // true when feature explanations were stripped for safety
}
