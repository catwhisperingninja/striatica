import { useAppStore } from '../stores/useAppStore'
import GrowthCurveChart from './GrowthCurveChart'
import { COLORS } from '../config/rendering'

const ROLE_LABELS: Record<string, string> = {
  source: 'Source',
  intermediate: 'Processing',
  sink: 'Output',
}

export default function DetailPanel() {
  const dataset = useAppStore((s) => s.dataset)
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const viewMode = useAppStore((s) => s.viewMode)
  const circuitData = useAppStore((s) => s.circuitData)
  const getCircuitsForFeature = useAppStore((s) => s.getCircuitsForFeature)

  if (!dataset || selectedIndex === null) {
    return (
      <div className="w-[260px] shrink-0 bg-[--color-panel] border-l border-[--color-panel-border] backdrop-blur-xl p-4 flex items-center justify-center">
        <p className="text-xs text-gray-600">Click a point to inspect</p>
      </div>
    )
  }

  const feature = dataset.features[selectedIndex]
  if (!feature) return null

  const clusterLabel = dataset.clusterLabels[selectedIndex]
  const clusterColor = clusterLabel >= 0
    ? COLORS.clusters[clusterLabel % COLORS.clusters.length]
    : '#374151'

  const actPercent = Math.min(feature.fracNonzero * 10000, 100)

  return (
    <div className="w-[260px] shrink-0 bg-[--color-panel] border-l border-[--color-panel-border] backdrop-blur-xl p-4 overflow-y-auto">
      <div className="text-[13px] font-bold text-gray-200 mb-1">
        {feature.explanation || `Feature #${feature.index}`}
      </div>
      <div className="text-[11px] text-gray-600 font-mono mb-3">
        {dataset.model}@{dataset.layer}:{feature.index}
      </div>

      {feature.explanation ? (
        <div className="mb-3.5">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
            Explanation
          </div>
          <div className="text-xs text-gray-300 leading-relaxed">
            {feature.explanation}
          </div>
        </div>
      ) : dataset.semanticsRedacted ? (
        <div className="mb-3.5 border border-red-900/40 rounded-md px-2.5 py-2 bg-red-950/20">
          <div className="text-[10px] font-semibold text-red-400/80 uppercase tracking-wide mb-1">
            Semantic Labels Redacted
          </div>
          <div className="text-[10px] text-red-300/60 leading-relaxed">
            Feature explanations for this model are withheld by default.
            Interpretability data applied to capable models can reveal
            safety-relevant circuits. If published or ingested into training
            data, this could enable circumvention of AI alignment mechanisms.
          </div>
        </div>
      ) : null}

      <div className="mb-3.5">
        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
          Statistics
        </div>
        <Stat label="Max activation" value={feature.maxAct.toFixed(2)} />
        <Stat label="Activation freq" value={feature.fracNonzero.toExponential(2)} />
        <Stat label="Cluster" value={`#${clusterLabel}`} color={clusterColor} />
      </div>

      {(() => {
        const circuits = getCircuitsForFeature(feature.index)
        return circuits.length > 0 ? (
          <div className="mb-3.5">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
              Circuit Membership
            </div>
            <div className="text-[10px] text-cyan-400 mb-1">
              In {circuits.length} circuit{circuits.length > 1 ? 's' : ''}
            </div>
            {circuits.map((id) => (
              <div key={id} className="text-[10px] text-gray-400 py-0.5 truncate">
                {id}
              </div>
            ))}
          </div>
        ) : (
          <div className="mb-3.5">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
              Circuit Membership
            </div>
            <div className="text-[10px] text-gray-600 italic">
              Not in any circuit
            </div>
          </div>
        )
      })()}

      <div className="mb-3.5">
        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
          Activation Strength
        </div>
        <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${actPercent}%`,
              background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.roles.source})`,
            }}
          />
        </div>
      </div>

      {feature.posTokens.length > 0 && (
        <div className="mb-3.5">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
            Top Activating Tokens
          </div>
          <div className="flex flex-wrap gap-1">
            {feature.posTokens.map((t, i) => (
              <span key={i} className="text-[10px] text-gray-300 bg-gray-800 px-1.5 py-0.5 rounded">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {dataset.growthCurves?.[selectedIndex] && (
        <GrowthCurveChart curve={dataset.growthCurves[selectedIndex]} />
      )}

      {viewMode === 'circuits' && circuitData && (() => {
        const circuitNode = circuitData.nodes.find((n) => n.featureIndex === selectedIndex)
        if (!circuitNode) return null
        const connectedEdges = circuitData.edges.filter(
          (e) => e.source === selectedIndex || e.target === selectedIndex
        )
        return (
          <div className="mb-3.5">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
              Circuit Role
            </div>
            <Stat
              label="Role"
              value={ROLE_LABELS[circuitNode.role] ?? circuitNode.role}
              color={COLORS.roles[circuitNode.role]}
            />
            <Stat label="Activation" value={circuitNode.activation.toFixed(3)} />
            {connectedEdges.length > 0 && (
              <div className="mt-2">
                <div className="text-[10px] text-gray-500 mb-1">
                  Connections ({connectedEdges.length})
                </div>
                {connectedEdges.slice(0, 8).map((edge, i) => {
                  const other = edge.source === selectedIndex ? edge.target : edge.source
                  const dir = edge.source === selectedIndex ? '\u2192' : '\u2190'
                  const otherFeat = dataset.features[other]
                  return (
                    <div key={i} className="text-[9px] text-gray-500 truncate py-px">
                      {dir} #{other} <span className="text-gray-600">({edge.weight.toFixed(2)})</span>
                      {otherFeat?.explanation && (
                        <span className="text-gray-600 ml-1">{otherFeat.explanation.slice(0, 25)}</span>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })()}

      <div className="mt-4 space-y-1.5">
        <button
          className="block w-full py-1.5 bg-gray-900 border border-[--color-panel-border] rounded-md text-[11px] text-gray-400 cursor-pointer hover:border-[--color-cluster-0] hover:text-[--color-cluster-0] transition-colors text-center"
          onClick={() => {
            window.open(
              `https://neuronpedia.org/${dataset.model}/${dataset.layer}/${feature.index}`,
              '_blank'
            )
          }}
        >
          Open in Neuronpedia
        </button>
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between py-0.5 text-[11px] border-b border-gray-800">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200 font-mono" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  )
}
