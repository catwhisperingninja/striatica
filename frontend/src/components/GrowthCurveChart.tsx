import type { GrowthCurve } from '../types/feature'
import { COLORS } from '../config/rendering'

const W = 220
const H = 100
const PAD = 20

interface Props {
  curve: GrowthCurve
}

export default function GrowthCurveChart({ curve }: Props) {
  if (!curve.log_r.length) return null

  const { log_r, log_v, slope, intercept } = curve

  const xMin = Math.min(...log_r)
  const xMax = Math.max(...log_r)
  const yMin = Math.min(...log_v)
  const yMax = Math.max(...log_v)
  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1

  const toX = (v: number) => PAD + ((v - xMin) / xRange) * (W - 2 * PAD)
  const toY = (v: number) => H - PAD - ((v - yMin) / yRange) * (H - 2 * PAD)

  const dataPath = log_r
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${toX(r).toFixed(1)} ${toY(log_v[i]).toFixed(1)}`)
    .join(' ')

  const fitY0 = slope * xMin + intercept
  const fitY1 = slope * xMax + intercept
  const fitPath = `M ${toX(xMin).toFixed(1)} ${toY(fitY0).toFixed(1)} L ${toX(xMax).toFixed(1)} ${toY(fitY1).toFixed(1)}`

  return (
    <div className="mb-3.5">
      <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        VGT Growth Curve
      </div>
      <svg width={W} height={H} className="bg-gray-900/50 rounded">
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#374151" strokeWidth={0.5} />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#374151" strokeWidth={0.5} />
        <path d={fitPath} fill="none" stroke="#A3D739" strokeWidth={1} strokeDasharray="4 2" opacity={0.7} />
        <path d={dataPath} fill="none" stroke={COLORS.accent} strokeWidth={1.5} />
        {log_r.map((r, i) => (
          <circle key={i} cx={toX(r)} cy={toY(log_v[i])} r={2} fill={COLORS.accent} />
        ))}
        <text x={W / 2} y={H - 2} textAnchor="middle" fill="#6b7280" fontSize={8}>log(radius)</text>
        <text x={4} y={H / 2} textAnchor="middle" fill="#6b7280" fontSize={8} transform={`rotate(-90, 4, ${H / 2})`}>log(count)</text>
        <text x={W - PAD} y={12} textAnchor="end" fill="#A3D739" fontSize={9} fontFamily="monospace">
          dim ≈ {slope.toFixed(1)}
        </text>
      </svg>
      <div className="text-[9px] text-gray-600 mt-1">
        Straight = manifold · Kinked = stratified space
      </div>
    </div>
  )
}
