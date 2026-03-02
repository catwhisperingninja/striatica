// frontend/src/three/CircuitLabels.tsx
// Renders ONE label per circuit on the most-connected feature.
// Diagonal leader line touching the point. Technical name + category shown.
// Distance-based opacity. Famous circuits get larger labels.

import { useRef, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { useAppStore, type CircuitRepFeatures } from '../stores/useAppStore'
import { LABELS } from '../config/rendering'

/**
 * Famous circuits — get prominent styling.
 * Maps circuit ID → short category name + citation.
 */
const FAMOUS_CIRCUITS: Record<string, { category: string; paper: string }> = {
  'coact-capital-of-france': { category: 'Factual Recall', paper: 'Wang et al. 2022' },
  'coact-cat-sat-on': { category: 'Syntax Prediction', paper: 'Elhage et al. 2021' },
  'coact-moon-landing': { category: 'Fact Retrieval', paper: 'Meng et al. 2022' },
  'coact-once-upon-a-time': { category: 'Induction Circuit', paper: 'Olsson et al. 2022' },
  'coact-fibonacci-code': { category: 'Code Generation', paper: 'Code Interp.' },
}

interface LabelData {
  position: [number, number, number]
  featureIndex: number
  circuitId: string
  technicalName: string // e.g. "coact-capital-of-france"
  category: string | null // e.g. "Factual Recall" (only for famous)
  paper: string | null
  isFamous: boolean
  edgeCount: number
}

function buildLabels(
  repFeatures: CircuitRepFeatures,
  positions: number[],
): LabelData[] {
  const labels: LabelData[] = []

  for (const [circuitId, rep] of repFeatures) {
    const i = rep.featureIndex * 3
    if (i + 2 >= positions.length) continue

    const famous = FAMOUS_CIRCUITS[circuitId]

    labels.push({
      position: [positions[i], positions[i + 1], positions[i + 2]],
      featureIndex: rep.featureIndex,
      circuitId,
      technicalName: circuitId,
      category: famous?.category ?? null,
      paper: famous?.paper ?? null,
      isFamous: !!famous,
      edgeCount: rep.edgeCount,
    })
  }

  return labels
}

export default function CircuitLabels() {
  const dataset = useAppStore((s) => s.dataset)
  const circuitRepFeatures = useAppStore((s) => s.circuitRepFeatures)

  const labels = useMemo(() => {
    if (!dataset || circuitRepFeatures.size === 0) return []
    return buildLabels(circuitRepFeatures, dataset.positions)
  }, [dataset, circuitRepFeatures])

  if (labels.length === 0) return null

  return (
    <group>
      {labels.map((label) => (
        <CircuitLabel key={label.circuitId} label={label} />
      ))}
    </group>
  )
}

function CircuitLabel({ label }: { label: LabelData }) {
  const groupRef = useRef<THREE.Group>(null)
  const { camera } = useThree()
  const opacityRef = useRef(0)
  const htmlRef = useRef<HTMLDivElement>(null)

  const fadeInDist = label.isFamous ? LABELS.famousFadeIn : LABELS.regularFadeIn
  const fadeOutDist = label.isFamous ? LABELS.famousFadeOut : LABELS.regularFadeOut

  useFrame(() => {
    if (!groupRef.current || !htmlRef.current) return

    const dist = camera.position.distanceTo(groupRef.current.position)

    let targetOpacity = 0
    if (dist < fadeInDist) {
      targetOpacity = 1
    } else if (dist < fadeOutDist) {
      targetOpacity = 1 - (dist - fadeInDist) / (fadeOutDist - fadeInDist)
    }

    if (label.isFamous && dist < fadeOutDist) {
      targetOpacity = Math.max(targetOpacity, LABELS.famousMinOpacity)
    }

    opacityRef.current += (targetOpacity - opacityRef.current) * LABELS.fadeSpeed
    const opacity = opacityRef.current

    htmlRef.current.style.opacity = opacity < LABELS.hideThreshold ? '0' : String(opacity.toFixed(2))
    htmlRef.current.style.pointerEvents = opacity < LABELS.pointerThreshold ? 'none' : 'auto'
  })

  // Position the Html exactly at the feature point (no offset — the diagonal
  // leader line will visually bridge from the point to the text)
  return (
    <group ref={groupRef} position={label.position}>
      <Html
        center={false}
        style={{ pointerEvents: 'none' }}
        zIndexRange={[10, 0]}
      >
        <div
          ref={htmlRef}
          style={{
            opacity: 0,
            transition: 'none',
            position: 'relative',
            whiteSpace: 'nowrap',
            userSelect: 'none',
          }}
        >
          {/* Container positioned to the upper-right with the diagonal leader
              line starting from the origin (the point) */}
          <svg
            width={label.isFamous ? 28 : 18}
            height={label.isFamous ? 20 : 14}
            style={{
              position: 'absolute',
              left: 0,
              top: label.isFamous ? -20 : -14,
              overflow: 'visible',
            }}
          >
            {/* Diagonal leader line from origin (0,bottom) up-right to (width,0) */}
            <line
              x1={0}
              y1={label.isFamous ? 20 : 14}
              x2={label.isFamous ? 28 : 18}
              y2={0}
              stroke={label.isFamous ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.22)'}
              strokeWidth={label.isFamous ? 1.2 : 0.8}
            />
          </svg>
          {/* Label text — positioned at end of the leader line */}
          <div
            style={{
              position: 'absolute',
              left: label.isFamous ? 30 : 19,
              top: label.isFamous ? -24 : -16,
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
              fontSize: label.isFamous ? '10px' : '8px',
              fontWeight: label.isFamous ? 500 : 400,
              color: label.isFamous ? 'rgba(255,255,255,0.88)' : 'rgba(200,200,200,0.65)',
              lineHeight: '1.3',
              textShadow: label.isFamous
                ? '0 0 6px rgba(6,182,212,0.5), 0 1px 3px rgba(0,0,0,0.85)'
                : '0 1px 3px rgba(0,0,0,0.9)',
              letterSpacing: '0.015em',
            }}
          >
            {/* Technical name always shown */}
            <div style={{ opacity: label.isFamous ? 1 : 0.9 }}>
              {label.technicalName}
            </div>
            {/* Category + citation for famous circuits */}
            {label.isFamous && label.category && (
              <div
                style={{
                  fontSize: '8px',
                  fontWeight: 400,
                  color: 'rgba(6,182,212,0.75)',
                  marginTop: '1px',
                }}
              >
                {label.category} · {label.paper}
              </div>
            )}
          </div>
        </div>
      </Html>
    </group>
  )
}
