// frontend/src/components/DebugConsole.tsx
import { useEffect, useRef, useState, useCallback } from 'react'
import { useAppStore } from '../stores/useAppStore'
import { cameraSync } from '../utils/cameraSync'

interface LogEntry {
  time: string
  message: string
}

const MAX_LOG = 3

export default function DebugConsole() {
  const [open, setOpen] = useState(true)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [copied, setCopied] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)
  const prevState = useRef<Record<string, unknown>>({})

  const pushLog = useCallback((msg: string) => {
    const now = new Date()
    const time = `${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`
    setLogs((prev) => [...prev.slice(-(MAX_LOG - 1)), { time, message: msg }])
  }, [])

  // Toggle with backtick key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '`' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Subscribe to store changes and log meaningful diffs
  useEffect(() => {
    // Log all tracked fields EXCEPT hoveredIndex (too noisy)
    const tracked = [
      'selectedIndex', 'viewMode', 'colorMode',
      'edgeThreshold', 'flyKey', 'loading', 'error',
      'currentDatasetFile',
    ] as const

    const unsub = useAppStore.subscribe((state) => {
      for (const key of tracked) {
        const val = state[key]
        if (val !== prevState.current[key]) {
          const display = val === null ? 'null' : String(val)
          pushLog(`${key}: ${prevState.current[key] === undefined ? '(init)' : String(prevState.current[key])} → ${display}`)
          prevState.current[key] = val
        }
      }

      // hoveredIndex: only log null↔non-null transitions (not every index change)
      const hovered = state.hoveredIndex
      const prevHovered = prevState.current.hoveredIndex as number | null | undefined
      const wasNull = prevHovered === null || prevHovered === undefined
      const isNull = hovered === null
      if (wasNull !== isNull) {
        pushLog(`hover: ${wasNull ? 'none' : `#${prevHovered}`} → ${isNull ? 'none' : `#${hovered}`}`)
      }
      prevState.current.hoveredIndex = hovered

      // Track selectedClusters separately (Set)
      const clusters = state.selectedClusters
      const prevClusters = prevState.current.selectedClusters as Set<number> | undefined
      const curStr = `{${[...clusters].join(',')}}`
      const prevStr = prevClusters ? `{${[...prevClusters].join(',')}}` : '(init)'
      if (curStr !== prevStr) {
        pushLog(`selectedClusters: ${prevStr} → ${curStr}`)
        prevState.current.selectedClusters = new Set(clusters)
      }

      // Track flyTarget
      const ft = state.flyTarget
      const prevFt = prevState.current.flyTarget as [number, number, number] | null | undefined
      const ftStr = ft ? `[${ft.map((v) => v.toFixed(2)).join(', ')}]` : 'null'
      const prevFtStr = prevFt ? `[${prevFt.map((v) => v.toFixed(2)).join(', ')}]` : prevFt === null ? 'null' : '(init)'
      if (ftStr !== prevFtStr) {
        pushLog(`flyTarget: ${prevFtStr} → ${ftStr}`)
        prevState.current.flyTarget = ft
      }

      // Track circuit manifest load
      const manifest = state.circuitManifest
      const prevManifest = prevState.current.circuitManifest
      if (manifest !== prevManifest) {
        const count = manifest?.circuits.length ?? 0
        pushLog(`circuitManifest: ${prevManifest === undefined ? '(init)' : 'null'} → ${manifest ? `${count} circuits` : 'null'}`)
        prevState.current.circuitManifest = manifest
      }

      // Track circuit data selection changes
      const circuit = state.circuitData
      const prevCircuit = prevState.current.circuitData as { name?: string } | null | undefined
      const circuitName = circuit?.name ?? 'null'
      const prevCircuitName = prevCircuit?.name ?? (prevCircuit === undefined ? '(init)' : 'null')
      if (circuitName !== prevCircuitName) {
        pushLog(`circuit: ${prevCircuitName} → ${circuitName}`)
        prevState.current.circuitData = circuit
      }
    })

    return unsub
  }, [pushLog])

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  // Current state snapshot
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const hoveredIndex = useAppStore((s) => s.hoveredIndex)
  const viewMode = useAppStore((s) => s.viewMode)
  const colorMode = useAppStore((s) => s.colorMode)
  const selectedClusters = useAppStore((s) => s.selectedClusters)
  const flyTarget = useAppStore((s) => s.flyTarget)
  const flyKey = useAppStore((s) => s.flyKey)
  const edgeThreshold = useAppStore((s) => s.edgeThreshold)
  const dataset = useAppStore((s) => s.dataset)
  const circuitData = useAppStore((s) => s.circuitData)
  const circuitManifest = useAppStore((s) => s.circuitManifest)
  const loading = useAppStore((s) => s.loading)
  const error = useAppStore((s) => s.error)
  const currentDatasetFile = useAppStore((s) => s.currentDatasetFile)

  // Poll camera position from shared mutable object (~4fps)
  const [camPosStr, setCamPosStr] = useState('0.00, 0.00, 0.00')
  useEffect(() => {
    const id = setInterval(() => {
      setCamPosStr(
        `${cameraSync.position[0].toFixed(2)}, ${cameraSync.position[1].toFixed(2)}, ${cameraSync.position[2].toFixed(2)}`,
      )
    }, 250)
    return () => clearInterval(id)
  }, [])

  const selectedFeature = dataset && selectedIndex !== null
    ? dataset.features[selectedIndex]
    : null

  // Copy full state + log to clipboard
  const handleCopy = () => {
    const state = [
      `viewMode: ${viewMode}`,
      `colorMode: ${colorMode}`,
      `selectedIndex: ${selectedIndex}`,
      `hoveredIndex: ${hoveredIndex}`,
      `selectedClusters: ${selectedClusters.size === 0 ? '(none)' : `{${[...selectedClusters].join(', ')}}`}`,
      `flyTarget: ${flyTarget ? `[${flyTarget.map((v) => v.toFixed(2)).join(', ')}]` : 'null'}`,
      `flyKey: ${flyKey}`,
      `edgeThreshold: ${edgeThreshold.toFixed(2)}`,
      `loading: ${loading}`,
      `error: ${error ?? 'null'}`,
      `points: ${dataset?.numFeatures.toLocaleString() ?? '0'}`,
      `manifest: ${circuitManifest ? `${circuitManifest.circuits.length} circuits` : 'null'}`,
      `circuit: ${circuitData ? `${circuitData.name} (${circuitData.nodes.length}n)` : 'null'}`,
    ]
    if (selectedFeature) {
      state.push(
        `sel.feature: #${selectedFeature.index}`,
        `sel.explain: ${selectedFeature.explanation ?? '(none)'}`,
        `sel.cluster: ${selectedIndex !== null ? dataset?.clusterLabels[selectedIndex] : 'null'}`,
      )
    }
    const logText = logs.map((e) => `${e.time} ${e.message}`).join('\n')
    const text = `--- STATE ---\n${state.join('\n')}\n\n--- LOG (last ${logs.length}) ---\n${logText}`
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  if (!open) {
    return (
      <div className="fixed top-12 right-[270px] z-[9999]">
        <button
          onClick={() => setOpen(true)}
          className="px-2 py-1 text-[10px] font-mono text-yellow-400 bg-gray-900 border border-yellow-600 rounded cursor-pointer hover:bg-yellow-900/30 transition-colors"
          title="Debug Console (`)"
        >
          DEBUG
        </button>
      </div>
    )
  }

  return (
    <div className="fixed top-12 right-[270px] z-[9999] w-[320px] max-h-[80vh] bg-black/95 border border-yellow-700/50 rounded-lg font-mono text-[10px] text-gray-300 shadow-2xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-gray-800 bg-gray-900/50">
        <span className="text-[9px] font-bold text-gray-400 uppercase tracking-wider">Debug Console</span>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="text-[9px] text-yellow-500 hover:text-yellow-300 cursor-pointer"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <button
            onClick={() => setLogs([])}
            className="text-[9px] text-gray-600 hover:text-gray-400 cursor-pointer"
          >
            Clear
          </button>
          <button
            onClick={() => setOpen(false)}
            className="text-gray-500 hover:text-gray-300 cursor-pointer"
          >
            ×
          </button>
        </div>
      </div>

      {/* State panel */}
      <div className="px-2 py-1.5 border-b border-gray-800 space-y-0.5">
        <Row label="dataset" value={currentDatasetFile ?? 'null'} />
        <Row label="viewMode" value={viewMode} />
        <Row label="colorMode" value={colorMode} />
        <Row label="selectedIndex" value={selectedIndex === null ? 'null' : String(selectedIndex)} highlight={selectedIndex !== null} />
        <Row label="hoveredIndex" value={hoveredIndex === null ? 'null' : String(hoveredIndex)} />
        <Row label="clusters" value={selectedClusters.size === 0 ? '(none)' : `{${[...selectedClusters].join(', ')}}`} />
        <Row label="flyTarget" value={flyTarget ? `[${flyTarget.map((v) => v.toFixed(2)).join(', ')}]` : 'null'} />
        <Row label="edgeThreshold" value={edgeThreshold.toFixed(2)} />
        <Row label="manifest" value={circuitManifest ? `${circuitManifest.circuits.length} circuits` : 'null'} />
        <Row label="circuit" value={circuitData ? `${circuitData.name} (${circuitData.nodes.length}n)` : 'null'} highlight={!!circuitData} />
        <Row label="error" value={error ?? 'null'} highlight={!!error} highlightColor="text-red-400" />
        <Row label="camera" value={camPosStr} />
        {selectedFeature && (
          <>
            <div className="border-t border-gray-800 pt-0.5 mt-1" />
            <Row label="sel.feature" value={`#${selectedFeature.index}`} highlight />
            <Row label="sel.explain" value={selectedFeature.explanation?.slice(0, 40) || '(none)'} />
            <Row label="sel.cluster" value={selectedIndex !== null ? String(dataset?.clusterLabels[selectedIndex]) : 'null'} />
          </>
        )}
      </div>

      {/* Log panel */}
      <div ref={logRef} className="flex-1 overflow-y-auto px-2 py-1 min-h-[40px] max-h-[60px]">
        {logs.length === 0 ? (
          <div className="text-gray-600 italic">Interact with the scene...</div>
        ) : (
          logs.map((entry, i) => (
            <div key={i} className="flex gap-1.5 leading-tight py-px">
              <span className="text-gray-600 shrink-0">{entry.time}</span>
              <span className="text-gray-400">{entry.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function Row({ label, value, highlight, highlightColor }: {
  label: string
  value: string
  highlight?: boolean
  highlightColor?: string
}) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className={highlight ? (highlightColor ?? 'text-cyan-400') : 'text-gray-300'}>
        {value}
      </span>
    </div>
  )
}
