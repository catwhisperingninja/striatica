import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { BufferGeometry, Color, Float32BufferAttribute, ShaderMaterial, Vector3 } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import DebugConsole from '../components/DebugConsole'
import SelectionRing from '../three/SelectionRing'
import CameraTracker from '../three/CameraTracker'
import { CAMERA, DOF, GRAPH_PILOT as STYLE, POINTS } from '../config/rendering'
import { graphUrl } from '../utils/neuronpediaUrls'
import vertexShader from '../shaders/point.vert.glsl?raw'
import fragmentShader from '../shaders/point.frag.glsl?raw'

type Layout = 'spectral' | 'pca'
type Position = [number, number, number]
interface Node {
  node_id: string
  feature_type: string
  layer: string | number
  ctx_idx: number | null
  featureAnchor: { layer: number; featureIndex: number } | null
  layoutRow: number | null
}
interface Edge { source: string; target: string; weight: number }
interface Graph {
  nodes: Node[]
  edges: Edge[]
  metadata: { model: string; slug: string; rawGraphSha256: string }
}
interface PilotData {
  keys: [number, number][]
  layouts: Record<Layout, Position[]>
  retained: Record<Layout, number[]>
  graphs: Graph[]
  report: { neighboursExcludingSelf: number; inputSha256: string } & Record<Layout, { meanRetention: number; perLayer: { layer: number; meanRetention: number }[] }>
}

function nodeLabel(node: Node) {
  const a = node.featureAnchor
  return `${a ? `L${a.layer} #${a.featureIndex}` : `${node.feature_type} · layer ${node.layer}`} · ctx ${node.ctx_idx ?? '?'}`
}

export default function TracedGraphPilot() {
  const [data, setData] = useState<PilotData | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/__pilot/no-umap.json', { signal: controller.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error('Prepared pilot cache unavailable. This dev-only slice has no production-data fallback.')
        return r.json() as Promise<PilotData>
      })
      .then(setData)
      .catch((e: Error) => { if (!controller.signal.aborted) setError(e.message) })
    return () => controller.abort()
  }, [])
  if (!data) return <main className="p-8"><h1 className="text-xl">No-UMAP graph pilot</h1><p role={error ? 'alert' : 'status'}>{error ?? 'Loading existing, label-free pilot outputs…'}</p><a className="text-cyan-300" href="/">Return to atlas</a></main>
  return <Pilot data={data} />
}

function Pilot({ data }: { data: PilotData }) {
  const [graphIndex, setGraphIndex] = useState(0)
  const [layout, setLayout] = useState<Layout>('spectral')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [scope, setScope] = useState('incident')
  const controls = useRef<OrbitControlsImpl>(null)
  const graph = data.graphs[graphIndex]
  const nodeById = useMemo(() => new Map(graph.nodes.map((n) => [n.node_id, n])), [graph])
  const strongest = useMemo(() => graph.edges.filter((e) =>
    nodeById.get(e.source)!.layoutRow !== null && nodeById.get(e.target)!.layoutRow !== null,
  ).reduce<Edge | null>(
    (best, e) => !best || Math.abs(e.weight) > Math.abs(best.weight) ? e : best, null), [graph, nodeById])
  const selected = nodeById.get(selectedId ?? strongest?.source ?? '') ?? null
  const selectedRow = selected?.layoutRow ?? null
  const rows = useMemo(() => [...new Set(graph.nodes.flatMap((n) => n.layoutRow === null ? [] : [n.layoutRow]))], [graph])
  // One translation and uniform scale for the entire union, never a per-graph refit.
  const positions = useMemo(() => {
    const raw = data.layouts[layout]
    const center = raw.reduce((v, p) => v.add(new Vector3(...p)), new Vector3()).divideScalar(raw.length)
    const centered = raw.map((p) => new Vector3(...p).sub(center))
    const radius = Math.max(...centered.map((v) => v.length()))
    return centered.map((v) => v.multiplyScalar(STYLE.sceneRadius / radius).toArray() as Position)
  }, [data, layout])
  const incident = useMemo(() => graph.edges.filter((e) => e.source === selected?.node_id || e.target === selected?.node_id), [graph, selected])
  const visible = scope === 'all' ? graph.edges : incident
  const { spatial, unpositioned, coincident } = useMemo(() => {
    const spatial: { edge: Edge; from: Position; to: Position }[] = []
    let unpositioned = 0, coincident = 0
    for (const edge of visible) {
      const fromRow = nodeById.get(edge.source)!.layoutRow
      const toRow = nodeById.get(edge.target)!.layoutRow
      if (fromRow === null || toRow === null) { unpositioned++; continue }
      const from = positions[fromRow], to = positions[toRow]
      if (from.every((value, axis) => value === to[axis])) { coincident++; continue }
      spatial.push({ edge, from, to })
    }
    return { spatial, unpositioned, coincident }
  }, [visible, nodeById, positions])
  const neighbours = useMemo(() => new Set(incident.flatMap((e) => [nodeById.get(e.source)!.layoutRow, nodeById.get(e.target)!.layoutRow])), [incident, nodeById])
  const instances = selectedRow === null ? [] : graph.nodes.filter((n) => n.layoutRow === selectedRow)
  const specials = graph.nodes.filter((n) => n.layoutRow === null)
  const retained = (100 * data.report[layout].meanRetention).toFixed(2)
  const spectralRetention = (100 * data.report.spectral.meanRetention).toFixed(2)
  const pcaRetention = (100 * data.report.pca.meanRetention).toFixed(2)
  const lost = (100 * (1 - data.report.spectral.meanRetention)).toFixed(1)
  const worseLayers = data.report.spectral.perLayer.filter((s) =>
    s.meanRetention < data.report.pca.perLayer.find((p) => p.layer === s.layer)!.meanRetention,
  ).map((s) => s.layer).join(', ')
  const changeGraph = (index: number) => {
    const anchor = selected?.featureAnchor
    const shared = anchor && data.graphs[index].nodes.find((n) => n.featureAnchor?.layer === anchor.layer && n.featureAnchor.featureIndex === anchor.featureIndex)
    setSelectedId(shared ? shared.node_id : null)
    setGraphIndex(index)
  }
  const chooseRow = (row: number) => setSelectedId(graph.nodes.find((n) => n.layoutRow === row)!.node_id)
  const selectStyle = 'w-full bg-gray-900 border border-gray-700 rounded px-2 py-2 text-sm'
  return (
    <main className="h-screen flex flex-col overflow-hidden text-gray-200">
      <DebugConsole initialOpen={false} positionClassName="top-4 right-4 lg:top-12 lg:right-[270px]" extraState={{
        'pilot.graph': graph.metadata.slug, 'pilot.layout': layout, 'pilot.scope': scope,
        'pilot.instance': selected?.node_id ?? 'none', 'pilot.anchorRow': String(selectedRow),
        'pilot.nodes': String(graph.nodes.length), 'pilot.edges': String(graph.edges.length),
        'pilot.spatial': String(spatial.length), 'pilot.unpositioned': String(unpositioned),
        'pilot.coincident': String(coincident),
      }} />
      <header className="border-b border-gray-800 px-5 py-3 pr-20 lg:pr-5 flex flex-wrap items-center justify-between gap-2">
        <div><h1 className="text-lg font-semibold">Striatica <span className="text-cyan-300">/ no-UMAP pilot</span></h1><p className="text-xs text-gray-400">Real multilayer graphs · shared decoder layout · not the production atlas</p></div>
        <a href="/" className="text-sm text-gray-400 hover:text-white">Return to atlas →</a>
      </header>
      <p className="px-5 py-3 text-sm bg-amber-950/30 text-amber-200 border-b border-amber-900/50" role="note">
        <strong>{retained}% neighbour retention</strong> · Spectral {spectralRetention}% / PCA {pcaRetention}%. {lost}% lost with spectral. {worseLayers && <>Layers {worseLayers} are worse than PCA. </>}Proximity is not causality.
      </p>
      <div className="flex flex-1 min-h-0 flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        <aside className="w-full lg:w-60 shrink-0 p-4 space-y-5 border-r border-gray-800 lg:overflow-y-auto" aria-label="Pilot controls">
          <label className="block text-xs text-gray-400">Attribution graph<select aria-label="Attribution graph" className={`${selectStyle} mt-1`} value={graphIndex} onChange={(e) => changeGraph(Number(e.target.value))}>{data.graphs.map((g, i) => <option key={g.metadata.slug} value={i}>{g.metadata.slug.replace('gemma-fact-', '')}</option>)}</select></label>
          <label className="block text-xs text-gray-400">Shared 3D layout<select aria-label="Shared 3D layout" className={`${selectStyle} mt-1`} value={layout} onChange={(e) => setLayout(e.target.value as Layout)}><option value="spectral">Spectral candidate</option><option value="pca">PCA yardstick</option></select></label>
          <label className="block text-xs text-gray-400">Attribution edges<select aria-label="Attribution edges" className={`${selectStyle} mt-1`} value={scope} onChange={(e) => setScope(e.target.value)}><option value="incident">Selected instance only</option><option value="all">All graph edges</option></select></label>
          <button className={`${selectStyle} hover:bg-gray-800 cursor-pointer`} onClick={() => controls.current?.reset()}>Reset camera</button>
          <div className="text-xs space-y-2 text-gray-400" aria-label="Graph accounting">
            <p className="text-gray-100">{graph.nodes.length.toLocaleString()} node instances<br />{graph.edges.length.toLocaleString()} signed directed edges</p>
            <p>{rows.length} decoder anchors · {graph.nodes.length - specials.length} feature instances</p>
            <p>{specials.length} special nodes outside decoder space</p>
            <p>{data.keys.length} union features · {new Set(data.keys.map((k) => k[0])).size} layers<br />Normalized decoder rows · cosine k={data.report.neighboursExcludingSelf}</p>
            <p>Layer colours: cyan → violet (early → late). Shared features keep one position across graphs; instances remain distinct.</p>
            <p>Drag to orbit · scroll to zoom<br />Backtick toggles debug</p>
            <a className="text-cyan-300 hover:text-white" href={graphUrl(graph.metadata.model, graph.metadata.slug)} target="_blank" rel="noreferrer">Original graph on Neuronpedia ↗</a>
          </div>
        </aside>
        <section className="relative flex-1 min-h-[360px]" aria-label="Shared decoder space">
          <Canvas frameloop="demand" camera={{ position: STYLE.cameraPosition, fov: CAMERA.fov }} onCreated={({ raycaster }) => { raycaster.params.Points.threshold = STYLE.raycastThreshold }} style={{ background: STYLE.background }}>
            <OrbitControls ref={controls} makeDefault enableDamping={false} rotateSpeed={CAMERA.rotateSpeed} zoomSpeed={CAMERA.zoomSpeed} panSpeed={CAMERA.panSpeed} />
            <CameraTracker controlsRef={controls} />
            <DecoderPoints key={layout} positions={positions} rows={rows} keys={data.keys} selectedRow={selectedRow} neighbours={neighbours} onSelect={chooseRow} />
            <AttributionEdges edges={spatial} all={scope === 'all'} />
            {selectedRow !== null && <SelectionRing position={positions[selectedRow]} />}
          </Canvas>
          <div className="absolute bottom-3 left-4 right-4 pointer-events-none text-xs text-gray-400 bg-black/80 p-2 rounded" aria-label="Spatial edge accounting">
            <span style={{ color: STYLE.positiveColor }}>+ positive</span> · <span style={{ color: STYLE.negativeColor }}>− negative</span> · arrowheads point source → target<br />
            {spatial.length.toLocaleString()} spatial edges / {visible.length.toLocaleString()} in scope · {unpositioned} unpositioned · {coincident} coincident<br />
            Non-spatial connections remain in the instance inspector. No geometry-neighbour edges are drawn.
          </div>
        </section>
        <aside className="w-full lg:w-80 shrink-0 border-l border-gray-800 p-4 space-y-4 lg:overflow-y-auto" aria-label="Instance inspector">
          <h2 className="font-semibold text-sm">Instance inspector</h2>
          <p className="text-xs text-gray-400">Starts at the source of the largest absolute-weight edge between decoder anchors. Click an anchor or edge endpoint to inspect another instance.</p>
          {selected && <>
            <h3 className="text-cyan-200 text-sm" data-testid="selected-node">{nodeLabel(selected)}</h3>
            <p className="text-xs break-all text-gray-400" data-testid="selected-id">{selected.node_id}</p>
            {selectedRow === null ? <p className="text-sm text-amber-200">Special node: no decoder coordinates. Connections are preserved below.</p> : <>
              <label className="block text-xs text-gray-400">Token instance at this anchor<select aria-label="Token instance at this anchor" value={selected.node_id} className={`${selectStyle} mt-1`} onChange={(e) => setSelectedId(e.target.value)}>{instances.map((n) => <option key={n.node_id} value={n.node_id}>{nodeLabel(n)}</option>)}</select></label>
              <p className="text-xs text-gray-400">{instances.length} distinct instances at one anchor. This feature retains {data.retained[layout][selectedRow]}/{data.report.neighboursExcludingSelf} neighbours in {layout}.</p>
            </>}
          </>}
          <details><summary className="text-sm cursor-pointer hover:text-cyan-200">Special nodes ({specials.length})</summary><div className="max-h-44 overflow-y-auto mt-2 space-y-1">{specials.map((n) => <button className="block text-left text-xs text-gray-400 hover:text-white cursor-pointer" key={n.node_id} onClick={() => setSelectedId(n.node_id)}>{nodeLabel(n)}</button>)}</div></details>
          <h3 className="text-sm">Incident edges ({incident.length})</h3>
          <p className="text-xs text-gray-400">Original signed weights, including special endpoints. No merging or normalization. All {graph.edges.length.toLocaleString()} graph edges remain loaded.</p>
          <div className="max-h-72 overflow-auto"><table className="text-xs w-full" aria-label="Incident edge records"><thead className="sticky top-0 bg-black text-gray-400"><tr><th className="text-left">Source → target</th><th className="text-right">Weight</th></tr></thead><tbody>{incident.map((e, i) => <tr key={i} className="border-b border-gray-900"><td className="py-2 pr-2">{[e.source, e.target].map((id, j) => <span key={j}>{j === 1 && ' → '}<button title={id} className="text-left hover:text-white cursor-pointer" onClick={() => setSelectedId(id)}>{nodeLabel(nodeById.get(id)!)}</button></span>)}</td><td className="text-right font-mono" style={{ color: e.weight < 0 ? STYLE.negativeColor : STYLE.positiveColor }}>{e.weight}</td></tr>)}</tbody></table></div>
        </aside>
      </div>
    </main>
  )
}

function DecoderPoints({ positions, rows, keys, selectedRow, neighbours, onSelect }: {
  positions: Position[]; rows: number[]; keys: [number, number][]; selectedRow: number | null
  neighbours: Set<number | null>; onSelect: (row: number) => void
}) {
  const material = useRef<ShaderMaterial>(null)
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const attributes = useMemo(() => {
    const maxLayer = Math.max(...keys.map((k) => k[0]))
    const colors = rows.flatMap((row) => (row === selectedRow
      ? new Color(STYLE.selectedColor)
      : new Color().setHSL(STYLE.layerHueStart + STYLE.layerHueSpan * keys[row][0] / maxLayer, STYLE.layerSaturation, STYLE.layerLightness)).toArray())
    return {
      positions: new Float32Array(rows.flatMap((row) => positions[row])),
      colors: new Float32Array(colors),
      sizes: new Float32Array(rows.map((row) => row === selectedRow ? STYLE.selectedSize : STYLE.pointSize)),
      opacity: new Float32Array(rows.map((row) => neighbours.has(row) ? STYLE.pointOpacity : STYLE.contextOpacity)),
    }
  }, [positions, rows, keys, selectedRow, neighbours])
  const uniforms = useMemo(() => ({
    uFocusDist: { value: new Vector3(...STYLE.cameraPosition).length() },
    uSizeScale: { value: POINTS.sizeScale }, uSizeMin: { value: POINTS.sizeMin }, uSizeMax: { value: POINTS.sizeMax },
    uSoftRange: { value: DOF.softRange }, uMinFocusDist: { value: DOF.minFocusDist },
    uCrispEdge: { value: DOF.crispEdge }, uSoftEdge: { value: DOF.softEdge }, uDepthFadeFactor: { value: DOF.depthFadeFactor },
  }), [])
  useFrame(({ camera }) => {
    if (material.current) material.current.uniforms.uFocusDist.value = controls ? camera.position.distanceTo(controls.target) : camera.position.length()
  })
  return <points onClick={(e) => { e.stopPropagation(); if (e.index !== undefined) onSelect(rows[e.index]) }}>
    <bufferGeometry>
      <bufferAttribute attach="attributes-position" args={[attributes.positions, 3]} />
      <bufferAttribute attach="attributes-aColor" args={[attributes.colors, 3]} />
      <bufferAttribute attach="attributes-aSize" args={[attributes.sizes, 1]} />
      <bufferAttribute attach="attributes-aOpacity" args={[attributes.opacity, 1]} />
    </bufferGeometry>
    <shaderMaterial ref={material} vertexShader={vertexShader} fragmentShader={fragmentShader} uniforms={uniforms} transparent depthWrite={false} />
  </points>
}

function AttributionEdges({ edges, all }: { edges: { edge: Edge; from: Position; to: Position }[]; all: boolean }) {
  const geometry = useMemo(() => {
    const positions: number[] = [], colors: number[] = []
    for (const { edge, from, to } of edges) {
      const source = new Vector3(...from), target = new Vector3(...to)
      const direction = target.clone().sub(source).normalize()
      const axis = direction.x === 0 && direction.y === 0 ? new Vector3(1, 0, 0) : new Vector3(0, 0, 1)
      const side = new Vector3().crossVectors(direction, axis).normalize()
      const length = Math.min(STYLE.arrowLength, source.distanceTo(target) / 3)
      const back = target.clone().addScaledVector(direction, -length)
      const width = Math.min(STYLE.arrowWidth, length)
      positions.push(...from, ...to, ...back.clone().addScaledVector(side, width).toArray(), ...to, ...back.clone().addScaledVector(side, -width).toArray(), ...to)
      const color = new Color(edge.weight < 0 ? STYLE.negativeColor : STYLE.positiveColor)
      for (let i = 0; i < 6; i++) colors.push(...color.toArray())
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new Float32BufferAttribute(positions, 3))
    g.setAttribute('color', new Float32BufferAttribute(colors, 3))
    return g
  }, [edges])
  useEffect(() => () => geometry.dispose(), [geometry])
  return <lineSegments geometry={geometry}><lineBasicMaterial vertexColors transparent depthWrite={false} opacity={all ? STYLE.allEdgesOpacity : STYLE.edgeOpacity} /></lineSegments>
}
