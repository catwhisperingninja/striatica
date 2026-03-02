import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useAppStore } from '../stores/useAppStore'
import { cameraSync } from '../utils/cameraSync'
import { MINIMAP, COLORS } from '../config/rendering'

// ── Minimap-internal components (rendered inside the secondary Canvas) ──

/** Point the orthographic camera at the origin on mount. */
function LookAtOrigin() {
  const { camera } = useThree()
  useMemo(() => { camera.lookAt(0, 0, 0) }, [camera])
  return null
}

/** Downsampled point cloud for spatial context. */
function MinimapPoints() {
  const dataset = useAppStore((s) => s.dataset)

  const geometry = useMemo(() => {
    if (!dataset) return null
    const { positions, clusterLabels } = dataset
    const n = dataset.numFeatures
    const step = Math.max(1, Math.floor(n / MINIMAP.maxPoints))

    const CLUSTER_COLS = COLORS.clusters.map((hex) => new THREE.Color(hex))
    const NOISE = new THREE.Color('#374151')

    const pts: number[] = []
    const cols: number[] = []
    for (let i = 0; i < n; i += step) {
      pts.push(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
      const c = clusterLabels[i] < 0 ? NOISE : CLUSTER_COLS[clusterLabels[i] % CLUSTER_COLS.length]
      cols.push(c.r, c.g, c.b)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3))
    return geo
  }, [dataset])

  if (!geometry) return null
  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={MINIMAP.pointSize}
        vertexColors
        transparent
        opacity={MINIMAP.pointOpacity}
        sizeAttenuation={false}
        depthTest={false}
      />
    </points>
  )
}

/** Bright sphere at the currently selected feature. */
function SelectedMarker() {
  const selectedIndex = useAppStore((s) => s.selectedIndex)
  const dataset = useAppStore((s) => s.dataset)

  if (selectedIndex === null || !dataset) return null
  const i = selectedIndex * 3
  return (
    <mesh position={[dataset.positions[i], dataset.positions[i + 1], dataset.positions[i + 2]]}>
      <sphereGeometry args={[MINIMAP.selectedSize, 8, 8]} />
      <meshBasicMaterial color="#A3D739" transparent opacity={0.9} depthTest={false} />
    </mesh>
  )
}

/** RGB axis lines through the origin. */
function Axes() {
  const geometry = useMemo(() => {
    const s = MINIMAP.axisLength
    const positions = new Float32Array([
      -s, 0, 0, s, 0, 0,
      0, -s, 0, 0, s, 0,
      0, 0, -s, 0, 0, s,
    ])
    const colors = new Float32Array([
      1, 0.3, 0.3, 1, 0.3, 0.3,
      0.3, 1, 0.3, 0.3, 1, 0.3,
      0.3, 0.3, 1, 0.3, 0.3, 1,
    ])
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    return geo
  }, [])

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial vertexColors transparent opacity={MINIMAP.axisOpacity} depthTest={false} />
    </lineSegments>
  )
}

/** Wireframe camera frustum that tracks the main camera's pose every frame. */
function CameraFrustum() {
  const groupRef = useRef<THREE.Group>(null)

  // 12 edges × 2 endpoints = 24 vertices × 3 floats
  const { geometry, positionAttr } = useMemo(() => {
    const attr = new THREE.Float32BufferAttribute(new Float32Array(24 * 3), 3)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', attr)
    return { geometry: geo, positionAttr: attr }
  }, [])

  useFrame(() => {
    if (!groupRef.current) return

    const { position, target, fov, aspect } = cameraSync

    // Position frustum group at main camera location
    groupRef.current.position.set(position[0], position[1], position[2])

    // Direction to target
    const dx = target[0] - position[0]
    const dy = target[1] - position[1]
    const dz = target[2] - position[2]
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (dist < 0.001) return

    // Orient group to face the target (Three.js lookAt points -Z at target)
    groupRef.current.lookAt(target[0], target[1], target[2])

    // Compute frustum dimensions in local space (opening along -Z)
    const length = Math.max(MINIMAP.frustumMinLength, Math.min(dist * MINIMAP.frustumLengthFraction, MINIMAP.frustumMaxLength))
    const near = length * 0.04
    const far = length
    const halfFov = (fov / 2) * Math.PI / 180
    const nH = near * Math.tan(halfFov)
    const nW = nH * aspect
    const fH = far * Math.tan(halfFov)
    const fW = fH * aspect

    const v = positionAttr.array as Float32Array
    let idx = 0
    const edge = (ax: number, ay: number, az: number, bx: number, by: number, bz: number) => {
      v[idx++] = ax; v[idx++] = ay; v[idx++] = az
      v[idx++] = bx; v[idx++] = by; v[idx++] = bz
    }

    // Near plane rectangle (narrow end, at camera)
    edge(-nW, +nH, near, +nW, +nH, near)
    edge(+nW, +nH, near, +nW, -nH, near)
    edge(+nW, -nH, near, -nW, -nH, near)
    edge(-nW, -nH, near, -nW, +nH, near)
    // Far plane rectangle (wide end, facing the model)
    edge(-fW, +fH, far, +fW, +fH, far)
    edge(+fW, +fH, far, +fW, -fH, far)
    edge(+fW, -fH, far, -fW, -fH, far)
    edge(-fW, -fH, far, -fW, +fH, far)
    // Connecting edges (near → far)
    edge(-nW, +nH, near, -fW, +fH, far)
    edge(+nW, +nH, near, +fW, +fH, far)
    edge(+nW, -nH, near, +fW, -fH, far)
    edge(-nW, -nH, near, -fW, -fH, far)

    positionAttr.needsUpdate = true
  })

  return (
    <group ref={groupRef}>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial color={MINIMAP.frustumColor} transparent opacity={MINIMAP.frustumOpacity} depthTest={false} />
      </lineSegments>
    </group>
  )
}

/** Small dot at the orbit controls target (where the camera is looking). */
function TargetDot() {
  const ref = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (!ref.current) return
    ref.current.position.set(cameraSync.target[0], cameraSync.target[1], cameraSync.target[2])
  })

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.015, 6, 6]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.4} depthTest={false} />
    </mesh>
  )
}

/** Broadcasts camera position from cameraSync to a React callback at ~10fps. */
function CameraPositionBroadcaster({ onUpdate }: { onUpdate: (x: number, y: number, z: number) => void }) {
  const frameCount = useRef(0)
  useFrame(() => {
    // Throttle to every 6th frame (~10fps at 60fps) to avoid excessive re-renders
    if (++frameCount.current % 6 !== 0) return
    onUpdate(cameraSync.position[0], cameraSync.position[1], cameraSync.position[2])
  })
  return null
}

// ── Main Minimap Component ──────────────────────────────────────────

export default function Minimap() {
  const dataset = useAppStore((s) => s.dataset)
  const [camPos, setCamPos] = useState<[number, number, number]>([0, 0, 0])

  const handleCamUpdate = useRef((x: number, y: number, z: number) => {
    setCamPos([x, y, z])
  }).current

  if (!dataset) return null

  return (
    <div className="absolute bottom-4 right-4 w-[160px] h-[120px] bg-[--color-panel] border border-[--color-panel-border] rounded-lg backdrop-blur-lg overflow-hidden pointer-events-none">
      {/* XYZ axis legend — upper right */}
      <div className="absolute top-1 right-1.5 z-10 font-mono text-[8px] leading-[11px] select-none">
        <div className="flex items-center gap-1">
          <span className="font-bold" style={{ color: '#ff4d4d' }}>X</span>
          <span className="text-gray-400">{camPos[0].toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="font-bold" style={{ color: '#4dff4d' }}>Y</span>
          <span className="text-gray-400">{camPos[1].toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="font-bold" style={{ color: '#4d4dff' }}>Z</span>
          <span className="text-gray-400">{camPos[2].toFixed(2)}</span>
        </div>
      </div>
      <Canvas
        orthographic
        camera={{
          position: MINIMAP.cameraPosition,
          zoom: MINIMAP.cameraZoom,
          near: 0.1,
          far: 20,
        }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <LookAtOrigin />
        <MinimapPoints />
        <Axes />
        <CameraFrustum />
        <TargetDot />
        <SelectedMarker />
        <CameraPositionBroadcaster onUpdate={handleCamUpdate} />
      </Canvas>
    </div>
  )
}
