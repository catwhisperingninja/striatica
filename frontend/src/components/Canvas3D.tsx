// frontend/src/components/Canvas3D.tsx
import { useEffect, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useAppStore } from '../stores/useAppStore'
import { loadDataset, loadCircuitManifest } from '../utils/dataLoader'
import { CAMERA, DATASET } from '../config/rendering'
import PointCloudView from '../views/PointCloudView'
import CircuitGraphView from '../views/CircuitGraphView'
import FlyToCamera from '../three/FlyToCamera'
import CameraTracker from '../three/CameraTracker'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

export default function Canvas3D() {
  const setDataset = useAppStore((s) => s.setDataset)
  const setLoading = useAppStore((s) => s.setLoading)
  const setError = useAppStore((s) => s.setError)
  const setCircuitManifest = useAppStore((s) => s.setCircuitManifest)
  const buildCircuitMembership = useAppStore((s) => s.buildCircuitMembership)
  const dataset = useAppStore((s) => s.dataset)
  const viewMode = useAppStore((s) => s.viewMode)
  const controlsRef = useRef<OrbitControlsImpl>(null)

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams(window.location.search)
    const datasetFile = params.get('dataset')
    const datasetPath = datasetFile ? `/data/${datasetFile}` : DATASET.defaultPath
    loadDataset(datasetPath)
      .then(setDataset)
      .catch((e: Error) => setError(e.message))
    loadCircuitManifest()
      .then((manifest) => {
        setCircuitManifest(manifest)
        buildCircuitMembership()
      })
      .catch((e: Error) => console.warn('Circuit manifest load failed:', e.message))
  }, [setDataset, setLoading, setError, setCircuitManifest, buildCircuitMembership])

  return (
    <Canvas
      camera={{ position: CAMERA.defaultPosition, fov: CAMERA.fov }}
      gl={{ antialias: false, alpha: false }}
      style={{ background: '#000' }}
    >
      {/* No damping: snappy linear stop on mouse release (AE linear keyframe style) */}
      <OrbitControls
        ref={controlsRef}
        enableDamping={false}
        rotateSpeed={CAMERA.rotateSpeed}
        zoomSpeed={CAMERA.zoomSpeed}
        panSpeed={CAMERA.panSpeed}
      />
      {dataset && viewMode === 'pointCloud' && <PointCloudView />}
      {dataset && viewMode === 'circuits' && <CircuitGraphView />}
      <FlyToCamera controlsRef={controlsRef} />
      <CameraTracker controlsRef={controlsRef} />
    </Canvas>
  )
}
