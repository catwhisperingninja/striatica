// frontend/src/components/Canvas3D.tsx
import { useEffect, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useAppStore } from '../stores/useAppStore'
import { loadDataset, loadCircuitManifest, listDatasets } from '../utils/dataLoader'
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
  const setAvailableDatasets = useAppStore((s) => s.setAvailableDatasets)
  const dataset = useAppStore((s) => s.dataset)
  const viewMode = useAppStore((s) => s.viewMode)
  const controlsRef = useRef<OrbitControlsImpl>(null)

  useEffect(() => {
    setLoading(true)

    // Determine which dataset file to load (URL param > default)
    const params = new URLSearchParams(window.location.search)
    const datasetFile = params.get('dataset')
    const defaultFile = DATASET.defaultPath.replace(/^\/data\//, '')
    const targetFile = datasetFile || defaultFile

    // Discover available datasets and load the target
    listDatasets().then((entries) => {
      setAvailableDatasets(entries)
      // If the target exists in the manifest, use it; otherwise fall back to first available
      const match = entries.find((e) => e.file === targetFile) ?? entries[0]
      const fileToLoad = match?.file ?? targetFile
      useAppStore.setState({ currentDatasetFile: fileToLoad })
      return loadDataset(`/data/${fileToLoad}`)
    })
      .then(setDataset)
      .catch((e: Error) => setError(e.message))

    // Only load GPT-2 circuits when viewing GPT-2 data — circuit feature indices
    // are model-specific and would display wrong labels on other datasets
    const isGpt2 = targetFile.startsWith('gpt2-small')
    if (isGpt2) {
      loadCircuitManifest()
        .then((manifest) => {
          setCircuitManifest(manifest)
          buildCircuitMembership()
        })
        .catch((e: Error) => console.warn('Circuit manifest load failed:', e.message))
    }
  }, [setDataset, setLoading, setError, setCircuitManifest, buildCircuitMembership, setAvailableDatasets])

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
