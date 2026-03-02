// frontend/src/App.tsx
import { useEffect } from 'react'
import TopBar from './components/TopBar'
import NavPanel from './components/NavPanel'
import CircuitPanel from './components/CircuitPanel'
import Canvas3D from './components/Canvas3D'
import DetailPanel from './components/DetailPanel'
import StatusBar from './components/StatusBar'
import Minimap from './three/Minimap'
import DebugConsole from './components/DebugConsole'
import { useAppStore } from './stores/useAppStore'

export default function App() {
  const viewMode = useAppStore((s) => s.viewMode)
  const setViewMode = useAppStore((s) => s.setViewMode)

  // ⌘P (Mac) / Ctrl+P (Win/Linux) — toggle between Points and Circuits
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'p') {
        e.preventDefault() // prevent browser print dialog
        const current = useAppStore.getState().viewMode
        setViewMode(current === 'pointCloud' ? 'circuits' : 'pointCloud')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [setViewMode])

  return (
    <div className="h-screen flex flex-col overflow-hidden relative">
      <DebugConsole />
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        {viewMode === 'circuits' ? <CircuitPanel /> : <NavPanel />}

        {/* 3D Canvas area */}
        <div className="flex-1 relative bg-black">
          <Canvas3D />
          <Minimap />
        </div>

        {/* Detail Panel */}
        <DetailPanel />
      </div>

      <StatusBar />
    </div>
  )
}
