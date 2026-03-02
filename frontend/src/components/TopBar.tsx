// frontend/src/components/TopBar.tsx
import { useAppStore } from '../stores/useAppStore'
import type { ViewMode } from '../types/feature'

const VIEW_TABS: { label: string; mode: ViewMode; ready: boolean }[] = [
  { label: 'Point Cloud', mode: 'pointCloud', ready: true },
  { label: 'Circuits', mode: 'circuits', ready: true },
  { label: 'Local Dim', mode: 'localDim', ready: false },
]

export default function TopBar() {
  const viewMode = useAppStore((s) => s.viewMode)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const resetSelection = useAppStore((s) => s.resetSelection)
  const colorMode = useAppStore((s) => s.colorMode)
  const setColorMode = useAppStore((s) => s.setColorMode)
  const hasLocalDim = useAppStore((s) => !!(s.dataset?.localDimensions?.length))

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-[--color-panel] border-b border-[--color-panel-border] backdrop-blur-xl z-10">
      <span className="text-sm font-bold tracking-wide text-[--color-cluster-0]">
        striatica
      </span>
      <div className="w-px h-5 bg-[--color-panel-border]" />

      <select disabled title="Multi-dataset support coming soon" className="bg-gray-900 text-gray-500 border border-[--color-panel-border] rounded-md px-2 py-1 text-xs cursor-not-allowed opacity-60">
        <option>GPT2-small</option>
      </select>

      <select disabled title="Multi-dataset support coming soon" className="bg-gray-900 text-gray-500 border border-[--color-panel-border] rounded-md px-2 py-1 text-xs cursor-not-allowed opacity-60">
        <option>Layer 6 (residual)</option>
      </select>

      <div className="flex items-center gap-1 text-xs">
        <span className="text-zinc-500">Color:</span>
        <button
          onClick={() => setColorMode('cluster')}
          className={`px-2 py-0.5 rounded cursor-pointer ${
            colorMode === 'cluster'
              ? 'bg-[#A3D739]/20 text-[#A3D739]'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Cluster
        </button>
        <button
          onClick={() => setColorMode('localDim')}
          disabled={!hasLocalDim}
          title={!hasLocalDim ? 'Local dimension data not loaded — run pipeline with VGT estimation' : 'Color by local intrinsic dimension'}
          className={`px-2 py-0.5 rounded cursor-pointer ${
            colorMode === 'localDim'
              ? 'bg-[#A3D739]/20 text-[#A3D739]'
              : 'text-zinc-400 hover:text-zinc-200'
          } disabled:opacity-30 disabled:cursor-not-allowed`}
        >
          Local Dim
        </button>
      </div>

      <button
        onClick={resetSelection}
        className="px-2 py-0.5 text-[11px] text-gray-600 hover:text-gray-300 cursor-pointer transition-colors"
        title="Clear all selections"
      >
        Reset
      </button>

      <div className="flex gap-0.5 ml-auto">
        {VIEW_TABS.map((tab) => {
          const isActive = viewMode === tab.mode
          return (
            <button
              key={tab.mode}
              disabled={!tab.ready}
              onClick={() => tab.ready && setViewMode(tab.mode)}
              className={`px-3 py-1 text-[11px] rounded transition-colors ${
                isActive
                  ? 'text-[--color-cluster-0] bg-[#0e2a31] cursor-pointer'
                  : tab.ready
                    ? 'text-gray-500 hover:text-gray-200 cursor-pointer'
                    : 'text-gray-700 cursor-not-allowed opacity-40'
              }`}
              title={!tab.ready ? 'Coming soon' : `${tab.label} (⌘P to toggle)`}
            >
              {tab.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
