import { lazy, StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

const GraphPilot = lazy(() => import('./views/TracedGraphPilot'))
const showPilot = import.meta.env.DEV && new URLSearchParams(window.location.search).get('pilot') === 'traced'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Suspense fallback={<p>Loading viewer…</p>}>
      {showPilot ? <GraphPilot /> : <App />}
    </Suspense>
  </StrictMode>,
)
