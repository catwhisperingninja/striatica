// frontend/vite.config.ts
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import path from 'node:path'

/**
 * Vite plugin: serves GET /data/datasets.json by scanning public/data/ for
 * dataset JSON files. Each file is read just enough to extract model + layer
 * from the top-level keys, so the TopBar dropdown can be populated without
 * loading multi-MB dataset files.
 */
function datasetManifestPlugin(): Plugin {
  return {
    name: 'dataset-manifest',
    configureServer(server) {
      server.middlewares.use('/data/datasets.json', (_req, res) => {
        const dataDir = path.resolve(__dirname, 'public/data')
        const entries: { file: string; model: string; layer: string; numFeatures: number }[] = []
        try {
          for (const name of fs.readdirSync(dataDir)) {
            if (!name.endsWith('.json') || name === 'datasets.json') continue
            // Skip subdirectories (e.g. circuits/)
            const fullPath = path.join(dataDir, name)
            if (!fs.statSync(fullPath).isFile()) continue
            try {
              // Read just the first 1KB to parse top-level fields
              const fd = fs.openSync(fullPath, 'r')
              const buf = Buffer.alloc(1024)
              fs.readSync(fd, buf, 0, 1024, 0)
              fs.closeSync(fd)
              const snippet = buf.toString('utf8')
              const modelMatch = snippet.match(/"model"\s*:\s*"([^"]+)"/)
              const layerMatch = snippet.match(/"layer"\s*:\s*"([^"]+)"/)
              const numMatch = snippet.match(/"numFeatures"\s*:\s*(\d+)/)
              entries.push({
                file: name,
                model: modelMatch?.[1] ?? name.replace('.json', ''),
                layer: layerMatch?.[1] ?? 'unknown',
                numFeatures: numMatch ? parseInt(numMatch[1], 10) : 0,
              })
            } catch {
              // Skip unreadable files
            }
          }
        } catch {
          // dataDir doesn't exist yet
        }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(entries))
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), datasetManifestPlugin()],
  server: { port: 5173 },
})
