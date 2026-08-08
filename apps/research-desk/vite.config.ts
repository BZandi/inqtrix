import { brotliCompressSync, constants, gzipSync } from 'node:zlib'
import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const apiProxyTarget = process.env.VITE_INQTRIX_API_BASE_URL || 'http://localhost:5100'

// Below this size the encoding headers and a second cache entry cost
// more than the saved bytes.
const COMPRESSIBLE_MIN_BYTES = 1_024
const COMPRESSIBLE = /\.(?:js|mjs|css|svg|json|map|txt|wasm)$/

/** Emit `.br`/`.gz` siblings for static assets at BUILD time.
 *
 * The gateway serves these as-is; it never compresses per request,
 * because squeezing a multi-MB bundle on every cache miss costs more
 * than the smaller payload saves. Already-compressed formats (png,
 * woff2, …) are skipped — recompressing them only adds bytes.
 */
function precompressAssets(): Plugin {
  return {
    apply: 'build',
    generateBundle(_options, bundle) {
      for (const [fileName, output] of Object.entries(bundle)) {
        if (!COMPRESSIBLE.test(fileName)) continue
        const source = output.type === 'asset' ? output.source : output.code
        const raw = typeof source === 'string' ? Buffer.from(source) : Buffer.from(source)
        if (raw.byteLength < COMPRESSIBLE_MIN_BYTES) continue
        this.emitFile({
          fileName: `${fileName}.gz`,
          source: gzipSync(raw, { level: 9 }),
          type: 'asset',
        })
        this.emitFile({
          fileName: `${fileName}.br`,
          source: brotliCompressSync(raw, {
            params: {
              [constants.BROTLI_PARAM_QUALITY]: 11,
              [constants.BROTLI_PARAM_SIZE_HINT]: raw.byteLength,
            },
          }),
          type: 'asset',
        })
      }
    },
    name: 'inqtrix-precompress-assets',
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), precompressAssets()],
  resolve: {
    alias: [
      {
        find: '@',
        replacement: path.resolve(__dirname, './src'),
      },
    ],
  },
  server: {
    port: 5173,
    proxy: {
      '/collaboration': {
        target: apiProxyTarget,
        ws: true,
      },
      '/health': apiProxyTarget,
      '/v1': apiProxyTarget,
      '/api': apiProxyTarget,
    },
  },
})
