import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const apiProxyTarget = process.env.VITE_INQTRIX_API_BASE_URL || 'http://localhost:5100'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
