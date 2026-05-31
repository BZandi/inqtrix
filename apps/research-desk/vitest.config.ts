import path from 'node:path'

import { defineConfig } from 'vitest/config'

// Pure-logic unit tests (node environment, no DOM). Mirrors the `@` alias from
// vite.config.ts so specs can import application modules the same way the app
// does. Component rendering is verified in the browser (preview), not here.
export default defineConfig({
  resolve: {
    alias: [
      {
        find: '@',
        replacement: path.resolve(__dirname, './src'),
      },
    ],
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
})
