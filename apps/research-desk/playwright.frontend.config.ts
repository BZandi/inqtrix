import path from 'node:path'
import { tmpdir } from 'node:os'

import { defineConfig } from '@playwright/test'

const port = 5187
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH

export default defineConfig({
  fullyParallel: false,
  outputDir: path.join(tmpdir(), 'inqtrix-research-desk-playwright'),
  testDir: './browser-tests',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
    launchOptions: executablePath ? { executablePath } : undefined,
  },
  webServer: {
    command: `pnpm exec vite --host 127.0.0.1 --port ${port} --strictPort`,
    reuseExistingServer: true,
    timeout: 120_000,
    url: `http://127.0.0.1:${port}/browser-tests/fixtures/editor-lifecycle.html`,
  },
})
