import { defineConfig, devices, type Project } from '@playwright/test'

import {
  COLLABORATION_TRANSPORTS,
  collaborationE2EConfiguration,
  type CollaborationTransport,
} from './e2e/config'
import {
  MOBILE_ONLY_SCENARIO_TAG,
  MOBILE_SCENARIO_TAG,
} from './e2e/release-contract'

const projects = COLLABORATION_TRANSPORTS.flatMap((transport) => [
  project(transport, 'desktop'),
  project(transport, 'mobile'),
])

export default defineConfig({
  expect: { timeout: 15_000 },
  forbidOnly: Boolean(process.env.CI) || collaborationE2EConfiguration.mode === 'release',
  fullyParallel: false,
  outputDir: 'e2e/.results',
  projects,
  reporter: collaborationE2EConfiguration.mode === 'release'
    ? [
        ['list', { printSteps: true }],
        ['./e2e/release-reporter.ts'],
      ]
    : [['list', { printSteps: true }]],
  retries: 0,
  testDir: './e2e',
  testMatch: 'collaboration.spec.ts',
  timeout: 60_000,
  workers: 1,
})

function project(
  transport: CollaborationTransport,
  formFactor: 'desktop' | 'mobile',
): Project {
  const endpoint = collaborationE2EConfiguration.stack?.transports[transport].baseURL
  const device = formFactor === 'desktop' ? devices['Desktop Chrome'] : devices['Pixel 7']
  return {
    grep: formFactor === 'mobile'
      ? new RegExp(`${MOBILE_SCENARIO_TAG}|@layout`)
      : undefined,
    grepInvert: formFactor === 'desktop'
      ? new RegExp(MOBILE_ONLY_SCENARIO_TAG)
      : undefined,
    metadata: { formFactor, transport },
    name: `${transport}-${formFactor}`,
    use: {
      ...device,
      baseURL: endpoint ?? 'http://127.0.0.1:9',
      screenshot: 'only-on-failure',
      trace: 'off',
      video: 'off',
    },
  }
}
