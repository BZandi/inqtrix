import type { VerificationAdapter } from '../adapter.ts'
import { runCommand } from '../command-runner.ts'
import {
  environmentCheck,
  executableCheck,
  failed,
  passed,
  repositoryFileCheck,
} from '../preflight.ts'
import { playwrightPreflight } from './shared.ts'

const ENGINE = 'agent-desk-live' as const

export const agentDeskLiveAdapter: VerificationAdapter = {
  engine: ENGINE,
  profiles: ['agent-desk'],
  async preflight(context) {
    const baseURL = context.environment.INQTRIX_E2E_BASE_URL
      ?? 'http://127.0.0.1:8080'
    let baseURLCheck
    let origin: string | null = null
    try {
      const parsed = new URL(baseURL)
      const valid = ['http:', 'https:'].includes(parsed.protocol)
        && !parsed.username
        && !parsed.password
        && !parsed.search
        && !parsed.hash
      origin = valid ? parsed.origin : null
      baseURLCheck = valid
        ? passed(ENGINE, 'base-url', 'The system base URL is structurally valid.')
        : failed(ENGINE, 'base-url', 'The system base URL must be a credential-free HTTP(S) URL.')
    } catch {
      baseURLCheck = failed(ENGINE, 'base-url', 'The system base URL is invalid.')
    }
    // Live stack health: a strict live profile never starts product
    // services itself — an unreachable or kernel-less stack is BLOCKED,
    // not a failing run. An auth-gated capability manifest still proves
    // reachability; the engine re-asserts the flag after login.
    let capabilityCheck
    if (!origin) {
      capabilityCheck = failed(
        ENGINE,
        'agent-kernel-capability',
        'Capability manifest not checked: invalid base URL.',
      )
    } else {
      try {
        const response = await fetch(`${origin}/v1/capabilities`, {
          signal: AbortSignal.timeout(10_000),
        })
        if (response.status === 401 || response.status === 403) {
          capabilityCheck = passed(
            ENGINE,
            'agent-kernel-capability',
            'The stack is reachable; the manifest is auth-gated and is re-checked in-engine after login.',
          )
        } else if (response.ok) {
          const manifest = await response.json() as {
            features?: { agent_kernel?: unknown }
          }
          capabilityCheck = manifest.features?.agent_kernel === true
            ? passed(
                ENGINE,
                'agent-kernel-capability',
                'The stack publishes features.agent_kernel = true.',
              )
            : failed(
                ENGINE,
                'agent-kernel-capability',
                'The stack does not publish features.agent_kernel = true.',
              )
        } else {
          capabilityCheck = failed(
            ENGINE,
            'agent-kernel-capability',
            `The capability manifest returned HTTP ${response.status}.`,
          )
        }
      } catch {
        capabilityCheck = failed(
          ENGINE,
          'agent-kernel-capability',
          'The live stack is unreachable — start the canonical stack first.',
        )
      }
    }
    const executable = context.environment.PLAYWRIGHT_EXECUTABLE_PATH
    // Deliberately tester-only: the engine drives ONE authenticated
    // account and cleans its own resources through owner APIs — no
    // admin credential is ever used (data minimization).
    return [
      repositoryFileCheck(
        context,
        ENGINE,
        'engine-entrypoint',
        'e2e/engines/agent-desk-live.mjs',
      ),
      ...playwrightPreflight(ENGINE, executable ? [] : ['chromium']),
      executableCheck(context, ENGINE, 'browser-executable', executable),
      environmentCheck(context, ENGINE, 'tester-email', 'INQTRIX_E2E_TESTER_EMAIL'),
      environmentCheck(context, ENGINE, 'user-password', 'INQTRIX_E2E_USER_PASSWORD'),
      baseURLCheck,
      capabilityCheck,
    ]
  },
  async execute(context, cleanupLedger) {
    // The engine writes the per-scenario sidecar file itself; runCommand
    // reads it back, so no scenarios are synthesized here.
    return await runCommand(context, cleanupLedger, {
      args: ['e2e/engines/agent-desk-live.mjs'],
      command: process.execPath,
      engine: ENGINE,
      productLifecycle: true,
    })
  },
}
