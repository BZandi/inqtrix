// Agent-Desk live engine (profile: agent-desk).
//
// Drives ONE kernel run through the VISIBLE Agent Desk against the
// already-running canonical stack: submit a run-prefixed question,
// watch the live tool activity, assert the clickable-citation answer
// experience, collect console/network hygiene, and leave zero
// residual resources. Scenario results are written to the orchestrator
// sidecar file (INQTRIX_VERIFICATION_SCENARIO_RESULTS_PATH); resources
// register through the shared product-lifecycle IPC channel.

import { chmod, mkdir, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { chromium } from '@playwright/test'

import {
  assertFixture as assert,
  fetchActorJson as fetchJson,
} from '../../tests/verification/fixtures/api.mjs'
import {
  agentSessionTitleForRun,
  assertVerificationRunId,
} from '../../tests/verification/fixtures/run-scope.mjs'
import {
  createSessionFixtures,
} from '../../tests/verification/fixtures/sessions.mjs'
import {
  VerificationLifecycleClient,
} from '../../tests/verification/fixtures/lifecycle-client.mjs'

const baseURL = process.env.INQTRIX_E2E_BASE_URL ?? 'http://127.0.0.1:8080'
const ignoreHTTPSErrors =
  process.env.INQTRIX_E2E_IGNORE_HTTPS_ERRORS === '1'
const testerEmail = requiredEnvironment('INQTRIX_E2E_TESTER_EMAIL')
const userPassword = requiredEnvironment('INQTRIX_E2E_USER_PASSWORD')
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH
const runKey = assertVerificationRunId(
  requiredEnvironment('INQTRIX_VERIFICATION_RUN_ID'),
)
const reportDirectory = requiredEnvironment('INQTRIX_VERIFICATION_REPORT_DIR')
const screenshotDirectory = join(reportDirectory, 'screenshots')

const lifecycle = new VerificationLifecycleClient({
  reportDirectory,
  runId: runKey,
})

const SCENARIOS = [
  'agent.kernel-run-submits',
  'agent.tool-activity-visible',
  'agent.answer-citations-clickable',
  'agent.console-network-clean',
  'agent.cleanup-integrity',
]
const scenarioStatus = new Map(SCENARIOS.map((id) => [id, 'failed']))
const networkFindings = []

let browser
let tester
let runId = null
let exitCode = 1

try {
  progress('Browser starten')
  await mkdir(screenshotDirectory, { recursive: true })
  browser = await chromium.launch({ executablePath, headless: true })
  const sessionFixtures = createSessionFixtures({
    baseURL,
    browser,
    ignoreHTTPSErrors,
    lifecycle,
    parseCollaborationFrame: () => null,
    runId: runKey,
    screenshotDirectory,
  })
  tester = await sessionFixtures.loginActor(
    testerEmail,
    userPassword,
    'AgentTester',
    'user',
  )
  // Network hygiene on top of the shared pageerror/console collectors.
  tester.page.on('requestfailed', (request) => {
    const failure = request.failure()?.errorText ?? 'failed'
    if (failure.includes('net::ERR_ABORTED')) return
    networkFindings.push(`requestfailed:${failure}`)
  })
  tester.page.on('response', (response) => {
    if (response.status() >= 500) {
      networkFindings.push(`http:${response.status()}`)
    }
  })

  const capabilities = await fetchJson(tester, 'GET', '/v1/capabilities')
  assert(
    capabilities.features?.agent_kernel === true,
    'The stack does not enable the agent kernel (features.agent_kernel).',
  )
  // The composer submits the published default engine — this profile
  // verifies the KERNEL answer experience, so a workspace_agent default
  // would silently test the wrong engine.
  assert(
    capabilities.agent?.default_mode === 'agent_kernel',
    'The stack default engine is not agent_kernel '
      + `(${capabilities.agent?.default_mode ?? 'unset'}).`,
  )

  progress('Agent Desk oeffnen')
  await tester.page.goto('/', { waitUntil: 'domcontentloaded' })
  await tester.page.getByRole('button', { name: 'Agent Desk' }).click()
  const composer = tester.page.getByTestId('agent-composer-input')
  await composer.waitFor({ state: 'visible', timeout: 20_000 })

  // Auto autonomy: the ONE web search must execute without a manual
  // approval click (the gate itself has its own scenario elsewhere).
  await tester.page
    .getByRole('button', { name: 'Ausführung', exact: true })
    .click()
  await tester.page
    .getByRole('menuitem')
    .filter({ hasText: 'Zwischenfreigaben' })
    .first()
    .click()
  await tester.page.keyboard.press('Escape')

  const question = `${agentSessionTitleForRun(runKey)}: Was ist heute der `
    + 'aktuelle Stand der EU-KI-Regulierung? Antworte mit Quellen.'
  await composer.fill(question)

  progress('Kernel-Lauf einreichen')
  const submitResponse = tester.page.waitForResponse(
    (response) =>
      response.request().method() === 'POST'
      && new URL(response.url()).pathname === '/v1/runs',
    { timeout: 30_000 },
  )
  await tester.page.getByTestId('agent-submit').click()
  const submitted = await submitResponse
  assert(
    submitted.status() === 202 || submitted.status() === 200,
    `Run submission returned HTTP ${submitted.status()}.`,
  )
  runId = (await submitted.json()).run_id
  assert(typeof runId === 'string' && runId.length > 0, 'No run id returned.')

  // Register the run IMMEDIATELY after creation (§15): the run-bound
  // session title is deterministic from our own question, so no server
  // round-trip may delay the ledger entry. LIFO cleanup then deletes
  // runs before the session; every delete is 404-tolerant.
  const deterministicTitle = question.trim().slice(0, 80)
  await lifecycle.register({
    credential: 'user',
    id: runId,
    kind: 'agent_run',
    ownerEmail: testerEmail,
    sessionTitle: deterministicTitle,
  })
  const session = await waitFor(
    async () => {
      const listing = await fetchJson(tester, 'GET', '/v1/agent-sessions')
      return listing.data?.find(
        (candidate) =>
          typeof candidate.title === 'string'
          && candidate.title.startsWith(`${runKey} `),
      ) ?? null
    },
    30_000,
    'run-titled agent session in the sync store',
  )
  await lifecycle.register({
    credential: 'user',
    id: session.id,
    kind: 'agent_session',
    ownerEmail: testerEmail,
    title: session.title,
  })
  scenarioStatus.set('agent.kernel-run-submits', 'passed')
  progress('Lauf laeuft — Aktivitaet beobachten')

  // Live tool activity: at least one activity row must appear WHILE the
  // run is active and carry a non-empty detail (the literal query).
  let activitySeen = false
  const activityProbe = (async () => {
    const row = tester.page.getByTestId('agent-activity-item').first()
    try {
      await row.waitFor({ state: 'visible', timeout: 180_000 })
      const text = ((await row.textContent()) ?? '').trim()
      activitySeen = text.length > 0
    } catch {
      activitySeen = false
    }
  })()

  const terminal = await waitFor(
    async () => {
      const summary = await fetchJson(tester, 'GET', `/v1/runs/${runId}`)
      return ['completed', 'failed', 'cancelled'].includes(summary.status)
        ? summary
        : null
    },
    600_000,
    'terminal kernel run state',
  )
  await activityProbe
  // A fanned-out run leaves child runs behind: register every child for
  // cleanup too (LIFO deletes them before the root and the session).
  const children = await fetchJson(
    tester,
    'GET',
    `/v1/runs/${runId}/children`,
  ).catch(() => null)
  for (const child of children?.data ?? []) {
    if (typeof child.run_id === 'string' && child.run_id) {
      await lifecycle.register({
        credential: 'user',
        id: child.run_id,
        kind: 'agent_run',
        ownerEmail: testerEmail,
        sessionTitle: deterministicTitle,
      })
    }
  }
  assert(
    terminal.status === 'completed',
    `The kernel run ended ${terminal.status}: ${terminal.error ?? ''}`,
  )
  // The run and every owned resource are terminal and registered before
  // answer presentation is inspected. A later UI assertion must not turn
  // this independently proven cleanup contract red.
  scenarioStatus.set('agent.cleanup-integrity', 'passed')
  if (activitySeen) {
    scenarioStatus.set('agent.tool-activity-visible', 'passed')
  }

  progress('Antwort-Erlebnis pruefen')
  const answer = tester.page.getByTestId('agent-answer')
  await answer.waitFor({ state: 'visible', timeout: 60_000 })
  // The answer container exists from answer.started onwards. References
  // arrive with the canonical artifact-detail reconciliation after
  // answer.ready, so the source owner — not the already-visible container —
  // is the anchored readiness signal for citation assertions.
  const sources = tester.page.getByTestId('agent-sources')
  await sources.waitFor({ state: 'visible', timeout: 30_000 })
  const citationAnchors = answer.locator('a[href^="#kref-"]')
  const citationCount = await citationAnchors.count()
  assert(citationCount > 0, 'The answer rendered no citation chips.')
  await tester.page.screenshot({
    fullPage: true,
    path: join(screenshotDirectory, 'agent-answer-ready.png'),
  })
  // Clicking a citation must react visibly: a web ref opens a popup, a
  // knowledge ref opens the evidence view (canvas panel).
  const popupPromise = tester.context
    .waitForEvent('page', { timeout: 5_000 })
    .catch(() => null)
  await citationAnchors.first().click()
  const popup = await popupPromise
  if (popup) {
    await popup.close()
  } else {
    await tester.page
      .getByText(/Beleg [KW]\d+/)
      .first()
      .waitFor({ state: 'visible', timeout: 10_000 })
  }
  scenarioStatus.set('agent.answer-citations-clickable', 'passed')

} catch (error) {
  progress(`Engine-Fehler: ${sanitize(String(error?.message ?? error))}`)
  exitCode = 1
} finally {
  try {
    if (runId && tester) {
      // Never leave a live run behind: cancel and await the terminal
      // state so the ledger's DELETE cannot race a running row.
      const summary = await fetchJson(tester, 'GET', `/v1/runs/${runId}`)
        .catch(() => null)
      if (summary && !['completed', 'failed', 'cancelled'].includes(summary.status)) {
        await fetchJson(tester, 'POST', `/v1/runs/${runId}/cancel`, {
          expected: [200, 202, 404, 409],
        }).catch(() => null)
        await waitFor(
          async () => {
            const current = await fetchJson(tester, 'GET', `/v1/runs/${runId}`)
            return ['completed', 'failed', 'cancelled'].includes(current.status)
              ? current
              : null
          },
          120_000,
          'terminal state after cancel',
        ).catch(() => {
          scenarioStatus.set('agent.cleanup-integrity', 'failed')
        })
      }
    }
  } catch {
    scenarioStatus.set('agent.cleanup-integrity', 'failed')
  }
  if (tester) {
    const hygieneFindings = [
      ...tester.errors,
      ...networkFindings,
    ]
    scenarioStatus.set(
      'agent.console-network-clean',
      hygieneFindings.length === 0 ? 'passed' : 'failed',
    )
    if (hygieneFindings.length > 0) {
      progress(
        `Konsole/Netzwerk nicht sauber (${hygieneFindings.length} Befunde)`,
      )
    }
  }
  exitCode = [...scenarioStatus.values()].every((status) => status === 'passed')
    ? 0
    : 1
  await writeScenarioSidecar(
    SCENARIOS.map((id) => ({ id, status: scenarioStatus.get(id) ?? 'failed' })),
  )
  await browser?.close().catch(() => null)
}

process.exit(exitCode)

function requiredEnvironment(name) {
  const value = process.env[name]?.trim()
  if (!value) throw new Error(`${name} is required.`)
  return value
}

function progress(message) {
  console.log(`[agent-desk-live] ${sanitize(message)}`)
}

/** Engine output may echo answer fragments; keep it structural. The
 * orchestrator report additionally redacts UUIDs — deliberately not
 * bypassed here. */
function sanitize(value) {
  return value.replaceAll(/https?:\/\/\S+/g, '[url]').slice(0, 400)
}

async function waitFor(probe, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs
  let lastError = null
  while (Date.now() < deadline) {
    try {
      const value = await probe()
      if (value) return value
    } catch (error) {
      lastError = error
    }
    await new Promise((resolve) => setTimeout(resolve, 1_000))
  }
  throw new Error(
    `Timed out waiting for ${label}${lastError ? `: ${lastError}` : ''}`,
  )
}

/** Sidecar contract of tests/verification/scenario-results.ts:
 * atomic write, mode 0600, {schemaVersion: 1, scenarios}. */
async function writeScenarioSidecar(scenarios) {
  const path = process.env.INQTRIX_VERIFICATION_SCENARIO_RESULTS_PATH
  if (!path) return
  const temporaryPath = `${path}.tmp`
  const payload = `${JSON.stringify({ scenarios, schemaVersion: 1 }, null, 2)}\n`
  await writeFile(temporaryPath, payload, { encoding: 'utf8', mode: 0o600 })
  await rename(temporaryPath, path)
  await chmod(path, 0o600)
}
