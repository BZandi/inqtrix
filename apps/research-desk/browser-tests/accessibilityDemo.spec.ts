import { writeFile } from 'node:fs/promises'

import { expect, test, type Page, type TestInfo } from '@playwright/test'
import axe from 'axe-core'

const WORKSPACES = [
  'Research Desk',
  'Knowledge Desk',
  'Chat',
  'Editor',
  'Agent Desk',
  'Prompt Library',
  'Datenbank',
  'Einstellungen',
] as const

const VIEWPORTS = [
  { height: 900, label: 'desktop', width: 1_440 },
  { height: 844, label: 'mobile', width: 390 },
] as const

type AxeViolation = {
  help: string
  helpUrl: string
  id: string
  impact: string | null
  nodes: Array<{
    failureSummary: string | null
    target: string[]
  }>
  tags: string[]
}

type SurfaceViolation = AxeViolation & {
  surface: string
  viewport: string
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear()
    window.sessionStorage.clear()
  })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await installBackendFreeDiscoveryRoutes(page)
})

test('demo workspaces have no automated WCAG A or AA violations', async ({ page }, testInfo) => {
  test.setTimeout(180_000)
  await page.setViewportSize(VIEWPORTS[0])
  await page.goto('/', { waitUntil: 'domcontentloaded' })
  await enableDemoMode(page)
  await installAxe(page)

  const violations: SurfaceViolation[] = []
  for (const viewport of VIEWPORTS) {
    await page.setViewportSize(viewport)
    for (const workspace of WORKSPACES) {
      const navigation = page.getByRole('button', { exact: true, name: workspace })
      await expect(navigation).toBeVisible()
      await navigation.click()
      await expect(navigation).toHaveAttribute('aria-pressed', 'true')
      await settleLayout(page)
      if (testInfo.project.name === 'chromium') {
        await captureSurfaceScreenshot(page, testInfo, viewport.label, workspace)
      }
      violations.push(...(await scanVisibleSurface(page)).map((violation) => ({
        ...violation,
        surface: workspace,
        viewport: viewport.label,
      })))
    }
  }

  const deduplicated = deduplicateViolations(violations)
  if (deduplicated.length > 0) {
    const evidencePath = testInfo.outputPath('axe-wcag-a-aa-violations.json')
    await writeFile(evidencePath, `${JSON.stringify(deduplicated, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
    await testInfo.attach('axe-wcag-a-aa-violations', {
      contentType: 'application/json',
      path: evidencePath,
    })
  }
  expect(
    deduplicated,
    accessibilityFailureMessage(testInfo, deduplicated),
  ).toEqual([])
})

test('new chats start as localized empty sessions in both UI languages', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })
  await enableDemoMode(page)

  const chatNavigation = page
    .getByRole('navigation', { name: 'Ansichten' })
    .getByRole('button', { exact: true, name: 'Chat' })
  await chatNavigation.click()
  await expect(chatNavigation).toHaveAttribute('aria-pressed', 'true')

  const history = page.getByRole('complementary')
  const conversation = page.getByTestId('chat-conversation-panel')
  await history.getByRole('button', { exact: true, name: 'Neuer Chat' }).click()
  await expect(conversation.getByRole('button', { exact: true, name: 'Neuer Chat' })).toBeVisible()
  await expect(conversation.locator('[title="Bereit für eine freie Frage."]')).toBeVisible()
  await expect(conversation.getByRole('heading', { name: 'Direkt fragen oder Kontext nutzen.' })).toBeVisible()
  await expect(conversation.getByText(
    'New conversation ready. Ask a question or sketch the research you want to derive from it.',
    { exact: true },
  )).toHaveCount(0)

  await page.getByRole('button', { exact: true, name: 'EN' }).click()
  await history.getByRole('button', { exact: true, name: 'New chat' }).click()
  await expect(conversation.getByRole('button', { exact: true, name: 'New chat' })).toBeVisible()
  await expect(conversation.locator('[title="Ready for an open question."]')).toBeVisible()
  await expect(conversation.getByRole('heading', { name: 'Ask directly or use context.' })).toBeVisible()
  await expect(conversation.getByText(
    'New conversation ready. Ask a question or sketch the research you want to derive from it.',
    { exact: true },
  )).toHaveCount(0)
})

async function installBackendFreeDiscoveryRoutes(page: Page): Promise<void> {
  const jsonResponse = (body: unknown) => ({
    body: JSON.stringify(body),
    contentType: 'application/json',
    status: 200,
  })
  await page.route('**/api/auth/config', (route) => route.fulfill(jsonResponse({
    auth_mode: 'none',
    auth_required: false,
    csrf_header: 'x-csrf-token',
    csrf_required: false,
    login_methods: [],
    pat_available: false,
    provider_name: null,
    registration: { needs_owner: false, self_service: false },
    supports_logout: false,
  })))
  await page.route('**/health', (route) => route.fulfill(jsonResponse({
    auth_mode: 'none',
    auth_required: false,
    llm: { provider: 'fixture', status: 'unavailable' },
    search: { provider: 'fixture', status: 'unavailable' },
    status: 'degraded',
  })))
  await page.route('**/v1/capabilities', (route) => route.fulfill(jsonResponse({
    algorithms: [],
    features: {
      embedding_provider: false,
      knowledge: false,
      multi_stack: false,
      openapi: false,
    },
    timeouts: {
      chat_wait_seconds: 3_630,
      editor_wait_seconds: 630,
      text_wait_seconds: 630,
    },
  })))
  await page.route('**/v1/runs?*', (route) => route.fulfill(jsonResponse({
    data: [],
    next_cursor: null,
  })))
}

async function captureSurfaceScreenshot(
  page: Page,
  testInfo: TestInfo,
  viewport: string,
  workspace: string,
): Promise<void> {
  const surface = workspace.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')
  const name = `${viewport}-${surface}`
  const evidencePath = testInfo.outputPath(`${name}.png`)
  await page.screenshot({ animations: 'disabled', path: evidencePath })
  await testInfo.attach(name, { contentType: 'image/png', path: evidencePath })
}

async function enableDemoMode(page: Page): Promise<void> {
  const settings = page.getByRole('button', { exact: true, name: 'Einstellungen' })
  await expect(settings).toBeVisible({ timeout: 30_000 })
  await settings.click()

  const switchControl = page.getByRole('switch', { name: 'Demo-Modus' })
  await expect(switchControl).not.toBeChecked()
  await switchControl.click()

  // Enabling demo mode replaces the workspace and returns to Research Desk.
  // Re-open Settings and assert the control that owns the mode state.
  await settings.click()
  await expect(page.getByRole('switch', { name: 'Demo-Modus' })).toBeChecked()
  await expect(page.getByRole('button', { exact: true, name: 'Knowledge Desk' })).toBeVisible()
  await expect(page.getByRole('button', { exact: true, name: 'Agent Desk' })).toBeVisible()
}

async function installAxe(page: Page): Promise<void> {
  await page.addScriptTag({ content: axe.source })
  await expect.poll(
    () => page.evaluate(() => typeof (window as Window & { axe?: unknown }).axe),
    { message: 'The versioned axe runtime must be available in the browser.' },
  ).toBe('object')
}

async function scanVisibleSurface(page: Page): Promise<AxeViolation[]> {
  return await page.evaluate(async () => {
    const runtime = (window as Window & {
      axe?: {
        run: (
          context: Document,
          options: Record<string, unknown>,
        ) => Promise<{ violations: AxeViolation[] }>
      }
    }).axe
    if (!runtime) throw new Error('axe is unavailable in the browser context.')
    const result = await runtime.run(document, {
      resultTypes: ['violations'],
      runOnly: {
        type: 'tag',
        values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'],
      },
    })
    return result.violations.map((violation) => ({
      help: violation.help,
      helpUrl: violation.helpUrl,
      id: violation.id,
      impact: violation.impact,
      nodes: violation.nodes.map((node) => ({
        failureSummary: node.failureSummary,
        target: node.target.map(String),
      })),
      tags: violation.tags,
    }))
  })
}

async function settleLayout(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await document.fonts.ready
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    })
  })
}

function deduplicateViolations(violations: SurfaceViolation[]): SurfaceViolation[] {
  const unique = new Map<string, SurfaceViolation>()
  for (const violation of violations) {
    const key = JSON.stringify([
      violation.viewport,
      violation.surface,
      violation.id,
      violation.nodes.map((node) => node.target),
    ])
    unique.set(key, violation)
  }
  return [...unique.values()]
}

function accessibilityFailureMessage(
  testInfo: TestInfo,
  violations: readonly SurfaceViolation[],
): string {
  if (violations.length === 0) return ''
  const summary = violations.map((violation) => (
    `${violation.viewport}/${violation.surface}: ${violation.id} `
    + `(${violation.impact ?? 'unknown'}) on `
    + violation.nodes.map((node) => node.target.join(' ')).join(', ')
  ))
  return [
    `${testInfo.project.name} reported ${violations.length} WCAG A/AA violations.`,
    ...summary,
  ].join('\n')
}
