import { expect, test, type Route } from '@playwright/test'

const fixturePath = '/browser-tests/fixtures/editor-lifecycle.html'

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error
  })
})

test('A to B rerender never exposes A through B surface or controller registration', async ({ page }) => {
  const pendingSessions = new Map<string, Route>()
  await page.route('**/v1/editor/documents/*/collaboration/session', async (route) => {
    const match = new URL(route.request().url()).pathname.match(/documents\/([^/]+)\/collaboration/)
    if (!match) return route.abort()
    pendingSessions.set(match[1], route)
  })

  await page.goto(`${fixturePath}?scenario=switch`)
  const state = page.getByTestId('switch-state')
  const surface = page.getByTestId('switch-surface')
  await expect(state).toHaveAttribute('data-requested-document-id', 'doc-a')
  await expect(surface.locator('.ProseMirror')).toContainText('Projection DOC-A')
  await expect(surface.locator('.ProseMirror')).toHaveAttribute('contenteditable', 'false')
  await expect.poll(() => pendingSessions.has('doc-a')).toBe(true)

  await page.getByTestId('switch-document').click()
  await expect(state).toHaveAttribute('data-requested-document-id', 'doc-b')
  await expect(state).toHaveAttribute('data-handle-document-id', 'doc-b')
  await expect(surface.locator('.ProseMirror')).toContainText('Projection DOC-B')
  await expect(surface.locator('.ProseMirror')).not.toContainText('Projection DOC-A')
  await expect(surface.locator('.ProseMirror')).toHaveAttribute('contenteditable', 'false')
  await expect(state).toHaveAttribute('data-binding-document-id', 'none')
  await expect(state).toHaveAttribute('data-registered-editor-id', 'doc-b')
  await expect.poll(() => pendingSessions.has('doc-b')).toBe(true)

  const snapshots = await page.evaluate(() => window.__collaborationRenderSnapshots ?? [])
  expect(snapshots).toContainEqual(expect.objectContaining({
    controllerRegistered: false,
    handleDocumentId: 'doc-b',
    lifecycleStatus: 'inactive',
    requestedDocumentId: 'doc-b',
  }))
  expect(snapshots.some((snapshot) => (
    snapshot.requestedDocumentId === 'doc-b' && snapshot.handleDocumentId === 'doc-a'
  ))).toBe(false)

  await pendingSessions.get('doc-a')!.fulfill({
    contentType: 'application/json',
    json: {
      access: 'edit',
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      initial_write_mode: 'edit',
      lease_token: 'late-a-lease',
      protocol_version: 1,
      room: 'browser-fixture:doc-a:g1',
      schema_version: 1,
      user: { color: '#2563EB', id: 'test-user', name: 'Test User' },
      websocket_path: '/collaboration',
    },
  })
  await page.waitForTimeout(50)
  await expect(state).toHaveAttribute('data-requested-document-id', 'doc-b')
  await expect(state).toHaveAttribute('data-handle-document-id', 'doc-b')
  await expect(state).toHaveAttribute('data-registered-editor-id', 'doc-b')
  await expect(surface.locator('.ProseMirror')).not.toContainText('Projection DOC-A')
})

test('activation success closes the writable body window before delayed exact hydration', async ({ page }) => {
  await page.goto(`${fixturePath}?scenario=activation`)
  const state = page.getByTestId('activation-state')
  const editor = page.getByTestId('activation-surface').locator('.ProseMirror')

  await expect(state).toHaveAttribute('data-content-mode', 'markdown')
  await expect(state).toHaveAttribute('data-autosave-kind', 'legacy_body')
  await expect(editor).toHaveAttribute('contenteditable', 'true')
  await expect(editor).toContainText('Writable legacy body')

  await page.getByTestId('activation-success').click()
  await expect(state).toHaveAttribute('data-content-mode', 'collaboration')
  await expect(state).toHaveAttribute('data-autosave-kind', 'collaboration_metadata')
  await expect(state).toHaveAttribute('data-document-updated-at', '2099-01-01T00:00:00.000Z')
  await expect(editor).toHaveAttribute('contenteditable', 'false')
  await expect(editor).toContainText('Writable legacy body')
  const postActivationChanges = await state.getAttribute('data-body-change-count')
  await editor.click()
  await page.keyboard.type(' must not persist')
  await expect(state).toHaveAttribute('data-body-change-count', postActivationChanges!)
  await expect(state).toHaveAttribute('data-autosave-kind', 'collaboration_metadata')
  await expect(editor).not.toContainText('must not persist')

  await page.getByTestId('detail-success').click()
  await expect(state).toHaveAttribute('data-document-updated-at', '2026-07-15T08:05:00.000Z')
  await expect(editor).toHaveAttribute('contenteditable', 'false')
  await expect(editor).toContainText('Exact hydrated projection')
  await expect(editor).not.toContainText('Writable legacy body')
})

test('narrow topbar keeps long title and four-participant status in separate bounded tracks', async ({ page }) => {
  await page.setViewportSize({ height: 800, width: 360 })
  await page.goto(`${fixturePath}?scenario=topbar`)

  const status = page.getByRole('status')
  await expect(status).toHaveAccessibleName(/4 (?:participants|Teilnehmende): Ada Lovelace, Lin Chen, Max Weber, Zoe Smith/)
  await expect(status.locator('[data-participant-id]')).toHaveCount(3)
  await expect(status).toContainText('+1')

  const geometry = await page.evaluate(() => {
    const rect = (selector: string) => {
      const value = document.querySelector<HTMLElement>(selector)!.getBoundingClientRect()
      return { left: value.left, right: value.right, width: value.width }
    }
    const title = document.querySelector<HTMLElement>('[data-testid="long-title"]')!
    return {
      actions: rect('[data-editor-topbar-actions]'),
      header: rect('[data-editor-topbar]'),
      leading: rect('[data-editor-topbar-leading]'),
      titleClipped: title.scrollWidth > title.clientWidth,
      toolbar: rect('[data-editor-topbar-toolbar]'),
    }
  })

  expect(geometry.header.left).toBeGreaterThanOrEqual(0)
  expect(geometry.header.right).toBeLessThanOrEqual(360)
  expect(geometry.leading.left).toBeGreaterThanOrEqual(geometry.header.left)
  expect(geometry.leading.right).toBeLessThanOrEqual(geometry.toolbar.left + 0.5)
  expect(geometry.toolbar.right).toBeLessThanOrEqual(geometry.actions.left + 0.5)
  expect(geometry.actions.right).toBeLessThanOrEqual(geometry.header.right)
  expect(geometry.titleClipped).toBe(true)
})

test('initial and same-event Suggest policies govern the first document transaction', async ({ page }) => {
  await page.goto(`${fixturePath}?scenario=initial-suggest-policy`)

  const initial = page.getByTestId('initial-suggest-editor')
  await expect(initial.locator('ins[data-suggestion-id]')).toContainText('immediate')

  await page.goto(`${fixturePath}?scenario=view-policy`)
  const viewOnly = page.getByTestId('initial-view-editor').locator('.ProseMirror')
  await expect(viewOnly).toContainText('View')
  await expect(viewOnly).not.toContainText('blocked')

  await page.goto(`${fixturePath}?scenario=mode-switch-policy`)
  const switched = page.getByTestId('mode-switch-editor')
  await expect(switched.locator('ins[data-suggestion-id]')).toHaveCount(0)
  await page.getByTestId('switch-to-suggest').click()
  await expect(switched.locator('ins[data-suggestion-id]')).toContainText('switched')
})

test('Simple Markup reveals every active compound suggestion and hides other deletions', async ({ page }) => {
  await page.goto(`${fixturePath}?scenario=simple-markup`)
  const markup = page.getByTestId('simple-markup-editor')
  const activeDeletion = markup.locator('[data-review-suggestion-id="delete-active"]')
  const activeInsertion = markup.locator('[data-review-suggestion-id="insert-active"]')
  const inactiveDeletion = markup.locator('[data-review-suggestion-id="delete-inactive"]')

  await expect(activeDeletion).toHaveAttribute('data-review-active', 'true')
  await expect(activeInsertion).toHaveAttribute('data-review-active', 'true')
  await expect(activeDeletion).not.toHaveCSS('display', 'none')
  await expect(inactiveDeletion).not.toHaveAttribute('data-review-active', 'true')
  await expect(inactiveDeletion).toHaveCSS('display', 'none')

  await expect(page.getByTestId('canonical-source')).toHaveText('new')
  const canonicalDiff = page.getByTestId('canonical-diff')
  await expect(canonicalDiff.locator('.editor-document-diff-token-delete')).toHaveText('old')
  await expect(canonicalDiff.locator('.editor-document-diff-token-insert')).toHaveText('new')
})

test('view-only assistant controls explain the restriction and never dispatch work', async ({ page }) => {
  await page.goto(`${fixturePath}?scenario=view-ai`)

  const privateControls = page.getByTestId('private-suggestion-controls')
  const batchInspector = page.getByTestId('batch-inspector')
  const accept = privateControls.getByRole('button', { exact: true, name: 'Accept' })
  const acceptAll = privateControls.getByRole('button', { exact: true, name: 'Accept all' })
  const reject = privateControls.getByRole('button', { exact: true, name: 'Reject' })
  const rejectAll = privateControls.getByRole('button', { exact: true, name: 'Reject all' })
  const diffAnchor = page.getByRole('button', { exact: true, name: 'Set comparison anchor' })
  await expect(accept).toBeEnabled()
  await expect(acceptAll).toBeEnabled()
  await expect(reject).toBeEnabled()
  await expect(rejectAll).toBeEnabled()
  await expect(diffAnchor).toBeEnabled()

  await batchInspector.getByRole('button', { name: /^(Accept all|Alle annehmen)$/ }).click()
  const batchDialog = page.getByRole('dialog')
  await expect(batchDialog).toBeVisible()
  await expect(batchDialog.getByRole('button', { name: /^(Accept all|Alle annehmen)$/ })).toBeEnabled()

  await page.getByTestId('downgrade-ai-access').evaluate((button: HTMLButtonElement) => {
    button.click()
  })
  await expect(batchDialog).toBeHidden()
  await expect(page.getByRole('note')).toContainText('AI editing is unavailable with view-only access.')
  await expect(accept).toBeDisabled()
  await expect(acceptAll).toBeDisabled()
  await expect(reject).toBeDisabled()
  await expect(rejectAll).toBeDisabled()
  await expect(page.getByRole('button', { name: /read-only/i })).toBeDisabled()
  const send = page.getByRole('button', { name: 'Send' })
  await expect(send).toBeDisabled()
  const composer = page.locator(
    '.mention-composer-prose[aria-label="Describe what should change in this document..."]',
  )
  await composer.press('Enter')
  await expect(page.getByTestId('ai-invocations')).toHaveText('0')
  await expect(page.getByTestId('anchor-invocations')).toHaveText('0')
  await expect(page.getByTestId('decision-invocations')).toHaveText('0')
  await expect(page.getByTestId('publish-invocations')).toHaveText('0')
})
