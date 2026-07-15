import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import type { Download, Locator, Page, TestInfo } from '@playwright/test'
import * as Y from 'yjs'

import {
  collaborationSocketWindow,
  installCollaborationWebSocketObserver as installWebSocketObserverInPage,
  type CollaborationSocketObserverState,
} from './browser-observer'
import { CollaborationFixtureControlClient } from './control'
import type { CollaborationE2EStack } from './config'
import { expect, test, type CollaborationHarness } from './fixtures'
import {
  controlBoundsViolations,
  type Bounds as LayoutBounds,
  type ControlGeometry,
  type Viewport,
} from './layout'
import {
  assertTransportFingerprint,
  observeTransportFingerprint,
} from './transport-fingerprint'
import { parseCollaborationProtocolSession } from './protocol-session'

const labels = {
  de: {
    accept: 'Annehmen',
    acceptAll: 'Alle annehmen',
    accessRevoked: 'Zugriff entzogen',
    all: 'Alle',
    assistant: 'Assistenz',
    author: 'Person',
    changes: 'Änderungen',
    closeInspector: 'Inspector schließen',
    connected: 'Verbunden',
    display: 'Anzeige',
    edit: 'Bearbeiten',
    exportBackup: 'Backup herunterladen',
    final: 'Final',
    importFile: 'Aus Datei importieren',
    live: 'Live',
    menu: 'Menü',
    open: 'Offen',
    original: 'Original',
    reject: 'Ablehnen',
    rejectAll: 'Alle ablehnen',
    reconnecting: 'Erneute Verbindung',
    saved: 'Gespeichert',
    saving: 'Wird gespeichert',
    showInspector: 'Editor-Assistent einblenden',
    showTree: 'Dateibaum einblenden',
    simple: 'Einfach',
    source: 'Source',
    sourceEditor: 'Markdown Source',
    sourceReadOnly: 'Quelltext ist in der Zusammenarbeit schreibgeschützt.',
    suggest: 'Vorschlagen',
    type: 'Art',
    viewLocked: 'Diese Freigabe ist schreibgeschützt.',
  },
  en: {
    accept: 'Accept',
    acceptAll: 'Accept all',
    accessRevoked: 'Access revoked',
    all: 'All',
    assistant: 'Assistant',
    author: 'Person',
    changes: 'Changes',
    closeInspector: 'Close inspector',
    connected: 'Connected',
    display: 'Display',
    edit: 'Edit',
    exportBackup: 'Download backup',
    final: 'Final',
    importFile: 'Import from file',
    live: 'Live',
    menu: 'Menu',
    open: 'Open',
    original: 'Original',
    reject: 'Reject',
    rejectAll: 'Reject all',
    reconnecting: 'Reconnecting',
    saved: 'Saved',
    saving: 'Saving',
    showInspector: 'Show editor assistant',
    showTree: 'Show file tree',
    simple: 'Simple',
    source: 'Source',
    sourceEditor: 'Markdown source',
    sourceReadOnly: 'Source is read-only during collaboration.',
    suggest: 'Suggest',
    type: 'Type',
    viewLocked: 'This share is read-only.',
  },
} as const

test.describe('two-user editor collaboration', () => {
  test('@transport-fingerprint @mobile each endpoint exposes its shipped transport fingerprint', async ({ configuredStack, request }) => {
    if (!configuredStack) {
      throw new Error('Playwright did not stop a test skipped for an unavailable stack.')
    }
    const { stack, transport } = configuredStack
    const baseURL = stack.transports[transport].baseURL
    if (!baseURL) throw new Error(`${transport} baseURL is unavailable after fixture validation.`)

    const observation = await observeTransportFingerprint(request, baseURL)
    expect(assertTransportFingerprint(transport, observation)).toBe(transport)
  })

  test('@direct-edit @mobile direct edit propagates with the remote caret at the author position', async ({ collaboration }) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    await Promise.all([
      openDocument(ownerPage, stack.documents.directEdit, stack.locale),
      openDocument(collaboratorPage, stack.documents.directEdit, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])

    const marker = uniqueMarker('direct')
    const collaboratorEditor = editor(collaboratorPage)
    await appendMarker(collaboratorEditor, marker)
    await expectTextOccurrences(editor(ownerPage), marker, 1)
    await expectRemoteCaretAtMarker(ownerPage, marker, stack.collaborator.displayName)

    await selectMarker(collaboratorEditor, marker)
    await expect.poll(() => collaborationSelectionPresentation(ownerPage)).toMatchObject({
      count: expect.any(Number),
      opaqueCount: 0,
      unclassifiedCount: 0,
    })
    expect((await collaborationSelectionPresentation(ownerPage)).count).toBeGreaterThan(0)

    await removeMarker(collaboratorEditor, marker)
    await expectTextOccurrences(editor(ownerPage), marker, 0)
  })

  test('@concurrent-edits @mobile concurrent owner and collaborator edits converge exactly once', async ({ collaboration }, testInfo) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.concurrent,
      'fixture.documents.concurrent',
      testInfo,
    )
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    await Promise.all([
      chooseWriteMode(ownerPage, stack.locale, 'edit'),
      chooseWriteMode(collaboratorPage, stack.locale, 'edit'),
    ])

    const ownerMarker = uniqueMarker('concurrent-owner')
    const collaboratorMarker = uniqueMarker('concurrent-collaborator')
    await Promise.all([
      appendMarker(editor(ownerPage), ownerMarker),
      appendMarker(editor(collaboratorPage), collaboratorMarker),
    ])
    for (const page of [ownerPage, collaboratorPage]) {
      await expectTextOccurrences(editor(page), ownerMarker, 1)
      await expectTextOccurrences(editor(page), collaboratorMarker, 1)
    }

    await removeMarker(editor(ownerPage), ownerMarker)
    await removeMarker(editor(ownerPage), collaboratorMarker)
    for (const page of [ownerPage, collaboratorPage]) {
      await expectTextOccurrences(editor(page), ownerMarker, 0)
      await expectTextOccurrences(editor(page), collaboratorMarker, 0)
    }
  })

  test('@suggestions @mobile suggestions can be accepted and rejected by the owner', async ({ collaboration }) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const suggestionFixture = stack.documents.suggestion
    const documentId = suggestionFixture.documentId
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    const share = await activeEditorShare(ownerPage, documentId, stack.collaborator.userId)
    expect(share.permission).toBe(suggestionFixture.expectedPermission)
    const schemaVersion = (await editorDocumentDetail(ownerPage, documentId))
      .collaboration.schema_version
    const sessionResult = await browserApi(
      collaboratorPage,
      `/v1/editor/documents/${documentId}/collaboration/session`,
      'POST',
      { protocol_version: 1, schema_version: schemaVersion },
    )
    requireApiSuccess(sessionResult)
    const suggestSession = parseCollaborationProtocolSession(sessionResult.payload)
    expect(suggestSession.access).toBe(suggestionFixture.expectedPermission)
    expect(suggestSession.initialWriteMode).toBe(suggestionFixture.expectedPermission)
    expect(suggestSession.userId).toBe(suggestionFixture.expectedAuthorId)

    await requireProjectionFlush(ownerPage, documentId)
    const beforeManipulatedUpdate = await editorDocumentDetail(ownerPage, documentId)
    const manipulatedMarker = uniqueMarker('suggest-direct-yjs-rejected')
    const manipulatedUpdate = createIndependentYjsUpdate(manipulatedMarker)
    await openBrowserProtocolProbe(collaboratorPage, sessionResult.payload)
    try {
      await sendBrowserProtocolProbeUpdate(collaboratorPage, manipulatedUpdate.bytes)
      await expect.poll(async () => {
        const state = await browserProtocolProbeState(collaboratorPage)
        return { closeCodes: state.closeCodes, errors: state.errors }
      }, { timeout: 30_000 }).toEqual({ closeCodes: [4403], errors: [] })
      await requireProjectionFlush(ownerPage, documentId)
      const afterManipulatedUpdate = await editorDocumentDetail(ownerPage, documentId)
      expect(afterManipulatedUpdate.collaboration.persisted_sequence).toBe(
        beforeManipulatedUpdate.collaboration.persisted_sequence,
      )
      expect(afterManipulatedUpdate.collaboration.projection_sequence).toBe(
        beforeManipulatedUpdate.collaboration.projection_sequence,
      )
      expect(afterManipulatedUpdate.content_markdown).not.toContain(manipulatedMarker)
      expect((await browserProtocolProbeState(collaboratorPage)).durableAckHashes)
        .not.toContain(manipulatedUpdate.hash)
      await expectTextOccurrences(editor(ownerPage), manipulatedMarker, 0)
      await expectTextOccurrences(editor(collaboratorPage), manipulatedMarker, 0)
    } finally {
      await closeBrowserProtocolProbe(collaboratorPage)
    }

    await chooseWriteMode(collaboratorPage, stack.locale, 'suggest')

    const acceptedMarker = uniqueMarker('accept')
    await appendMarker(editor(collaboratorPage), acceptedMarker)
    await expectSuggestionIdentity(
      ownerPage,
      collaboratorPage,
      acceptedMarker,
      suggestionFixture.expectedAuthorId,
    )
    await decideVisibleSuggestion(ownerPage, stack.locale, acceptedMarker, 'accept')
    await expect(editor(ownerPage)).toContainText(acceptedMarker)
    await expect(editor(collaboratorPage)).toContainText(acceptedMarker)

    await chooseWriteMode(ownerPage, stack.locale, 'edit')
    await removeMarker(editor(ownerPage), acceptedMarker)
    await expect(editor(collaboratorPage)).not.toContainText(acceptedMarker)

    const rejectedMarker = uniqueMarker('reject')
    await appendMarker(editor(collaboratorPage), rejectedMarker)
    await expectSuggestionIdentity(
      ownerPage,
      collaboratorPage,
      rejectedMarker,
      suggestionFixture.expectedAuthorId,
    )
    await decideVisibleSuggestion(ownerPage, stack.locale, rejectedMarker, 'reject')
    await expect(editor(ownerPage)).not.toContainText(rejectedMarker)
    await expect(editor(collaboratorPage)).not.toContainText(rejectedMarker)
  })

  test('@ime @mobile genuine Chromium IME composition creates one shared suggestion', async ({ collaboration }) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const suggestionFixture = stack.documents.suggestion
    await Promise.all([
      openDocument(ownerPage, suggestionFixture.documentId, stack.locale),
      openDocument(collaboratorPage, suggestionFixture.documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    await chooseWriteMode(collaboratorPage, stack.locale, 'suggest')

    const marker = `${uniqueMarker('ime')}-日本`
    const compositionEvents = await composeMarkerWithChromiumIme(
      collaboratorPage,
      editor(collaboratorPage),
      marker,
    )
    expect(compositionEvents.some((event) => (
      event.isTrusted && event.type === 'compositionstart'
    ))).toBe(true)
    expect(compositionEvents.some((event) => (
      event.isTrusted && event.type === 'compositionend'
    ))).toBe(true)
    expect(compositionEvents.some((event) => (
      event.isTrusted && event.type === 'input' && event.isComposing
    ))).toBe(true)
    await expectTextOccurrences(editor(collaboratorPage), marker, 1)
    await expectTextOccurrences(editor(ownerPage), marker, 1)

    const ownerMarks = ownerPage.locator('.editor-prose ins[data-suggestion-id]', {
      hasText: marker,
    })
    const collaboratorMarks = collaboratorPage.locator(
      '.editor-prose ins[data-suggestion-id]',
      { hasText: marker },
    )
    await expect(ownerMarks).toHaveCount(1)
    await expect(collaboratorMarks).toHaveCount(1)
    const suggestionId = await ownerMarks.getAttribute('data-suggestion-id')
    expect(suggestionId).toBeTruthy()
    expect(await collaboratorMarks.getAttribute('data-suggestion-id')).toBe(suggestionId)

    await ensureInspector(ownerPage, stack.locale)
    await ownerPage.getByRole('tab', {
      name: new RegExp(`^${labels[stack.locale].changes}`),
    }).click()
    await expect(
      ownerPage.locator('[data-inspector-change-id]', { hasText: marker }),
    ).toHaveCount(1)

    await decideVisibleSuggestion(ownerPage, stack.locale, marker, 'reject')
    await expectTextOccurrences(editor(ownerPage), marker, 0)
    await expectTextOccurrences(editor(collaboratorPage), marker, 0)
  })

  test('@revocation @mobile revocation closes with 4403, hides the document with 404, and reconnects after restore', async ({ collaboration }) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = stack.documents.revocation
    await installWebSocketObserver(collaboratorPage)
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])

    const share = await activeEditorShare(ownerPage, documentId, stack.collaborator.userId)
    let restoreRequired = false
    try {
      requireApiSuccess(await browserApi(ownerPage, `/v1/shares/${share.id}`, 'DELETE'))
      restoreRequired = true
      await expect.poll(() => observedCloseCodes(collaboratorPage), { timeout: 30_000 })
        .toContain(4403)
      const deniedSession = await browserApi(
        collaboratorPage,
        `/v1/editor/documents/${documentId}/collaboration/session`,
        'POST',
        { protocol_version: 1, schema_version: 1 },
      )
      expect(deniedSession.status).toBe(404)
      const hiddenDocument = await browserApi(
        collaboratorPage,
        `/v1/editor/documents/${documentId}`,
        'GET',
      )
      expect(hiddenDocument.status).toBe(404)
      await ensureInspector(collaboratorPage, stack.locale)
      await expect(
        collaboratorPage.getByText(labels[stack.locale].accessRevoked, { exact: true }),
      ).toBeVisible({ timeout: 30_000 })

      await restoreEditorShare(
        ownerPage,
        collaboratorPage,
        documentId,
        stack.collaborator.userId,
        share.permission,
      )
      restoreRequired = false
      await collaboratorPage.reload({ waitUntil: 'domcontentloaded' })
      await openDocument(collaboratorPage, documentId, stack.locale, false)
      await waitForConnected(collaboratorPage, stack.locale)
    } finally {
      if (restoreRequired) {
        await restoreEditorShare(
          ownerPage,
          collaboratorPage,
          documentId,
          stack.collaborator.userId,
          share.permission,
        )
      }
    }
  })

  test('@permission-downgrade @mobile the same live edit socket rejects updates after downgrade', async ({ collaboration }, testInfo) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.downgrade,
      'fixture.documents.downgrade',
      testInfo,
    )
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])

    let share = await activeEditorShare(ownerPage, documentId, stack.collaborator.userId)
    expect(share.permission).toBe('edit')
    const schemaVersion = (await editorDocumentDetail(ownerPage, documentId))
      .collaboration.schema_version
    const editSession = await browserApi(
      collaboratorPage,
      `/v1/editor/documents/${documentId}/collaboration/session`,
      'POST',
      { protocol_version: 1, schema_version: schemaVersion },
    )
    requireApiSuccess(editSession)
    expect((editSession.payload as { access?: unknown } | null)?.access).toBe('edit')
    await openBrowserProtocolProbe(collaboratorPage, editSession.payload)
    await requireProjectionFlush(ownerPage, documentId)
    const beforeRejectedUpdate = await editorDocumentDetail(ownerPage, documentId)
    const rejectedMarker = uniqueMarker('downgrade-rejected-protocol-update')
    const rejectedUpdate = createIndependentYjsUpdate(rejectedMarker)
    await armBrowserProtocolProbeUpdate(collaboratorPage, rejectedUpdate.bytes)
    let restoreRequired = false
    try {
      share = await updateEditorSharePermission(ownerPage, share, 'view')
      restoreRequired = true
      await commitBrowserProtocolProbePolicyChange(collaboratorPage)
      await expect.poll(
        async () => {
          const state = await browserProtocolProbeState(collaboratorPage)
          return {
            challenged: state.authChallenges > state.authChallengesAtArm,
            errors: state.errors,
            rejected: state.authenticationDenied
              || state.scopes.includes('readonly')
              || state.closeCodes.includes(4403),
            sentAfterChallenge: state.updateSentAfterChallenge,
          }
        },
        { timeout: 30_000 },
      ).toEqual({
        challenged: true,
        errors: [],
        rejected: true,
        sentAfterChallenge: true,
      })
      await expect(editor(collaboratorPage)).toHaveAttribute('contenteditable', 'false', {
        timeout: 30_000,
      })
      await ensureInspector(collaboratorPage, stack.locale)
      await expect(
        collaboratorPage.getByText(labels[stack.locale].viewLocked, { exact: true }),
      ).toBeVisible({ timeout: 30_000 })

      const viewSession = await browserApi(
        collaboratorPage,
        `/v1/editor/documents/${documentId}/collaboration/session`,
        'POST',
        { protocol_version: 1, schema_version: schemaVersion },
      )
      requireApiSuccess(viewSession)
      expect((viewSession.payload as { access?: unknown } | null)?.access).toBe('view')

      await requireProjectionFlush(ownerPage, documentId)
      const afterRejectedUpdate = await editorDocumentDetail(ownerPage, documentId)
      expect(afterRejectedUpdate.collaboration.persisted_sequence).toBe(
        beforeRejectedUpdate.collaboration.persisted_sequence,
      )
      expect(afterRejectedUpdate.collaboration.projection_sequence).toBe(
        beforeRejectedUpdate.collaboration.projection_sequence,
      )
      expect(afterRejectedUpdate.content_markdown).not.toContain(rejectedMarker)
      const probeAfterRejection = await browserProtocolProbeState(collaboratorPage)
      expect(probeAfterRejection.durableAckHashes).not.toContain(rejectedUpdate.hash)
      await expectTextOccurrences(editor(ownerPage), rejectedMarker, 0)
      await expectTextOccurrences(editor(collaboratorPage), rejectedMarker, 0)

      share = await updateEditorSharePermission(ownerPage, share, 'edit')
      restoreRequired = false
      await collaboratorPage.reload({ waitUntil: 'domcontentloaded' })
      await openDocument(collaboratorPage, documentId, stack.locale, false)
      await waitForConnected(collaboratorPage, stack.locale)
      await expect(editor(collaboratorPage)).toHaveAttribute('contenteditable', 'true')
      const marker = uniqueMarker('downgrade-restored')
      await appendMarker(editor(collaboratorPage), marker)
      await expectTextOccurrences(editor(ownerPage), marker, 1)
      await removeMarker(editor(collaboratorPage), marker)
      await expectTextOccurrences(editor(ownerPage), marker, 0)
    } finally {
      await closeBrowserProtocolProbe(collaboratorPage)
      if (restoreRequired) {
        const current = await activeEditorShare(
          ownerPage,
          documentId,
          stack.collaborator.userId,
        )
        if (current.permission !== 'edit') {
          await updateEditorSharePermission(ownerPage, current, 'edit')
        }
      }
    }
  })

  test('@reconciliation @mobile a lost durable ACK reconciles exactly once after reconnect and reload', async ({ collaboration }, testInfo) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.reconciliation,
      'fixture.documents.reconciliation',
      testInfo,
    )
    const controls = requireFixtureControls(stack, testInfo)
    const client = new CollaborationFixtureControlClient(controls)
    await installWebSocketObserver(collaboratorPage)
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])

    await ensureInspector(collaboratorPage, stack.locale)
    await expect(
      collaboratorPage.getByRole('status', {
        exact: true,
        name: labels[stack.locale].saved,
      }).first(),
    ).toBeVisible({ timeout: 30_000 })
    await expect(
      collaboratorPage.getByRole('status', {
        exact: true,
        name: labels[stack.locale].saving,
      }),
    ).toHaveCount(0)
    await closeMobileInspector(collaboratorPage, stack.locale)
    await expect.poll(() => observedActiveCollaborationSocketId(collaboratorPage))
      .not.toBeNull()
    const originalSocketId = await observedActiveCollaborationSocketId(collaboratorPage)
    if (originalSocketId === null) {
      throw new Error('No active collaboration socket was observable before arming lost ACK.')
    }
    const baselineObserverState = await observedCollaborationSocketState(collaboratorPage)
    const baselineOrder = Math.max(0, ...baselineObserverState.events.map((event) => event.order))

    const marker = uniqueMarker('lost-ack')
    const operation = await client.armLostAck(documentId, stack.collaborator.userId)
    expect(operation.state).toBe('armed')
    let durableSequence = 0
    try {
      await appendMarker(editor(collaboratorPage), marker)
      const triggered = await client.waitForState(operation.operationId, 'triggered')
      expect(triggered.closeCode).toBe(1012)
      expect(triggered.durableSequence).not.toBeNull()
      expect(triggered.durableSequence!).toBeGreaterThan(0)
      durableSequence = triggered.durableSequence!
      await expect.poll(() => observedCloseCodes(collaboratorPage), { timeout: 30_000 })
        .toContain(1012)
      await expect.poll(async () => (
        await observedCollaborationSocketState(collaboratorPage)
      ).pendingFrameDecodes).toBe(0)
      const observerState = await observedCollaborationSocketState(collaboratorPage)
      const originalSocketWindow = collaborationSocketWindow(
        observerState.events,
        originalSocketId,
        baselineOrder,
        1012,
      )
      expect(
        originalSocketWindow,
        'The original browser collaboration socket must observe the fixture close.',
      ).not.toBeNull()
      expect(
        originalSocketWindow!.durableAcks,
        'No durable ACK may reach the browser data path before the forced close.',
      ).toEqual([])
      expect(
        originalSocketWindow!.protocolErrors,
        'The browser observer must decode every collaboration frame in the fault window.',
      ).toEqual([])
      await ensureInspector(collaboratorPage, stack.locale)
      await expect(
        collaboratorPage.getByText(labels[stack.locale].reconnecting, { exact: true }),
      ).toBeVisible({ timeout: 20_000 })
      await expect(editor(collaboratorPage)).toHaveAttribute('contenteditable', 'false')
    } finally {
      const restored = await client.restore(operation.operationId)
      expect(restored.state).toBe('ready')
    }
    await waitForConnected(collaboratorPage, stack.locale)
    const reconnectedSocketId = await observedActiveCollaborationSocketId(collaboratorPage)
    expect(reconnectedSocketId).not.toBeNull()
    expect(reconnectedSocketId).not.toBe(originalSocketId)
    const reconciled = await client.waitForDurabilityReconciliation(operation.operationId)
    expect(reconciled.durabilityReconciled).toBe(true)
    expect(reconciled.pendingDurabilityCount).toBe(0)
    expect(reconciled.durableSequence).toBe(durableSequence)
    expect(reconciled.reconciliationSequence).not.toBeNull()
    expect(reconciled.reconciliationSequence!).toBeGreaterThanOrEqual(
      durableSequence,
    )
    await ensureInspector(collaboratorPage, stack.locale)
    await expect(
      collaboratorPage.getByRole('status', {
        exact: true,
        name: labels[stack.locale].saved,
      }).first(),
    ).toBeVisible({ timeout: 30_000 })
    await expect(
      collaboratorPage.getByRole('status', {
        exact: true,
        name: labels[stack.locale].saving,
      }),
    ).toHaveCount(0)
    await closeMobileInspector(collaboratorPage, stack.locale)
    await requireProjectionFlush(ownerPage, documentId)
    await Promise.all([
      ownerPage.reload({ waitUntil: 'domcontentloaded' }),
      collaboratorPage.reload({ waitUntil: 'domcontentloaded' }),
    ])
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale, false),
      openDocument(collaboratorPage, documentId, stack.locale, false),
    ])
    await expectTextOccurrences(editor(ownerPage), marker, 1)
    await expectTextOccurrences(editor(collaboratorPage), marker, 1)
    await removeMarker(editor(ownerPage), marker)
    await expectTextOccurrences(editor(collaboratorPage), marker, 0)
  })

  test('@outage @mobile sidecar outage exposes a stale projection and recovers every durable marker', async ({ collaboration }, testInfo) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.outage,
      'fixture.documents.outage',
      testInfo,
    )
    const controls = requireFixtureControls(stack, testInfo)
    const client = new CollaborationFixtureControlClient(controls)
    await installWebSocketObserver(collaboratorPage)
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    assertFastApiHealth(await browserApi(ownerPage, '/health', 'GET'))
    const before = await editorDocumentDetail(ownerPage, documentId)
    const marker = uniqueMarker('outage')
    const operation = await client.armOutage(documentId, stack.collaborator.userId)
    expect(operation.state).toBe('armed')
    try {
      await appendMarker(editor(collaboratorPage), marker)
      const outage = await client.waitForState(operation.operationId, 'outage')
      expect(outage.outageLayer).toBe('collaboration_sidecar')
      expect(outage.closeCode).toBe(4503)
      expect(outage.durableSequence).not.toBeNull()
      expect(outage.projectionSequence).not.toBeNull()
      expect(outage.durableSequence!).toBeGreaterThan(outage.projectionSequence!)
      await expect.poll(() => observedCloseCodes(collaboratorPage), { timeout: 30_000 })
        .toContain(4503)
      await ensureInspector(collaboratorPage, stack.locale)
      await expect(
        collaboratorPage.getByText(labels[stack.locale].reconnecting, { exact: true }),
      ).toBeVisible({ timeout: 20_000 })
      await expect(editor(collaboratorPage)).toHaveAttribute('contenteditable', 'false')

      const unavailableSession = await browserApi(
        collaboratorPage,
        `/v1/editor/documents/${documentId}/collaboration/session`,
        'POST',
        { protocol_version: 1, schema_version: 1 },
      )
      expect(unavailableSession.status).toBe(503)
      const unavailableFlush = await browserApi(
        ownerPage,
        `/v1/editor/documents/${documentId}/collaboration/projection:flush`,
        'POST',
      )
      expect(unavailableFlush.status).toBe(503)
      assertFastApiHealth(await browserApi(ownerPage, '/health', 'GET'))

      const stale = await editorDocumentDetail(ownerPage, documentId)
      expect(stale.collaboration.persisted_sequence).toBeGreaterThanOrEqual(
        outage.durableSequence!,
      )
      expect(stale.collaboration.projection_sequence).toBeLessThan(
        outage.durableSequence!,
      )
      expect(stale.collaboration.projection_updated_at).toBe(
        before.collaboration.projection_updated_at,
      )
      expect(stale.content_markdown).not.toContain(marker)
    } finally {
      const restored = await client.restore(operation.operationId)
      expect(restored.state).toBe('ready')
    }
    await waitForConnected(collaboratorPage, stack.locale)
    await requireProjectionFlush(ownerPage, documentId)
    const recovered = await editorDocumentDetail(ownerPage, documentId)
    expect(recovered.collaboration.projection_sequence).toBe(
      recovered.collaboration.persisted_sequence,
    )
    expect(recovered.collaboration.projection_updated_at).not.toBe(
      before.collaboration.projection_updated_at,
    )
    expectTextCount(recovered.content_markdown, marker, 1)
    await ownerPage.reload({ waitUntil: 'domcontentloaded' })
    await openDocument(ownerPage, documentId, stack.locale, false)
    await expectTextOccurrences(editor(ownerPage), marker, 1)
    await removeMarker(editor(ownerPage), marker)
    await expectTextOccurrences(editor(collaboratorPage), marker, 0)
  })

  test('@gateway-outage @mobile FastAPI gateway outage is independently observable and recovers durable state', async ({ collaboration }, testInfo) => {
    testInfo.setTimeout(90_000)
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.gatewayOutage,
      'fixture.documents.gatewayOutage',
      testInfo,
    )
    const controls = requireFixtureControls(stack, testInfo)
    const client = new CollaborationFixtureControlClient(controls)
    await installWebSocketObserver(collaboratorPage)
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    assertFastApiHealth(await browserApi(ownerPage, '/health', 'GET'))
    const schemaVersion = (await editorDocumentDetail(ownerPage, documentId))
      .collaboration.schema_version
    const marker = uniqueMarker('gateway-outage-durable')
    await appendMarker(editor(collaboratorPage), marker)
    await expectTextOccurrences(editor(ownerPage), marker, 1)
    await requireProjectionFlush(ownerPage, documentId)
    expectTextCount((await editorDocumentDetail(ownerPage, documentId)).content_markdown, marker, 1)

    const operation = await client.armGatewayOutage(documentId, stack.collaborator.userId)
    expect(operation.state).toBe('armed')
    expect(operation.outageLayer).toBe('fastapi_gateway')
    try {
      const outage = await client.waitForState(operation.operationId, 'outage')
      expect(outage.outageLayer).toBe('fastapi_gateway')
      await expect.poll(
        () => observedCloseCodes(collaboratorPage),
        { timeout: 30_000 },
      ).toContain(1006)
      await expect.poll(async () => isPublicGatewayUnavailable(
        await browserApi(ownerPage, '/health', 'GET', undefined, 3_000),
      ), { timeout: 30_000 }).toBe(true)
      await expect.poll(async () => isPublicGatewayUnavailable(await browserApi(
        collaboratorPage,
        `/v1/editor/documents/${documentId}/collaboration/session`,
        'POST',
        { protocol_version: 1, schema_version: schemaVersion },
        3_000,
      )), { timeout: 30_000 }).toBe(true)

      await ensureInspector(collaboratorPage, stack.locale)
      await expect(
        collaboratorPage.getByText(labels[stack.locale].reconnecting, { exact: true }),
      ).toBeVisible({ timeout: 20_000 })
      await expect(editor(collaboratorPage)).toHaveAttribute('contenteditable', 'false')
    } finally {
      const restored = await client.restore(operation.operationId)
      expect(restored.state).toBe('ready')
      expect(restored.outageLayer).toBe('fastapi_gateway')
    }

    await expect.poll(async () => isFastApiHealth(
      await browserApi(ownerPage, '/health', 'GET', undefined, 3_000),
    ), { timeout: 30_000 }).toBe(true)
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    await requireProjectionFlush(ownerPage, documentId)
    await Promise.all([
      ownerPage.reload({ waitUntil: 'domcontentloaded' }),
      collaboratorPage.reload({ waitUntil: 'domcontentloaded' }),
    ])
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale, false),
      openDocument(collaboratorPage, documentId, stack.locale, false),
    ])
    await expectTextOccurrences(editor(ownerPage), marker, 1)
    await expectTextOccurrences(editor(collaboratorPage), marker, 1)
    await removeMarker(editor(ownerPage), marker)
    await expectTextOccurrences(editor(collaboratorPage), marker, 0)
  })

  test('@private-anchors @mobile private AI and comment anchors remain visible only to their creator', async ({ collaboration }, testInfo) => {
    testInfo.setTimeout(90_000)
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const missingPrivacyFixture = 'External fixture prerequisite missing: fixture.privateAnchors must define one private AI marker and one private comment marker per user.'
    const fixture = stack.privateAnchors
    if (fixture === null) {
      process.stdout.write(`[SKIP ${testInfo.project.name}] ${missingPrivacyFixture}\n`)
      test.skip(true, missingPrivacyFixture)
      throw new Error('Playwright did not stop a test marked skipped for missing private anchors.')
    }
    await Promise.all([
      openDocument(ownerPage, fixture.documentId, stack.locale),
      openDocument(collaboratorPage, fixture.documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    await Promise.all([
      openAssistant(ownerPage, stack.locale),
      openAssistant(collaboratorPage, stack.locale),
    ])

    const ownerBefore = await privateAnchorSnapshot(ownerPage, fixture.owner)
    const collaboratorBefore = await privateAnchorSnapshot(
      collaboratorPage,
      fixture.collaborator,
    )
    await expectPrivateAnchorPrivacy(ownerPage, fixture.collaborator, collaboratorBefore)
    await expectPrivateAnchorPrivacy(collaboratorPage, fixture.owner, ownerBefore)

    const rebaseMarker = `${uniqueMarker('anchor-rebase')} `
    await insertMarkerAtStart(editor(collaboratorPage), rebaseMarker)
    await expectTextOccurrences(editor(ownerPage), rebaseMarker, 1)
    const ownerRebased = await privateAnchorSnapshot(ownerPage, fixture.owner)
    expect(ownerRebased.aiId).toBe(ownerBefore.aiId)
    expect(ownerRebased.commentId).toBe(ownerBefore.commentId)
    expect(ownerRebased.aiOffset).toBeGreaterThan(ownerBefore.aiOffset)
    expect(ownerRebased.commentOffset).toBeGreaterThan(ownerBefore.commentOffset)

    await Promise.all([
      ownerPage.reload({ waitUntil: 'domcontentloaded' }),
      collaboratorPage.reload({ waitUntil: 'domcontentloaded' }),
    ])
    await Promise.all([
      openDocument(ownerPage, fixture.documentId, stack.locale, false),
      openDocument(collaboratorPage, fixture.documentId, stack.locale, false),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    await Promise.all([
      openAssistant(ownerPage, stack.locale),
      openAssistant(collaboratorPage, stack.locale),
    ])
    const ownerReloaded = await privateAnchorSnapshot(ownerPage, fixture.owner)
    const collaboratorReloaded = await privateAnchorSnapshot(
      collaboratorPage,
      fixture.collaborator,
    )
    expect(ownerReloaded).toEqual(ownerRebased)
    expect(collaboratorReloaded.aiId).toBe(collaboratorBefore.aiId)
    expect(collaboratorReloaded.commentId).toBe(collaboratorBefore.commentId)
    await expectPrivateAnchorPrivacy(ownerPage, fixture.collaborator, collaboratorReloaded)
    await expectPrivateAnchorPrivacy(collaboratorPage, fixture.owner, ownerReloaded)

    await removeMarker(editor(ownerPage), rebaseMarker)
    await expectTextOccurrences(editor(collaboratorPage), rebaseMarker, 0)
  })

  test('@detached-transfer @mobile a collaboration export imports as a detached document without reconnecting', async ({ collaboration }, testInfo) => {
    testInfo.setTimeout(90_000)
    const { ownerPage, stack, transport } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.detachedTransfer,
      'fixture.documents.detachedTransfer',
      testInfo,
    )
    await openDocument(ownerPage, documentId, stack.locale)
    await waitForConnected(ownerPage, stack.locale)
    const marker = uniqueMarker('detached-transfer')
    await appendMarker(editor(ownerPage), marker)
    await requireProjectionFlush(ownerPage, documentId)
    await forceDownloadProjectPicker(ownerPage)

    const downloadPromise = ownerPage.waitForEvent('download')
    await triggerProjectAction(ownerPage, stack.locale, labels[stack.locale].exportBackup)
    const download = await downloadPromise
    const zipFiles = parseStoredZip(await downloadBytes(download))
    const documentFile = zipFiles.find((file) => (
      file.contents.includes(`document_id: ${documentId}`)
      && file.contents.includes('detached_from_collaboration: true')
    ))
    expect(documentFile, 'Export must contain a detached collaboration document.').toBeTruthy()
    expectTextCount(documentFile!.contents, marker, 1)

    const uploadRoot = await writeProjectUploadDirectory(zipFiles)
    const endpoint = stack.transports[transport]
    const browser = ownerPage.context().browser()
    if (!browser) {
      throw new Error('Detached transfer requires a browser-backed Playwright context.')
    }
    const detachedContext = await browser.newContext({
      baseURL: endpoint.baseURL!,
      storageState: endpoint.ownerStorageState,
      viewport: ownerPage.viewportSize() ?? undefined,
    })
    try {
      const detachedPage = await detachedContext.newPage()
      await detachedPage.goto('./', { waitUntil: 'domcontentloaded' })
      await expect(
        detachedPage.getByRole('button', { name: 'Editor', exact: true }),
      ).toBeVisible({ timeout: 20_000 })
      await forceDownloadProjectPicker(detachedPage)

      let collaborationSessionAttempts = 0
      let collaborationSocketAttempts = 0
      await detachedContext.route(
        '**/v1/editor/documents/*/collaboration/session',
        async (route) => {
          collaborationSessionAttempts += 1
          await route.abort('blockedbyclient')
        },
      )
      detachedPage.on('websocket', (socket) => {
        if (new URL(socket.url()).pathname === '/collaboration') {
          collaborationSocketAttempts += 1
        }
      })

      const chooserPromise = detachedPage.waitForEvent('filechooser')
      await triggerProjectAction(
        detachedPage,
        stack.locale,
        labels[stack.locale].importFile,
      )
      const chooser = await chooserPromise
      await chooser.setFiles(uploadRoot)
      await detachedPage.getByRole('button', { name: 'Editor', exact: true }).click()
      await expectTextOccurrences(editor(detachedPage), marker, 1)
      await expect(editor(detachedPage)).toHaveAttribute('contenteditable', 'true')
      const activeDocument = detachedPage.locator(
        '[data-editor-document-id]:has(button[aria-pressed="true"])',
      )
      await expect(activeDocument).toHaveCount(1)
      expect(await activeDocument.getAttribute('data-editor-document-id')).not.toBe(documentId)
      await detachedPage.waitForTimeout(1_500)
      expect(collaborationSessionAttempts).toBe(0)
      expect(collaborationSocketAttempts).toBe(0)
    } finally {
      await detachedContext.close()
      await rm(uploadRoot, { force: true, recursive: true })
      await removeMarker(editor(ownerPage), marker)
    }
  })

  test('@protocol-rejection @mobile incompatible schema and wire frames are rejected explicitly', async ({ collaboration }, testInfo) => {
    const { ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = requireCapabilityDocument(
      stack.documents.protocol,
      'fixture.documents.protocol',
      testInfo,
    )
    await openDocument(ownerPage, documentId, stack.locale)
    const detail = await editorDocumentDetail(ownerPage, documentId)
    const schemaVersion = detail.collaboration.schema_version

    const protocolConflict = await browserApi(
      ownerPage,
      `/v1/editor/documents/${documentId}/collaboration/session`,
      'POST',
      { protocol_version: 2_147_483_647, schema_version: schemaVersion },
    )
    expect(protocolConflict.status).toBe(409)
    expect(apiFailureReason(protocolConflict)).toBe('protocol_conflict')

    const schemaConflict = await browserApi(
      ownerPage,
      `/v1/editor/documents/${documentId}/collaboration/session`,
      'POST',
      { protocol_version: 1, schema_version: schemaVersion + 1 },
    )
    expect(schemaConflict.status).toBe(409)
    expect(apiFailureReason(schemaConflict)).toBe('schema_conflict')

    const validSession = await browserApi(
      ownerPage,
      `/v1/editor/documents/${documentId}/collaboration/session`,
      'POST',
      { protocol_version: 1, schema_version: schemaVersion },
    )
    requireApiSuccess(validSession)
    expect(await closeCodeForTextFrame(ownerPage, validSession.payload)).toBe(4409)
  })

  test('@source-readonly @mobile Source stays read-only while its shared projection updates', async ({ collaboration }) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    await Promise.all([
      openDocument(ownerPage, stack.documents.directEdit, stack.locale),
      openDocument(collaboratorPage, stack.documents.directEdit, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])

    await ownerPage.getByRole('button', {
      exact: true,
      name: labels[stack.locale].source,
    }).click()
    const source = ownerPage.getByLabel(labels[stack.locale].sourceEditor, { exact: true })
    await expect(source).toBeVisible()
    await expect(source).not.toHaveAttribute('contenteditable', 'true')
    const editButton = ownerPage.getByRole('button', {
      exact: true,
      name: labels[stack.locale].edit,
    }).first()
    const suggestButton = ownerPage.getByRole('button', {
      exact: true,
      name: labels[stack.locale].suggest,
    }).first()
    await expect(editButton).toBeDisabled()
    await expect(suggestButton).toBeDisabled()
    await editButton.locator('xpath=..').hover()
    await expect(
      ownerPage.getByText(labels[stack.locale].sourceReadOnly, { exact: true }),
    ).toBeVisible()

    const sharedMarker = uniqueMarker('source-shared-projection')
    const rejectedMarker = uniqueMarker('source-rejected-input')
    await appendMarker(editor(collaboratorPage), sharedMarker)
    await expectTextOccurrences(source, sharedMarker, 1)
    await source.click()
    await ownerPage.keyboard.insertText(rejectedMarker)
    await expectTextOccurrences(source, rejectedMarker, 0)
    const detail = await editorDocumentDetail(ownerPage, stack.documents.directEdit)
    expect(detail.content_markdown).not.toContain(rejectedMarker)

    await removeMarker(editor(collaboratorPage), sharedMarker)
    await expectTextOccurrences(source, sharedMarker, 0)
    await ownerPage.getByRole('button', {
      exact: true,
      name: labels[stack.locale].live,
    }).click()
    await expect(source).toHaveCount(0)
    await waitForConnected(ownerPage, stack.locale)
    await expectTextOccurrences(editor(ownerPage), rejectedMarker, 0)
  })

  test('@layout editor and Inspector surfaces stay bounded without control overlap', async ({ collaboration }, testInfo) => {
    const { collaboratorPage, ownerPage, stack } = requireCollaboration(collaboration)
    const documentId = stack.documents.suggestion.documentId
    await Promise.all([
      openDocument(ownerPage, documentId, stack.locale),
      openDocument(collaboratorPage, documentId, stack.locale),
    ])
    await Promise.all([
      waitForConnected(ownerPage, stack.locale),
      waitForConnected(collaboratorPage, stack.locale),
    ])
    await chooseWriteMode(collaboratorPage, stack.locale, 'suggest')
    const marker = uniqueMarker('layout-expanded-change')
    await appendMarker(editor(collaboratorPage), marker)
    await ensureInspector(ownerPage, stack.locale)
    const changesTab = ownerPage.getByRole('tab', {
      name: new RegExp(`^${labels[stack.locale].changes}`),
    })
    await changesTab.click()
    await expect(changesTab).toHaveAttribute('data-state', 'active')
    const inspector = ownerPage.locator('.inqtrix-contained-panel', { has: changesTab })
    const changeRow = inspector.locator('[data-inspector-change-id]', { hasText: marker })
    await expect(changeRow).toHaveCount(1)
    const changeToggle = changeRow.locator('button[aria-expanded]').first()
    await changeToggle.click()
    await expect(changeToggle).toHaveAttribute('aria-expanded', 'true')

    const displayControl = inspector.getByLabel(labels[stack.locale].display, { exact: true })
    await expect(displayControl).toBeVisible()
    for (const display of ['simple', 'all', 'final', 'original'] as const) {
      await expect(displayControl.getByRole('button', {
        exact: true,
        name: labels[stack.locale][display],
      })).toBeVisible()
    }
    await expect(inspector.getByRole('combobox', {
      exact: true,
      name: labels[stack.locale].author,
    })).toBeVisible()
    await expect(inspector.getByRole('combobox', {
      exact: true,
      name: labels[stack.locale].type,
    })).toBeVisible()
    await expect(changeRow.getByRole('button', {
      exact: true,
      name: labels[stack.locale].accept,
    })).toBeVisible()
    await expect(changeRow.getByRole('button', {
      exact: true,
      name: labels[stack.locale].reject,
    })).toBeVisible()
    await expect(inspector.getByRole('button', {
      exact: true,
      name: labels[stack.locale].acceptAll,
    })).toBeVisible()
    await expect(inspector.getByRole('button', {
      exact: true,
      name: labels[stack.locale].rejectAll,
    })).toBeVisible()

    const layout = await wholeEditorLayout(ownerPage)
    expect(layout.horizontalOverflow).toBeLessThanOrEqual(1)
    expect(layout.missingSurfaces).toEqual([])
    expect(layout.outOfViewportSurfaces).toEqual([])
    expect(layout.outOfBoundsControls).toEqual([])
    expect(layout.editorControlOverlaps).toEqual([])
    expect(layout.inspectorControlOverlaps).toEqual([])
    expect(layout.canvasContainedByEditor).toBe(true)

    const formFactor = testInfo.project.metadata.formFactor
    expect(['desktop', 'mobile']).toContain(formFactor)
    if (formFactor === 'mobile') {
      expect(layout.mobileDialogPresent).toBe(true)
      expect(layout.dialogContainedByViewport).toBe(true)
    } else {
      expect(layout.mobileDialogPresent).toBe(false)
      expect(layout.editorInspectorOverlapArea).toBeLessThanOrEqual(1)
    }

    await changeRow.getByRole('button', {
      exact: true,
      name: labels[stack.locale].reject,
    }).click()
    await expect(changeRow).toHaveCount(0)
    await expectTextOccurrences(editor(ownerPage), marker, 0)
    await expectTextOccurrences(editor(collaboratorPage), marker, 0)
  })

  test('@mobile-only @mobile-drawer mobile tree and inspector drawers are modal and exclusive', async ({ collaboration }) => {
    const { ownerPage, stack } = requireCollaboration(collaboration)
    await openDocument(ownerPage, stack.documents.directEdit, stack.locale)

    await ownerPage.getByRole('button', { name: labels[stack.locale].showTree }).click()
    const treeDrawer = ownerPage.locator('#editor-file-tree-panel[role="dialog"]')
    await expect(treeDrawer).toBeVisible()
    await expect(ownerPage.locator('#editor-comments-panel[role="dialog"]')).toHaveCount(0)
    await expect(ownerPage.locator('[role="dialog"]')).toHaveCount(1)
    await ownerPage.keyboard.press('Escape')
    await expect(treeDrawer).toHaveCount(0)

    await ownerPage.getByRole('button', { name: labels[stack.locale].showInspector }).click()
    const inspectorDrawer = ownerPage.locator('#editor-comments-panel[role="dialog"]')
    await expect(inspectorDrawer).toBeVisible()
    await expect(ownerPage.locator('#editor-file-tree-panel[role="dialog"]')).toHaveCount(0)
    await expect(ownerPage.locator('[role="dialog"]')).toHaveCount(1)
    await ownerPage.getByRole('button', { name: labels[stack.locale].closeInspector }).click()
    await expect(inspectorDrawer).toHaveCount(0)
  })
})

function requireCollaboration(
  collaboration: CollaborationHarness | null,
): CollaborationHarness {
  if (!collaboration) {
    throw new Error(
      'The collaboration fixture was not initialized and did not mark the test as skipped.',
    )
  }
  return collaboration
}

function requireCapabilityDocument(
  documentId: string | null,
  field: string,
  testInfo: TestInfo,
): string {
  if (documentId) return documentId
  const reason = `External fixture prerequisite missing: ${field} must be a non-empty string.`
  process.stdout.write(`[SKIP ${testInfo.project.name}] ${reason}\n`)
  testInfo.skip(true, reason)
  throw new Error(`Playwright did not stop a test skipped for missing ${field}.`)
}

function requireFixtureControls(
  stack: CollaborationE2EStack,
  testInfo: TestInfo,
): NonNullable<CollaborationE2EStack['controls']> {
  if (stack.controls) return stack.controls
  const details = stack.capabilityReasons.controls.join('; ')
  const reason = `External fixture prerequisite missing: fixture.controls must provide lost-ACK, sidecar-outage, FastAPI-gateway-outage, status, restore, and restart operations${details ? ` (${details})` : ''}.`
  process.stdout.write(`[SKIP ${testInfo.project.name}] ${reason}\n`)
  testInfo.skip(true, reason)
  throw new Error('Playwright did not stop a test skipped for missing fixture.controls.')
}

async function openDocument(
  page: Page,
  documentId: string,
  locale: 'de' | 'en',
  navigate = true,
): Promise<void> {
  if (navigate) await page.goto('./', { waitUntil: 'domcontentloaded' })
  const editorNavigation = page.getByRole('button', { name: 'Editor', exact: true })
  await expect(editorNavigation).toBeVisible({ timeout: 20_000 })
  await editorNavigation.click()

  const documentRow = page.locator(`[data-editor-document-id="${documentId}"]`).first()
  if (!await documentRow.isVisible().catch(() => false)) {
    const showTree = page.getByRole('button', { name: labels[locale].showTree })
    if (await showTree.isVisible().catch(() => false)) await showTree.click()
  }
  await expect(documentRow).toBeVisible({ timeout: 20_000 })
  await documentRow.click()
  const mobileTree = page.locator('#editor-file-tree-panel[role="dialog"]')
  if (await mobileTree.isVisible().catch(() => false)) {
    await page.keyboard.press('Escape')
    await expect(mobileTree).toHaveCount(0)
  }
  await expect(editor(page)).toBeVisible({ timeout: 20_000 })
}

function editor(page: Page): Locator {
  return page.locator('.editor-prose[contenteditable="true"], .editor-prose[contenteditable="false"]').first()
}

async function ensureInspector(page: Page, locale: 'de' | 'en'): Promise<void> {
  const changes = page.getByRole('tab', { name: new RegExp(`^${labels[locale].changes}`) })
  if (!await changes.isVisible().catch(() => false)) {
    await page.getByRole('button', { name: labels[locale].showInspector }).click()
  }
  await expect(changes).toBeVisible()
}

async function openAssistant(page: Page, locale: 'de' | 'en'): Promise<void> {
  await ensureInspector(page, locale)
  const assistant = page.getByRole('tab', { name: labels[locale].assistant, exact: true })
  await assistant.click()
  await expect(assistant).toHaveAttribute('data-state', 'active')
}

async function waitForConnected(page: Page, locale: 'de' | 'en'): Promise<void> {
  await ensureInspector(page, locale)
  await expect(page.getByText(labels[locale].connected, { exact: true })).toBeVisible({ timeout: 30_000 })
  await closeMobileInspector(page, locale)
}

async function closeMobileInspector(page: Page, locale: 'de' | 'en'): Promise<void> {
  const drawer = page.locator('#editor-comments-panel[role="dialog"]')
  if (!await drawer.isVisible().catch(() => false)) return
  await page.getByRole('button', { name: labels[locale].closeInspector }).click()
  await expect(drawer).toHaveCount(0)
}

async function wholeEditorLayout(page: Page): Promise<{
  canvasContainedByEditor: boolean
  dialogContainedByViewport: boolean
  editorControlOverlaps: string[]
  editorInspectorOverlapArea: number
  horizontalOverflow: number
  inspectorControlOverlaps: string[]
  missingSurfaces: string[]
  mobileDialogPresent: boolean
  outOfBoundsControls: string[]
  outOfViewportSurfaces: string[]
}> {
  const snapshot = await page.evaluate(() => {
    type Bounds = { bottom: number; left: number; right: number; top: number }
    const viewport = {
      height: document.documentElement.clientHeight,
      width: document.documentElement.clientWidth,
    }
    const bounds = (element: Element): Bounds => {
      const rect = element.getBoundingClientRect()
      return { bottom: rect.bottom, left: rect.left, right: rect.right, top: rect.top }
    }
    const intersectsViewport = (rect: Bounds): boolean => (
      Math.min(rect.right, viewport.width) - Math.max(rect.left, 0) > 1
      && Math.min(rect.bottom, viewport.height) - Math.max(rect.top, 0) > 1
    )
    const visible = (element: Element): boolean => {
      const style = getComputedStyle(element)
      const rect = bounds(element)
      return (
        style.display !== 'none'
        && style.visibility !== 'hidden'
        && rect.right - rect.left > 1
        && rect.bottom - rect.top > 1
        && intersectsViewport(rect)
      )
    }
    const contained = (inner: Bounds, outer: Bounds): boolean => (
      inner.left >= outer.left - 1
      && inner.right <= outer.right + 1
      && inner.top >= outer.top - 1
      && inner.bottom <= outer.bottom + 1
    )
    const inViewport = (rect: Bounds): boolean => contained(rect, {
      bottom: viewport.height,
      left: 0,
      right: viewport.width,
      top: 0,
    })
    const canvas = Array.from(document.querySelectorAll<HTMLElement>('.editor-prose'))
      .find(visible) ?? null
    const editorMain = canvas?.closest('main') ?? null
    const canvasViewport = canvas?.closest('[data-radix-scroll-area-viewport]')
      ?? canvas?.parentElement
      ?? null
    const inspector = Array.from(
      document.querySelectorAll<HTMLElement>('.inqtrix-contained-panel'),
    ).find((panel) => visible(panel) && panel.querySelector('[role="tab"]')) ?? null
    const mobileDialog = inspector?.closest<HTMLElement>('[role="dialog"]') ?? null

    const surfaces: Array<[string, Element | null]> = [
      ['editor', editorMain],
      ['editor-canvas-viewport', canvasViewport],
      ['inspector', inspector],
    ]
    if (mobileDialog) surfaces.push(['mobile-inspector-dialog', mobileDialog])
    const missingSurfaces = surfaces
      .filter(([, element]) => element === null)
      .map(([name]) => name)
    const outOfViewportSurfaces = surfaces
      .filter(([, element]) => element !== null && !inViewport(bounds(element)))
      .map(([name]) => name)

    const controlSelector = 'button, input, select, textarea, [role="status"], [role="tab"]'
    const controls = (scope: Element | null, excluded: Element | null = null): Element[] => {
      if (!scope) return []
      return [...new Set(Array.from(scope.querySelectorAll(controlSelector)))]
        .filter((element) => !excluded?.contains(element) && visible(element))
    }
    const controlName = (element: Element, index: number): string => {
      const accessible = element.getAttribute('aria-label')
        ?? element.textContent?.trim()
        ?? element.tagName.toLowerCase()
      return `${element.tagName.toLowerCase()}[${index}]:${accessible.slice(0, 60)}`
    }
    const editorControls = controls(editorMain, inspector)
    const inspectorControls = controls(inspector)
    const controlGeometry = (elements: Element[]): ControlGeometry[] => elements.map(
      (element, index) => ({
        bounds: bounds(element),
        name: controlName(element, index),
      }),
    )
    const overlaps = (elements: Element[]): string[] => {
      const collisions: string[] = []
      for (let leftIndex = 0; leftIndex < elements.length; leftIndex += 1) {
        for (let rightIndex = leftIndex + 1; rightIndex < elements.length; rightIndex += 1) {
          const left = elements[leftIndex]!
          const right = elements[rightIndex]!
          if (left.contains(right) || right.contains(left)) continue
          const leftBounds = bounds(left)
          const rightBounds = bounds(right)
          const overlapWidth = Math.min(leftBounds.right, rightBounds.right)
            - Math.max(leftBounds.left, rightBounds.left)
          const overlapHeight = Math.min(leftBounds.bottom, rightBounds.bottom)
            - Math.max(leftBounds.top, rightBounds.top)
          if (overlapWidth > 1 && overlapHeight > 1) {
            collisions.push(
              `${controlName(left, leftIndex)} <> ${controlName(right, rightIndex)}`,
            )
          }
        }
      }
      return collisions
    }
    const overlapArea = (left: Element | null, right: Element | null): number => {
      if (!left || !right) return 0
      const leftBounds = bounds(left)
      const rightBounds = bounds(right)
      return Math.max(
        0,
        Math.min(leftBounds.right, rightBounds.right) - Math.max(leftBounds.left, rightBounds.left),
      ) * Math.max(
        0,
        Math.min(leftBounds.bottom, rightBounds.bottom) - Math.max(leftBounds.top, rightBounds.top),
      )
    }
    const editorBounds = editorMain ? bounds(editorMain) : null
    const canvasBounds = canvasViewport ? bounds(canvasViewport) : null
    return {
      canvasContainedByEditor: Boolean(
        editorBounds && canvasBounds && contained(canvasBounds, editorBounds),
      ),
      dialogContainedByViewport: mobileDialog ? inViewport(bounds(mobileDialog)) : false,
      editorBounds,
      editorControlGeometry: controlGeometry(editorControls),
      editorControlOverlaps: overlaps(editorControls),
      editorInspectorOverlapArea: overlapArea(editorMain, inspector),
      horizontalOverflow: document.documentElement.scrollWidth - viewport.width,
      inspectorBounds: inspector ? bounds(inspector) : null,
      inspectorControlGeometry: controlGeometry(inspectorControls),
      inspectorControlOverlaps: overlaps(inspectorControls),
      missingSurfaces,
      mobileDialogPresent: mobileDialog !== null,
      outOfViewportSurfaces,
      viewport,
    }
  })
  const {
    editorBounds,
    editorControlGeometry,
    inspectorBounds,
    inspectorControlGeometry,
    viewport,
    ...layout
  } = snapshot as typeof snapshot & {
    editorBounds: LayoutBounds | null
    editorControlGeometry: ControlGeometry[]
    inspectorBounds: LayoutBounds | null
    inspectorControlGeometry: ControlGeometry[]
    viewport: Viewport
  }
  return {
    ...layout,
    outOfBoundsControls: [
      ...controlBoundsViolations(editorControlGeometry, editorBounds, viewport),
      ...controlBoundsViolations(inspectorControlGeometry, inspectorBounds, viewport),
    ],
  }
}

async function chooseWriteMode(
  page: Page,
  locale: 'de' | 'en',
  mode: 'edit' | 'suggest',
): Promise<void> {
  const button = page.getByRole('button', { name: labels[locale][mode], exact: true }).first()
  await expect(button).toBeVisible()
  if (await button.isEnabled()) await button.click()
  await expect(button).toHaveAttribute('aria-pressed', 'true')
}

async function appendMarker(surface: Locator, marker: string): Promise<void> {
  await focusEditorEnd(surface)
  await surface.pressSequentially(` ${marker}`, { delay: 5 })
}

async function insertMarkerAtStart(surface: Locator, marker: string): Promise<void> {
  await surface.evaluate((element) => {
    const selection = window.getSelection()
    const range = document.createRange()
    range.selectNodeContents(element)
    range.collapse(true)
    selection?.removeAllRanges()
    selection?.addRange(range)
    ;(element as HTMLElement).focus()
    document.dispatchEvent(new Event('selectionchange'))
  })
  await surface.pressSequentially(marker, { delay: 5 })
}

type ImeEventRecord = {
  data: string | null
  inputType: string | null
  isComposing: boolean
  isTrusted: boolean
  type: string
}

async function composeMarkerWithChromiumIme(
  page: Page,
  surface: Locator,
  marker: string,
): Promise<ImeEventRecord[]> {
  await focusEditorEnd(surface)
  await surface.evaluate((element) => {
    const records: Array<{
      data: string | null
      inputType: string | null
      isComposing: boolean
      isTrusted: boolean
      type: string
    }> = []
    const record = (event: Event) => {
      const input = event as InputEvent
      const composition = event as CompositionEvent
      records.push({
        data: input.data ?? composition.data ?? null,
        inputType: input.inputType ?? null,
        isComposing: input.isComposing ?? false,
        isTrusted: event.isTrusted,
        type: event.type,
      })
    }
    for (const type of ['beforeinput', 'compositionend', 'compositionstart', 'compositionupdate', 'input']) {
      element.addEventListener(type, record)
    }
    ;(window as unknown as { __inqtrixImeEvents: typeof records }).__inqtrixImeEvents = records
  })
  const session = await page.context().newCDPSession(page)
  try {
    await session.send('Input.imeSetComposition', {
      selectionEnd: marker.length,
      selectionStart: marker.length,
      text: marker,
    })
    await session.send('Input.insertText', { text: marker })
  } finally {
    await session.detach()
  }
  return surface.evaluate(() => (
    (window as unknown as { __inqtrixImeEvents: ImeEventRecord[] }).__inqtrixImeEvents
  ))
}

async function focusEditorEnd(surface: Locator): Promise<void> {
  await surface.evaluate((element) => {
    const selection = window.getSelection()
    const range = document.createRange()
    range.selectNodeContents(element)
    range.collapse(false)
    selection?.removeAllRanges()
    selection?.addRange(range)
    ;(element as HTMLElement).focus()
    document.dispatchEvent(new Event('selectionchange'))
  })
}

async function expectTextOccurrences(
  surface: Locator,
  marker: string,
  expected: number,
): Promise<void> {
  await expect.poll(async () => {
    const content = await surface.textContent()
    return (content ?? '').split(marker).length - 1
  }).toBe(expected)
}

async function selectMarker(surface: Locator, marker: string): Promise<void> {
  const selected = await surface.evaluate((element, needle) => {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const offset = node.textContent?.indexOf(needle) ?? -1
      if (offset < 0) continue
      const range = document.createRange()
      range.setStart(node, offset)
      range.setEnd(node, offset + needle.length)
      const selection = window.getSelection()
      selection?.removeAllRanges()
      selection?.addRange(range)
      ;(element as HTMLElement).focus()
      document.dispatchEvent(new Event('selectionchange'))
      return true
    }
    return false
  }, marker)
  expect(selected, `Expected marker ${marker} to be selectable`).toBe(true)
}

async function expectRemoteCaretAtMarker(
  page: Page,
  marker: string,
  displayName: string,
): Promise<void> {
  await expect.poll(async () => page.evaluate(({ author, needle }) => {
    const labels = Array.from(
      document.querySelectorAll<HTMLElement>('.inqtrix-collaboration-caret-label'),
    ).filter((candidate) => candidate.textContent?.trim() === author)
    const label = labels[0]
    const caret = label?.closest<HTMLElement>('.inqtrix-collaboration-caret')
    const root = caret?.closest<HTMLElement>('.editor-prose')
    if (!label || !caret || !root) return { exactIdentityCount: labels.length, atMarker: false }

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const offset = node.textContent?.indexOf(needle) ?? -1
      if (offset < 0) continue
      const markerRange = document.createRange()
      markerRange.setStart(node, offset + needle.length - 1)
      markerRange.setEnd(node, offset + needle.length)
      const markerRect = markerRange.getBoundingClientRect()
      const caretRect = caret.getBoundingClientRect()
      const sameLine = Math.abs(caretRect.top - markerRect.top) <= 8
      const adjacent = Math.abs(caretRect.left - markerRect.right) <= 12
      return { atMarker: adjacent && sameLine, exactIdentityCount: labels.length }
    }
    return { atMarker: false, exactIdentityCount: labels.length }
  }, { author: displayName, needle: marker })).toEqual({
    atMarker: true,
    exactIdentityCount: 1,
  })
}

type PrivateAnchorDescriptor = NonNullable<CollaborationE2EStack['privateAnchors']>['owner']

type PrivateAnchorSnapshot = {
  aiId: string
  aiOffset: number
  commentId: string
  commentOffset: number
}

async function privateAnchorSnapshot(
  page: Page,
  descriptor: PrivateAnchorDescriptor,
): Promise<PrivateAnchorSnapshot> {
  await expect(page.getByText(descriptor.commentText, { exact: true })).toBeVisible()
  await expect(page.getByText(descriptor.aiText, { exact: true })).toBeVisible()
  const aiDecoration = page.locator('.editor-prose [data-suggestion-id]', {
    hasText: descriptor.aiAnchorText,
  })
  const commentDecoration = page.locator('.editor-prose [data-editor-comment-anchor]', {
    hasText: descriptor.commentAnchorText,
  })
  await expect(aiDecoration).toHaveCount(1)
  await expect(commentDecoration).toHaveCount(1)
  await expect(aiDecoration).toHaveText(descriptor.aiAnchorText)
  await expect(commentDecoration).toHaveText(descriptor.commentAnchorText)
  const aiId = await aiDecoration.getAttribute('data-suggestion-id')
  const commentId = await commentDecoration.getAttribute('data-editor-comment-anchor')
  expect(aiId).toBeTruthy()
  expect(commentId).toBeTruthy()
  return {
    aiId: aiId!,
    aiOffset: await decorationTextOffset(aiDecoration),
    commentId: commentId!,
    commentOffset: await decorationTextOffset(commentDecoration),
  }
}

async function decorationTextOffset(decoration: Locator): Promise<number> {
  return decoration.evaluate((element) => {
    const root = element.closest('.editor-prose')
    if (!root) throw new Error('Private anchor decoration is outside the collaboration editor.')
    const range = document.createRange()
    range.selectNodeContents(root)
    range.setEndBefore(element)
    return range.toString().length
  })
}

async function expectPrivateAnchorPrivacy(
  page: Page,
  foreignDescriptor: PrivateAnchorDescriptor,
  foreignSnapshot: PrivateAnchorSnapshot,
): Promise<void> {
  await expect(page.getByText(foreignDescriptor.commentText, { exact: true })).toHaveCount(0)
  await expect(page.getByText(foreignDescriptor.aiText, { exact: true })).toHaveCount(0)
  await expect(page.locator('.editor-prose [data-suggestion-id]', {
    hasText: foreignDescriptor.aiAnchorText,
  })).toHaveCount(0)
  await expect(page.locator('.editor-prose [data-editor-comment-anchor]', {
    hasText: foreignDescriptor.commentAnchorText,
  })).toHaveCount(0)
  const counts = await page.evaluate(({ aiId, commentId }) => ({
    ai: Array.from(document.querySelectorAll('[data-suggestion-id]'))
      .filter((element) => element.getAttribute('data-suggestion-id') === aiId).length,
    comment: Array.from(document.querySelectorAll('[data-editor-comment-anchor]'))
      .filter((element) => element.getAttribute('data-editor-comment-anchor') === commentId).length,
  }), {
    aiId: foreignSnapshot.aiId,
    commentId: foreignSnapshot.commentId,
  })
  expect(counts).toEqual({ ai: 0, comment: 0 })
}

async function collaborationSelectionPresentation(page: Page): Promise<{
  count: number
  opaqueCount: number
  unclassifiedCount: number
}> {
  return page.evaluate(() => {
    const selections = Array.from(
      document.querySelectorAll<HTMLElement>('.collaboration-carets__selection'),
    )
    return {
      count: selections.length,
      opaqueCount: selections.filter((selection) => {
        const style = getComputedStyle(selection)
        return (
          style.backgroundColor !== 'rgba(0, 0, 0, 0)'
          && style.backgroundColor !== 'transparent'
        ) || style.boxShadow !== 'none'
      }).length,
      unclassifiedCount: selections.filter((selection) => (
        !selection.classList.contains('inqtrix-collaboration-selection')
        || selection.dataset.collaborationSelection !== 'transparent'
      )).length,
    }
  })
}

async function installWebSocketObserver(page: Page): Promise<void> {
  await page.addInitScript(installWebSocketObserverInPage)
}

async function observedCloseCodes(page: Page): Promise<number[]> {
  const state = await observedCollaborationSocketState(page)
  return state.events.flatMap((event) => event.kind === 'close' ? [event.code] : [])
}

async function observedCollaborationSocketState(
  page: Page,
): Promise<CollaborationSocketObserverState> {
  return page.evaluate(() => {
    const state = (window as unknown as {
      __inqtrixCollaborationSocketObserver?: CollaborationSocketObserverState
    }).__inqtrixCollaborationSocketObserver
    return state
      ? {
          events: [...state.events].sort((left, right) => left.order - right.order),
          pendingFrameDecodes: state.pendingFrameDecodes,
        }
      : { events: [], pendingFrameDecodes: 0 }
  })
}

async function observedActiveCollaborationSocketId(page: Page): Promise<number | null> {
  const state = await observedCollaborationSocketState(page)
  const active = new Map<number, number>()
  for (const event of state.events) {
    if (event.kind === 'open') active.set(event.socketId, event.order)
    else if (event.kind === 'close') active.delete(event.socketId)
  }
  const latest = [...active.entries()].sort((left, right) => right[1] - left[1])[0]
  return latest?.[0] ?? null
}

async function removeMarker(surface: Locator, marker: string): Promise<void> {
  const selected = await surface.evaluate((element, needle) => {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const offset = node.textContent?.indexOf(needle) ?? -1
      if (offset < 0) continue
      const range = document.createRange()
      range.setStart(node, offset)
      range.setEnd(node, offset + needle.length)
      const selection = window.getSelection()
      selection?.removeAllRanges()
      selection?.addRange(range)
      ;(element as HTMLElement).focus()
      return true
    }
    return false
  }, marker)
  expect(selected, `Expected marker ${marker} to be selectable`).toBe(true)
  await surface.press('Backspace')
}

async function decideVisibleSuggestion(
  ownerPage: Page,
  locale: 'de' | 'en',
  marker: string,
  decision: 'accept' | 'reject',
): Promise<void> {
  await ensureInspector(ownerPage, locale)
  await ownerPage.getByRole('tab', { name: new RegExp(`^${labels[locale].changes}`) }).click()
  const row = ownerPage.locator('[data-inspector-change-id]', { hasText: marker }).first()
  await expect(row).toBeVisible({ timeout: 20_000 })
  await row.locator('button').first().click()
  await row.getByRole('button', { name: labels[locale][decision], exact: true }).click()
  await expect(row).toBeHidden({ timeout: 20_000 })
  await closeMobileInspector(ownerPage, locale)
}

async function expectSuggestionIdentity(
  ownerPage: Page,
  collaboratorPage: Page,
  marker: string,
  expectedAuthorId: string,
): Promise<void> {
  const ownerMark = ownerPage.locator('.editor-prose ins[data-suggestion-id]', {
    hasText: marker,
  })
  const collaboratorMark = collaboratorPage.locator(
    '.editor-prose ins[data-suggestion-id]',
    { hasText: marker },
  )
  await expect(ownerMark).toHaveCount(1)
  await expect(collaboratorMark).toHaveCount(1)
  const suggestionId = await ownerMark.getAttribute('data-suggestion-id')
  expect(suggestionId).toMatch(
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
  )
  expect(await collaboratorMark.getAttribute('data-suggestion-id')).toBe(suggestionId)
  expect(await ownerMark.getAttribute('data-suggestion-author-id')).toBe(expectedAuthorId)
  expect(await collaboratorMark.getAttribute('data-suggestion-author-id')).toBe(
    expectedAuthorId,
  )
}

type StoredZipFile = {
  contents: string
  path: string
}

async function forceDownloadProjectPicker(page: Page): Promise<void> {
  await page.evaluate(() => {
    Object.defineProperty(window, 'showDirectoryPicker', {
      configurable: true,
      value: undefined,
    })
  })
}

async function triggerProjectAction(
  page: Page,
  locale: 'de' | 'en',
  actionLabel: string,
): Promise<void> {
  const directButton = page.getByRole('button', { name: actionLabel, exact: true })
  if (await directButton.isVisible().catch(() => false)) {
    await directButton.click()
    return
  }
  await page.getByRole('button', { name: labels[locale].menu, exact: true }).click()
  await page.getByRole('menuitem', { name: actionLabel, exact: true }).click()
}

async function downloadBytes(download: Download): Promise<Buffer> {
  const stream = await download.createReadStream()
  if (!stream) throw new Error('Project export download did not expose a readable stream.')
  const chunks: Buffer[] = []
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  return Buffer.concat(chunks)
}

function parseStoredZip(bytes: Buffer): StoredZipFile[] {
  const decoder = new TextDecoder('utf-8', { fatal: true })
  const files: StoredZipFile[] = []
  let offset = 0
  while (offset + 4 <= bytes.length) {
    const signature = bytes.readUInt32LE(offset)
    if (signature === 0x02014b50 || signature === 0x06054b50) break
    if (signature !== 0x04034b50 || offset + 30 > bytes.length) {
      throw new Error('Project export is not a valid stored ZIP archive.')
    }
    const method = bytes.readUInt16LE(offset + 8)
    const compressedSize = bytes.readUInt32LE(offset + 18)
    const uncompressedSize = bytes.readUInt32LE(offset + 22)
    const nameLength = bytes.readUInt16LE(offset + 26)
    const extraLength = bytes.readUInt16LE(offset + 28)
    if (method !== 0 || compressedSize !== uncompressedSize) {
      throw new Error('Project export ZIP must use the uncompressed project format.')
    }
    const nameStart = offset + 30
    const dataStart = nameStart + nameLength + extraLength
    const dataEnd = dataStart + compressedSize
    if (dataEnd > bytes.length) throw new Error('Project export ZIP entry is truncated.')
    const path = decoder.decode(bytes.subarray(nameStart, nameStart + nameLength))
    if (
      !path
      || path.startsWith('/')
      || path.split('/').some((segment) => segment === '..')
    ) {
      throw new Error('Project export ZIP contains an unsafe entry path.')
    }
    files.push({
      contents: decoder.decode(bytes.subarray(dataStart, dataEnd)),
      path,
    })
    offset = dataEnd
  }
  if (files.length === 0) throw new Error('Project export ZIP contains no project files.')
  return files
}

async function writeProjectUploadDirectory(files: StoredZipFile[]): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'inqtrix-detached-transfer-'))
  for (const file of files) {
    const target = join(root, file.path)
    await mkdir(dirname(target), { recursive: true })
    await writeFile(target, file.contents, 'utf8')
  }
  return root
}

type ShareRecord = {
  accepted_at: number | null
  id: string
  permission: 'edit' | 'suggest' | 'view'
  recipient_user_id: string
  revision: number
}

type BrowserApiResult = {
  contentType: string
  ok: boolean
  payload: unknown
  status: number
  transportError: boolean
}

type EditorDocumentDetail = {
  collaboration: {
    persisted_sequence: number
    projection_sequence: number
    projection_updated_at: number | null
    schema_version: number
  }
  content_markdown: string
}

async function editorDocumentDetail(
  page: Page,
  documentId: string,
): Promise<EditorDocumentDetail> {
  const result = await browserApi(page, `/v1/editor/documents/${documentId}`, 'GET')
  requireApiSuccess(result)
  const payload = result.payload as Partial<EditorDocumentDetail> | null
  const collaboration = payload?.collaboration
  if (
    typeof payload?.content_markdown !== 'string'
    || !collaboration
    || !Number.isSafeInteger(collaboration.persisted_sequence)
    || !Number.isSafeInteger(collaboration.projection_sequence)
    || !Number.isSafeInteger(collaboration.schema_version)
    || (
      collaboration.projection_updated_at !== null
      && typeof collaboration.projection_updated_at !== 'number'
    )
  ) {
    throw new Error('Editor document detail omitted the collaboration projection contract.')
  }
  return payload as EditorDocumentDetail
}

async function requireProjectionFlush(page: Page, documentId: string): Promise<void> {
  requireApiSuccess(await browserApi(
    page,
    `/v1/editor/documents/${documentId}/collaboration/projection:flush`,
    'POST',
  ))
}

function expectTextCount(content: string, marker: string, expected: number): void {
  expect(content.split(marker).length - 1).toBe(expected)
}

function apiFailureReason(result: BrowserApiResult): string | null {
  const payload = result.payload as {
    detail?: { error?: { reason?: unknown }; reason?: unknown }
    error?: { reason?: unknown }
  } | null
  const reason = payload?.error?.reason
    ?? payload?.detail?.error?.reason
    ?? payload?.detail?.reason
  return typeof reason === 'string' ? reason : null
}

type BrowserProtocolProbeState = {
  authChallenges: number
  authChallengesAtArm: number
  authenticationDenied: boolean
  closeCodes: number[]
  durableAckHashes: string[]
  errors: string[]
  open: boolean
  scopes: string[]
  synced: boolean
  updateSentAfterChallenge: boolean
}

function createIndependentYjsUpdate(marker: string): { bytes: Uint8Array; hash: string } {
  const document = new Y.Doc()
  try {
    const paragraph = new Y.XmlElement('paragraph')
    const text = new Y.XmlText()
    text.insert(0, marker)
    paragraph.insert(0, [text])
    document.getXmlFragment('content').insert(0, [paragraph])
    const bytes = Y.encodeStateAsUpdate(document)
    return {
      bytes,
      hash: createHash('sha256').update(bytes).digest('hex'),
    }
  } finally {
    document.destroy()
  }
}

async function openBrowserProtocolProbe(page: Page, payload: unknown): Promise<void> {
  const session = parseCollaborationProtocolSession(payload)
  const emptyDocument = new Y.Doc()
  let syncOne: Uint8Array
  let syncTwo: Uint8Array
  try {
    syncOne = encodeProtocolFrame(
      session.room,
      0,
      encodeProtocolVarUint(0),
      encodeProtocolBytes(Y.encodeStateVector(emptyDocument)),
    )
    syncTwo = encodeProtocolFrame(
      session.room,
      0,
      encodeProtocolVarUint(1),
      encodeProtocolBytes(Y.encodeStateAsUpdate(emptyDocument)),
    )
  } finally {
    emptyDocument.destroy()
  }
  const authentication = encodeProtocolFrame(
    session.room,
    2,
    encodeProtocolVarUint(0),
    encodeProtocolString(session.leaseToken),
    encodeProtocolString('4.3.0'),
  )

  await page.evaluate(({ auth, room, sync1, sync2, websocketPath }) => {
    type ProbeRuntime = BrowserProtocolProbeState & {
      authenticationFrame: Uint8Array
      challengeHeld: boolean
      encodeUpdate: (update: Uint8Array) => Uint8Array
      pendingUpdateFrame: Uint8Array | null
      policyChangeCommitted: boolean
      releasePolicyChallenge: () => void
      socket: WebSocket
      syncTwoFrame: Uint8Array
    }
    type ProbeWindow = Window & typeof globalThis & {
      __inqtrixPermissionProbe?: ProbeRuntime
    }

    const decodeBase64 = (value: string): Uint8Array => Uint8Array.from(
      atob(value),
      (character) => character.charCodeAt(0),
    )
    const concat = (...parts: Uint8Array[]): Uint8Array => {
      const result = new Uint8Array(parts.reduce((total, part) => total + part.length, 0))
      let offset = 0
      for (const part of parts) {
        result.set(part, offset)
        offset += part.length
      }
      return result
    }
    const encodeVarUint = (value: number): Uint8Array => {
      const bytes: number[] = []
      let remaining = value
      do {
        let byte = remaining & 0x7f
        remaining = Math.floor(remaining / 128)
        if (remaining > 0) byte |= 0x80
        bytes.push(byte)
      } while (remaining > 0)
      return Uint8Array.from(bytes)
    }
    const encodeBytes = (value: Uint8Array): Uint8Array => concat(
      encodeVarUint(value.length),
      value,
    )
    const encodeString = (value: string): Uint8Array => encodeBytes(
      new TextEncoder().encode(value),
    )
    const encodeFrame = (type: number, ...parts: Uint8Array[]): Uint8Array => concat(
      encodeString(room),
      encodeVarUint(type),
      ...parts,
    )

    class Decoder {
      private offset = 0

      constructor(private readonly bytes: Uint8Array) {}

      get remaining(): number {
        return this.bytes.length - this.offset
      }

      readVarUint(): number {
        let value = 0
        let multiplier = 1
        while (this.offset < this.bytes.length) {
          const byte = this.bytes[this.offset++]!
          value += (byte & 0x7f) * multiplier
          if ((byte & 0x80) === 0) return value
          multiplier *= 128
          if (!Number.isSafeInteger(value) || multiplier > 2 ** 49) break
        }
        throw new Error('invalid varuint')
      }

      readBytes(): Uint8Array {
        const length = this.readVarUint()
        const end = this.offset + length
        if (end > this.bytes.length) throw new Error('truncated bytes')
        const value = this.bytes.subarray(this.offset, end)
        this.offset = end
        return value
      }

      readString(): string {
        return new TextDecoder().decode(this.readBytes())
      }
    }

    const url = new URL(websocketPath, window.location.href)
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    if (url.host !== window.location.host || url.username || url.password) {
      throw new Error('Collaboration websocket_path must stay on the current host.')
    }
    const socket = new WebSocket(url)
    socket.binaryType = 'arraybuffer'
    const sendBinary = (frame: Uint8Array): void => {
      const copy = new Uint8Array(frame.byteLength)
      copy.set(frame)
      socket.send(copy.buffer)
    }
    const runtime: ProbeRuntime = {
      authChallenges: 0,
      authChallengesAtArm: 0,
      authenticationDenied: false,
      authenticationFrame: decodeBase64(auth),
      challengeHeld: false,
      closeCodes: [],
      durableAckHashes: [],
      encodeUpdate: (update) => encodeFrame(
        0,
        encodeVarUint(2),
        encodeBytes(update),
      ),
      errors: [],
      open: false,
      pendingUpdateFrame: null,
      policyChangeCommitted: true,
      releasePolicyChallenge: () => {},
      scopes: [],
      socket,
      synced: false,
      syncTwoFrame: decodeBase64(sync2),
      updateSentAfterChallenge: false,
    }
    runtime.releasePolicyChallenge = () => {
      if (runtime.pendingUpdateFrame && !runtime.updateSentAfterChallenge) {
        sendBinary(runtime.pendingUpdateFrame)
        runtime.updateSentAfterChallenge = true
      }
      sendBinary(runtime.authenticationFrame)
      runtime.challengeHeld = false
    }
    ;(window as ProbeWindow).__inqtrixPermissionProbe?.socket.close(1000)
    ;(window as ProbeWindow).__inqtrixPermissionProbe = runtime

    socket.addEventListener('open', () => {
      runtime.open = true
      sendBinary(runtime.authenticationFrame)
      sendBinary(decodeBase64(sync1))
    })
    socket.addEventListener('close', (event) => {
      runtime.open = false
      runtime.closeCodes.push(event.code)
    })
    socket.addEventListener('error', () => {
      if (runtime.closeCodes.length === 0) runtime.errors.push('websocket transport error')
    })
    socket.addEventListener('message', (event) => {
      try {
        if (!(event.data instanceof ArrayBuffer)) throw new Error('non-binary frame')
        const bytes = new Uint8Array(event.data)
        if (bytes.length === 1 && bytes[0] === 9) {
          sendBinary(Uint8Array.of(10))
          return
        }
        if (bytes.length === 1 && bytes[0] === 10) return
        const decoder = new Decoder(bytes)
        if (decoder.readString() !== room) throw new Error('wrong room')
        const type = decoder.readVarUint()
        if (type === 2) {
          const subtype = decoder.readVarUint()
          if (subtype === 0) {
            runtime.authChallenges += 1
            if (!runtime.policyChangeCommitted) {
              runtime.challengeHeld = true
              return
            }
            runtime.releasePolicyChallenge()
          } else if (subtype === 1) {
            runtime.authenticationDenied = true
            if (decoder.remaining > 0) decoder.readString()
          } else if (subtype === 2) {
            runtime.scopes.push(decoder.readString())
          } else {
            throw new Error('unknown auth subtype')
          }
          return
        }
        if (type === 0 || type === 4) {
          const subtype = decoder.readVarUint()
          if (subtype === 0) {
            decoder.readBytes()
            sendBinary(runtime.syncTwoFrame)
          } else if (subtype === 1 || subtype === 2) {
            decoder.readBytes()
            if (subtype === 1) runtime.synced = true
          } else {
            throw new Error('unknown sync subtype')
          }
          return
        }
        if (type === 5) {
          const stateless = JSON.parse(decoder.readString()) as {
            hash?: unknown
            type?: unknown
          }
          if (
            stateless.type === 'durable_ack'
            && typeof stateless.hash === 'string'
            && /^[a-f0-9]{64}$/.test(stateless.hash)
          ) {
            runtime.durableAckHashes.push(stateless.hash)
          }
          return
        }
        if (type === 1) decoder.readBytes()
        else if (type === 6) decoder.readString()
        else if (type === 7 && decoder.remaining > 0) decoder.readString()
        else if (type === 8) decoder.readVarUint()
      } catch {
        runtime.errors.push('invalid collaboration protocol frame')
      }
    })
  }, {
    auth: Buffer.from(authentication).toString('base64'),
    room: session.room,
    sync1: Buffer.from(syncOne).toString('base64'),
    sync2: Buffer.from(syncTwo).toString('base64'),
    websocketPath: session.websocketPath,
  })

  await expect.poll(async () => {
    const state = await browserProtocolProbeState(page)
    return {
      errors: state.errors,
      open: state.open,
      readWrite: state.scopes.includes('read-write'),
      synced: state.synced,
    }
  }, { timeout: 30_000 }).toEqual({
    errors: [],
    open: true,
    readWrite: true,
    synced: true,
  })
}

async function sendBrowserProtocolProbeUpdate(
  page: Page,
  update: Uint8Array,
): Promise<void> {
  await page.evaluate((encodedUpdate) => {
    type ProbeRuntime = BrowserProtocolProbeState & {
      encodeUpdate: (bytes: Uint8Array) => Uint8Array
      socket: WebSocket
    }
    const runtime = (window as unknown as {
      __inqtrixPermissionProbe?: ProbeRuntime
    }).__inqtrixPermissionProbe
    if (!runtime || !runtime.open || !runtime.synced) {
      throw new Error('Raw collaboration protocol probe is not ready.')
    }
    const updateBytes = Uint8Array.from(
      atob(encodedUpdate),
      (character) => character.charCodeAt(0),
    )
    const frame = runtime.encodeUpdate(updateBytes)
    const copy = new Uint8Array(frame.byteLength)
    copy.set(frame)
    runtime.socket.send(copy.buffer)
  }, Buffer.from(update).toString('base64'))
}

async function armBrowserProtocolProbeUpdate(page: Page, update: Uint8Array): Promise<void> {
  await page.evaluate((encodedUpdate) => {
    type ProbeRuntime = BrowserProtocolProbeState & {
      challengeHeld: boolean
      encodeUpdate: (bytes: Uint8Array) => Uint8Array
      pendingUpdateFrame: Uint8Array | null
      policyChangeCommitted: boolean
      socket: WebSocket
    }
    const runtime = (window as unknown as {
      __inqtrixPermissionProbe?: ProbeRuntime
    }).__inqtrixPermissionProbe
    if (!runtime || !runtime.open || !runtime.synced) {
      throw new Error('Raw collaboration permission probe is not ready.')
    }
    runtime.authChallengesAtArm = runtime.authChallenges
    runtime.challengeHeld = false
    runtime.policyChangeCommitted = false
    const bytes = Uint8Array.from(
      atob(encodedUpdate),
      (character) => character.charCodeAt(0),
    )
    runtime.pendingUpdateFrame = runtime.encodeUpdate(bytes)
  }, Buffer.from(update).toString('base64'))
}

async function commitBrowserProtocolProbePolicyChange(page: Page): Promise<void> {
  await page.evaluate(() => {
    const runtime = (window as unknown as {
      __inqtrixPermissionProbe?: {
        challengeHeld: boolean
        policyChangeCommitted: boolean
        releasePolicyChallenge: () => void
      }
    }).__inqtrixPermissionProbe
    if (!runtime) throw new Error('Raw collaboration permission probe is unavailable.')
    runtime.policyChangeCommitted = true
    if (runtime.challengeHeld) runtime.releasePolicyChallenge()
  })
}

async function browserProtocolProbeState(page: Page): Promise<BrowserProtocolProbeState> {
  return page.evaluate(() => {
    const runtime = (window as unknown as {
      __inqtrixPermissionProbe?: BrowserProtocolProbeState
    }).__inqtrixPermissionProbe
    if (!runtime) throw new Error('Raw collaboration permission probe is unavailable.')
    return {
      authChallenges: runtime.authChallenges,
      authChallengesAtArm: runtime.authChallengesAtArm,
      authenticationDenied: runtime.authenticationDenied,
      closeCodes: [...runtime.closeCodes],
      durableAckHashes: [...runtime.durableAckHashes],
      errors: [...runtime.errors],
      open: runtime.open,
      scopes: [...runtime.scopes],
      synced: runtime.synced,
      updateSentAfterChallenge: runtime.updateSentAfterChallenge,
    }
  })
}

async function closeBrowserProtocolProbe(page: Page): Promise<void> {
  await page.evaluate(() => {
    const holder = window as unknown as {
      __inqtrixPermissionProbe?: { socket: WebSocket }
    }
    const runtime = holder.__inqtrixPermissionProbe
    if (runtime?.socket.readyState === WebSocket.OPEN) runtime.socket.close(1000)
    delete holder.__inqtrixPermissionProbe
  }).catch(() => {})
}

function encodeProtocolFrame(
  room: string,
  type: number,
  ...payloads: Uint8Array[]
): Uint8Array {
  return concatProtocolBytes(
    encodeProtocolString(room),
    encodeProtocolVarUint(type),
    ...payloads,
  )
}

function encodeProtocolString(value: string): Uint8Array {
  return encodeProtocolBytes(new TextEncoder().encode(value))
}

function encodeProtocolBytes(value: Uint8Array): Uint8Array {
  return concatProtocolBytes(encodeProtocolVarUint(value.length), value)
}

function encodeProtocolVarUint(value: number): Uint8Array {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error('Protocol varuint must be a non-negative safe integer.')
  }
  const bytes: number[] = []
  let remaining = value
  do {
    let byte = remaining & 0x7f
    remaining = Math.floor(remaining / 128)
    if (remaining > 0) byte |= 0x80
    bytes.push(byte)
  } while (remaining > 0)
  return Uint8Array.from(bytes)
}

function concatProtocolBytes(...parts: Uint8Array[]): Uint8Array {
  const bytes = new Uint8Array(parts.reduce((total, part) => total + part.length, 0))
  let offset = 0
  for (const part of parts) {
    bytes.set(part, offset)
    offset += part.length
  }
  return bytes
}

async function closeCodeForTextFrame(page: Page, payload: unknown): Promise<number> {
  const websocketPath = (
    payload as { websocket_path?: unknown } | null
  )?.websocket_path
  if (typeof websocketPath !== 'string' || !websocketPath.trim()) {
    throw new Error('Collaboration session omitted websocket_path.')
  }
  return page.evaluate(async (path) => {
    const url = new URL(path, window.location.origin)
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    if (url.host !== window.location.host || url.username || url.password) {
      throw new Error('Collaboration websocket_path must stay on the current host.')
    }
    return new Promise<number>((resolve, reject) => {
      const socket = new WebSocket(url)
      const timer = window.setTimeout(() => {
        socket.close()
        reject(new Error('Collaboration gateway did not reject a text frame within 10 seconds.'))
      }, 10_000)
      socket.addEventListener('open', () => socket.send('invalid-text-frame'))
      socket.addEventListener('close', (event) => {
        window.clearTimeout(timer)
        resolve(event.code)
      })
    })
  }, websocketPath)
}

async function activeEditorShare(page: Page, documentId: string, recipientId: string): Promise<ShareRecord> {
  const params = new URLSearchParams({ resource_id: documentId, resource_type: 'editor_document' })
  const result = await browserApi(page, `/v1/shares?${params.toString()}`, 'GET')
  requireApiSuccess(result)
  const shares = (result.payload as { data?: ShareRecord[] } | null)?.data ?? []
  const share = shares.find((candidate) => candidate.recipient_user_id === recipientId)
  if (!share?.accepted_at || !Number.isSafeInteger(share.revision)) {
    throw new Error('The collaboration fixture requires an accepted, revisioned collaborator share.')
  }
  return share
}

async function updateEditorSharePermission(
  ownerPage: Page,
  share: ShareRecord,
  permission: ShareRecord['permission'],
): Promise<ShareRecord> {
  const result = await browserApi(ownerPage, `/v1/shares/${share.id}`, 'PATCH', {
    expected_revision: share.revision,
    permission,
  })
  requireApiSuccess(result)
  const updated = result.payload as Partial<ShareRecord> | null
  if (
    !updated
    || updated.id !== share.id
    || updated.permission !== permission
    || !Number.isSafeInteger(updated.revision)
    || updated.revision !== share.revision + 1
  ) {
    throw new Error('Share permission update omitted its compare-and-swap revision contract.')
  }
  return updated as ShareRecord
}

async function restoreEditorShare(
  ownerPage: Page,
  collaboratorPage: Page,
  documentId: string,
  recipientId: string,
  permission: ShareRecord['permission'],
): Promise<void> {
  const params = new URLSearchParams({ resource_id: documentId, resource_type: 'editor_document' })
  const listed = await browserApi(ownerPage, `/v1/shares?${params.toString()}`, 'GET')
  requireApiSuccess(listed)
  let share = ((listed.payload as { data?: ShareRecord[] } | null)?.data ?? [])
    .find((candidate) => candidate.recipient_user_id === recipientId)
  if (!share) {
    const created = await browserApi(ownerPage, '/v1/shares', 'POST', {
      invitees: [{ permission, user_id: recipientId }],
      resource_id: documentId,
      resource_type: 'editor_document',
    })
    requireApiSuccess(created)
    share = (created.payload as { data?: ShareRecord[] } | null)?.data?.[0]
  }
  if (!share) throw new Error('The collaboration share could not be restored.')
  if (share.accepted_at === null) {
    requireApiSuccess(await browserApi(collaboratorPage, `/v1/shares/${share.id}/accept`, 'POST'))
  }
  if (share.permission !== permission) {
    if (!Number.isSafeInteger(share.revision)) {
      throw new Error('The collaboration share restore response omitted revision.')
    }
    await updateEditorSharePermission(ownerPage, share, permission)
  }
}

async function browserApi(
  page: Page,
  path: string,
  method: 'DELETE' | 'GET' | 'PATCH' | 'POST',
  body?: unknown,
  timeoutMs = 10_000,
): Promise<BrowserApiResult> {
  return page.evaluate(async ({ requestBody, requestMethod, requestPath, requestTimeoutMs }) => {
    const headers = new Headers({ Accept: 'application/json' })
    if (requestBody !== undefined) headers.set('Content-Type', 'application/json')
    if (requestMethod !== 'GET') {
      const cookies = Object.fromEntries(document.cookie.split(';').map((entry) => {
        const [name, ...value] = entry.trim().split('=')
        return [name, decodeURIComponent(value.join('='))]
      }))
      const csrf = cookies['__Host-inqtrix_csrf'] ?? cookies.inqtrix_csrf
      if (csrf) headers.set('X-CSRF-Token', csrf)
    }
    try {
      const response = await fetch(requestPath, {
        body: requestBody === undefined ? undefined : JSON.stringify(requestBody),
        credentials: 'same-origin',
        headers,
        method: requestMethod,
        signal: AbortSignal.timeout(requestTimeoutMs),
      })
      let payload: unknown = null
      try {
        payload = await response.json()
      } catch {
        payload = null
      }
      return {
        contentType: response.headers.get('content-type')?.toLowerCase() ?? '',
        ok: response.ok,
        payload,
        status: response.status,
        transportError: false,
      }
    } catch {
      return {
        contentType: '',
        ok: false,
        payload: null,
        status: 0,
        transportError: true,
      }
    }
  }, {
    requestBody: body,
    requestMethod: method,
    requestPath: path,
    requestTimeoutMs: timeoutMs,
  })
}

function requireApiSuccess(result: BrowserApiResult): void {
  if (result.ok) return
  const reason = apiFailureReason(result)
  throw new Error(`Collaboration fixture API request failed with HTTP ${result.status}${
    reason ? ` (${reason})` : ''
  }.`)
}

function assertFastApiHealth(result: BrowserApiResult): void {
  if (isFastApiHealth(result)) return
  throw new Error('The public /health probe did not return the Inqtrix FastAPI JSON contract.')
}

function isFastApiHealth(result: BrowserApiResult): boolean {
  const payload = isRecord(result.payload) ? result.payload : null
  const validProvider = (value: unknown): boolean => (
    isRecord(value)
    && typeof value.provider === 'string'
    && value.provider.trim().length > 0
    && typeof value.status === 'string'
    && value.status.trim().length > 0
  )
  return (
    result.ok
    && result.status === 200
    && result.contentType.includes('application/json')
    && payload?.status === 'ok'
    && validProvider(payload.llm)
    && validProvider(payload.search)
    && typeof payload.auth_mode === 'string'
    && isRecord(payload.legal)
  )
}

function isPublicGatewayUnavailable(result: BrowserApiResult): boolean {
  return !result.ok && (
    result.transportError
    || [500, 502, 503, 504].includes(result.status)
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function uniqueMarker(kind: string): string {
  return `inqtrix-e2e-${kind}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}
