import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import {
  editorDocumentShareDisabled,
  EditorRemoteDeletionRecoveryBar,
  editorRecoveryMarkdown,
} from './EditorWorkspace'
import type { EditorDocumentRecord } from '@/features/project/types'

describe('editor recovery markdown', () => {
  it('keeps the unconfirmed document body and appends unsent comment drafts', () => {
    expect(editorRecoveryMarkdown({
      commentDrafts: ['First local note', 'Second **draft**'],
      documentMarkdown: '# Unconfirmed body\n\nChanged locally.',
      locale: 'en',
      title: 'Original.md',
    })).toBe(
      '# Unconfirmed body\n\nChanged locally.'
      + '\n\n---\n\n## Unsent comment drafts'
      + '\n\n### Draft 1\n\nFirst local note'
      + '\n\n### Draft 2\n\nSecond **draft**',
    )
  })

  it('preserves a comment-only recovery without copying the confirmed body', () => {
    expect(editorRecoveryMarkdown({
      commentDrafts: ['Noch nicht gesendet'],
      documentMarkdown: '',
      locale: 'de',
      title: 'Remote gelöscht.md',
    })).toBe(
      '# Remote gelöscht.md'
      + '\n\n---\n\n## Nicht gesendete Kommentarentwürfe'
      + '\n\n### Entwurf 1\n\nNoch nicht gesendet',
    )
  })

  it('renders a visibly anchored recovery boundary with explicit decisions', () => {
    const markup = renderToStaticMarkup(createElement(
      EditorRemoteDeletionRecoveryBar,
      {
        locale: 'de',
        onDiscard: () => undefined,
        onSaveAsNew: () => undefined,
      },
    ))

    expect(markup).toContain('data-editor-recovery-artifact')
    expect(markup).toContain('role="alert"')
    expect(markup).toContain('Als neues Dokument speichern')
    expect(markup).toContain('Verwerfen')
    expect(markup).toContain('wird nicht automatisch synchronisiert')
  })

  it('keeps sharing disabled until the recovery copy is promoted', () => {
    const document = {
      contentMarkdown: '# Local recovery',
      createdAt: '2026-07-29T10:00:00.000Z',
      folderId: null,
      id: 'editor-recovery-local',
      recovery: {
        capturedAt: '2026-07-29T10:00:00.000Z',
        originalDocumentId: 'deleted-server-document',
        reason: 'remote_deleted',
      },
      revision: 0,
      source: 'blank',
      title: 'Recovered.md',
      updatedAt: '2026-07-29T10:00:00.000Z',
    } satisfies EditorDocumentRecord

    expect(editorDocumentShareDisabled({
      canFlushForShare: true,
      document,
      isDirty: false,
      sharingAvailable: true,
    })).toBe(true)
    expect(editorDocumentShareDisabled({
      canFlushForShare: true,
      document: { ...document, recovery: undefined },
      isDirty: false,
      sharingAvailable: true,
    })).toBe(false)
  })
})
