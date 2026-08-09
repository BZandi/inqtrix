import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { LocaleProvider } from '@/i18n/LocaleProvider'
import { EditorCollaborationStatus } from './EditorInspector'
import { EditorTopBarLayout } from './EditorTopBarLayout'
import { buildEditorCollaborationStatusModel } from './model'

const participants = [
  { color: '#2563EB', id: 'ada', name: 'Ada Lovelace' },
  { color: '#047857', id: 'lin', name: 'Lin Chen' },
  { color: '#B45309', id: 'max', name: 'Max Weber' },
  { color: '#7C3AED', id: 'zoe', name: 'Zoe Smith' },
]

function statusMarkup() {
  const model = buildEditorCollaborationStatusModel({
    access: 'edit',
    active: true,
    canEdit: true,
    connectionStatus: 'connected',
    durabilityStatus: 'saved',
    participants,
    projectionUpdatedAt: '2026-07-15T10:02:00.000Z',
    synced: true,
  })
  return renderToStaticMarkup(
    <LocaleProvider>
      <EditorCollaborationStatus collaborationExpected model={model} variant="topbar" />
    </LocaleProvider>,
  )
}

describe('EditorCollaborationStatus', () => {
  it('announces projection confirmation and every participant while previewing only three', () => {
    const markup = statusMarkup()

    expect(markup).toContain('role="status"')
    expect(markup).toMatch(
      /<div(?=[^>]*role="status")(?=[^>]*data-editor-status-kind="saved")[^>]*>/,
    )
    expect(markup).not.toMatch(/<span[^>]*data-editor-status-kind=/)
    expect(markup).toContain('Bestätigter Stand')
    expect(markup).toContain('4 Teilnehmende: Ada Lovelace, Lin Chen, Max Weber, Zoe Smith')
    expect(markup).toContain('data-participant-count="4"')
    expect(markup.match(/data-participant-id=/g)).toHaveLength(3)
    expect(markup).toContain('+1')
  })

  it('keeps a long title and four-person status in bounded narrow topbar tracks', () => {
    const longTitle = 'A very long collaboration document title that must stay inside its own track'
    const markup = renderToStaticMarkup(
      <EditorTopBarLayout
        actions={(
          <LocaleProvider>
            <EditorCollaborationStatus
              collaborationExpected
              model={buildEditorCollaborationStatusModel({
                access: 'edit',
                active: true,
                canEdit: true,
                connectionStatus: 'connected',
                durabilityStatus: 'saved',
                participants,
                synced: true,
              })}
              variant="topbar"
            />
          </LocaleProvider>
        )}
        leading={<span className="min-w-0 truncate">{longTitle}</span>}
        primary={<span>Mode and status</span>}
        toolbar={<span>Toolbar</span>}
      />,
    )

    expect(markup).toContain('data-editor-topbar="true"')
    expect(markup).toContain('data-editor-topbar-leading="true"')
    expect(markup).toContain('data-editor-topbar-toolbar="true"')
    expect(markup).toContain('data-editor-topbar-primary="true"')
    expect(markup).toContain('data-editor-topbar-actions="true"')
    expect(markup).toContain('overflow-hidden')
    expect(markup).not.toContain('overflow-x-auto')
    expect(markup).toContain('Mode and status')
    expect(markup).toContain(`>${longTitle}</span>`)
    expect(markup).toContain('data-participant-count="4"')
  })

  it('does not duplicate punctuation between a notice and projection metadata', () => {
    const markup = renderToStaticMarkup(
      <LocaleProvider>
        <EditorCollaborationStatus
          collaborationExpected
          model={buildEditorCollaborationStatusModel({
            access: 'view',
            active: true,
            canEdit: false,
            connectionStatus: 'access_revoked',
            durabilityStatus: 'saved',
            notice: 'Der Zugriff wurde entzogen.',
            participants: [],
            projectionUpdatedAt: '2026-07-15T10:02:00.000Z',
            synced: false,
          })}
          variant="topbar"
        />
      </LocaleProvider>,
    )

    expect(markup).toContain(
      'Zugriff entzogen. Der Zugriff wurde entzogen. Bestätigter Stand',
    )
    expect(markup).not.toContain('entzogen..')
  })
})
