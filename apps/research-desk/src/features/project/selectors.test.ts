import { describe, expect, it } from 'vitest'
import { createEmptyProjectState } from './seedProject'
import {
  chatRuleOptions,
  chatAttachmentsFromRefs,
  chatContextRefKey,
  completedReportOptions,
  dedupeChatContextRefs,
  displayRelativeAge,
  isResearchDeskRun,
  projectAllKnowledgeItems,
  projectChatRules,
  projectKnowledgeItems,
  projectKnowledgeSessionSections,
  projectKnowledgeSessions,
  referenceDocsFromRefs,
} from './selectors'
import type {
  ChatRuleRecord,
  FileAssetRecord,
  FileGroupRecord,
  KnowledgeSessionGroupRecord,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
  ProjectState,
  ResearchRunRecord,
} from './types'

describe('displayRelativeAge', () => {
  const now = new Date('2026-06-26T12:00:00.000Z')

  it('formats compact German age labels', () => {
    expect(displayRelativeAge('2026-06-26T11:59:20.000Z', 'de', now)).toBe('Gerade eben')
    expect(displayRelativeAge('2026-06-26T11:54:00.000Z', 'de', now)).toBe('6 Min.')
    expect(displayRelativeAge('2026-06-26T10:00:00.000Z', 'de', now)).toBe('2 Std.')
    expect(displayRelativeAge('2026-06-23T12:00:00.000Z', 'de', now)).toBe('3 Tage')
    expect(displayRelativeAge('2026-06-05T12:00:00.000Z', 'de', now)).toBe('3 W')
  })

  it('formats compact English age labels', () => {
    expect(displayRelativeAge('2026-06-26T11:59:20.000Z', 'en', now)).toBe('Just now')
    expect(displayRelativeAge('2026-06-26T11:54:00.000Z', 'en', now)).toBe('6 min')
    expect(displayRelativeAge('2026-06-26T10:00:00.000Z', 'en', now)).toBe('2 h')
    expect(displayRelativeAge('2026-06-23T12:00:00.000Z', 'en', now)).toBe('3 d')
    expect(displayRelativeAge('2026-06-05T12:00:00.000Z', 'en', now)).toBe('3 w')
  })

  it('returns an empty label for invalid dates', () => {
    expect(displayRelativeAge('not-a-date', 'de', now)).toBe('')
    expect(displayRelativeAge('2026-06-26T12:00:00.000Z', 'en', new Date('invalid'))).toBe('')
  })
})

function makeReportRun(
  runId: string,
  mode: ResearchRunRecord['mode'],
): ResearchRunRecord {
  // Only the fields completedReportOptions reads matter here; the cast keeps
  // the test focused on the mode filter, not on a full run-record fixture.
  return {
    mode,
    result: { markdown: `# ${runId}` },
    runId,
    status: 'completed',
    summary: { title: runId },
  } as unknown as ResearchRunRecord
}

function makeAsset(id: string, label: string, overrides: Partial<FileAssetRecord> = {}): FileAssetRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    extractedText: `${label} content`,
    fileName: `${label}.txt`,
    groupId: null,
    id,
    label,
    mimeType: 'text/plain',
    origin: 'library',
    pageCount: null,
    parseStatus: 'parsed',
    parseWarning: null,
    sectionId: 'file-section-temp',
    sizeBytes: 12,
    textTruncated: false,
    title: `${label}.txt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeRule(id: string, label: string, overrides: Partial<ChatRuleRecord> = {}): ChatRuleRecord {
  return {
    contentMarkdown: `${label} prompt`,
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    label,
    title: `${label} prompt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeKnowledgeItem(
  id: string,
  sessionId: string,
  overrides: Partial<KnowledgeThreadItemRecord> = {},
): KnowledgeThreadItemRecord {
  return {
    collectionTitles: [],
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    progress: { steps: [] },
    question: id,
    requestedProfile: null,
    runId: null,
    sessionId,
    status: 'completed',
    ...overrides,
  }
}

function makeKnowledgeSession(
  id: string,
  title: string,
  overrides: Partial<KnowledgeSessionRecord> = {},
): KnowledgeSessionRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeKnowledgeSessionGroup(
  id: string,
  title: string,
  overrides: Partial<KnowledgeSessionGroupRecord> = {},
): KnowledgeSessionGroupRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function stateWith(assets: FileAssetRecord[], groups: FileGroupRecord[] = []): ProjectState {
  const base = createEmptyProjectState()
  return {
    ...base,
    fileAssetOrder: assets.map((asset) => asset.id),
    fileAssets: Object.fromEntries(assets.map((asset) => [asset.id, asset])),
    fileGroupOrder: groups.map((group) => group.id),
    fileGroups: Object.fromEntries(groups.map((group) => [group.id, group])),
  }
}

function stateWithRules(rules: ChatRuleRecord[]): ProjectState {
  const base = createEmptyProjectState()
  return {
    ...base,
    chatRuleOrder: rules.map((rule) => rule.id),
    chatRules: Object.fromEntries(rules.map((rule) => [rule.id, rule])),
  }
}

describe('chatContextRefKey', () => {
  it('produces a stable key per reference kind', () => {
    expect(chatContextRefKey({ fileId: 'f1', kind: 'file-asset' })).toBe('file-asset:f1')
    expect(chatContextRefKey({ groupId: 'g1', kind: 'file-group' })).toBe('file-group:g1')
  })
})

describe('dedupeChatContextRefs', () => {
  it('keeps the first occurrence of each reference', () => {
    const refs = dedupeChatContextRefs([
      { fileId: 'f1', kind: 'file-asset' },
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
    expect(refs).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
  })
})

describe('chatAttachmentsFromRefs', () => {
  it('resolves a file-asset reference to one attachment', () => {
    const state = stateWith([makeAsset('f1', 'alpha')])
    const attachments = chatAttachmentsFromRefs(state, [{ fileId: 'f1', kind: 'file-asset' }])
    expect(attachments).toHaveLength(1)
    expect(attachments[0]).toMatchObject({ contentMarkdown: 'alpha content', kind: 'file-asset', label: 'alpha' })
  })

  it('expands a file-group reference to one attachment per member', () => {
    const group: FileGroupRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'g1',
      sectionId: 'file-section-temp',
      title: 'Dossier',
      updatedAt: '2026-01-01T00:00:00.000Z',
    }
    const state = stateWith(
      [makeAsset('f1', 'alpha', { groupId: 'g1' }), makeAsset('f2', 'beta', { groupId: 'g1' })],
      [group],
    )
    const attachments = chatAttachmentsFromRefs(state, [{ groupId: 'g1', kind: 'file-group' }])
    expect(attachments.map((attachment) => attachment.label)).toEqual(['alpha', 'beta'])
    expect(attachments.every((attachment) => attachment.kind === 'file-group')).toBe(true)
  })

  it('renders context-pack rules with linked database files as a stored snapshot', () => {
    const state = {
      ...stateWith([makeAsset('f1', 'alpha')]),
      chatRuleOrder: ['r1'],
      chatRules: {
        r1: makeRule('r1', 'cv', {
          category: 'context',
          contentMarkdown: 'Use this profile context:\n{{context}}\nThen answer.',
          linkedContextRefs: [{ fileId: 'f1', kind: 'file-asset' }],
        }),
      },
    }

    const attachments = chatAttachmentsFromRefs(state, [{ kind: 'chat-rule', ruleId: 'r1' }])

    expect(attachments).toHaveLength(1)
    expect(attachments[0]).toMatchObject({ kind: 'chat-rule', label: 'cv' })
    expect(attachments[0].contentMarkdown).toContain('Use this profile context:')
    expect(attachments[0].contentMarkdown).toContain('--- [1] @files:alpha ---')
    expect(attachments[0].contentMarkdown).toContain('alpha content')
    expect(attachments[0].contentMarkdown).toContain('Then answer.')
    expect(attachments[0].contentMarkdown).not.toContain('{{context}}')
  })
})

describe('projectChatRules', () => {
  it('normalizes legacy rules without new prompt-library fields', () => {
    const state = stateWithRules([makeRule('r1', 'legacy')])

    expect(projectChatRules(state)[0]).toMatchObject({
      category: 'instruction',
      includeInAutocomplete: true,
      linkedContextRefs: [],
      visibility: { chat: true, editor: true },
    })
  })
})

describe('knowledge selectors', () => {
  it('filters thread items to the selected knowledge session', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const state: ProjectState = {
      ...base,
      knowledgeItemOrder: ['ki-1', 'ki-2'],
      knowledgeItems: {
        'ki-1': makeKnowledgeItem('ki-1', defaultSessionId),
        'ki-2': makeKnowledgeItem('ki-2', 'ks-2'),
      },
      knowledgeSessionOrder: [defaultSessionId, 'ks-2'],
      knowledgeSessions: {
        ...base.knowledgeSessions,
        'ks-2': {
          createdAt: '2026-01-01T00:00:00.000Z',
          id: 'ks-2',
          title: 'Second',
          updatedAt: '2026-01-01T00:00:00.000Z',
        },
      },
      selectedKnowledgeSessionId: 'ks-2',
    }

    expect(projectKnowledgeItems(state).map((item) => item.id)).toEqual(['ki-2'])
    expect(projectAllKnowledgeItems(state).map((item) => item.id)).toEqual(['ki-1', 'ki-2'])
    expect(projectKnowledgeSessions(state).map((session) => session.id)).toEqual([defaultSessionId, 'ks-2'])
  })

  it('groups knowledge sessions into folder sections and keeps invalid memberships ungrouped', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const state: ProjectState = {
      ...base,
      knowledgeSessionGroupMemberships: {
        'ks-2': 'kg-1',
        'ks-3': 'missing-group',
      },
      knowledgeSessionGroupOrder: ['kg-1'],
      knowledgeSessionGroups: {
        'kg-1': makeKnowledgeSessionGroup('kg-1', 'Client work'),
      },
      knowledgeSessionOrder: [defaultSessionId, 'ks-2', 'ks-3'],
      knowledgeSessions: {
        ...base.knowledgeSessions,
        'ks-2': makeKnowledgeSession('ks-2', 'Inside folder'),
        'ks-3': makeKnowledgeSession('ks-3', 'Loose session'),
      },
    }

    const sections = projectKnowledgeSessionSections(state)

    expect(sections).toHaveLength(2)
    expect(sections[0]).toMatchObject({ groupId: 'kg-1', kind: 'group' })
    expect(sections[0].sessions.map((session) => session.id)).toEqual(['ks-2'])
    expect(sections[1]).toMatchObject({ groupId: null, kind: 'ungrouped' })
    expect(sections[1].sessions.map((session) => session.id)).toEqual([defaultSessionId, 'ks-3'])
  })
})

describe('chatRuleOptions', () => {
  it('filters autocomplete options by surface visibility and autocomplete status', () => {
    const state = stateWithRules([
      makeRule('r1', 'chat-only', {
        category: 'instruction',
        visibility: { chat: true, editor: false },
      }),
      makeRule('r2', 'editor-only', {
        category: 'function',
        visibility: { chat: false, editor: true },
      }),
      makeRule('r3', 'hidden-autocomplete', {
        category: 'context',
        includeInAutocomplete: false,
        visibility: { chat: true, editor: true },
      }),
    ])

    expect(chatRuleOptions(state, 'chat').map((rule) => rule.label)).toEqual(['chat-only'])
    expect(chatRuleOptions(state, 'editor').map((rule) => rule.label)).toEqual(['editor-only'])
    expect(chatRuleOptions(state).map((rule) => rule.category)).toEqual(['instruction', 'function', 'context'])
  })
})

describe('referenceDocsFromRefs', () => {
  it('maps file and research-report refs to ReferenceDoc DTOs and skips chat rules', () => {
    const base = stateWith([makeAsset('f1', 'alpha', { pageCount: 3 })])
    const report = {
      result: { markdown: 'report body' },
      runId: 'r1',
      status: 'completed',
      summary: { title: 'My Report' },
    } as unknown as ProjectState['researchRuns'][string]
    const state: ProjectState = {
      ...base,
      researchRunOrder: ['r1'],
      researchRuns: { r1: report },
    }
    const docs = referenceDocsFromRefs(state, [
      { fileId: 'f1', kind: 'file-asset' },
      { kind: 'research-report', runId: 'r1' },
      { kind: 'chat-rule', ruleId: 'does-not-exist' },
    ])
    expect(docs).toEqual([
      { content: 'alpha content', label: 'alpha', pageCount: 3, sizeBytes: 12 },
      { content: 'report body', label: 'my-report', pageCount: null, sizeBytes: undefined },
    ])
  })
})

describe('isResearchDeskRun', () => {
  it('excludes only knowledge-mode runs; research/direct_llm/legacy count as desk runs', () => {
    expect(isResearchDeskRun({ mode: 'knowledge' } as ResearchRunRecord)).toBe(false)
    expect(isResearchDeskRun({ mode: 'research' } as ResearchRunRecord)).toBe(true)
    expect(isResearchDeskRun({ mode: 'direct_llm' } as ResearchRunRecord)).toBe(true)
    // A run with no mode is a legacy record and stays visible as a report.
    expect(isResearchDeskRun({} as ResearchRunRecord)).toBe(true)
  })
})

describe('completedReportOptions', () => {
  it('offers research/direct_llm/legacy reports but never knowledge-mode runs', () => {
    const state = createEmptyProjectState()
    state.researchRuns = {
      r1: makeReportRun('r1', 'research'),
      r2: makeReportRun('r2', 'direct_llm'),
      r3: makeReportRun('r3', 'knowledge'),
      r4: makeReportRun('r4', undefined),
    }
    state.researchRunOrder = ['r1', 'r2', 'r3', 'r4']

    const runIds = completedReportOptions(state).map((option) => option.runId)

    // The editor "import report" + chat @research lists must match the desk:
    // knowledge runs (the "Wissen" thread) never leak in as importable reports.
    expect(runIds).toEqual(['r1', 'r2', 'r4'])
  })
})
