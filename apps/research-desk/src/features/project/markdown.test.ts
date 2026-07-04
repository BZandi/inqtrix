import { describe, expect, it } from 'vitest'
import { createDefaultFileLibrarySections } from '@/features/files/sections'
import { createEmptyProjectState } from './seedProject'
import {
  buildProjectFiles,
  parseChatThread,
  parseChatRule,
  parseFileAsset,
  parseProjectManifest,
  serializeChatThread,
  serializeChatRule,
  serializeFileAsset,
  serializeProjectManifest,
} from './markdown'
import type {
  ChatRuleRecord,
  ChatThreadRecord,
  FileAssetRecord,
  FileGroupRecord,
  ProjectState,
  ResearchRunRecord,
  VectorIndexRecord,
} from './types'

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

function makeRule(overrides: Partial<ChatRuleRecord> = {}): ChatRuleRecord {
  return {
    contentMarkdown: 'Follow the profile context.',
    createdAt: '2026-01-01T00:00:00.000Z',
    id: 'rule-1',
    label: 'profile',
    title: 'Profile',
    updatedAt: '2026-01-02T00:00:00.000Z',
    ...overrides,
  }
}

function makeResearchRun(id: string, overrides: Partial<ResearchRunRecord> = {}): ResearchRunRecord {
  return {
    agentOverrides: {},
    createdAt: '2026-01-01T00:00:00.000Z',
    events: [],
    finishedAt: '2026-01-01T00:01:00.000Z',
    metrics: { claims: 0, queries: 0, rounds: '0', sources: 0 },
    mode: 'research',
    phaseState: { activePhase: 'answer', completedPhases: ['analysis', 'planning', 'search', 'evaluation', 'answer'] },
    result: {
      markdown: `# ${id}`,
      references: [],
      topClaims: [],
      topSources: [],
    },
    runId: id,
    source: 'api',
    stack: 'test-stack',
    startedAt: '2026-01-01T00:00:01.000Z',
    status: 'completed',
    submittedAt: '2026-01-01T00:00:00.000Z',
    summary: { title: id },
    ...overrides,
  }
}

describe('serializeChatRule / parseChatRule', () => {
  it('parses legacy rule files with prompt-library defaults', () => {
    const parsed = parseChatRule([
      '---',
      'created_at: 2026-01-01T00:00:00.000Z',
      'kind: inqtrix.chat_rule',
      'label: legacy',
      'rule_id: rule-legacy',
      'schema_version: 1',
      'title: Legacy',
      'updated_at: 2026-01-02T00:00:00.000Z',
      '---',
      'Legacy instruction.',
    ].join('\n'))

    expect(parsed).toEqual({
      category: 'instruction',
      contentMarkdown: 'Legacy instruction.',
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'rule-legacy',
      includeInAutocomplete: true,
      label: 'legacy',
      linkedContextRefs: [],
      title: 'Legacy',
      updatedAt: '2026-01-02T00:00:00.000Z',
      visibility: { chat: true, editor: true },
    })
  })

  it('round-trips prompt-library frontmatter additively', () => {
    const rule = makeRule({
      category: 'context',
      includeInAutocomplete: false,
      linkedContextRefs: [
        { fileId: 'file-1', kind: 'file-asset' },
        { groupId: 'group-1', kind: 'file-group' },
        { kind: 'chat-rule', ruleId: 'ignored-linked-rule' },
      ],
      visibility: { chat: true, editor: false },
    })

    const file = serializeChatRule(rule)
    const parsed = parseChatRule(file.contents)

    expect(file.path).toBe('rules/profile.md')
    expect(parsed).toEqual({
      ...rule,
      linkedContextRefs: [
        { fileId: 'file-1', kind: 'file-asset' },
        { groupId: 'group-1', kind: 'file-group' },
      ],
    })
  })
})

describe('parseFrontmatter line-ending tolerance (Windows/CRLF)', () => {
  const ruleLines = [
    '---',
    'created_at: 2026-01-01T00:00:00.000Z',
    'kind: inqtrix.chat_rule',
    'label: legacy',
    'rule_id: rule-legacy',
    'schema_version: 1',
    'title: Legacy',
    'updated_at: 2026-01-02T00:00:00.000Z',
    '---',
    'Legacy instruction.',
  ]
  const expected: ChatRuleRecord & Record<string, unknown> = {
    category: 'instruction',
    contentMarkdown: 'Legacy instruction.',
    createdAt: '2026-01-01T00:00:00.000Z',
    id: 'rule-legacy',
    includeInAutocomplete: true,
    label: 'legacy',
    linkedContextRefs: [],
    title: 'Legacy',
    updatedAt: '2026-01-02T00:00:00.000Z',
    visibility: { chat: true, editor: true },
  }

  it('accepts CRLF frontmatter from a Windows autocrlf checkout', () => {
    expect(parseChatRule(ruleLines.join('\r\n'))).toEqual(expected)
  })

  it('accepts lone-CR frontmatter from a legacy-Mac checkout', () => {
    expect(parseChatRule(ruleLines.join('\r'))).toEqual(expected)
  })
})

describe('serializeChatThread / parseChatThread', () => {
  it('round-trips assistant request context metadata additively', () => {
    const thread: ChatThreadRecord = {
      createdAt: '2026-06-26T12:00:00.000Z',
      id: 'ct_1',
      messages: [
        {
          contentMarkdown: 'Use knowledge mode.',
          createdAt: '2026-06-26T12:01:00.000Z',
          id: 'cm_u1',
          role: 'user',
        },
        {
          contentMarkdown: 'Knowledge-grounded answer.',
          createdAt: '2026-06-26T12:02:00.000Z',
          id: 'cm_a1',
          requestContext: { knowledgeCollectionIds: ['kc_1', 'kc_2'] },
          role: 'assistant',
        },
      ],
      preview: 'Knowledge-grounded answer.',
      source: 'api',
      title: 'Knowledge retry',
      updatedAt: '2026-06-26T12:02:00.000Z',
    }

    const parsed = parseChatThread(serializeChatThread(thread).contents)

    expect(parsed.messages[1].requestContext).toEqual({ knowledgeCollectionIds: ['kc_1', 'kc_2'] })
    expect(parsed.messages.map((message) => message.contentMarkdown)).toEqual([
      'Use knowledge mode.',
      'Knowledge-grounded answer.',
    ])
  })
})

describe('serializeFileAsset / parseFileAsset', () => {
  it('round-trips an asset with all metadata', () => {
    const asset = makeAsset('f1', 'alpha', {
      groupId: 'g1',
      origin: 'chat',
      pageCount: 3,
      parseStatus: 'partial',
      parseWarning: 'Document shortened',
      sizeBytes: 42,
      textTruncated: true,
    })
    const file = serializeFileAsset(asset)
    expect(file.path).toBe('files/alpha.md')
    expect(parseFileAsset(file.contents)).toEqual(asset)
  })

  it('round-trips a null group and page count', () => {
    const asset = makeAsset('f2', 'beta')
    expect(parseFileAsset(serializeFileAsset(asset).contents)).toEqual(asset)
  })
})

describe('project file export plan', () => {
  it('exports completed research runs but skips knowledge-mode runs', () => {
    const researchRun = makeResearchRun('run-research')
    const knowledgeRun = makeResearchRun('run-knowledge', { mode: 'knowledge' })
    const state: ProjectState = {
      ...createEmptyProjectState(),
      researchRunOrder: ['run-research', 'run-knowledge'],
      researchRuns: {
        'run-knowledge': knowledgeRun,
        'run-research': researchRun,
      },
    }

    const paths = buildProjectFiles(state).map((file) => file.path)

    expect(paths.some((path) => path.includes('run-research'))).toBe(true)
    expect(paths.some((path) => path.includes('run-knowledge'))).toBe(false)
  })
})

describe('project manifest file library', () => {
  it('round-trips pinned explorer UI state additively', () => {
    const base = createEmptyProjectState()
    const state: ProjectState = {
      ...base,
      ui: {
        ...base.ui,
        pinnedExplorer: {
          chatThreadIds: ['ct-1'],
          editorDocumentIds: ['doc-1'],
          knowledgeSessionIds: ['ks-1'],
        },
      },
    }

    const data = parseProjectManifest(serializeProjectManifest(state).contents)

    expect((data.ui as Record<string, unknown>).pinnedExplorer).toEqual({
      chatThreadIds: ['ct-1'],
      editorDocumentIds: ['doc-1'],
      knowledgeSessionIds: ['ks-1'],
    })
  })

  it('serializes and re-parses sections, groups and asset order', () => {
    const sections = createDefaultFileLibrarySections('2026-01-01T00:00:00.000Z')
    const group: FileGroupRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'g1',
      sectionId: sections[1].id,
      title: 'Group',
      updatedAt: '2026-01-01T00:00:00.000Z',
    }
    const asset = makeAsset('f1', 'alpha', { groupId: 'g1', sectionId: sections[1].id })
    const state: ProjectState = {
      ...createEmptyProjectState(),
      fileAssetOrder: ['f1'],
      fileAssets: { f1: asset },
      fileGroupOrder: ['g1'],
      fileGroups: { g1: group },
      fileLibrarySectionOrder: sections.map((section) => section.id),
      fileLibrarySections: Object.fromEntries(sections.map((section) => [section.id, section])),
    }

    const data = parseProjectManifest(serializeProjectManifest(state).contents)

    expect(data.file_section_order).toEqual(sections.map((section) => section.id))
    expect((data.file_sections as unknown[]).length).toBe(3)
    expect(data.file_group_order).toEqual(['g1'])
    expect(data.file_asset_order).toEqual(['f1'])
  })
})

describe('project manifest vector indexes', () => {
  it('round-trips vector index order and pending/embedded members', () => {
    const index: VectorIndexRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      dims: 3072,
      handle: 'eu-recht',
      id: 'idx1',
      members: [
        { fileId: 'f1', state: 'embedded' },
        { fileId: 'f2', state: 'pending' },
      ],
      model: 'text-embedding-3-large',
      status: 'stale',
      title: 'EU Recht',
      updatedAt: '2026-01-02T00:00:00.000Z',
    }
    const state: ProjectState = {
      ...createEmptyProjectState(),
      vectorIndexOrder: ['idx1'],
      vectorIndexes: { idx1: index },
    }

    const data = parseProjectManifest(serializeProjectManifest(state).contents)

    expect(data.vector_index_order).toEqual(['idx1'])
    const serialized = data.vector_indexes as Array<Record<string, unknown>>
    expect(serialized).toHaveLength(1)
    expect(serialized[0].handle).toBe('eu-recht')
    expect(serialized[0].members).toEqual([
      { fileId: 'f1', state: 'embedded' },
      { fileId: 'f2', state: 'pending' },
    ])
  })
})
