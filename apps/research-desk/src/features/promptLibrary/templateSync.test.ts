import { describe, expect, it } from 'vitest'
import {
  canDeleteRule,
  canEditRule,
  isSameSyncedRule,
  ruleFromTemplate,
  staleSyncedRuleIds,
  templatePayloadFromRule,
} from './templateSync'
import type { PromptTemplateInfo } from './templateSync'
import type { ChatRuleRecord } from '@/features/project/types'

const TEMPLATE: PromptTemplateInfo = {
  category: 'instruction',
  content_markdown: 'Fasse zusammen.',
  created_at: 1_700_000_000,
  id: 'pt_abc',
  include_in_autocomplete: true,
  label: 'briefing',
  title: 'Briefing',
  updated_at: 1_700_000_100,
  visibility: { chat: true, editor: false },
}

const RULE: ChatRuleRecord = {
  category: 'instruction',
  contentMarkdown: 'Fasse zusammen.',
  createdAt: '2024-01-01T00:00:00.000Z',
  id: 'pt_abc',
  includeInAutocomplete: true,
  label: 'briefing',
  serverTemplateId: 'pt_abc',
  title: 'Briefing',
  updatedAt: '2024-01-01T00:00:00.000Z',
  visibility: { chat: true, editor: false },
}

describe('ruleFromTemplate', () => {
  it('maps the wire record onto the local rule shape', () => {
    const rule = ruleFromTemplate(TEMPLATE)
    expect(rule.id).toBe('pt_abc')
    expect(rule.serverTemplateId).toBe('pt_abc')
    expect(rule.label).toBe('briefing')
    expect(rule.visibility).toEqual({ chat: true, editor: false })
    expect(rule.createdAt).toBe(new Date(1_700_000_000 * 1000).toISOString())
    expect(rule.access).toBeUndefined()
  })

  it('carries the shared-in access annotation', () => {
    const rule = ruleFromTemplate({
      ...TEMPLATE,
      access: { permission: 'view', via: 'share' },
    })
    expect(rule.access).toEqual({ permission: 'view', via: 'share' })
  })

  it('carries the EXACT server timestamp as the precondition anchor', () => {
    // A sub-millisecond float that the ISO `updatedAt` would truncate —
    // serverUpdatedAt must keep it verbatim so the 409 guard matches.
    const micros = 1_781_307_605.652035
    const rule = ruleFromTemplate({ ...TEMPLATE, updated_at: micros })
    expect(rule.serverUpdatedAt).toBe(micros)
    // The ISO mirror necessarily loses the sub-millisecond part —
    // proving why the precondition cannot be derived from it.
    expect(new Date(rule.updatedAt).getTime() / 1000).not.toBe(micros)
  })

  it('preserves browser-local context refs from the existing record', () => {
    const existing = {
      ...RULE,
      linkedContextRefs: [{ fileId: 'f1', kind: 'file-asset' as const }],
    }
    expect(ruleFromTemplate(TEMPLATE, existing).linkedContextRefs).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
    ])
    expect(ruleFromTemplate(TEMPLATE).linkedContextRefs).toBeUndefined()
  })
})

describe('isSameSyncedRule', () => {
  it('treats the hydrated mirror of an unchanged record as a no-op', () => {
    const incoming = ruleFromTemplate(TEMPLATE)
    expect(isSameSyncedRule(incoming, ruleFromTemplate(TEMPLATE))).toBe(true)
  })

  it('detects server-side changes', () => {
    const incoming = ruleFromTemplate(TEMPLATE)
    expect(
      isSameSyncedRule(
        incoming,
        ruleFromTemplate({ ...TEMPLATE, title: 'Anders' }),
      ),
    ).toBe(false)
    expect(
      isSameSyncedRule(
        incoming,
        ruleFromTemplate({
          ...TEMPLATE,
          access: { permission: 'view', via: 'share' },
        }),
      ),
    ).toBe(false)
  })
})

describe('templatePayloadFromRule', () => {
  it('round-trips the writable fields', () => {
    expect(templatePayloadFromRule(RULE)).toEqual({
      category: 'instruction',
      content_markdown: 'Fasse zusammen.',
      include_in_autocomplete: true,
      label: 'briefing',
      title: 'Briefing',
      visibility: { chat: true, editor: false },
    })
  })

  it('treats an uncategorized rule as null category', () => {
    expect(
      templatePayloadFromRule({ ...RULE, category: undefined }).category,
    ).toBeNull()
  })
})

describe('staleSyncedRuleIds', () => {
  it('flags synced rules whose server record vanished', () => {
    const local = [
      RULE,
      { ...RULE, id: 'local-1', serverTemplateId: undefined },
      { ...RULE, id: 'pt_gone', serverTemplateId: 'pt_gone' },
    ]
    expect(staleSyncedRuleIds(local, new Set(['pt_abc']))).toEqual(['pt_gone'])
  })
})

describe('permission gates', () => {
  it('blocks edits for view shares and deletion for any share', () => {
    const viewShared = {
      ...RULE,
      access: { permission: 'view' as const, via: 'share' as const },
    }
    const editShared = {
      ...RULE,
      access: { permission: 'edit' as const, via: 'share' as const },
    }
    expect(canEditRule(RULE)).toBe(true)
    expect(canEditRule(null)).toBe(true)
    expect(canEditRule(viewShared)).toBe(false)
    expect(canEditRule(editShared)).toBe(true)
    expect(canDeleteRule(RULE)).toBe(true)
    expect(canDeleteRule(viewShared)).toBe(false)
    expect(canDeleteRule(editShared)).toBe(false)
    expect(canDeleteRule(null)).toBe(false)
  })
})
