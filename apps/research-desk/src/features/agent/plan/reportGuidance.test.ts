import { describe, expect, it } from 'vitest'

import type { AgentApprovalRecord } from '../model'
import {
  decidedReportGuidance,
  decidedReportRuleLabels,
} from './reportGuidance'

function approval(
  overrides: Partial<AgentApprovalRecord> = {},
): AgentApprovalRecord {
  return {
    approvalId: 'a1',
    kind: 'plan',
    status: 'approved',
    subjectType: 'plan',
    subjectId: 'p1',
    payload: {},
    decision: 'approve',
    note: '',
    createdAt: 0,
    decidedAt: 1,
    ...overrides,
  }
}

describe('decidedReportGuidance', () => {
  it('reads the requirement that is in force', () => {
    expect(
      decidedReportGuidance([
        approval({ decisionPayload: { report_guidance: ' Als Tabelle. ' } }),
      ]),
    ).toBe('Als Tabelle.')
  })

  it('lets a later gate clear it', () => {
    // The user withdrew the requirement at the replan gate; showing the
    // old text back would tell them something is in force that is not.
    expect(
      decidedReportGuidance([
        approval({ decisionPayload: { report_guidance: 'Als Tabelle.' } }),
        approval({
          approvalId: 'a2',
          kind: 'replan',
          decisionPayload: { report_guidance: '' },
        }),
      ]),
    ).toBe('')
  })

  it('keeps the standing one when a later gate says nothing', () => {
    // Absence is not a decision — the server's resume keeps the old
    // value in exactly this case, so the read-back must agree.
    expect(
      decidedReportGuidance([
        approval({ decisionPayload: { report_guidance: 'Als Tabelle.' } }),
        approval({ approvalId: 'a2', kind: 'replan', decisionPayload: {} }),
      ]),
    ).toBe('Als Tabelle.')
  })

  it('ignores gates that cannot carry it and undecided ones', () => {
    expect(
      decidedReportGuidance([
        approval({
          kind: 'tool',
          decisionPayload: { report_guidance: 'fremd' },
        }),
        approval({
          approvalId: 'a2',
          status: 'pending',
          decidedAt: null,
          decisionPayload: { report_guidance: 'noch nicht entschieden' },
        }),
      ]),
    ).toBe('')
  })
})

describe('decided requirement parts', () => {
  it('shows the typed text, not the composed prompt block', () => {
    // The composed value carries `[Regel: …]` markers because the model
    // needs to tell the origins apart; the user should see what they
    // wrote, plus the rule names beside it.
    const approvals = [
      approval({
        decisionPayload: {
          report_guidance:
            '[Regel: sprechzettel]\nGliedere.\n[Ende Regel: sprechzettel]'
            + '\n\n[Freie Vorgabe]\nZielgruppe: Laien.',
          report_requirement: {
            free_text: 'Zielgruppe: Laien.',
            rules: [
              { template_id: 'r1', label: 'sprechzettel', revision: 1 },
            ],
          },
        },
      }),
    ]
    expect(decidedReportGuidance(approvals)).toBe('Zielgruppe: Laien.')
    expect(decidedReportRuleLabels(approvals)).toEqual(['sprechzettel'])
  })

  it('falls back to the composed value for older decisions', () => {
    expect(
      decidedReportGuidance([
        approval({ decisionPayload: { report_guidance: 'Als Tabelle.' } }),
      ]),
    ).toBe('Als Tabelle.')
    expect(
      decidedReportRuleLabels([
        approval({ decisionPayload: { report_guidance: 'Als Tabelle.' } }),
      ]),
    ).toEqual([])
  })
})
