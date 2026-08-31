import type { AgentApprovalRecord } from '../model'

/**
 * The result requirement that is actually in force for a run.
 *
 * It lives in the plan gate's decision payload, not in the plan, so a
 * finished plan cannot show it from the plan alone. Reading it back is
 * the difference between a requirement the user can verify and one they
 * have to remember: the value shaped every section of the report.
 *
 * Presence decides, in decision order — an approval that carried no
 * requirement key leaves the earlier one standing (the server's resume
 * does the same), while an explicitly emptied one clears it. Only plan
 * gates can carry it; the server rejects it anywhere else.
 */
export function decidedReportGuidance(
  approvals: readonly AgentApprovalRecord[],
): string {
  let effective = ''
  for (const approval of approvals) {
    if (approval.kind !== 'plan' && approval.kind !== 'replan') continue
    if (approval.status === 'pending') continue
    const payload = approval.decisionPayload
    // The parts, when the decision kept them: `report_guidance` holds
    // the COMPOSED text with its `[Regel: …]` / `[Freie Vorgabe]`
    // markers, which is what the model reads — showing that back would
    // be showing the user the prompt. Older decisions have only the
    // composed value and fall back to it.
    if (payload && 'report_requirement' in payload) {
      const requirement = payload.report_requirement
      const freeText =
        requirement && typeof requirement === 'object'
          ? (requirement as { free_text?: unknown }).free_text
          : undefined
      effective = typeof freeText === 'string' ? freeText.trim() : ''
      continue
    }
    if (!payload || !('report_guidance' in payload)) continue
    const value = payload.report_guidance
    effective = typeof value === 'string' ? value.trim() : ''
  }
  return effective
}

/**
 * Labels of the library rules the decided gate ran under.
 *
 * The composed requirement in the prompt carries `[Regel: …]` markers,
 * but showing those to the user would be showing them the prompt. The
 * decision keeps the parts for exactly this: naming what was attached.
 */
export function decidedReportRuleLabels(
  approvals: readonly AgentApprovalRecord[],
): string[] {
  let labels: string[] = []
  for (const approval of approvals) {
    if (approval.kind !== 'plan' && approval.kind !== 'replan') continue
    if (approval.status === 'pending') continue
    const payload = approval.decisionPayload
    if (!payload || !('report_requirement' in payload)) continue
    const requirement = payload.report_requirement
    if (!requirement || typeof requirement !== 'object') {
      labels = []
      continue
    }
    const rules = (requirement as { rules?: unknown }).rules
    labels = Array.isArray(rules)
      ? rules
        .map((rule) =>
          rule && typeof rule === 'object'
            ? String((rule as { label?: unknown }).label ?? '')
            : '',
        )
        .filter((label) => label.length > 0)
      : []
  }
  return labels
}
