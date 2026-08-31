import type { AgentRunRecord } from './model'

/**
 * Scroll identity and growth signal of the agent transcript.
 *
 * The Agent Desk had none of the three scroll behaviours the other
 * desks have: no following while a run produces messages, no user
 * override (there was nothing to override), and no position memory —
 * a session switch even remounted the surface and landed at the top of
 * a long transcript. The mechanism for all three already exists in
 * `features/scroll/`; only these two pure values were missing to
 * connect it, so the agent inherits the SAME contract as chat and
 * knowledge instead of growing a fourth variant.
 */

/** Namespaced memory key, matching `chat:<id>` / `knowledge:<id>`. */
export function agentScrollKey(
  sessionId: string | null | undefined,
): string | null {
  const id = (sessionId ?? '').trim()
  return id ? `agent:${id}` : null
}

/**
 * A value that changes exactly when the transcript grew.
 *
 * `lastSequence` is the highest event sequence applied to a run, so it
 * advances on every appended step, streamed token and status change —
 * the three things that make the transcript taller. Every run of the
 * session contributes: an older run can still gain events (a late
 * artifact update), and following must not miss that.
 */
export function agentTranscriptVersion(
  runs: readonly AgentRunRecord[],
): string {
  return runs
    .map((run) => `${run.runId}:${run.lastSequence}:${run.status}`)
    .join('|')
}
