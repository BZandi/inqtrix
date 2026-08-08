/**
 * `/`-menu detection for the agent composer — the
 * slash twin of `detectCollectionMention` (same adapter pattern; the
 * shared composer mention module stays untouched). A slash token only
 * counts at the start of the text or after whitespace, so paths and
 * URLs never open the menu.
 */

export type SkillSlashState = {
  /** Char index of the `/` in the textarea value. */
  start: number
  query: string
}

export function detectSkillSlash(
  value: string,
  caret: number,
): SkillSlashState | null {
  const safeCaret = Math.max(0, Math.min(caret, value.length))
  const beforeCaret = value.slice(0, safeCaret)
  const match = /(?:^|\s)\/([a-z0-9-]*)$/i.exec(beforeCaret)
  if (!match) return null
  return {
    query: match[1].toLowerCase(),
    start: safeCaret - match[1].length - 1,
  }
}

/** Direct, one-message routes. These are execution commands rather than
 * reusable skills or best-effort tool hints. */
export const EXECUTION_DIRECTIVE_OPTIONS = [
  { id: 'quick_web', token: 'web' },
  { id: 'knowledge_only', token: 'wissen' },
] as const

export type ExecutionDirectiveId =
  (typeof EXECUTION_DIRECTIVE_OPTIONS)[number]['id']
