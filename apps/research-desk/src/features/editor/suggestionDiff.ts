import { diffLines, diffWordsWithSpace } from 'diff'

export type SuggestionDiffSegment = {
  text: string
  type: 'delete' | 'equal' | 'insert'
}

export type SuggestionDiffDisplay = 'block' | 'inline'
export type SuggestionReviewSurface = 'editor' | 'panel'

export type SuggestionDiffPlan = {
  display: SuggestionDiffDisplay
  reviewSurface: SuggestionReviewSurface
  segments: SuggestionDiffSegment[]
}

export const INLINE_DIFF_MAX_CHARS = 720
export const INLINE_DIFF_MAX_CHANGED_SEGMENTS = 14

/**
 * Word-level diff between the anchored original text and a proposed rewrite.
 * Whitespace is preserved so the rendered diff reads like the source passage.
 */
export function suggestionDiffSegments(original: string, proposed: string): SuggestionDiffSegment[] {
  return diffWordsWithSpace(original, proposed).map((part) => ({
    text: part.value,
    type: part.added ? 'insert' : part.removed ? 'delete' : 'equal',
  }))
}

export function markdownDiffSegments(originalMarkdown: string, currentMarkdown: string): SuggestionDiffSegment[] {
  return diffLines(originalMarkdown, currentMarkdown)
    .map((part) => ({
      text: part.value,
      type: (part.added ? 'insert' : part.removed ? 'delete' : 'equal') as SuggestionDiffSegment['type'],
    }))
    .filter((segment) => segment.text.length > 0)
}

export function suggestionDiffPlan(original: string, proposed: string): SuggestionDiffPlan {
  const segments = suggestionDiffSegments(original, proposed)
  const display = shouldUseBlockDiff(original, proposed, segments) ? 'block' : 'inline'
  return {
    display,
    reviewSurface: display === 'block' ? 'editor' : 'panel',
    segments,
  }
}

export function hasVisibleDiff(segments: SuggestionDiffSegment[]): boolean {
  return segments.some((segment) => segment.type !== 'equal' && segment.text.trim().length > 0)
}

function shouldUseBlockDiff(
  original: string,
  proposed: string,
  segments: SuggestionDiffSegment[],
) {
  const originalLength = original.trim().length
  const proposedLength = proposed.trim().length
  if (original.includes('\n') || proposed.includes('\n')) return true
  if (originalLength > INLINE_DIFF_MAX_CHARS || proposedLength > INLINE_DIFF_MAX_CHARS) return true
  const changedSegments = segments.filter((segment) => segment.type !== 'equal' && segment.text.trim()).length
  return changedSegments > INLINE_DIFF_MAX_CHANGED_SEGMENTS
}
