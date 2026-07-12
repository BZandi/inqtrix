/**
 * Comment-kind order and per-kind visual metadata (icon, tone classes,
 * localized label), shared by the comments panel in `EditorWorkspace` and the
 * bubble-menu comment composer in `core/MarkdownEditorSurface`.
 */
import { MessagesSquare, SearchCheck, Sparkles } from '@/components/icons'
import type { EditorCommentKind } from '@/features/project/types'
import type { EditorCopy } from './editorCopy'

export const COMMENT_KIND_ORDER: EditorCommentKind[] = ['collect', 'inline_edit', 'evidence_review']

export type CommentKindMeta = {
  Icon: typeof MessagesSquare
  accentText: string
  bgClass: string
  borderClass: string
  dotClass: string
  label: string
  selectedBgClass: string
  selectedBorderClass: string
}

export function commentKindMeta(kind: EditorCommentKind, copy: EditorCopy): CommentKindMeta {
  if (kind === 'inline_edit') {
    return {
      Icon: Sparkles,
      accentText: 'text-warning',
      bgClass: 'bg-warning-subtle/20',
      borderClass: 'border-l-warning',
      dotClass: 'bg-warning',
      label: copy.kindInline,
      selectedBgClass: 'bg-warning-subtle/45',
      selectedBorderClass: 'border-warning',
    }
  }
  if (kind === 'evidence_review') {
    return {
      Icon: SearchCheck,
      accentText: 'text-success',
      bgClass: 'bg-success-subtle/20',
      borderClass: 'border-l-success',
      dotClass: 'bg-success',
      label: copy.kindEvidence,
      selectedBgClass: 'bg-success-subtle/45',
      selectedBorderClass: 'border-success',
    }
  }
  return {
    Icon: MessagesSquare,
    accentText: 'text-brand',
    bgClass: 'bg-brand-subtle/25',
    borderClass: 'border-l-brand',
    dotClass: 'bg-brand',
    label: copy.kindCollect,
    selectedBgClass: 'bg-brand-subtle/45',
    selectedBorderClass: 'border-brand',
  }
}
