import { BookOpen, FileText, FolderOpen, Paperclip } from '@/components/icons'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type {
  ChatRuleOption,
  CompletedReportOption,
  FileGroupMentionOption,
  FileMentionOption,
} from '@/features/project/selectors'
import type { ChatContextReferenceRecord } from '@/features/project/types'

export type MentionKind = 'research' | 'rules' | 'files' | 'filegroups'

export type MentionMatch = {
  end: number
  kind: MentionKind | 'root'
  query: string
  start: number
}

export type MentionCategoryLabels = {
  files: string
  filegroups: string
  research: string
  rules: string
}

export type MentionSources = {
  fileGroupOptions: FileGroupMentionOption[]
  fileOptions: FileMentionOption[]
  reportOptions: CompletedReportOption[]
  ruleOptions: ChatRuleOption[]
}

export type MentionOption = {
  label: string
  prefix?: '@research:' | '@rules:' | '@files:' | '@filegroups:'
  ref?: ChatContextReferenceRecord
  title: string
  type: MentionKind
}

const ALL_KINDS: MentionKind[] = ['research', 'rules', 'files', 'filegroups']

const CATEGORY_PREFIX: Record<MentionKind, MentionOption['prefix']> = {
  research: '@research:',
  rules: '@rules:',
  files: '@files:',
  filegroups: '@filegroups:',
}

/**
 * Build mention options for the current trigger. `enabledKinds` restricts which
 * categories are offered: the chat composer enables all, the editor composer
 * everything except research reports. With a single enabled kind the root "@"
 * lists that kind's items directly instead of an intermediate category step.
 */
export function buildMentionOptions(
  match: MentionMatch,
  sources: MentionSources,
  categoryLabels: MentionCategoryLabels,
  enabledKinds: MentionKind[] = ALL_KINDS,
): MentionOption[] {
  const query = match.query.toLowerCase()
  if (match.kind === 'root') {
    if (enabledKinds.length === 1) {
      return mentionItemsForKind(enabledKinds[0], '', sources)
    }
    return enabledKinds.map((kind) => ({
      label: CATEGORY_PREFIX[kind] ?? `@${kind}:`,
      prefix: CATEGORY_PREFIX[kind],
      title: categoryLabels[kind],
      type: kind,
    }))
  }
  return enabledKinds.includes(match.kind)
    ? mentionItemsForKind(match.kind, query, sources)
    : []
}

function mentionItemsForKind(
  kind: MentionKind,
  query: string,
  sources: MentionSources,
): MentionOption[] {
  if (kind === 'research') {
    return sources.reportOptions
      .filter((report) => matchesMentionQuery(query, report.label, report.title))
      .slice(0, 8)
      .map((report) => ({
        label: report.label,
        ref: { kind: 'research-report', runId: report.runId },
        title: report.title,
        type: 'research' as const,
      }))
  }
  if (kind === 'files') {
    return sources.fileOptions
      .filter((file) => matchesMentionQuery(query, file.label, file.title))
      .slice(0, 8)
      .map((file) => ({
        label: file.label,
        ref: { fileId: file.fileId, kind: 'file-asset' },
        title: file.title,
        type: 'files' as const,
      }))
  }
  if (kind === 'filegroups') {
    return sources.fileGroupOptions
      .filter((group) => matchesMentionQuery(query, group.label, group.title))
      .slice(0, 8)
      .map((group) => ({
        label: group.label,
        ref: { groupId: group.groupId, kind: 'file-group' },
        title: group.title,
        type: 'filegroups' as const,
      }))
  }
  return sources.ruleOptions
    .filter((rule) => matchesMentionQuery(query, rule.label, rule.title))
    .slice(0, 8)
    .map((rule) => ({
      label: rule.label,
      ref: { kind: 'chat-rule', ruleId: rule.ruleId },
      title: rule.title,
      type: 'rules' as const,
    }))
}

const MENTION_KIND_BY_TOKEN: Record<string, MentionKind> = {
  research: 'research',
  rules: 'rules',
  files: 'files',
  filegroups: 'filegroups',
}

export function detectMentionTrigger(value: string, cursor: number): MentionMatch | null {
  const prefix = value.slice(0, cursor)
  const rootMatch = /(?:^|\s)(@)$/i.exec(prefix)
  if (rootMatch) {
    return { end: cursor, kind: 'root', query: '', start: cursor - rootMatch[1].length }
  }
  const match = /(?:^|\s)(@(rules|research|files|filegroups):([a-z0-9-]*))$/i.exec(prefix)
  if (!match) return null
  return {
    end: cursor,
    kind: MENTION_KIND_BY_TOKEN[match[2].toLowerCase()] ?? 'rules',
    query: match[3].toLowerCase(),
    start: cursor - match[1].length,
  }
}

/**
 * Resolve `@kind:label` mentions typed inline into context references. Used by
 * both the chat composer and the editor composer (via `enabledKinds`), so a
 * typed mention is never silently dropped. Unknown labels are reported back so
 * the caller can surface a visible notice.
 */
export function resolveInlineMentions(
  contentMarkdown: string,
  sources: MentionSources,
  enabledKinds: MentionKind[] = ALL_KINDS,
): { error: string | null; refs: ChatContextReferenceRecord[] } {
  const refs: ChatContextReferenceRecord[] = []
  const unknown: string[] = []
  const seen = new Set<string>()
  const reportByLabel = new Map(sources.reportOptions.map((report) => [report.label, report]))
  const ruleByLabel = new Map(sources.ruleOptions.map((rule) => [rule.label, rule]))
  const fileByLabel = new Map(sources.fileOptions.map((file) => [file.label, file]))
  const groupByLabel = new Map(sources.fileGroupOptions.map((group) => [group.label, group]))
  const regex = /(?:^|\s)@(rules|research|files|filegroups):([^\s]+)/gi
  let match = regex.exec(contentMarkdown)

  const push = (ref: ChatContextReferenceRecord, key: string) => {
    if (seen.has(key)) return
    refs.push(ref)
    seen.add(key)
  }

  while (match) {
    const kind = MENTION_KIND_BY_TOKEN[match[1].toLowerCase()] ?? 'rules'
    const rawLabel = match[2].replace(/[.,;!?)]$/, '').toLowerCase()
    if (!enabledKinds.includes(kind)) {
      match = regex.exec(contentMarkdown)
      continue
    }
    if (kind === 'rules') {
      const rule = ruleByLabel.get(rawLabel)
      if (rule) push({ kind: 'chat-rule', ruleId: rule.ruleId }, `chat-rule:${rule.ruleId}`)
      else unknown.push(`@rules:${rawLabel}`)
    } else if (kind === 'research') {
      const report = reportByLabel.get(rawLabel)
      if (report) push({ kind: 'research-report', runId: report.runId }, `research-report:${report.runId}`)
      else unknown.push(`@research:${rawLabel}`)
    } else if (kind === 'files') {
      const file = fileByLabel.get(rawLabel)
      if (file) push({ fileId: file.fileId, kind: 'file-asset' }, `file-asset:${file.fileId}`)
      else unknown.push(`@files:${rawLabel}`)
    } else {
      const group = groupByLabel.get(rawLabel)
      if (group) push({ groupId: group.groupId, kind: 'file-group' }, `file-group:${group.groupId}`)
      else unknown.push(`@filegroups:${rawLabel}`)
    }
    match = regex.exec(contentMarkdown)
  }

  return { error: unknown.length > 0 ? unknown.join(', ') : null, refs }
}

function matchesMentionQuery(query: string, label: string, title: string) {
  if (!query) return true
  return label.includes(query) || title.toLowerCase().includes(query)
}

const MENTION_ICON: Record<MentionKind, typeof BookOpen> = {
  research: FileText,
  rules: BookOpen,
  files: Paperclip,
  filegroups: FolderOpen,
}

export function MentionAutocomplete({
  activeIndex,
  onSelect,
  options,
}: {
  activeIndex: number
  onSelect: (option: MentionOption) => void
  options: MentionOption[]
}) {
  const { t } = useLocale()
  return (
    <div className="absolute bottom-full left-0 z-30 mb-2 w-full max-w-lg overflow-hidden rounded-lg border border-border bg-popover shadow-lg">
      <div className="border-b border-border px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        <span>{t.chat.mentionResults}</span>
        <span className="float-right normal-case tracking-normal text-muted-foreground/75">
          {t.chat.mentionEscHint}
        </span>
      </div>
      <div className="max-h-64 overflow-y-auto p-1">
        {options.map((option, index) => {
          const Icon = MENTION_ICON[option.type]
          return (
            <button
              className={cn(
                'flex w-full min-w-0 items-start gap-2 rounded-md px-2 py-2 text-left transition-colors',
                index === activeIndex ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/70',
              )}
              key={`${option.type}-${option.label}`}
              onMouseDown={(event) => {
                event.preventDefault()
                onSelect(option)
              }}
              type="button"
            >
              <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold">
                  {option.prefix ? option.label : `@${option.type}:${option.label}`}
                </span>
                <span className="block truncate text-xs text-muted-foreground">
                  {option.title}
                </span>
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
