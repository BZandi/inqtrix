import type { ReportReference, ResearchSource } from '@/features/researchRuns/types'

export function normalizeReportReferences(
  value: unknown,
  markdown: string,
  topSources: readonly ResearchSource[],
): ReportReference[] {
  if (Array.isArray(value)) {
    return value.flatMap((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return []
      const record = item as Record<string, unknown>
      const url = optionalString(record.url)?.trim()
      if (!url) return []
      return [{
        label: optionalString(record.label)?.trim() || 'Quelle',
        tier: optionalString(record.tier)?.trim() || tierForUrl(url, topSources),
        url,
      }]
    })
  }

  return reportReferencesFromMarkdown(markdown, topSources)
}

export function reportReferencesFromMarkdown(
  markdown: string,
  topSources: readonly ResearchSource[],
): ReportReference[] {
  const references: ReportReference[] = []
  let inReferenceSection = false
  const seenUrls = new Set<string>()

  for (const line of markdown.split(/\r?\n/)) {
    if (/^##\s+(Referenzen|References)\s*$/i.test(line.trim())) {
      inReferenceSection = true
      continue
    }
    if (!inReferenceSection) continue
    if (/^---+\s*$/.test(line.trim()) || /^##\s+/.test(line.trim())) break

    const match = line.match(/^\s*[-*]\s+\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/)
    if (!match) continue
    const label = match[1].trim() || 'Quelle'
    const url = match[2].trim()
    if (!url || seenUrls.has(url)) continue
    seenUrls.add(url)
    references.push({ label, tier: tierForUrl(url, topSources), url })
  }

  return references
}

function tierForUrl(url: string, topSources: readonly ResearchSource[]) {
  return topSources.find((source) => source.url === url)?.tier ?? 'unknown'
}

function optionalString(value: unknown) {
  return typeof value === 'string' ? value : undefined
}
