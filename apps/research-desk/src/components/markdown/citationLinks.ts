const CITATION_HREF_PREFIX = '#kref-'

export type CitationLabelPredicate = (label: string) => boolean

/**
 * Turn known citation labels into synthetic local Markdown links without
 * coupling the shared Markdown renderer to one evidence system.
 *
 * Bracketed labels are linkified when they belong to the caller's vocabulary.
 * Bare labels are intentionally stricter: they must also occur in
 * ``knownLabels`` so ordinary prose cannot become a dead evidence link.
 */
export function linkifyCitationLabels(
  markdown: string,
  isCitationLabel: CitationLabelPredicate,
  knownLabels?: ReadonlySet<string>,
  options: { requireKnownBracketed?: boolean } = {},
): string {
  let out = markdown.replace(
    /\[([A-Z]\d+)\](?!\()/g,
    (match, label: string) => (
      isCitationLabel(label)
      && (!options.requireKnownBracketed || knownLabels?.has(label))
        ? citationLink(label)
        : match
    ),
  )
  if (!knownLabels || knownLabels.size === 0) return out

  out = out.replace(
    /(^|[^\w[\]()/#-])((?:[A-Z]\d+){2,})\b/g,
    (match, prefix: string, run: string) => {
      const labels = splitKnownCitationRun(run, knownLabels, isCitationLabel)
      return labels
        ? `${prefix}${labels.map((label) => citationLink(label)).join('')}`
        : match
    },
  )
  return out.replace(
    /(^|[^\w[\]()/#-])([A-Z]\d+)\b/g,
    (match, prefix: string, label: string) => (
      isCitationLabel(label) && knownLabels.has(label)
        ? `${prefix}${citationLink(label)}`
        : match
    ),
  )
}

/** Extract a caller-approved label from a synthetic evidence link. */
export function citationLabelFromHref(
  href: string | null | undefined,
  isCitationLabel: CitationLabelPredicate,
): string | null {
  if (!href) return null
  const index = href.indexOf(CITATION_HREF_PREFIX)
  if (index === -1) return null
  const label = href.slice(index + CITATION_HREF_PREFIX.length)
  return isCitationLabel(label) ? label : null
}

function citationLink(label: string): string {
  return `[${label}](${CITATION_HREF_PREFIX}${label})`
}

function splitKnownCitationRun(
  run: string,
  knownLabels: ReadonlySet<string>,
  isCitationLabel: CitationLabelPredicate,
): string[] | null {
  const labels = [...knownLabels].filter(isCitationLabel)
  const memo = new Map<number, string[] | null>()
  const splitFrom = (offset: number): string[] | null => {
    if (offset === run.length) return []
    const cached = memo.get(offset)
    if (cached !== undefined) return cached
    const candidates = labels
      .filter((label) => run.startsWith(label, offset))
      .sort((a, b) => b.length - a.length)
    for (const label of candidates) {
      const rest = splitFrom(offset + label.length)
      if (rest) {
        const result = [label, ...rest]
        memo.set(offset, result)
        return result
      }
    }
    memo.set(offset, null)
    return null
  }
  const result = splitFrom(0)
  return result && result.length > 1 ? result : null
}
