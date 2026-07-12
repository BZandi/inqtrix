const MARKDOWN_LANGUAGE_ALIASES: Record<string, string> = {
  js: 'javascript',
  py: 'python',
  shell: 'shellscript',
  text: 'plaintext',
  ts: 'typescript',
}

export const MARKDOWN_COMMON_LANGUAGES = [
  'plaintext',
  'python',
  'bash',
  'sh',
  'shellscript',
  'json',
  'jsonc',
  'javascript',
  'typescript',
  'tsx',
  'jsx',
  'css',
  'html',
  'markdown',
] as const

const SUPPORTED_MARKDOWN_LANGUAGE_SET = new Set<string>(MARKDOWN_COMMON_LANGUAGES)

export type MarkdownCodeBlock = {
  code: string
  language: string
}

export function extractMarkdownCodeLanguages(markdown: string): string[] {
  const languages = new Set<string>()
  for (const block of extractMarkdownCodeBlocks(markdown)) {
    languages.add(block.language)
  }

  return [...languages]
}

export function extractMarkdownCodeBlocks(markdown: string): MarkdownCodeBlock[] {
  const blocks: MarkdownCodeBlock[] = []
  const lines = markdown.split(/\r?\n/)

  for (let index = 0; index < lines.length; index += 1) {
    const opening = /^(?: {0,3})(`{3,}|~{3,})([^\n\r`]*)$/.exec(lines[index] ?? '')
    if (!opening) continue

    const marker = opening[1]
    const markerChar = marker[0]
    const markerLength = marker.length
    const language = normalizeMarkdownCodeLanguage(opening[2])
    const bodyStart = index + 1
    let bodyEnd = bodyStart

    for (; bodyEnd < lines.length; bodyEnd += 1) {
      const line = lines[bodyEnd] ?? ''
      const leadingTrimmed = line.trimStart()
      const indent = line.length - leadingTrimmed.length
      if (indent > 3) continue
      const markerMatch = new RegExp(`^\\${markerChar}{${markerLength},}\\s*$`).exec(leadingTrimmed)
      if (markerMatch) break
    }

    if (bodyEnd >= lines.length) break
    if (language) {
      blocks.push({
        code: lines.slice(bodyStart, bodyEnd).join('\n'),
        language,
      })
    }
    index = bodyEnd
  }

  return blocks
}

export function plainCodeLanguageFromClassName(className: unknown): string | null {
  if (Array.isArray(className)) {
    for (const item of className) {
      const language = plainCodeLanguageFromClassName(item)
      if (language) return language
    }
    return null
  }

  if (typeof className !== 'string') return null

  for (const token of className.split(/\s+/)) {
    if (!token.startsWith('language-')) continue
    const language = normalizeMarkdownCodeLanguage(token.slice('language-'.length))
    if (language) return language
  }

  return null
}

/** The RAW fence language token (lowercased) WITHOUT the Shiki
 * whitelist — for renderers that dispatch on languages the highlighter
 * does not know (the mermaid figure). `plainCodeLanguageFromClassName`
 * stays the highlighter-facing, whitelist-normalized reading. */
export function rawCodeLanguageFromClassName(className: unknown): string | null {
  if (Array.isArray(className)) {
    for (const item of className) {
      const language = rawCodeLanguageFromClassName(item)
      if (language) return language
    }
    return null
  }

  if (typeof className !== 'string') return null

  for (const token of className.split(/\s+/)) {
    if (!token.startsWith('language-')) continue
    const raw = token.slice('language-'.length).trim().toLowerCase()
    if (raw) return raw
  }

  return null
}

function normalizeMarkdownCodeLanguage(value: unknown): string | null {
  if (typeof value !== 'string') return null

  const token = value
    .trim()
    .split(/\s+/)[0]
    ?.replace(/^language-/i, '')
    .toLowerCase()

  if (!token) return null

  const normalized = MARKDOWN_LANGUAGE_ALIASES[token] ?? token
  return SUPPORTED_MARKDOWN_LANGUAGE_SET.has(normalized) ? normalized : null
}
