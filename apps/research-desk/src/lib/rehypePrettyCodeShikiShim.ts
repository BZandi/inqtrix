export function getSingletonHighlighter(): never {
  throw new Error('The chat Markdown renderer must provide its own rehype-pretty-code highlighter.')
}
