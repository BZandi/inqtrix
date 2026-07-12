/**
 * Small DOM helpers shared by the editor shell (`EditorWorkspace`) and the
 * markdown surface (`core/MarkdownEditorSurface`).
 */

export function resetExternalContentFlag(ref: { current: boolean }) {
  if (globalThis.queueMicrotask) {
    globalThis.queueMicrotask(() => {
      ref.current = false
    })
    return
  }
  globalThis.setTimeout(() => {
    ref.current = false
  }, 0)
}

export function escapeCssIdentifier(value: string) {
  if (globalThis.CSS?.escape) return globalThis.CSS.escape(value)
  return value.replace(/["\\]/g, '\\$&')
}
