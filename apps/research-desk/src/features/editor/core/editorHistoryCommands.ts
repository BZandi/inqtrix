import type { Editor } from '@tiptap/react'

export type EditorHistoryCommand = 'redo' | 'undo'

/** Collaboration deliberately removes local history until the Yjs binding is
 * ready. Treat those temporarily absent commands as disabled UI, not as an
 * exceptional editor state. */
export function canRunEditorHistoryCommand(
  editor: Editor | null,
  command: EditorHistoryCommand,
): boolean {
  if (!editor || editor.isDestroyed) return false
  const commands = editor.can()
  const candidate = commands[command]
  return typeof candidate === 'function' && candidate.call(commands)
}

export function runEditorHistoryCommand(
  editor: Editor | null,
  command: EditorHistoryCommand,
): boolean {
  if (!editor || editor.isDestroyed) return false
  const chain = editor.chain().focus()
  const candidate = chain[command]
  if (typeof candidate !== 'function') return false
  return candidate.call(chain).run()
}
