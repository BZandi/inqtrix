import type { Editor } from '@tiptap/react'

/**
 * Prompt for a URL and apply/replace/clear the link mark on the current
 * selection. Shared by the header toolbar and the selection bubble menu so the
 * link behaviour (and the Link extension's safe-URL validation) lives in exactly
 * one place. A cancelled prompt (`null`) is a no-op; an empty URL clears the
 * link. `extendMarkRange('link')` makes editing or clearing an existing link
 * cover its whole range even from a collapsed caret.
 */
export function promptSetLink(editor: Editor): void {
  const previousUrl = editor.getAttributes('link').href as string | undefined
  const url = window.prompt('URL', previousUrl ?? 'https://')
  if (url === null) return
  const trimmed = url.trim()
  if (!trimmed) {
    editor.chain().focus().extendMarkRange('link').unsetLink().run()
    return
  }
  editor.chain().focus().extendMarkRange('link').setLink({ href: trimmed }).run()
}
