import type { Editor } from '@tiptap/react'
import { describe, expect, it, vi } from 'vitest'

import {
  canRunEditorHistoryCommand,
  runEditorHistoryCommand,
} from './editorHistoryCommands'

function editorWithoutHistory(): Editor {
  return {
    can: () => ({}),
    chain: () => ({
      focus: () => ({}),
    }),
    isDestroyed: false,
  } as unknown as Editor
}

describe('editor history command guards', () => {
  it('keeps collaboration toolbars inert while history is not installed', () => {
    const editor = editorWithoutHistory()

    expect(canRunEditorHistoryCommand(editor, 'undo')).toBe(false)
    expect(canRunEditorHistoryCommand(editor, 'redo')).toBe(false)
    expect(runEditorHistoryCommand(editor, 'undo')).toBe(false)
    expect(runEditorHistoryCommand(editor, 'redo')).toBe(false)
  })

  it('runs an available history command through the focused chain', () => {
    const run = vi.fn(() => true)
    const undo = vi.fn(() => ({ run }))
    const focus = vi.fn(() => ({ undo }))
    const canUndo = vi.fn(() => true)
    const editor = {
      can: () => ({ undo: canUndo }),
      chain: () => ({ focus }),
      isDestroyed: false,
    } as unknown as Editor

    expect(canRunEditorHistoryCommand(editor, 'undo')).toBe(true)
    expect(runEditorHistoryCommand(editor, 'undo')).toBe(true)
    expect(canUndo).toHaveBeenCalledOnce()
    expect(focus).toHaveBeenCalledOnce()
    expect(undo).toHaveBeenCalledOnce()
    expect(run).toHaveBeenCalledOnce()
  })
})
