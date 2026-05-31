import { describe, expect, it } from 'vitest'
import type { JSONContent } from '@tiptap/core'
import {
  instructionTextFromDoc,
  mentionDocFromText,
  mentionTextFromDoc,
  pillRefsFromDoc,
  type LabelResolver,
} from './mentionDoc'

function pill(refId: string, refKind: string, refLabel: string): JSONContent {
  return { attrs: { refId, refKind, refLabel }, type: 'mentionPill' }
}

function paragraph(...content: JSONContent[]): JSONContent {
  return { content, type: 'paragraph' }
}

const TWO_PILLS: JSONContent = {
  content: [
    paragraph(
      { text: 'Compare ', type: 'text' },
      pill('f1', 'file-asset', 'alpha'),
      { text: ' with ', type: 'text' },
      pill('g1', 'file-group', 'beta'),
    ),
  ],
  type: 'doc',
}

describe('instructionTextFromDoc', () => {
  it('renders pills as [N] in reading order', () => {
    expect(instructionTextFromDoc(TWO_PILLS)).toBe('Compare [1] with [2]')
  })

  it('returns an empty string for an empty doc', () => {
    expect(instructionTextFromDoc({ content: [paragraph()], type: 'doc' })).toBe('')
  })
})

describe('mentionTextFromDoc', () => {
  it('renders pills as @kind:label for the round-trip', () => {
    expect(mentionTextFromDoc(TWO_PILLS)).toBe('Compare @files:alpha with @filegroups:beta')
  })
})

describe('pillRefsFromDoc', () => {
  it('extracts positional references in reading order', () => {
    expect(pillRefsFromDoc(TWO_PILLS)).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
  })
})

describe('mentionDocFromText', () => {
  const resolve: LabelResolver = (kind, label) => {
    if (kind === 'file-asset' && label === 'alpha') return { id: 'f1', label: 'alpha' }
    if (kind === 'file-group' && label === 'beta') return { id: 'g1', label: 'beta' }
    return null
  }

  it('turns resolvable @kind:label tokens into pills and round-trips', () => {
    const doc = mentionDocFromText('Compare @files:alpha with @filegroups:beta', resolve)
    expect(pillRefsFromDoc(doc)).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
    expect(mentionTextFromDoc(doc)).toBe('Compare @files:alpha with @filegroups:beta')
    expect(instructionTextFromDoc(doc)).toBe('Compare [1] with [2]')
  })

  it('leaves unresolvable tokens as plain text', () => {
    const doc = mentionDocFromText('Use @files:missing here', resolve)
    expect(pillRefsFromDoc(doc)).toEqual([])
    expect(mentionTextFromDoc(doc)).toBe('Use @files:missing here')
  })
})
