export const EDITOR_COLLABORATION_PROTOCOL_VERSION = 1
export const EDITOR_SCHEMA_VERSION = 1
export const EDITOR_YJS_FRAGMENT = 'content'
export const EDITOR_ROOM_PREFIX = 'inqtrix-editor-v1'

export const EDITOR_SCHEMA_DEPENDENCY_VERSIONS = Object.freeze({
  editorSchema: '0.1.0',
  hocuspocusTransformer: '4.3.0',
  prosemirrorSuggestChanges: '0.1.8',
  tiptap: '3.27.1',
  tiptapYjs: '3.0.6',
  yjs: '13.6.31',
})

// Bump the matching token whenever compatibility-relevant behavior changes
// without changing a declarative ProseMirror schema field.
export const EDITOR_SCHEMA_BEHAVIOR_INPUTS = Object.freeze({
  extensionConfiguration: 'editor-extensions-v1',
  linkPolicy: 'http-https-mailto-noopener-v1',
  markdownProjection: 'gfm-final-original-v1',
  relativePositions: 'yjs-relative-position-base64-v1',
  suggestionTransform: 'adjacent-semantic-modification-pairs-inline-only-table-topology-guard-v5',
  yjsValidation: 'canonical-v1-full-consumption-novel-delta-and-live-state-v7',
})

export const EDITOR_BLOCK_SUGGESTIONS_SUPPORTED = false

// Code blocks admit inline suggestion marks on their text content. No block
// parent advertises node-level suggestion support to the upstream transform.
export const EDITOR_SUGGESTION_BLOCK_PARENTS = Object.freeze([
  'codeBlock',
])

export function editorCollaborationRoom(documentId: string, generation: number): string {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(documentId)) {
    throw new Error('Invalid editor document id')
  }
  if (!Number.isSafeInteger(generation) || generation < 1) {
    throw new Error('Collaboration generation must be a positive integer')
  }
  return `${EDITOR_ROOM_PREFIX}:${documentId}:g${generation}`
}

export function parseEditorCollaborationRoom(room: string): {
  documentId: string
  generation: number
} | null {
  const match = /^inqtrix-editor-v1:([A-Za-z0-9][A-Za-z0-9_-]{0,127}):g([1-9][0-9]*)$/.exec(room)
  if (!match) return null
  const generation = Number(match[2])
  if (!Number.isSafeInteger(generation)) return null
  return { documentId: match[1]!, generation }
}
