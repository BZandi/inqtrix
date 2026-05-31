/// <reference types="vite/client" />

// mammoth ships no type declarations and resolves its browser build via the
// package.json "browser" field; declare only the narrow surface we use.
declare module 'mammoth' {
  export function extractRawText(input: { arrayBuffer: ArrayBuffer }): Promise<{
    messages: unknown[]
    value: string
  }>
}
