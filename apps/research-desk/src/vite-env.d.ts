/// <reference types="vite/client" />

// Build-time constant from the `define` block in vite.config.ts,
// single-sourced from `__version__` in src/inqtrix/__init__.py.
declare const __APP_VERSION__: string

// mammoth ships no type declarations and resolves its browser build via the
// package.json "browser" field; declare only the narrow surface we use.
declare module 'mammoth' {
  export function extractRawText(input: { arrayBuffer: ArrayBuffer }): Promise<{
    messages: unknown[]
    value: string
  }>
}
