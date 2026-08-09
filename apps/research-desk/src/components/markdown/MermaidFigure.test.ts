import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Mermaid appends a temporary `#d<renderId>` container to document.body for
// measurement. On SUCCESS it removes it itself; on a PARSE ERROR the node
// used to leak as a direct body child below the viewport. That single stray
// gave the document real scroll extent (html/body are overflow: visible), so
// a wheel over any surface without an inner scroller — the left app rail —
// scrolled the WHOLE shell upward. The cleanup contract in
// ensureMermaidRender is what this file pins.
//
// The suite runs in the node environment on purpose (repo convention), so
// the test stubs exactly the DOM surface the contract touches: body children
// plus the two lookups the render path makes.

type StubElement = { id: string; remove: () => void }

function installDomStub() {
  const bodyChildren: StubElement[] = []
  const makeElement = (): StubElement => {
    const el: StubElement = {
      id: '',
      remove: () => {
        const index = bodyChildren.indexOf(el)
        if (index !== -1) bodyChildren.splice(index, 1)
      },
    }
    return el
  }
  const documentStub = {
    body: {
      appendChild: (el: StubElement) => { bodyChildren.push(el) },
      get children() { return [...bodyChildren] },
    },
    createElement: () => makeElement(),
    documentElement: {},
    getElementById: (id: string) => bodyChildren.find((el) => el.id === id) ?? null,
  }
  vi.stubGlobal('document', documentStub)
  vi.stubGlobal('getComputedStyle', () => ({ getPropertyValue: () => '' }))
  return bodyChildren
}

vi.mock('mermaid', () => ({
  default: {
    initialize: () => {},
    render: (id: string) => {
      const stray = (globalThis.document as unknown as {
        createElement: () => StubElement
        body: { appendChild: (el: StubElement) => void }
      }).createElement()
      stray.id = `d${id}`
      ;(globalThis.document as unknown as {
        body: { appendChild: (el: StubElement) => void }
      }).body.appendChild(stray)
      throw new Error('No diagram type detected')
    },
  },
}))

let bodyChildren: StubElement[]

beforeEach(() => {
  bodyChildren = installDomStub()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('mermaid temp-container cleanup', () => {
  it('removes the leaked render container when the diagram fails to parse', async () => {
    const { ensureMermaidRender } = await import('./MermaidFigure')
    await ensureMermaidRender('kaputt', 'light', 'standard', 'standard')

    const strays = bodyChildren.filter((el) => el.id.startsWith('dinqtrix-mermaid'))
    // The error itself stays loud (the figure renders the message + source);
    // only the invisible body-level side effect must be gone.
    expect(strays).toHaveLength(0)
  })
})
