import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'
import * as browser from '../dist/browser.js'
import * as core from '../dist/core.js'
import * as protocol from '../dist/protocolEntry.js'
import * as server from '../dist/server.js'

describe('built package boundaries', () => {
  it('publishes explicit source-backed workspace entry points', async () => {
    const packageJson = JSON.parse(await readFile(
      new URL('../package.json', import.meta.url),
      'utf8',
    ))

    expect(Object.keys(packageJson.exports).sort()).toEqual([
      '.',
      './browser',
      './core',
      './protocol',
      './server',
    ])
    expect(core.EDITOR_SCHEMA_DEPENDENCY_VERSIONS.editorSchema).toBe(packageJson.version)
  })

  it('emits usable runtime entry points without crossing browser/server boundaries', () => {
    expect(core.createEditorSchemaExtensions).toBeTypeOf('function')
    expect(core).not.toHaveProperty('createRelativePositionAdapter')
    expect(core).not.toHaveProperty('editorJsonToYDoc')

    expect(browser.createRelativePositionAdapter).toBeTypeOf('function')
    expect(browser).not.toHaveProperty('editorJsonToYDoc')

    expect(server.editorJsonToYDoc).toBeTypeOf('function')
    expect(server).not.toHaveProperty('createRelativePositionAdapter')

    expect(protocol.editorCollaborationRoom).toBeTypeOf('function')
    expect(protocol.EDITOR_COLLABORATION_PROTOCOL_VERSION).toBe(1)
    expect(protocol).not.toHaveProperty('createEditorSchemaExtensions')
  })
})
