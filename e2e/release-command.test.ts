import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname } from 'node:path'
import { describe, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  assertReleaseE2EArguments,
  assertReleaseE2EEnvironment,
  executeReleaseE2E,
} from './release-command.ts'

const runner = fileURLToPath(new URL('./release-command.ts', import.meta.url))
const repositoryRoot = dirname(dirname(runner))

describe('collaboration E2E release command', () => {
  test('rejects every forwarded argument before spawning Playwright', () => {
    for (const argument of [
      '--help',
      '--list',
      '--reporter=line',
      '--grep=@direct-edit',
      'collaboration.spec.ts',
    ]) {
      let spawnCalls = 0
      assert.throws(
        () => executeReleaseE2E([argument], {}, () => {
          spawnCalls += 1
          return { signal: null, status: 0 }
        }),
        /accepts no command-line arguments.*reporter overrides/s,
      )
      assert.equal(spawnCalls, 0)
    }
  })

  test('spawns only the fixed Playwright test command in release mode', () => {
    const calls: Array<{
      args: readonly string[]
      command: string
      cwd: string | URL | undefined
      environment: NodeJS.ProcessEnv | undefined
      mode: string | undefined
      stdio: unknown
    }> = []
    const environment = {
      CI: '1',
      HTTPS_PROXY: 'http://proxy.example.test:8080',
      INQTRIX_E2E_CONTROL_TOKEN: 'test-only-control-value',
      INQTRIX_E2E_FIXTURE: '/tmp/test-fixture.json',
      PATH: '/test-path',
      PLAYWRIGHT_BROWSERS_PATH: '/test-browsers',
    }
    const status = executeReleaseE2E([], environment, (command, args, options) => {
      calls.push({
        args,
        command,
        cwd: options.cwd,
        environment: options.env,
        mode: options.env?.INQTRIX_E2E_MODE,
        stdio: options.stdio,
      })
      return { signal: null, status: 17 }
    })

    assert.equal(status, 17)
    assert.equal(calls.length, 1)
    assert.equal(calls[0]?.command, process.execPath)
    assert.match(calls[0]?.args[0] ?? '', /@playwright[\\/]test[\\/]cli\.js$/)
    assert.deepEqual(calls[0]?.args.slice(1), ['test'])
    assert.equal(calls[0]?.cwd, repositoryRoot)
    for (const [name, value] of Object.entries(environment)) {
      assert.equal(calls[0]?.environment?.[name], value)
    }
    assert.equal(calls[0]?.mode, 'release')
    assert.equal(calls[0]?.stdio, 'inherit')
  })

  test('the executable guard rejects bypass switches before fixture preflight', () => {
    for (const argument of ['--help', '--list', '--reporter=json', '--grep=@layout']) {
      const result = spawnSync(process.execPath, [
        '--disable-warning=MODULE_TYPELESS_PACKAGE_JSON',
        '--experimental-strip-types',
        runner,
        argument,
      ], { encoding: 'utf8' })

      assert.equal(result.status, 1)
      assert.match(result.stderr, /accepts no command-line arguments/)
      assert.doesNotMatch(result.stderr, /INQTRIX_E2E_FIXTURE/)
      assert.equal(result.stdout, '')
    }
  })

  test('absolute invocation from a non-repository cwd reaches fixture preflight and fails closed', () => {
    const cwd = mkdtempSync('/tmp/inqtrix-release-cwd-')
    const result = spawnSync(process.execPath, [
      '--disable-warning=MODULE_TYPELESS_PACKAGE_JSON',
      '--experimental-strip-types',
      runner,
    ], {
      cwd,
      encoding: 'utf8',
      env: cleanReleaseEnvironment(),
    })

    assert.match(cwd, /^\/tmp\/inqtrix-release-cwd-/)
    assert.equal(result.status, 1)
    assert.match(result.stderr, /INQTRIX_E2E_FIXTURE is not set/)
    assert.doesNotMatch(result.stderr, /Cannot find module|Unknown file extension/)
  })

  for (const [name, value] of [
    ['PWTEST_WATCH', '1'],
    ['PW_TEST_SOURCE_TRANSFORM', '/tmp/transform.js'],
    ['PW_TEST_REPORTER', 'line'],
    ['PLAYWRIGHT_JSON_OUTPUT_NAME', '/tmp/results.json'],
  ] as const) {
    test(`rejects ${name} in a child process before scenarios can be bypassed`, () => {
      const result = spawnSync(process.execPath, [
        '--disable-warning=MODULE_TYPELESS_PACKAGE_JSON',
        '--experimental-strip-types',
        runner,
      ], {
        cwd: tmpdir(),
        encoding: 'utf8',
        env: { ...cleanReleaseEnvironment(), [name]: value },
      })

      assert.equal(result.status, 1)
      assert.match(result.stderr, new RegExp(`forbids test-runner environment controls: ${name}`))
      assert.doesNotMatch(result.stderr, /INQTRIX_E2E_FIXTURE is not set/)
      assert.equal(result.stdout, '')
    })
  }

  test('rejects all runner prefixes while preserving explicit browser runtime controls', () => {
    for (const name of [
      'NODE_OPTIONS',
      'PWDEBUG',
      'PWTEST_CUSTOM_BYPASS',
      'PW_TEST_CUSTOM_BYPASS',
      'PLAYWRIGHT_TEST_BASE_URL',
    ]) {
      assert.throws(
        () => assertReleaseE2EEnvironment({ [name]: 'test-value' }),
        new RegExp(name),
      )
    }
    assert.doesNotThrow(() => assertReleaseE2EEnvironment({
      CI: '1',
      HTTP_PROXY: 'http://proxy.example.test',
      INQTRIX_E2E_FIXTURE: '/tmp/fixture.json',
      PLAYWRIGHT_BROWSERS_PATH: '/tmp/browsers',
    }))
  })

  test('the argument assertion accepts only the empty release invocation', () => {
    assert.doesNotThrow(() => assertReleaseE2EArguments([]))
    assert.throws(() => assertReleaseE2EArguments(['--workers=1']))
  })
})

function cleanReleaseEnvironment(): NodeJS.ProcessEnv {
  return Object.fromEntries(
    Object.entries(process.env).filter(([name]) => (
      name !== 'INQTRIX_E2E_FIXTURE'
      && name !== 'NODE_OPTIONS'
      && name !== 'PWDEBUG'
      && !name.startsWith('PWTEST_')
      && !name.startsWith('PW_TEST_')
      && !name.startsWith('PLAYWRIGHT_')
    )),
  )
}
