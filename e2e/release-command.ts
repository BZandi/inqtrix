import { spawnSync, type SpawnSyncOptions } from 'node:child_process'
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

type Environment = NodeJS.ProcessEnv

type ReleaseSpawnResult = {
  error?: Error
  signal: NodeJS.Signals | null
  status: number | null
}

type ReleaseSpawn = (
  command: string,
  args: readonly string[],
  options: SpawnSyncOptions,
) => ReleaseSpawnResult

const RELEASE_ARGUMENT_ERROR = [
  'Collaboration release E2E accepts no command-line arguments.',
  '--help, --list, reporter overrides, and all other Playwright overrides are forbidden;',
  'use e2e:dev or e2e:list for developer controls.',
].join(' ')

const MODULE_PATH = fileURLToPath(import.meta.url)
const REPOSITORY_ROOT = resolve(dirname(MODULE_PATH), '..')
const ALLOWED_PLAYWRIGHT_ENVIRONMENT = new Set([
  'PLAYWRIGHT_BROWSERS_PATH',
  'PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST',
  'PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT',
  'PLAYWRIGHT_DOWNLOAD_HOST',
  'PLAYWRIGHT_FIREFOX_DOWNLOAD_HOST',
  'PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD',
  'PLAYWRIGHT_WEBKIT_DOWNLOAD_HOST',
])

export function assertReleaseE2EEnvironment(environment: Environment): void {
  const forbidden = Object.keys(environment)
    .filter((name) => (
      name === 'NODE_OPTIONS'
      || name === 'PWDEBUG'
      || name.startsWith('PWTEST_')
      || name.startsWith('PW_TEST_')
      || (name.startsWith('PLAYWRIGHT_') && !ALLOWED_PLAYWRIGHT_ENVIRONMENT.has(name))
    ))
    .sort()
  if (forbidden.length > 0) {
    throw new Error(
      `Collaboration release E2E forbids test-runner environment controls: ${forbidden.join(', ')}.`,
    )
  }
}

export function assertReleaseE2EArguments(args: readonly string[]): void {
  if (args.length === 0) return
  throw new Error(RELEASE_ARGUMENT_ERROR)
}

export function executeReleaseE2E(
  args: readonly string[] = process.argv.slice(2),
  environment: Environment = process.env,
  spawn: ReleaseSpawn = spawnSync,
): number {
  assertReleaseE2EArguments(args)
  assertReleaseE2EEnvironment(environment)
  const require = createRequire(import.meta.url)
  const playwrightCli = require.resolve('@playwright/test/cli')
  const result = spawn(process.execPath, [playwrightCli, 'test'], {
    cwd: REPOSITORY_ROOT,
    env: { ...environment, INQTRIX_E2E_MODE: 'release' },
    stdio: 'inherit',
  })
  if (result.error) {
    throw new Error('Collaboration release E2E could not start Playwright.')
  }
  if (result.signal) {
    throw new Error('Collaboration release E2E Playwright process terminated by a signal.')
  }
  return result.status ?? 1
}

function main(): void {
  try {
    process.exitCode = executeReleaseE2E()
  } catch (error) {
    const message = error instanceof Error
      ? error.message
      : 'Collaboration release E2E failed before Playwright started.'
    process.stderr.write(`${message}\n`)
    process.exitCode = 1
  }
}

if (resolve(process.argv[1] ?? '') === MODULE_PATH) main()
