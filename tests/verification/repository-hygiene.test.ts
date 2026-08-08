import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import {
  readFileSync,
  readdirSync,
} from 'node:fs'
import { basename, dirname, resolve } from 'node:path'
import { describe, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import { readInqtrixVersion } from './report.ts'

const REPOSITORY_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../..',
)

const PRIVATE_PATH_EXAMPLES = [
  'AGENTS.md',
  'agents.md',
  'Agents.md',
  'CLAUDE.md',
  'claude.md',
  'Claude.md',
  'CURSOR.md',
  'cursor.md',
  'Cursor.md',
  '.cursor/rules/private.mdc',
  '.claude/private.md',
] as const

const FORBIDDEN_SOURCE_MARKERS = [
  {
    label: 'private plan milestone',
    pattern: new RegExp('\\b' + 'plan[ -]' + 'M\\d+[a-z]?', 'i'),
  },
  {
    label: 'private program phase',
    pattern: new RegExp('\\b' + 'Programm-' + '\\d+', 'i'),
  },
  {
    label: 'private decision identifier',
    pattern: new RegExp('\\b' + '(?:decision|Entscheidung)\\s+' + 'E\\d+', 'i'),
  },
  {
    label: 'private ADR identifier',
    pattern: new RegExp('\\b' + 'ADR-' + '[A-Z][A-Z0-9_-]*-\\d+', 'i'),
  },
  {
    label: 'priority-labelled implementation history',
    pattern: new RegExp(
      '\\b' + 'P\\d+\\s*(?:fix|bug|finding|review|regression|lesson|incident)',
      'i',
    ),
  },
  {
    label: 'incident chronology',
    pattern: new RegExp('\\b' + 'live\\s+incident' + '\\b', 'i'),
  },
  {
    label: 'dated implementation decision',
    pattern: new RegExp(
      '\\b' + '(?:user\\s+decision|deferred)\\s+20\\d{2}' + '\\b',
      'i',
    ),
  },
  {
    label: 'migration creation chronology',
    pattern: new RegExp('\\b' + 'Create\\s+Date\\s*:', 'i'),
  },
  {
    label: 'removed baseline metadata',
    pattern: new RegExp(
      '(?:`|")' + '_established' + '(?:`|")',
      'i',
    ),
  },
] as const

describe('repository version-control boundary', () => {
  test('keeps every private agent-memory path untracked', () => {
    const tracked = git(['ls-files', '-z'])
      .split('\0')
      .filter(Boolean)
      .filter(isPrivateAgentPath)
    assert.deepEqual(
      tracked,
      [],
      `Private agent-memory paths are tracked:\n${tracked.join('\n')}`,
    )
  })

  test('keeps representative spelling variants ignored', () => {
    const result = spawnSync(
      'git',
      ['check-ignore', '--no-index', '--stdin'],
      {
        cwd: REPOSITORY_ROOT,
        encoding: 'utf8',
        input: `${PRIVATE_PATH_EXAMPLES.join('\n')}\n`,
        stdio: ['pipe', 'pipe', 'pipe'],
      },
    )
    assert.equal(
      result.status,
      0,
      `git check-ignore failed: ${result.stderr.trim()}`,
    )
    assert.deepEqual(
      new Set(result.stdout.trim().split('\n').filter(Boolean)),
      new Set(PRIVATE_PATH_EXAMPLES),
    )
  })
})

describe('timeless versioned source', () => {
  test('contains no known private plan, incident, or chronology markers', () => {
    const violations: string[] = []
    for (const relativePath of trackedTextPaths()) {
      const source = readFileSync(resolve(REPOSITORY_ROOT, relativePath), 'utf8')
      for (const [index, line] of source.split(/\r?\n/).entries()) {
        for (const marker of FORBIDDEN_SOURCE_MARKERS) {
          if (marker.pattern.test(line)) {
            violations.push(
              `${relativePath}:${index + 1}: ${marker.label}: ${line.trim()}`,
            )
          }
        }
      }
    }
    assert.deepEqual(
      violations,
      [],
      `Versioned source contains private chronology:\n${violations.join('\n')}`,
    )
  })

  test('binds every eval baseline to the current Inqtrix version', () => {
    const baselineDirectory = resolve(REPOSITORY_ROOT, 'tests/eval/baselines')
    const expectedVersion = readInqtrixVersion()
    const violations: string[] = []
    for (const name of readdirSync(baselineDirectory).filter(
      (entry) => entry.endsWith('.json'),
    )) {
      const baseline = JSON.parse(
        readFileSync(resolve(baselineDirectory, name), 'utf8'),
      ) as Record<string, unknown>
      if (baseline._inqtrix_version !== expectedVersion) {
        violations.push(
          `${name}: _inqtrix_version must be ${expectedVersion}`,
        )
      }
      if (
        typeof baseline._context !== 'string'
        || baseline._context.trim().length === 0
      ) {
        violations.push(`${name}: _context must be a non-empty string`)
      }
      if ('_established' in baseline) {
        violations.push(`${name}: _established is not permitted`)
      }
    }
    assert.deepEqual(violations, [], violations.join('\n'))
  })
})

function trackedTextPaths(): string[] {
  return git(['ls-files', '-z'])
    .split('\0')
    .filter(Boolean)
    .filter((relativePath) => (
      relativePath !== 'package-lock.json'
      && relativePath !== 'uv.lock'
      && !relativePath.startsWith('docs/archive/github-actions/')
    ))
    .filter((relativePath) => {
      const content = readFileSync(resolve(REPOSITORY_ROOT, relativePath))
      return !content.includes(0)
    })
}

function isPrivateAgentPath(relativePath: string): boolean {
  const normalized = relativePath.replaceAll('\\', '/').toLowerCase()
  return (
    normalized.startsWith('.cursor/')
    || normalized.startsWith('.claude/')
    || ['agents.md', 'claude.md', 'cursor.md'].includes(basename(normalized))
  )
}

function git(args: readonly string[]): string {
  const result = spawnSync('git', [...args], {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  assert.equal(
    result.status,
    0,
    `git ${args.join(' ')} failed: ${result.stderr.trim()}`,
  )
  return result.stdout
}
