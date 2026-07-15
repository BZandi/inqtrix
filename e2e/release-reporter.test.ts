import assert from 'node:assert/strict'
import { test } from 'node:test'

import type {
  FullConfig,
  FullResult,
  Suite,
  TestCase,
  TestResult,
} from '@playwright/test/reporter'

import CollaborationReleaseReporter from './release-reporter.ts'
import {
  RELEASE_DESKTOP_SCENARIOS,
  RELEASE_MOBILE_SCENARIOS,
} from './release-contract.ts'

test('release scenario contract mandates critical shared and read-only behavior everywhere', () => {
  for (const tag of [
    '@concurrent-edits',
    '@gateway-outage',
    '@layout',
    '@permission-downgrade',
    '@revocation',
    '@suggestions',
    '@detached-transfer',
    '@source-readonly',
    '@transport-fingerprint',
  ]) {
    assert((RELEASE_DESKTOP_SCENARIOS as readonly string[]).includes(tag), `${tag} must be required on desktop`)
    assert((RELEASE_MOBILE_SCENARIOS as readonly string[]).includes(tag), `${tag} must be required on mobile`)
  }
})

test('release reporter converts any required runtime skip into a failed run', async () => {
  const reporter = new CollaborationReleaseReporter()
  reporter.onTestEnd({
    parent: { project: () => ({ name: 'vite-mobile' }) },
    title: '@outage @mobile sidecar outage',
  } as unknown as TestCase, {
    status: 'skipped',
  } as TestResult)

  const originalWrite = process.stderr.write
  let output = ''
  process.stderr.write = ((chunk: string | Uint8Array) => {
    output += String(chunk)
    return true
  }) as typeof process.stderr.write
  try {
    const result = await reporter.onEnd({
      duration: 1,
      startTime: new Date(0),
      status: 'passed',
    } satisfies FullResult)
    assert.deepEqual(result, { status: 'failed' })
    assert.match(output, /required test skipped: vite-mobile/)
  } finally {
    process.stderr.write = originalWrite
  }
})

test('release reporter fails when a transport project loses required scenarios', async () => {
  const reporter = new CollaborationReleaseReporter()
  reporter.onBegin({} as FullConfig, {
    allTests: () => [],
  } as unknown as Suite)

  const originalWrite = process.stderr.write
  let output = ''
  process.stderr.write = ((chunk: string | Uint8Array) => {
    output += String(chunk)
    return true
  }) as typeof process.stderr.write
  try {
    const result = await reporter.onEnd({
      duration: 1,
      startTime: new Date(0),
      status: 'passed',
    } satisfies FullResult)
    assert.deepEqual(result, { status: 'failed' })
    assert.match(output, /vite-desktop selected no tests/)
    assert.match(output, /dist-mobile selected no tests/)
  } finally {
    process.stderr.write = originalWrite
  }
})
