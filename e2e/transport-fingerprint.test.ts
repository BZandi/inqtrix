import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  assertTransportFingerprint,
  type TransportObservation,
} from './transport-fingerprint.ts'

const observations = {
  dist: observation({ serverHeader: 'uvicorn' }),
  nginx: observation({ serverHeader: 'nginx/1.27.5' }),
  vite: observation({
    serverHeader: '',
    viteClientContentType: 'text/javascript',
    viteClientMarker: true,
  }),
} as const

test('hardcoded runtime evidence distinguishes Vite, nginx, and dist', () => {
  assert.equal(assertTransportFingerprint('vite', observations.vite), 'vite')
  assert.equal(assertTransportFingerprint('nginx', observations.nginx), 'nginx')
  assert.equal(assertTransportFingerprint('dist', observations.dist), 'dist')
})

test('three URLs reaching any one transport cannot satisfy the release matrix', () => {
  for (const [actual, sameObservation] of Object.entries(observations)) {
    const accepted = (['vite', 'nginx', 'dist'] as const).filter((expected) => {
      try {
        assertTransportFingerprint(expected, sameObservation)
        return true
      } catch {
        return false
      }
    })
    assert.deepEqual(accepted, [actual])
  }
})

test('SPA fallbacks and missing server identity fail closed', () => {
  assert.throws(
    () => assertTransportFingerprint('vite', observation({
      viteClientContentType: 'text/html',
      viteClientMarker: false,
    })),
    /Vite client marker absent/,
  )
  assert.throws(
    () => assertTransportFingerprint('dist', observation({ serverHeader: '' })),
    /Server header was <missing>/,
  )
})

function observation(
  overrides: Partial<TransportObservation> = {},
): TransportObservation {
  return {
    rootContentType: 'text/html; charset=utf-8',
    rootStatus: 200,
    serverHeader: '',
    viteClientContentType: 'text/html; charset=utf-8',
    viteClientMarker: false,
    viteClientStatus: 200,
    ...overrides,
  }
}
