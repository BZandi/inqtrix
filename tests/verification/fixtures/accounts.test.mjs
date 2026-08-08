import assert from 'node:assert/strict'
import test from 'node:test'

import {
  ensureTemporaryUsers,
  temporaryUserBelongsToRun,
  temporaryUserDescriptors,
} from './accounts.mjs'

test('temporary user provisioning derives default identities from its run', async () => {
  const runId = 'inqv-account-default-identities'
  const expectedDescriptors = temporaryUserDescriptors(runId)
  const registrations = []
  const requests = []
  const actor = {
    context: {
      request: {
        async fetch(path, options) {
          requests.push({ method: options.method, path })
          if (options.method === 'GET') return jsonResponse(200, { users: [] })
          return jsonResponse(201, {
            email: options.data.email,
            id: `user-${requests.length}`,
          })
        },
      },
    },
    csrf: 'csrf',
    label: 'fixture owner',
    user: { id: 'owner' },
    workspaceId: 'workspace',
  }
  const lifecycle = {
    async register(resource) {
      registrations.push(resource)
      return `cleanup-${registrations.length}`
    },
  }

  const users = await ensureTemporaryUsers({
    adminActor: actor,
    lifecycle,
    password: 'fixture password',
    runId,
  })

  assert.deepEqual(
    users.map(({ displayName, email }) => ({ displayName, email })),
    expectedDescriptors,
  )
  assert.deepEqual(
    registrations.map(({ email, id, kind }) => ({ email, id, kind })),
    expectedDescriptors.map(({ email }) => ({
      email,
      id: `${runId}:${email}`,
      kind: 'temporary_user',
    })),
  )
  assert.equal(requests.filter(({ method }) => method === 'GET').length, 1)
  assert.equal(requests.filter(({ method }) => method === 'POST').length, 4)
})

test('temporary user scope supports the 24-user soak cohort without widening runs', () => {
  const firstRun = 'inqv-account-soak-identities-a'
  const secondRun = 'inqv-account-soak-identities-b'
  const descriptors = temporaryUserDescriptors(firstRun, 24)
  assert.equal(descriptors.length, 24)
  assert.equal(new Set(descriptors.map(({ email }) => email)).size, 24)
  assert(descriptors.every(({ email }) => temporaryUserBelongsToRun(email, firstRun)))
  assert(descriptors.every(({ email }) => !temporaryUserBelongsToRun(email, secondRun)))
  assert.throws(() => temporaryUserDescriptors(firstRun, 25), /at most 24/)
})

function jsonResponse(status, body) {
  return {
    status: () => status,
    text: async () => JSON.stringify(body),
  }
}
