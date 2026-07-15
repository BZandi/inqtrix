import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, test } from 'node:test'

import {
  assertReleaseE2EConfiguration,
  loadCollaborationE2EConfiguration,
  releasePreflightReasons,
  resolveCollaborationE2EMode,
} from './config.ts'

describe('collaboration E2E release preflight', () => {
  test('developer mode reports an absent fixture without pretending it ran', () => {
    const configuration = loadCollaborationE2EConfiguration({}, '/workspace')

    assert.equal(configuration.mode, 'dev')
    assert.equal(configuration.stack, null)
    assert.deepEqual(configuration.reasons, ['INQTRIX_E2E_FIXTURE is not set'])
  })

  test('release mode rejects an absent fixture', () => {
    const environment = { INQTRIX_E2E_MODE: 'release' }
    const configuration = loadCollaborationE2EConfiguration(environment, '/workspace')

    assert.throws(
      () => assertReleaseE2EConfiguration(configuration, environment),
      /INQTRIX_E2E_FIXTURE is not set/,
    )
  })

  test('release mode requires private anchors, fault controls, and critical documents', () => {
    const complete = validFixture()
    const fixture = {
      ...complete,
      controls: undefined,
      documents: {
        directEdit: complete.documents.directEdit,
        revocation: complete.documents.revocation,
        suggestion: complete.documents.suggestion,
      },
      privateAnchors: undefined,
    }
    const { configuration, environment } = loadFixture(fixture)

    assert.deepEqual(releasePreflightReasons(configuration, environment), [
      'fixture.documents.concurrent is required in release mode',
      'fixture.documents.downgrade is required in release mode',
      'fixture.documents.detachedTransfer is required in release mode',
      'fixture.documents.gatewayOutage is required in release mode',
      'fixture.documents.reconciliation is required in release mode',
      'fixture.documents.outage is required in release mode',
      'fixture.documents.protocol is required in release mode',
      'fixture.privateAnchors is required in release mode',
      'fixture.controls is required in release mode',
    ])
  })

  test('release mode rejects path aliases on one origin', () => {
    const fixture = validFixture()
    fixture.transports.dist.baseURL = 'https://vite.example.test/dist/'
    const { configuration, environment } = loadFixture(fixture)

    assert.match(
      releasePreflightReasons(configuration, environment).join('\n'),
      /must use three distinct origins/,
    )
  })

  test('configured application base paths remain navigable roots', () => {
    const fixture = validFixture()
    fixture.transports.vite.baseURL = 'https://vite.example.test/inqtrix'
    const { configuration } = loadFixture(fixture)

    assert.equal(
      configuration.stack?.transports.vite.baseURL,
      'https://vite.example.test/inqtrix/',
    )
  })

  test('release mode rejects transport URLs that could carry credentials', () => {
    const fixture = validFixture()
    fixture.transports.dist.baseURL = 'https://user:password@dist.example.test'
    const { configuration, environment } = loadFixture(fixture)

    assert.match(
      releasePreflightReasons(configuration, environment).join('\n'),
      /dist baseURL must not contain credentials/,
    )
  })

  test('release mode requires concrete private anchor ranges, not marker cards alone', () => {
    const fixture = validFixture()
    fixture.privateAnchors.owner.aiAnchorText = ''
    const { configuration, environment } = loadFixture(fixture)

    assert.match(
      releasePreflightReasons(configuration, environment).join('\n'),
      /fixture\.privateAnchors\.owner\.aiAnchorText must be a non-empty string/,
    )
    assert.match(
      releasePreflightReasons(configuration, environment).join('\n'),
      /fixture\.privateAnchors is required in release mode/,
    )
  })

  test('release mode binds the suggestion document to a real suggest principal', () => {
    const permissionFixture = validFixture()
    permissionFixture.documents.suggestion.expectedPermission = 'edit'
    const permission = loadFixture(permissionFixture)
    assert.match(
      releasePreflightReasons(permission.configuration, permission.environment).join('\n'),
      /expectedPermission must equal "suggest"/,
    )

    const identityFixture = validFixture()
    identityFixture.documents.suggestion.expectedAuthorId = '00000000-0000-4000-8000-000000000003'
    const identity = loadFixture(identityFixture)
    assert.match(
      releasePreflightReasons(identity.configuration, identity.environment).join('\n'),
      /expectedAuthorId must equal fixture\.users\.collaborator\.userId/,
    )
  })

  test('release mode requires a distinct FastAPI gateway fault document and control', () => {
    const fixture = validFixture()
    fixture.documents.gatewayOutage = ''
    fixture.controls.armGatewayOutagePath = ''
    const { configuration, environment } = loadFixture(fixture)
    const reasons = releasePreflightReasons(configuration, environment).join('\n')

    assert.match(reasons, /fixture\.documents\.gatewayOutage is required in release mode/)
    assert.match(reasons, /fixture\.controls\.armGatewayOutagePath must be a non-empty string/)
    assert.match(reasons, /fixture\.controls is required in release mode/)
  })

  test('a complete release fixture passes without weakening prerequisites', () => {
    const { configuration, environment } = loadFixture(validFixture())

    assert.equal(configuration.mode, 'release')
    assert.deepEqual(releasePreflightReasons(configuration, environment), [])
    assert.doesNotThrow(() => assertReleaseE2EConfiguration(configuration, environment))
  })

  test('release mode fails when the named control authorization value is absent', () => {
    const { configuration, environment } = loadFixture(validFixture())
    delete environment.INQTRIX_E2E_CONTROL_TOKEN

    assert.deepEqual(releasePreflightReasons(configuration, environment), [
      'INQTRIX_E2E_CONTROL_TOKEN is required for fixture control authorization in release mode',
    ])
  })

  test('mode parsing rejects ambiguous values', () => {
    assert.equal(resolveCollaborationE2EMode({}), 'dev')
    assert.throws(
      () => resolveCollaborationE2EMode({ INQTRIX_E2E_MODE: 'strict' }),
      /must be "dev" or "release"/,
    )
  })
})

function loadFixture(fixture: unknown): {
  configuration: ReturnType<typeof loadCollaborationE2EConfiguration>
  environment: Record<string, string>
} {
  const directory = mkdtempSync(join(tmpdir(), 'inqtrix-e2e-config-'))
  writeFileSync(join(directory, 'owner.json'), '{}')
  writeFileSync(join(directory, 'collaborator.json'), '{}')
  writeFileSync(join(directory, 'fixture.json'), JSON.stringify(fixture))
  const environment = {
    INQTRIX_E2E_CONTROL_TOKEN: 'test-only-control-token',
    INQTRIX_E2E_FIXTURE: 'fixture.json',
    INQTRIX_E2E_MODE: 'release',
  }
  return {
    configuration: loadCollaborationE2EConfiguration(environment, directory),
    environment,
  }
}

function validFixture() {
  return {
    controls: {
      armGatewayOutagePath: '/v1/test/collaboration/gateway-outage:arm',
      armLostAckPath: '/v1/test/collaboration/lost-ack:arm',
      armOutagePath: '/v1/test/collaboration/outage:arm',
      authorizationEnv: 'INQTRIX_E2E_CONTROL_TOKEN',
      baseURL: 'https://control.example.test',
      operationStatusPath: '/v1/test/collaboration/operation:status',
      restartPath: '/v1/test/collaboration/restart',
      restorePath: '/v1/test/collaboration/restore',
    },
    documents: {
      concurrent: 'concurrent-document',
      detachedTransfer: 'detached-transfer-document',
      directEdit: 'direct-document',
      downgrade: 'downgrade-document',
      gatewayOutage: 'gateway-outage-document',
      outage: 'outage-document',
      protocol: 'protocol-document',
      reconciliation: 'reconciliation-document',
      revocation: 'revocation-document',
      suggestion: {
        documentId: 'suggestion-document',
        expectedAuthorId: '00000000-0000-4000-8000-000000000002',
        expectedPermission: 'suggest',
      },
    },
    locale: 'en',
    privateAnchors: {
      collaborator: {
        aiAnchorText: 'collaborator-ai-anchor',
        aiText: 'collaborator-ai-marker',
        commentAnchorText: 'collaborator-comment-anchor',
        commentText: 'collaborator-comment-marker',
      },
      documentId: 'private-anchor-document',
      owner: {
        aiAnchorText: 'owner-ai-anchor',
        aiText: 'owner-ai-marker',
        commentAnchorText: 'owner-comment-anchor',
        commentText: 'owner-comment-marker',
      },
    },
    transports: {
      dist: { baseURL: 'https://dist.example.test' },
      nginx: { baseURL: 'https://nginx.example.test' },
      vite: { baseURL: 'https://vite.example.test' },
    },
    users: {
      collaborator: {
        displayName: 'Collaborator User',
        storageState: './collaborator.json',
        userId: '00000000-0000-4000-8000-000000000002',
      },
      owner: {
        displayName: 'Owner User',
        storageState: './owner.json',
      },
    },
    version: 2,
  }
}
