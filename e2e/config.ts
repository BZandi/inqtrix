import { existsSync, readFileSync } from 'node:fs'
import { dirname, isAbsolute, resolve } from 'node:path'

export const COLLABORATION_TRANSPORTS = ['vite', 'nginx', 'dist'] as const
export const COLLABORATION_E2E_MODES = ['dev', 'release'] as const

export type CollaborationTransport = typeof COLLABORATION_TRANSPORTS[number]
export type CollaborationE2EMode = typeof COLLABORATION_E2E_MODES[number]
export type CollaborationLocale = 'de' | 'en'

type Environment = Record<string, string | undefined>

type RawUser = {
  displayName?: unknown
  storageState?: unknown
  userId?: unknown
}

type RawTransport = {
  baseURL?: unknown
  collaboratorStorageState?: unknown
  ownerStorageState?: unknown
}

type RawSuggestionDocument = {
  documentId?: unknown
  expectedAuthorId?: unknown
  expectedPermission?: unknown
}

type RawControls = {
  armGatewayOutagePath?: unknown
  armLostAckPath?: unknown
  armOutagePath?: unknown
  authorizationEnv?: unknown
  baseURL?: unknown
  operationStatusPath?: unknown
  restartPath?: unknown
  restorePath?: unknown
}

type RawFixture = {
  controls?: RawControls
  documents?: {
    concurrent?: unknown
    detachedTransfer?: unknown
    directEdit?: unknown
    downgrade?: unknown
    gatewayOutage?: unknown
    outage?: unknown
    protocol?: unknown
    reconciliation?: unknown
    revocation?: unknown
    suggestion?: RawSuggestionDocument
  }
  locale?: unknown
  privateAnchors?: {
    collaborator?: {
      aiAnchorText?: unknown
      aiText?: unknown
      commentAnchorText?: unknown
      commentText?: unknown
    }
    documentId?: unknown
    owner?: {
      aiAnchorText?: unknown
      aiText?: unknown
      commentAnchorText?: unknown
      commentText?: unknown
    }
  }
  transports?: Partial<Record<CollaborationTransport, RawTransport>>
  users?: {
    collaborator?: RawUser
    owner?: RawUser
  }
  version?: unknown
}

export type CollaborationControlFixture = {
  authorizationEnv: string
  baseURL: string
  paths: {
    armGatewayOutage: string
    armLostAck: string
    armOutage: string
    operationStatus: string
    restart: string
    restore: string
  }
}

export type CollaborationE2EStack = {
  capabilityReasons: {
    controls: string[]
    privateAnchors: string[]
  }
  collaborator: {
    displayName: string
    storageState: string
    userId: string
  }
  controls: CollaborationControlFixture | null
  documents: {
    concurrent: string | null
    detachedTransfer: string | null
    directEdit: string
    downgrade: string | null
    gatewayOutage: string | null
    outage: string | null
    protocol: string | null
    reconciliation: string | null
    revocation: string
    suggestion: {
      documentId: string
      expectedAuthorId: string
      expectedPermission: 'suggest'
    }
  }
  locale: CollaborationLocale
  owner: {
    displayName: string
    storageState: string
  }
  privateAnchors: {
    collaborator: {
      aiAnchorText: string
      aiText: string
      commentAnchorText: string
      commentText: string
    }
    documentId: string
    owner: {
      aiAnchorText: string
      aiText: string
      commentAnchorText: string
      commentText: string
    }
  } | null
  transports: Record<CollaborationTransport, {
    baseURL: string | null
    collaboratorStorageState: string
    ownerStorageState: string
    reasons: string[]
  }>
}

export type CollaborationE2EConfiguration = {
  mode: CollaborationE2EMode
  reasons: string[]
  stack: CollaborationE2EStack | null
}

const TRANSPORT_ENV: Record<CollaborationTransport, string> = {
  dist: 'INQTRIX_E2E_DIST_BASE_URL',
  nginx: 'INQTRIX_E2E_NGINX_BASE_URL',
  vite: 'INQTRIX_E2E_VITE_BASE_URL',
}

export function resolveCollaborationE2EMode(
  environment: Environment = process.env,
): CollaborationE2EMode {
  const value = environment.INQTRIX_E2E_MODE ?? 'dev'
  if (value === 'dev' || value === 'release') return value
  throw new Error('INQTRIX_E2E_MODE must be "dev" or "release".')
}

export function loadCollaborationE2EConfiguration(
  environment: Environment = process.env,
  workingDirectory = process.cwd(),
): CollaborationE2EConfiguration {
  const mode = resolveCollaborationE2EMode(environment)
  const fixturePath = environment.INQTRIX_E2E_FIXTURE
  if (!fixturePath) {
    return { mode, reasons: ['INQTRIX_E2E_FIXTURE is not set'], stack: null }
  }

  const absoluteFixturePath = resolve(workingDirectory, fixturePath)
  if (!existsSync(absoluteFixturePath)) {
    return {
      mode,
      reasons: [`INQTRIX_E2E_FIXTURE does not exist: ${absoluteFixturePath}`],
      stack: null,
    }
  }

  let raw: RawFixture
  try {
    raw = JSON.parse(readFileSync(absoluteFixturePath, 'utf8')) as RawFixture
  } catch {
    return {
      mode,
      reasons: [`INQTRIX_E2E_FIXTURE is not valid JSON: ${absoluteFixturePath}`],
      stack: null,
    }
  }

  const reasons: string[] = []
  if (raw.version !== 2) reasons.push('fixture.version must equal 2')
  const locale = raw.locale === 'de' || raw.locale === 'en' ? raw.locale : null
  if (!locale) reasons.push('fixture.locale must be "de" or "en"')

  const owner = parseUser(raw.users?.owner, 'fixture.users.owner', false, absoluteFixturePath, reasons)
  const collaborator = parseUser(
    raw.users?.collaborator,
    'fixture.users.collaborator',
    true,
    absoluteFixturePath,
    reasons,
  )
  const documents = {
    concurrent: optionalCapabilityString(raw.documents?.concurrent),
    detachedTransfer: optionalCapabilityString(raw.documents?.detachedTransfer),
    directEdit: requiredString(raw.documents?.directEdit, 'fixture.documents.directEdit', reasons),
    downgrade: optionalCapabilityString(raw.documents?.downgrade),
    gatewayOutage: optionalCapabilityString(raw.documents?.gatewayOutage),
    outage: optionalCapabilityString(raw.documents?.outage),
    protocol: optionalCapabilityString(raw.documents?.protocol),
    reconciliation: optionalCapabilityString(raw.documents?.reconciliation),
    revocation: requiredString(raw.documents?.revocation, 'fixture.documents.revocation', reasons),
    suggestion: parseSuggestionDocument(raw.documents?.suggestion, reasons),
  }
  if (
    collaborator
    && documents.suggestion
    && documents.suggestion.expectedAuthorId !== collaborator.userId
  ) {
    reasons.push(
      'fixture.documents.suggestion.expectedAuthorId must equal fixture.users.collaborator.userId',
    )
  }
  const privateReasons: string[] = []
  const privateAnchors = parsePrivateAnchors(raw.privateAnchors, privateReasons)
  const controlReasons: string[] = []
  const controls = parseControls(raw.controls, controlReasons)
  const suggestion = documents.suggestion

  if (
    !owner
    || !collaborator
    || !locale
    || !documents.directEdit
    || !documents.revocation
    || !suggestion
  ) {
    return { mode, reasons, stack: null }
  }

  const fixtureDirectory = dirname(absoluteFixturePath)
  const transports = Object.fromEntries(COLLABORATION_TRANSPORTS.map((transport) => {
    const rawTransport = raw.transports?.[transport]
    const endpoint = nonEmptyString(environment[TRANSPORT_ENV[transport]])
      ?? nonEmptyString(rawTransport?.baseURL)
    const transportReasons: string[] = []
    if (!endpoint) {
      transportReasons.push(
        `${TRANSPORT_ENV[transport]} is not set and fixture.transports.${transport}.baseURL is absent`,
      )
    } else if (!isHttpUrl(endpoint)) {
      transportReasons.push(`${transport} baseURL must use http:// or https://`)
    } else if (hasUrlSecrets(endpoint)) {
      transportReasons.push(
        `${transport} baseURL must not contain credentials, query, or fragment data`,
      )
    }

    const ownerOverride = resolveOptionalPath(rawTransport?.ownerStorageState, fixtureDirectory)
    const collaboratorOverride = resolveOptionalPath(
      rawTransport?.collaboratorStorageState,
      fixtureDirectory,
    )
    const ownerState = ownerOverride ?? owner.storageState
    const collaboratorState = collaboratorOverride ?? collaborator.storageState
    if (!existsSync(ownerState)) transportReasons.push(`owner storageState does not exist: ${ownerState}`)
    if (!existsSync(collaboratorState)) {
      transportReasons.push(`collaborator storageState does not exist: ${collaboratorState}`)
    }

    return [transport, {
      baseURL: endpoint && isHttpUrl(endpoint) && !hasUrlSecrets(endpoint)
        ? normalizeHttpUrl(endpoint)
        : null,
      collaboratorStorageState: collaboratorState,
      ownerStorageState: ownerState,
      reasons: transportReasons,
    }]
  })) as CollaborationE2EStack['transports']

  return {
    mode,
    reasons,
    stack: {
      capabilityReasons: {
        controls: controlReasons,
        privateAnchors: privateReasons,
      },
      collaborator,
      controls,
      documents: { ...documents, suggestion },
      locale,
      owner,
      privateAnchors,
      transports,
    },
  }
}

export function releasePreflightReasons(
  configuration: CollaborationE2EConfiguration,
  environment: Environment = process.env,
): string[] {
  const reasons = [...configuration.reasons]
  const { stack } = configuration
  if (!stack) return reasons.length > 0 ? reasons : ['collaboration stack is unavailable']

  for (const transport of COLLABORATION_TRANSPORTS) {
    reasons.push(...stack.transports[transport].reasons)
  }
  const endpoints = COLLABORATION_TRANSPORTS
    .map((transport) => stack.transports[transport].baseURL)
    .filter((value): value is string => value !== null)
  const origins = endpoints.map((endpoint) => new URL(endpoint).origin)
  if (origins.length === COLLABORATION_TRANSPORTS.length && new Set(origins).size !== origins.length) {
    reasons.push('vite, nginx, and dist base URLs must use three distinct origins')
  }

  if (!stack.documents.concurrent) {
    reasons.push('fixture.documents.concurrent is required in release mode')
  }
  if (!stack.documents.downgrade) {
    reasons.push('fixture.documents.downgrade is required in release mode')
  }
  if (!stack.documents.detachedTransfer) {
    reasons.push('fixture.documents.detachedTransfer is required in release mode')
  }
  if (!stack.documents.gatewayOutage) {
    reasons.push('fixture.documents.gatewayOutage is required in release mode')
  }

  if (!stack.documents.reconciliation) {
    reasons.push('fixture.documents.reconciliation is required in release mode')
  }
  if (!stack.documents.outage) {
    reasons.push('fixture.documents.outage is required in release mode')
  }
  if (!stack.documents.protocol) {
    reasons.push('fixture.documents.protocol is required in release mode')
  }
  reasons.push(...stack.capabilityReasons.privateAnchors)
  if (!stack.privateAnchors) {
    reasons.push('fixture.privateAnchors is required in release mode')
  }
  reasons.push(...stack.capabilityReasons.controls)
  if (!stack.controls) {
    reasons.push('fixture.controls is required in release mode')
  } else if (!nonEmptyString(environment[stack.controls.authorizationEnv])) {
    reasons.push(
      `${stack.controls.authorizationEnv} is required for fixture control authorization in release mode`,
    )
  }
  return uniqueStrings(reasons)
}

export function assertReleaseE2EConfiguration(
  configuration: CollaborationE2EConfiguration,
  environment: Environment = process.env,
): void {
  if (configuration.mode !== 'release') return
  const reasons = releasePreflightReasons(configuration, environment)
  if (reasons.length === 0) return
  throw new Error(`Collaboration release E2E preflight failed:\n- ${reasons.join('\n- ')}`)
}

function parsePrivateAnchors(
  raw: RawFixture['privateAnchors'],
  reasons: string[],
): CollaborationE2EStack['privateAnchors'] {
  if (raw === undefined) return null
  const documentId = requiredString(raw.documentId, 'fixture.privateAnchors.documentId', reasons)
  const ownerAiAnchor = requiredString(
    raw.owner?.aiAnchorText,
    'fixture.privateAnchors.owner.aiAnchorText',
    reasons,
  )
  const ownerAi = requiredString(raw.owner?.aiText, 'fixture.privateAnchors.owner.aiText', reasons)
  const ownerCommentAnchor = requiredString(
    raw.owner?.commentAnchorText,
    'fixture.privateAnchors.owner.commentAnchorText',
    reasons,
  )
  const ownerComment = requiredString(
    raw.owner?.commentText,
    'fixture.privateAnchors.owner.commentText',
    reasons,
  )
  const collaboratorAi = requiredString(
    raw.collaborator?.aiText,
    'fixture.privateAnchors.collaborator.aiText',
    reasons,
  )
  const collaboratorAiAnchor = requiredString(
    raw.collaborator?.aiAnchorText,
    'fixture.privateAnchors.collaborator.aiAnchorText',
    reasons,
  )
  const collaboratorCommentAnchor = requiredString(
    raw.collaborator?.commentAnchorText,
    'fixture.privateAnchors.collaborator.commentAnchorText',
    reasons,
  )
  const collaboratorComment = requiredString(
    raw.collaborator?.commentText,
    'fixture.privateAnchors.collaborator.commentText',
    reasons,
  )
  if (
    !documentId
    || !ownerAiAnchor
    || !ownerAi
    || !ownerCommentAnchor
    || !ownerComment
    || !collaboratorAiAnchor
    || !collaboratorAi
    || !collaboratorCommentAnchor
    || !collaboratorComment
  ) return null
  return {
    collaborator: {
      aiAnchorText: collaboratorAiAnchor,
      aiText: collaboratorAi,
      commentAnchorText: collaboratorCommentAnchor,
      commentText: collaboratorComment,
    },
    documentId,
    owner: {
      aiAnchorText: ownerAiAnchor,
      aiText: ownerAi,
      commentAnchorText: ownerCommentAnchor,
      commentText: ownerComment,
    },
  }
}

function parseSuggestionDocument(
  raw: RawSuggestionDocument | undefined,
  reasons: string[],
): CollaborationE2EStack['documents']['suggestion'] | null {
  if (!isRecord(raw)) {
    reasons.push('fixture.documents.suggestion must be an object')
    return null
  }
  const documentId = requiredString(
    raw.documentId,
    'fixture.documents.suggestion.documentId',
    reasons,
  )
  const expectedAuthorId = requiredString(
    raw.expectedAuthorId,
    'fixture.documents.suggestion.expectedAuthorId',
    reasons,
  )
  const expectedPermission = requiredString(
    raw.expectedPermission,
    'fixture.documents.suggestion.expectedPermission',
    reasons,
  )
  if (expectedPermission && expectedPermission !== 'suggest') {
    reasons.push('fixture.documents.suggestion.expectedPermission must equal "suggest"')
  }
  if (expectedAuthorId && !isUuid(expectedAuthorId)) {
    reasons.push('fixture.documents.suggestion.expectedAuthorId must be a UUID')
  }
  if (
    !documentId
    || !expectedAuthorId
    || expectedPermission !== 'suggest'
    || !isUuid(expectedAuthorId)
  ) return null
  return { documentId, expectedAuthorId, expectedPermission }
}

function parseControls(
  raw: RawControls | undefined,
  reasons: string[],
): CollaborationControlFixture | null {
  if (raw === undefined) return null
  const baseURL = requiredString(raw.baseURL, 'fixture.controls.baseURL', reasons)
  const authorizationEnv = requiredString(
    raw.authorizationEnv,
    'fixture.controls.authorizationEnv',
    reasons,
  )
  const paths = {
    armGatewayOutage: requiredControlPath(
      raw.armGatewayOutagePath,
      'fixture.controls.armGatewayOutagePath',
      reasons,
    ),
    armLostAck: requiredControlPath(
      raw.armLostAckPath,
      'fixture.controls.armLostAckPath',
      reasons,
    ),
    armOutage: requiredControlPath(
      raw.armOutagePath,
      'fixture.controls.armOutagePath',
      reasons,
    ),
    operationStatus: requiredControlPath(
      raw.operationStatusPath,
      'fixture.controls.operationStatusPath',
      reasons,
    ),
    restart: requiredControlPath(raw.restartPath, 'fixture.controls.restartPath', reasons),
    restore: requiredControlPath(raw.restorePath, 'fixture.controls.restorePath', reasons),
  }
  if (baseURL && (!isHttpUrl(baseURL) || hasUrlSecrets(baseURL))) {
    reasons.push('fixture.controls.baseURL must be an HTTP(S) URL without credentials, query, or fragment')
  }
  if (authorizationEnv && !/^[A-Z][A-Z0-9_]*$/.test(authorizationEnv)) {
    reasons.push('fixture.controls.authorizationEnv must name an uppercase environment variable')
  }
  if (!baseURL || !authorizationEnv || Object.values(paths).some((value) => !value) || reasons.length > 0) {
    return null
  }
  return {
    authorizationEnv,
    baseURL: normalizeHttpUrl(baseURL),
    paths,
  }
}

function parseUser(
  raw: RawUser | undefined,
  field: string,
  requireUserId: boolean,
  fixturePath: string,
  reasons: string[],
): { displayName: string; storageState: string; userId: string } | null {
  const displayName = requiredString(raw?.displayName, `${field}.displayName`, reasons)
  const storageState = requiredString(raw?.storageState, `${field}.storageState`, reasons)
  const userId = requireUserId
    ? requiredString(raw?.userId, `${field}.userId`, reasons)
    : nonEmptyString(raw?.userId) ?? ''
  if (requireUserId && userId && !isUuid(userId)) {
    reasons.push(`${field}.userId must be a UUID`)
  }
  if (!displayName || !storageState || (requireUserId && (!userId || !isUuid(userId)))) return null
  return {
    displayName,
    storageState: resolveOptionalPath(storageState, dirname(fixturePath))!,
    userId,
  }
}

function requiredControlPath(value: unknown, field: string, reasons: string[]): string {
  const path = requiredString(value, field, reasons)
  if (!path) return ''
  if (!path.startsWith('/') || path.startsWith('//') || path.includes('?') || path.includes('#')) {
    reasons.push(`${field} must be an absolute path without query or fragment`)
    return ''
  }
  return path
}

function requiredString(value: unknown, field: string, reasons: string[]): string {
  const parsed = nonEmptyString(value)
  if (!parsed) reasons.push(`${field} must be a non-empty string`)
  return parsed ?? ''
}

function optionalCapabilityString(value: unknown): string | null {
  return nonEmptyString(value)
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function isUuid(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
}

function resolveOptionalPath(value: unknown, baseDirectory: string): string | null {
  const parsed = nonEmptyString(value)
  if (!parsed) return null
  return isAbsolute(parsed) ? parsed : resolve(baseDirectory, parsed)
}

function normalizeHttpUrl(value: string): string {
  const url = new URL(value)
  if (!url.pathname.endsWith('/')) url.pathname = `${url.pathname}/`
  return url.toString()
}

function isHttpUrl(value: string): boolean {
  try {
    return ['http:', 'https:'].includes(new URL(value).protocol)
  } catch {
    return false
  }
}

function hasUrlSecrets(value: string): boolean {
  try {
    const url = new URL(value)
    return Boolean(url.username || url.password || url.search || url.hash)
  } catch {
    return true
  }
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)]
}

export const collaborationE2EConfiguration = loadCollaborationE2EConfiguration()
assertReleaseE2EConfiguration(collaborationE2EConfiguration)
