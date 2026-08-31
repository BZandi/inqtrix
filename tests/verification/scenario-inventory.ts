import type {
  VerificationEngine,
  VerificationProfile,
} from './model.ts'

export const MOBILE_SCENARIO_TAG = '@mobile'
export const MOBILE_ONLY_SCENARIO_TAG = '@mobile-only'
export const CHROMIUM_ONLY_SCENARIO_TAG = '@chromium-only'

export const COLLABORATION_BROWSER_TARGETS = [
  { browser: 'chromium', formFactor: 'desktop' },
  { browser: 'chromium', formFactor: 'mobile' },
  { browser: 'firefox', formFactor: 'desktop' },
  { browser: 'webkit', formFactor: 'desktop' },
] as const

export type CollaborationBrowserTarget =
  typeof COLLABORATION_BROWSER_TARGETS[number]

export type ScenarioDefinition = {
  browsers?: readonly CollaborationBrowserTarget['browser'][]
  cleanup: string
  destructive: boolean
  engine: VerificationEngine
  formFactors?: readonly ('desktop' | 'mobile')[]
  id: string
  prerequisites: readonly string[]
  profiles: readonly VerificationProfile[]
  selectorTag?: `@${string}`
  testTitle?: string
  title: string
}

export const PROFILE_ENGINE_ORDER: Readonly<
  Record<VerificationProfile, readonly VerificationEngine[]>
> = {
  'agent-desk': ['agent-desk-live'],
  'chat-prompt': ['chat-prompt-live'],
  'edge-conformance': ['web-edge-containers'],
  'fault-injection': ['collaboration-playwright'],
  'load-capacity': ['collaboration-load'],
  'load-ramp': ['collaboration-load'],
  'load-smoke': ['collaboration-load'],
  'load-soak': ['collaboration-load'],
  'owner-setup': ['owner-setup-live'],
  'system-smoke': ['collaboration-playwright', 'editor-system-live'],
  'ui-fixture': ['ui-fixture-playwright'],
}

export const PROFILE_DESCRIPTIONS: Readonly<Record<VerificationProfile, string>> = {
  'agent-desk': (
    'Kernel answer experience through the visible Agent Desk against '
    + 'the live stack: run submission, tool activity, clickable '
    + 'citations, console/network hygiene, and cleanup integrity.'
  ),
  'chat-prompt': (
    'Chat and Prompt Library browser matrix against the live stack: '
    + 'real completions, reload ordering, duplicated tabs, offline '
    + 'recovery, and library CRUD in Chromium, Firefox, and WebKit.'
  ),
  'edge-conformance': 'Black-box parity of the packaged Python and nginx web edges.',
  'fault-injection': 'Controlled collaboration failure and recovery scenarios.',
  'load-capacity': 'Fixed high-capacity collaboration workload and resilience budgets.',
  'load-ramp': (
    'Local, resource-gated socket fan-out ladder (20/100/250/500/1000) '
    + 'from a capped identity pool. A controlled stop reports the highest '
    + 'reached rung; it never discharges the production capacity gate.'
  ),
  'load-smoke': (
    'Run-ID-bound 20-socket collaboration protocol, visibility, and '
    + 'durability workload with automatic temporary-resource provisioning.'
  ),
  'load-soak': (
    'Thirty-minute, 25-identity mixed collaboration soak with phased '
    + 'network shaping, real product APIs, and resource recovery budgets.'
  ),
  'owner-setup': (
    'One visible first-run local-owner session through setup, initial '
    + 'persistence, user administration, and pointer logout on a fresh stack.'
  ),
  'system-smoke': 'External multiuser system behavior through visible browser flows.',
  'ui-fixture': 'Deterministic browser fixtures without external product services.',
}

export const SCENARIO_INVENTORY: readonly ScenarioDefinition[] = [
  uiScenario(
    'ui.lifecycle-isolation',
    'Document rerender isolation',
    'A to B rerender never exposes A through B surface or controller registration',
  ),
  uiScenario(
    'ui.modal-focus-race',
    'Modal launcher teardown focus',
    'modal focus wins the launcher teardown race and restores on Escape',
  ),
  uiScenario(
    'ui.editor-share-focus',
    'Editor share dialog focus handoff',
    'editor share menu returns focus after Escape and click-outside',
  ),
  uiScenario(
    'ui.explorer-action-tooltip',
    'Explorer action tooltip handoff',
    'explorer action tooltip yields while its modal owns interaction',
  ),
  uiScenario(
    'ui.mobile-collaboration-presence',
    'Bounded mobile collaboration identity labels',
    'long collaboration labels stay inside the mobile editor at both edges',
  ),
  uiScenario(
    'ui.profile-menu-tooltip',
    'Profile menu and tooltip ownership',
    'profile menu suppresses its trigger tooltip while actions are open',
  ),
  uiScenario(
    'ui.activation-hydration',
    'Activation and exact hydration ordering',
    'activation success closes the writable body window before delayed exact hydration',
  ),
  uiScenario(
    'ui.narrow-topbar',
    'Narrow topbar bounded layout',
    'narrow topbar keeps long title and four-participant status in separate bounded tracks',
  ),
  uiScenario(
    'ui.save-status',
    'Stable save-status acknowledgement',
    'save status suppresses acknowledgement flicker and keeps its label track stable',
  ),
  uiScenario(
    'ui.initial-suggest-policy',
    'Initial Suggest transaction policy',
    'initial and same-event Suggest policies govern the first document transaction',
  ),
  uiScenario(
    'ui.suggestion-undo',
    'Rejected Suggest undo retry',
    'rejected Suggest undo emits no Yjs mutation and remains retryable',
  ),
  uiScenario(
    'ui.simple-markup',
    'Simple Markup compound suggestions',
    'Simple Markup reveals every active compound suggestion and hides other deletions',
  ),
  uiScenario(
    'ui.slash-heading',
    'Schema-valid slash heading',
    'slash heading creates a server-valid collaboration document',
  ),
  uiScenario(
    'ui.slash-matrix',
    'Schema-valid slash command matrix',
    'all eleven slash actions keep the collaboration document schema-valid',
  ),
  uiScenario(
    'ui.slash-keyboard',
    'Slash search and keyboard behavior',
    'slash search, keyboard selection, escape, and suggest guards are coherent',
  ),
  uiScenario(
    'ui.slash-reversible',
    'Reversible slash suggestion updates',
    'all reversible slash suggestions emit complete server-valid Yjs V1 updates',
  ),
  uiScenario(
    'ui.comment-density',
    'Sixty-thread inspector density',
    '60-thread inspector stays compact, progressive, and single-composer',
  ),
  uiScenario(
    'ui.view-only-assistant',
    'View-only assistant restriction',
    'view-only assistant controls explain the restriction and never dispatch work',
  ),
  uiScenario(
    'ui.mobile-file-upload-action',
    'Mobile file-library upload action',
    'mobile file library keeps its primary upload action fully visible',
  ),
  uiScenario(
    'ui.accessibility-demo',
    'Automated WCAG A/AA scan across demo workspaces',
    'demo workspaces have no automated WCAG A or AA violations',
  ),
  uiScenario(
    'ui.chat-empty-session',
    'Localized empty chat creation',
    'new chats start as localized empty sessions in both UI languages',
  ),
  {
    cleanup: 'The externally prepared run-scoped stack and its volumes are removed after the browser report is retained.',
    destructive: true,
    engine: 'owner-setup-live',
    id: 'auth.owner-setup-visible',
    prerequisites: [
      'Fresh loopback-bound local-auth stack with no owner',
      'One explicit Playwright browser target',
      'Protected synthetic owner credentials',
    ],
    profiles: ['owner-setup'],
    title: 'Visible owner setup creates an authenticated administrator session',
  },
  {
    cleanup: 'Covered by removal of the externally prepared run-scoped stack and volumes.',
    destructive: true,
    engine: 'owner-setup-live',
    id: 'auth.initial-server-mutation',
    prerequisites: ['The newly created owner session without a page reload'],
    profiles: ['owner-setup'],
    title: 'The first project mutation persists and the owning sync badge confirms success',
  },
  {
    cleanup: 'The synthetic account is removed with the externally prepared run-scoped stack and volumes.',
    destructive: true,
    engine: 'owner-setup-live',
    id: 'auth.admin-user-create',
    prerequisites: [
      'The authenticated owner session',
      'Protected credentials for a distinct synthetic user',
    ],
    profiles: ['owner-setup'],
    title: 'The owner creates a distinct local user through the visible administration UI',
  },
  {
    cleanup: 'Logout destroys the only browser session; the external stack cleanup removes retained audit data.',
    destructive: true,
    engine: 'owner-setup-live',
    id: 'auth.pointer-logout',
    prerequisites: ['The authenticated owner profile menu'],
    profiles: ['owner-setup'],
    title: 'A pointer-driven profile logout destroys exactly one server session',
  },
  {
    cleanup: 'Observational scenario with no resources of its own.',
    destructive: false,
    engine: 'owner-setup-live',
    id: 'auth.console-network-clean',
    prerequisites: ['The complete owner-setup browser flow'],
    profiles: ['owner-setup'],
    title: 'The first-run flow has no unexplained console errors, failed requests, CSRF rejection, or server error',
  },
  collaborationScenario(
    'system.transport-fingerprint',
    'Observable transport identity',
    '@transport-fingerprint',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.direct-edit',
    'Direct edit visibility and durable persistence',
    '@direct-edit',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.large-state-latency',
    'Large collaboration state remains responsive under concurrent visible edits',
    '@large-state-latency',
    ['system-smoke'],
    ['desktop'],
    false,
    ['chromium'],
  ),
  collaborationScenario(
    'system.remote-caret',
    'Remote caret identity at the author position',
    '@remote-caret',
    ['system-smoke'],
    ['desktop'],
  ),
  collaborationScenario(
    'system.remote-selection',
    'Remote selection presentation',
    '@remote-selection',
    ['system-smoke'],
    ['desktop'],
  ),
  collaborationScenario(
    'system.concurrent-edits',
    'Concurrent edits converge exactly once',
    '@concurrent-edits',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.ai-suggestion-accept',
    'An assistant suggestion publishes and reaches the second session',
    '@ai-suggestion-accept',
    ['system-smoke'],
    ['desktop'],
  ),
  collaborationScenario(
    'system.collaboration-stays-connected',
    'Two writing sessions stay connected without a reconnect banner',
    '@stays-connected',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.suggestions',
    'Suggestion acceptance and rejection',
    '@suggestions',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.suggestion-undo',
    'Suggestion undo remains connected and durable',
    '@suggestion-undo',
    ['system-smoke'],
    ['desktop'],
  ),
  collaborationScenario(
    'system.ime',
    'Genuine browser IME collaboration',
    '@ime',
    ['system-smoke'],
    ['desktop', 'mobile'],
    false,
    ['chromium'],
  ),
  collaborationScenario(
    'system.source-readonly',
    'Read-only Source projection',
    '@source-readonly',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.layout',
    'Bounded editor and Inspector layout',
    '@layout',
    ['system-smoke'],
  ),
  collaborationScenario(
    'system.mobile-drawers',
    'Exclusive mobile tree and Inspector drawers',
    '@mobile-drawer',
    ['system-smoke'],
    ['mobile'],
  ),
  {
    cleanup: 'The engine deletes run-prefixed documents, closes contexts, and disables temporary users in finally.',
    destructive: true,
    engine: 'editor-system-live',
    id: 'system.multiuser-live-matrix',
    prerequisites: [
      'External product stack',
      'Two authenticated seed accounts',
      'Admin user-management permission',
    ],
    profiles: ['system-smoke'],
    title: 'Six-user document, permission, comment, and responsive-layout matrix',
  },
  {
    cleanup: 'The run-question-bound agent session and its runs are registered on creation and deleted through owner APIs.',
    destructive: true,
    engine: 'agent-desk-live',
    id: 'agent.kernel-run-submits',
    prerequisites: [
      'External product stack with the agent kernel enabled',
      'Authenticated tester seed account',
    ],
    profiles: ['agent-desk'],
    title: 'A composer submission creates a session and a running kernel run',
  },
  {
    cleanup: 'Covered by the shared agent session/run cleanup registration.',
    destructive: true,
    engine: 'agent-desk-live',
    id: 'agent.tool-activity-visible',
    prerequisites: ['A running kernel run that performs a source-tool call'],
    profiles: ['agent-desk'],
    title: 'Tool activity appears live with the literal query and settles',
  },
  {
    cleanup: 'Covered by the shared agent session/run cleanup registration.',
    destructive: true,
    engine: 'agent-desk-live',
    id: 'agent.answer-citations-clickable',
    prerequisites: ['A completed kernel answer with evidence references'],
    profiles: ['agent-desk'],
    title: 'The chat answer renders citation chips and a source list that open',
  },
  {
    cleanup: 'No resources of its own; observational over the shared run.',
    destructive: false,
    engine: 'agent-desk-live',
    id: 'agent.console-network-clean',
    prerequisites: ['The shared agent-desk run'],
    profiles: ['agent-desk'],
    title: 'Zero console errors, failed requests, and 5xx responses across the flow',
  },
  {
    cleanup: 'Asserts the terminal run state and complete cleanup registration itself.',
    destructive: false,
    engine: 'agent-desk-live',
    id: 'agent.cleanup-integrity',
    prerequisites: ['The shared agent-desk run'],
    profiles: ['agent-desk'],
    title: 'Every created agent resource is registered and terminal before exit',
  },
  {
    cleanup: 'The run-titled thread is registered on creation and deleted through the visible UI; the ledger pass is 404-tolerant.',
    destructive: true,
    engine: 'chat-prompt-live',
    id: 'chat.live-turn-roundtrip',
    prerequisites: [
      'External product stack',
      'Authenticated tester seed account',
    ],
    profiles: ['chat-prompt'],
    title: 'A visible chat question produces a persisted assistant answer in Chromium, Firefox, and WebKit',
  },
  {
    cleanup: 'Covered by the shared chat-thread cleanup registration.',
    destructive: false,
    engine: 'chat-prompt-live',
    id: 'chat.reload-turn-order',
    prerequisites: ['The shared per-browser chat thread'],
    profiles: ['chat-prompt'],
    title: 'A real reload preserves the user-before-assistant turn order in every browser',
  },
  {
    cleanup: 'Covered by the shared chat-thread cleanup registration.',
    destructive: false,
    engine: 'chat-prompt-live',
    id: 'chat.duplicate-tab-consistency',
    prerequisites: ['The shared per-browser chat thread'],
    profiles: ['chat-prompt'],
    title: 'A duplicated tab reads the same thread consistently without duplicate sends',
  },
  {
    cleanup: 'Covered by the shared chat-thread cleanup registration.',
    destructive: true,
    engine: 'chat-prompt-live',
    id: 'chat.network-recovery',
    prerequisites: ['The shared per-browser chat thread'],
    profiles: ['chat-prompt'],
    title: 'An offline send fails visibly without losing input and completes after recovery',
  },
  {
    cleanup: 'Asserts the visible thread deletion and zero server residue itself.',
    destructive: true,
    engine: 'chat-prompt-live',
    id: 'chat.cleanup-integrity',
    prerequisites: ['The shared per-browser chat thread'],
    profiles: ['chat-prompt'],
    title: 'The visible thread deletion leaves zero run-bound chat residue per browser',
  },
  {
    cleanup: 'The run-titled template is registered on creation and deleted through the visible UI; the ledger pass is 404-tolerant.',
    destructive: true,
    engine: 'chat-prompt-live',
    id: 'prompt.live-owner-crud',
    prerequisites: [
      'External product stack',
      'Authenticated tester seed account',
    ],
    profiles: ['chat-prompt'],
    title: 'Prompt-template create, edit, revision, reload, and search work in Chromium, Firefox, and WebKit',
  },
  {
    cleanup: 'Covered by the shared prompt-template cleanup registration.',
    destructive: true,
    engine: 'chat-prompt-live',
    id: 'prompt.live-mobile-viewport',
    prerequisites: ['The Chromium chat-prompt session'],
    profiles: ['chat-prompt'],
    title: 'The live mobile viewport keeps template creation and deletion reachable',
  },
  {
    cleanup: 'Asserts the visible template deletion and zero server residue itself.',
    destructive: true,
    engine: 'chat-prompt-live',
    id: 'prompt.cleanup-integrity',
    prerequisites: ['The shared per-browser prompt template'],
    profiles: ['chat-prompt'],
    title: 'The visible template deletion leaves zero run-bound prompt residue per browser',
  },
  {
    cleanup: 'No resources of its own; observational over all chat-prompt contexts.',
    destructive: false,
    engine: 'chat-prompt-live',
    id: 'chatprompt.console-network-clean',
    prerequisites: ['The shared chat-prompt browser sessions'],
    profiles: ['chat-prompt'],
    title: 'Zero unexplained console errors, failed requests, and 5xx responses outside the offline window',
  },
  collaborationScenario(
    'fault.revocation',
    'Live access revocation and hidden 404',
    '@revocation',
    ['fault-injection'],
    undefined,
    true,
  ),
  collaborationScenario(
    'fault.permission-downgrade',
    'Live edit-to-view downgrade',
    '@permission-downgrade',
    ['fault-injection'],
    undefined,
    true,
  ),
  collaborationScenario(
    'fault.lost-ack',
    'Lost durable acknowledgement reconciliation',
    '@reconciliation',
    ['fault-injection'],
    undefined,
    true,
  ),
  collaborationScenario(
    'fault.sidecar-outage',
    'Collaboration-sidecar outage, restart, and recovery',
    '@outage',
    ['fault-injection'],
    undefined,
    true,
  ),
  collaborationScenario(
    'fault.gateway-outage',
    'Public gateway outage and recovery',
    '@gateway-outage',
    ['fault-injection'],
    undefined,
    true,
  ),
  collaborationScenario(
    'fault.private-anchors',
    'Private anchor isolation during remote edits',
    '@private-anchors',
    ['fault-injection'],
  ),
  collaborationScenario(
    'system.detached-transfer',
    'Detached export and import',
    '@detached-transfer',
    ['system-smoke'],
  ),
  collaborationScenario(
    'fault.protocol-rejection',
    'Incompatible protocol and schema rejection',
    '@protocol-rejection',
    ['fault-injection'],
    undefined,
    true,
  ),
  loadScenario(
    'load-smoke.protocol',
    'Small authenticated Yjs protocol cohort',
    'load-smoke',
  ),
  loadScenario(
    'load-smoke.durability',
    'Small visibility and durable-ack sample',
    'load-smoke',
  ),
  loadScenario(
    'load-smoke.reconstruction',
    'Small observer reconstruction check',
    'load-smoke',
  ),
  loadScenario(
    'load-soak.identity-matrix',
    'Twenty-five distinct identities and ordered role cohorts',
    'load-soak',
  ),
  loadScenario(
    'load-soak.comments-and-navigation',
    'Fifty comment threads and repeated reader navigation',
    'load-soak',
  ),
  loadScenario(
    'load-soak.network-phases',
    'Normal, latency, bandwidth, packet-loss, and normalization phases',
    'load-soak',
    true,
  ),
  loadScenario(
    'load-soak.durability',
    'Phase-anchored visibility, durable acknowledgement, and reconstruction',
    'load-soak',
  ),
  loadScenario(
    'load-soak.feature-activity',
    'Staggered cost-bounded Research, Knowledge, and Agent activity',
    'load-soak',
  ),
  loadScenario(
    'load-soak.resource-recovery',
    'Sockets, memory, CPU, pools, and final recovery remain bounded',
    'load-soak',
  ),
  loadScenario(
    'load-ramp.ladder',
    'Resource-gated socket fan-out ladder with an honest controlled stop',
    'load-ramp',
  ),
  loadScenario(
    'load-ramp.integrity',
    'Exact reconstruction and zero lost or duplicate mutations at every reached rung',
    'load-ramp',
  ),
  loadScenario(
    'load-capacity.latency',
    'Fixed visible-update, durable-ack, and API latency budgets',
    'load-capacity',
  ),
  loadScenario(
    'load-capacity.rotation',
    'Connected and scheduled lease rotation',
    'load-capacity',
  ),
  loadScenario(
    'load-capacity.restart',
    'Ungraceful restart and observer reconstruction',
    'load-capacity',
    true,
  ),
  edgeScenario(
    'edge.static-spa-cache',
    'Static assets, hard asset 404s, and SPA cache behavior',
  ),
  edgeScenario(
    'edge.readiness-contract',
    'Ready and degraded dependency status relay across edge adapters',
  ),
  edgeScenario(
    'edge.http-streaming-cookies',
    'SSE streaming and duplicate Set-Cookie relay',
  ),
  edgeScenario(
    'edge.hop-by-hop-headers',
    'Request and response hop-by-hop header removal',
  ),
  edgeScenario(
    'edge.request-body-limit',
    'Header-independent request-body ceiling',
  ),
  edgeScenario(
    'edge.websocket-contract',
    'Binary WebSocket query and close behavior',
  ),
  edgeScenario(
    'edge.backend-recovery',
    'Backend outage and same-process recovery',
  ),
  edgeScenario(
    'edge.guest-security-and-redaction',
    'Normalized guest/share-link privacy and log redaction',
  ),
  edgeScenario(
    'edge.runtime-hardening',
    'Non-root, read-only, capability-dropped runtime',
  ),
] as const

validateScenarioInventory(SCENARIO_INVENTORY)

export function scenariosForProfile(
  profile: VerificationProfile,
): readonly ScenarioDefinition[] {
  return SCENARIO_INVENTORY.filter((scenario) => scenario.profiles.includes(profile))
}

export function scenarioIdsForProfile(profile: VerificationProfile): string[] {
  return scenariosForProfile(profile).map((scenario) => scenario.id)
}

export function uiScenarioForTestTitle(
  title: string,
): ScenarioDefinition | undefined {
  return SCENARIO_INVENTORY.find((scenario) => (
    scenario.engine === 'ui-fixture-playwright'
    && scenario.testTitle === title
  ))
}

export function collaborationScenarioForTags(
  profile: VerificationProfile,
  tags: readonly string[],
): ScenarioDefinition | undefined {
  return scenariosForProfile(profile).find((scenario) => (
    scenario.engine === 'collaboration-playwright'
    && scenario.selectorTag !== undefined
    && tags.includes(scenario.selectorTag)
  ))
}

export function requiredPlaywrightTags(
  profile: VerificationProfile,
  formFactor: 'desktop' | 'mobile',
  browser: CollaborationBrowserTarget['browser'] = 'chromium',
): string[] {
  return scenariosForProfile(profile)
    .filter((scenario) => (
      scenario.engine === 'collaboration-playwright'
      && scenario.selectorTag
      && (!scenario.formFactors || scenario.formFactors.includes(formFactor))
      && (!scenario.browsers || scenario.browsers.includes(browser))
    ))
    .map((scenario) => scenario.selectorTag!)
}

export function playwrightGrep(profile: VerificationProfile): string {
  const tags = new Set([
    ...requiredPlaywrightTags(profile, 'desktop'),
    ...requiredPlaywrightTags(profile, 'mobile'),
  ])
  if (tags.size === 0) {
    throw new Error(`Profile ${profile} has no collaboration Playwright scenarios.`)
  }
  return [...tags].join('|')
}

function uiScenario(
  id: string,
  title: string,
  testTitle: string,
): ScenarioDefinition {
  return {
    cleanup: 'Playwright closes the isolated browser context and fixture server.',
    destructive: false,
    engine: 'ui-fixture-playwright',
    id,
    prerequisites: ['Local browser runtime'],
    profiles: ['ui-fixture'],
    testTitle,
    title,
  }
}

function edgeScenario(
  id: string,
  title: string,
): ScenarioDefinition {
  return {
    cleanup: 'The orchestrator removes exact run-labelled containers, network, and image tags, then verifies zero residual resources.',
    destructive: false,
    engine: 'web-edge-containers',
    id,
    prerequisites: [
      'Explicit Podman or Docker engine',
      'Local image build capability',
    ],
    profiles: ['edge-conformance'],
    title,
  }
}

function collaborationScenario(
  id: string,
  title: string,
  selectorTag: `@${string}`,
  profiles: readonly VerificationProfile[],
  formFactors: readonly ('desktop' | 'mobile')[] = ['desktop', 'mobile'],
  destructive = false,
  browsers?: readonly CollaborationBrowserTarget['browser'][],
): ScenarioDefinition {
  return {
    browsers,
    cleanup: destructive
      ? 'External fixture controls restore the affected capability; disposable documents are removed by fixture ownership.'
      : 'Disposable fixture documents remain externally owned and are cleaned by the fixture lifecycle.',
    destructive,
    engine: 'collaboration-playwright',
    formFactors,
    id,
    prerequisites: ['Version-2 collaboration fixture', 'Isolated owner and collaborator identities'],
    profiles,
    selectorTag,
    title,
  }
}

function loadScenario(
  id: string,
  title: string,
  profile: 'load-smoke' | 'load-soak' | 'load-ramp' | 'load-capacity',
  destructive = false,
): ScenarioDefinition {
  return {
    cleanup: 'The load engine closes every socket and stops the lease-rotation supervisor in finally.',
    destructive,
    engine: 'collaboration-load',
    id,
    prerequisites: ['Version-2 lease/session fixture'],
    profiles: [profile],
    title,
  }
}

function validateScenarioInventory(inventory: readonly ScenarioDefinition[]): void {
  const ids = new Set<string>()
  for (const scenario of inventory) {
    if (ids.has(scenario.id)) throw new Error(`Duplicate verification scenario: ${scenario.id}`)
    ids.add(scenario.id)
    if (scenario.profiles.length === 0) {
      throw new Error(`Verification scenario ${scenario.id} has no profile.`)
    }
    for (const profile of scenario.profiles) {
      if (!PROFILE_ENGINE_ORDER[profile].includes(scenario.engine)) {
        throw new Error(
          `Verification scenario ${scenario.id} uses engine ${scenario.engine} outside profile ${profile}.`,
        )
      }
    }
  }
  for (const [profile, engines] of Object.entries(PROFILE_ENGINE_ORDER) as Array<
    [VerificationProfile, readonly VerificationEngine[]]
  >) {
    const selected = inventory.filter((scenario) => scenario.profiles.includes(profile))
    if (selected.length === 0) throw new Error(`Verification profile ${profile} has no scenarios.`)
    for (const engine of engines) {
      if (!selected.some((scenario) => scenario.engine === engine)) {
        throw new Error(`Verification profile ${profile} has no scenario for engine ${engine}.`)
      }
    }
  }
}
