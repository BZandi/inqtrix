export const RELEASE_DESKTOP_SCENARIOS = [
  '@direct-edit',
  '@concurrent-edits',
  '@suggestions',
  '@ime',
  '@revocation',
  '@permission-downgrade',
  '@reconciliation',
  '@outage',
  '@gateway-outage',
  '@private-anchors',
  '@detached-transfer',
  '@protocol-rejection',
  '@source-readonly',
  '@transport-fingerprint',
  '@layout',
] as const

export const RELEASE_MOBILE_SCENARIOS = [
  ...RELEASE_DESKTOP_SCENARIOS,
  '@mobile-drawer',
] as const

export const MOBILE_SCENARIO_TAG = '@mobile'
export const MOBILE_ONLY_SCENARIO_TAG = '@mobile-only'
