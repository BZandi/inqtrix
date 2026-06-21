/**
 * Frontend auth-mode vocabulary, mirroring the server's `auth_mode`.
 *
 * Two derived predicates keep the `oidc || local || ldap` test from being
 * re-spelled across the component tree (Designprinzip #4 — define shared
 * logic once):
 *  - cookie-session modes share the BFF session/CSRF machinery, so the SPA
 *    probes `/api/auth/session` and gates sharing/quota/admin on a live
 *    session for all three (ADR-AUTH-3);
 *  - password modes (`local`/`ldap`) render the credential login form and
 *    drive the first-run owner setup gate.
 */
export type AuthMode = 'none' | 'apikey' | 'oidc' | 'local' | 'ldap'

/** Modes whose identity rides a server-side cookie session. */
export function isCookieSessionMode(mode: AuthMode | undefined | null): boolean {
  return mode === 'oidc' || mode === 'local' || mode === 'ldap'
}

/** Modes that authenticate via an identifier + password form. */
export function isPasswordMode(mode: AuthMode | undefined | null): boolean {
  return mode === 'local' || mode === 'ldap'
}
