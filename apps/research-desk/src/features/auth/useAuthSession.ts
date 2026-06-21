import { useCallback, useEffect, useState } from 'react'
import {
  buildLoginUrl,
  fetchAuthSession,
  logoutSession,
  type AuthSessionInfo,
} from '@/api/inqtrixClient'

export type AuthSessionState = {
  /** `unknown` until the first probe answers; the UI shows neither
   * login nor logout while unknown to avoid a button flash. */
  status: 'anonymous' | 'authenticated' | 'unknown'
  displayName: string | null
  email: string | null
  /** Instance role (`admin`/`user`) from the session payload; `null`
   * while anonymous/unknown. Drives the admin-surface gate. */
  role: string | null
  /** Stable subject of the signed-in identity (self-row detection). */
  sub: string | null
  /** The user's canonical project namespace (cross-device), resolved
   * server-side from the session; `null` while anonymous/unknown or before a
   * namespace has been adopted. The desk scopes the project to this (not the
   * browser-local id) when authenticated. */
  projectNamespace: string | null
}

const ANONYMOUS: AuthSessionState = {
  status: 'anonymous',
  displayName: null,
  email: null,
  role: null,
  sub: null,
  projectNamespace: null,
}

/**
 * Cookie-session state for the SPA. Probes `GET /api/auth/session` only
 * when the server reports a cookie-session mode (`oidc`/`local`/`ldap`)
 * — in `none`/`apikey` deployments (and demo mode) the hook stays inert
 * and reports `anonymous` without any network traffic. `login` is the
 * OIDC full-page redirect (local/ldap use the credential form, then
 * `refresh()`); `logout` destroys the server-side session and re-probes.
 */
export function useAuthSession(active: boolean, workspaceId?: string) {
  const [session, setSession] = useState<AuthSessionState>(
    active ? { ...ANONYMOUS, status: 'unknown' } : ANONYMOUS,
  )

  const refresh = useCallback(async () => {
    if (!active) return
    try {
      // Send the browser's namespace as the CANDIDATE: on a first authenticated
      // boot the server adopts it as the user's canonical project namespace and
      // returns it in `project_namespace`; thereafter it returns the already-
      // adopted value (the same on every device, so the data follows the user).
      const info: AuthSessionInfo = await fetchAuthSession({ workspaceId })
      setSession(
        info.authenticated
          ? {
              status: 'authenticated',
              displayName: info.display_name ?? null,
              email: info.email ?? null,
              role: info.role ?? null,
              sub: info.sub ?? null,
              projectNamespace: info.project_namespace ?? null,
            }
          : ANONYMOUS,
      )
    } catch {
      // Server unreachable: treat as anonymous; the existing health
      // banner already surfaces connectivity problems.
      setSession(ANONYMOUS)
    }
  }, [active, workspaceId])

  useEffect(() => {
    if (!active) {
      setSession(ANONYMOUS)
      return
    }
    // `active` (= cookie-session mode) usually flips true only AFTER health
    // resolves post-mount, so the mount-time initializer cannot seed
    // `unknown`. Seed it here, before the probe, so the desk withholds the
    // auth-lock form while the session is being checked (no lock flash for an
    // already-authenticated user reloading the page).
    setSession((current) =>
      current.status === 'authenticated'
        ? current
        : { ...ANONYMOUS, status: 'unknown' },
    )
    void refresh()
  }, [active, refresh])

  const login = useCallback(() => {
    window.location.assign(
      buildLoginUrl(undefined, window.location.pathname || '/'),
    )
  }, [])

  const logout = useCallback(async () => {
    try {
      await logoutSession()
    } finally {
      await refresh()
    }
  }, [refresh])

  return { session, login, logout, refresh }
}
