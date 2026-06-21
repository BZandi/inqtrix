import { useCallback, useEffect, useRef, useState } from 'react'

import {
  type AdminUser,
  createAdminUser,
  listAdminUsers,
  resetUserPassword,
  setAdminUserDisabled,
  setAdminUserRole,
} from '@/api/inqtrixClient'
import { seedAdminUsers } from './demo'

type AdminUsersStatus = 'idle' | 'loading' | 'ready' | 'error'

export type AdminUsersState = {
  /** True when the admin user list is in scope (instance admin, or demo). */
  available: boolean
  demo: boolean
  users: AdminUser[]
  status: AdminUsersStatus
  error: string | null
  /** Last failed mutation; surfaced as a banner (No Silent Fallbacks). */
  mutationError: string | null
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Instance user administration for the Settings admin panel. Mirrors the
 * quota-admin hook shape: a generation-guarded reload, a single mutation
 * runner that re-reads the list on success AND failure (so the table never
 * shows stale rows after a partial write), and a demo branch that mutates
 * local state so the digital twin is fully interactive offline.
 */
export function useAdminUsers({
  demo,
  enabled,
}: {
  demo: boolean
  enabled: boolean
}) {
  const [state, setState] = useState<AdminUsersState>({
    available: false,
    demo,
    error: null,
    mutationError: null,
    status: 'idle',
    users: [],
  })
  const generationRef = useRef(0)

  const reload = useCallback(async () => {
    // Bump the generation on EVERY reload (incl. the demo/disabled early
    // returns), so a live fetch started before a demo toggle cannot resolve
    // and clobber the seeded/empty state with stale rows.
    const generation = ++generationRef.current
    if (!enabled) {
      setState({
        available: false,
        demo,
        error: null,
        mutationError: null,
        status: 'idle',
        users: [],
      })
      return
    }
    if (demo) {
      setState({
        available: true,
        demo,
        error: null,
        mutationError: null,
        status: 'ready',
        users: seedAdminUsers(),
      })
      return
    }
    setState((current) => ({ ...current, available: true, status: 'loading' }))
    try {
      const { users } = await listAdminUsers()
      if (generationRef.current !== generation) return
      setState({
        available: true,
        demo,
        error: null,
        mutationError: null,
        status: 'ready',
        users,
      })
    } catch (error) {
      if (generationRef.current !== generation) return
      setState((current) => ({
        ...current,
        available: true,
        error: messageOf(error),
        status: 'error',
        users: [],
      }))
    }
  }, [demo, enabled])

  useEffect(() => {
    void reload()
  }, [reload])

  const runMutation = useCallback(
    async (action: () => Promise<void>) => {
      setState((current) => ({ ...current, mutationError: null }))
      try {
        await action()
        await reload()
      } catch (error) {
        await reload()
        setState((current) => ({ ...current, mutationError: messageOf(error) }))
      }
    },
    [reload],
  )

  const setRole = useCallback(
    (subject: string, role: 'admin' | 'user') => {
      if (demo) {
        setState((current) => ({
          ...current,
          users: current.users.map((user) =>
            user.subject === subject ? { ...user, instance_role: role } : user,
          ),
        }))
        return Promise.resolve()
      }
      return runMutation(async () => {
        await setAdminUserRole(subject, role)
      })
    },
    [demo, runMutation],
  )

  const setDisabled = useCallback(
    (subject: string, disabled: boolean) => {
      if (demo) {
        setState((current) => ({
          ...current,
          users: current.users.map((user) =>
            user.subject === subject ? { ...user, disabled } : user,
          ),
        }))
        return Promise.resolve()
      }
      return runMutation(async () => {
        await setAdminUserDisabled(subject, disabled)
      })
    },
    [demo, runMutation],
  )

  /**
   * Create a local account. Unlike the role/disable mutations this REthrows
   * on failure so the dialog can show the inline error (e.g. 409 duplicate
   * email) and keep the entered values. The caller reveals the chosen
   * initial password once on success.
   */
  const createUser = useCallback(
    async (input: {
      email: string
      password: string
      instanceRole?: 'admin' | 'user'
      displayName?: string
    }) => {
      if (demo) {
        setState((current) => ({
          ...current,
          users: [
            ...current.users,
            {
              subject: `demo-${input.email}`,
              email: input.email,
              display_name: input.displayName || input.email,
              instance_role: input.instanceRole ?? 'user',
              disabled: false,
              last_login_at: null,
            },
          ],
        }))
        return
      }
      await createAdminUser(input)
      await reload()
    },
    [demo, reload],
  )

  /**
   * Admin sets a new password for a local account (forgotten-password
   * recovery). REthrows on failure so the dialog shows the inline error; the
   * caller reveals the chosen password once. Demo is a client-only no-op (the
   * revealed value is the password the admin typed).
   */
  const resetPassword = useCallback(
    async (subject: string, password: string) => {
      if (demo) return
      await resetUserPassword(subject, password)
    },
    [demo],
  )

  return { createUser, reload, resetPassword, setDisabled, setRole, state }
}
