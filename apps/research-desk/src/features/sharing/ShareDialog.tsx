import { Search, X } from '@/components/icons'
import { useEffect, useMemo, useRef, useState } from 'react'
import { searchUsers } from '@/api/inqtrixClient'
import { InitialsAvatar } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { searchDemoUsers } from './demoShares'
import {
  personLabel,
  selectableSearchResults,
  sharePermissionLabel,
  sharePermissionsForResource,
  toggleSelectedUser,
} from './shareModel'
import type { SharePermissionValue, UserSearchResult } from './types'
import { useShares } from './useShares'

const SEARCH_DEBOUNCE_MS = 250
const MIN_QUERY_LENGTH = 2

type ShareDialogProps = {
  /** Demo mode resolves users + shares from seeded data (no backend). */
  demo?: boolean
  onChanged?: () => void
  onClose: () => void
  ownerEmail: string | null
  ownerName: string | null
  refreshToken: number
  resourceId: string
  resourceTitle: string
  resourceType: string
}

/**
 * Google-Drive-style share dialog: typeahead over the user mirror,
 * batch selection as chips with ONE permission picker, and the live
 * access list below (owner pinned, revoke hover-revealed). Server is
 * the only source of truth — every mutation re-reads the listing.
 */
export function ShareDialog({
  demo = false,
  onChanged,
  onClose,
  ownerEmail,
  ownerName,
  refreshToken,
  resourceId,
  resourceTitle,
  resourceType,
}: ShareDialogProps) {
  const { locale, t } = useLocale()
  const { grant, reload, revoke, state, updatePermission } = useShares(
    resourceType,
    resourceId,
    demo,
    refreshToken,
  )
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<readonly UserSearchResult[]>([])
  const [searchError, setSearchError] = useState(false)
  const [searchPending, setSearchPending] = useState(false)
  const [selected, setSelected] = useState<readonly UserSearchResult[]>([])
  const [permission, setPermission] = useState<SharePermissionValue>('view')
  const [submitState, setSubmitState] = useState<'error-grant' | 'error-revoke' | 'error-update' | 'idle' | 'submitting'>('idle')
  const [updatingShareId, setUpdatingShareId] = useState<string | null>(null)
  const permissionOptions = sharePermissionsForResource(resourceType)
  const permissionLabels = {
    edit: t.sharing.permissionEdit,
    view: t.sharing.permissionView,
  }

  useEffect(() => {
    if (!permissionOptions.includes(permission)) setPermission('view')
  }, [permission, permissionOptions])

  const trimmedQuery = query.trim()
  // Generation guard (same pattern as useShares): a slow response for
  // an EARLIER query must never overwrite the current query's results.
  const searchGenerationRef = useRef(0)
  useEffect(() => {
    const generation = ++searchGenerationRef.current
    if (trimmedQuery.length < MIN_QUERY_LENGTH) {
      setResults([])
      setSearchError(false)
      setSearchPending(false)
      return undefined
    }
    setSearchError(false)
    setSearchPending(true)
    const lookup = demo
      ? (value: string) => Promise.resolve(searchDemoUsers(value))
      : searchUsers
    const timeout = window.setTimeout(() => {
      lookup(trimmedQuery)
        .then((found) => {
          if (searchGenerationRef.current !== generation) return
          setResults(found)
          setSearchError(false)
          setSearchPending(false)
        })
        .catch(() => {
          if (searchGenerationRef.current !== generation) return
          setResults([])
          setSearchError(true)
          setSearchPending(false)
        })
    }, SEARCH_DEBOUNCE_MS)
    return () => window.clearTimeout(timeout)
  }, [demo, trimmedQuery])

  const existingUserIds = useMemo(() => {
    const userIds = new Set(state.records.map((record) => record.recipient_user_id))
    for (const user of selected) userIds.add(user.id)
    return userIds
  }, [selected, state.records])
  const visibleResults = selectableSearchResults(results, existingUserIds)

  const submitGrant = async () => {
    if (selected.length === 0 || submitState === 'submitting') return
    setSubmitState('submitting')
    try {
      await grant(selected.map((user) => ({ permission, userId: user.id })))
      setSelected([])
      setQuery('')
      setResults([])
      setSubmitState('idle')
      onChanged?.()
    } catch {
      setSubmitState('error-grant')
    }
  }

  const submitRevoke = async (shareId: string) => {
    try {
      await revoke(shareId)
      setSubmitState('idle')
      onChanged?.()
    } catch {
      setSubmitState('error-revoke')
    }
  }

  const submitPermissionUpdate = async (
    shareId: string,
    nextPermission: SharePermissionValue,
    expectedRevision: number,
  ) => {
    setUpdatingShareId(shareId)
    try {
      await updatePermission(shareId, nextPermission, expectedRevision)
      setSubmitState('idle')
      onChanged?.()
    } catch {
      await reload()
      setSubmitState('error-update')
    } finally {
      setUpdatingShareId(null)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-background/75 px-4 py-8 backdrop-blur"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section
        aria-modal="true"
        className="w-full max-w-lg overflow-hidden rounded-lg border border-border bg-background shadow-xl"
        role="dialog"
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h2 className="t-section truncate text-foreground">
              {t.sharing.dialogTitle}
              {resourceTitle ? ` · ${resourceTitle}` : ''}
            </h2>
            <p className="t-meta text-muted-foreground">{t.sharing.dialogHint}</p>
          </div>
          <button
            aria-label={t.sharing.close}
            className="grid size-7 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="space-y-3 px-4 py-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              autoFocus
              className="h-8 w-full rounded-md border border-border bg-background pl-8 pr-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus-visible:border-brand"
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t.sharing.searchPlaceholder}
              type="text"
              value={query}
            />
          </div>

          {trimmedQuery.length > 0 && trimmedQuery.length < MIN_QUERY_LENGTH && (
            <p className="t-meta text-muted-foreground">
              {t.sharing.searchMinChars}
            </p>
          )}
          {trimmedQuery.length >= MIN_QUERY_LENGTH && (
            <div className="overflow-hidden rounded-md border border-border">
              {visibleResults.length === 0 ? (
                <p className={cn(
                  'px-3 py-2 t-meta',
                  searchError ? 'text-destructive' : 'text-muted-foreground',
                )}>
                  {searchPending
                    ? '…'
                    : searchError
                      ? t.sharing.searchFailed
                      : t.sharing.searchEmpty}
                </p>
              ) : (
                <ul className="max-h-44 overflow-y-auto">
                  {visibleResults.map((user) => (
                    <li key={user.id}>
                      <button
                        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left hover:bg-accent"
                        onMouseDown={(event) => {
                          event.preventDefault()
                          setSelected((current) => toggleSelectedUser(current, user))
                          setQuery('')
                          setResults([])
                          setSearchError(false)
                        }}
                        type="button"
                      >
                        <InitialsAvatar
                          displayName={user.display_name}
                          email={user.email}
                          size="sm"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate t-list text-foreground">
                          {personLabel(user.display_name, user.email, user.id)}
                          </span>
                          {user.email && (
                            <span className="block truncate t-meta-sm text-muted-foreground">
                              {user.email}
                            </span>
                          )}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {selected.length > 0 && (
            <div className="space-y-2.5">
              <div className="flex flex-wrap items-center gap-1.5">
                {selected.map((user) => (
                  <span
                    className="inline-flex h-6 items-center gap-1.5 rounded-full bg-brand-subtle pl-1 pr-1.5 text-brand"
                    key={user.id}
                  >
                    <InitialsAvatar
                      displayName={user.display_name}
                      email={user.email}
                      size="sm"
                    />
                    <span className="max-w-40 truncate t-meta-sm font-medium">
                      {personLabel(user.display_name, user.email, user.id)}
                    </span>
                    <button
                      aria-label={t.sharing.removeSelected}
                      className="grid size-3.5 place-items-center rounded-full hover:bg-brand/15"
                      onClick={() => setSelected((current) => toggleSelectedUser(current, user))}
                      type="button"
                    >
                      <X className="size-3" />
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex items-center justify-between gap-3">
                <div
                  aria-label={t.sharing.permissionLabel}
                  className="flex h-7 items-center gap-0.5 rounded-md bg-surface p-0.5"
                  role="group"
                >
                  {permissionOptions.map((value) => (
                    <button
                      aria-pressed={permission === value}
                      className={cn(
                        'h-6 rounded px-2 text-xs font-medium transition-colors',
                        permission === value
                          ? 'bg-background text-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground',
                      )}
                      key={value}
                      onClick={() => setPermission(value)}
                      type="button"
                    >
                      {sharePermissionLabel(value, locale, permissionLabels)}
                    </button>
                  ))}
                </div>
                <Button
                  className="h-8 bg-brand px-3 text-xs text-brand-foreground hover:bg-brand/90"
                  disabled={submitState === 'submitting'}
                  onClick={() => void submitGrant()}
                  size="sm"
                  type="button"
                >
                  {submitState === 'submitting' ? t.sharing.granting : t.sharing.grantCta}
                </Button>
              </div>
            </div>
          )}

          {submitState === 'error-grant' && (
            <p className="t-meta text-destructive">{t.sharing.shareFailed}</p>
          )}
          {submitState === 'error-revoke' && (
            <p className="t-meta text-destructive">{t.sharing.revokeFailed}</p>
          )}
          {submitState === 'error-update' && (
            <p className="t-meta text-destructive">{t.sharing.updateFailed}</p>
          )}

          <div>
            <h3 className="t-caption text-muted-foreground">
              {t.sharing.peopleAndInvites}
            </h3>
            <ul className="mt-1.5 space-y-0.5">
              <li className="flex items-center gap-2 rounded-md px-1.5 py-1.5">
                <InitialsAvatar displayName={ownerName} email={ownerEmail} size="sm" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate t-list text-foreground">
                    {personLabel(ownerName, ownerEmail, t.sharing.you)}
                  </span>
                  {ownerEmail && (
                    <span className="block truncate t-meta-sm text-muted-foreground">
                      {ownerEmail}
                    </span>
                  )}
                </span>
                <span className="shrink-0 t-meta text-muted-foreground">
                  {t.sharing.owner}
                </span>
              </li>
              {state.status === 'error' ? (
                <li className="px-1.5 py-1.5 t-meta text-destructive">
                  {t.sharing.loadFailed}
                </li>
              ) : state.records.length === 0 && state.status === 'ready' ? (
                <li className="px-1.5 py-1.5 t-meta text-muted-foreground">
                  {t.sharing.noShares}
                </li>
              ) : (
                state.records.map((record) => (
                  <li
                    className="group flex items-center gap-2 rounded-md px-1.5 py-1.5 hover:bg-accent/50"
                    key={record.id}
                  >
                    <InitialsAvatar
                      displayName={record.display_name}
                      email={record.email}
                      size="sm"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate t-list text-foreground">
                        {personLabel(record.display_name, record.email, record.recipient_user_id)}
                      </span>
                      {record.email && (
                        <span className="block truncate t-meta-sm text-muted-foreground">
                          {record.email}
                        </span>
                      )}
                    </span>
                    {record.accepted_at === null ? (
                      <span className="shrink-0 rounded-md border border-warning/25 bg-warning-subtle px-1.5 py-0.5 t-meta-sm font-medium text-warning">
                        {t.sharing.pending}
                      </span>
                    ) : null}
                    <div
                      aria-label={t.sharing.permissionLabel}
                      className="flex h-7 shrink-0 items-center gap-0.5 rounded-md bg-surface p-0.5"
                      role="group"
                    >
                      {permissionOptions.map((value) => (
                        <button
                          aria-pressed={record.permission === value}
                          className={cn(
                            'h-6 rounded px-2 text-xs font-medium transition-colors',
                            record.permission === value
                              ? 'bg-background text-foreground shadow-sm'
                              : 'text-muted-foreground hover:text-foreground',
                          )}
                          disabled={updatingShareId === record.id}
                          key={value}
                          onClick={() => {
                            if (record.permission === value) return
                            void submitPermissionUpdate(record.id, value, record.revision)
                          }}
                          type="button"
                        >
                          {sharePermissionLabel(value, locale, permissionLabels)}
                        </button>
                      ))}
                    </div>
                    <button
                      aria-label={t.sharing.revoke}
                      className="grid size-6 shrink-0 place-items-center rounded-md text-muted-foreground opacity-0 transition-opacity hover:bg-accent hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
                      onClick={() => void submitRevoke(record.id)}
                      title={t.sharing.revoke}
                      type="button"
                    >
                      <X className="size-3.5" />
                    </button>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
