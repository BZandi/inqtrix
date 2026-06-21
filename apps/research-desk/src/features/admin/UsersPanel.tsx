import { type FormEvent, useState } from 'react'

import { AlertTriangle, Check, Copy, KeyRound, Plus } from '@/components/icons'
import { type AdminUser, hasHttpStatus } from '@/api/inqtrixClient'
import { InitialsAvatar } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import {
  Table,
  TableBody,
  TableCell,
  TableEmpty,
  TableHead,
  TableHeader,
  TableRow,
  TableSkeleton,
} from '@/components/ui/table'
import type { AuthMode } from '@/features/auth/authMode'
import {
  MIN_PASSWORD_LENGTH,
  isPasswordAcceptable,
} from '@/features/auth/passwordPolicy'
import { StatusBadge } from '@/features/settings/parts'
import { useLocale } from '@/i18n/LocaleProvider'
import { canDisable, canSetRole, isSelf, sortUsers } from './adminModel'
import type { useAdminUsers } from './useAdminUsers'

function formatDate(seconds: number | null, locale: string): string | null {
  if (seconds == null) return null
  return new Date(seconds * 1000).toLocaleDateString(locale, {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  })
}

/** A cryptographically-random, policy-passing initial password suggestion. */
function suggestPassword(): string {
  const alphabet =
    'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789-_'
  const bytes = new Uint32Array(20)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (value) => alphabet[value % alphabet.length]).join('')
}

export function UsersPanel({
  admin,
  mode,
  sessionSub,
}: {
  admin: ReturnType<typeof useAdminUsers>
  mode: AuthMode
  sessionSub: string | null
}) {
  const { locale, t } = useLocale()
  const { createUser, resetPassword, setDisabled, setRole, state } = admin
  const [createOpen, setCreateOpen] = useState(false)
  const [resetTarget, setResetTarget] = useState<AdminUser | null>(null)
  const rows = sortUsers(state.users)

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <p className="t-meta max-w-xl text-muted-foreground">
          {t.adminUsers.description}
        </p>
        {mode === 'local' || state.demo ? (
          <Button
            className="shrink-0 gap-1.5 bg-brand text-brand-foreground hover:bg-brand/90"
            onClick={() => setCreateOpen(true)}
            size="sm"
          >
            <Plus className="size-4" />
            {t.adminUsers.createButton}
          </Button>
        ) : null}
      </div>

      {state.mutationError ? (
        <p className="flex gap-1.5 text-xs font-medium text-destructive" role="alert">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>{t.adminUsers.mutationFailed}</span>
        </p>
      ) : null}

      <Table variant="fluid">
        <TableHeader>
          <TableRow>
            <TableHead>{t.adminUsers.colUser}</TableHead>
            <TableHead>{t.adminUsers.colRole}</TableHead>
            <TableHead>{t.adminUsers.colStatus}</TableHead>
            <TableHead>{t.adminUsers.colLastLogin}</TableHead>
            <TableHead className="text-right">{t.adminUsers.colActions}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {state.status === 'loading' ? (
            <TableSkeleton colSpan={5} />
          ) : rows.length === 0 ? (
            <TableEmpty colSpan={5} title={t.adminUsers.empty} />
          ) : (
            rows.map((user) => {
              const self = isSelf(user, sessionSub)
              const roleGuardUser = canSetRole(rows, user, sessionSub, 'user')
              const disableGuard = canDisable(rows, user, sessionSub)
              const lockReason = (
                guard: ReturnType<typeof canDisable>,
              ): string | undefined =>
                guard.allowed
                  ? undefined
                  : guard.reason === 'self'
                    ? t.adminUsers.lockedSelf
                    : t.adminUsers.lockedLastAdmin
              return (
                <TableRow key={user.subject}>
                  <TableCell>
                    <div className="flex items-center gap-2.5">
                      <InitialsAvatar
                        displayName={user.display_name}
                        email={user.email}
                      />
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="t-list truncate text-foreground">
                            {user.display_name ?? user.email}
                          </span>
                          {self ? (
                            <StatusBadge
                              density="table"
                              label={t.adminUsers.you}
                              tone="brand"
                            />
                          ) : null}
                        </div>
                        <span className="t-meta-sm block truncate text-muted-foreground">
                          {user.email}
                        </span>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Select
                      disabled={!roleGuardUser.allowed && user.instance_role === 'admin'}
                      onValueChange={(value) =>
                        void setRole(user.subject, value as 'admin' | 'user')
                      }
                      value={user.instance_role}
                    >
                      <SelectTrigger
                        className="w-32 border-border bg-background shadow-none disabled:bg-surface/60 disabled:text-muted-foreground disabled:opacity-100"
                        density="table"
                        title={lockReason(roleGuardUser)}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="admin">
                          {t.adminUsers.roleAdmin}
                        </SelectItem>
                        <SelectItem value="user">
                          {t.adminUsers.roleUser}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell>
                    <StatusBadge
                      density="table"
                      label={
                        user.disabled
                          ? t.adminUsers.statusDisabled
                          : t.adminUsers.statusActive
                      }
                      tone={user.disabled ? 'destructive' : 'success'}
                    />
                  </TableCell>
                  <TableCell>
                    <span className="t-mono tabular-nums text-muted-foreground">
                      {formatDate(user.last_login_at, locale) ??
                        t.adminUsers.neverLoggedIn}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    {/* The Status column already shows active/disabled; the
                        switch's aria-label carries the action verb, so no
                        (easily-inverted) inline text label is needed. */}
                    <div className="flex items-center justify-end gap-1">
                      {mode === 'local' || state.demo ? (
                        <Button
                          aria-label={t.adminUsers.resetButton}
                          className="size-7 text-muted-foreground"
                          onClick={() => setResetTarget(user)}
                          size="icon"
                          title={t.adminUsers.resetButton}
                          variant="ghost"
                        >
                          <KeyRound className="icon-sm" />
                        </Button>
                      ) : null}
                      <Switch
                        aria-label={
                          user.disabled
                            ? t.adminUsers.enable
                            : t.adminUsers.disable
                        }
                        checked={!user.disabled}
                        disabled={!user.disabled && !disableGuard.allowed}
                        onCheckedChange={(checked) =>
                          void setDisabled(user.subject, !checked)
                        }
                        density="table"
                        title={
                          user.disabled ? undefined : lockReason(disableGuard)
                        }
                      />
                    </div>
                  </TableCell>
                </TableRow>
              )
            })
          )}
        </TableBody>
      </Table>

      {createOpen ? (
        <CreateUserDialog
          createUser={createUser}
          onClose={() => setCreateOpen(false)}
        />
      ) : null}

      {resetTarget ? (
        <ResetPasswordDialog
          onClose={() => setResetTarget(null)}
          resetPassword={resetPassword}
          user={resetTarget}
        />
      ) : null}
    </div>
  )
}

function ResetPasswordDialog({
  onClose,
  resetPassword,
  user,
}: {
  onClose: () => void
  resetPassword: ReturnType<typeof useAdminUsers>['resetPassword']
  user: AdminUser
}) {
  const { t } = useLocale()
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [revealed, setRevealed] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await resetPassword(user.subject, password)
      setRevealed(password)
    } catch {
      setError(t.adminUsers.resetFailed)
      setSubmitting(false)
    }
  }

  if (revealed) {
    return (
      <Dialog
        closeLabel={t.adminUsers.done}
        description={user.email ?? undefined}
        dismissable={false}
        footer={
          <Button onClick={onClose} size="sm">
            {t.adminUsers.done}
          </Button>
        }
        onClose={onClose}
        open
        title={t.adminUsers.resetCreatedTitle}
      >
        <div className="rounded-md border border-warning/25 bg-warning/10 p-3">
          <p className="t-label text-foreground">
            {t.adminUsers.resetCreatedHint}
          </p>
          <div className="mt-2 flex items-center gap-2">
            <code className="t-mono flex-1 select-all break-all rounded bg-background px-2 py-1.5 text-foreground">
              {revealed}
            </code>
            <Button
              className="shrink-0 gap-1.5"
              onClick={() => {
                void navigator.clipboard?.writeText(revealed)
                setCopied(true)
              }}
              size="sm"
              variant="outline"
            >
              {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
              {copied ? t.adminUsers.copied : t.adminUsers.copyPassword}
            </Button>
          </div>
        </div>
      </Dialog>
    )
  }

  const canSubmit = isPasswordAcceptable(password) && !submitting
  return (
    <Dialog
      closeLabel={t.adminTokens.cancel}
      description={t.adminUsers.resetDescription}
      onClose={onClose}
      open
      title={t.adminUsers.resetTitle(user.display_name ?? user.email ?? '')}
    >
      <form className="space-y-4" onSubmit={handleSubmit}>
        <div>
          <label className="t-label mb-1.5 block text-foreground" htmlFor="reset-password">
            {t.adminUsers.fieldPassword}
          </label>
          <div className="flex items-center gap-2">
            <Input
              autoComplete="new-password"
              autoFocus
              id="reset-password"
              onChange={(event) => setPassword(event.target.value)}
              type="text"
              value={password}
            />
            <Button
              className="shrink-0"
              onClick={() => setPassword(suggestPassword())}
              size="sm"
              type="button"
              variant="outline"
            >
              {t.adminUsers.generatePassword}
            </Button>
          </div>
          {password && !isPasswordAcceptable(password) ? (
            <p className="t-hint mt-1 text-warning">{`min. ${MIN_PASSWORD_LENGTH}`}</p>
          ) : null}
        </div>
        {error ? (
          <p className="flex gap-1.5 text-xs font-medium text-destructive" role="alert">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{error}</span>
          </p>
        ) : null}
        <div className="flex items-center justify-end gap-2 pt-1">
          <Button onClick={onClose} size="sm" type="button" variant="ghost">
            {t.adminTokens.cancel}
          </Button>
          <Button
            className="bg-brand text-brand-foreground hover:bg-brand/90"
            disabled={!canSubmit}
            size="sm"
            type="submit"
          >
            {submitting ? t.adminUsers.resetting : t.adminUsers.resetSubmit}
          </Button>
        </div>
      </form>
    </Dialog>
  )
}

function CreateUserDialog({
  createUser,
  onClose,
}: {
  createUser: ReturnType<typeof useAdminUsers>['createUser']
  onClose: () => void
}) {
  const { t } = useLocale()
  const [email, setEmail] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<'admin' | 'user'>('user')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [created, setCreated] = useState<{ email: string; password: string } | null>(null)
  const [copied, setCopied] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await createUser({
        email: email.trim(),
        password,
        instanceRole: role,
        displayName: displayName.trim() || undefined,
      })
      setCreated({ email: email.trim(), password })
    } catch (caught) {
      setError(
        hasHttpStatus(caught, 409)
          ? t.adminUsers.duplicateEmail
          : t.adminUsers.createFailed,
      )
      setSubmitting(false)
    }
  }

  if (created) {
    return (
      <Dialog
        closeLabel={t.adminUsers.done}
        dismissable={false}
        footer={
          <Button onClick={onClose} size="sm">
            {t.adminUsers.done}
          </Button>
        }
        onClose={onClose}
        open
        title={t.adminUsers.createdTitle}
      >
        <p className="t-meta text-muted-foreground">{created.email}</p>
        <div className="mt-3 rounded-md border border-warning/25 bg-warning/10 p-3">
          <p className="t-label text-foreground">{t.adminUsers.createdHint}</p>
          <div className="mt-2 flex items-center gap-2">
            <code className="t-mono flex-1 select-all break-all rounded bg-background px-2 py-1.5 text-foreground">
              {created.password}
            </code>
            <Button
              className="shrink-0 gap-1.5"
              onClick={() => {
                void navigator.clipboard?.writeText(created.password)
                setCopied(true)
              }}
              size="sm"
              variant="outline"
            >
              {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
              {copied ? t.adminUsers.copied : t.adminUsers.copyPassword}
            </Button>
          </div>
        </div>
      </Dialog>
    )
  }

  const canSubmit =
    email.includes('@') && isPasswordAcceptable(password) && !submitting

  return (
    <Dialog
      closeLabel={t.adminTokens.cancel}
      description={t.adminUsers.createDescription}
      onClose={onClose}
      open
      title={t.adminUsers.createTitle}
    >
      <form className="space-y-4" onSubmit={handleSubmit}>
        <div>
          <label className="t-label mb-1.5 block text-foreground" htmlFor="admin-create-email">
            {t.adminUsers.fieldEmail}
          </label>
          <Input
            autoFocus
            id="admin-create-email"
            onChange={(event) => setEmail(event.target.value)}
            type="email"
            value={email}
          />
        </div>
        <div>
          <label className="t-label mb-1.5 block text-foreground" htmlFor="admin-create-name">
            {t.adminUsers.fieldDisplayName}
          </label>
          <Input
            id="admin-create-name"
            onChange={(event) => setDisplayName(event.target.value)}
            value={displayName}
          />
          <p className="t-hint mt-1 text-muted-foreground">
            {t.adminUsers.fieldDisplayNameHint}
          </p>
        </div>
        <div>
          <label className="t-label mb-1.5 block text-foreground" htmlFor="admin-create-password">
            {t.adminUsers.fieldPassword}
          </label>
          <div className="flex items-center gap-2">
            <Input
              autoComplete="new-password"
              id="admin-create-password"
              onChange={(event) => setPassword(event.target.value)}
              type="text"
              value={password}
            />
            <Button
              className="shrink-0"
              onClick={() => setPassword(suggestPassword())}
              size="sm"
              type="button"
              variant="outline"
            >
              {t.adminUsers.generatePassword}
            </Button>
          </div>
          {password && !isPasswordAcceptable(password) ? (
            <p className="t-hint mt-1 text-warning">
              {`min. ${MIN_PASSWORD_LENGTH}`}
            </p>
          ) : null}
        </div>
        <div>
          <label className="t-label mb-1.5 block text-foreground" htmlFor="admin-create-role">
            {t.adminUsers.fieldRole}
          </label>
          <Select onValueChange={(value) => setRole(value as 'admin' | 'user')} value={role}>
            <SelectTrigger className="w-full" id="admin-create-role">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="user">{t.adminUsers.roleUser}</SelectItem>
              <SelectItem value="admin">{t.adminUsers.roleAdmin}</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {error ? (
          <p className="flex gap-1.5 text-xs font-medium text-destructive" role="alert">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{error}</span>
          </p>
        ) : null}
        <div className="flex items-center justify-end gap-2 pt-1">
          <Button onClick={onClose} size="sm" type="button" variant="ghost">
            {t.adminTokens.cancel}
          </Button>
          <Button
            className="bg-brand text-brand-foreground hover:bg-brand/90"
            disabled={!canSubmit}
            size="sm"
            type="submit"
          >
            {submitting ? t.adminUsers.creating : t.adminUsers.createSubmit}
          </Button>
        </div>
      </form>
    </Dialog>
  )
}
