# Create and manage users

## Scope

How accounts come into being and how an admin manages them — across every auth
mode, and through the web app's Administration section. Covers first-owner
setup (`local`), adding accounts, instance roles, disabling, and personal
access tokens. It does **not** cover workspace-level sharing roles (a separate
concept — see below) or the OIDC/LDAP wiring itself (see
[Auth modes](../deployment/auth-modes.md),
[Connect to an existing LDAP](connect-to-existing-ldap.md)).

## How a user arrives, per mode

There is no single "create user" button that fits all modes — where identity
lives depends on the auth mode.

| Mode | User store | First user | More users | Sharing |
|---|---|---|---|---|
| `none` | none (one anonymous principal) | – | – | no (no scoped identity) |
| `apikey` | none (one static principal) | – | – | no |
| `local` | Postgres + argon2id | **first-run owner setup** | admin creates them (initial password) | yes |
| `ldap` | your directory + a JIT mirror | first bind (or admin group) | exist in the directory | yes |
| `oidc` | your IdP + a JIT mirror | first login (or allowed group) | IdP + invitations | yes |

`local` is the default and the only mode with an in-app "create
account" action; the single-operator `none`/`apikey` modes have no scoped
identity and therefore no user management or sharing.

## Two role concepts — do not conflate them

- **Instance role** (`admin` / `user`) — instance-wide. `admin` unlocks the
  Administration section. The `/api/auth/session` payload's `role` field is the
  instance role; the UI gate is default-closed (anything that is not exactly
  `admin` yields no admin surface).
- **Workspace role** (viewer / commenter / editor / owner) — per workspace, for
  sharing runs/collections/templates. Managed in the share dialog, not here.

Promoting someone to instance `admin` does not change their workspace roles, and
vice versa.

## `local`: first-run owner setup

On the first start of a `local` deployment the web app shows an **owner setup**
screen instead of the lock screen. The account you create there:

- becomes the **instance admin**;
- is logged in immediately (the server sets the session cookie);
- is the only owner — the setup screen never appears again (a second attempt is
  refused, race-safe).

The gate is driven by `GET /api/setup/status` (`{ "needs_owner": true }` only in
`local` mode with no owner yet); every other mode reports `false` and the gate
is inert.

```bash
# Probe the gate (local mode, fresh DB)
curl -s http://localhost:8080/api/setup/status        # -> {"needs_owner":true}
```

## Adding more accounts (`local`)

An admin opens **Settings → Administration → Users → Create user** and enters an
email, optional display name, an initial password (the dialog can generate a
strong one), and a role. The initial password is shown **once** — copy it and
share it securely with the person; it is never retrievable again. There is no
email delivery; you hand the credentials over out-of-band.

Behind the UI this is `POST /v1/admin/users` (admin-session-only; a personal
access token can never administer users):

```bash
curl -s -X POST http://localhost:8080/v1/admin/users \
  -H 'Content-Type: application/json' -H "X-CSRF-Token: $CSRF" -b cookies.txt \
  -d '{"email":"mara@example.com","password":"correct-horse-battery-staple","instance_role":"user"}'
```

Account-creation policy is `INQTRIX_LOCAL_REGISTRATION` — `closed` (default:
owner + admin-created accounts only) or `open` (a public self-signup route,
logged loudly).

For `ldap`/`oidc`, users already live in the directory/IdP — there is no
local "create"; they appear in the mirror on first login.

## Managing users (all multi-user modes)

In **Settings → Administration → Users** an admin can, per row:

- **Change the instance role** (admin ↔ user) via the role select;
- **Disable / enable** via the status switch — disabling is a complete cut-off:
  the mirror flag is set, live sessions are purged, personal access tokens are
  revoked, and (for `local`) the password credential is disabled so login is
  refused. Re-enabling clears the flag.

Two invariants protect the deployment from locking itself out, enforced
server-side (atomic) and reflected in the UI (the control is disabled with a
tooltip):

- you **cannot disable or demote yourself** (the "Sie"/"You" row);
- the **last active admin** can be neither demoted nor disabled.

## Passwords (local mode)

- **Change your own password** — a signed-in local user opens **Settings →
  Account → Security → Change password**, enters the current password and a new
  one. The session stays valid. (`ldap`/`oidc` passwords live upstream, so this
  only appears for `local`.)
- **Forgot a password** — there is no email reset; an admin sets a new one in
  **Settings → Administration → Users → (row) → Reset password**. The new
  password is shown **once** (copy + hand over securely); the user's live
  sessions are ended so the old password stops working immediately. PATs are
  left intact (a reset is not a full disable). To cut off all access instead,
  disable the account.
- **Brute-force protection** — repeated failed logins for the same
  `(email, IP)` are throttled (10 attempts / 5 min → 60 s lockout by default,
  `INQTRIX_LOGIN_RATE_LIMIT_*`); behind a proxy, also rate-limit at the edge
  ([Deploy to production](deploy-to-production.md#login-rate-limiting)).

## Personal access tokens

Any signed-in user (not just admins) manages **their own** API tokens under
**Settings → Account → Access tokens** — list, create, revoke. A created
token's plaintext is shown **once**; copy it immediately. Use it as a Bearer
token for the native API:

```bash
curl -s http://localhost:8080/v1/runs -H "Authorization: Bearer ipat_..."
```

Tokens are session-scoped to manage (a token can never mint or revoke tokens),
and a disabled user's tokens are revoked by the disable cascade.

## Related docs

- [Auth modes](../deployment/auth-modes.md) — choosing a mode; how each authenticates.
- [Connect to an existing LDAP](connect-to-existing-ldap.md) — bind to your directory.
- [Build a UI on Inqtrix](build-a-ui-on-inqtrix.md) — the native `/v1/runs` API and PATs.
- [Stack quickstart](../getting-started/stack-quickstart.md) — one-command start.
