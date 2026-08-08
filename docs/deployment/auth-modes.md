# Authentication modes

> Files: `src/inqtrix/auth/principal.py`, `src/inqtrix/auth/api_key.py`, `src/inqtrix/auth/oidc.py`, `src/inqtrix/server/routers/auth.py`, `deploy/compose/dex-config.yaml`

## Scope

Authentication is a building block behind the `AuthProvider` seam: every
request resolves to a `Principal` (downstream code never holds
`Principal | None`), and the active mode is decided exactly once at startup
by `resolve_auth_mode()` and logged (`auth_mode=...` in the startup line).
Misconfiguration raises at startup instead of silently downgrading.

| Mode | Scenario | Principal | Credential per request |
|---|---|---|---|
| `none` | Single user, local machine, or behind your own gateway | anonymous, `user_id=null` | none |
| `apikey` | One shared token for scripts and a trusted UI | static, `user_id=null` | `Authorization: Bearer <key>` |
| `oidc` | Multi-user browser deployment against an identity provider | canonical `users.id` UUID (`oidc_session`) | session cookie + CSRF token |
| `local` | Multi-user with native email/password accounts (no external IdP) | canonical `users.id` UUID (`oidc_session`) | session cookie + CSRF token |
| `ldap` | Multi-user bound to an existing LDAP/AD directory | canonical `users.id` UUID (`oidc_session`) | session cookie + CSRF token |

### Mode resolution when `INQTRIX_AUTH_MODE` is unset

The default is the sentinel `infer`, which derives the mode for backwards
compatibility: a non-empty `INQTRIX_SERVER_API_KEY` means `apikey`, an empty
one means `none`. Pre-mode deployments keep behaving bit-for-bit without
touching their configuration.

An explicit `INQTRIX_AUTH_MODE` always wins, and contradictions fail loudly:

| Explicit value | Contradiction | Behaviour |
|---|---|---|
| `oidc` | issuer, client id, client secret, or session secret missing | `RuntimeError` at startup naming the missing variables |
| `oidc` | neither `INQTRIX_OIDC_REDIRECT_URL` nor `INQTRIX_PUBLIC_BASE_URL` set | `RuntimeError` at startup |
| `apikey` | `INQTRIX_SERVER_API_KEY` empty | `RuntimeError` at startup |
| `local` | `INQTRIX_SESSION_SECRET` or `INQTRIX_PAT_PEPPER` missing | `RuntimeError` at startup |
| `ldap` | LDAP URL, bind DN/password, search base, or a session/PAT secret missing | `RuntimeError` at startup naming the missing variables |
| `none` | `INQTRIX_SERVER_API_KEY` set | gate deliberately disabled; WARNING logged |

### Gated vs. open routes

Open by design in every mode, because UIs and probes need them before
credentials exist: `/health`, `/v1/models`, `/v1/capabilities`,
`/v1/stacks` (multi-stack apps), and `/api/auth/config` (pre-login
discovery — the active mode, login methods, SSO provider name, whether
self-registration/owner-setup apply; `no-store`). Every other route takes the principal
dependency: `/v1/chat/completions`, `/v1/text/improvements`,
`/v1/editor/*`, `/v1/runs*`, `/v1/test/run`, and — when their feature
gates register them — `/v1/files*`, `/v1/knowledge/*`, `/v1/sources/*`.
In `oidc` mode the `/api/auth/*` routes are additionally mounted; they are
the credential surface itself (only `/api/auth/logout` requires an
authenticated session).

`GET /v1/stacks` is capability discovery, not a required business-data read.
A single-stack deployment may answer 404 to indicate that no stack registry is
available; the research desk treats that result as an expected capability
outcome. Likewise, a 404 while deleting an already-absent synchronized asset is
an idempotent success for that delete. Neither optional 404 is a global server
synchronization failure.

## Mode `none`

No configuration. Leave `INQTRIX_SERVER_API_KEY` empty (or set
`INQTRIX_AUTH_MODE=none` explicitly). Every request resolves to the
anonymous principal:

```bash
curl http://127.0.0.1:5100/v1/runs -X POST \
     -H "Content-Type: application/json" \
     -d '{"question":"Wie ist der Stand bei Smart Metern?"}'
```

## Mode `apikey`

| Variable | Required | Notes |
|---|---|---|
| `INQTRIX_SERVER_API_KEY` | yes | the single shared token |
| `INQTRIX_AUTH_MODE` | no | `apikey` is inferred from the key; set it explicitly to fail loudly if the key goes missing |

The gate is byte-identical to the legacy static Bearer check: constant-time
comparison via `hmac.compare_digest`, and 401 responses keep the historical
envelope `{"error": {"message": ..., "type": "unauthorized"}}` with a
`WWW-Authenticate: Bearer` header.

```bash
curl http://127.0.0.1:5100/v1/chat/completions -X POST \
     -H "Authorization: Bearer dev-secret-xxxxx" \
     -H "Content-Type: application/json" \
     -d '{"model":"research-agent","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Mode `oidc`

| Variable | Required | Notes |
|---|---|---|
| `INQTRIX_AUTH_MODE` | yes | must be `oidc` (never inferred) |
| `INQTRIX_OIDC_ISSUER` | yes | pinned issuer URL; discovery and every id_token must echo it |
| `INQTRIX_OIDC_CLIENT_ID` | yes | OAuth client registered at the IdP |
| `INQTRIX_OIDC_CLIENT_SECRET` | yes | confidential-client secret (the server is the OAuth client, not the browser) |
| `INQTRIX_SESSION_SECRET` | yes | CSRF-token derivation secret; rotating it invalidates outstanding CSRF tokens, sessions survive |
| `INQTRIX_PAT_PEPPER` | yes | Server-side HMAC pepper for personal access tokens; required from first boot even before any token exists. Rotating it invalidates every issued token |
| `INQTRIX_OIDC_REDIRECT_URL` | one of the two | absolute callback URL registered at the IdP (exact string match) |
| `INQTRIX_PUBLIC_BASE_URL` | one of the two | empty redirect URL derives `{base}/api/auth/callback` |
| `INQTRIX_SESSION_MAX_AGE_SECONDS` | no | absolute session lifetime, default `28800` (8 h); expiry yields 401 and the SPA re-runs the login redirect — no silent refresh |

### Browser flow

```
GET /api/auth/login            302 to the IdP authorization endpoint
                               (code flow, PKCE S256, state, nonce;
                                a short-lived flow cookie binds the
                                transaction to this browser)
        ... user authenticates at the IdP ...
GET /api/auth/callback?code=&state=
                               server-side code exchange + id_token
                               validation; 303 back into the SPA with
                               an HttpOnly session cookie and a
                               JS-readable CSRF cookie
GET /api/auth/session          SPA bootstrap: {"authenticated": true,
                               "user": {"id": "<uuid>", "email": ...,
                               "display_name": ..., "role": ...},
                               "project_namespace": ...,
                               "csrf_token": ...} — or {"authenticated": false}
POST /api/auth/logout          destroys the server-side session
                               (CSRF-protected like every unsafe request)
```

Every unsafe method (`POST`/`PUT`/`PATCH`/`DELETE`) of a cookie-authenticated
request must carry the bootstrap token in the `X-CSRF-Token` header; a
missing or invalid token returns 403 with `"type": "csrf_error"`.

After session bootstrap, the Research Desk also sends
`X-Inqtrix-Expected-User-Id` on protected API requests. This is not an
authentication credential: the server still resolves the cookie/PAT normally,
then compares the live principal with the user whose state the SPA rendered.
If another tab changed the cookie session, the request is rejected before the
domain operation with `409 principal_changed` and the SPA reloads. API clients
that omit the additive header keep the existing contract. The per-user SSE
stream repeats the live check before data and quiet keepalive frames.

Session, login-flow, and user records follow `INQTRIX_STORAGE_BACKEND`:
`memory` (default, single process, logins lost on restart) or `postgres`
(logins survive restarts and replica switches). See
[Settings and env](../configuration/settings-and-env.md).

## Bring your own IdP

Inqtrix speaks only standard OIDC — discovery, authorization code + PKCE —
and hardwires no identity provider. Dex, Keycloak, Entra ID, Okta, and
authentik are configuration, not code. `(issuer, subject)` is retained only
inside the authentication adapter as the external-account binding. A
successful login resolves that binding to the deployment-local `users.id`
UUID; every ownership, membership, quota, sharing, session, PAT, audit, and
actor decision uses that UUID. Email is display metadata, never an identity
key.

Beyond the connection variables above, the claim-mapping contract:

| Variable | Default | Purpose |
|---|---|---|
| `INQTRIX_OIDC_SCOPES` | `openid profile email` | authorization-request scopes |
| `INQTRIX_OIDC_USERNAME_CLAIM` | `preferred_username` | display username; falls back to email, then `sub`; dot-paths descend nested claims |
| `INQTRIX_OIDC_EMAIL_CLAIM` | `email` | email claim |
| `INQTRIX_OIDC_GROUPS_CLAIM` | `groups` | group memberships; a JSON array, or a string split on `INQTRIX_OIDC_CLAIM_SEPARATORS`; dot-path capable |
| `INQTRIX_OIDC_ROLES_CLAIM` | `roles` | role assignments for admin elevation; dot-path capable (Keycloak `realm_access.roles` / `resource_access.<client>.roles`); may equal the groups claim |
| `INQTRIX_OIDC_ALLOWED_GROUPS` | empty | comma-separated admission allowlist; no overlap is a visible 403; `*` admits any authenticated user; empty disables the gate |
| `INQTRIX_OIDC_ALLOWED_DOMAINS` | empty | comma-separated email-domain admission allowlist (case-insensitive), orthogonal to groups; a login without an email is rejected |
| `INQTRIX_OIDC_ADMIN_ROLES` | empty | comma-separated role values that grant instance-admin on login (grant-only) |
| `INQTRIX_OIDC_ADMIN_GROUPS` | empty | comma-separated group values that grant instance-admin on login (grant-only) |
| `INQTRIX_OIDC_CLAIM_SEPARATORS` | space + comma | characters a string-valued group/role claim is split on (a JSON array is used as-is) |
| `INQTRIX_OIDC_GROUPS_STRIP_PATH_PREFIX` | `false` | strip one leading `/` from group values (Keycloak full-path groups) |
| `INQTRIX_OIDC_SKIP_EMAIL_VERIFIED` | `false` | explicitly bypass the requirement that `email_verified` is exactly `true`; missing and `false` are rejected alike, and enabling this bypass logs a warning |
| `INQTRIX_OIDC_DISCOVERY_URL` | empty | metadata URL override; the document's `issuer` must still match `INQTRIX_OIDC_ISSUER` |
| `INQTRIX_OIDC_USERINFO_FALLBACK` | `true` | fetch userinfo when the id_token lacks mapped claims (incl. the roles claim when admin elevation is configured) |
| `INQTRIX_OIDC_PROVIDER_NAME` | empty | SSO button display name surfaced by `GET /api/auth/config` (e.g. `Okta`) |
| `INQTRIX_OIDC_CA_CERT` | empty | PEM CA bundle for IdPs behind a private CA |

Group and role values are matched **literally** — configure the exact value
the IdP emits (a GUID, a name, a `/path`), not a friendly alias.

Provider mapping notes:

- **Entra ID** omits `email_verified` — set
  `INQTRIX_OIDC_SKIP_EMAIL_VERIFIED=true`. Groups arrive as object GUIDs by
  default (match the GUID). Genuine group **overage** (a distributed
  `_claim_names`/`_claim_sources` pointer, not an inline array) cannot be
  resolved from the token: it surfaces as a visible error when groups gate
  admission and a logged warning otherwise — never a silent empty-group
  admit. App roles arrive in the `roles` claim.
- **Okta** needs an extra `groups` entry in `INQTRIX_OIDC_SCOPES` for group
  claims; its thin id_tokens rely on the userinfo fallback (default on).
- **Keycloak** nests roles — dot-paths reach them:
  `INQTRIX_OIDC_ROLES_CLAIM=realm_access.roles` (realm) or
  `resource_access.<client>.roles` (client). Group paths arrive as
  `/Parent/Child`; set `INQTRIX_OIDC_GROUPS_STRIP_PATH_PREFIX=true` to drop
  the leading slash.

**Admin elevation from claims:** a user whose roles match
`INQTRIX_OIDC_ADMIN_ROLES`, or whose groups match `INQTRIX_OIDC_ADMIN_GROUPS`,
is promoted to instance-admin on login. This is **grant-only** — a non-match
never demotes; revocation is the admin UI's job (the same last-admin-guarded
path the LDAP admin-group mapping uses).

### Dex as the dev reference IdP

The compose stack ships Dex behind the profile `oidc` — a development
reference, not an architectural dependency:

```bash
docker compose \
  -f deploy/compose/compose.stack.yaml \
  -f deploy/compose/compose.dev-ports.yaml \
  --env-file deploy/.env.stack.secrets.local \
  --env-file deploy/.env.stack.local \
  --profile oidc up -d dex
```

The bundled `deploy/compose/dex-config.yaml` registers issuer
`http://dex.localhost:5556/dex`, client `inqtrix-local` (secret from
`INQTRIX_OIDC_CLIENT_SECRET` in the selected secrets file), callbacks for the
API on port 5100 and the Vite dev server on 5173, and a demo user
`admin@example.com` / `password`.

Visible configuration (`deploy/.env.stack.local`):

```dotenv
INQTRIX_AUTH_MODE=oidc
INQTRIX_OIDC_ISSUER=http://dex.localhost:5556/dex
INQTRIX_OIDC_CLIENT_ID=inqtrix-local
INQTRIX_OIDC_REDIRECT_URL=http://127.0.0.1:5100/api/auth/callback
INQTRIX_OIDC_INSECURE_DEV_COOKIES=true
```

Credentials (`deploy/.env.stack.secrets.local`, mode `0600`):

```dotenv
INQTRIX_OIDC_CLIENT_SECRET=replace-with-a-local-dev-secret
INQTRIX_SESSION_SECRET=replace-with-an-independent-session-secret
INQTRIX_PAT_PEPPER=replace-with-an-independent-pat-pepper
```

`INQTRIX_OIDC_INSECURE_DEV_COOKIES=true` drops the `Secure` flag and the
`__Host-` prefix so login works over plain `http://127.0.0.1` in every
browser (Safari rejects Secure cookies on loopback HTTP). The activation is
logged loudly; never enable it in production.

A copy-paste, run-it-now recipe (start Dex, log in through the browser,
verify over curl) is the
[OIDC stack walkthrough](../../examples/webserver_stacks/oidc_stack.md).

## Mode `local`

Native email/password accounts — the **default** — for multi-user
deployments without an external IdP. The first visit creates the instance
owner; the in-app admin area then manages users, invitations, and access
tokens. It reuses the OIDC cookie session + CSRF machinery verbatim
under the synthetic issuer `local`; only the login transport
differs (a server-side password check via `POST /api/auth/login/local`).
Requires `INQTRIX_SESSION_SECRET`, `INQTRIX_PAT_PEPPER`, and the Postgres
backend for durable accounts. Self-signup is off by default
(`INQTRIX_LOCAL_REGISTRATION=open` mounts a public signup route). Walkthrough:
[Create and manage users](../how-to/create-and-manage-users.md).

## Mode `ldap`

Bind logins against a directory you already run (OpenLDAP, Active Directory,
FreeIPA, 389-DS) — search-then-bind, no passwords stored, configuration only.
Like `local` it reuses the cookie session machinery under the synthetic issuer
`ldap`. Admin-group membership grants instance-admin on login (grant-only).
Required: `INQTRIX_LDAP_URL`, `INQTRIX_LDAP_BIND_DN` / `_PASSWORD`,
`INQTRIX_LDAP_USER_SEARCH_BASE`, plus the session/PAT secrets and Postgres.
Full reference and per-directory cheat sheet:
[Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md); a
copy-paste recipe is the
[LDAP stack walkthrough](../../examples/webserver_stacks/ldap_stack.md).

## Security properties

- **BFF pattern** — the server is a confidential OAuth client; tokens are
  validated at login and then discarded. The browser only ever holds an
  opaque session id, never a token.
- **PKCE S256 + state + nonce** on every authorization request; a
  short-lived flow cookie binds the transaction to the initiating browser
  (login-CSRF defense).
- **id_token validation** — algorithm allowlist (RS256/ES256), pinned
  issuer, audience, expiry, one-time nonce; JWKS cached with a TTL and
  cooldown-limited refresh on unknown key ids.
- **CSRF** — OWASP signed double-submit: the token is an HMAC over the
  session id, delivered in a non-HttpOnly cookie and required back in the
  `X-CSRF-Token` header on every unsafe method. A successful
  `GET /api/auth/session` refreshes that readable cookie from the current
  server secret. The SPA uses this only for a typed `csrf_error`: concurrent
  mutations share one bootstrap and each original request is retried at most
  once. Other 403 responses, guest tokens, and bearer-token requests are never
  retried through this path.
- **Cookies** — HttpOnly session cookie with the `__Host-` prefix,
  `Secure`, and `SameSite=Lax` in secure mode.
- **`Cache-Control: no-store`** on every `/api/auth/*` response.
- **Live account status** — cookie sessions and PATs resolve the canonical
  user on every request. Disabling a user invalidates existing credentials;
  re-enabling the account does not restore old sessions or PATs.

## Canonical identity and direct sharing

`users.id` (`UUID`) is the only application-level user identifier. Public
session, user-search, admin, quota, workspace-member, and sharing contracts use
`user.id` or `user_id`; they do not expose or accept IdP subjects. Anonymous
and static-key deployments deliberately keep `user_id=null`, create no
synthetic user, and do not expose sharing.

Resource sharing is a direct user-to-resource consent contract. It supports
`view` and `edit` for `run`, `knowledge_collection`, `prompt_template`, and
`skill_template`. Collaboration-mode `editor_document` resources additionally
support `suggest`; that value is rejected for every other resource type. There
are no local sharing groups, inherited workspace resource rights, direct file
shares, or `comment`/`manage` permissions.
Workspace roles remain useful for workspace administration and, when enabled,
for constraining which users may receive a direct share; they do not themselves
grant access to a resource.

The lifecycle routes are:

| Route | Behaviour |
|---|---|
| `GET /v1/users/search?q=...` | Returns `id`, display name, and email for share recipients. |
| `POST /v1/shares` | Atomically creates only new pending shares. Body: `{"resource_type":"prompt_template","resource_id":"pt_...","invitees":[{"user_id":"<uuid>","permission":"edit"}]}`. Duplicate/invalid invitees return 400 with no writes; an existing active share returns 409. |
| `GET /v1/shares?resource_type=...&resource_id=...` | Owner-only lifecycle view for one resource. |
| `PATCH /v1/shares/{share_id}` | Owner-only permission update with `permission` and integer `expected_revision`; stale revisions return 409 with `current_revision`. Acceptance is retained for this same share. |
| `POST /v1/shares/{share_id}/accept` | Recipient-only, idempotent acceptance. Pending becomes active; an already accepted share returns its current record; revoked or foreign ids are indistinguishable 404s. |
| `DELETE /v1/shares/{share_id}` | Owner revokes, or recipient declines/leaves; returns 204. |
| `GET /v1/shares/inbox` | Incoming pending and accepted lifecycle records. It is not the resource list. |
| `GET /v1/shares/mine` | Owner summary of resources with outgoing shares. It is not the resource list. |

Revoking and sharing again creates a new share id and requires fresh consent.
Regular resource lists return owned and accepted-shared records together with
an `access` object: `{"mode":"unscoped"}`, `{"mode":"owner"}`, or
`{"mode":"shared","permission":"view|edit"}`; editor documents may also
return `permission="suggest"`. Authorization is checked
live at each read or mutation boundary; a list result or frontend cache is not
an authorization grant. Owner-only operations include resource deletion and
share management. An `edit` recipient may mutate the supported resource but
cannot re-share or delete it.

TLS, CORS, and request limits are covered in
[Security hardening](security-hardening.md).

## Personal access tokens (oidc, local, ldap)

Browser sessions cannot serve scripts or CI; a signed-in user (in any
cookie-session mode) mints per-user Bearer credentials instead of falling
back to a shared key.
Shape `ipat_<token_id>_<secret>`; only the peppered HMAC of the secret
is stored, and the plaintext is shown exactly once at creation.

| Route | Effect |
|-------|--------|
| `POST /api/auth/tokens` | Create (body: `{"name": "ci-runner", "expires_in_days": 30}`; expiry optional). Returns the plaintext ONCE. 409 at the per-user cap. |
| `GET /api/auth/tokens` | List own tokens (never secret material). |
| `DELETE /api/auth/tokens/{token_id}` | Revoke own token; foreign/unknown ids are indistinguishable 404s. |

Requests authenticate with `Authorization: Bearer ipat_...` — the
header routes exclusively to token verification (no cookie fallback)
and needs no CSRF header. Token management itself is session-only: a
PAT cannot mint or revoke PATs. Tuning: `INQTRIX_PAT_MAX_PER_USER`
(default 10), `INQTRIX_PAT_DEFAULT_TTL_DAYS` (default 0 = no expiry).
Tokens persist only with the postgres storage backend; the memory
default logs a loud warning that they vanish on restart.

PATs can access ordinary resource APIs according to their principal, but they
cannot obtain a live editor collaboration lease. That transport requires the
cookie session id so lease rotation and immediate session revocation remain
bound to the active browser login.

## Invitation-gated registration (oidc mode)

`INQTRIX_REGISTRATION=invite` closes registration: a first-time login
is admitted only when an open invitation matches the login email, and
the acceptance simultaneously grants the invited workspace
membership — checked at the callback BEFORE any user record or
session exists. Existing users always pass (and still collect newly
opened invitations on their next login); disabled users are denied in
every mode. The setting requires the postgres storage backend
(memory invitations would evaporate on restart and lock everyone
out) — the contradiction fails loudly at startup. The default
`open` keeps the historical admit-everyone behaviour.

| Route | Effect |
|-------|--------|
| `POST /v1/workspaces` | Create a workspace; the creating admin becomes its OWNER (`{"name": "Team"}`). Instance admin only — a non-admin session gets 404 (workspace creation is platform administration, not self-serve). |
| `GET /v1/workspaces` | The caller's memberships. |
| `POST /v1/workspaces/{id}/invitations` | Invite an email (`{"email": ..., "role": "viewer\|commenter\|editor\|owner", "expires_in_days": 14}`); OWNER only; one open invitation per email and workspace (409 on duplicates). |
| `GET /v1/workspaces/{id}/invitations` | List (OWNER only). |
| `DELETE /v1/workspaces/{id}/invitations/{invitation_id}` | Revoke an open invitation (OWNER only). |

## Instance-admin quota administration

Per-user usage quotas (when `INQTRIX_QUOTA_ENABLED=true`) are tenant-wide
platform administration, so the admin surface is gated on `instance_role ==
"admin"` — never on workspace ownership. It lives under `/v1/admin/quota*`;
denials hide behind 404 (a non-admin session, a PAT, or anonymous), and every
mutation is audited (`quota.override` / `quota.override_cleared` /
`quota.reset`). The self-meter `GET /v1/quota/usage` is separate and available
to any scoped principal.

With `INQTRIX_QUOTA_ENABLED=false`, quota accounting is disabled for every
user, independently of account count, instance role, and workspace membership.
No workspace is created as a side effect of creating or promoting an admin.
List parameters named `limit` remain pagination controls and do not enable a
quota.

| Route | Effect |
|-------|--------|
| `GET /v1/admin/quota` | Overview: metered users with usage, effective limits, and operator ceilings. |
| `PUT /v1/admin/quota/limits` | Set the tenant default or a per-user override (`{"user_id": "<uuid>\|default", "dimension": ..., "value": <int>=0>}`; `0` = unlimited). |
| `DELETE /v1/admin/quota/limits?user_id=...&dimension=...` | Drop a limit so it falls back to the next layer (204). |
| `POST /v1/admin/quota/reset` | Zero one user's current-window flow usage (`{"user_id": "<uuid>", "dimension": ...}`; stock dimensions are a 400). |

(Before v0.2.0 these lived under `/v1/workspaces/{id}/quota*` and were gated on
the workspace OWNER; the move to `/v1/admin/quota*` aligns them with the
instance-admin axis.)

## Instance-admin workspace management

The instance admin (`instance_role == "admin"`) manages the deployment's
workspaces and their membership — creating spaces and positioning users into
them — on the platform-administration axis, independent of workspace
ownership. Available in every cookie-session mode (`oidc`/`local`/`ldap`) when
a user mirror and the membership store are wired; denials hide behind 404 (a
non-admin session, a PAT, or anonymous). Every mutation is audited.

Workspace membership is optional for an instance admin. Multiple admins with
no workspace membership are supported; workspace creation and assignment are
explicit administrative actions rather than account-creation side effects.

| Route | Effect |
|-------|--------|
| `GET /v1/admin/workspaces` | List every workspace with its member count. |
| `POST /v1/admin/workspaces` | Create a workspace (`{"name": "Team"}`); the admin becomes its OWNER. |
| `PATCH /v1/admin/workspaces/{id}` | Rename (`{"name": "..."}`). |
| `DELETE /v1/admin/workspaces/{id}` | Delete the workspace; memberships cascade. |
| `GET /v1/admin/workspaces/{id}/members` | Members with display name + email. |
| `POST /v1/admin/workspaces/{id}/members` | Assign a user (`{"user_id": "<uuid>", "role": "viewer\|commenter\|editor\|owner"}`); an unknown or disabled user is 404. |
| `PATCH /v1/admin/workspaces/{id}/members/{user_id}` | Change a member's role; demoting the last OWNER is refused (409). |
| `DELETE /v1/admin/workspaces/{id}/members/{user_id}` | Remove a member; removing the last OWNER is refused (409). |

### Confining sharing to workspaces

By default resource sharing is tenant-wide: any authenticated user may be a
share target, and the share typeahead (`GET /v1/users/search`) searches the
whole tenant. Set `INQTRIX_SHARING_RESTRICT_TO_WORKSPACE_MEMBERS=true` to
confine collaboration to workspace boundaries — a user may then only share
with people they share at least one workspace with, and the typeahead is
scoped the same way. The typeahead remains convenience only; grant, accept,
and every live resource access enforce the same boundary fail-closed.

The constraint is continuous. Removing a member or deleting a workspace
revokes pending and accepted shares in either direction when that was the last
common workspace; another common workspace preserves them. Startup reconciles
existing shares before readiness whenever the setting is enabled. Turning the
setting off stops this workspace check but does not resurrect previously
revoked shares.

## Related docs

- [Security hardening](security-hardening.md)
- [Editor collaboration](editor-collaboration.md)
- [Settings and env](../configuration/settings-and-env.md)
- [React UI](react-ui.md)
