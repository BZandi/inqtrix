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
| `none` | Single user, local machine, or behind your own gateway | `__anonymous__` | none |
| `apikey` | One shared token for scripts and a trusted UI | `__static__` | `Authorization: Bearer <key>` |
| `oidc` | Multi-user browser deployment against an identity provider | IdP subject (`oidc_session`) | session cookie + CSRF token |
| `local` | Multi-user with native email/password accounts (no external IdP) | local user (`oidc_session`, issuer `local`) | session cookie + CSRF token |
| `ldap` | Multi-user bound to an existing LDAP/AD directory | directory user (`oidc_session`, issuer `ldap`) | session cookie + CSRF token |

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
                               "sub": ..., "email": ..., "display_name": ...,
                               "csrf_token": ...} — or {"authenticated": false}
POST /api/auth/logout          destroys the server-side session
                               (CSRF-protected like every unsafe request)
```

Every unsafe method (`POST`/`PUT`/`PATCH`/`DELETE`) of a cookie-authenticated
request must carry the bootstrap token in the `X-CSRF-Token` header; a
missing or invalid token returns 403 with `"type": "csrf_error"`.

Session, login-flow, and user records follow `INQTRIX_STORAGE_BACKEND`:
`memory` (default, single process, logins lost on restart) or `postgres`
(logins survive restarts and replica switches). See
[Settings and env](../configuration/settings-and-env.md).

## Bring your own IdP

Inqtrix speaks only standard OIDC — discovery, authorization code + PKCE —
and hardwires no identity provider (ADR-AUTH-1). Dex, Keycloak, Entra ID,
Okta, and authentik are configuration, not code. The identity anchor is
`(issuer, subject)`; email is display metadata, never an identity key.

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
| `INQTRIX_OIDC_SKIP_EMAIL_VERIFIED` | `false` | accept tokens without `email_verified=true` |
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
docker compose -f deploy/compose/compose.dev.yaml --profile oidc up -d
```

The bundled `deploy/compose/dex-config.yaml` registers issuer
`http://127.0.0.1:5556/dex`, client `inqtrix-local` (secret from
`INQTRIX_OIDC_CLIENT_SECRET`, compose default `inqtrix-dev-oidc-secret`),
callbacks for the API on port 5100 and the Vite dev server on 5173, and a
demo user `admin@example.com` / `password`.

```dotenv
INQTRIX_AUTH_MODE=oidc
INQTRIX_OIDC_ISSUER=http://127.0.0.1:5556/dex
INQTRIX_OIDC_CLIENT_ID=inqtrix-local
INQTRIX_OIDC_CLIENT_SECRET=inqtrix-dev-oidc-secret
INQTRIX_OIDC_REDIRECT_URL=http://127.0.0.1:5100/api/auth/callback
INQTRIX_SESSION_SECRET=dev-session-secret-xxxxx
INQTRIX_OIDC_INSECURE_DEV_COOKIES=true
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
(ADR-AUTH-3) under the synthetic issuer `local`; only the login transport
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
  `X-CSRF-Token` header on every unsafe method.
- **Cookies** — HttpOnly session cookie with the `__Host-` prefix,
  `Secure`, and `SameSite=Lax` in secure mode.
- **`Cache-Control: no-store`** on every `/api/auth/*` response.

TLS, CORS, and request limits are covered in
[Security hardening](security-hardening.md).

## Related docs

- [Security hardening](security-hardening.md)
- [Web server mode](webserver-mode.md)
- [Settings and env](../configuration/settings-and-env.md)
- [React UI](react-ui.md)

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

| Route | Effect |
|-------|--------|
| `GET /v1/admin/quota` | Overview: metered subjects with usage, effective limits, and operator ceilings. |
| `PUT /v1/admin/quota/limits` | Set the tenant default or a per-user override (`{"subject_id": "<sub>\|__quota_default__", "dimension": ..., "value": <int>=0>}`; `0` = unlimited). |
| `DELETE /v1/admin/quota/limits?subject_id=...&dimension=...` | Drop a limit so it falls back to the next layer (204). |
| `POST /v1/admin/quota/reset` | Zero one subject's current-window flow usage (`{"subject_id": ..., "dimension": ...}`; stock dimensions are a 400). |

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

| Route | Effect |
|-------|--------|
| `GET /v1/admin/workspaces` | List every workspace with its member count. |
| `POST /v1/admin/workspaces` | Create a workspace (`{"name": "Team"}`); the admin becomes its OWNER. |
| `PATCH /v1/admin/workspaces/{id}` | Rename (`{"name": "..."}`). |
| `DELETE /v1/admin/workspaces/{id}` | Delete the workspace; memberships cascade. |
| `GET /v1/admin/workspaces/{id}/members` | Members with display name + email. |
| `POST /v1/admin/workspaces/{id}/members` | Assign a user (`{"sub": ..., "role": "viewer\|commenter\|editor\|owner"}`); an unknown or disabled user is 404. |
| `PATCH /v1/admin/workspaces/{id}/members/{sub}` | Change a member's role; demoting the last OWNER is refused (409). |
| `DELETE /v1/admin/workspaces/{id}/members/{sub}` | Remove a member; removing the last OWNER is refused (409). |

### Confining sharing to workspaces

By default resource sharing is tenant-wide: any authenticated user may be a
share target, and the share typeahead (`GET /v1/users/search`) searches the
whole tenant. Set `INQTRIX_SHARING_RESTRICT_TO_WORKSPACE_MEMBERS=true` to
confine collaboration to workspace boundaries — a user may then only share
with people they share at least one workspace with, and the typeahead is
scoped the same way. The check runs at grant time (`POST /v1/shares` returns a
400 for a non-co-member), so the typeahead filter is convenience, not the
boundary. Default `false` keeps the historical tenant-wide behaviour
byte-identical and never revokes an existing grant.

