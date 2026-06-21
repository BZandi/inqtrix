# Connect to an existing LDAP

## Scope

How to authenticate Research Desk users against a directory you **already
run** (OpenLDAP, Active Directory, FreeIPA, 389-DS, …) with
`INQTRIX_AUTH_MODE=ldap` — no Dex, no second identity provider, configuration
only. This page covers the env vars, the search-then-bind flow, attribute
mapping, admin-group mapping, and TLS. It does **not** cover running an LDAP
server yourself (Inqtrix never runs one — it binds to yours); a disposable
test directory is in the appendix. For the broader picture of every auth mode
see [Auth modes](../deployment/auth-modes.md); for managing the users that
arrive, [Create and manage users](create-and-manage-users.md).

## What `ldap` mode does

Inqtrix performs the classic **search-then-bind** on each login, the same flow
a directory admin would script:

1. Bind as a read-only **service account** (`INQTRIX_LDAP_BIND_DN`).
2. **Search** for the user under `INQTRIX_LDAP_USER_SEARCH_BASE` with the
   filter `INQTRIX_LDAP_USER_SEARCH_FILTER` — the typed login name is
   `escape_filter_chars`-escaped before it is substituted, so a value like
   `*)(uid=*` cannot widen the filter (LDAP injection is structurally
   rejected). Exactly one match is required.
3. **Re-bind** as the found user DN with the submitted password — that bind is
   the password check; Inqtrix never reads or stores the password hash.
4. Map attributes to an identity and mint the same cookie session as every
   other cookie mode (ADR-AUTH-3: the session kind is the transport, not the
   IdP). A wrong password and an unknown user fail identically.

The directory stays the source of truth for credentials; Inqtrix keeps only a
local mirror row (display name, email, instance role) for the admin list and
sharing typeaheads. It requires the Postgres backend (cookie sessions + the
mirror); on the memory backend sessions evaporate on restart.

## Configuration

Pick `ldap` mode and point the bind/search variables at your directory. Every
value is a constructor argument behind the scenes (Constructor-First); only the
settings bridge reads the environment.

```dotenv
INQTRIX_AUTH_MODE=ldap

# Connection (use ldaps:// or set INQTRIX_LDAP_START_TLS=true — see TLS below)
INQTRIX_LDAP_URL=ldaps://ldap.example.com:636

# Read-only service account used for the search step (least privilege)
INQTRIX_LDAP_BIND_DN=cn=inqtrix-svc,ou=services,dc=example,dc=com
INQTRIX_LDAP_BIND_PASSWORD=CHANGE_ME_SERVICE_PASSWORD

# Where and how to find the user. {username} is escaped before substitution.
INQTRIX_LDAP_USER_SEARCH_BASE=ou=people,dc=example,dc=com
INQTRIX_LDAP_USER_SEARCH_FILTER=(uid={username})

# Cookie-session secrets (shared with local/oidc) + durable backend
INQTRIX_SESSION_SECRET=CHANGE_ME_SESSION_SECRET
INQTRIX_PAT_PEPPER=CHANGE_ME_PAT_PEPPER
INQTRIX_STORAGE_BACKEND=postgres
INQTRIX_DATABASE_URL=postgresql+asyncpg://inqtrix:...@postgres:5432/inqtrix
```

### Every LDAP variable

| Variable | Required | Effect |
|---|---|---|
| `INQTRIX_LDAP_URL` | yes | Server URL. `ldaps://…:636` for implicit TLS, or `ldap://…:389` plus `INQTRIX_LDAP_START_TLS=true` for StartTLS. |
| `INQTRIX_LDAP_BIND_DN` | yes | Service-account DN that performs the user search. Read-only is enough. |
| `INQTRIX_LDAP_BIND_PASSWORD` | yes | Service-account password. A failed service bind is logged distinctly (operator misconfig) but still surfaces the uniform login error to the client. |
| `INQTRIX_LDAP_USER_SEARCH_BASE` | yes | Base DN for the user search (e.g. `ou=people,dc=example,dc=com`). |
| `INQTRIX_LDAP_USER_SEARCH_FILTER` | no (`(uid={username})`) | Search filter with a single `{username}` placeholder. For Active Directory use `(sAMAccountName={username})` or `(userPrincipalName={username})`. |
| `INQTRIX_LDAP_EMAIL_ATTR` | no (`mail`) | Attribute mapped to email; falls back to the login name when absent. |
| `INQTRIX_LDAP_DISPLAY_NAME_ATTR` | no (`cn`) | Attribute mapped to the display name; falls back to email. |
| `INQTRIX_LDAP_ID_ATTR` | no (`entryUUID`) | Stable identity anchor. Use a value that never changes — `entryUUID` (OpenLDAP) or `objectGUID` (AD). Falls back to the user DN if absent. |
| `INQTRIX_LDAP_ADMIN_GROUP_DN` | no | Members of this group DN (via `memberOf`) become instance admins on login (see below). |
| `INQTRIX_LDAP_START_TLS` | no (`false`) | Issue StartTLS on an `ldap://` connection. |
| `INQTRIX_LDAP_CA_CERT` | no | PEM CA-bundle path for a private-CA directory. |
| `INQTRIX_LDAP_TLS_VALIDATE` | no (`true`) | Verify the server certificate. Setting `false` logs a loud warning — trusted networks only, never production. |

> The identity anchor is `INQTRIX_LDAP_ID_ATTR`, not the email or the DN: emails
> and DNs move, the anchor must not. If you point it at a renamable attribute,
> a renamed user becomes a new identity (and loses their shares).

### Per-directory cheat sheet

The only per-directory differences are the **login filter** and the **id /
email / display / group** attributes; everything else is identical. Set these
and nothing else changes:

| Directory | `_USER_SEARCH_FILTER` | `_ID_ATTR` | `_EMAIL_ATTR` | `_DISPLAY_NAME_ATTR` | Notes |
|---|---|---|---|---|---|
| **OpenLDAP / 389-DS** | `(uid={username})` | `entryUUID` | `mail` | `cn` | `memberOf` needs the OpenLDAP `memberof` overlay (389-DS: the MemberOf plugin); without it the admin-group mapping sees no groups. |
| **Active Directory** | `(sAMAccountName={username})` or `(userPrincipalName={username})` | `objectGUID` | `mail` | `displayName` | `objectGUID` is binary — Inqtrix renders it as the canonical GUID string. `memberOf` is native. |
| **FreeIPA** | `(uid={username})` | `ipaUniqueID` | `mail` | `displayName` | `memberOf` is native. |
| **LLDAP** (test only) | `(uid={username})` | `uid` | `mail` | `cn` | Admin bind DN `cn=admin,ou=people,<base>`; admin group `cn=lldap_admin,ou=groups,<base>`. |

For a copy-paste, run-it-now recipe (compose profile + web-UI user + curl
login) see the [LDAP stack walkthrough](../../examples/webserver_stacks/ldap_stack.md).

### Who is an admin

The first instance admin is established one of two ways, both opt-in:

- **Group-driven** — set `INQTRIX_LDAP_ADMIN_GROUP_DN`; any user whose
  `memberOf` includes that group DN is promoted to instance admin on login. DN
  comparison folds case, RFC 4514 hex escapes (`\20`), and the optional
  whitespace AD emits around RDN separators, so `CN=Admins, OU=…` matches
  `cn=admins,ou=…`.
- **First-login owner** — the very first LDAP user to sign in becomes the
  instance admin when no admin exists yet (a guarded, race-safe promotion).

Group membership only ever **grants** admin on login; **revoking** admin is the
admin API's job (Settings → Administration → Users), so a directory change
never silently strips a UI-granted role or trips the last-admin guard. A
disabled user (disabled via the admin UI) is refused at login even though the
directory bind still succeeds — the mirror is the source of truth for the
Inqtrix-side disable.

## TLS

Use transport security in production. Either:

```dotenv
INQTRIX_LDAP_URL=ldaps://ldap.example.com:636
```

or StartTLS on the cleartext port:

```dotenv
INQTRIX_LDAP_URL=ldap://ldap.example.com:389
INQTRIX_LDAP_START_TLS=true
```

For a private CA, point `INQTRIX_LDAP_CA_CERT` at the PEM bundle. Disabling
`INQTRIX_LDAP_TLS_VALIDATE` is for trusted-network debugging only and is logged
loudly.

## Verify

```bash
# auth_mode reflects the running provider
curl -s http://localhost:8080/health | grep -o '"auth_mode":"[a-z]*"'   # -> "auth_mode":"ldap"

# A correct login establishes the cookie session
curl -s -i -c cookies.txt -X POST http://localhost:8080/api/auth/login/ldap \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"<her-directory-password>"}'        # -> 200, sets cookies

# A wrong password (or unknown user) is an indistinguishable 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8080/api/auth/login/ldap \
  -H 'Content-Type: application/json' -d '{"username":"alice","password":"nope"}'  # -> 401
```

In the web app the lock screen shows a username + password form; on success the
user lands in Research Desk, and a directory admin sees the admin section.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Every login is 401, logs say "LDAP-Service-Bind fehlgeschlagen" | `INQTRIX_LDAP_BIND_DN`/`_PASSWORD` wrong, or the server is unreachable from the api container. |
| Login 401, logs say the search returned 0 or >1 hits | `INQTRIX_LDAP_USER_SEARCH_BASE` / `_FILTER` does not resolve to exactly one entry (tighten the filter). |
| Logins work but nobody is admin | Set `INQTRIX_LDAP_ADMIN_GROUP_DN` to the exact group DN, or use the first-login owner, or promote via Settings → Administration → Users. |
| Sessions drop on restart | You are on the memory backend — set `INQTRIX_STORAGE_BACKEND=postgres`. |
| TLS handshake fails | Private CA not trusted — set `INQTRIX_LDAP_CA_CERT`; do not disable validation in production. |

## Appendix: a disposable test directory (LLDAP)

To try `ldap` mode without your corporate directory, use the throwaway
[LLDAP](https://github.com/lldap/lldap) directory (a tiny LDAP server with a
web UI) wired into the dev compose stack under the `ldap` profile. The
step-by-step recipe — start it, create a user, log in, verify — is the
[LDAP stack walkthrough](../../examples/webserver_stacks/ldap_stack.md):

```bash
podman compose -f deploy/compose/compose.dev.yaml --profile ldap up -d lldap
```

Or run it standalone (local testing only — never a production directory):

```bash
docker run --rm -p 3890:3890 -p 17170:17170 \
  -e LLDAP_LDAP_USER_PASS=admin-password \
  -e LLDAP_LDAP_BASE_DN=dc=example,dc=com \
  lldap/lldap:stable
```

Create a user in the LLDAP web UI (`http://localhost:17170`), then point
Inqtrix at it — LLDAP's admin bind DN is `cn=admin` (not `uid=admin`):

```dotenv
INQTRIX_AUTH_MODE=ldap
INQTRIX_LDAP_URL=ldap://host.docker.internal:3890
INQTRIX_LDAP_BIND_DN=cn=admin,ou=people,dc=example,dc=com
INQTRIX_LDAP_BIND_PASSWORD=admin-password
INQTRIX_LDAP_USER_SEARCH_BASE=ou=people,dc=example,dc=com
INQTRIX_LDAP_USER_SEARCH_FILTER=(uid={username})
INQTRIX_LDAP_ID_ATTR=uid
```

## Related docs

- [Auth modes](../deployment/auth-modes.md) — every mode and how users arrive in each.
- [Create and manage users](create-and-manage-users.md) — owner setup, roles, disabling.
- [Writing a custom auth provider](writing-a-custom-auth-provider.md) — when env config is not enough.
- [Stack quickstart](../getting-started/stack-quickstart.md) — one-command Stack-mode start.
