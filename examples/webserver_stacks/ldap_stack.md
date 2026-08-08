# LDAP stack walkthrough (bind against a directory with LLDAP)

> Files: `deploy/compose/compose.stack.yaml`, `deploy/compose/compose.dev-ports.yaml`, `src/inqtrix/auth/ldap.py`, `src/inqtrix/server/routers/auth.py`
>
> Every `INQTRIX_LDAP_*` / `INQTRIX_AUTH_MODE` variable (defaults, allowed values) is defined in [`docs/configuration/settings-and-env.md`](../../docs/configuration/settings-and-env.md), the single source of truth for env vars; this page is the walkthrough.

## Scope

End-to-end recipe for `INQTRIX_AUTH_MODE=ldap`: start a throwaway
[LLDAP](https://github.com/lldap/lldap) directory from the dev compose
stack, create a demo user in its web UI, point any webserver-stack
script (or `python -m inqtrix`) at it, and verify a bind login over
curl. LLDAP is the recommended dev directory, never an architecture
component — Inqtrix binds to **any** RFC 4511 directory (OpenLDAP,
Active Directory, FreeIPA, 389-DS); only the search filter and attribute
names change, all as configuration (see the per-directory cheat sheet in
[Connect to an existing LDAP](../../docs/how-to/connect-to-existing-ldap.md)).

Inqtrix never runs an LDAP server and never stores the password: it does
the portable **search-then-bind** (bind as a read-only service account,
search for the user DN, then re-bind as that DN with the user's
password). LDAP logins reuse the exact session-cookie + CSRF + PAT
machinery as OIDC; only the login transport differs.

## 1. Start the directory

LLDAP is gated behind the compose profile `ldap`; without that profile Compose
does not start the bundled directory and leaves the configured authentication
mode unchanged. Start from the named local pair described in the
[local infrastructure guide](../../docs/development/local-infrastructure.md#start--stop). Keep the
LDAP topology in `deploy/.env.stack.local`, and add these credentials only to
`deploy/.env.stack.secrets.local`:

```dotenv
INQTRIX_LDAP_BIND_PASSWORD=replace-with-a-local-bind-password
INQTRIX_LLDAP_JWT_SECRET=replace-with-an-independent-jwt-secret
INQTRIX_LLDAP_KEY_SEED=replace-with-an-independent-key-seed
```

```bash
podman compose \
  -f deploy/compose/compose.stack.yaml \
  -f deploy/compose/compose.dev-ports.yaml \
  --env-file deploy/.env.stack.secrets.local \
  --env-file deploy/.env.stack.local \
  --profile ldap up -d lldap
```

(`docker compose` works identically.) It publishes two loopback ports:
LDAP on `3890` (the host-side API binds here) and a setup web UI on
`17170`. The compose service bootstraps the base DN `dc=example,dc=com`
and the admin account `cn=admin,ou=people,dc=example,dc=com` with the
password from `INQTRIX_LDAP_BIND_PASSWORD` (default
is intentionally absent; use the value from the selected secret file).

## 2. Create a demo user

Open `http://127.0.0.1:17170` and sign in as `admin` with the selected
`INQTRIX_LDAP_BIND_PASSWORD`. Create a user — e.g. user id `bob`, email
`bob@example.com`, and set a password. (Optionally add `bob` to the
built-in `lldap_admin` group to exercise the admin-group mapping in
step 4.)

## 3. Configure and start the server

Every webserver-stack script builds its auth provider through
`create_app(...)`, so LDAP needs env vars only — no code change:

| Variable | Dev value | Notes |
|---|---|---|
| `INQTRIX_AUTH_MODE` | `ldap` | Explicit mode; missing connection settings fail loud at startup. |
| `INQTRIX_LDAP_URL` | `ldap://127.0.0.1:3890` | `ldaps://host:636` or add `INQTRIX_LDAP_START_TLS=true` for real directories — never bind plaintext over an untrusted network. |
| `INQTRIX_LDAP_BIND_DN` | `cn=admin,ou=people,dc=example,dc=com` | Read-only service account that can search users. |
| `INQTRIX_LDAP_BIND_PASSWORD` | the value selected above | Same value feeds the LLDAP container and host-side API; keep it in the private companion file and export it only into the API process. |
| `INQTRIX_LDAP_USER_SEARCH_BASE` | `ou=people,dc=example,dc=com` | Subtree searched for the login. |
| `INQTRIX_LDAP_USER_SEARCH_FILTER` | `(uid={username})` | `{username}` is RFC 4515-escaped before the search (LDAP-injection defense). AD uses `(sAMAccountName={username})`. |
| `INQTRIX_LDAP_ID_ATTR` | `uid` | Stable subject id. LLDAP keys on `uid`; OpenLDAP uses `entryUUID`, AD `objectGUID` (binary — rendered as a canonical GUID). Prefer a non-reassignable attribute in production. |
| `INQTRIX_LDAP_ADMIN_GROUP_DN` | `cn=lldap_admin,ou=groups,dc=example,dc=com` | Optional; members are granted instance-admin on login (grant-only). |
| `INQTRIX_SESSION_SECRET` | any random string | CSRF-token derivation; required in ldap mode. |
| `INQTRIX_PAT_PEPPER` | any random string | HMAC pepper for personal access tokens; required in ldap mode. |
| `INQTRIX_OIDC_INSECURE_DEV_COOKIES` | `true` | Drops the `Secure` flag and `__Host-` prefix so the session cookie works over plain `http://127.0.0.1`. NEVER in production; activation is loudly logged. |

`mail` (email) and `cn` (display name) are the Inqtrix defaults and match
LLDAP, so no override is needed.

Install the project once with either `uv sync --extra dev` or a standard
environment created with `python -m venv .venv` followed by
`python -m pip install -e ".[dev]"`. The following block uses plain Python;
uv users can replace its final command with
`uv run python examples/webserver_stacks/anthropic_perplexity.py`.

```bash
INQTRIX_AUTH_MODE=ldap \
INQTRIX_LDAP_URL=ldap://127.0.0.1:3890 \
INQTRIX_LDAP_BIND_DN=cn=admin,ou=people,dc=example,dc=com \
INQTRIX_LDAP_BIND_PASSWORD=replace-with-the-same-local-bind-password \
INQTRIX_LDAP_USER_SEARCH_BASE=ou=people,dc=example,dc=com \
INQTRIX_LDAP_USER_SEARCH_FILTER='(uid={username})' \
INQTRIX_LDAP_ID_ATTR=uid \
INQTRIX_LDAP_ADMIN_GROUP_DN=cn=lldap_admin,ou=groups,dc=example,dc=com \
INQTRIX_SESSION_SECRET=dev-session-secret-change-me \
INQTRIX_PAT_PEPPER=dev-pat-pepper-change-me \
INQTRIX_OIDC_INSECURE_DEV_COOKIES=true \
python examples/webserver_stacks/anthropic_perplexity.py
```

Any other stack script accepts the same variables. The env-only server starts
with `uv run python -m inqtrix` or, after the pip installation,
`python -m inqtrix`. The startup log line records `auth_mode=ldap`. A missing
connection setting (URL, bind DN/password, search base) fails loud at startup,
naming the variable.

## 4. Verify over curl

Discover the active mode and login method without any credential
(always available, even pre-login):

```bash
curl http://127.0.0.1:5100/api/auth/config
# {"auth_mode": "ldap", "auth_required": true,
#  "login_methods": [{"kind": "password", "label": "LDAP", "identifier": "username"}],
#  "supports_logout": true, "csrf_required": true, "csrf_header": "X-CSRF-Token", ...}
```

Bind login — a JSON `POST` with the username and password (search-then-
bind happens server-side). On success it sets the HttpOnly session cookie
and the JS-readable CSRF cookie:

```bash
curl -i -X POST http://127.0.0.1:5100/api/auth/login/ldap \
  -H 'Content-Type: application/json' \
  -d '{"identifier": "bob", "password": "<the password you set>"}'
# HTTP/1.1 200 ... set-cookie: inqtrix_session=...; set-cookie: inqtrix_csrf=...
```

A wrong password (or an unknown user, or an ambiguous search) returns a
uniform `401 Ungueltige Anmeldedaten` — the directory never reveals which
of the three it was. Repeated failures are throttled
(`INQTRIX_LOGIN_RATE_LIMIT_*`). Reuse the session cookie to confirm the
identity:

```bash
curl http://127.0.0.1:5100/api/auth/session \
  -H 'Cookie: inqtrix_session=<value from the login response>'
# {"authenticated": true, "sub": "bob", "email": "bob@example.com",
#  "display_name": "bob", "role": "admin"|"user", "csrf_token": "..."}
```

If you added `bob` to `lldap_admin`, `role` is `admin` (the admin-group
mapping). State-changing routes then require that `csrf_token` in the
`X-CSRF-Token` header (OWASP signed double-submit), exactly as in OIDC
mode.

## Stack mode (bundled LLDAP)

The Stack-mode compose ships the same throwaway LLDAP under the `ldap`
profile, but **internal** to the compose network (only the setup web UI
is published):

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets \
  --env-file deploy/.env.stack \
  --profile ldap up -d
```

Set `INQTRIX_AUTH_MODE=ldap` in `deploy/.env.stack` and point Inqtrix at the
service by name (no host port for LDAP — the API container reaches it over the
network). Keep only visible directory configuration in this file:

```dotenv
INQTRIX_AUTH_MODE=ldap
INQTRIX_LDAP_URL=ldap://lldap:3890
INQTRIX_LDAP_BIND_DN=cn=admin,ou=people,dc=example,dc=com
INQTRIX_LDAP_USER_SEARCH_BASE=ou=people,dc=example,dc=com
INQTRIX_LDAP_USER_SEARCH_FILTER=(uid={username})
INQTRIX_LDAP_ID_ATTR=uid
```

Put the corresponding credential in `deploy/.env.stack.secrets`:

```dotenv
INQTRIX_LDAP_BIND_PASSWORD=replace-with-the-lldap-bind-password
```

Open the setup UI on `http://127.0.0.1:17170` to create users, then sign
in at `http://localhost:8080`.

## Pointing at a real directory instead

Skip the LLDAP container and set `INQTRIX_LDAP_*` at your own directory.
The only per-directory differences are the **login filter** and the
**id/email/display attributes** — all configuration, no code. The cheat
sheet (Active Directory `sAMAccountName`/`objectGUID`, OpenLDAP
`uid`/`entryUUID` + the `memberof` overlay note, FreeIPA) is in
[Connect to an existing LDAP](../../docs/how-to/connect-to-existing-ldap.md).
For enterprises that already front their directory with an IdP, the
OIDC-bridge path (Dex/Keycloak in front of LDAP/AD, Inqtrix speaks OIDC)
is usually preferable — see [OIDC stack](oidc_stack.md).

## Related docs

- [Connect to an existing LDAP](../../docs/how-to/connect-to-existing-ldap.md) — full config reference + per-IdP cheat sheet
- [Auth modes](../../docs/deployment/auth-modes.md) — none / apikey / oidc / local / ldap
- [`webserver_stacks/README.md`](README.md) — env-var matrix, logging, TLS
