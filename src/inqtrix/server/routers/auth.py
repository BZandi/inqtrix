"""OIDC BFF routes (``/api/auth/*``) — mounted only in oidc mode.

The browser never sees a token: ``/login`` redirects to the IdP with
PKCE S256 + state + nonce, ``/callback`` exchanges the code
server-side, validates the id_token, and sets the opaque session
cookie; ``/session`` bootstraps the SPA (identity + CSRF token);
``/logout`` destroys the server-side session. A short-lived flow
cookie binds the transaction to the initiating browser — a forged
callback with an attacker's state fails the cookie comparison (login
CSRF). The flow cookie is SameSite=Lax on purpose: Strict cookies are
dropped on the cross-site IdP redirect and would loop every login.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from inqtrix.auth.oidc import OidcExchangeError, make_pkce_pair
from inqtrix.auth.provisioning import apply_admin_grant
from inqtrix.auth.sessions import (
    FLOW_TTL_SECONDS,
    AuthSession,
    LoginFlow,
    new_session_id,
)
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.auth.oidc import OidcAuthProvider

log = logging.getLogger("inqtrix")

_FLOW_COOKIE_SECURE = "__Host-inqtrix_oidc"
_FLOW_COOKIE_DEV = "inqtrix_oidc"


def build_auth_router(provider: "OidcAuthProvider") -> APIRouter:
    """Bind the BFF routes against one cookie-session provider instance.

    OIDC mounts the IdP redirect flow; local (and, later, ldap) mount the
    first-run setup + password login routes. The session / logout / PAT
    routes are identical across modes (local/ldap sessions are the same
    cookie-session ``kind`` as OIDC, ADR-AUTH-3), so password modes route
    through :func:`_build_password_auth_router`. Workspaces are no longer
    bootstrapped on login: workspace management is an explicit instance-admin
    action (``/v1/admin/workspaces``), so the owner setup mints only the
    instance admin and the first workspace is created deliberately.
    """
    if provider.mode in {"local", "ldap"}:
        return _build_password_auth_router(provider)

    async def no_store(response: Response) -> None:
        # The auth surface emits identity facts and the CSRF token —
        # never cacheable, declared at the source instead of relying
        # on every intermediary's defaults.
        response.headers["Cache-Control"] = "no-store"

    router = APIRouter(dependencies=[Depends(no_store)])
    users = provider.users
    flow_cookie = (
        _FLOW_COOKIE_SECURE if provider.secure_cookies else _FLOW_COOKIE_DEV
    )

    def _set_cookie(
        response: Response,
        name: str,
        value: str,
        *,
        http_only: bool,
        max_age: int | None = None,
    ) -> None:
        response.set_cookie(
            name,
            value,
            httponly=http_only,
            secure=provider.secure_cookies,
            samesite="lax",
            path="/",
            max_age=max_age,
        )

    def _clear_cookie(response: Response, name: str) -> None:
        response.delete_cookie(
            name, path="/", secure=provider.secure_cookies
        )

    @router.get("/api/auth/login")
    async def login(request: Request, next: str = "/"):
        """Start one authorization-code transaction and redirect."""
        next_path = next if next.startswith("/") and not next.startswith("//") else "/"
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        code_verifier, code_challenge = make_pkce_pair()
        await provider.flows.put(
            LoginFlow(
                state=state,
                code_verifier=code_verifier,
                nonce=nonce,
                next_path=next_path,
                expires_at=time.time() + FLOW_TTL_SECONDS,
            )
        )
        url = await provider.client.authorization_url(
            state=state, nonce=nonce, code_challenge=code_challenge
        )
        response = RedirectResponse(url, status_code=302)
        _set_cookie(
            response,
            flow_cookie,
            state,
            http_only=True,
            max_age=int(FLOW_TTL_SECONDS),
        )
        return response

    @router.get("/api/auth/callback")
    async def callback(request: Request):
        """Exchange the code, validate identity, establish the session."""
        error = request.query_params.get("error")
        if error:
            log.warning("OIDC-Callback meldet Fehler: %s", error)
            return error_response(
                400,
                f"Anmeldung beim Identity-Provider fehlgeschlagen ({error})",
                "oidc_error",
            )
        code = request.query_params.get("code", "")
        state = request.query_params.get("state", "")
        if not code or not state:
            return error_response(
                400, "Login-Antwort unvollstaendig", "oidc_error"
            )
        if request.cookies.get(flow_cookie, "") != state:
            # Login-CSRF defense: the browser completing the flow must
            # be the browser that started it.
            log.warning(
                "OIDC-Callback ohne passendes Flow-Cookie verworfen."
            )
            return error_response(
                400, "Login-Transaktion ungueltig", "oidc_error"
            )
        flow = await provider.flows.consume(state)
        if flow is None:
            return error_response(
                400,
                "Login-Transaktion abgelaufen oder bereits verwendet",
                "oidc_error",
            )
        try:
            tokens = await provider.client.exchange_code(
                code=code, code_verifier=flow.code_verifier
            )
            claims = await provider.client.validate_id_token(
                tokens.get("id_token", ""), nonce=flow.nonce
            )
            claims = await _maybe_merge_userinfo(provider, claims, tokens)
            subject, email, display_name, groups = (
                provider.resolve_identity(claims)
            )
        except OidcExchangeError as exc:
            log.warning("OIDC-Login abgelehnt: %s", exc)
            return error_response(403, str(exc), "oidc_error")

        issuer = str(claims.get("iss", ""))
        if provider.registration_gate is not None:
            # Admission BEFORE any user record or session exists: a
            # rejected stranger leaves no trace beyond the audit entry.
            from inqtrix.auth.invitations import RegistrationDenied

            try:
                await provider.registration_gate.admit(
                    tenant_id="default",
                    issuer=issuer,
                    sub=subject,
                    email=email or "",
                )
            except RegistrationDenied as exc:
                log.warning(
                    "Registrierung abgelehnt: sub=%s", subject
                )
                return error_response(403, str(exc), "registration_denied")
        if users is not None:
            await users.record_login(
                tenant_id="default",
                issuer=issuer,
                subject=subject,
                email=email or "",
                email_verified=claims.get("email_verified") is True,
                display_name=display_name,
            )
            # Admin elevation from IdP claims (configured admin roles or
            # groups), with the same grant-only semantics as the LDAP
            # admin-group path; on a fresh instance the first login still
            # bootstraps the owner.
            await apply_admin_grant(
                users,
                tenant_id="default",
                issuer=issuer,
                subject=subject,
                is_admin=provider.map_admin(claims, groups),
                first_login_owner=True,
            )
        session = AuthSession(
            id=new_session_id(),
            sub=subject,
            issuer=issuer,
            email=email,
            display_name=display_name,
            groups=groups,
            csrf_random=secrets.token_hex(16),
            created_at=time.time(),
            expires_at=time.time() + provider.session_max_age_seconds,
        )
        await provider.sessions.create(session)
        log.info(
            "OIDC-Login erfolgreich: sub=%s session=%s",
            subject,
            session.id[:8],
        )
        response = RedirectResponse(flow.next_path, status_code=303)
        _set_cookie(
            response, provider.session_cookie, session.id, http_only=True
        )
        from inqtrix.auth.oidc import make_csrf_token

        _set_cookie(
            response,
            provider.csrf_cookie,
            make_csrf_token(
                provider.session_secret, session.id, session.csrf_random
            ),
            http_only=False,
        )
        _clear_cookie(response, flow_cookie)
        return response

    @router.get("/api/auth/session")
    async def session_info(request: Request):
        """SPA bootstrap: identity facts plus the CSRF token."""
        return await provider.session_payload(request)

    @router.post("/api/auth/logout")
    async def logout(request: Request):
        """Destroy the server-side session (CSRF-protected)."""
        principal_dep = provider.build_principal_dependency()
        principal = await principal_dep(request)
        if principal.session_id:
            await provider.sessions.delete(principal.session_id)
        response = JSONResponse({"logged_out": True})
        _clear_cookie(response, provider.session_cookie)
        _clear_cookie(response, provider.csrf_cookie)
        return response

    if provider.pat_service is not None:
        _register_token_routes(router, provider)

    return router


_MIN_PASSWORD_LEN = 12

_RATE_LIMITED_MSG = (
    "Zu viele fehlgeschlagene Anmeldeversuche. Bitte spaeter erneut versuchen."
)


def _login_throttle_key(mode: str, identifier: str, request: Request) -> str:
    """Brute-force throttle key: ``mode:lower(identifier):client_ip``."""
    from inqtrix.auth.ratelimit import client_ip

    return f"{mode}:{identifier.strip().lower()}:{client_ip(request)}"


async def _read_json_object(request: Request):
    """Parse a JSON object body, or return ``(None, error_response)``."""
    try:
        body = await request.json()
    except Exception:
        return None, error_response(
            400, "Ungueltiger JSON-Body", "invalid_request_error"
        )
    if not isinstance(body, dict):
        return None, error_response(
            400, "Ungueltiger JSON-Body", "invalid_request_error"
        )
    return body, None


def _build_password_auth_router(provider) -> APIRouter:
    """Routes for password-based cookie-session modes (local; ldap later).

    Mounts the first-run owner setup and the password login route, plus the
    mode-agnostic session / logout / PAT routes. Sessions are minted exactly
    like the OIDC callback mints them, so the inherited per-request path
    (Bearer -> PAT, cookie -> session, CSRF) resolves them unchanged.
    """
    import secrets as _secrets
    import time as _time

    from inqtrix.auth.oidc import make_csrf_token

    async def no_store(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    router = APIRouter(dependencies=[Depends(no_store)])
    users = provider.users

    def _set_cookie(
        response: Response, name: str, value: str, *, http_only: bool
    ) -> None:
        response.set_cookie(
            name,
            value,
            httponly=http_only,
            secure=provider.secure_cookies,
            samesite="lax",
            path="/",
        )

    def _clear_cookie(response: Response, name: str) -> None:
        response.delete_cookie(
            name, path="/", secure=provider.secure_cookies
        )

    async def _mint_session(
        *,
        subject: str,
        issuer: str,
        email: str | None,
        display_name: str | None,
        status_code: int = 200,
    ) -> JSONResponse:
        session = AuthSession(
            id=new_session_id(),
            sub=subject,
            issuer=issuer,
            email=email,
            display_name=display_name,
            groups=(),
            csrf_random=_secrets.token_hex(16),
            created_at=_time.time(),
            expires_at=_time.time() + provider.session_max_age_seconds,
        )
        await provider.sessions.create(session)
        response = JSONResponse({"authenticated": True}, status_code=status_code)
        _set_cookie(
            response, provider.session_cookie, session.id, http_only=True
        )
        _set_cookie(
            response,
            provider.csrf_cookie,
            make_csrf_token(
                provider.session_secret, session.id, session.csrf_random
            ),
            http_only=False,
        )
        return response

    if provider.mode == "local":
        from inqtrix.auth.credentials import (
            LOCAL_ISSUER,
            CredentialError,
            LocalCredential,
            hash_password,
            new_subject,
            verify_password,
        )

        def _validate_email_password(body: dict):
            email = str(body.get("email", "")).strip()
            password = str(body.get("password", ""))
            if "@" not in email or len(email) > 320:
                return None, error_response(
                    400, "Ungueltige E-Mail-Adresse", "invalid_request_error"
                )
            if len(password) < _MIN_PASSWORD_LEN:
                return None, error_response(
                    400,
                    f"Passwort muss mindestens {_MIN_PASSWORD_LEN} Zeichen lang sein",
                    "invalid_request_error",
                )
            return (email, password), None

        @router.get("/api/setup/status")
        async def setup_status():
            """First-run gate: whether the owner account still needs creating."""
            count = await provider.credentials.count(tenant_id="default")
            return {"needs_owner": count == 0}

        @router.post("/api/setup/owner", status_code=201)
        async def setup_owner(request: Request):
            """Create the first owner exactly once, then lock permanently."""
            body, error = await _read_json_object(request)
            if error is not None:
                return error
            parsed, error = _validate_email_password(body)
            if error is not None:
                return error
            email, password = parsed
            display_name = str(body.get("display_name", "")).strip() or email
            credential = LocalCredential(
                subject=new_subject(),
                email=email,
                password_hash=hash_password(password),
                display_name=display_name,
                created_at=_time.time(),
            )
            created = await provider.credentials.create(
                credential, tenant_id="default", allow_first_only=True
            )
            if not created:
                # Idempotent permanent lock: the owner exists already.
                return error_response(
                    409, "Owner ist bereits eingerichtet", "setup_locked"
                )
            if users is not None:
                await users.record_login(
                    tenant_id="default",
                    issuer=LOCAL_ISSUER,
                    subject=credential.subject,
                    email=email,
                    email_verified=True,
                    display_name=display_name,
                )
                # The first owner is the instance admin, unconditionally.
                await users.set_instance_role(
                    tenant_id="default",
                    issuer=LOCAL_ISSUER,
                    subject=credential.subject,
                    role="admin",
                )
            log.info("Lokaler Owner angelegt: sub=%s", credential.subject)
            return await _mint_session(
                subject=credential.subject,
                issuer=LOCAL_ISSUER,
                email=email,
                display_name=display_name,
                status_code=201,
            )

        @router.post("/api/auth/login/local")
        async def login_local(request: Request):
            """Email/password login -> the same session+CSRF cookies as OIDC."""
            body, error = await _read_json_object(request)
            if error is not None:
                return error
            # Accept "identifier" (shared with the ldap form) or "email".
            email = str(
                body.get("identifier") or body.get("email") or ""
            ).strip()
            password = str(body.get("password", ""))
            limiter = getattr(provider, "login_rate_limiter", None)
            throttle_key = _login_throttle_key("local", email, request)
            if limiter is not None and limiter.locked(throttle_key):
                return error_response(429, _RATE_LIMITED_MSG, "rate_limited")
            try:
                credential = await provider.authenticator.authenticate(
                    email, password
                )
            except CredentialError:
                if limiter is not None:
                    limiter.record_failure(throttle_key)
                return error_response(
                    401, "Ungueltige Anmeldedaten", "unauthorized"
                )
            if limiter is not None:
                limiter.reset(throttle_key)
            if users is not None:
                await users.record_login(
                    tenant_id="default",
                    issuer=LOCAL_ISSUER,
                    subject=credential.subject,
                    email=credential.email,
                    email_verified=True,
                    display_name=credential.display_name,
                )
            return await _mint_session(
                subject=credential.subject,
                issuer=LOCAL_ISSUER,
                email=credential.email,
                display_name=credential.display_name,
            )

        @router.post("/api/auth/password")
        async def change_password(request: Request):
            """Self-service password change for the signed-in local user.

            Session-gated (the caller must be an authenticated local session)
            and re-verifies the CURRENT password before replacing the hash —
            so a hijacked-but-not-authenticated request cannot change it. CSRF
            is enforced by the principal dependency on this unsafe method.
            """
            resolved, error = await _session_principal_for_tokens(
                provider, request
            )
            if error is not None:
                return error
            _principal, session = resolved
            if session.issuer != LOCAL_ISSUER:
                return error_response(
                    400,
                    "Passwortwechsel ist nur fuer lokale Konten verfuegbar",
                    "invalid_request_error",
                )
            body, parse_error = await _read_json_object(request)
            if parse_error is not None:
                return parse_error
            current = str(body.get("current_password", ""))
            new_password = str(body.get("new_password", ""))
            if len(new_password) < _MIN_PASSWORD_LEN:
                return error_response(
                    400,
                    f"Passwort muss mindestens {_MIN_PASSWORD_LEN} Zeichen lang sein",
                    "invalid_request_error",
                )
            credential = await provider.credentials.get_by_subject(
                tenant_id="default", subject=session.sub
            )
            if credential is None or not verify_password(
                credential.password_hash, current
            ):
                return error_response(
                    401, "Aktuelles Passwort ist falsch", "unauthorized"
                )
            await provider.credentials.set_password(
                tenant_id="default",
                subject=session.sub,
                password_hash=hash_password(new_password),
            )
            return JSONResponse({"changed": True})

    if provider.mode == "ldap":
        from starlette.concurrency import run_in_threadpool

        from inqtrix.auth.ldap import LDAP_ISSUER, LdapError

        @router.post("/api/auth/login/ldap")
        async def login_ldap(request: Request):
            """LDAP bind login -> the same session+CSRF cookies as OIDC."""
            body, error = await _read_json_object(request)
            if error is not None:
                return error
            username = str(
                body.get("identifier") or body.get("username") or ""
            ).strip()
            password = str(body.get("password", ""))
            limiter = getattr(provider, "login_rate_limiter", None)
            throttle_key = _login_throttle_key("ldap", username, request)
            if limiter is not None and limiter.locked(throttle_key):
                return error_response(429, _RATE_LIMITED_MSG, "rate_limited")
            try:
                # ldap3 is blocking — keep it off the event loop.
                identity = await run_in_threadpool(
                    provider.ldap_client.authenticate, username, password
                )
            except LdapError:
                if limiter is not None:
                    limiter.record_failure(throttle_key)
                return error_response(
                    401, "Ungueltige Anmeldedaten", "unauthorized"
                )
            if limiter is not None:
                limiter.reset(throttle_key)
            if users is not None:
                # Disabled enforcement parity with local (the authenticator
                # checks credential.disabled_at) and oidc (the registration
                # gate checks the mirror): the directory itself has no
                # knowledge of an Inqtrix-side disable, so the mirror is the
                # source of truth. Reject BEFORE recording the login or
                # (re-)granting any role, so a disabled LDAP user can neither
                # re-establish a session nor be re-promoted to admin.
                existing = await users.find_user(
                    tenant_id="default",
                    issuer=LDAP_ISSUER,
                    subject=identity.subject,
                )
                if existing is not None and existing.disabled_at is not None:
                    return error_response(
                        401, "Ungueltige Anmeldedaten", "unauthorized"
                    )
                await users.record_login(
                    tenant_id="default",
                    issuer=LDAP_ISSUER,
                    subject=identity.subject,
                    email=identity.email,
                    email_verified=True,
                    display_name=identity.display_name,
                )
                # Admin-group membership GRANTS instance-admin (grant-only,
                # never demotes — see apply_admin_grant); on a fresh instance
                # the first LDAP login may bootstrap the owner.
                await apply_admin_grant(
                    users,
                    tenant_id="default",
                    issuer=LDAP_ISSUER,
                    subject=identity.subject,
                    is_admin=identity.is_admin,
                    first_login_owner=provider.first_login_owner,
                )
            return await _mint_session(
                subject=identity.subject,
                issuer=LDAP_ISSUER,
                email=identity.email,
                display_name=identity.display_name,
            )

    @router.get("/api/auth/session")
    async def session_info(request: Request):
        """SPA bootstrap: identity facts plus the CSRF token."""
        return await provider.session_payload(request)

    @router.post("/api/auth/logout")
    async def logout(request: Request):
        """Destroy the server-side session (CSRF-protected)."""
        principal_dep = provider.build_principal_dependency()
        principal = await principal_dep(request)
        if principal.session_id:
            await provider.sessions.delete(principal.session_id)
        response = JSONResponse({"logged_out": True})
        _clear_cookie(response, provider.session_cookie)
        _clear_cookie(response, provider.csrf_cookie)
        return response

    if provider.pat_service is not None:
        _register_token_routes(router, provider)

    return router


async def _session_principal_for_tokens(
    provider: "OidcAuthProvider", request: Request
):
    """Token management is SESSION-only: a PAT can never mint or
    revoke PATs (a leaked token must not be able to extend itself or
    cut off its siblings)."""
    principal_dep = provider.build_principal_dependency()
    principal = await principal_dep(request)
    if principal.kind != "oidc_session":
        return None, error_response(
            403,
            "Zugriffstokens koennen nur in einer Browser-Sitzung "
            "verwaltet werden.",
            "forbidden",
        )
    session = await provider.sessions.get(principal.session_id or "")
    if session is None:
        return None, error_response(401, "Sitzung abgelaufen", "unauthorized")
    return (principal, session), None


def _token_payload(record) -> dict:
    """List/create response shape — never the hash, never the secret."""
    return {
        "token_id": record.token_id,
        "name": record.name,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
        "last_used_at": record.last_used_at,
        "scopes": list(record.scopes),
    }


def _register_token_routes(
    router: APIRouter, provider: "OidcAuthProvider"
) -> None:
    """Personal-access-token management under the BFF surface.

    Lives on the auth router on purpose: the ``no_store`` dependency
    already covers it, and the one-time plaintext emission must never
    be cacheable.
    """
    from inqtrix.auth.pat import PatLimitExceeded

    @router.post("/api/auth/tokens", status_code=201)
    async def create_token(request: Request):
        resolved, error = await _session_principal_for_tokens(
            provider, request
        )
        if error is not None:
            return error
        _principal, session = resolved
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        if not isinstance(body, dict):
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        name = str(body.get("name", "")).strip()
        if not name or len(name) > 120:
            return error_response(
                400,
                "Feld 'name' muss 1 bis 120 Zeichen lang sein",
                "invalid_request_error",
            )
        expires_in_days = body.get("expires_in_days")
        if expires_in_days is not None and (
            not isinstance(expires_in_days, int) or expires_in_days < 1
        ):
            return error_response(
                400,
                "Feld 'expires_in_days' muss eine positive Ganzzahl sein",
                "invalid_request_error",
            )
        try:
            minted = await provider.pat_service.create_token(
                tenant_id="default",
                owner_issuer=session.issuer,
                owner_sub=session.sub,
                name=name,
                expires_in_days=expires_in_days,
            )
        except PatLimitExceeded as exc:
            return error_response(
                409,
                f"Maximale Anzahl an Zugriffstokens erreicht ({exc})",
                "pat_limit",
            )
        payload = _token_payload(minted.record)
        # The ONE plaintext emission — shown here, never stored,
        # never logged, never retrievable again.
        payload["token"] = minted.plaintext
        return JSONResponse(payload, status_code=201)

    @router.get("/api/auth/tokens")
    async def list_tokens(request: Request):
        resolved, error = await _session_principal_for_tokens(
            provider, request
        )
        if error is not None:
            return error
        _principal, session = resolved
        tokens = await provider.pat_service.list_tokens(
            tenant_id="default",
            owner_issuer=session.issuer,
            owner_sub=session.sub,
        )
        return {"tokens": [_token_payload(record) for record in tokens]}

    @router.delete("/api/auth/tokens/{token_id}")
    async def revoke_token(token_id: str, request: Request):
        resolved, error = await _session_principal_for_tokens(
            provider, request
        )
        if error is not None:
            return error
        _principal, session = resolved
        revoked = await provider.pat_service.revoke_token(
            tenant_id="default",
            token_id=token_id,
            owner_issuer=session.issuer,
            owner_sub=session.sub,
        )
        if not revoked:
            # Foreign or unknown ids are indistinguishable — denial
            # hidden behind absence, consistent with the permission
            # layer's 404-not-403 convention.
            return error_response(404, "Token nicht gefunden", "not_found")
        return {"revoked": True}


async def _maybe_merge_userinfo(
    provider: "OidcAuthProvider", claims: dict, tokens: dict
) -> dict:
    """Merge userinfo claims when the id_token is thin (opt-in)."""
    from inqtrix.auth.oidc import claim_path

    needed = [
        provider.username_claim,
        provider.email_claim,
        provider.groups_claim,
    ]
    if provider.admin_elevation_enabled:
        # Only pull userinfo for the roles claim when admin-from-claims is
        # configured; otherwise the roles value changes nothing and the
        # extra round-trip would be pure waste.
        needed.append(provider.roles_claim)
    if not provider.userinfo_fallback or all(
        claim_path(claims, path) is not None for path in needed
    ):
        return claims
    access_token = tokens.get("access_token", "")
    if not access_token:
        return claims
    extra = await provider.client.userinfo(access_token)
    if extra.get("sub") not in (None, claims.get("sub")):
        # OIDC Core: userinfo sub MUST match the id_token sub.
        log.warning(
            "Userinfo-Subject weicht vom id_token ab — Antwort verworfen."
        )
        return claims
    # id_token claims win over userinfo — but a null claim is an
    # ABSENT claim (several IdPs emit explicit nulls), not a value
    # that may clobber the userinfo answer.
    merged = dict(extra)
    merged.update(
        {key: value for key, value in claims.items() if value is not None}
    )
    return merged
