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
import uuid
from typing import TYPE_CHECKING, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from inqtrix.auth.log_redaction import (
    pseudonymous_log_reference,
    stable_pseudonym,
)
from inqtrix.auth.lifecycle import LoginCommand, UserDisabledError
from inqtrix.auth.oidc import OidcExchangeError, make_pkce_pair
from inqtrix.auth.sessions import (
    FLOW_TTL_SECONDS,
    AuthSession,
    LoginFlow,
    new_session_id,
)
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.auth.oidc import OidcAuthProvider
    from inqtrix.auth.permissions import AuditSink
    from inqtrix.auth.principal import Principal
    from inqtrix.services.audit_service import AuditService

log = logging.getLogger("inqtrix")


def _login_audit(audit: "AuditSink | None"):
    """Fail-safe audit writer for the AuthN catalog rows, or ``None``.

    Login failures/lockouts are lifecycle telemetry: they must never
    turn a 401 into a 500, so they go through the warning-loud
    AuditService instead of the fail-loud direct sink path.
    """
    if audit is None:
        return None
    from inqtrix.services.audit_service import AuditService

    return AuditService(audit)


async def _destroy_session_with_audit(
    provider: "OidcAuthProvider",
    auditor: "AuditService | None",
    principal: "Principal",
) -> None:
    """Delete one cookie session without retaining its bearer credential."""
    session_id = principal.session_id
    if not session_id:
        return
    await provider.sessions.delete(session_id)
    if auditor is not None and principal.user_id is not None:
        await auditor.record_event(
            tenant_id=principal.tenant_id or "default",
            action="auth.logout",
            resource_type="session",
            resource_id=stable_pseudonym("ses", session_id),
            actor_user_id=principal.user_id,
        )

_FLOW_COOKIE_SECURE = "__Host-inqtrix_oidc"
_FLOW_COOKIE_DEV = "inqtrix_oidc"


def build_auth_router(
    provider: "OidcAuthProvider",
    principal_dependency: Callable[..., "Principal"] | None = None,
    audit: "AuditSink | None" = None,
) -> APIRouter:
    """Bind the BFF routes against one cookie-session provider instance.

    OIDC mounts the IdP redirect flow; local (and, later, ldap) mount the
    first-run setup + password login routes. The session / logout / PAT
    routes are identical across modes (local/ldap sessions are the same
    cookie-session ``kind`` as OIDC), so password modes route
    through :func:`_build_password_auth_router`. Workspaces are no longer
    bootstrapped on login: workspace management is an explicit instance-admin
    action (``/v1/admin/workspaces``), so the owner setup mints only the
    instance admin and the first workspace is created deliberately.
    """
    if principal_dependency is None:
        from inqtrix.auth.principal_generation import (
            bind_principal_generation,
        )

        principal_dependency = bind_principal_generation(
            provider.build_principal_dependency()
        )
    if provider.mode in {"local", "ldap"}:
        return _build_password_auth_router(
            provider, principal_dependency, audit=audit
        )

    async def no_store(response: Response) -> None:
        # The auth surface emits identity facts and the CSRF token —
        # never cacheable, declared at the source instead of relying
        # on every intermediary's defaults.
        response.headers["Cache-Control"] = "no-store"

    router = APIRouter(dependencies=[Depends(no_store)])
    users = provider.users
    auditor = _login_audit(audit)
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
            log.warning("OIDC-Callback meldet einen Provider-Fehler")
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
            log.warning(
                "OIDC-Login abgelehnt (error_type=%s)",
                type(exc).__name__,
            )
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
                    "Registrierung abgelehnt: kind=oidc_session reason=invite"
                )
                return error_response(403, str(exc), "registration_denied")
        if users is None:
            return error_response(
                503, "Nutzerverzeichnis ist nicht verfuegbar", "server_error"
            )
        session = AuthSession(
            id=new_session_id(),
            user_id=uuid.uuid4(),
            issuer=issuer,
            subject=subject,
            email=email,
            display_name=display_name,
            groups=groups,
            csrf_random=secrets.token_hex(16),
            created_at=time.time(),
            expires_at=time.time() + provider.session_max_age_seconds,
        )
        if provider.lifecycle is None:
            return error_response(
                503, "Nutzerverwaltung ist nicht verfuegbar", "server_error"
            )
        from inqtrix.auth.invitations import RegistrationDenied

        try:
            admitted_user = await provider.lifecycle.provision_login(
                LoginCommand(
                    tenant_id="default",
                    issuer=issuer,
                    subject=subject,
                    email=email or "",
                    email_verified=claims.get("email_verified") is True,
                    display_name=display_name,
                    session=session,
                    is_admin=provider.map_admin(claims, groups),
                    first_login_owner=True,
                    invitation_required=bool(
                        provider.registration_gate is not None
                        and provider.registration_gate.registration == "invite"
                    ),
                )
            )
        except RegistrationDenied as exc:
            log.warning(
                "Registrierung atomar abgelehnt: "
                "kind=oidc_session reason=invite"
            )
            return error_response(403, str(exc), "registration_denied")
        except UserDisabledError:
            log.warning(
                "OIDC-Login fuer deaktiviertes Konto abgelehnt: "
                "kind=oidc_session reason=disabled"
            )
            return error_response(403, "Konto ist deaktiviert.", "registration_denied")
        log.info(
            "OIDC-Login erfolgreich: actor_ref=%s kind=oidc_session",
            pseudonymous_log_reference("usr", admitted_user.user_id),
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
    async def session_info(request: Request, response: Response):
        """SPA bootstrap and repair the readable double-submit cookie.

        The opaque session remains valid across a deliberate
        ``session_secret`` rotation, while a token minted with the previous
        secret does not.  Returning the freshly derived value only in JSON
        leaves the SPA's cookie/header pair stale, so every subsequent
        mutation (including logout) keeps failing.  A successful bootstrap
        therefore refreshes the cookie from the same value it returns.
        """
        payload = await provider.session_payload(request)
        csrf_token = payload.get("csrf_token")
        if payload.get("authenticated") is True and isinstance(csrf_token, str):
            _set_cookie(
                response,
                provider.csrf_cookie,
                csrf_token,
                http_only=False,
            )
        return payload

    @router.post("/api/auth/logout")
    async def logout(request: Request):
        """Destroy the server-side session (CSRF-protected)."""
        principal = await principal_dependency(request)
        await _destroy_session_with_audit(provider, auditor, principal)
        response = JSONResponse({"logged_out": True})
        _clear_cookie(response, provider.session_cookie)
        _clear_cookie(response, provider.csrf_cookie)
        return response

    if provider.pat_service is not None:
        _register_token_routes(router, provider, principal_dependency)

    return router


_MIN_PASSWORD_LEN = 12

_RATE_LIMITED_MSG = (
    "Zu viele fehlgeschlagene Anmeldeversuche. Bitte spaeter erneut versuchen."
)


_DEFAULT_TRUSTED_PROXY_HOPS = 1


def _login_throttle_key(
    mode: str, identifier: str, request: Request, trusted_proxy_hops: int
) -> str:
    """Brute-force throttle key: ``mode:lower(identifier):client_ip``.

    ``trusted_proxy_hops`` is the provider's configured reverse-proxy depth,
    threaded to :func:`client_ip` so the client IP is read from the trusted
    (right) end of ``X-Forwarded-For`` and cannot be spoofed per request to
    evade the lockout.
    """
    from inqtrix.auth.ratelimit import client_ip

    ip = client_ip(request, trusted_proxy_hops)
    return f"{mode}:{identifier.strip().lower()}:{ip}"


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


def _build_password_auth_router(
    provider,
    principal_dependency: Callable[..., "Principal"],
    audit: "AuditSink | None" = None,
) -> APIRouter:
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
    auditor = _login_audit(audit)

    async def _audit_login_failure(
        *, mode: str, identifier: str, reason: str, locked_now: bool
    ) -> None:
        """auth.login_failed (+ auth.lockout on the transition).

        The attempted identifier is the attacked resource — standard
        AuthN-audit practice (BSI A3); the trail is admin-only with its
        own retention. Steady-state 429s during an active lockout are
        deliberately NOT re-audited (volume without information)."""
        if auditor is None:
            return
        await auditor.record_event(
            tenant_id="default",
            action="auth.login_failed",
            resource_type="account",
            resource_id=identifier.strip().lower(),
            outcome="failure",
            detail={"reason": reason},
            origin={"auth_method": mode},
        )
        if locked_now:
            await auditor.record_event(
                tenant_id="default",
                action="auth.lockout",
                resource_type="account",
                resource_id=identifier.strip().lower(),
                outcome="denied",
                origin={"auth_method": mode},
            )

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

    def _new_session(
        *,
        user_id: uuid.UUID,
        subject: str,
        issuer: str,
        email: str | None,
        display_name: str | None,
    ) -> AuthSession:
        return AuthSession(
            id=new_session_id(),
            user_id=user_id,
            issuer=issuer,
            subject=subject,
            email=email,
            display_name=display_name,
            groups=(),
            csrf_random=_secrets.token_hex(16),
            created_at=_time.time(),
            expires_at=_time.time() + provider.session_max_age_seconds,
        )

    def _session_response(
        session: AuthSession, *, status_code: int = 200
    ) -> JSONResponse:
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
            if users is None:
                return error_response(
                    503, "Nutzerverzeichnis ist nicht verfuegbar", "server_error"
                )
            display_name = str(body.get("display_name", "")).strip() or email
            credential = LocalCredential(
                user_id=uuid.uuid4(),
                subject=new_subject(),
                email=email,
                password_hash=hash_password(password),
                display_name=display_name,
                created_at=_time.time(),
            )
            if provider.lifecycle is None:
                return error_response(
                    503, "Nutzerverwaltung ist nicht verfuegbar", "server_error"
                )
            session = _new_session(
                user_id=credential.user_id,
                subject=credential.subject,
                issuer=LOCAL_ISSUER,
                email=email,
                display_name=display_name,
            )
            user = await provider.lifecycle.create_local_account(
                tenant_id="default",
                credential=credential,
                role="admin",
                session=session,
                first_only=True,
            )
            if user is None:
                # Idempotent permanent lock: the owner exists already.
                return error_response(
                    409, "Owner ist bereits eingerichtet", "setup_locked"
                )
            log.info(
                "Lokaler Owner angelegt: actor_ref=%s kind=oidc_session",
                pseudonymous_log_reference("usr", user.user_id),
            )
            return _session_response(session, status_code=201)

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
            throttle_key = _login_throttle_key(
                "local",
                email,
                request,
                getattr(
                    provider, "trusted_proxy_hops", _DEFAULT_TRUSTED_PROXY_HOPS
                ),
            )
            if limiter is not None and limiter.locked(throttle_key):
                return error_response(429, _RATE_LIMITED_MSG, "rate_limited")
            try:
                credential = await provider.authenticator.authenticate(
                    email, password
                )
            except CredentialError:
                locked_now = False
                if limiter is not None:
                    limiter.record_failure(throttle_key)
                    locked_now = limiter.locked(throttle_key)
                await _audit_login_failure(
                    mode="local",
                    identifier=email,
                    reason="invalid_credentials",
                    locked_now=locked_now,
                )
                return error_response(
                    401, "Ungueltige Anmeldedaten", "unauthorized"
                )
            if limiter is not None:
                limiter.reset(throttle_key)
            if users is None or provider.lifecycle is None:
                return error_response(
                    503, "Nutzerverzeichnis ist nicht verfuegbar", "server_error"
                )
            session = _new_session(
                user_id=credential.user_id,
                subject=credential.subject,
                issuer=LOCAL_ISSUER,
                email=credential.email,
                display_name=credential.display_name,
            )
            try:
                await provider.lifecycle.provision_login(
                    LoginCommand(
                        tenant_id="default",
                        issuer=LOCAL_ISSUER,
                        subject=credential.subject,
                        email=credential.email,
                        email_verified=True,
                        display_name=credential.display_name,
                        session=session,
                    )
                )
            except UserDisabledError:
                await _audit_login_failure(
                    mode="local",
                    identifier=email,
                    reason="account_disabled",
                    locked_now=False,
                )
                return error_response(
                    401, "Ungueltige Anmeldedaten", "unauthorized"
                )
            return _session_response(session)

        @router.post("/api/auth/password")
        async def change_password(request: Request):
            """Self-service password change for the signed-in local user.

            Session-gated (the caller must be an authenticated local session)
            and re-verifies the CURRENT password before replacing the hash —
            so a hijacked-but-not-authenticated request cannot change it. CSRF
            is enforced by the principal dependency on this unsafe method.
            """
            resolved, error = await _session_principal_for_tokens(
                provider, request, principal_dependency
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
            credential = await provider.credentials.get_by_user_id(
                tenant_id="default", user_id=session.user_id
            )
            if credential is None or not verify_password(
                credential.password_hash, current
            ):
                return error_response(
                    401, "Aktuelles Passwort ist falsch", "unauthorized"
                )
            await provider.credentials.set_password(
                tenant_id="default",
                user_id=session.user_id,
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
            throttle_key = _login_throttle_key(
                "ldap",
                username,
                request,
                getattr(
                    provider, "trusted_proxy_hops", _DEFAULT_TRUSTED_PROXY_HOPS
                ),
            )
            if limiter is not None and limiter.locked(throttle_key):
                return error_response(429, _RATE_LIMITED_MSG, "rate_limited")
            try:
                # ldap3 is blocking — keep it off the event loop.
                identity = await run_in_threadpool(
                    provider.ldap_client.authenticate, username, password
                )
            except LdapError:
                locked_now = False
                if limiter is not None:
                    limiter.record_failure(throttle_key)
                    locked_now = limiter.locked(throttle_key)
                await _audit_login_failure(
                    mode="ldap",
                    identifier=username,
                    reason="invalid_credentials",
                    locked_now=locked_now,
                )
                return error_response(
                    401, "Ungueltige Anmeldedaten", "unauthorized"
                )
            if limiter is not None:
                limiter.reset(throttle_key)
            if users is not None and provider.lifecycle is not None:
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
                    await _audit_login_failure(
                        mode="ldap",
                        identifier=username,
                        reason="account_disabled",
                        locked_now=False,
                    )
                    return error_response(
                        401, "Ungueltige Anmeldedaten", "unauthorized"
                    )
            else:
                return error_response(
                    503, "Nutzerverzeichnis ist nicht verfuegbar", "server_error"
                )
            session = _new_session(
                user_id=uuid.uuid4(),
                subject=identity.subject,
                issuer=LDAP_ISSUER,
                email=identity.email,
                display_name=identity.display_name,
            )
            try:
                await provider.lifecycle.provision_login(
                    LoginCommand(
                        tenant_id="default",
                        issuer=LDAP_ISSUER,
                        subject=identity.subject,
                        email=identity.email,
                        email_verified=True,
                        display_name=identity.display_name,
                        session=session,
                        is_admin=identity.is_admin,
                        first_login_owner=provider.first_login_owner,
                    )
                )
            except UserDisabledError:
                return error_response(
                    401, "Ungueltige Anmeldedaten", "unauthorized"
                )
            return _session_response(session)

    @router.get("/api/auth/session")
    async def session_info(request: Request, response: Response):
        """SPA bootstrap and repair the readable double-submit cookie."""
        payload = await provider.session_payload(request)
        csrf_token = payload.get("csrf_token")
        if payload.get("authenticated") is True and isinstance(csrf_token, str):
            _set_cookie(
                response,
                provider.csrf_cookie,
                csrf_token,
                http_only=False,
            )
        return payload

    @router.post("/api/auth/logout")
    async def logout(request: Request):
        """Destroy the server-side session (CSRF-protected)."""
        principal = await principal_dependency(request)
        await _destroy_session_with_audit(provider, auditor, principal)
        response = JSONResponse({"logged_out": True})
        _clear_cookie(response, provider.session_cookie)
        _clear_cookie(response, provider.csrf_cookie)
        return response

    if provider.pat_service is not None:
        _register_token_routes(router, provider, principal_dependency)

    return router


async def _session_principal_for_tokens(
    provider: "OidcAuthProvider",
    request: Request,
    principal_dependency: Callable[..., "Principal"],
):
    """Token management is SESSION-only: a PAT can never mint or
    revoke PATs (a leaked token must not be able to extend itself or
    cut off its siblings)."""
    principal = await principal_dependency(request)
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
    router: APIRouter,
    provider: "OidcAuthProvider",
    principal_dependency: Callable[..., "Principal"],
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
            provider, request, principal_dependency
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
                owner_user_id=session.user_id,
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
            provider, request, principal_dependency
        )
        if error is not None:
            return error
        _principal, session = resolved
        tokens = await provider.pat_service.list_tokens(
            tenant_id="default",
            owner_user_id=session.user_id,
        )
        return {"tokens": [_token_payload(record) for record in tokens]}

    @router.delete("/api/auth/tokens/{token_id}")
    async def revoke_token(token_id: str, request: Request):
        resolved, error = await _session_principal_for_tokens(
            provider, request, principal_dependency
        )
        if error is not None:
            return error
        _principal, session = resolved
        revoked = await provider.pat_service.revoke_token(
            tenant_id="default",
            token_id=token_id,
            owner_user_id=session.user_id,
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
