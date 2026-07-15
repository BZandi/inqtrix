"""Serve the Research Desk via uvicorn with streaming backend proxies.

Standalone Python launcher: serves the pre-built React frontend at
``apps/research-desk/dist/`` plus a streaming reverse-proxy for the
Inqtrix backend (HTTP on ``/v1/*``, ``/api/*``, ``/health``, and
``/collaboration/instance``; binary WebSocket on ``/collaboration``). The
browser sees a single origin, so the
React bundle works without ``VITE_INQTRIX_API_BASE_URL`` and without CORS on
the backend.

Use this launcher when the target host has only Python available
(no nginx, no Docker), when verifying the production build locally
without container tooling, or while prototyping a single-process
deployment before adding nginx. For real two-pod production prefer
the nginx pattern documented in ``docs/deployment/react-ui.md``.

Behaviour tracks the nginx template
(``deploy/nginx/inqtrix.conf.template``) so the launcher can stand in
for the ``web`` container: SPA fallback to ``index.html``, the
no-cache/immutable cache split (see :class:`_SpaStaticFiles`), the
    trusted public ``Host`` / ``X-Forwarded-Host`` / ``X-Forwarded-Proto``
    plus ``X-Forwarded-For`` / ``X-Real-IP`` on proxied requests, the raw request target
forwarded byte-identically (encoded path segments and repeated query
keys survive), and 502 Bad Gateway when the backend is unreachable
(60s connect timeout, matching nginx's default
``proxy_connect_timeout``).

Prerequisites
-------------
- ``pnpm run ui:build`` or ``npm run ui:build`` has produced
  ``apps/research-desk/dist/``.
- An Inqtrix backend is reachable at ``INQTRIX_BACKEND_URL``.

Usage
-----
::

    uv run python scripts/run_research_desk.py

Environment variables
---------------------
- ``RESEARCH_DESK_HOST``  Bind host (default ``127.0.0.1``).
- ``RESEARCH_DESK_PORT``  Bind port (default ``8080``).
- ``INQTRIX_BACKEND_URL`` Backend origin to proxy to
  (default ``http://localhost:5100``).
- ``INQTRIX_PUBLIC_BASE_URL`` Explicit browser origin when a trusted reverse
  proxy terminates TLS before the launcher (for example
  ``https://desk.example``). Incoming forwarding headers are never trusted.
- ``INQTRIX_DIST_DIR``    Override the ``dist/`` location (default:
  the ``apps/research-desk/dist`` directory inside this repository).
- ``INQTRIX_COLLABORATION_MAX_FRAME_BYTES`` Maximum collaboration frame size
  (default ``2097152``; valid range 65536 through 16777216 bytes).
- ``INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES`` Maximum inbound collaboration
  frame queue depth (default ``32``; valid range 1 through 256 frames).
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import AsyncIterator, NamedTuple
from urllib.parse import urlsplit, urlunsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.types import Scope
from starlette.websockets import WebSocketState
from websockets.asyncio.client import (
    ClientConnection,
    connect as websocket_connect,
)
from websockets.exceptions import ConnectionClosed

from inqtrix.settings import CollaborationSettings

log = logging.getLogger("research-desk-launcher")

# .mjs must resolve to a JavaScript MIME type or browsers reject the ES module
# ("Failed to fetch dynamically imported module"). Python's builtin map gained
# .mjs only in 3.12 and platform mime.types files vary, so the launcher
# registers the mapping itself instead of trusting the host (the nginx
# template pins default_type for /assets/ for the same reason).
mimetypes.add_type("text/javascript", ".mjs")

_HOP_BY_HOP = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
})
"""Per RFC 7230 these headers describe a single transport hop and MUST NOT
be forwarded by an intermediary. ``content-length`` is re-derived by httpx
for the upstream request and would otherwise collide; ``host`` is stripped
here and re-applied verbatim in :func:`_proxy_request_headers` (nginx
``proxy_set_header Host $host``)."""

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

_ASSETS_PREFIX = "assets/"
"""Mount-relative URL prefix of Vite's hashed build output. Filenames under it
are content-hashed (a changed file gets a new name), which is the property
that makes the immutable cache policy below safe. The literal matches Vite's
default ``build.assetsDir`` AND nginx's ``location /assets/`` — overriding
the Vite setting requires updating all three together."""

_ASSETS_CACHE_CONTROL = "public, max-age=31536000, immutable"
"""Cache policy for hashed assets — equivalent to the nginx template's
``expires 1y`` + ``Cache-Control "public, immutable"`` pair."""

_DEFAULT_CACHE_CONTROL = "no-cache"
"""Cache policy for every non-asset path, most importantly ``index.html``:
forces a cheap ETag revalidation per load so a redeploy is picked up
immediately. Without it browsers apply heuristic caching and a stale
``index.html`` keeps requesting hashed bundles the redeploy has deleted."""

_DEFAULT_COLLABORATION_MAX_FRAME_BYTES = int(
    CollaborationSettings.model_fields["max_frame_bytes"].default
)


class _PublicOrigin(NamedTuple):
    """Validated public origin used to overwrite forwarding metadata."""

    scheme: str
    authority: str


def _parse_public_origin(value: str | None) -> _PublicOrigin | None:
    """Validate an optional public HTTP(S) origin for proxy trust decisions."""
    if value is None or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "INQTRIX_PUBLIC_BASE_URL must be an absolute HTTP(S) origin"
        ) from exc
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "INQTRIX_PUBLIC_BASE_URL must be an absolute HTTP(S) origin"
        )
    normalized_host = hostname.lower()
    if ":" in normalized_host:
        normalized_host = f"[{normalized_host}]"
    default_port = 443 if scheme == "https" else 80
    authority = (
        normalized_host
        if port is None or port == default_port
        else f"{normalized_host}:{port}"
    )
    return _PublicOrigin(scheme=scheme, authority=authority)


def _validate_collaboration_max_frame_bytes(value: int) -> int:
    """Validate the legacy frame-only launcher override canonically.

    Args:
        value: Maximum accepted binary WebSocket message size in bytes.

    Returns:
        The validated integer for constructor-first propagation.

    Raises:
        ValueError: When the value is not an integer within the shared bounds.
    """
    try:
        settings = CollaborationSettings.model_validate(
            {"max_frame_bytes": value}
        )
    except ValueError as exc:
        raise ValueError(
            "INQTRIX_COLLABORATION_MAX_FRAME_BYTES is outside the "
            "CollaborationSettings bounds"
        ) from exc
    return settings.max_frame_bytes


def _parse_collaboration_max_frame_bytes(raw_value: str) -> int:
    """Parse the environment-facing collaboration frame limit."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "INQTRIX_COLLABORATION_MAX_FRAME_BYTES must be an integer"
        ) from exc
    return _validate_collaboration_max_frame_bytes(value)


def _resolve_collaboration_settings(
    settings: CollaborationSettings | None,
    *,
    legacy_max_frame_bytes: int | None,
) -> CollaborationSettings:
    """Resolve one validated transport contract without duplicated bounds."""
    if settings is not None:
        if (
            legacy_max_frame_bytes is not None
            and legacy_max_frame_bytes != settings.max_frame_bytes
        ):
            raise ValueError(
                "collaboration_settings conflicts with "
                "collaboration_max_frame_bytes"
            )
        return settings
    payload: dict[str, object] = {}
    if legacy_max_frame_bytes is not None:
        payload["max_frame_bytes"] = _validate_collaboration_max_frame_bytes(
            legacy_max_frame_bytes
        )
    return CollaborationSettings.model_validate(payload)


_WEBSOCKET_FORWARD_HEADERS = (
    "authorization",
    "cookie",
    "x-csrf-token",
    "x-inqtrix-workspace-id",
)
"""Application headers safe to copy into the backend WebSocket handshake.

WebSocket transport headers are generated by the upstream client and must not
be duplicated. ``Host``, ``Origin`` and ``User-Agent`` use dedicated client
parameters instead.
"""


def _resolve_dist_dir() -> Path:
    """Resolve the dist/ directory; fail loudly if missing.

    Returns:
        Absolute :class:`Path` to the React build output directory. The
        path is resolved either from ``INQTRIX_DIST_DIR`` (when set and
        non-empty) or from the repository-relative default
        ``apps/research-desk/dist`` next to this script.

    Raises:
        RuntimeError: When the resolved directory does not exist. The
            error message names the resolved path and instructs the
            operator to run ``pnpm run ui:build`` / ``npm run ui:build`` or correct
            ``INQTRIX_DIST_DIR``, matching the Inqtrix project policy
            of loud failure for missing deployment inputs.
    """
    explicit = os.getenv("INQTRIX_DIST_DIR", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent
        path = (repo_root / "apps" / "research-desk" / "dist").resolve()
    if not path.is_dir():
        raise RuntimeError(
            f"dist/ not found at {path}. Run `pnpm run ui:build` or "
            f"`npm run ui:build` first, or "
            f"set INQTRIX_DIST_DIR to point at an existing build."
        )
    return path


def _filter_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with hop-by-hop fields removed.

    Request direction (client to upstream) only. A ``dict`` collapses
    repeated field names, which is acceptable here because browsers send
    the request headers the proxy forwards as singletons; the response
    direction must preserve duplicates instead (see
    :func:`_relay_response_headers`).

    Args:
        headers: Header mapping from a Starlette ``Request``.

    Returns:
        Plain ``dict`` containing only headers safe to forward.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _relay_response_headers(headers: httpx.Headers) -> list[tuple[bytes, bytes]]:
    """Build raw relay headers from an upstream response, keeping duplicates.

    The response is relayed field-by-field instead of through a ``dict``
    because HTTP allows a field name to occur multiple times and
    ``Set-Cookie`` is the one field where folding those occurrences into a
    single comma-joined value changes meaning: RFC 6265 forbids the fold,
    and browsers parse each ``Set-Cookie`` line as exactly one cookie. The
    login response carries two of them (session and CSRF token); a
    dict-shaped relay silently dropped the CSRF cookie in every browser,
    which then failed every cookie-authenticated non-GET request with 403.
    nginx (the template this launcher mirrors) forwards the fields
    unmerged, so this is also the parity behaviour.

    Args:
        headers: ``httpx.Headers`` of the upstream response;
            ``multi_items()`` yields every occurrence of every field in
            wire order.

    Returns:
        Raw ``(name, value)`` byte pairs for
        ``starlette.responses.Response.raw_headers``, hop-by-hop fields
        removed, names lowercased (matching Starlette's own header
        normalisation).
    """
    return [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in headers.multi_items()
        if name.lower() not in _HOP_BY_HOP
    ]


def _proxy_request_headers(
    request: Request,
    public_origin: _PublicOrigin | None,
) -> dict[str, str]:
    """Build upstream headers: hop-by-hop stripped, forwarding added.

    Mirrors the ``proxy_set_header`` block of the nginx template. The
    With an explicit public origin, its authority and scheme overwrite all
    incoming Host forwarding metadata. Direct HTTP mode derives these values
    from the current ASGI connection and Host, never from incoming
    ``X-Forwarded-Proto`` or ``X-Forwarded-Host``. The client address is appended to any
    incoming ``X-Forwarded-For`` chain (``$proxy_add_x_forwarded_for``
    semantics) and exposed as ``X-Real-IP``; the backend's login rate
    limiter keys on these, so without them every browser behind the
    launcher would share a single bucket.

    Args:
        request: Incoming Starlette request. ``request.client`` can be
            ``None`` on non-socket transports; the address headers are
            then omitted rather than invented.

    Returns:
        Header dict for the ``httpx`` upstream request. Lookups and
        overrides key on lowercase names, which is safe because ASGI
        servers deliver header names lowercased.
    """
    headers = _filter_headers(request.headers)
    headers.pop("x-forwarded-proto", None)
    headers.pop("x-forwarded-host", None)
    host = (
        public_origin.authority
        if public_origin is not None
        else request.headers.get("host")
    )
    if host:
        headers["host"] = host
        headers["x-forwarded-host"] = host
    client_host = request.client.host if request.client else None
    if client_host:
        prior = request.headers.get("x-forwarded-for")
        headers["x-forwarded-for"] = (
            f"{prior}, {client_host}" if prior else client_host
        )
        headers["x-real-ip"] = client_host
    headers["x-forwarded-proto"] = (
        public_origin.scheme if public_origin is not None else request.url.scheme
    )
    return headers


def _proxy_websocket_headers(
    websocket: WebSocket,
    public_origin: _PublicOrigin | None,
) -> dict[str, str]:
    """Build application and forwarding headers for the backend handshake.

    Args:
        websocket: Incoming browser WebSocket. Its transport-specific
            handshake fields are intentionally excluded because the upstream
            client generates a fresh, valid handshake.

    Returns:
        Headers that preserve authentication context and reverse-proxy client
        metadata without duplicating WebSocket protocol fields.
    """
    headers = {
        name: value
        for name in _WEBSOCKET_FORWARD_HEADERS
        if (value := websocket.headers.get(name)) is not None
    }
    client_host = websocket.client.host if websocket.client else None
    if client_host:
        prior = websocket.headers.get("x-forwarded-for")
        headers["x-forwarded-for"] = (
            f"{prior}, {client_host}" if prior else client_host
        )
        headers["x-real-ip"] = client_host
    authority = (
        public_origin.authority
        if public_origin is not None
        else websocket.headers.get("host")
    )
    if not authority:
        raise ValueError("WebSocket Host is required for trusted forwarding")
    headers["x-forwarded-host"] = authority
    headers["x-forwarded-proto"] = (
        public_origin.scheme
        if public_origin is not None
        else ("https" if websocket.url.scheme == "wss" else "http")
    )
    return headers


def _websocket_upstream(
    websocket: WebSocket,
    backend_url: str,
    public_origin: _PublicOrigin | None,
) -> tuple[str, str, int, str | None]:
    """Resolve the public handshake URI and private TCP destination.

    The FastAPI collaboration gateway validates the browser ``Origin`` against
    the public ``Host``. A naive ``ws://backend/collaboration`` connection
    replaces that Host with the private service name and rejects local
    deployments using a non-default public port. ``websockets`` supports
    separating URI authority from TCP destination: the URI below retains the
    browser authority for the handshake while ``host`` and ``port`` direct the
    socket to ``INQTRIX_BACKEND_URL``.

    Args:
        websocket: Incoming browser socket, including its public Host and raw
            query string.
        backend_url: HTTP(S) origin of the FastAPI backend.

    Returns:
        Tuple of upstream WebSocket URI, TCP host, TCP port, and optional TLS
        server name. The raw query string is retained byte-for-byte.

    Raises:
        ValueError: When ``backend_url`` is not an absolute HTTP(S) origin.
    """
    backend = urlsplit(backend_url)
    if backend.scheme not in {"http", "https"} or backend.hostname is None:
        raise ValueError("backend_url must be an absolute HTTP(S) origin")
    scheme = "wss" if backend.scheme == "https" else "ws"
    authority = (
        public_origin.authority
        if public_origin is not None
        else (websocket.headers.get("host") or backend.netloc)
    )
    query = bytes(websocket.scope.get("query_string", b"")).decode("ascii")
    target = urlunsplit((scheme, authority, "/collaboration", query, ""))
    port = backend.port or (443 if backend.scheme == "https" else 80)
    server_hostname = backend.hostname if backend.scheme == "https" else None
    return target, backend.hostname, port, server_hostname


def _safe_websocket_close_code(code: int) -> int:
    """Return a close code that Starlette may send to a browser."""
    if code in {1000, 1001, 1002, 1003, 1008, 1009, 1011, 1012}:
        return code
    if 4000 <= code <= 4999:
        return code
    return 1011


async def _close_websocket(
    websocket: WebSocket, code: int, reason: str = ""
) -> None:
    """Close a browser WebSocket on any ASGI connection state."""
    with suppress(RuntimeError, WebSocketDisconnect):
        if websocket.application_state is WebSocketState.CONNECTING:
            await websocket.accept()
        if websocket.application_state is WebSocketState.CONNECTED:
            await websocket.close(
                code=_safe_websocket_close_code(code),
                reason=reason[:123],
            )


async def _relay_browser_websocket(
    browser: WebSocket,
    upstream: ClientConnection,
    max_frame_bytes: int,
) -> None:
    """Relay binary browser messages to FastAPI until either side closes."""
    while True:
        message = await browser.receive()
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            await upstream.close(
                code=_safe_websocket_close_code(
                    int(message.get("code") or 1000)
                )
            )
            return
        if message_type != "websocket.receive":
            continue
        payload = message.get("bytes")
        if payload is None:
            await upstream.close(code=4409, reason="binary_frames_required")
            await _close_websocket(browser, 4409, "binary_frames_required")
            return
        if len(payload) > max_frame_bytes:
            await upstream.close(code=1009, reason="message_too_big")
            await _close_websocket(browser, 1009, "message_too_big")
            return
        await upstream.send(payload)


async def _relay_upstream_websocket(
    upstream: ClientConnection,
    browser: WebSocket,
    max_frame_bytes: int,
) -> None:
    """Relay binary FastAPI messages to the browser and propagate closure."""
    try:
        async for payload in upstream:
            if isinstance(payload, str):
                log.warning(
                    "Backend emitted an unexpected text WebSocket message."
                )
                await _close_websocket(browser, 1011, "invalid_upstream_frame")
                return
            if len(payload) > max_frame_bytes:
                await upstream.close(code=1009, reason="message_too_big")
                await _close_websocket(browser, 1009, "message_too_big")
                return
            await browser.send_bytes(bytes(payload))
    except ConnectionClosed as exc:
        await _close_websocket(browser, exc.code, exc.reason)
        return
    await _close_websocket(
        browser,
        upstream.close_code or 1000,
        upstream.close_reason or "",
    )


class _SpaStaticFiles(StaticFiles):
    """Static app with SPA fallback and the nginx template's cache split.

    Encodes the three static-serving behaviours of
    ``deploy/nginx/inqtrix.conf.template`` that plain ``StaticFiles``
    does not provide:

    - Unknown paths outside ``/assets/`` serve ``index.html``
      (``try_files $uri $uri/ /index.html``), so the SPA also loads on
      deep links and post-login redirects to sub-paths. The root path
      ``/`` resolves through the same fallback, which is why the mount
      runs with ``html=False``: Starlette's ``html`` mode would add
      nothing here except a ``404.html`` branch that RETURNS the 404
      page instead of raising — silently bypassing this fallback if a
      build ever shipped a ``404.html`` — plus one wasted stat per miss.
    - Missing files under ``/assets/`` stay hard 404s (``try_files $uri
      =404``). Falling back to ``index.html`` there would hand HTML to
      module loaders and mask a stale-bundle problem instead of
      surfacing it.
    - ``/assets/`` responses cache immutably, everything else — most
      importantly ``index.html``, the only unhashed entry point — is
      ``no-cache`` (see the module constants for the redeploy
      rationale). The header is set on every response, including 304 and
      206, matching nginx's ``add_header``/``expires`` behaviour.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve ``path``, falling back to ``index.html`` for SPA routes.

        Args:
            path: Mount-relative file path derived from the URL (no
                leading slash, e.g. ``assets/index-abc.js``).
            scope: ASGI scope of the request, forwarded to the base
                implementation for method/range handling.

        Returns:
            The file response with the cache-control policy applied.

        Raises:
            HTTPException: 404 for missing ``/assets/`` files, plus
                whatever non-404 errors the base implementation raises
                (405 for unsupported methods, 401 on permission errors).
        """
        is_asset = path.startswith(_ASSETS_PREFIX)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or is_asset:
                raise
            response = await super().get_response("index.html", scope)
        response.headers["cache-control"] = (
            _ASSETS_CACHE_CONTROL if is_asset else _DEFAULT_CACHE_CONTROL
        )
        return response


async def _relay_upstream(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield the upstream body, always releasing the upstream response.

    A transport failure mid-stream (backend restart during a long SSE
    response) must not escape as an unhandled exception: it would tear
    down the client connection with a traceback and skip the upstream
    cleanup, leaking the pooled ``httpx`` connection on every dropped
    stream. Instead the relay logs the abort and ends the body quietly —
    to the browser this looks exactly like a killed nginx upstream. The
    ``finally`` also runs on client disconnect (``GeneratorExit``), so
    the upstream is released on every exit path without relying on a
    Starlette ``background`` callback, which is skipped on errors.
    """
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    except httpx.TransportError as exc:
        log.warning("Upstream stream aborted mid-response: %s", exc)
    finally:
        await response.aclose()


def build_app(
    dist_dir: Path,
    backend_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    collaboration_max_frame_bytes: int | None = None,
    collaboration_settings: CollaborationSettings | None = None,
    public_base_url: str | None = None,
) -> FastAPI:
    """Wire the launcher FastAPI app: streaming proxy plus static mount.

    The app registers the API proxy routes first and then mounts the
    static directory at ``/`` via :class:`_SpaStaticFiles` (SPA fallback
    to ``index.html`` plus the no-cache/immutable cache split). Route
    registration order matters: API paths must win over the static
    catch-all.

    Args:
        dist_dir: Absolute path to the React build output. Caller must
            ensure existence (use :func:`_resolve_dist_dir`).
        backend_url: Origin (scheme + host + optional port) of the
            Inqtrix backend. Trailing slashes are not normalized here;
            the caller already strips them in :func:`main`.
        transport: Optional ``httpx.AsyncBaseTransport`` for tests; when
            ``None`` the underlying ``httpx.AsyncClient`` uses the
            default network transport. Production callers never pass
            this; the parameter exists so :class:`httpx.MockTransport`
            can be injected from unit tests without monkey-patching.
        collaboration_max_frame_bytes: Largest accepted binary WebSocket
            message in either relay direction. Retained as a backwards-
            compatible override for callers that do not yet pass the full
            collaboration settings contract.
        collaboration_settings: Canonical validated frame-size and queue-depth
            contract shared with the FastAPI collaboration gateway.
        public_base_url: Explicit browser origin trusted when TLS terminates at
            a reverse proxy before this launcher. Empty keeps direct HTTP mode.

    Returns:
        A wired :class:`FastAPI` instance with a lifespan that closes
        the shared ``httpx.AsyncClient`` on shutdown. The instance is
        intended for ``uvicorn.run`` or :class:`fastapi.testclient.TestClient`.
    """
    collaboration_settings = _resolve_collaboration_settings(
        collaboration_settings,
        legacy_max_frame_bytes=collaboration_max_frame_bytes,
    )
    collaboration_max_frame_bytes = collaboration_settings.max_frame_bytes
    collaboration_max_queued_frames = collaboration_settings.max_queued_frames
    public_origin = _parse_public_origin(public_base_url)

    # Connects fail after 60s like nginx's default proxy_connect_timeout, so a
    # black-holed backend yields a 502 instead of hanging forever; read/write
    # stay unbounded because SSE responses are long-lived.
    client = httpx.AsyncClient(
        base_url=backend_url,
        timeout=httpx.Timeout(connect=60.0, read=None, write=None, pool=None),
        transport=transport,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "Serving %s and proxying /v1, /api, /health, "
            "/collaboration/instance, /collaboration "
            "to %s",
            dist_dir,
            backend_url,
        )
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(
        title="Inqtrix Research Desk launcher",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.websocket("/collaboration")
    async def proxy_collaboration(websocket: WebSocket) -> None:
        """Relay the browser's binary collaboration socket to FastAPI."""
        try:
            target, tcp_host, tcp_port, server_hostname = _websocket_upstream(
                websocket, backend_url, public_origin
            )
            destination: dict[str, object] = {
                "host": tcp_host,
                "port": tcp_port,
            }
            if server_hostname is not None:
                destination["server_hostname"] = server_hostname
            async with websocket_connect(
                target,
                origin=websocket.headers.get("origin"),
                additional_headers=_proxy_websocket_headers(
                    websocket, public_origin
                ),
                user_agent_header=websocket.headers.get("user-agent"),
                compression=None,
                proxy=None,
                open_timeout=60,
                close_timeout=3,
                ping_interval=20,
                ping_timeout=20,
                max_size=collaboration_max_frame_bytes,
                max_queue=collaboration_max_queued_frames,
                **destination,
            ) as upstream:
                await websocket.accept()
                browser_to_backend = asyncio.create_task(
                    _relay_browser_websocket(
                        websocket,
                        upstream,
                        collaboration_max_frame_bytes,
                    )
                )
                backend_to_browser = asyncio.create_task(
                    _relay_upstream_websocket(
                        upstream,
                        websocket,
                        collaboration_max_frame_bytes,
                    )
                )
                done, pending = await asyncio.wait(
                    {browser_to_backend, backend_to_browser},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()
        except WebSocketDisconnect:
            return
        except ConnectionClosed as exc:
            await _close_websocket(websocket, exc.code, exc.reason)
        except Exception as exc:
            # Authentication credentials travel in cookies or WebSocket auth
            # messages, never URLs. Keep the target and exception body out of
            # logs because connection metadata may still be sensitive.
            log.warning(
                "FastAPI collaboration WebSocket upstream is unavailable "
                "(%s).",
                type(exc).__name__,
            )
            await _close_websocket(
                websocket, 4503, "collaboration_gateway_unavailable"
            )

    async def _proxy(request: Request) -> Response:
        # Forward the ORIGINAL request target (nginx `proxy_pass` semantics):
        # rebuilding it from the decoded route parameter would collapse
        # percent-encoded separators (%2F, %3F) and repeated query keys.
        raw_path = request.scope.get("raw_path") or request.url.path.encode(
            "utf-8"
        )
        target = bytes(raw_path).split(b"?", 1)[0]
        if request.url.query:
            target += b"?" + request.url.query.encode("ascii")
        # A request body exists only when the client declared one (RFC 9112:
        # Content-Length or Transfer-Encoding); streaming it upstream avoids
        # buffering large uploads in launcher memory.
        has_body = (
            "content-length" in request.headers
            or "transfer-encoding" in request.headers
        )
        upstream = client.build_request(
            method=request.method,
            url=httpx.URL(raw_path=target),
            headers=_proxy_request_headers(request, public_origin),
            content=request.stream() if has_body else None,
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.TransportError as exc:
            # nginx answers 502 Bad Gateway here; an unhandled ConnectError
            # would surface as an opaque 500. The backend origin stays out of
            # the client-visible body — it belongs in the operator log only.
            log.warning(
                "Backend %s unreachable for %s %s: %s",
                backend_url,
                request.method,
                request.url.path,
                exc,
            )
            return JSONResponse(
                {"detail": "Backend unreachable"},
                status_code=502,
            )
        relay = StreamingResponse(
            _relay_upstream(response),
            status_code=response.status_code,
        )
        # Assigned raw instead of via the ``headers=`` mapping: a mapping
        # cannot carry duplicate fields (Set-Cookie). Content-type arrives
        # through the relayed fields, so no ``media_type`` is needed.
        relay.raw_headers = _relay_response_headers(response.headers)
        return relay

    @app.api_route("/health", methods=_PROXY_METHODS)
    async def proxy_health(request: Request) -> Response:
        return await _proxy(request)

    @app.get("/collaboration/instance")
    async def proxy_collaboration_instance(request: Request) -> Response:
        """Forward the content-free production instance probe to FastAPI."""
        return await _proxy(request)

    @app.api_route("/v1/{full_path:path}", methods=_PROXY_METHODS)
    async def proxy_v1(request: Request) -> Response:
        return await _proxy(request)

    @app.api_route("/api/{full_path:path}", methods=_PROXY_METHODS)
    async def proxy_api(request: Request) -> Response:
        """Forward the ``/api/*`` surface to the backend.

        Covers the auth BFF (``/api/auth/*``), the local-auth setup wizard
        (``/api/setup/*``), and admin routes (``/api/admin/*``). Without this
        the same-origin production path cannot log in. Session cookies and the
        ``X-CSRF-Token`` header pass through unchanged via
        ``_proxy_request_headers``.
        """
        return await _proxy(request)

    app.mount(
        "/", _SpaStaticFiles(directory=str(dist_dir), html=False), name="frontend"
    )

    return app


def main() -> None:
    """Resolve env vars, wire the app, and start uvicorn.

    The function is intentionally thin: all configuration knobs live in
    environment variables so the call site reduces to ``uv run python
    scripts/run_research_desk.py``. Constructor-First applies — env
    parsing happens here, never inside :func:`build_app`, so the latter
    stays test-friendly.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dist_dir = _resolve_dist_dir()
    backend_url = os.getenv("INQTRIX_BACKEND_URL", "http://localhost:5100").rstrip("/")
    public_base_url = os.getenv("INQTRIX_PUBLIC_BASE_URL", "").strip() or None
    collaboration_settings = CollaborationSettings(_env_file=None)
    app = build_app(
        dist_dir,
        backend_url,
        collaboration_settings=collaboration_settings,
        public_base_url=public_base_url,
    )
    uvicorn.run(
        app,
        host=os.getenv("RESEARCH_DESK_HOST", "127.0.0.1"),
        port=int(os.getenv("RESEARCH_DESK_PORT", "8080")),
        log_level="info",
        timeout_keep_alive=300,
        ws_max_size=collaboration_settings.max_frame_bytes,
        ws_max_queue=collaboration_settings.max_queued_frames,
    )


if __name__ == "__main__":
    main()
