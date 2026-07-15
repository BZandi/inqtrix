"""Unit tests for ``scripts/run_research_desk.py``.

The launcher is a standalone script outside the ``inqtrix`` package
namespace, so the module is loaded by absolute path through
``importlib`` rather than imported directly. Tests cover the
behaviours that matter for deployment correctness: SPA fallback and
the cache-policy split, streaming HTTP proxy routing, binary WebSocket
relay, 502 mapping for an unreachable backend, and loud failure on a
missing ``dist/`` directory.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.sync.server import ServerConnection, serve


def _json_response(status_code: int, payload: object) -> httpx.Response:
    """Build a streaming Mock response.

    ``httpx.Response(json=...)`` reads the stream eagerly, which leaves
    the response in a ``is_stream_consumed=True`` state and makes
    ``aiter_raw()`` raise ``StreamConsumed`` when the proxy iterates it.
    Using an explicit ``stream=`` parameter keeps the body unread until
    the production code iterates it through Starlette.
    """
    body = json.dumps(payload).encode("utf-8")
    return httpx.Response(
        status_code,
        stream=httpx.ByteStream(body),
        headers={
            "content-type": "application/json",
            "content-length": str(len(body)),
        },
    )


def _load_launcher_module():
    """Import ``scripts/run_research_desk.py`` by absolute path."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "run_research_desk.py"
    spec = importlib.util.spec_from_file_location("run_research_desk", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher_module()


class _FakeWebSocketUpstream:
    """Deterministic ``websockets`` peer for launcher relay contracts."""

    def __init__(self, *, incoming: tuple[bytes | str, ...] = ()) -> None:
        self.incoming = incoming
        self.sent: list[bytes] = []
        self.close_calls: list[tuple[int, str]] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self._queue: asyncio.Queue[bytes | str] | None = None

    async def __aenter__(self) -> "_FakeWebSocketUpstream":
        self._queue = asyncio.Queue()
        for payload in self.incoming:
            self._queue.put_nowait(payload)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, exc, traceback
        return False

    async def send(self, payload: bytes) -> None:
        self.sent.append(bytes(payload))

    async def close(self, *, code: int, reason: str = "") -> None:
        self.close_calls.append((code, reason))
        self.close_code = code
        self.close_reason = reason

    def __aiter__(self) -> "_FakeWebSocketUpstream":
        return self

    async def __anext__(self) -> bytes | str:
        assert self._queue is not None
        return await self._queue.get()


class _WebSocketConnector:
    """Capture launcher upstream options and return a scripted peer."""

    def __init__(self, upstream: _FakeWebSocketUpstream) -> None:
        self.upstream = upstream
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, target: str, **kwargs: Any) -> _FakeWebSocketUpstream:
        self.calls.append((target, dict(kwargs)))
        return self.upstream


@pytest.fixture
def fake_dist(tmp_path: Path) -> Path:
    """Create a minimal dist/ layout with index.html and one asset."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body>fake-react-app</body></html>",
        encoding="utf-8",
    )
    (dist / "assets" / "main.js").write_text(
        "console.log('fake bundle');", encoding="utf-8"
    )
    return dist


@pytest.fixture
def websocket_backend() -> Iterator[tuple[str, dict[str, object]]]:
    """Run a real local WebSocket endpoint for binary relay verification."""
    captured: dict[str, object] = {}
    reply = b"\x00backend\xffreply"

    def handler(connection: ServerConnection) -> None:
        request = connection.request
        assert request is not None
        captured["path"] = request.path
        captured["host"] = request.headers["Host"]
        captured["origin"] = request.headers["Origin"]
        captured["cookie"] = request.headers["Cookie"]
        captured["xff"] = request.headers["X-Forwarded-For"]
        captured["proto"] = request.headers["X-Forwarded-Proto"]
        captured["forwarded_host"] = request.headers["X-Forwarded-Host"]
        captured["payload"] = connection.recv(timeout=5)
        connection.send(reply)

    server = serve(handler, "127.0.0.1", 0, compression=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.socket.getsockname()[:2]
        yield f"http://{host}:{port}", captured
    finally:
        server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_static_mount_serves_index_html(fake_dist: Path) -> None:
    """The root path delivers ``index.html`` from the mounted dist/."""
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "fake-react-app" in response.text
    assert response.headers["content-type"].startswith("text/html")


def test_static_mount_serves_assets(fake_dist: Path) -> None:
    """Hashed asset paths are served directly without proxy roundtrip."""
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        response = client.get("/assets/main.js")
    assert response.status_code == 200
    assert "fake bundle" in response.text


def test_v1_proxies_to_backend(fake_dist: Path) -> None:
    """``GET /v1/runs`` reaches the upstream and returns its body."""
    received: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        received["method"] = request.method
        received["auth"] = request.headers.get("authorization")
        return _json_response(200, {"data": ["proxied"]})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/runs",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    assert response.json() == {"data": ["proxied"]}
    assert received["url"] == "http://backend.invalid/v1/runs"
    assert received["method"] == "GET"
    assert received["auth"] == "Bearer test-token"


def test_health_proxies_to_backend(fake_dist: Path) -> None:
    """``GET /health`` reaches the upstream ``/health`` endpoint."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://backend.invalid/health"
        return _json_response(200, {"status": "ok"})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_collaboration_instance_probe_proxies_to_backend(fake_dist: Path) -> None:
    """The dist launcher cannot serve SPA HTML at the release probe path."""
    payload = {
        "contract": "inqtrix-collaboration-instance-v1",
        "service": "inqtrix-collaboration",
        "status": "ready",
        "instance_id": "node-dist",
        "epoch": 9,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "http://backend.invalid/collaboration/instance"
        )
        assert request.method == "GET"
        return _json_response(200, payload)

    app = launcher.build_app(
        fake_dist,
        "http://backend.invalid",
        transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.get("/collaboration/instance")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == payload


def test_api_proxies_to_backend(fake_dist: Path) -> None:
    """``GET /api/auth/session`` reaches the upstream ``/api/*`` path.

    Without this the same-origin production login (auth BFF, local-auth
    setup wizard, admin routes) is unreachable behind the launcher.
    """
    received: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        received["method"] = request.method
        return _json_response(200, {"authenticated": False})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        response = client.get("/api/auth/session")

    assert response.status_code == 200
    assert response.json() == {"authenticated": False}
    assert received["url"] == "http://backend.invalid/api/auth/session"
    assert received["method"] == "GET"


def test_collaboration_relays_binary_websocket_to_fastapi_backend(
    fake_dist: Path,
    websocket_backend: tuple[str, dict[str, object]],
) -> None:
    """``/collaboration`` is a binary relay to the configured FastAPI URL.

    The upstream handshake keeps the browser-facing Host and Origin even
    though the TCP connection goes to the private backend address. This is
    required by FastAPI's same-origin gateway check and proves that the
    launcher doesn't connect browsers directly to the Node service.
    """
    backend_url, captured = websocket_backend
    app = launcher.build_app(fake_dist, backend_url)
    payload = b"\x00browser\x80update\xff"

    with TestClient(app) as client:
        with client.websocket_connect(
            "/collaboration?room=doc%2Fone&client=browser%2Bone",
            headers={
                "Host": "desk.example:8443",
                "Origin": "http://desk.example:8443",
                "Cookie": "__Host-inqtrix_session=session-value",
            },
        ) as websocket:
            websocket.send_bytes(payload)
            assert websocket.receive_bytes() == b"\x00backend\xffreply"

    assert captured == {
        "path": "/collaboration?room=doc%2Fone&client=browser%2Bone",
        "host": "desk.example:8443",
        "origin": "http://desk.example:8443",
        "cookie": "__Host-inqtrix_session=session-value",
        "xff": "testclient",
        "proto": "http",
        "forwarded_host": "desk.example:8443",
        "payload": payload,
    }


def test_collaboration_tls_forwarding_uses_explicit_public_origin(
    fake_dist: Path,
    websocket_backend: tuple[str, dict[str, object]],
) -> None:
    """Outer TLS metadata comes only from configured public origin."""
    backend_url, captured = websocket_backend
    app = launcher.build_app(
        fake_dist,
        backend_url,
        public_base_url="https://Desk.Example:8443/",
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/collaboration",
            headers={
                "Host": "launcher.internal:8080",
                "Origin": "https://desk.example:8443",
                "Cookie": "__Host-inqtrix_session=session-value",
                "X-Forwarded-Proto": "http",
                "X-Forwarded-Host": "attacker.example",
            },
        ) as websocket:
            websocket.send_bytes(b"tls-update")
            assert websocket.receive_bytes() == b"\x00backend\xffreply"

    assert captured["host"] == "desk.example:8443"
    assert captured["origin"] == "https://desk.example:8443"
    assert captured["proto"] == "https"
    assert captured["forwarded_host"] == "desk.example:8443"
    assert captured["payload"] == b"tls-update"


def test_collaboration_unavailable_closes_with_retryable_gateway_code(
    fake_dist: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable FastAPI upstream is distinct from an internal fault."""

    def unavailable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("upstream unavailable")

    monkeypatch.setattr(launcher, "websocket_connect", unavailable)
    app = launcher.build_app(fake_dist, "http://backend.invalid")

    with TestClient(app) as client:
        with client.websocket_connect("/collaboration") as websocket:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                websocket.receive_bytes()

    assert exc_info.value.code == 4503
    assert exc_info.value.reason == "collaboration_gateway_unavailable"


def test_collaboration_rejects_oversized_browser_frame_before_upstream_send(
    fake_dist: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher closes 1009 before forwarding an oversized browser frame."""
    limit = 65_536
    settings = launcher.CollaborationSettings.model_validate(
        {
            "max_frame_bytes": limit,
            "max_queued_frames": 5,
        }
    )
    upstream = _FakeWebSocketUpstream()
    connector = _WebSocketConnector(upstream)
    monkeypatch.setattr(launcher, "websocket_connect", connector)
    app = launcher.build_app(
        fake_dist,
        "http://backend.invalid",
        collaboration_settings=settings,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/collaboration") as websocket:
            websocket.send_bytes(b"x" * (limit + 1))
            with pytest.raises(WebSocketDisconnect) as exc_info:
                websocket.receive_bytes()

    assert exc_info.value.code == 1009
    assert exc_info.value.reason == "message_too_big"
    assert upstream.sent == []
    assert (1009, "message_too_big") in upstream.close_calls
    assert connector.calls[0][1]["max_size"] == limit
    assert connector.calls[0][1]["max_queue"] == 5


def test_collaboration_rejects_oversized_upstream_frame_before_browser_send(
    fake_dist: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reverse relay closes both peers with 1009 for oversized output."""
    limit = 65_536
    upstream = _FakeWebSocketUpstream(incoming=(b"x" * (limit + 1),))
    connector = _WebSocketConnector(upstream)
    monkeypatch.setattr(launcher, "websocket_connect", connector)
    app = launcher.build_app(
        fake_dist,
        "http://backend.invalid",
        collaboration_max_frame_bytes=limit,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/collaboration") as websocket:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                websocket.receive_bytes()

    assert exc_info.value.code == 1009
    assert exc_info.value.reason == "message_too_big"
    assert upstream.sent == []
    assert (1009, "message_too_big") in upstream.close_calls


@pytest.mark.parametrize(
    "frame_limit",
    [True, 65_535, 16 * 1_048_576 + 1],
)
def test_collaboration_frame_limit_rejects_invalid_configuration(
    fake_dist: Path,
    frame_limit: object,
) -> None:
    """Launcher construction fails loudly for values outside shared bounds."""
    with pytest.raises(
        ValueError,
        match="INQTRIX_COLLABORATION_MAX_FRAME_BYTES",
    ):
        launcher.build_app(
            fake_dist,
            "http://backend.invalid",
            collaboration_max_frame_bytes=frame_limit,
        )


@pytest.mark.parametrize("queue_limit", [0, 257])
def test_collaboration_queue_limit_rejects_invalid_configuration(
    queue_limit: int,
) -> None:
    """The canonical launcher settings bound the parser's frame queue."""
    with pytest.raises(ValueError):
        launcher.CollaborationSettings.model_validate(
            {"max_queued_frames": queue_limit}
        )


@pytest.mark.parametrize(
    "public_base_url",
    [
        "desk.example",
        "ftp://desk.example",
        "https://user@desk.example",
        "https://desk.example/path",
        "https://desk.example?tenant=a",
    ],
)
def test_public_base_url_rejects_non_origin_values(
    fake_dist: Path,
    public_base_url: str,
) -> None:
    """TLS forwarding cannot be configured from an ambiguous URL."""
    with pytest.raises(ValueError, match="INQTRIX_PUBLIC_BASE_URL"):
        launcher.build_app(
            fake_dist,
            "http://backend.invalid",
            public_base_url=public_base_url,
        )


def test_main_propagates_non_default_collaboration_transport_limits(
    fake_dist: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One environment value configures both relay and browser parser limits."""
    configured_frame_bytes = 3 * 1_048_576
    configured_queue = 9
    app = object()
    captured: dict[str, object] = {}

    def build_app(
        dist_dir: Path,
        backend_url: str,
        *,
        collaboration_settings: Any,
        public_base_url: str | None,
    ) -> object:
        captured["dist_dir"] = dist_dir
        captured["backend_url"] = backend_url
        captured["app_frame_limit"] = collaboration_settings.max_frame_bytes
        captured["app_queue_limit"] = collaboration_settings.max_queued_frames
        captured["public_base_url"] = public_base_url
        return app

    def run(received_app: object, **kwargs: object) -> None:
        captured["uvicorn_app"] = received_app
        captured["uvicorn_options"] = kwargs

    monkeypatch.setenv(
        "INQTRIX_COLLABORATION_MAX_FRAME_BYTES",
        str(configured_frame_bytes),
    )
    monkeypatch.setenv(
        "INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES",
        str(configured_queue),
    )
    monkeypatch.setenv("INQTRIX_COLLABORATION_ENABLED", "false")
    monkeypatch.setenv("INQTRIX_PUBLIC_BASE_URL", "https://desk.example")
    monkeypatch.setattr(launcher, "_resolve_dist_dir", lambda: fake_dist)
    monkeypatch.setattr(launcher, "build_app", build_app)
    monkeypatch.setattr(launcher.uvicorn, "run", run)

    launcher.main()

    assert captured["app_frame_limit"] == configured_frame_bytes
    assert captured["app_queue_limit"] == configured_queue
    assert captured["public_base_url"] == "https://desk.example"
    assert captured["uvicorn_app"] is app
    assert (
        captured["uvicorn_options"]["ws_max_size"]
        == configured_frame_bytes
    )
    assert captured["uvicorn_options"]["ws_max_queue"] == configured_queue


def test_api_proxy_forwards_csrf_and_cookie(fake_dist: Path) -> None:
    """Unsafe ``/api`` calls forward the session cookie + CSRF header.

    The OIDC/local double-submit CSRF check needs both the ``__Host-``
    cookie and the ``X-CSRF-Token`` header to survive the proxy hop.
    """
    received: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["csrf"] = request.headers.get("x-csrf-token")
        received["cookie"] = request.headers.get("cookie")
        received["url"] = str(request.url)
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.post(
            "/api/auth/logout",
            headers={
                "X-CSRF-Token": "csrf-abc",
                "Cookie": "__Host-inqtrix_session=sid; __Host-inqtrix_csrf=csrf-abc",
            },
        )

    assert received["url"] == "http://backend.invalid/api/auth/logout"
    assert received["csrf"] == "csrf-abc"
    assert received["cookie"] == "__Host-inqtrix_session=sid; __Host-inqtrix_csrf=csrf-abc"


def test_proxy_relays_duplicate_set_cookie_fields(fake_dist: Path) -> None:
    """Duplicate ``Set-Cookie`` fields reach the client as separate headers.

    The local/OIDC login answers with TWO cookies (session + CSRF token).
    Folding them into one comma-joined field — the natural result of a
    dict-shaped header relay — makes every browser drop the CSRF cookie:
    a ``Set-Cookie`` line is parsed as exactly ONE cookie, and the folded
    remainder degrades to an unknown attribute. Without the CSRF cookie
    every subsequent cookie-authenticated non-GET request fails with 403.
    """
    session_cookie = (
        "__Host-inqtrix_session=sess; HttpOnly; Path=/; SameSite=lax; Secure"
    )
    csrf_cookie = "__Host-inqtrix_csrf=token; Path=/; SameSite=lax; Secure"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({"authenticated": True}).encode("utf-8")
        return httpx.Response(
            200,
            stream=httpx.ByteStream(body),
            headers=[
                ("content-type", "application/json"),
                ("content-length", str(len(body))),
                ("set-cookie", session_cookie),
                ("set-cookie", csrf_cookie),
            ],
        )

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        response = client.post("/api/auth/login/local", json={})

    assert response.status_code == 200
    assert response.headers.get_list("set-cookie") == [
        session_cookie,
        csrf_cookie,
    ]


def test_proxy_forwards_workspace_header(fake_dist: Path) -> None:
    """The X-Inqtrix-Workspace-Id header is passed through verbatim."""
    received: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["workspace"] = request.headers.get("x-inqtrix-workspace-id")
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.get(
            "/v1/runs",
            headers={"X-Inqtrix-Workspace-Id": "ws-abc-123"},
        )

    assert received["workspace"] == "ws-abc-123"


def test_unknown_path_falls_back_to_index_html(fake_dist: Path) -> None:
    """Deep links serve ``index.html`` (nginx ``try_files`` parity).

    Plain ``StaticFiles(html=True)`` answers 404 here; the SPA must load
    on any client-side route, so the launcher has to mirror the nginx
    fallback.
    """
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        response = client.get("/settings/deep/link")
    assert response.status_code == 200
    assert "fake-react-app" in response.text
    assert response.headers["cache-control"] == "no-cache"


def test_missing_asset_stays_hard_404(fake_dist: Path) -> None:
    """A missing hashed bundle surfaces as 404, never as ``index.html``.

    Falling back under ``/assets/`` would hand HTML to the browser's
    module loader and mask a stale-``index.html`` deployment problem.
    """
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        response = client.get("/assets/gone-abc123.js")
    assert response.status_code == 404
    assert "fake-react-app" not in response.text


def test_cache_policy_splits_index_and_assets(fake_dist: Path) -> None:
    """``index.html`` revalidates per load, hashed assets cache forever."""
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        index = client.get("/")
        asset = client.get("/assets/main.js")
    assert index.headers["cache-control"] == "no-cache"
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_proxy_adds_forwarding_headers(fake_dist: Path) -> None:
    """Proxied requests carry the client address and scheme.

    The backend's login rate limiter keys on ``X-Forwarded-For``;
    without these headers every browser behind the launcher would share
    one bucket.
    """
    received: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["xff"] = request.headers.get("x-forwarded-for")
        received["real_ip"] = request.headers.get("x-real-ip")
        received["proto"] = request.headers.get("x-forwarded-proto")
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.get("/v1/runs")

    assert received["xff"] == "testclient"
    assert received["real_ip"] == "testclient"
    assert received["proto"] == "http"


def test_proxy_extends_forwarded_chain(fake_dist: Path) -> None:
    """Client-chain metadata extends, but untrusted origin headers do not."""
    received: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["xff"] = request.headers.get("x-forwarded-for")
        received["proto"] = request.headers.get("x-forwarded-proto")
        received["forwarded_host"] = request.headers.get("x-forwarded-host")
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.get(
            "/v1/runs",
            headers={
                "X-Forwarded-For": "203.0.113.7",
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "attacker.example",
            },
        )

    assert received["xff"] == "203.0.113.7, testclient"
    assert received["proto"] == "http"
    assert received["forwarded_host"] == "testserver"


def test_proxy_uses_explicit_public_origin_for_http_forwarding(
    fake_dist: Path,
) -> None:
    """HTTP and WebSocket paths share one configured TLS trust anchor."""
    received: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["host"] = request.headers.get("host")
        received["proto"] = request.headers.get("x-forwarded-proto")
        received["forwarded_host"] = request.headers.get("x-forwarded-host")
        return _json_response(200, {})

    app = launcher.build_app(
        fake_dist,
        "http://backend.invalid",
        transport=httpx.MockTransport(handler),
        public_base_url="https://Desk.Example:8443/",
    )

    with TestClient(app) as client:
        client.get(
            "/v1/runs",
            headers={
                "Host": "launcher.internal:8080",
                "X-Forwarded-Proto": "http",
                "X-Forwarded-Host": "attacker.example",
            },
        )

    assert received == {
        "host": "desk.example:8443",
        "proto": "https",
        "forwarded_host": "desk.example:8443",
    }


def test_backend_unreachable_maps_to_502(fake_dist: Path) -> None:
    """A connect failure answers 502 Bad Gateway like nginx, not a 500."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        response = client.get("/v1/runs")

    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"]
    assert "backend.invalid" not in response.text


def test_spa_fallback_survives_present_404_html(fake_dist: Path) -> None:
    """A ``404.html`` in dist/ must not hijack the SPA fallback.

    With ``html=True`` Starlette RETURNS the 404 page for unknown paths
    instead of raising, which would silently bypass the ``index.html``
    fallback; the launcher therefore mounts with ``html=False``.
    """
    (fake_dist / "404.html").write_text("custom not-found page", encoding="utf-8")
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        response = client.get("/settings/deep/link")
    assert response.status_code == 200
    assert "fake-react-app" in response.text


def test_mjs_assets_served_as_javascript(fake_dist: Path) -> None:
    """ES-module assets get a JavaScript MIME type on every platform.

    Browsers hard-reject module scripts served as octet-stream; the
    launcher registers the ``.mjs`` mapping itself instead of trusting
    the host's mimetypes database (nginx pins a default_type for the
    same reason).
    """
    (fake_dist / "assets" / "worker.mjs").write_text(
        "export {};", encoding="utf-8"
    )
    app = launcher.build_app(fake_dist, "http://backend.invalid")
    with TestClient(app) as client:
        response = client.get("/assets/worker.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_proxy_forwards_original_host(fake_dist: Path) -> None:
    """The browser's Host header reaches the backend (nginx ``$host``)."""
    received: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["host"] = request.headers.get("host")
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.get("/v1/runs")

    assert received["host"] == "testserver"


def test_proxy_preserves_raw_target(fake_dist: Path) -> None:
    """Encoded path segments and repeated query keys survive verbatim.

    Rebuilding the URL from the decoded route parameter would turn
    ``a%2Fb`` into ``a/b`` and collapse ``tag=1&tag=2`` to the last
    value; nginx forwards the raw request target and so must the
    launcher.
    """
    received: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.get("/v1/items/a%2Fb?tag=1&tag=2")

    assert received["url"] == "http://backend.invalid/v1/items/a%2Fb?tag=1&tag=2"


def test_proxy_streams_request_body(fake_dist: Path) -> None:
    """Upload bodies reach the backend through the streaming path."""
    received: dict[str, bytes] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        received["body"] = await request.aread()
        return _json_response(200, {})

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        client.post("/v1/files", content=b"chunk-one chunk-two")

    assert received["body"] == b"chunk-one chunk-two"


class _AbortingStream(httpx.AsyncByteStream):
    """Upstream body that dies after the first chunk (backend restart)."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"data: one\n\n"
        raise httpx.ReadError("upstream reset")

    async def aclose(self) -> None:
        self.closed = True


def test_midstream_abort_ends_quietly_and_closes_upstream(
    fake_dist: Path,
) -> None:
    """A backend dying mid-stream truncates the body and frees the upstream.

    Without the relay guard the ReadError escapes as an unhandled ASGI
    exception and the upstream response is never closed (pooled
    connection leak on every dropped SSE stream).
    """
    stream = _AbortingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    app = launcher.build_app(
        fake_dist, "http://backend.invalid", transport=transport
    )

    with TestClient(app) as client:
        response = client.get("/v1/runs/xyz/events")

    assert response.status_code == 200
    assert response.text == "data: one\n\n"
    assert stream.closed


def test_resolve_dist_dir_missing_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dist/ directory triggers a loud RuntimeError."""
    monkeypatch.setenv("INQTRIX_DIST_DIR", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError, match="dist/ not found"):
        launcher._resolve_dist_dir()


def test_resolve_dist_dir_explicit_path(
    fake_dist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit INQTRIX_DIST_DIR overrides the repository default."""
    monkeypatch.setenv("INQTRIX_DIST_DIR", str(fake_dist))
    resolved = launcher._resolve_dist_dir()
    assert resolved == fake_dist.resolve()


def test_filter_headers_drops_hop_by_hop() -> None:
    """The hop-by-hop filter removes connection-management headers."""
    raw = {
        "Authorization": "Bearer x",
        "Host": "example.com",
        "Connection": "keep-alive",
        "Content-Length": "42",
        "X-Custom": "passthrough",
    }
    filtered = launcher._filter_headers(raw)
    assert "X-Custom" in filtered
    assert "Authorization" in filtered
    assert "Host" not in filtered
    assert "Connection" not in filtered
    assert "Content-Length" not in filtered
