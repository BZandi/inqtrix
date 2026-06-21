"""Unit tests for ``scripts/run_research_desk.py``.

The launcher is a standalone script outside the ``inqtrix`` package
namespace, so the module is loaded by absolute path through
``importlib`` rather than imported directly. Tests cover the three
behaviours that matter for deployment correctness: static-mount
fallback, streaming proxy routing, and loud failure on a missing
``dist/`` directory.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


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
