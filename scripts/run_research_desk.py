"""Serve the Research Desk via uvicorn with a streaming backend proxy.

Standalone Python launcher: serves the pre-built React frontend at
``apps/research-desk/dist/`` plus a streaming reverse-proxy for the
Inqtrix backend (``/v1/*`` and ``/health``). The browser sees a single
origin, so the React bundle works without ``VITE_INQTRIX_API_BASE_URL``
and without CORS on the backend.

Use this launcher when the target host has only Python available
(no nginx, no Docker), when verifying the production build locally
without container tooling, or while prototyping a single-process
deployment before adding nginx. For real two-pod production prefer
the nginx pattern documented in ``docs/deployment/react-ui.md``.

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
- ``INQTRIX_DIST_DIR``    Override the ``dist/`` location (default:
  the ``apps/research-desk/dist`` directory inside this repository).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("research-desk-launcher")

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
be forwarded by an intermediary. ``host`` and ``content-length`` are
re-derived by httpx for the upstream request and would otherwise collide."""

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


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

    Used symmetrically on the request (client to upstream) and response
    (upstream to client) paths so that connection management headers
    are not leaked through the proxy.

    Args:
        headers: Header mapping from a Starlette ``Request`` or an
            ``httpx.Response``.

    Returns:
        Plain ``dict`` containing only headers safe to forward.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def build_app(
    dist_dir: Path,
    backend_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Wire the launcher FastAPI app: streaming proxy plus static mount.

    The app registers the API proxy routes first and then mounts the
    static directory at ``/``. The mount uses ``html=True`` so unknown
    paths inside the SPA serve ``index.html``. Route registration order
    matters: API paths must win over the static catch-all.

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

    Returns:
        A wired :class:`FastAPI` instance with a lifespan that closes
        the shared ``httpx.AsyncClient`` on shutdown. The instance is
        intended for ``uvicorn.run`` or :class:`fastapi.testclient.TestClient`.
    """
    client = httpx.AsyncClient(
        base_url=backend_url,
        timeout=None,
        transport=transport,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "Serving %s and proxying /v1, /health to %s",
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

    async def _proxy(request: Request, upstream_path: str) -> StreamingResponse:
        upstream = client.build_request(
            method=request.method,
            url=httpx.URL(upstream_path, params=request.query_params),
            headers=_filter_headers(request.headers),
            content=await request.body(),
        )
        response = await client.send(upstream, stream=True)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=_filter_headers(response.headers),
            media_type=response.headers.get("content-type"),
            background=response.aclose,
        )

    @app.get("/health")
    async def proxy_health(request: Request) -> StreamingResponse:
        return await _proxy(request, "/health")

    @app.api_route("/v1/{full_path:path}", methods=_PROXY_METHODS)
    async def proxy_v1(full_path: str, request: Request) -> StreamingResponse:
        return await _proxy(request, f"/v1/{full_path}")

    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

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
    app = build_app(dist_dir, backend_url)
    uvicorn.run(
        app,
        host=os.getenv("RESEARCH_DESK_HOST", "127.0.0.1"),
        port=int(os.getenv("RESEARCH_DESK_PORT", "8080")),
        log_level="info",
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
