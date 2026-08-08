"""Outermost ASGI middleware binding the per-request log context.

Every HTTP request gets a request id — an incoming ``X-Request-ID``
(validated, so a client cannot inject log content) or a fresh hex uuid.
The id is bound into :mod:`inqtrix.observability.context` for the
lifetime of the request (so EVERY log line inside carries it, including
error paths — the reason this middleware must wrap all others) and is
echoed as an ``X-Request-ID`` response header so a support report can
quote the exact id to grep for.

The middleware additionally binds the request ORIGIN facts (client ip
resolved with the same trusted-proxy-hops policy as the login throttle,
plus the user agent) into :mod:`inqtrix.auth.request_origin` — read
ONLY by audit writers, never by the log formatter (PII stays out of
operational logs).

It also opens the SERVER span for the request. That span is what makes
the trace chain real: without an active span, ``inject_traceparent``
writes nothing into the persisted run payload, so the worker would open
a disconnected root span and "one trace from request to LLM call" would
be a claim rather than a fact. An incoming ``traceparent`` is extracted
first, so an upstream caller's trace continues through Inqtrix. Probe
and scrape endpoints are excluded — they would otherwise dominate the
trace volume without ever being read.

Pure ASGI on purpose: no BaseHTTPMiddleware task hop, no body access,
nothing measurable on the hot path.
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from starlette.datastructures import Headers, MutableHeaders

from inqtrix.auth.ratelimit import client_ip
from inqtrix.auth.request_origin import (
    bind_request_origin,
    reset_request_origin,
)
from inqtrix.observability.context import bind_log_context, reset_log_context

# Conservative charset/length: anything else is replaced, never echoed.
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

# Liveness/readiness/scrape traffic is high-frequency and carries no
# diagnostic value in a trace waterfall.
_UNTRACED_PATHS = frozenset({"/health", "/readyz", "/metrics"})

# Service-to-service coordination (collaboration policy polling, lease
# renewal, compaction). It is continuous background chatter — measured
# live, ONE knowledge run's window carried 40 such spans against 21 for
# the run itself, which would bury every user-initiated trace.
_UNTRACED_PREFIXES = ("/internal/",)


class RequestContextMiddleware:
    """Bind ``request_id`` + origin for the request; echo the id back."""

    def __init__(
        self, app: Callable[..., Any], *, trusted_proxy_hops: int = 1
    ) -> None:
        self.app = app
        self._trusted_proxy_hops = trusted_proxy_hops

    async def __call__(
        self, scope: dict, receive: Callable[..., Any], send: Callable[..., Any]
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        incoming = headers.get("x-request-id", "")
        request_id = (
            incoming
            if _REQUEST_ID_PATTERN.fullmatch(incoming)
            else uuid.uuid4().hex
        )
        tokens = bind_log_context(request_id=request_id)
        # client_ip expects a Request-shaped object; a namespace over the
        # raw scope avoids constructing a full Request on the hot path
        # while reusing the ONE XFF-from-the-right implementation.
        client = scope.get("client")
        peer = (
            SimpleNamespace(host=client[0])
            if isinstance(client, (tuple, list)) and client
            else None
        )
        origin_tokens = bind_request_origin(
            ip=client_ip(
                SimpleNamespace(client=peer, headers=headers),
                self._trusted_proxy_hops,
            ),
            user_agent=headers.get("user-agent", ""),
        )

        async def send_with_request_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(
                    "X-Request-ID", request_id
                )
            await send(message)

        try:
            with _server_span(scope, headers, request_id):
                await self.app(scope, receive, send_with_request_id)
        finally:
            # The server reuses tasks/threads across requests in some
            # execution paths — never let request context leak forward.
            reset_request_origin(origin_tokens)
            reset_log_context(tokens)


@contextmanager
def _server_span(
    scope: dict, headers: Headers, request_id: str
) -> Iterator[None]:
    """Open the request's server span, continuing an upstream trace.

    Degrades to a plain no-op without the observability extra or with
    tracing off — the request path must never depend on telemetry.
    """
    path = scope.get("path", "")
    if path in _UNTRACED_PATHS or path.startswith(_UNTRACED_PREFIXES):
        yield
        return
    try:
        from opentelemetry import trace as otel_trace

        from inqtrix.observability import semconv
        from inqtrix.observability.propagation import (
            extract_incoming_context,
        )
    except Exception:  # noqa: BLE001 — extra not installed
        yield
        return
    parent = extract_incoming_context(headers)
    method = str(scope.get("method", "") or "")
    tracer = otel_trace.get_tracer("inqtrix.server")
    with tracer.start_as_current_span(
        f"{method} {path}".strip(),
        context=parent,
        kind=otel_trace.SpanKind.SERVER,
        attributes={
            semconv.HTTP_REQUEST_METHOD: method,
            semconv.URL_PATH: path,
            semconv.INQTRIX_REQUEST_ID: request_id,
        },
    ):
        yield
