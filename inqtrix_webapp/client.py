"""Thin synchronous HTTP client for the Inqtrix OpenAI-compatible server.

This module is consumed by ``webapp.py`` (Streamlit UI). It deliberately
avoids importing anything from the ``inqtrix`` package: the UI is a
pure HTTP consumer and must stay deployable against a remote server.

Endpoints covered:

* ``GET  /health``                 — readiness probe, drives the sidebar badge.
* ``GET  /v1/models``              — single-stack fallback for discovery.
* ``GET  /v1/stacks``              — multi-stack discovery (5 s server cache).
* ``POST /v1/chat/completions``    — blocking + SSE streaming chat.

The SSE parser understands the chunk shape produced by
:mod:`inqtrix.server.streaming`: progress markers are emitted as
``f"> \\`{msg}\\`\\n>\\n"`` content deltas and are teased apart from
answer tokens so the UI can render progress inside an ``st.status``
container while streaming the answer into ``st.write_stream``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterator, Literal

import httpx

log = logging.getLogger("inqtrix_webapp")

_DEFAULT_BASE_URL = "http://localhost:5100"
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=1800.0, write=30.0, pool=5.0)
_DEFAULT_SHORT_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)

_PROGRESS_RE = re.compile(r"^>\s`([^`]+)`\s*\n?>?\s*\n?$")


# ------------------------------------------------------------------ #
# Environment accessors
# ------------------------------------------------------------------ #


def get_base_url() -> str:
    """Return the Inqtrix server base URL, without trailing slash."""
    raw = os.environ.get("INQTRIX_WEBAPP_BASE_URL", "").strip() or _DEFAULT_BASE_URL
    return raw.rstrip("/")


def get_api_key() -> str | None:
    """Return the bearer API key if ``INQTRIX_WEBAPP_API_KEY`` is set."""
    key = os.environ.get("INQTRIX_WEBAPP_API_KEY", "").strip()
    return key or None


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


# ------------------------------------------------------------------ #
# Discovery endpoints
# ------------------------------------------------------------------ #


def fetch_health(base_url: str | None = None) -> dict[str, Any]:
    """Call ``GET /health``.

    Returns a dict shaped like ``{"status": "ok"|"degraded"|"unreachable",
    "http_status": int, "llm": {...}, "search": {...}, ...}``. Never
    raises — network errors surface as ``status="unreachable"``.
    """
    url = f"{base_url or get_base_url()}/health"
    try:
        with httpx.Client(timeout=_DEFAULT_SHORT_TIMEOUT) as client:
            resp = client.get(url)
        payload: dict[str, Any] = {}
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001 — health endpoint may return text
            payload = {}
        payload["http_status"] = resp.status_code
        if resp.status_code == 200 and payload.get("status") != "degraded":
            payload.setdefault("status", "ok")
        elif "status" not in payload:
            payload["status"] = "degraded"
        return payload
    except Exception as exc:  # noqa: BLE001 — health must not crash the UI
        log.debug("Health probe failed: %s", exc)
        return {"status": "unreachable", "error": str(exc), "http_status": 0}


def fetch_stacks(base_url: str | None = None) -> dict[str, Any] | None:
    """Call ``GET /v1/stacks`` (multi-stack discovery).

    Returns the parsed payload ``{"default": str, "stacks": [...]}`` or
    ``None`` when the server is single-stack (endpoint returns 404) or
    unreachable.
    """
    url = f"{base_url or get_base_url()}/v1/stacks"
    try:
        with httpx.Client(timeout=_DEFAULT_SHORT_TIMEOUT) as client:
            resp = client.get(url)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            log.debug("Stacks endpoint returned %s", resp.status_code)
            return None
        data = resp.json()
        if not isinstance(data, dict) or "stacks" not in data:
            return None
        return data
    except Exception as exc:  # noqa: BLE001
        log.debug("Stacks discovery failed: %s", exc)
        return None


def fetch_models_fallback(base_url: str | None = None) -> list[str]:
    """Call ``GET /v1/models`` — used when ``/v1/stacks`` is 404."""
    url = f"{base_url or get_base_url()}/v1/models"
    try:
        with httpx.Client(timeout=_DEFAULT_SHORT_TIMEOUT) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [entry.get("id", "") for entry in data.get("data", []) if entry.get("id")]
    except Exception as exc:  # noqa: BLE001
        log.debug("Models fallback failed: %s", exc)
        return []


# ------------------------------------------------------------------ #
# Chat completions
# ------------------------------------------------------------------ #


def _build_chat_body(
    messages: list[dict[str, Any]],
    *,
    stack: str | None,
    agent_overrides: dict[str, Any] | None,
    include_progress: bool,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": messages,
        "stream": stream,
        "include_progress": include_progress,
    }
    if stack:
        body["stack"] = stack
    if agent_overrides:
        cleaned = {k: v for k, v in agent_overrides.items() if v is not None}
        if cleaned:
            body["agent_overrides"] = cleaned
    return body


def _extract_error(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return f"HTTP {resp.status_code}: {resp.text[:500]}"
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict) and "message" in err:
        msg = err["message"]
        available = err.get("available_stacks")
        if available:
            return f"{msg} (verfügbar: {', '.join(available)})"
        return str(msg)
    return f"HTTP {resp.status_code}: {resp.text[:500]}"


def call_chat(
    messages: list[dict[str, Any]],
    *,
    stack: str | None = None,
    agent_overrides: dict[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Call ``POST /v1/chat/completions`` without streaming.

    Returns the OpenAI-compatible response dict on success, or a
    ``{"error": {"message": ...}}`` dict on failure. The UI decides
    how to surface errors.
    """
    url = f"{base_url or get_base_url()}/v1/chat/completions"
    body = _build_chat_body(
        messages,
        stack=stack,
        agent_overrides=agent_overrides,
        include_progress=False,
        stream=False,
    )
    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            resp = client.post(url, json=body, headers=headers)
    except Exception as exc:  # noqa: BLE001
        return {"error": {"message": f"Verbindungsfehler: {exc}"}}
    if resp.status_code != 200:
        return {"error": {"message": _extract_error(resp)}}
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": {"message": f"Antwort nicht parsebar: {exc}"}}


def _parse_progress_content(content: str) -> str | None:
    """Detect the progress-chunk shape produced by ``inqtrix.server.streaming``.

    Progress chunks look like ``"> `<msg>`\\n>\\n"``. Everything else
    (answer tokens, error messages, the ``\\n\\n---\\n\\n`` separator)
    is treated as a normal delta.
    """
    match = _PROGRESS_RE.match(content)
    if match is None:
        return None
    return match.group(1).strip()


def _iter_sse_events(resp: httpx.Response) -> Iterator[str]:
    """Yield the raw ``data:`` payloads from an SSE response."""
    buffer: list[str] = []
    for line in resp.iter_lines():
        if line == "":
            if buffer:
                yield "\n".join(buffer)
                buffer = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())
    if buffer:
        yield "\n".join(buffer)


def stream_chat(
    messages: list[dict[str, Any]],
    *,
    stack: str | None = None,
    agent_overrides: dict[str, Any] | None = None,
    include_progress: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Iterator[tuple[Literal["progress", "delta", "error", "done"], str]]:
    """Stream a chat completion via SSE.

    Yields ``(kind, payload)`` tuples:

    * ``("progress", "<msg>")``  — a single progress line from the agent
      (``plan.round_0``, ``search.round_0.q1`` etc.).
    * ``("delta", "<text>")``    — a token/chunk of the final answer.
    * ``("error", "<msg>")``     — a transport or server error; the stream
      terminates after yielding this.
    * ``("done", "")``           — clean terminator (not strictly needed by
      callers but emitted once after the final chunk so the UI can close
      its ``st.status`` container reliably).
    """
    url = f"{base_url or get_base_url()}/v1/chat/completions"
    body = _build_chat_body(
        messages,
        stack=stack,
        agent_overrides=agent_overrides,
        include_progress=include_progress,
        stream=True,
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **_auth_headers(api_key),
    }
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    resp.read()
                    yield ("error", _extract_error(resp))
                    return
                for raw in _iter_sse_events(resp):
                    if raw == "[DONE]":
                        yield ("done", "")
                        return
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = evt.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if not content:
                        continue
                    progress_msg = _parse_progress_content(content)
                    if progress_msg is not None:
                        yield ("progress", progress_msg)
                    else:
                        yield ("delta", content)
    except httpx.ReadTimeout:
        yield ("error", "Zeitüberschreitung beim Lesen der Antwort")
    except httpx.ConnectError as exc:
        yield ("error", f"Server nicht erreichbar: {exc}")
    except Exception as exc:  # noqa: BLE001 — UI must survive any upstream error
        yield ("error", f"Unerwarteter Fehler: {exc}")
