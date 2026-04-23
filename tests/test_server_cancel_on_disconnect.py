"""Tests for the implicit-cancel-on-disconnect pathway (ADR-MS-5/MS-6)."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import inqtrix.server.routes as routes_module
import inqtrix.server.streaming as streaming_module
from inqtrix.exceptions import AgentCancelled
from inqtrix.providers.base import ProviderContext
from inqtrix.server.app import create_app
from inqtrix.server.session import SessionStore
from inqtrix.server.streaming import stream_response
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings
from inqtrix.state import check_cancel_event, initial_state


# ------------------------------------------------------------------ #
# state.check_cancel_event — unit
# ------------------------------------------------------------------ #


def test_check_cancel_event_noop_when_field_missing():
    state = {"question": "x"}
    check_cancel_event(state)  # must not raise


def test_check_cancel_event_noop_when_event_unset():
    event = threading.Event()
    state = {"question": "x", "_cancel_event": event}
    check_cancel_event(state)  # must not raise


def test_check_cancel_event_raises_when_set():
    event = threading.Event()
    event.set()
    state = {"question": "x", "_cancel_event": event}
    with pytest.raises(AgentCancelled):
        check_cancel_event(state)


def test_initial_state_accepts_optional_cancel_event():
    event = threading.Event()
    state = initial_state("question", cancel_event=event)
    assert state["_cancel_event"] is event


def test_initial_state_omits_cancel_event_when_none():
    state = initial_state("question")
    assert "_cancel_event" not in state


# ------------------------------------------------------------------ #
# graph.run — cancel handling
# ------------------------------------------------------------------ #


def test_graph_run_returns_cancelled_state_when_cancelled(monkeypatch):
    """Mock the LangGraph compiled agent so .invoke raises AgentCancelled."""
    from inqtrix import graph as graph_module

    class _StubAgent:
        def invoke(self, state):
            raise AgentCancelled("simulated client disconnect")

    monkeypatch.setattr(graph_module, "get_agent", lambda *a, **kw: _StubAgent())

    providers = SimpleNamespace(llm=SimpleNamespace(), search=SimpleNamespace())
    strategies = SimpleNamespace()
    settings = AgentSettings()
    cancel_event = threading.Event()
    cancel_event.set()

    result = graph_module.run(
        "question",
        providers=providers,
        strategies=strategies,
        settings=settings,
        cancel_event=cancel_event,
    )
    assert result["answer"] == ""
    assert result["result_state"]["cancelled"] is True


# ------------------------------------------------------------------ #
# stream_response — disconnect handling
# ------------------------------------------------------------------ #


class _DummyLLM:
    def complete(self, *a, **kw): return "ok"
    def summarize_parallel(self, *a, **kw): return ("", 0, 0)
    def is_available(self): return True


class _DummySearch:
    def search(self, *a, **kw):
        return {"answer": "", "citations": [], "related_questions": [],
                "_prompt_tokens": 0, "_completion_tokens": 0}
    def is_available(self): return True


@pytest.mark.asyncio
async def test_stream_response_passes_cancel_event_to_agent_run(monkeypatch):
    """Verify the cancel_event arrives in agent_run as a kwarg."""
    captured: dict[str, Any] = {}

    def fake_run(question, *, history, progress_queue, prev_session,
                 providers, strategies, settings, cancel_event=None):
        captured["cancel_event"] = cancel_event
        return {"answer": "Hallo Welt", "result_state": {}}

    monkeypatch.setattr(streaming_module, "agent_run", fake_run)

    event = threading.Event()
    chunks = [
        chunk
        async for chunk in stream_response(
            "Frage", "", None,
            providers=None, strategies=None, settings=AgentSettings(),
            session_store=SimpleNamespace(save=lambda *a, **kw: None),
            cancel_event=event,
        )
    ]
    assert captured["cancel_event"] is event
    assert any("Hallo " in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_stream_response_sets_cancel_event_on_disconnect(monkeypatch):
    """Simulate a Request whose receive() yields http.disconnect; cancel_event must be set.

    Replaces the previous polling-based ``is_disconnected`` test: the
    new pathway (ADR-WS-11) spawns a watcher task that blocks on
    ``await request.receive()`` and acts on the first
    ``http.disconnect`` message uvicorn emits. The fake request below
    delays the disconnect by a tiny amount so the streaming loop has a
    chance to enter the progress-read path before the cancel fires.
    """

    def slow_run(question, *, history, progress_queue, prev_session,
                 providers, strategies, settings, cancel_event=None):
        # Simulate a long-running agent: keep emitting progress until cancelled.
        if progress_queue is not None:
            for i in range(200):
                if cancel_event is not None and cancel_event.is_set():
                    break
                progress_queue.put(("progress", f"step {i}"))
                time.sleep(0.05)
        return {"answer": "never delivered", "result_state": {}}

    monkeypatch.setattr(streaming_module, "agent_run", slow_run)

    class _FakeRequest:
        def __init__(self):
            self._delivered = False

        async def receive(self) -> dict:
            # First call: tiny await so the streaming loop can yield the
            # role chunk and start polling. Then deliver http.disconnect.
            if not self._delivered:
                self._delivered = True
                await asyncio.sleep(0.2)
                return {"type": "http.disconnect"}
            # Subsequent receive() calls would normally block forever;
            # the watcher task should have exited after the first
            # disconnect, so this branch is unreachable in practice.
            await asyncio.sleep(60)
            return {"type": "http.disconnect"}

        async def is_disconnected(self) -> bool:
            # Kept for backwards-compat with any caller that might still
            # probe it; the new watcher path does not call this.
            return self._delivered

    event = threading.Event()
    request = _FakeRequest()

    chunks: list[str] = []
    async for chunk in stream_response(
        "Frage", "", None,
        providers=None, strategies=None, settings=AgentSettings(),
        session_store=SimpleNamespace(save=lambda *a, **kw: None),
        request=request,
        cancel_event=event,
    ):
        chunks.append(chunk)

    assert event.is_set(), "Cancel event must be set after detected disconnect"
    # The generator returns early on disconnect; no [DONE] tail expected.
    assert not any("data: [DONE]" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_watch_disconnect_signals_cancel_on_http_disconnect():
    """Direct unit test for the watcher helper: http.disconnect -> cancel_event.set()."""
    from inqtrix.server.streaming import _watch_disconnect

    class _ImmediateDisconnectRequest:
        async def receive(self) -> dict:
            return {"type": "http.disconnect"}

    event = threading.Event()
    await _watch_disconnect(_ImmediateDisconnectRequest(), event)
    assert event.is_set(), "Watcher must set cancel_event on http.disconnect"


@pytest.mark.asyncio
async def test_watch_disconnect_treats_receive_error_as_disconnect():
    """If receive() raises (e.g. transport torn down), watcher must still cancel."""
    from inqtrix.server.streaming import _watch_disconnect

    class _BrokenRequest:
        async def receive(self) -> dict:
            raise RuntimeError("transport gone")

    event = threading.Event()
    await _watch_disconnect(_BrokenRequest(), event)
    assert event.is_set(), (
        "Watcher must set cancel_event when receive() crashes — treating an "
        "unreadable transport as a disconnect avoids burning tokens for a run "
        "whose response can no longer be delivered."
    )


@pytest.mark.asyncio
async def test_watch_disconnect_cancellation_propagates():
    """The watcher must propagate asyncio.CancelledError so cleanup awaits resolve."""
    from inqtrix.server.streaming import _watch_disconnect

    class _SilentRequest:
        async def receive(self) -> dict:
            await asyncio.sleep(60)  # block forever
            return {"type": "http.disconnect"}

    event = threading.Event()
    task = asyncio.create_task(_watch_disconnect(_SilentRequest(), event))
    await asyncio.sleep(0.05)  # let the task enter receive()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not event.is_set(), (
        "Watcher must NOT set cancel_event when its task is externally "
        "cancelled (normal stream completion path) — only when the client "
        "actually disconnects."
    )


@pytest.mark.asyncio
async def test_stream_response_cleans_up_watcher_on_normal_completion(monkeypatch):
    """Normal stream completion must cancel the watcher task without leaking."""
    monkeypatch.setattr(
        streaming_module,
        "agent_run",
        lambda question, *, history, progress_queue, prev_session,
        providers, strategies, settings, cancel_event=None: {
            "answer": "ok", "result_state": {},
        },
    )

    class _NeverDisconnectRequest:
        async def receive(self) -> dict:
            await asyncio.sleep(60)
            return {"type": "http.disconnect"}

    request = _NeverDisconnectRequest()
    chunks: list[str] = []
    async for chunk in stream_response(
        "Frage", "", None,
        providers=None, strategies=None, settings=AgentSettings(),
        session_store=SimpleNamespace(save=lambda *a, **kw: None),
        request=request,
    ):
        chunks.append(chunk)

    # Stream completed normally with [DONE] sentinel; no leftover tasks.
    assert any("data: [DONE]" in chunk for chunk in chunks)
    pending = [
        t for t in asyncio.all_tasks() if "_watch_disconnect" in t.get_coro().__qualname__
    ]
    assert pending == [], f"Watcher task leaked after normal completion: {pending}"


# ------------------------------------------------------------------ #
# Backwards-compat — blocking path is unaffected
# ------------------------------------------------------------------ #


def test_chat_completions_blocking_path_unaffected(monkeypatch):
    """Blocking /v1/chat/completions does not need a cancel_event."""
    def fake_run(question, *, history, prev_session, providers, strategies, settings):
        return {
            "answer": "ok",
            "result_state": {},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    providers = ProviderContext(llm=_DummyLLM(), search=_DummySearch())
    app = create_app(settings=Settings(), providers=providers)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hallo"}], "stream": False},
        )
    assert response.status_code == 200
