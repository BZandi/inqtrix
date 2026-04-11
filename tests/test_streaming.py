"""Tests for SSE streaming helpers."""

from __future__ import annotations

from types import SimpleNamespace
import time

import pytest

from inqtrix.settings import AgentSettings
from inqtrix.streaming import guarded_stream, stream_response


@pytest.mark.asyncio
async def test_stream_response_includes_progress_by_default(monkeypatch):
    import inqtrix.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        prev_session,
        providers,
        strategies,
        settings,
    ):
        assert progress_queue is not None
        progress_queue.put(("progress", "Plane Suchanfragen (Runde 1/4)..."))
        return {"answer": "Hallo Welt", "result_state": {}}

    monkeypatch.setattr(streaming_module, "agent_run", fake_run)

    chunks = [
        chunk
        async for chunk in stream_response(
            "Meine Frage",
            "",
            None,
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            session_store=SimpleNamespace(save=lambda *a, **kw: None),
        )
    ]

    assert any("Plane Suchanfragen (Runde 1/4)..." in chunk for chunk in chunks)
    assert any("---" in chunk for chunk in chunks)
    assert any("Hallo " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_response_can_omit_progress(monkeypatch):
    import inqtrix.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        prev_session,
        providers,
        strategies,
        settings,
    ):
        assert progress_queue is None
        return {"answer": "Hallo Welt", "result_state": {}}

    monkeypatch.setattr(streaming_module, "agent_run", fake_run)

    chunks = [
        chunk
        async for chunk in stream_response(
            "Meine Frage",
            "",
            None,
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            session_store=SimpleNamespace(save=lambda *a, **kw: None),
            include_progress=False,
        )
    ]

    assert not any("> `" in chunk for chunk in chunks)
    assert not any("---" in chunk for chunk in chunks)
    assert any("Hallo " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_guarded_stream_passes_include_progress(monkeypatch):
    import asyncio
    import inqtrix.streaming as streaming_module

    captured: dict[str, object] = {}

    async def fake_stream_response(*args, **kwargs):
        captured["include_progress"] = kwargs["include_progress"]
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(streaming_module, "stream_response", fake_stream_response)

    chunks = [
        chunk
        async for chunk in guarded_stream(
            "Meine Frage",
            "",
            None,
            asyncio.Semaphore(1),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            session_store=SimpleNamespace(save=lambda *a, **kw: None),
            include_progress=False,
        )
    ]

    assert captured["include_progress"] is False
    assert chunks == ["data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_stream_response_returns_timeout_chunk(monkeypatch):
    import inqtrix.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        prev_session,
        providers,
        strategies,
        settings,
    ):
        time.sleep(1.2)
        return {"answer": "Zu spaet", "result_state": {}}

    monkeypatch.setattr(streaming_module, "agent_run", fake_run)

    settings = AgentSettings()
    settings.max_total_seconds = -29

    chunks = [
        chunk
        async for chunk in stream_response(
            "Meine Frage",
            "",
            None,
            providers=None,
            strategies=None,
            settings=settings,
            session_store=SimpleNamespace(save=lambda *a, **kw: None),
        )
    ]

    assert any("Request-Timeout erreicht" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
