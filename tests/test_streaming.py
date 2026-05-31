"""Tests for SSE streaming helpers."""

from __future__ import annotations

import json
import time

import pytest

from inqtrix.settings import AgentSettings
from inqtrix.server.streaming import guarded_stream, stream_response


@pytest.mark.asyncio
async def test_stream_response_includes_progress_by_default(monkeypatch):
    import inqtrix.server.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        providers,
        strategies,
        settings,
        cancel_event=None,
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
            providers=None,
            strategies=None,
            settings=AgentSettings(),
        )
    ]

    assert any("Plane Suchanfragen (Runde 1/4)..." in chunk for chunk in chunks)
    assert any("---" in chunk for chunk in chunks)
    assert any("Hallo " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_response_can_omit_progress(monkeypatch):
    import inqtrix.server.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        providers,
        strategies,
        settings,
        cancel_event=None,
    ):
        assert progress_queue is None
        return {"answer": "Hallo Welt", "result_state": {}}

    monkeypatch.setattr(streaming_module, "agent_run", fake_run)

    chunks = [
        chunk
        async for chunk in stream_response(
            "Meine Frage",
            "",
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            include_progress=False,
        )
    ]

    assert not any("> `" in chunk for chunk in chunks)
    assert not any("---" in chunk for chunk in chunks)
    assert any("Hallo " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_response_emits_model_resolution_metadata(monkeypatch):
    import inqtrix.server.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        providers,
        strategies,
        settings,
        cancel_event=None,
    ):
        return {
            "answer": "Hallo Welt",
            "result_state": {
                "node_model_resolutions": {
                    "direct_chat": {
                        "node": "direct_chat",
                        "model": "F",
                        "tier": "fast",
                        "effort": "none",
                        "model_source": "tier:fast",
                        "effort_source": "tier:fast",
                        "requested_tier": "fast",
                    }
                }
            },
        }

    monkeypatch.setattr(streaming_module, "agent_run", fake_run)

    chunks = [
        chunk
        async for chunk in stream_response(
            "Meine Frage",
            "",
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            include_progress=False,
        )
    ]

    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: {")
    ]
    metadata_payload = next(payload for payload in payloads if "inqtrix" in payload)
    assert metadata_payload["inqtrix"]["model_resolution"]["model"] == "F"
    answer_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload.get("choices", [{}])[0].get("delta", {}).get("content")
    )
    metadata_index = payloads.index(metadata_payload)
    assert metadata_index < answer_index


@pytest.mark.asyncio
async def test_guarded_stream_passes_include_progress(monkeypatch):
    import asyncio
    import inqtrix.server.streaming as streaming_module

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
            asyncio.Semaphore(1),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            include_progress=False,
        )
    ]

    assert captured["include_progress"] is False
    assert chunks == ["data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_stream_response_returns_timeout_chunk(monkeypatch):
    import inqtrix.server.streaming as streaming_module

    def fake_run(
        question,
        *,
        history,
        progress_queue,
        providers,
        strategies,
        settings,
        cancel_event=None,
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
            providers=None,
            strategies=None,
            settings=settings,
        )
    ]

    assert any("Request-Timeout erreicht" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
