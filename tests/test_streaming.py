"""Tests for SSE streaming helpers.

Streaming now dispatches through the AlgorithmRegistry: ``stream_response`` runs
``algorithm.run`` with a per-request ``RunContext``. Graph-backed modes reach the
LangGraph engine at the single seam ``inqtrix.research.web_research.run_web_graph``
(the same one the non-streamed path and native runs use), so these tests patch
THAT global and dispatch a real ``WebResearchAlgorithm``. The wire assertions
(progress blockquote, ``---`` separator, model_resolution chunk, ``[DONE]``) pin
that the OpenAI SSE output stays byte-stable across the convergence.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from inqtrix.core.results import RunRequest
from inqtrix.research.web_research import WebResearchAlgorithm
from inqtrix.server.streaming import guarded_stream, stream_response
from inqtrix.settings import AgentSettings


def _run_request(question: str = "Meine Frage") -> RunRequest:
    return RunRequest(mode="research", question=question, history="")


async def _collect(gen) -> list[str]:
    return [chunk async for chunk in gen]


@pytest.mark.asyncio
async def test_stream_response_includes_progress_by_default(monkeypatch):
    import inqtrix.research.web_research as web_research_module

    def fake_run_web_graph(question, **kwargs):
        progress_queue = kwargs.get("progress_queue")
        assert progress_queue is not None
        progress_queue.put(("progress", "Plane Suchanfragen (Runde 1/4)..."))
        return {"answer": "Hallo Welt", "result_state": {}}

    monkeypatch.setattr(web_research_module, "run_web_graph", fake_run_web_graph)

    chunks = await _collect(
        stream_response(
            "Meine Frage",
            algorithm=WebResearchAlgorithm(),
            runtime=None,
            run_request=_run_request(),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
        )
    )

    assert any("Plane Suchanfragen (Runde 1/4)..." in chunk for chunk in chunks)
    assert any("---" in chunk for chunk in chunks)
    assert any("Hallo " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_response_can_omit_progress(monkeypatch):
    import inqtrix.research.web_research as web_research_module

    def fake_run_web_graph(question, **kwargs):
        assert kwargs.get("progress_queue") is None
        return {"answer": "Hallo Welt", "result_state": {}}

    monkeypatch.setattr(web_research_module, "run_web_graph", fake_run_web_graph)

    chunks = await _collect(
        stream_response(
            "Meine Frage",
            algorithm=WebResearchAlgorithm(),
            runtime=None,
            run_request=_run_request(),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            include_progress=False,
        )
    )

    assert not any("> `" in chunk for chunk in chunks)
    assert not any("---" in chunk for chunk in chunks)
    assert any("Hallo " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_response_emits_model_resolution_metadata(monkeypatch):
    import inqtrix.research.web_research as web_research_module

    def fake_run_web_graph(question, **kwargs):
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

    monkeypatch.setattr(web_research_module, "run_web_graph", fake_run_web_graph)

    chunks = await _collect(
        stream_response(
            "Meine Frage",
            algorithm=WebResearchAlgorithm(),
            runtime=None,
            run_request=_run_request(),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            include_progress=False,
        )
    )

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
    import inqtrix.server.streaming as streaming_module

    captured: dict[str, object] = {}

    async def fake_stream_response(*args, **kwargs):
        captured["include_progress"] = kwargs["include_progress"]
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(streaming_module, "stream_response", fake_stream_response)

    chunks = await _collect(
        guarded_stream(
            "Meine Frage",
            "",
            asyncio.Semaphore(1),
            algorithm=WebResearchAlgorithm(),
            runtime=None,
            run_request=_run_request(),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
            include_progress=False,
        )
    )

    assert captured["include_progress"] is False
    assert chunks == ["data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_stream_response_dispatches_event_only_algorithm_answer_only():
    """An event-only algorithm (the knowledge-style path) streams its answer with
    NO granular progress lines: the chat surface wires no event_sink, so a mode
    that emits only structured events surfaces the answer + [DONE] and nothing
    between. Proves streaming dispatches ANY registry algorithm, not just the
    graph-backed ones — the core of the registry-first convergence.
    """
    from inqtrix.core.results import AgentResult

    class _EventOnlyAlgorithm:
        """Mimics KnowledgeAlgorithm: emits only via event_sink (unused here),
        never touches progress_queue, returns a full AgentResult synchronously."""

        def capabilities(self) -> dict:
            return {"supports_chat_completions": True}

        def run(self, request, *, runtime, context):
            # No progress_queue writes; event_sink is None on the chat surface.
            assert context.event_sink is None
            return AgentResult(
                answer="Belegte Antwort.",
                result_type="knowledge_result",
                raw={"answer": "Belegte Antwort.", "result_state": {}, "usage": {}},
            )

    chunks = await _collect(
        stream_response(
            "Meine Frage",
            algorithm=_EventOnlyAlgorithm(),
            runtime=None,
            run_request=_run_request(),
            providers=None,
            strategies=None,
            settings=AgentSettings(),
        )
    )

    assert not any("> `" in chunk for chunk in chunks)  # no progress lines
    assert not any("---" in chunk for chunk in chunks)  # and no separator
    assert any("Belegte " in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_response_returns_timeout_chunk(monkeypatch):
    import inqtrix.research.web_research as web_research_module

    def fake_run_web_graph(question, **kwargs):
        time.sleep(1.2)
        return {"answer": "Zu spaet", "result_state": {}}

    monkeypatch.setattr(web_research_module, "run_web_graph", fake_run_web_graph)

    settings = AgentSettings()
    settings.max_total_seconds = -29

    chunks = await _collect(
        stream_response(
            "Meine Frage",
            algorithm=WebResearchAlgorithm(),
            runtime=None,
            run_request=_run_request(),
            providers=None,
            strategies=None,
            settings=settings,
        )
    )

    assert any("Request-Timeout erreicht" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
