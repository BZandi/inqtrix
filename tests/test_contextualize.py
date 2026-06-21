"""Offline tests for ingestion-time chunk contextualization."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inqtrix.knowledge.contextualize import (
    CONTEXT_MARKER_APPLIED,
    CONTEXT_MARKER_FALLBACK,
    LLMChunkContextualizer,
)
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.base import LLMResponse
from inqtrix.services.knowledge_service import KnowledgeService

from tests.test_knowledge_engine import StubEmbeddings


class ScriptedContextLLM:
    """Returns a fixed payload for every contextualization prompt."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.prompts: list[str] = []

    def complete_with_metadata(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        content = (
            self._payload
            if isinstance(self._payload, str)
            else json.dumps(self._payload, ensure_ascii=False)
        )
        return LLMResponse(
            content=content,
            prompt_tokens=10,
            completion_tokens=5,
            model="stub-ctx",
            finish_reason="stop",
        )


def test_contexts_are_prefixed_in_chunk_order():
    llm = ScriptedContextLLM(["Kontext A", "Kontext B"])
    contextualizer = LLMChunkContextualizer(llm)

    result = contextualizer.contextualize(
        document_title="Artikel 26",
        document_text="Volltext...",
        chunks=["Absatz eins.", "Absatz zwei."],
    )

    assert result.marker == CONTEXT_MARKER_APPLIED
    assert result.texts == [
        "Kontext A\n\nAbsatz eins.",
        "Kontext B\n\nAbsatz zwei.",
    ]
    # One batched call per document, never per chunk.
    assert len(llm.prompts) == 1
    assert "CHUNK 2" in llm.prompts[0]


@pytest.mark.parametrize(
    "payload",
    [
        "gar kein JSON",
        ["nur", "ein", "Kontext", "zu", "viel"],
        [1, 2],
        {"falsch": "geformt"},
    ],
)
def test_bad_responses_degrade_loudly_to_raw_chunks(payload, caplog):
    contextualizer = LLMChunkContextualizer(ScriptedContextLLM(payload))

    import logging

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        result = contextualizer.contextualize(
            document_title="T", document_text="x", chunks=["a", "b"]
        )

    assert result.marker == CONTEXT_MARKER_FALLBACK
    assert result.texts == ["a", "b"]
    assert any("Kontextualisierung" in m for m in caplog.messages)


def test_llm_failure_degrades_loudly_not_fatally():
    class BrokenLLM:
        def complete_with_metadata(self, *args: Any, **kwargs: Any):
            raise RuntimeError("endpoint down")

    contextualizer = LLMChunkContextualizer(BrokenLLM())
    result = contextualizer.contextualize(
        document_title="T", document_text="x", chunks=["a"]
    )
    assert result.marker == CONTEXT_MARKER_FALLBACK
    assert result.texts == ["a"]


def test_empty_chunks_short_circuit_without_llm_call():
    llm = ScriptedContextLLM([])
    contextualizer = LLMChunkContextualizer(llm)
    result = contextualizer.contextualize(
        document_title="T", document_text="x", chunks=[]
    )
    assert result.texts == []
    assert llm.prompts == []


# ------------------------------------------------------------------ #
# Service integration
# ------------------------------------------------------------------ #


def make_service(contextualizer=None) -> KnowledgeService:
    return KnowledgeService(
        knowledge=KnowledgeProviderContext(
            embeddings=StubEmbeddings(),
            store=MemoryKnowledgeStore(),
            default_top_k=4,
            contextualizer=contextualizer,
        ),
        chunk_max_chars=2_000,
        max_document_chars=100_000,
    )


@pytest.mark.asyncio
async def test_ingestion_applies_contexts_and_records_the_marker():
    llm = ScriptedContextLLM(["Pflichten der Betreiber, Artikel 26."])
    service = make_service(LLMChunkContextualizer(llm))
    collection = await service.create_collection(name="K")

    document = await service.add_document(
        collection_id=collection.id,
        title="Artikel 26",
        text="Die Betreiber treffen geeignete Massnahmen.",
    )

    assert document.metadata["_chunk_context"] == CONTEXT_MARKER_APPLIED
    # The retrieval text (chunk) carries the prefix; the document text
    # stays original for the sources view.
    candidates = await service.search(query="Pflichten Betreiber Artikel")
    assert candidates[0].chunk.text.startswith(
        "Pflichten der Betreiber, Artikel 26."
    )
    assert document.text == "Die Betreiber treffen geeignete Massnahmen."


@pytest.mark.asyncio
async def test_without_contextualizer_ingestion_is_unchanged():
    service = make_service()
    collection = await service.create_collection(name="K")
    document = await service.add_document(
        collection_id=collection.id, title="T", text="Inhalt."
    )
    assert "_chunk_context" not in document.metadata


def test_container_requires_llm_when_contextualize_is_on():
    from inqtrix.server.container import build_knowledge_context
    from inqtrix.settings import KnowledgeSettings, Settings

    settings = Settings(
        knowledge=KnowledgeSettings(enabled=True, contextualize="on")
    )
    with pytest.raises(RuntimeError, match="LLM-Provider"):
        build_knowledge_context(settings, llm=None)
