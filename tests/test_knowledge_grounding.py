"""Quote-then-answer grounding tests: parsing, verification, wiring.

Pure unit tests cover the deterministic quote check (whitespace
tolerance, German quotation marks, paraphrase and out-of-range
rejection, fallback shapes); algorithm tests assert the answer prompt
gains the quote instruction, the quote block is stripped from the
user-facing answer, verification lands visibly in ``result_state`` and
the run event stream, and ``grounding=off`` restores the previous
prompt and output byte-for-byte.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from inqtrix.core.context import RunContext, RuntimeContext
from inqtrix.core.results import RunRequest
from inqtrix.knowledge.algorithm import KnowledgeAlgorithm
from inqtrix.knowledge.grounding import (
    GROUNDING_MARKER_FALLBACK,
    GROUNDING_MARKER_PARSED,
    check_grounding,
)
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.base import LLMResponse, ProviderContext
from inqtrix.services.knowledge_service import KnowledgeService
from inqtrix.settings import AgentSettings, Settings

from tests.contract._app import StubSearch
from tests.test_knowledge_engine import StubEmbeddings

EVIDENCE = [
    "[K1] Rahmenvertrag (Abschnitt 1)\n"
    "Die Haftung ist auf den Auftragswert begrenzt.",
    "[K2] Reiserichtlinie (Abschnitt 1)\n"
    "Die Verpflegungspauschale betraegt 28 Euro pro Tag.",
]


# ------------------------------------------------------------------ #
# check_grounding unit tests
# ------------------------------------------------------------------ #


def test_verified_quote_and_stripped_answer():
    content = (
        "ZITATE:\n"
        '[K1] "Die Haftung ist auf den Auftragswert begrenzt."\n'
        "\n"
        "ANTWORT:\n"
        "Die Haftung ist begrenzt [K1]."
    )
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_PARSED
    assert report.answer == "Die Haftung ist begrenzt [K1]."
    assert [(q.label, q.verified) for q in report.quotes] == [("K1", True)]


def test_whitespace_differences_do_not_fail_verification():
    content = (
        "ZITATE:\n"
        '[K1] "Die Haftung   ist auf den\nAuftragswert begrenzt."\n'
        "ANTWORT:\n"
        "Antwort [K1]."
    )
    # The quote regex is line-based; a wrapped quote is two lines and
    # only the complete single-line form parses — so feed a quote with
    # collapsed-vs-multiple spaces instead.
    content = content.replace("den\nAuftragswert", "den  Auftragswert")
    report = check_grounding(content, EVIDENCE)
    assert report.quotes[0].verified is True


def test_german_quotation_marks_are_accepted():
    content = (
        "ZITATE:\n"
        "[K2] „Die Verpflegungspauschale betraegt 28 Euro pro Tag.“\n"
        "ANTWORT:\n"
        "28 Euro [K2]."
    )
    report = check_grounding(content, EVIDENCE)
    assert report.quotes == [
        type(report.quotes[0])(
            label="K2",
            text="Die Verpflegungspauschale betraegt 28 Euro pro Tag.",
            verified=True,
        )
    ]


def test_smart_quotes_and_dashes_verify_against_ascii_source():
    # The PDF text-layer used curly quotes and an em dash; the model quoted the
    # ASCII forms. A genuinely verbatim quote must not fail on typography alone.
    evidence = ["[K1] X\nEr nannte “den Vertrag” — verbindlich."]
    content = (
        "ZITATE:\n"
        '[K1] "Er nannte "den Vertrag" - verbindlich."\n'
        "ANTWORT:\nAntwort [K1]."
    )
    assert check_grounding(content, evidence).quotes[0].verified is True


def test_ligature_and_case_differences_verify():
    # NFKC folds the ffi ligature; case folds. Still a verbatim substring.
    evidence = ["[K1] X\nDie EFFIZIENTE Abwicklung."]
    content = (
        "ZITATE:\n"
        '[K1] "die eﬃziente abwicklung."\n'
        "ANTWORT:\nAntwort [K1]."
    )
    assert check_grounding(content, evidence).quotes[0].verified is True


def test_paraphrase_is_visibly_unverified():
    content = (
        "ZITATE:\n"
        '[K1] "Die Haftung wurde vertraglich gedeckelt."\n'
        "ANTWORT:\n"
        "Antwort [K1]."
    )
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_PARSED
    assert report.quotes[0].verified is False


def test_out_of_range_label_is_unverified():
    content = 'ZITATE:\n[K7] "Die Haftung"\nANTWORT:\nAntwort.'
    report = check_grounding(content, EVIDENCE)
    assert report.quotes[0].label == "K7"
    assert report.quotes[0].verified is False


def test_missing_answer_section_falls_back_to_full_content():
    content = "Eine Antwort ohne das verlangte Format [K1]."
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_FALLBACK
    assert report.answer == content
    assert report.quotes == []


def test_answer_section_without_quotes_falls_back_but_strips():
    content = "ZITATE:\n\nANTWORT:\nNur die Antwort [K1]."
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_FALLBACK
    assert report.answer == "Nur die Antwort [K1]."


def test_empty_answer_after_separator_is_a_visible_fallback():
    """Truncated completions must not masquerade as clean parses."""
    content = 'ZITATE:\n[K1] "Die Haftung"\nANTWORT:\n'
    report = check_grounding(content, EVIDENCE)
    assert report.answer == content
    assert report.marker == GROUNDING_MARKER_FALLBACK
    assert report.quotes == []


def test_bold_separator_and_crlf_are_tolerated():
    content = (
        'ZITATE:\r\n[K1] "Die Haftung ist auf den Auftragswert begrenzt."\r\n'
        '**ANTWORT:**\r\nDie Haftung ist begrenzt [K1].'
    )
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_PARSED
    assert report.quotes[0].verified is True
    assert report.answer == 'Die Haftung ist begrenzt [K1].' 


# ------------------------------------------------------------------ #
# Algorithm wiring
# ------------------------------------------------------------------ #


class GroundedLLM:
    """Answers with a fixed quote-then-answer shaped completion."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.prompts: list[str] = []

    def complete(self, *args: Any, **kwargs: Any) -> str:
        return "ok"

    def complete_with_metadata(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(
            content=self._content,
            prompt_tokens=42,
            completion_tokens=11,
            model="stub-answer",
            finish_reason="stop",
        )

    def is_available(self) -> bool:
        return True


def make_algorithm(
    llm: GroundedLLM, *, grounding_enabled: bool, seed: bool = True
) -> tuple[KnowledgeAlgorithm, RunContext, RuntimeContext, list]:
    knowledge = KnowledgeProviderContext(
        embeddings=StubEmbeddings(),
        store=MemoryKnowledgeStore(),
        default_top_k=4,
    )
    service = KnowledgeService(
        knowledge=knowledge, chunk_max_chars=2_000, max_document_chars=100_000
    )
    if seed:
        async def _seed() -> None:
            collection = await service.create_collection(name="K")
            await service.add_document(
                collection_id=collection.id,
                title="Rahmenvertrag",
                text="Die Haftung ist auf den Auftragswert begrenzt.",
            )

        asyncio.run(_seed())
    algorithm = KnowledgeAlgorithm(
        knowledge=knowledge,
        gate_enabled=False,
        grounding_enabled=grounding_enabled,
    )
    settings = Settings(agent=AgentSettings())
    runtime = RuntimeContext(
        settings=settings,
        registry=None,
        providers=ProviderContext(llm=llm, search=StubSearch()),
        strategies=None,
    )
    events: list[tuple[str, dict]] = []
    context = RunContext(
        providers=runtime.providers,
        strategies=None,
        agent_settings=settings.agent,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    return algorithm, context, runtime, events


def run_question(algorithm, runtime, context):
    return algorithm.run(
        RunRequest(mode="knowledge", question="Wie ist die Haftung?"),
        runtime=runtime,
        context=context,
    )


def test_grounded_answer_is_verified_stripped_and_evented():
    llm = GroundedLLM(
        "ZITATE:\n"
        '[K1] "Die Haftung ist auf den Auftragswert begrenzt."\n'
        "\n"
        "ANTWORT:\n"
        "Die Haftung ist auf den Auftragswert begrenzt [K1]."
    )
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )

    result = run_question(algorithm, runtime, context)

    assert result.answer == (
        "Die Haftung ist auf den Auftragswert begrenzt [K1]."
    )
    assert "ZITATE:" not in result.answer
    assert "ZITATE:" in llm.prompts[0]
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["marker"] == GROUNDING_MARKER_PARSED
    assert state["quotes_total"] == 1
    assert state["quotes_verified"] == 1
    assert state["quotes"][0]["verified"] is True
    assert result.raw["result_state"]["answer"] == result.answer
    grounding_events = [
        payload
        for event, payload in events
        if event == "inqtrix.knowledge.grounding.checked"
    ]
    assert grounding_events == [
        {
            "marker": GROUNDING_MARKER_PARSED,
            "quotes_total": 1,
            "quotes_verified": 1,
        }
    ]


def test_unverified_quote_warns_but_answers(caplog):
    llm = GroundedLLM(
        "ZITATE:\n"
        '[K1] "Die Haftung wurde komplett ausgeschlossen."\n'
        "ANTWORT:\n"
        "Antwort [K1]."
    )
    algorithm, context, runtime, _events = make_algorithm(
        llm, grounding_enabled=True
    )

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        result = run_question(algorithm, runtime, context)

    assert result.answer == "Antwort [K1]."
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["quotes_verified"] == 0
    assert any("Knowledge-Grounding" in m for m in caplog.messages)


def test_unparseable_shape_falls_back_loudly(caplog):
    llm = GroundedLLM("Antwort ohne Format [K1].")
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        result = run_question(algorithm, runtime, context)

    assert result.answer == "Antwort ohne Format [K1]."
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["marker"] == GROUNDING_MARKER_FALLBACK
    assert any(GROUNDING_MARKER_FALLBACK in m for m in caplog.messages)
    assert [
        payload["marker"]
        for event, payload in events
        if event == "inqtrix.knowledge.grounding.checked"
    ] == [GROUNDING_MARKER_FALLBACK]


def test_grounding_off_restores_plain_prompt_and_passthrough():
    llm = GroundedLLM("Antwort mit Beleg [K1].")
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=False
    )

    result = run_question(algorithm, runtime, context)

    assert result.answer == "Antwort mit Beleg [K1]."
    assert "ZITATE:" not in llm.prompts[0]
    assert "AUSGABEFORMAT" not in llm.prompts[0]
    assert result.raw["result_state"]["knowledge_grounding"] == {
        "enabled": False
    }
    assert not any(
        event == "inqtrix.knowledge.grounding.checked"
        for event, _payload in events
    )


def test_refusal_path_records_no_grounding_check():
    llm = GroundedLLM("nie aufgerufen")
    algorithm, context, runtime, _events = make_algorithm(
        llm, grounding_enabled=True, seed=False
    )

    result = algorithm.run(
        RunRequest(mode="knowledge", question="?"),
        runtime=runtime,
        context=context,
    )

    assert "keine relevanten" in result.answer
    assert llm.prompts == []
    assert result.raw["result_state"]["knowledge_grounding"] == {
        "enabled": True
    }
