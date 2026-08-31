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


from inqtrix.core.context import RunContext, RuntimeContext
from inqtrix.core.results import RunRequest
from inqtrix.knowledge.algorithm import KnowledgeAlgorithm
from inqtrix.knowledge.grounding import (
    GROUNDING_MARKER_FORMAT_REPAIRED,
    GROUNDING_MARKER_FALLBACK,
    GROUNDING_MARKER_PARSED,
    GroundingFailureCode,
    GroundingStatus,
    check_grounding,
    quote_is_verbatim,
    strip_page_break_artifacts,
)
from inqtrix.runtime_logging import sanitize_event_payload
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
    assert report.status is GroundingStatus.VERIFIED
    assert report.publishable is True
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
    assert report.status is GroundingStatus.REJECTED_QUOTE
    assert report.failure_code is GroundingFailureCode.QUOTE_UNVERIFIED
    assert report.answer == ""
    assert report.publishable is False
    assert report.quotes[0].verified is False


def test_out_of_range_label_is_unverified():
    content = 'ZITATE:\n[K7] "Die Haftung"\nANTWORT:\nAntwort.'
    report = check_grounding(content, EVIDENCE)
    assert report.quotes[0].label == "K7"
    assert report.quotes[0].verified is False
    assert report.status is GroundingStatus.REJECTED_QUOTE


def test_missing_answer_section_is_not_publishable():
    content = "Eine Antwort ohne das verlangte Format [K1]."
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_FALLBACK
    assert report.status is GroundingStatus.REJECTED_FORMAT
    assert report.failure_code is GroundingFailureCode.FORMAT_INVALID
    assert report.answer == ""
    assert report.quotes == []


def test_answer_section_without_quotes_is_not_publishable():
    content = "ZITATE:\n\nANTWORT:\nNur die Antwort [K1]."
    report = check_grounding(content, EVIDENCE)
    assert report.marker == GROUNDING_MARKER_FALLBACK
    assert report.answer == ""
    assert report.status is GroundingStatus.REJECTED_FORMAT


def test_empty_answer_after_separator_is_a_visible_fallback():
    """Truncated completions must not masquerade as clean parses."""
    content = 'ZITATE:\n[K1] "Die Haftung"\nANTWORT:\n'
    report = check_grounding(content, EVIDENCE)
    assert report.answer == ""
    assert report.marker == GROUNDING_MARKER_FALLBACK
    assert report.status is GroundingStatus.REJECTED_FORMAT
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


def test_missing_zitate_header_is_rejected_even_with_a_valid_quote_line():
    report = check_grounding(
        '[K1] "Die Haftung ist auf den Auftragswert begrenzt."\n'
        "ANTWORT:\nAntwort [K1].",
        EVIDENCE,
    )

    assert report.status is GroundingStatus.REJECTED_FORMAT
    assert report.quotes == []
    assert report.answer == ""


def test_one_bounded_markdown_heading_repair_is_audited():
    report = check_grounding(
        "### ZITATE:\n"
        '[K1] "Die Haftung ist auf den Auftragswert begrenzt."\n'
        "### ANTWORT:\nAntwort [K1].",
        EVIDENCE,
    )

    assert report.status is GroundingStatus.VERIFIED
    assert report.marker == GROUNDING_MARKER_FORMAT_REPAIRED
    assert report.format_repaired is True
    assert report.answer == "Antwort [K1]."


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
    assert state["status"] == GroundingStatus.VERIFIED.value
    assert state["failure_code"] is None
    assert state["format_repaired"] is False
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
            "status": GroundingStatus.VERIFIED.value,
            "failure_code": None,
            "format_repaired": False,
            "quotes_total": 1,
            "quotes_verified": 1,
        }
    ]


def test_unverified_quote_fails_closed_with_safe_visible_reason(caplog):
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

    assert "nicht veröffentlicht" in result.answer
    assert "Antwort [K1]." not in result.answer
    assert result.successful is False
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.type
        == GroundingFailureCode.QUOTE_UNVERIFIED.value
    )
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["quotes_verified"] == 0
    assert state["status"] == GroundingStatus.REJECTED_QUOTE.value
    assert (
        state["failure_code"]
        == GroundingFailureCode.QUOTE_UNVERIFIED.value
    )
    assert result.raw["result_state"]["answer"] == result.answer
    assert any("Knowledge-Grounding" in m for m in caplog.messages)


def test_unparseable_shape_fails_closed_loudly(caplog):
    llm = GroundedLLM("Antwort ohne Format [K1].")
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        result = run_question(algorithm, runtime, context)

    assert "nicht veröffentlicht" in result.answer
    assert "Antwort ohne Format [K1]." not in result.answer
    assert result.successful is False
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.type
        == GroundingFailureCode.FORMAT_INVALID.value
    )
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["marker"] == GROUNDING_MARKER_FALLBACK
    assert state["status"] == GroundingStatus.REJECTED_FORMAT.value
    assert any("knowledge_grounding_format_invalid" in m for m in caplog.messages)
    assert [
        (payload["marker"], payload["status"], payload["failure_code"])
        for event, payload in events
        if event == "inqtrix.knowledge.grounding.checked"
    ] == [
        (
            GROUNDING_MARKER_FALLBACK,
            GroundingStatus.REJECTED_FORMAT.value,
            GroundingFailureCode.FORMAT_INVALID.value,
        )
    ]


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


# ------------------------------------------------------------------ #
# Page-break artifact tolerance (second evidence surface)
# ------------------------------------------------------------------ #

# The evidenced production shape: pdfminer renders the printed page
# number and a form feed INTO a sentence spanning a page break.
PAGE_BREAK_EVIDENCE = [
    "[K1] Commission Guidelines (Abschnitt 21)\n"
    "that element should be understood to\n"
    "emphasise the fact that AI systems are not passive, but actively "
    "impact the environments\n\n11\n\n\x0cin which they are deployed. "
    "Reference to 'physical or virtual environments' indicates that\n"
    "the influence"
]


def test_page_break_artifact_quote_verifies_without_page_number():
    # Regression: the model quotes the sentence as a human reads it —
    # without the printed page number. Must verify.
    content = (
        "ZITATE:\n"
        '[K1] "AI systems are not passive, but actively impact the '
        'environments in which they are deployed."\n'
        "ANTWORT:\n"
        "KI-Systeme wirken aktiv auf ihre Umgebung [K1]."
    )
    report = check_grounding(content, PAGE_BREAK_EVIDENCE)
    assert report.status is GroundingStatus.VERIFIED
    assert report.quotes[0].verified is True


def test_page_break_artifact_quote_with_page_number_still_verifies():
    # A model that faithfully includes the artifact stays valid (surface 1).
    content = (
        "ZITATE:\n"
        '[K1] "AI systems are not passive, but actively impact the '
        'environments 11 in which they are deployed."\n'
        "ANTWORT:\n"
        "Antwort [K1]."
    )
    assert check_grounding(content, PAGE_BREAK_EVIDENCE).quotes[0].verified


def test_number_without_form_feed_stays_content():
    # "Article 11" protection: a number NOT anchored by a form feed is
    # content — a quote that drops it must keep failing.
    evidence = [
        "[K1] AI Act\nThe requirements of Article 11 in conjunction "
        "with Annex IV apply to providers."
    ]
    content = (
        "ZITATE:\n"
        '[K1] "The requirements of Article in conjunction with Annex IV '
        'apply to providers."\n'
        "ANTWORT:\nAntwort [K1]."
    )
    assert check_grounding(content, evidence).quotes[0].verified is False


def test_paraphrase_still_fails_against_artifact_evidence():
    # The tolerance opens page-break artifacts, never paraphrase.
    content = (
        "ZITATE:\n"
        '[K1] "AI systems always change their environment actively."\n'
        "ANTWORT:\nAntwort [K1]."
    )
    report = check_grounding(content, PAGE_BREAK_EVIDENCE)
    assert report.status is GroundingStatus.REJECTED_QUOTE
    assert report.failure_code is GroundingFailureCode.QUOTE_UNVERIFIED


def test_strip_page_break_artifacts_shapes_and_idempotency():
    cases = {
        # blank-line-framed page number before the form feed (the
        # corpus-evidenced dominant shape, 347x)
        "a\n\n11\n\n\x0cb": "a\nb",
        # bare form feed (10x)
        "a\n\x0cCONTENTS": "a\nCONTENTS",
        # a number AFTER the form feed is never touched
        "a\x0c12\nb": "a\n12\nb",
        # multiple breaks in one text
        "a\n\n1\n\n\x0cb\n\n2\n\n\x0cc": "a\nb\nc",
        # no form feed: byte-identical, numbers untouched
        "Article 11 applies.\n\n12\n\nNext.": (
            "Article 11 applies.\n\n12\n\nNext."
        ),
        # a digit that merely ends a wrapped text line (single newline,
        # no blank-line frame) is CONTENT and survives the strip
        "price rose from 1\n2\n\x0c3 percent overall": (
            "price rose from 1\n2\n3 percent overall"
        ),
    }
    for source, expected in cases.items():
        stripped = strip_page_break_artifacts(source)
        assert stripped == expected, repr(source)
        assert strip_page_break_artifacts(stripped) == stripped


def test_content_digit_at_page_break_is_never_elidable():
    # Review finding: a content number adjacent to the form feed without
    # the blank-line frame (wrapped amounts, column values) must not be
    # removable — a quote eliding it stays unverified.
    evidence = ["[K1] Report\nprice rose from 1\n2\n\x0c3 percent overall"]
    content = (
        "ZITATE:\n"
        '[K1] "price rose from 1 3 percent overall"\n'
        "ANTWORT:\nAntwort [K1]."
    )
    assert check_grounding(content, evidence).quotes[0].verified is False
    evidence2 = [
        "[K1] Report\nThe penalty shall not exceed\n20\n\x0cmillion euros"
    ]
    content2 = (
        "ZITATE:\n"
        '[K1] "The penalty shall not exceed million euros"\n'
        "ANTWORT:\nAntwort [K1]."
    )
    assert check_grounding(content2, evidence2).quotes[0].verified is False


def test_zero_width_only_quote_never_verifies():
    # Review finding: a quote that normalizes to nothing ('' is a
    # substring of everything) must fail, exactly as in quote_is_verbatim.
    content = "ZITATE:\n[K1] ​\nANTWORT:\nAntwort [K1]."
    report = check_grounding(content, EVIDENCE)
    assert report.quotes[0].verified is False
    assert report.status is GroundingStatus.REJECTED_QUOTE


def test_quote_verifies_only_against_its_assigned_entry():
    # Review finding: K2's text under a K1 label is false provenance —
    # the check is bound to the ASSIGNED entry, never any-entry search.
    content = (
        "ZITATE:\n"
        '[K1] "Die Verpflegungspauschale betraegt 28 Euro pro Tag."\n'
        "ANTWORT:\nAntwort [K1]."
    )
    assert check_grounding(content, EVIDENCE).quotes[0].verified is False


def test_artifact_tolerated_flag_records_the_surface_used():
    clean_quote = (
        "ZITATE:\n"
        '[K1] "AI systems are not passive, but actively impact the '
        'environments in which they are deployed."\n'
        "ANTWORT:\nAntwort [K1]."
    )
    tolerated = check_grounding(clean_quote, PAGE_BREAK_EVIDENCE).quotes[0]
    assert tolerated.verified is True
    assert tolerated.artifact_tolerated is True
    raw_quote = (
        "ZITATE:\n"
        '[K1] "Reference to \'physical or virtual environments\'"\n'
        "ANTWORT:\nAntwort [K1]."
    )
    raw = check_grounding(raw_quote, PAGE_BREAK_EVIDENCE).quotes[0]
    assert raw.verified is True
    assert raw.artifact_tolerated is False


def test_quote_is_verbatim_covers_both_surfaces():
    # Direct pins for the helper the agent-side check reuses.
    evidence = PAGE_BREAK_EVIDENCE
    assert quote_is_verbatim(
        "actively impact the environments in which they are deployed",
        evidence,
    )
    assert quote_is_verbatim(
        "actively impact the environments 11 in which they are deployed",
        evidence,
    )
    assert not quote_is_verbatim("changes its environment", evidence)
    assert not quote_is_verbatim("", evidence)


# ------------------------------------------------------------------ #
# Visible one-shot answer regeneration
# ------------------------------------------------------------------ #


class QueuedGroundedLLM(GroundedLLM):
    """Pops one scripted completion per call — the regeneration seam."""

    def __init__(self, contents: list[str]) -> None:
        super().__init__(contents[0])
        self._queue = list(contents)

    def complete_with_metadata(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(
            content=self._queue.pop(0),
            prompt_tokens=42,
            completion_tokens=11,
            model="stub-answer",
            finish_reason="stop",
        )


PARAPHRASED_ATTEMPT = (
    "ZITATE:\n"
    '[K1] "Die Haftung wurde komplett ausgeschlossen."\n'
    "ANTWORT:\n"
    "Antwort [K1]."
)
VERBATIM_ATTEMPT = (
    "ZITATE:\n"
    '[K1] "Die Haftung ist auf den Auftragswert begrenzt."\n'
    "ANTWORT:\n"
    "Die Haftung ist begrenzt [K1]."
)


def test_unverified_quote_triggers_one_visible_regeneration(caplog):
    llm = QueuedGroundedLLM([PARAPHRASED_ATTEMPT, VERBATIM_ATTEMPT])
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )
    # Order probe: snapshot how many retry events exist at the moment of
    # each model call — the retry must be announced BEFORE call 2 runs,
    # not emitted retroactively after it.
    retry_seen_at_call: list[int] = []
    original_complete = llm.complete_with_metadata

    def probed_complete(prompt: str, **kwargs: Any) -> LLMResponse:
        retry_seen_at_call.append(
            sum(
                1
                for event, _payload in events
                if event == "inqtrix.knowledge.answer.retry"
            )
        )
        return original_complete(prompt, **kwargs)

    llm.complete_with_metadata = probed_complete  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        result = run_question(algorithm, runtime, context)

    assert retry_seen_at_call == [0, 1]

    # Second attempt published; the run is a success, not a failure.
    assert result.successful is True
    assert result.terminal_failure is None
    assert result.answer == "Die Haftung ist begrenzt [K1]."
    # Exactly two model calls; the retry prompt names the failed quote.
    assert len(llm.prompts) == 2
    assert "KORREKTUR" in llm.prompts[1]
    assert "Die Haftung wurde komplett ausgeschlossen." in llm.prompts[1]
    assert "KORREKTUR" not in llm.prompts[0]
    # Both calls are real spend: usage is additive across attempts.
    assert result.raw["usage"]["prompt_tokens"] == 84
    assert result.raw["usage"]["completion_tokens"] == 22
    # Both attempts stay visible in the result state.
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["status"] == GroundingStatus.VERIFIED.value
    assert [a["status"] for a in state["attempts"]] == [
        GroundingStatus.REJECTED_QUOTE.value,
        GroundingStatus.VERIFIED.value,
    ]
    # The regeneration is announced: retry event with counters only,
    # exactly one final grounding.checked, and a log line.
    retry_events = [
        payload
        for event, payload in events
        if event == "inqtrix.knowledge.answer.retry"
    ]
    assert retry_events == [
        {"attempt": 2, "quotes_total": 1, "quotes_unverified": 1}
    ]
    checked = [
        payload
        for event, payload in events
        if event == "inqtrix.knowledge.grounding.checked"
    ]
    assert len(checked) == 1
    assert checked[0]["status"] == GroundingStatus.VERIFIED.value
    assert any("Antwort-Regeneration" in m for m in caplog.messages)


def test_both_attempts_unverified_keep_fail_closed_terminal(caplog):
    llm = QueuedGroundedLLM([PARAPHRASED_ATTEMPT, PARAPHRASED_ATTEMPT])
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        result = run_question(algorithm, runtime, context)

    assert len(llm.prompts) == 2
    assert "nicht veröffentlicht" in result.answer
    assert "Antwort [K1]." not in result.answer
    assert result.successful is False
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.type
        == GroundingFailureCode.QUOTE_UNVERIFIED.value
    )
    state = result.raw["result_state"]["knowledge_grounding"]
    assert state["status"] == GroundingStatus.REJECTED_QUOTE.value
    assert len(state["attempts"]) == 2
    assert (
        len(
            [
                1
                for event, _payload in events
                if event == "inqtrix.knowledge.grounding.checked"
            ]
        )
        == 1
    )
    assert any("Knowledge-Grounding abgelehnt" in m for m in caplog.messages)


def test_verified_first_attempt_never_retries():
    llm = QueuedGroundedLLM([VERBATIM_ATTEMPT, "nie aufgerufen"])
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )

    result = run_question(algorithm, runtime, context)

    assert result.successful is True
    assert len(llm.prompts) == 1
    assert not any(
        event == "inqtrix.knowledge.answer.retry"
        for event, _payload in events
    )
    state = result.raw["result_state"]["knowledge_grounding"]
    assert [a["status"] for a in state["attempts"]] == [
        GroundingStatus.VERIFIED.value
    ]


def test_format_invalid_first_attempt_never_retries():
    # Only QUOTE_UNVERIFIED regenerates; format failures keep today's
    # immediate terminal behaviour.
    llm = QueuedGroundedLLM(["Antwort ohne Format [K1].", "nie aufgerufen"])
    algorithm, context, runtime, events = make_algorithm(
        llm, grounding_enabled=True
    )

    result = run_question(algorithm, runtime, context)

    assert len(llm.prompts) == 1
    assert result.successful is False
    assert (
        result.terminal_failure.type
        == GroundingFailureCode.FORMAT_INVALID.value
    )
    assert not any(
        event == "inqtrix.knowledge.answer.retry"
        for event, _payload in events
    )


def test_answer_retry_is_projected_and_forwarded_on_the_agent_path():
    # Review finding: deleting either allowlist entry kept the suite
    # green while silently hiding the retry from parent/mission streams.
    from inqtrix.runs.shared import _CHILD_PROJECTED_EVENTS

    assert "inqtrix.knowledge.answer.retry" in _CHILD_PROJECTED_EVENTS


def test_answer_retry_event_sanitizer_drops_quote_texts():
    # The event boundary carries counters only — a quote text smuggled
    # into the payload must not survive sanitization.
    sanitized = sanitize_event_payload(
        "inqtrix.knowledge.answer.retry",
        {
            "attempt": 2,
            "quotes_total": 3,
            "quotes_unverified": 1,
            "quote_text": "geheim",
            "quotes": [{"text": "geheim"}],
        },
    )
    assert sanitized == {
        "attempt": 2,
        "quotes_total": 3,
        "quotes_unverified": 1,
    }
