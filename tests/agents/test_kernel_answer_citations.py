"""Final kernel-answer citation repair and reference projection."""

from __future__ import annotations

from typing import Any

from inqtrix.agents.kernel.algorithm import (
    _result_references,
    _validate_kernel_answer_citations,
)
from inqtrix.agents.report_quality import CitationValidationFailed


LEDGER = [
    {"label": "W1", "url": "https://example.test/one"},
    {"label": "K2", "url": "inqtrix://documents/two#chunk-0"},
]


def test_unknown_only_citation_never_falls_back_to_the_whole_ledger() -> None:
    assert _result_references("Erfundene Zuordnung [W9].", LEDGER) == []


def test_mixed_valid_and_unknown_citations_keep_only_the_valid_subset() -> None:
    projected = _result_references(
        "Belegt [K2], aber diese Zuordnung ist unbekannt [W9].",
        LEDGER,
    )

    assert [reference["label"] for reference in projected] == ["K2"]


class _Deps:
    def __init__(self) -> None:
        self.llm = object()
        self.model = "model"
        self.reasoning_effort = None
        self.timeout = 30.0
        self.question = "What does GPT-5.6 Sol cost?"
        self.usage: list[tuple[int, int]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def check_abort(self) -> None:
        return

    def book_usage(self, prompt: int, completion: int) -> None:
        self.usage.append((prompt, completion))

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


def test_bounded_repair_is_used_before_reference_projection(monkeypatch) -> None:
    deps = _Deps()

    def repair(*_args, **_kwargs):
        return "Korrigierte Aussage [W1].", {
            "prompt_tokens": 11,
            "completion_tokens": 3,
        }

    monkeypatch.setattr(
        "inqtrix.agents.report_quality.validate_and_repair_citations",
        repair,
    )

    answer = _validate_kernel_answer_citations(  # type: ignore[arg-type]
        deps,
        "Aussage [W9].",
        LEDGER,
    )

    assert answer == "Korrigierte Aussage [W1]."
    assert deps.usage == [(11, 3)]
    assert [reference["label"] for reference in _result_references(answer, LEDGER)] == [
        "W1"
    ]
    assert deps.events[-1][1]["status"] == "repaired"


def test_failed_repair_marks_unknown_label_and_preserves_valid_citation(
    monkeypatch,
) -> None:
    deps = _Deps()

    def fail(*_args, **_kwargs):
        raise CitationValidationFailed(
            "still invalid",
            usage={"prompt_tokens": 7, "completion_tokens": 2},
        )

    monkeypatch.setattr(
        "inqtrix.agents.report_quality.validate_and_repair_citations",
        fail,
    )

    answer = _validate_kernel_answer_citations(  # type: ignore[arg-type]
        deps,
        "Belastbar [W1], erfunden [W9].",
        LEDGER,
    )

    assert "[W1]" in answer
    assert "[W9]" not in answer
    assert "[unsupported: W9]" in answer
    assert deps.usage == [(7, 2)]
    assert [reference["label"] for reference in _result_references(answer, LEDGER)] == [
        "W1"
    ]
    assert deps.events[-1][1]["status"] == "degraded"
