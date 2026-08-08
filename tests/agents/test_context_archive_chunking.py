"""Lossless multi-section context archive and imperative planner pin.

The archive must preserve compaction evictions beyond one section, and
the pinned web profile must prevent repeated ``plan_invalid`` results.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

from inqtrix.agents.kernel.deps import (
    CONTEXT_ARCHIVE_SECTION_CHARS,
    KernelDeps,
)
from inqtrix.agents.prompts import _planner_web_rule
from inqtrix.settings import AgentPlatformSettings


def _deps() -> KernelDeps:
    return KernelDeps(
        run_id="run_chunk_test",
        control=None,  # the chunker never touches the store directly
        platform=AgentPlatformSettings(),
        llm=None,
        model=None,
        reasoning_effort=None,
        timeout=1.0,
    )


def test_chunked_archive_preserves_overflow_and_closes_fences():
    deps = _deps()
    written: list[tuple[str, str]] = []

    def record(title: str, text: str) -> str:
        written.append((title, text))
        return f"art_section_{len(written)}"

    deps.append_context_archive = record  # type: ignore[method-assign]

    # ~3 sections worth of one open code fence: the cut lands INSIDE it.
    body = "```python\n" + ("wert = 1\n" * 6_000)
    ids = deps.append_context_archive_chunked("Komprimierter Verlauf", body)

    assert len(ids) == len(written) >= 2
    # Lossless: nothing beyond the fence markers is dropped.
    preserved = sum(len(text) for _, text in written)
    assert preserved >= len(body.strip())
    for index, (title, text) in enumerate(written, start=1):
        assert len(text) <= CONTEXT_ARCHIVE_SECTION_CHARS
        # Every section renders as valid markdown on read_canvas.
        assert text.count("```") % 2 == 0, f"section {index} leaks a fence"
        assert title == f"Komprimierter Verlauf ({index}/{len(written)})"
    # The reopened fence continues the cut code block.
    assert all(text.startswith("```") for _, text in written[1:])


def test_chunked_archive_keeps_short_sections_singular():
    deps = _deps()
    written: list[tuple[str, str]] = []
    deps.append_context_archive = (  # type: ignore[method-assign]
        lambda title, text: written.append((title, text)) or "art_single"
    )

    assert deps.append_context_archive_chunked("Titel", "kurzer Text") == [
        "art_single"
    ]
    assert written == [("Titel", "kurzer Text")]
    assert deps.append_context_archive_chunked("Titel", "   ") == []


def test_planner_pinned_profile_rule_is_imperative():
    pinned = _planner_web_rule(True, "deep")
    assert "MUESSEN params.profile=deep" in pinned
    assert "web_instant-Tasks tragen NIEMALS ein profile" in pinned
    # The ceiling variant keeps its explicit scope sentence.
    ceiling = _planner_web_rule(
        True, "schnell", max_profile="compact", max_instant_tasks=2
    )
    assert "NUR web_research-Tasks duerfen params.profile setzen" in ceiling
