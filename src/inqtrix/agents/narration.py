"""Narration lines for the Agent-Desk transcript (plan B2).

The main window shows the agent "thinking along" as short German prose.
Every line is built DETERMINISTICALLY from artifacts the run already
produced (discovery result, plan summary, task outcomes, memo outline) —
no extra LLM call, no invented chain-of-thought (Designprinzip 5:
grounded visibility, never storytelling). Paragraph granularity by
design: narration events are persisted rows that replay after a reload,
so the channel stays bounded (~10-30 events per run); the typewriter
feel is a frontend rendering concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inqtrix.agents.phase_models import DiscoveryResult

NARRATION_EVENT = "inqtrix.agent.narration"

_TEXT_LIMIT = 400
"""Per-line prose bound (mirrored by the sanitizer's text cap)."""


def first_sentences(text: str, limit: int = _TEXT_LIMIT) -> str:
    """Compact prose head: whole sentences up to *limit* characters.

    Falls back to a word-boundary cut when the first sentence alone
    exceeds the limit — a narration line must never end mid-word.
    """
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    head = normalized[:limit]
    sentence_end = max(
        head.rfind(". "), head.rfind("! "), head.rfind("? ")
    )
    if sentence_end >= limit // 3:
        return head[: sentence_end + 1]
    boundary = head.rfind(" ")
    if boundary < limit // 2:
        boundary = limit - 4
    return head[:boundary].rstrip() + " ..."


def discovery_narration(discovery: "DiscoveryResult | None") -> str:
    """One paragraph on what the exploration established."""
    if discovery is None:
        return ""
    facts = len(discovery.known_facts)
    gaps = len(discovery.gaps)
    if facts == 0 and gaps == 0:
        return (
            "Die Erkundung hat weder belegte Fakten noch offene "
            "Luecken ergeben."
        )
    parts = [
        "Ich habe "
        f"{facts} belegte{'n' if facts == 1 else ''} Fakt"
        f"{'' if facts == 1 else 'en'} und {gaps} offene "
        f"Luecke{'' if gaps == 1 else 'n'} identifiziert."
    ]
    if discovery.gaps:
        parts.append(
            "Wichtigste Luecke: "
            f"{first_sentences(discovery.gaps[0].description, 160)}"
        )
    return " ".join(part for part in parts if part)


def plan_narration(summary_markdown: str, task_count: int) -> str:
    """The plan's own summary prose, framed with the task count."""
    summary = first_sentences(summary_markdown)
    if not summary:
        return f"Ich schlage einen Plan mit {task_count} Aufgaben vor."
    return f"Mein Plan ({task_count} Aufgaben): {summary}"


def task_narration(title: str, summary: str) -> str:
    """One line per finished task: its title plus the result head."""
    head = first_sentences(summary, 240)
    if not head:
        return ""
    return f"{title}: {head}" if title else head


def synthesis_narration(title: str, section_count: int) -> str:
    """Announces the memo being written (the canvas shows it live)."""
    return (
        f"Ich schreibe jetzt das Memo '{title}' mit "
        f"{section_count} Abschnitten."
    )


def section_narration(title: str) -> str:
    """One line per completed memo section."""
    return f"Abschnitt '{title}' geschrieben."
