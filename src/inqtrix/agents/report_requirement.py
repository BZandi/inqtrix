"""Composition of the result requirement a plan gate decides.

The requirement can come from two places: prompt-library rules the user
attached, and the free text they typed at the gate. Both answer the same
question — what the result has to look like — so they travel as ONE
value in one state key and render in one prompt section.

Composition happens at decision time, on the server, and the composed
text is what gets stored: the plan is a contract, so editing a template
afterwards must not silently rewrite a requirement a run is already
working under. The library entry contributes its text, not a live link.

Every contribution carries an origin marker. Without them the critic can
report ``instruction_violation`` but cannot say WHICH requirement was
broken, and the read-back cannot tell a rule from typed text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

FREE_REQUIREMENT_MARKER = "[Freie Vorgabe]"
"""Origin marker of the text the user typed at the gate."""

REPORT_GUIDANCE_MAX_CHARS = 2_000
"""Ceiling for the typed half of the requirement, at BOTH entry points.

The requirement can be set before the run and at the plan gate, and the
two must agree: a text the composer accepts and the gate rejects (or the
reverse) would be a limit the user cannot predict.
"""

REPORT_RULE_IDS_MAX = 3
"""How many library rules one requirement may attach, at both entry points."""

MAX_COMPOSED_REQUIREMENT_CHARS = 20_000
"""Ceiling for the composed requirement.

Not a truncation — a refusal. Attached rules are whole documents (the
shipped 'Sprechzettel' template alone is ~17k characters) and the
requirement is repeated into every writing prompt of the run, so a stack
of them is a real cost. Exceeding this fails loudly and names the
offender; nothing is ever shortened behind the user's back.
"""


def rule_requirement_marker(label: str) -> str:
    """Origin marker of one attached library rule."""
    return f"[Regel: {label}]"


def compose_report_requirement(
    *,
    free_text: str = "",
    rules: "Sequence[tuple[str, str]]" = (),
) -> str:
    """The requirement block, rules first, typed text last.

    Args:
        free_text: What the user typed at the gate.
        rules: ``(label, content)`` of each attached library rule, in
            attachment order. Labels are resolved server-side from the
            caller's own catalog — a client-supplied label would be a
            free-text channel into the prompt.

    Returns:
        The composed block, or ``""`` when there is nothing to say.
    """
    parts: list[str] = []
    for label, content in rules:
        body = content.strip()
        if not body:
            continue
        marker = rule_requirement_marker(label)
        parts.append(f"{marker}\n{body}\n[Ende Regel: {label}]")
    typed = free_text.strip()
    if typed:
        parts.append(f"{FREE_REQUIREMENT_MARKER}\n{typed}")
    return "\n\n".join(parts)


def composed_requirement_is_oversized(composed: str) -> bool:
    """Whether the composed requirement exceeds the visible ceiling."""
    return len(composed) > MAX_COMPOSED_REQUIREMENT_CHARS
