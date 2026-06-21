"""Quote-then-answer grounding: deterministic verbatim-quote checks.

The answer prompt (``build_knowledge_answer_prompt(grounding=True)``)
requires the model to emit a ``ZITATE:`` block of verbatim,
``[K#]``-labelled quotes BEFORE the ``ANTWORT:`` section. Writing the
quotes first conditions the subsequent answer on literal evidence
(WebGPT/GopherCite lineage — the cheapest proven grounding step), and
because the quotes claim to be verbatim they can be validated WITHOUT
another LLM call: a whitespace-normalized substring check against the
exact evidence entries the model saw.

Failure policy (No Silent Fallbacks): a response without a parseable
quote block degrades to the unmodified answer with a loud log and the
``_knowledge_grounding_fallback`` marker; quotes that do not verify
stay in the report with ``verified=False`` so callers can surface
them — the check never blocks or rewrites the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GROUNDING_MARKER_PARSED = "_knowledge_grounding_parsed"
GROUNDING_MARKER_FALLBACK = "_knowledge_grounding_fallback"

_ANSWER_SPLIT = re.compile(
    r"^[ \t]*(?:\*\*)?ANTWORT:(?:\*\*)?[ \t]*\r?$", re.MULTILINE
)
_QUOTE_LINE = re.compile(
    r"^\[K(\d+)\][ \t]*(.+?)[ \t]*\r?$", re.MULTILINE
)
_QUOTE_CHARS = "\"'„“”»«‚‘’"


@dataclass(frozen=True)
class QuoteCheck:
    """One quote from the model's ``ZITATE:`` block, with its verdict.

    Attributes:
        label: The evidence label the model attributed the quote to,
            e.g. ``"K1"`` — 1-based position in the evidence block.
        text: The quote text with surrounding quotation marks removed.
        verified: ``True`` when the whitespace-normalized quote occurs
            verbatim in the evidence entry the label points at;
            ``False`` for paraphrases, ellipses, and out-of-range
            labels alike.
    """

    label: str
    text: str
    verified: bool


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of parsing and verifying one grounded answer.

    Attributes:
        answer: The user-facing answer with the quote block stripped;
            the unmodified model output when parsing fell back.
        quotes: Every parsed quote with its verification verdict —
            empty on fallback.
        marker: ``_knowledge_grounding_parsed`` or
            ``_knowledge_grounding_fallback`` (visible degradation).
    """

    answer: str
    quotes: list[QuoteCheck]
    marker: str


def _normalize(text: str) -> str:
    """Collapse all whitespace runs so line wrapping never fails a match."""
    return " ".join(text.split())


def check_grounding(
    content: str, evidence_texts: list[str]
) -> GroundingReport:
    """Parse the quote block and verify each quote against its evidence.

    Args:
        content: Raw model output, expected to contain a ``ZITATE:``
            block followed by an ``ANTWORT:`` section.
        evidence_texts: The chunks' SOURCE texts in label order
            (index 0 is ``K1``) — what the cited document actually
            contains, deliberately WITHOUT contextualization prefixes
            or rendering scaffolding.

    Returns:
        A :class:`GroundingReport`. Missing ``ANTWORT:`` separator or
        an empty post-separator answer yields the fallback marker with
        the ORIGINAL content as answer; an empty quote block yields the
        fallback marker with the STRIPPED answer — never an exception
        and never a silently degraded result.
    """
    split = _ANSWER_SPLIT.search(content)
    if split is None:
        return GroundingReport(
            answer=content, quotes=[], marker=GROUNDING_MARKER_FALLBACK
        )
    quote_section = content[: split.start()]
    answer = content[split.end() :].strip()
    if not answer:
        # Truncated completion (the output ends at the separator):
        # nothing user-facing exists to strip down to — keep the full
        # content AND report the degradation; a parsed marker here
        # would misreport scaffolding as a clean answer.
        return GroundingReport(
            answer=content, quotes=[], marker=GROUNDING_MARKER_FALLBACK
        )

    quotes: list[QuoteCheck] = []
    for match in _QUOTE_LINE.finditer(quote_section):
        index = int(match.group(1))
        text = match.group(2).strip().strip(_QUOTE_CHARS).strip()
        if not text:
            continue
        verified = (
            1 <= index <= len(evidence_texts)
            and _normalize(text) in _normalize(evidence_texts[index - 1])
        )
        quotes.append(
            QuoteCheck(label=f"K{index}", text=text, verified=verified)
        )
    if not quotes:
        return GroundingReport(
            answer=answer, quotes=[], marker=GROUNDING_MARKER_FALLBACK
        )
    return GroundingReport(
        answer=answer, quotes=quotes, marker=GROUNDING_MARKER_PARSED
    )
