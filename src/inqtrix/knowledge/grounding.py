"""Quote-then-answer grounding: deterministic verbatim-quote checks.

The answer prompt (``build_knowledge_answer_prompt(grounding=True)``)
requires the model to emit a ``ZITATE:`` block of verbatim,
``[K#]``-labelled quotes BEFORE the ``ANTWORT:`` section. Writing the
quotes first conditions the subsequent answer on literal evidence
(WebGPT/GopherCite lineage — the cheapest proven grounding step), and
because the quotes claim to be verbatim they can be validated WITHOUT
another LLM call: a substring check against the exact evidence entries
the model saw, tolerant only to formatting (whitespace, Unicode/typography
and case) so a genuinely verbatim quote is not failed by curly quotes,
ligatures or line wrapping — never to paraphrase.

Failure policy (No Silent Fallbacks): a response without a parseable
quote block degrades to the unmodified answer with a loud log and the
``_knowledge_grounding_fallback`` marker; quotes that do not verify
stay in the report with ``verified=False`` so callers can surface
them — the check never blocks or rewrites the answer.
"""

from __future__ import annotations

import re
import unicodedata
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

# Typographic variants the model or a PDF text-layer may emit for the SAME
# character: smart quotes -> ASCII quotes, dashes -> hyphen, zero-width
# characters dropped. Folded so a genuinely verbatim quote still verifies even
# when the source PDF used curly quotes or an em dash. Deliberately limited to
# encoding/typography (never paraphrase).
_TYPOGRAPHY = str.maketrans(
    {
        "„": '"', "“": '"', "”": '"', "»": '"', "«": '"',
        "‚": "'", "‘": "'", "’": "'",
        "–": "-", "—": "-", "―": "-",
        "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    }
)


@dataclass(frozen=True)
class QuoteCheck:
    """One quote from the model's ``ZITATE:`` block, with its verdict.

    Attributes:
        label: The evidence label the model attributed the quote to,
            e.g. ``"K1"`` — 1-based position in the evidence block.
        text: The quote text with surrounding quotation marks removed.
        verified: ``True`` when the quote occurs verbatim in the
            evidence entry the label points at, tolerant only to
            formatting (whitespace, Unicode/typography, case);
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
    """Fold typographic/encoding differences so a verbatim quote still matches.

    The check stays a verbatim-substring match — only formatting is tolerated,
    never paraphrase. NFKC folds ligatures, full-width and compatibility forms
    (and maps no-break spaces to plain spaces); smart quotes and dashes map to
    ASCII and zero-width characters drop (:data:`_TYPOGRAPHY`); whitespace runs
    collapse (line wrapping); case folds. A reworded quote still differs in its
    words and fails — no false ``verified``.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    return " ".join(folded.split()).casefold()


def quote_is_verbatim(text: str, evidence_texts: list[str]) -> bool:
    """Whether *text* appears verbatim in ANY of the evidence texts.

    The agent-side quote check (``agents/report_quality.verify_quotes``)
    reuses the ONE normalizer chain through this helper instead of the
    ``check_grounding`` answer-block contract — that parser is bound to
    per-label indices and the ``ANTWORT:`` scaffold, neither of which
    exists for memo quotes checked against the whole evidence ledger.

    Args:
        text: The quoted passage (without surrounding quote marks).
        evidence_texts: Stored source texts captured at retrieval time
            (internal chunks and web excerpts alike).

    Returns:
        ``True`` when the normalizer-folded quote is a substring of at
        least one normalizer-folded evidence text.
    """
    normalized = _normalize(text)
    if not normalized:
        return False
    return any(
        normalized in _normalize(evidence)
        for evidence in evidence_texts
        if evidence
    )


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
