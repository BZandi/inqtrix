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

Failure policy (No Silent Fallbacks): when grounding is enabled, only a
fully parsed response whose every labelled quote verifies is publishable.
Malformed output and unverified quotes produce a typed rejection.  A single,
deterministic format repair may accept Markdown heading adornment around the
two required section labels; it never invents a quote, changes evidence text,
or calls another model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

GROUNDING_MARKER_PARSED = "_knowledge_grounding_parsed"
GROUNDING_MARKER_FALLBACK = "_knowledge_grounding_fallback"
GROUNDING_MARKER_FORMAT_REPAIRED = "_knowledge_grounding_format_repaired"

_STRICT_QUOTE_HEADER = re.compile(
    r"^[ \t]*(?:\*\*)?ZITATE:(?:\*\*)?[ \t]*$"
)
_STRICT_ANSWER_HEADER = re.compile(
    r"^[ \t]*(?:\*\*)?ANTWORT:(?:\*\*)?[ \t]*$"
)
_REPAIRED_QUOTE_HEADER = re.compile(
    r"^[ \t]*#{1,6}[ \t]+(?:\*\*)?ZITATE:(?:\*\*)?[ \t]*$"
)
_REPAIRED_ANSWER_HEADER = re.compile(
    r"^[ \t]*#{1,6}[ \t]+(?:\*\*)?ANTWORT:(?:\*\*)?[ \t]*$"
)
_QUOTE_LINE = re.compile(
    r"^[ \t]*\[K(\d+)\][ \t]*(.+?)[ \t]*$"
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


class GroundingStatus(StrEnum):
    """Terminal decision of the deterministic Knowledge grounding gate."""

    VERIFIED = "verified"
    REJECTED_FORMAT = "rejected_format"
    REJECTED_QUOTE = "rejected_quote"


class GroundingFailureCode(StrEnum):
    """Stable failure vocabulary shared with native runs and Agent tasks."""

    FORMAT_INVALID = "knowledge_grounding_format_invalid"
    QUOTE_UNVERIFIED = "knowledge_grounding_quote_unverified"


def grounding_failure_message(
    code: GroundingFailureCode, *, language: str
) -> str:
    """Return the safe visible explanation for a rejected grounded answer."""

    german = str(language or "").lower().startswith("de")
    if code is GroundingFailureCode.QUOTE_UNVERIFIED:
        return (
            "Die Antwort wurde nicht veröffentlicht, weil mindestens ein "
            "als wörtlich ausgewiesenes Zitat nicht im zugeordneten "
            "Originalbeleg nachweisbar war."
            if german
            else "The answer was not published because at least one quote "
            "presented as verbatim could not be verified in its assigned "
            "original evidence."
        )
    return (
        "Die Antwort wurde nicht veröffentlicht, weil der erforderliche "
        "Belegblock nicht sicher gelesen und geprüft werden konnte."
        if german
        else "The answer was not published because the required evidence "
        "block could not be parsed and verified safely."
    )


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of parsing and verifying one grounded answer.

    Attributes:
        answer: Publishable user-facing answer with the quote block stripped.
            Empty for every rejected result so callers cannot accidentally
            expose the unchecked model completion.
        quotes: Every parsed quote with its verification verdict —
            empty when the output shape cannot be parsed.
        marker: Stable audit marker for strict parse, deterministic format
            repair, or format rejection.
        status: Typed terminal decision.  Only ``VERIFIED`` is publishable.
        failure_code: Stable run/task error type for rejected results.
        format_repaired: Whether the one bounded Markdown-heading repair was
            required.  No repair changes quote or answer content.
    """

    answer: str
    quotes: list[QuoteCheck]
    marker: str
    status: GroundingStatus
    failure_code: GroundingFailureCode | None = None
    format_repaired: bool = False

    @property
    def publishable(self) -> bool:
        """Whether this report may become a normal Knowledge answer."""

        return self.status is GroundingStatus.VERIFIED


@dataclass(frozen=True)
class _ParsedGroundingShape:
    """Internal parse result before label-bound quote verification."""

    answer: str
    quotes: list[tuple[int, str]]
    format_repaired: bool


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
        A :class:`GroundingReport`.  Missing/ambiguous headers, malformed quote
        lines, an empty quote block, or an empty answer are typed format
        rejections with no publishable answer.  A fully parsed response is
        still rejected unless every quote is verbatim in the evidence entry
        named by its label.
    """
    shape = _parse_grounding_shape(content, allow_heading_repair=False)
    if shape is None:
        # The sole repair is syntactic and bounded: Markdown heading markers
        # may adorn BOTH required headers.  Missing headers, invented section
        # names, prose in the quote block, or malformed quote lines remain a
        # rejection; no second model call and no fuzzy parser are allowed.
        shape = _parse_grounding_shape(content, allow_heading_repair=True)
    if shape is None:
        return GroundingReport(
            answer="",
            quotes=[],
            marker=GROUNDING_MARKER_FALLBACK,
            status=GroundingStatus.REJECTED_FORMAT,
            failure_code=GroundingFailureCode.FORMAT_INVALID,
        )

    quotes: list[QuoteCheck] = []
    for index, text in shape.quotes:
        verified = (
            1 <= index <= len(evidence_texts)
            and _normalize(text) in _normalize(evidence_texts[index - 1])
        )
        quotes.append(
            QuoteCheck(label=f"K{index}", text=text, verified=verified)
        )
    if any(not quote.verified for quote in quotes):
        return GroundingReport(
            answer="",
            quotes=quotes,
            marker=(
                GROUNDING_MARKER_FORMAT_REPAIRED
                if shape.format_repaired
                else GROUNDING_MARKER_PARSED
            ),
            status=GroundingStatus.REJECTED_QUOTE,
            failure_code=GroundingFailureCode.QUOTE_UNVERIFIED,
            format_repaired=shape.format_repaired,
        )
    return GroundingReport(
        answer=shape.answer,
        quotes=quotes,
        marker=(
            GROUNDING_MARKER_FORMAT_REPAIRED
            if shape.format_repaired
            else GROUNDING_MARKER_PARSED
        ),
        status=GroundingStatus.VERIFIED,
        format_repaired=shape.format_repaired,
    )


def _parse_grounding_shape(
    content: str, *, allow_heading_repair: bool
) -> _ParsedGroundingShape | None:
    """Parse exactly one quote section and one answer section.

    The repair path differs only in accepting a Markdown heading prefix on
    both required headers.  All other grammar stays identical and fail-closed.
    """

    lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    quote_header = (
        _REPAIRED_QUOTE_HEADER if allow_heading_repair else _STRICT_QUOTE_HEADER
    )
    answer_header = (
        _REPAIRED_ANSWER_HEADER if allow_heading_repair else _STRICT_ANSWER_HEADER
    )
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty or quote_header.fullmatch(lines[nonempty[0]]) is None:
        return None
    quote_index = nonempty[0]
    answer_indexes = [
        index
        for index in range(quote_index + 1, len(lines))
        if answer_header.fullmatch(lines[index]) is not None
    ]
    if len(answer_indexes) != 1:
        return None
    answer_index = answer_indexes[0]

    parsed_quotes: list[tuple[int, str]] = []
    for line in lines[quote_index + 1 : answer_index]:
        if not line.strip():
            continue
        match = _QUOTE_LINE.fullmatch(line)
        if match is None:
            return None
        text = match.group(2).strip().strip(_QUOTE_CHARS).strip()
        if not text:
            return None
        parsed_quotes.append((int(match.group(1)), text))
    answer = "\n".join(lines[answer_index + 1 :]).strip()
    if not parsed_quotes or not answer:
        return None
    return _ParsedGroundingShape(
        answer=answer,
        quotes=parsed_quotes,
        format_repaired=allow_heading_repair,
    )
