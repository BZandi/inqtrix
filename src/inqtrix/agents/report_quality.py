"""Deterministic report-quality building blocks shared by BOTH engines.

The mission phase machine (``synthesis.py``/``algorithm.py``) and the
cognitive kernel (``kernel/tools.py``) consume the SAME citation
validation, quote grounding, and evidence ranking — one definition,
never a per-engine copy (Designprinzip 4). Everything here except
:func:`validate_and_repair_citations` is pure and LLM-free.

Quote verification is EXCERPT-BASED: quoted passages are checked
verbatim against the evidence texts stored at retrieval time (internal
chunks and web excerpts alike). Nothing is re-fetched and model memory
is never consulted, so a quote that only exists beyond the stored
excerpt window stays visibly unverified rather than silently trusted.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable

from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.phase_models import CitationRepairText
from inqtrix.agents.patterns._structured import structured_call
from inqtrix.domains import SOURCE_TIER_WEIGHTS

if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider

_CITATION_LABEL = re.compile(r"\[([KW]\d+)\]")
_WEB_LABEL = re.compile(r"\[(W\d+)\]")
_QUOTE = re.compile(r"\"([^\"]{20,400})\"")
log = logging.getLogger("inqtrix")


class CitationValidationFailed(RuntimeError):
    """Raised when one bounded repair cannot restore citation integrity."""

    def __init__(self, message: str, *, usage: dict[str, int]) -> None:
        super().__init__(message)
        self.usage = dict(usage)
        """Token usage of the rejected synthesis and repair calls."""


def _merge_usage(*values: dict[str, int]) -> dict[str, int]:
    """Add token usage from one synthesis call and its optional repair."""
    return {
        "prompt_tokens": sum(int(value.get("prompt_tokens", 0)) for value in values),
        "completion_tokens": sum(
            int(value.get("completion_tokens", 0)) for value in values
        ),
    }


def validate_and_repair_citations(
    llm: "LLMProvider",
    *,
    markdown: str,
    known_labels: list[str],
    usage: dict[str, int],
    system: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
) -> tuple[str, dict[str, int]]:
    """Accept known labels or make exactly one complete-text repair call.

    Raises:
        CitationValidationFailed: The single bounded repair round still
            left unknown labels (or produced no text) — the caller fails
            loudly instead of shipping invented citations.
    """
    from inqtrix.agents.prompts import build_agent_citation_repair_prompt

    references = [{"label": label} for label in known_labels]
    unknown = unknown_citation_labels(markdown, references)
    if not unknown:
        return markdown, usage

    log.warning(
        "Agent synthesis used unknown citation labels %s; starting one "
        "bounded citation repair.",
        unknown,
    )
    repair = structured_call(
        llm,
        prompt=build_agent_citation_repair_prompt(
            markdown,
            allowed_labels=known_labels,
        ),
        model_cls=CitationRepairText,
        node="agent_citation_repair",
        system=system,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    repaired = repair.value
    repaired_markdown = (
        repaired.markdown.strip()
        if isinstance(repaired, CitationRepairText)
        else ""
    )
    remaining = unknown_citation_labels(repaired_markdown, references)
    merged_usage = _merge_usage(usage, repair.usage)
    if not repaired_markdown or remaining:
        raise CitationValidationFailed(
            "Citation repair did not produce a complete text with only known "
            f"labels (remaining={remaining or ['empty_output']}).",
            usage=merged_usage,
        )
    return (
        normalize_agent_markdown(repaired_markdown),
        merged_usage,
    )


def citation_coverage(markdown: str) -> dict[str, int]:
    """Deterministic citation stats the critic receives as facts."""
    paragraphs = [
        paragraph.strip()
        for paragraph in markdown.split("\n\n")
        if paragraph.strip() and not paragraph.lstrip().startswith("#")
    ]
    cited = sum(
        1 for paragraph in paragraphs if _CITATION_LABEL.search(paragraph)
    )
    return {
        "paragraphs": len(paragraphs),
        "cited_paragraphs": cited,
        "labels_used": len(set(_CITATION_LABEL.findall(markdown))),
    }


def cited_references(
    markdown: str, references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return known references actually cited by the rendered output.

    Order follows the canonical ledger, not textual citation order, so label
    identity stays stable across partial memo flushes and revisions.
    """
    used = set(_CITATION_LABEL.findall(markdown))
    return [
        dict(reference)
        for reference in references
        if str(reference.get("label") or "") in used
    ]


def unknown_citation_labels(
    markdown: str, references: list[dict[str, Any]]
) -> list[str]:
    """Return cited K/W labels absent from the canonical evidence ledger."""
    known = {str(reference.get("label") or "") for reference in references}
    return sorted(set(_CITATION_LABEL.findall(markdown)) - known)


def verify_quotes(
    markdown: str, references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verbatim-verify quoted passages against ALL stored evidence texts.

    Internal chunks AND web excerpts count — a quote attributed to a web
    source is checked against the excerpt captured at retrieval time,
    never against a re-fetch or model memory (CitationAgent pattern).
    Uses :func:`inqtrix.knowledge.grounding.quote_is_verbatim` — the
    platform's ONE normalizer chain (Prinzip 4). The earlier detour
    through ``check_grounding`` never verified anything: that parser
    expects ``[K#]`` lines and an ``ANTWORT:`` scaffold, so every memo
    quote silently came back unverified.
    Returns ``[{"quote", "verified"}]``; unverified stays visible.
    """
    from inqtrix.knowledge.grounding import quote_is_verbatim

    evidence_texts = [
        text
        for ref in references
        if (text := str(ref.get("excerpt") or ref.get("source_text") or ""))
    ]
    quotes = _QUOTE.findall(markdown)
    if not quotes:
        return []
    # Missing original text is not the same as "nothing to verify".  A web
    # result may legitimately carry only a URL or provider summary; treating
    # that state as an empty check list made the strict/tief critic believe no
    # unverified quote existed.  Preserve every quote as an explicit failed
    # verification until canonical evidence is available.
    if not evidence_texts:
        return [{"quote": quote, "verified": False} for quote in quotes]
    return [
        {"quote": quote, "verified": quote_is_verbatim(quote, evidence_texts)}
        for quote in quotes
    ]


def unverified_web_quotes(
    markdown: str, quote_checks: list[dict[str, Any]]
) -> list[str]:
    """Unverified quotes standing in a web-cited ([W#]) paragraph.

    The deterministic escalation trigger of the ``tief`` tier: a quoted
    claim that leans on a web citation but could not be matched against
    any stored excerpt. Paragraph scope keeps the attribution honest
    without a claim-to-source model that the ledger does not carry.
    """
    unverified = {
        str(check.get("quote") or "")
        for check in quote_checks
        if not check.get("verified")
    }
    if not unverified:
        return []
    flagged: list[str] = []
    for paragraph in markdown.split("\n\n"):
        if not _WEB_LABEL.search(paragraph):
            continue
        for quote in unverified:
            if quote and quote in paragraph and quote not in flagged:
                flagged.append(quote)
    return flagged


def rank_evidence(
    references: list[dict[str, Any]],
    *,
    budget: int,
    tier_for_url: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic relevance selection for the synthesis PROMPT digests.

    Below or at ``budget`` the ledger passes through untouched (no
    behavior change for typical runs); above it the top-``budget``
    references are kept in a stable order. The score is built only from
    signals the ledger actually carries (Designprinzip 7):

    - source tier (``domains.py`` weights; internal document references
      count as ``primary`` — the user selected that corpus deliberately),
    - cross-task corroboration (how many independent tasks surfaced the
      same source),
    - whether a real excerpt exists (only those can ground quotes).

    This caps what the outline/answer PROMPT sees; the citation ledger
    itself is never truncated — labels stay resolvable end-to-end.
    No LLM call.
    """
    if budget <= 0 or len(references) <= budget:
        return references

    def score(ref: dict[str, Any]) -> float:
        if ref.get("document_id") is not None:
            tier_weight = SOURCE_TIER_WEIGHTS["primary"]
        else:
            tier = str(ref.get("tier") or "")
            if not tier and tier_for_url is not None:
                tier = tier_for_url(str(ref.get("url") or ""))
            tier_weight = SOURCE_TIER_WEIGHTS.get(tier, SOURCE_TIER_WEIGHTS["unknown"])
        corroboration = 0.3 * min(len(ref.get("tasks") or []) - 1, 3)
        has_excerpt = bool(
            str(ref.get("excerpt") or ref.get("source_text") or "").strip()
        )
        return tier_weight + max(corroboration, 0.0) + (0.25 if has_excerpt else 0.0)

    ordered = sorted(
        range(len(references)),
        key=lambda index: (-score(references[index]), index),
    )
    kept = sorted(ordered[:budget])
    log.info(
        "rank_evidence capped the synthesis digest to %d of %d references.",
        budget,
        len(references),
    )
    return [references[index] for index in kept]
