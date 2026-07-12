"""Phase 6 — evidence merge, contradiction analysis, sufficiency (§4).

Dedup uses EXACTLY the citation identity of the research pipeline
(``doc:{document_id}#{chunk_index}`` for internal, canonical URL for
web) so a source cited by two tasks becomes ONE labelled reference.
Contradiction analysis is one mid-tier call over LEXICALLY OVERLAPPING
claim pairs only (``claim_signature`` prefilter — never O(n²) LLM).
Sufficiency is hybrid: a deterministic criterion→task check plus one
fast-tier gate-style judgement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from inqtrix.agents.patterns._structured import StructuredOutcome, structured_call
from inqtrix.agents.phase_models import (
    ContradictionReport,
    SufficiencyJudgement,
)
from inqtrix.agents.scheduler import TaskOutcome
from inqtrix.urls import normalize_url

if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider

GROUNDED_SUPPORT_MAX_CHARS = 600
"""Maximum persisted provider-answer support per cited URL.

``grounded_support`` is a bounded passage from the provider's synthesized,
grounded answer. It is explicitly not a verbatim source excerpt; real
``excerpt``/``source_text`` fields always remain the stronger evidence.
"""

_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_ANSWER_URL = re.compile(r"https?://[^\s\])}>,\"']+")
_SENTENCE_END = re.compile(
    r"(?<!\b[A-Za-zÄÖÜäöü])[.!?](?:[\"'”’)]*)\s+"
    r"(?=[A-ZÄÖÜ0-9#*_(])"
)


def dedup_key(ref: dict[str, Any]) -> str:
    """The citation identity (result.py precedent, decision E4)."""
    document_id = ref.get("document_id")
    if document_id is not None and ref.get("chunk_index") is not None:
        return f"doc:{document_id}#{ref['chunk_index']}"
    return normalize_url(str(ref.get("url", "") or ""))


def enrich_instant_evidence(
    grounded_answer: str,
    references: list[dict[str, Any]],
    *,
    support_chars: int = GROUNDED_SUPPORT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Associate cited URLs with provider-grounded supporting prose.

    The instant-search provider returns one grounded answer plus a source
    list. Some providers expose rich per-source snippets; Azure Foundry
    exposes only cited URLs and titles. For URLs that occur in the grounded
    answer, this helper deterministically captures the containing statement
    as ``grounded_support``. That field records what the provider answer said
    near the citation and must never be presented as a verbatim source quote.

    Args:
        grounded_answer: Full provider-grounded answer, potentially with
            inline Markdown links.
        references: Provider-neutral evidence dictionaries containing URLs.
        support_chars: Hard character cap for each persisted support passage.

    Returns:
        Fresh evidence dictionaries. Existing fields are preserved and an
        existing ``grounded_support`` is never overwritten.
    """
    if support_chars < 1:
        raise ValueError("support_chars must be at least 1")
    answer = str(grounded_answer or "").replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    enriched: list[dict[str, Any]] = []
    for raw in references:
        ref = dict(raw)
        if ref.get("grounded_support") or not answer:
            enriched.append(ref)
            continue
        url = str(ref.get("url") or "").strip()
        position = _citation_position(answer, url)
        if position >= 0:
            support = _grounded_statement(
                answer,
                position=position,
                url=url,
                support_chars=support_chars,
            )
            if support:
                ref["grounded_support"] = support
        enriched.append(ref)
    return enriched


def _citation_position(answer: str, url: str) -> int:
    target = normalize_url(url)
    if not target:
        return -1
    for match in _ANSWER_URL.finditer(answer):
        if normalize_url(match.group(0)) == target:
            return match.start()
    return -1


def _grounded_statement(
    answer: str,
    *,
    position: int,
    url: str,
    support_chars: int,
) -> str:
    preceding_break = answer.rfind("\n\n", 0, position)
    paragraph_start = 0 if preceding_break < 0 else preceding_break + 2
    paragraph_end = answer.find("\n\n", position)
    if paragraph_end < 0:
        paragraph_end = len(answer)
    paragraph = answer[paragraph_start:paragraph_end]
    relative_position = position - paragraph_start

    sentence_start = 0
    sentence_end = len(paragraph)
    for boundary in _SENTENCE_END.finditer(paragraph):
        boundary_end = boundary.end()
        if boundary_end <= relative_position:
            sentence_start = boundary_end
            continue
        sentence_end = boundary.end() - len(boundary.group(0)) + 1
        break

    statement = paragraph[sentence_start:sentence_end].strip()
    relative_in_statement = max(0, relative_position - sentence_start)
    if len(statement) > support_chars:
        # Citations usually follow the claim. Preserve more context before
        # the URL while retaining a small suffix for qualifiers.
        window_start = max(0, relative_in_statement - (support_chars * 4 // 5))
        window_end = min(len(statement), window_start + support_chars)
        if window_end - window_start < support_chars:
            window_start = max(0, window_end - support_chars)
        statement = statement[window_start:window_end]
        if window_start:
            statement = "…" + statement.lstrip()
        if window_end < len(paragraph[sentence_start:sentence_end].strip()):
            statement = statement.rstrip() + "…"

    statement = _MARKDOWN_LINK.sub(r"\1", statement)
    target_url = normalize_url(url)
    statement = _ANSWER_URL.sub(
        lambda match: (
            ""
            if normalize_url(match.group(0)) == target_url
            else match.group(0)
        ),
        statement,
    )
    statement = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", statement)
    statement = re.sub(r"^[ \t]*(?:[-*+] |#+ |>|\d+[.)] )", "", statement)
    statement = " ".join(statement.split()).strip(" -–—")
    if len(statement) > support_chars:
        statement = statement[: support_chars - 1].rstrip() + "…"
    return statement


def merge_evidence(
    outcomes: dict[str, TaskOutcome],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge task evidence into labelled references + collected claims.

    Returns ``(references, claims)``; each reference gains a stable
    ``label`` (``K#`` internal, ``W#`` web) in first-seen order — the
    labels the synthesis cites.
    """
    references: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    counters = {"K": 0, "W": 0}
    claims: list[dict[str, Any]] = []
    for task_id in sorted(outcomes):
        outcome = outcomes[task_id]
        for ref in outcome.evidence:
            key = dedup_key(ref)
            if not key:
                continue
            if key in seen:
                canonical = seen[key]
                if task_id not in canonical["tasks"]:
                    canonical["tasks"].append(task_id)
                # First-seen order owns the label, not evidence quality. A
                # later task may carry the real excerpt that an earlier
                # URL-only provider record lacked. Fill only missing fields;
                # an existing real source excerpt is never replaced by
                # provider-synthesized support or another duplicate.
                merge_missing_evidence_fields(canonical, ref)
                continue
            prefix = "K" if ref.get("document_id") is not None else "W"
            counters[prefix] += 1
            labelled = {
                **ref,
                "label": f"{prefix}{counters[prefix]}",
                "tasks": [task_id],
            }
            seen[key] = labelled
            references.append(labelled)
        for claim in outcome.claims:
            claims.append({**claim, "task_id": task_id})
    return references, claims


def merge_missing_evidence_fields(
    canonical: dict[str, Any], incoming: dict[str, Any]
) -> bool:
    """Fill absent evidence metadata without replacing stronger truth.

    Returns ``True`` when *canonical* changed. Stable identity fields and
    first-seen labels stay untouched; the function only fills an empty value.
    """
    changed = False
    for field, value in incoming.items():
        if field in {"label", "tasks"}:
            continue
        if _missing_evidence_value(
            canonical.get(field)
        ) and not _missing_evidence_value(value):
            canonical[field] = value
            changed = True
    return changed


def _missing_evidence_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return value == [] or value == {}


def overlapping_claim_pairs(
    claims: list[dict[str, Any]],
    *,
    signature: Any,
    limit: int = 12,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Lexically overlapping claim pairs (the LLM prefilter).

    *signature* is ``claim_signature(text) -> str`` from the claim
    consolidation strategy — the ONE lexical normalizer (Prinzip 4).
    """
    by_signature: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        text = str(claim.get("text", ""))
        if not text:
            continue
        by_signature.setdefault(signature(text), []).append(claim)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for bucket in by_signature.values():
        for index in range(len(bucket) - 1):
            pairs.append((bucket[index], bucket[index + 1]))
            if len(pairs) >= limit:
                return pairs
    return pairs


def run_contradiction_analysis(
    llm: "LLMProvider",
    *,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
) -> StructuredOutcome | None:
    """One mid-tier call over the overlapping pairs; ``None`` when there
    is nothing to compare (no call, no cost)."""
    if not pairs:
        return None
    lines = [
        f"Paar {index + 1}:\n  A: {a.get('text', '')}\n  B: {b.get('text', '')}"
        for index, (a, b) in enumerate(pairs)
    ]
    return structured_call(
        llm,
        prompt=(
            "Pruefe die folgenden Behauptungs-Paare auf Widersprueche. "
            "Nur echte inhaltliche Konflikte melden (hard = unvereinbar, "
            "soft = Spannungslage) mit wahrscheinlicher Ursache.\n\n"
            + "\n\n".join(lines)
        ),
        model_cls=ContradictionReport,
        node="agent_contradiction",
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def run_sufficiency_judgement(
    llm: "LLMProvider",
    *,
    success_criteria: list[str],
    evidence_digest: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
) -> StructuredOutcome:
    """The fast-tier three-way coverage verdict (gate semantics)."""
    criteria = "\n".join(f"- {c}" for c in success_criteria) or "- (keine)"
    return structured_call(
        llm,
        prompt=(
            f"Erfolgskriterien:\n{criteria}\n\n"
            f"Vorliegende Belege:\n{evidence_digest}\n\n"
            "Beurteile die Abdeckung: covered, partial oder uncovered, "
            "plus die Kriterien ohne ausreichende Belege."
        ),
        model_cls=SufficiencyJudgement,
        node="agent_sufficiency",
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def evidence_digest(
    references: list[dict[str, Any]],
    *,
    excerpt_chars: int = 200,
    labels: Iterable[str] | None = None,
) -> str:
    """Label -> compact prompt digest, optionally scoped to one section."""
    allowed = set(labels) if labels is not None else None
    lines = []
    for ref in references:
        if allowed is not None and ref.get("label") not in allowed:
            continue
        source_body = ref.get("excerpt") or ref.get("source_text")
        grounded_support = ref.get("grounded_support")
        if source_body:
            body = str(source_body)[:excerpt_chars]
        elif grounded_support:
            support = str(grounded_support)[:excerpt_chars]
            body = (
                "Geerdeter Antwortkontext (kein Quellenauszug): "
                f"{support}"
            )
        else:
            body = str(ref.get("title") or ref.get("url") or "")[
                :excerpt_chars
            ]
        lines.append(f"[{ref['label']}] {body}")
    return "\n".join(lines) or "(keine Belege)"
