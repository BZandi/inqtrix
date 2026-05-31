"""EvidenceLedger helpers: assembly, claim projection, and the single
record-driven evidence overview rendered for the final answer prompt."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from inqtrix.runtime_logging import make_record_id
from inqtrix.urls import domain_from_url, normalize_url

# Evidence content (snippets, passages, claims, the per-query synthesis) is
# stored and rendered in full -- there are no per-field character caps. The
# only length bound left is the answer-prompt render budget
# (``max_total_chars`` / ``max_record_chars``), which truncates visibly with a
# marker and reports an omitted-record count.

# Trust ranking for evidence records when the overview character budget is
# tight. Low-tier sources sink to the bottom but are not hard-excluded.
_SOURCE_CONTEXT_TIER_RANK: dict[str, int] = {
    "primary": 60,
    "mainstream": 50,
    "stakeholder": 40,
    "unknown": 25,
    "low": -100,
}

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_OVERVIEW_PATH_RE = re.compile(
    r"(^|/)(latest|category|categories|topics|tag|tags|events-calendar|ai-events|news)(/|$)",
    re.IGNORECASE,
)
_EVIDENCE_LABEL_RE = re.compile(r"^E(\d+)$")
_ADJACENT_CITATION_MARKER_RE = re.compile(
    r"(\[(?:E\d+|unmapped:[^\]]+|nicht-gerendert:[^\]]+)\])"
    r"(?=\[(?:E\d+|unmapped:|nicht-gerendert:))"
)


def evidence_id_for_citation(query_id: str, citation_record: dict[str, Any]) -> str:
    """Return the stable EvidenceRecord id for one citation record.

    Args:
        query_id: Query that produced the citation.
        citation_record: Normalized provider citation record.

    Returns:
        Stable EvidenceRecord identifier. The same query/citation/URL
        tuple always maps to the same id within and across runs.
    """
    canonical = normalize_url(str(citation_record.get("canonical_url", "") or ""))
    citation_id = str(citation_record.get("citation_id", "") or "")
    source_id = str(citation_record.get("source_id", "") or "")
    return make_record_id("ev", query_id, citation_id or source_id, canonical)


def _bounded_text(text: Any, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _full_text(text: Any) -> str:
    """Whitespace-normalized full text on a single line, with no length cap.

    Used for evidence content fields (claims, snippets, passages). The only
    length bound left is the answer-prompt render budget, which truncates
    visibly with a marker.
    """
    return " ".join(str(text or "").split())


def _full_block(text: Any) -> str:
    """Full block text that preserves paragraph/bullet newlines, no cap.

    Used for the per-query synthesis (summary and provider answer) where
    newlines carry list structure. Collapses only runs of spaces/tabs and
    excess blank lines.
    """
    raw = str(text or "")
    if not raw.strip():
        return ""
    normalized = re.sub(r"[ \t]+", " ", raw).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


_NUMERIC_CITATION_RE = re.compile(r"\[(?:web:)?(\d+)\]")


def _space_adjacent_evidence_labels(text: str) -> str:
    """Insert a space between adjacent evidence/citation diagnostic markers."""
    return _ADJACENT_CITATION_MARKER_RE.sub(r"\1 ", text)


def _remap_synthesis_citations(
    text: str,
    url_to_label: dict[str, str],
    rank_to_url: dict[str, str],
) -> str:
    """Rewrite a provider answer's inline citations to global ``[E#]`` labels.

    Maps Markdown-link citations (Azure ``([title](url))``) and numeric
    markers (Perplexity ``[2]`` / ``[web:2]``) to the stable evidence label
    of the cited source, so the synthesis cites the same ``[E#]`` labels as
    the rendered source blocks and carries no token-heavy inline URLs. The
    final answer LLM therefore only ever sees compact labels; the URLs are
    re-attached after synthesis. Citations that cannot be mapped are kept as
    visible diagnostic markers instead of disappearing silently. Numeric
    provider refs whose URL exists but whose source block is not visible
    become ``[nicht-gerendert:n]`` so they cannot be mistaken for valid
    evidence labels.

    Args:
        text: The provider's raw synthesized answer.
        url_to_label: Canonical URL to ``E#`` label for the query's sources.
        rank_to_url: Provider source rank/id to URL (Perplexity numeric ids).

    Returns:
        str: The answer with inline citations rewritten to ``[E#]`` labels.
    """
    if not text:
        return ""

    def _label_for_url(url: str) -> str | None:
        return url_to_label.get(normalize_url(str(url or "")))

    def _md(match: re.Match[str]) -> str:
        label = _label_for_url(match.group(2))
        return f"[{label}]" if label else f"{match.group(1)} [nicht-gerendert:url]"

    out = _MARKDOWN_LINK_RE.sub(_md, text)

    def _num(match: re.Match[str]) -> str:
        ref = match.group(1)
        url = rank_to_url.get(ref)
        if not url:
            return f"[unmapped:{ref}]"
        label = _label_for_url(url) if url else None
        return f"[{label}]" if label else f"[nicht-gerendert:{ref}]"

    out = _NUMERIC_CITATION_RE.sub(_num, out)
    out = re.sub(r"\(\s*(\[E\d+\])\s*\)", r"\1", out)
    out = re.sub(r"[ \t]+([.,;:)])", r"\1", out)
    out = _space_adjacent_evidence_labels(out)
    return out


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _passages_for_record(
    *,
    evidence_id: str,
    citation_record: dict[str, Any],
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []

    def add(origin: str, text: Any) -> None:
        bounded = _full_text(text)
        if not bounded:
            return
        if any(p["text"] == bounded for p in passages):
            return
        passages.append(
            {
                "passage_id": make_record_id("passage", evidence_id, origin, len(passages) + 1),
                "origin": origin,
                "text": bounded,
                "char_count": len(bounded),
            }
        )

    add("source_snippet", citation_record.get("snippet", ""))
    for claim in claims:
        add("claim_evidence_snippet", claim.get("evidence_snippet", ""))
    return passages


def _citation_set_for_records(citation_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citation_set: list[dict[str, Any]] = []
    for index, citation in enumerate(citation_records, start=1):
        citation_set.append(
            {
                "label": f"E1.{index}",
                "evidence_id": "",
                "citation_id": citation.get("citation_id", ""),
                "source_id": citation.get("source_id", ""),
                "url": normalize_url(str(citation.get("canonical_url", "") or "")),
                "role": "source",
                "title": citation.get("title", ""),
                "snippet": citation.get("snippet", ""),
            }
        )
    return citation_set


def _claim_citation_set(
    claim: dict[str, Any],
    citation_records: list[dict[str, Any]],
    *,
    evidence_id: str,
    allow_overview_without_snippet: bool = False,
) -> list[dict[str, Any]]:
    claim_urls = {
        normalize_url(str(url))
        for url in claim.get("source_urls", []) or []
        if normalize_url(str(url))
    }
    if not claim_urls:
        return []
    rows = []
    for index, citation in enumerate(citation_records, start=1):
        url = normalize_url(str(citation.get("canonical_url", "") or ""))
        if claim_urls and url not in claim_urls:
            continue
        if (
            not allow_overview_without_snippet
            and _is_overview_url(url)
            and not str(citation.get("snippet", "")).strip()
        ):
            continue
        rows.append(
            {
                "label": f"E1.{index}",
                "evidence_id": evidence_id,
                "citation_id": citation.get("citation_id", ""),
                "source_id": citation.get("source_id", ""),
                "url": url,
                "role": "source",
                "title": citation.get("title", ""),
                "snippet": citation.get("snippet", ""),
            }
        )
    return rows


def _is_overview_url(url: str) -> bool:
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return False
    return bool(_OVERVIEW_PATH_RE.search(path.rstrip("/")))


def _claim_payload(
    claim: dict[str, Any],
    *,
    citation_set: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "raw_claim_id": claim.get("raw_claim_id", ""),
        "claim_text": claim.get("claim_text", ""),
        "claim_type": claim.get("claim_type", "fact"),
        "polarity": claim.get("polarity", "affirmed"),
        "needs_primary": bool(claim.get("needs_primary", False)),
        "evidence_snippet": claim.get("evidence_snippet", ""),
        "verification_status": claim.get("verification_status", "unverified"),
        "verification_basis": claim.get("verification_basis", ""),
        "supporting_evidence_ids": claim.get("supporting_evidence_ids", []),
        "supporting_domain_count": int(claim.get("supporting_domain_count", 0) or 0),
        "published_date": claim.get("published_date", "unknown"),
        "signature": claim.get("signature", ""),
        "citation_set": citation_set,
    }
    return payload


def assemble_evidence_records(
    *,
    query_id: str,
    query: str,
    provider: str,
    source_records: list[dict[str, Any]],
    citation_records: list[dict[str, Any]],
    claim_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble per-source EvidenceRecords for one search result.

    One record per cited source (de-duplicated by EvidenceRecord id). The
    synthesized provider answer and the query summary are NOT stored on the
    records: they live once per query in ``state["query_synthesis"]`` so
    they are never duplicated across every source of a query. A record is
    ``report_eligible`` when it carries a citable URL, so URL-only sources
    (e.g. Azure Foundry, whose answer exposes no per-source body) still
    become tier-classified citation anchors instead of being dropped.

    Args:
        query_id: Stable query id.
        query: Query text.
        provider: Search provider class label.
        source_records: Normalized source registry rows for the query.
        citation_records: Normalized provider citation rows for the query.
        claim_entries: Raw claim-support entries already linked to source
            URLs and EvidenceRecord ids.

    Returns:
        EvidenceRecord dicts keyed by query plus citation/source. Each
        record carries source metadata, full text passages, and the raw
        claims supported by that source.
    """
    source_by_id = {str(record.get("source_id", "")): record for record in source_records}
    claims_by_evidence_id: dict[str, list[dict[str, Any]]] = {}
    for claim in claim_entries:
        for evidence_id in claim.get("evidence_ids", []) or []:
            claims_by_evidence_id.setdefault(str(evidence_id), []).append(claim)

    records: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()

    for citation_record in citation_records:
        evidence_id = evidence_id_for_citation(query_id, citation_record)
        if evidence_id in seen_evidence_ids:
            continue
        seen_evidence_ids.add(evidence_id)
        source_id = str(citation_record.get("source_id", "") or "")
        source_record = source_by_id.get(source_id, {})
        canonical_url = normalize_url(str(citation_record.get("canonical_url", "") or ""))
        claims = claims_by_evidence_id.get(evidence_id, [])
        passages = _passages_for_record(
            evidence_id=evidence_id,
            citation_record=citation_record,
            claims=claims,
        )
        citation_set = _citation_set_for_records([citation_record])
        for citation in citation_set:
            citation["evidence_id"] = evidence_id
        records.append(
            {
                "evidence_id": evidence_id,
                "record_type": "source",
                "report_eligible": bool(canonical_url),
                "query_id": query_id,
                "query": query,
                "source_id": source_id,
                "citation_id": str(citation_record.get("citation_id", "") or ""),
                "canonical_url": canonical_url,
                "domain": source_record.get("domain") or domain_from_url(canonical_url),
                "tier": source_record.get("tier", "unknown"),
                "tier_reason": source_record.get("tier_reason", ""),
                "provider": source_record.get("provider") or provider,
                "source_title": citation_record.get("title", ""),
                "source_snippet": citation_record.get("snippet", ""),
                "source_date": citation_record.get("source_date", "") or "unknown",
                "last_updated": citation_record.get("last_updated", "") or "unknown",
                "source_passages": passages,
                "citation_set": citation_set,
                "claims": [
                    _claim_payload(
                        claim,
                        citation_set=_claim_citation_set(
                            claim,
                            [citation_record],
                            evidence_id=evidence_id,
                            allow_overview_without_snippet=True,
                        ),
                    )
                    for claim in claims
                ],
            }
        )
    return records


def merge_evidence_records(
    existing: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append EvidenceRecords by id without duplicating existing rows."""
    by_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in existing + new_records:
        evidence_id = str(record.get("evidence_id", "") or "")
        if evidence_id:
            by_id[evidence_id] = record
    return list(by_id.values())


def derive_claim_ledger_from_evidence(
    evidence_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive the legacy raw claim ledger from EvidenceRecord claims."""
    by_raw_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in evidence_ledger:
        evidence_id = str(record.get("evidence_id", "") or "")
        if not evidence_id:
            continue
        for claim in record.get("claims", []) or []:
            raw_claim_id = str(claim.get("raw_claim_id", "") or "")
            if not raw_claim_id:
                raw_claim_id = make_record_id(
                    "raw_claim",
                    record.get("query_id", ""),
                    evidence_id,
                    claim.get("signature") or claim.get("claim_text", ""),
                )
            entry = by_raw_id.setdefault(
                raw_claim_id,
                {
                    "raw_claim_id": raw_claim_id,
                    "claim_text": claim.get("claim_text", ""),
                    "evidence_snippet": claim.get("evidence_snippet", ""),
                    "claim_type": claim.get("claim_type", "fact"),
                    "polarity": claim.get("polarity", "affirmed"),
                    "needs_primary": bool(claim.get("needs_primary", False)),
                    "source_urls": [],
                    "source_ids": [],
                    "citation_ids": [],
                    "evidence_ids": [],
                    "published_date": claim.get("published_date", "unknown"),
                    "signature": claim.get("signature", ""),
                    "round": int(record.get("round", 0) or 0),
                    "query": record.get("query", ""),
                    "query_id": record.get("query_id", ""),
                },
            )
            entry["round"] = int(claim.get("round", entry.get("round", 0)) or 0)
            claim_citations = claim.get("citation_set", []) or []
            if claim_citations:
                for citation in claim_citations:
                    _append_unique(entry["source_urls"], normalize_url(str(citation.get("url", ""))))
                    _append_unique(entry["source_ids"], str(citation.get("source_id", "")))
                    _append_unique(entry["citation_ids"], str(citation.get("citation_id", "")))
            else:
                _append_unique(entry["source_urls"], str(record.get("canonical_url", "")))
                _append_unique(entry["source_ids"], str(record.get("source_id", "")))
                _append_unique(entry["citation_ids"], str(record.get("citation_id", "")))
            _append_unique(entry["evidence_ids"], evidence_id)
            if claim_citations:
                entry.setdefault("citation_set", [])
                for citation in claim_citations:
                    if citation not in entry["citation_set"]:
                        entry["citation_set"].append(citation)
    return list(by_raw_id.values())


def project_claim_verification_to_evidence(
    evidence_ledger: list[dict[str, Any]],
    consolidated_claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project consolidated claim verification back onto EvidenceRecords."""
    by_raw_id: dict[str, dict[str, Any]] = {}
    for claim in consolidated_claims:
        for raw_claim_id in claim.get("member_claim_ids", []) or []:
            by_raw_id[str(raw_claim_id)] = claim

    for record in evidence_ledger:
        for claim in record.get("claims", []) or []:
            consolidated = by_raw_id.get(str(claim.get("raw_claim_id", "")))
            if not consolidated:
                continue
            claim["claim_id"] = consolidated.get("claim_id", "")
            claim["verification_status"] = consolidated.get("status", "unverified")
            claim["verification_basis"] = (
                consolidated.get("verification_basis")
                or consolidated.get("status_reason", "")
            )
            claim["supporting_evidence_ids"] = consolidated.get("supporting_evidence_ids", [])
            claim["supporting_domain_count"] = consolidated.get("supporting_domain_count", 0)
            claim["contradicting_evidence_ids"] = consolidated.get("contradicting_evidence_ids", [])
    return evidence_ledger


# --------------------------------------------------------------------------- #
# Single canonical evidence view for the final answer composer
# --------------------------------------------------------------------------- #
#
# The EvidenceLedger is the only persisted evidence truth. The final answer
# prompt consumes exactly one derived view: a record-driven Markdown overview
# rendered here. There is no separate report-evidence-bundle, prompt-evidence-
# unit, or rendered-context channel anymore -- verification standing is read
# directly from the projected claim fields on each record.

_VERIFICATION_RANK: dict[str, int] = {
    "cross-checked": 50,
    "primary-source": 42,
    "contested": 30,
    "single-source verified": 24,
    "source-context": 12,
    "unverified": 8,
}


@dataclass(slots=True)
class EvidenceOverview:
    """Rendered single-view evidence overview for the answer composer.

    Attributes:
        markdown: The full source-indexed Markdown block handed to the
            answer prompt. Empty when no report-eligible record exists.
        label_urls: Mapping from visible per-source citation labels
            (``E1`` ..) to their canonical URLs, used to expand and validate
            answer citations without a separate citation list.
        allowed_urls: Ordered unique URLs of visible source blocks -- the
            citation allowlist for the final answer.
        label_by_evidence_id: Stable EvidenceRecord-id to citation-label
            mapping. Multiple EvidenceRecords can intentionally share a
            label when they represent the same canonical URL.
        rendered_record_count: Number of EvidenceRecords that made it
            into ``markdown``.
        omitted_record_count: Report-eligible records dropped only because
            of the character budget. Surfaced so budget loss is visible
            instead of silent.
        rendered_evidence_ids: EvidenceRecord ids whose source block is
            visible in ``markdown``.
    """

    markdown: str
    label_urls: dict[str, str] = field(default_factory=dict)
    allowed_urls: list[str] = field(default_factory=list)
    label_by_evidence_id: dict[str, str] = field(default_factory=dict)
    rendered_record_count: int = 0
    omitted_record_count: int = 0
    rendered_evidence_ids: list[str] = field(default_factory=list)


def _normalized_text_key(text: Any) -> str:
    """Return a whitespace-normalized lowercase key for dedup comparisons."""
    return " ".join(str(text or "").split()).lower()


def _record_primary_url(record: dict[str, Any]) -> str:
    """Return the canonical URL a record is cited by, or its first citation URL."""
    url = normalize_url(str(record.get("canonical_url", "") or ""))
    if url:
        return url
    for citation in record.get("citation_set", []) or []:
        if not isinstance(citation, dict):
            continue
        citation_url = normalize_url(str(citation.get("url", "") or ""))
        if citation_url:
            return citation_url
    return ""


def _evidence_label_number(label: str) -> int | None:
    """Return the numeric part of an ``E#`` label, or ``None``."""
    match = _EVIDENCE_LABEL_RE.fullmatch(str(label or ""))
    if not match:
        return None
    return int(match.group(1))


def _next_evidence_label(used_labels: set[str]) -> str:
    """Return the next unused compact evidence label."""
    index = 1
    while f"E{index}" in used_labels:
        index += 1
    return f"E{index}"


def _assign_url_canonical_labels(
    eligible: list[dict[str, Any]],
    *,
    existing_label_by_evidence_id: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign one report label per canonical URL and project it to records.

    EvidenceRecords remain query/citation-specific ledger rows, but the
    report-facing citation label represents the source URL. Reusing existing
    ``E#`` labels keeps labels stable across repeated render calls while
    collapsing any older per-record duplicates to the URL's first label.
    """
    existing = dict(existing_label_by_evidence_id or {})
    sorted_records = sorted(eligible, key=_evidence_record_score, reverse=True)
    label_by_url: dict[str, str] = {}
    label_by_evidence_id: dict[str, str] = {}
    used_labels: set[str] = set()

    for record in sorted_records:
        evidence_id = str(record.get("evidence_id", "") or "")
        url = _record_primary_url(record)
        label = existing.get(evidence_id)
        if not url or _evidence_label_number(str(label or "")) is None:
            continue
        if url not in label_by_url and str(label) not in used_labels:
            label_by_url[url] = str(label)
            used_labels.add(str(label))
        if url in label_by_url and evidence_id:
            label_by_evidence_id[evidence_id] = label_by_url[url]

    for record in sorted_records:
        evidence_id = str(record.get("evidence_id", "") or "")
        url = _record_primary_url(record)
        if not url:
            continue
        label = label_by_url.get(url)
        if not label:
            label = _next_evidence_label(used_labels)
            label_by_url[url] = label
            used_labels.add(label)
        if evidence_id:
            label_by_evidence_id[evidence_id] = label

    return label_by_evidence_id, label_by_url


def _record_verification_label(record: dict[str, Any]) -> str:
    """Return the strongest verification standing across a record's claims.

    Reads the verification fields projected onto each claim by
    :func:`project_claim_verification_to_evidence`. Claimless records are
    ``source-context``: usable, single-source-attributed background.
    """
    claims = record.get("claims", []) or []
    if not claims:
        return "source-context"
    bases = {str(claim.get("verification_basis", "") or "") for claim in claims}
    statuses = {str(claim.get("verification_status", "") or "") for claim in claims}
    if "verified_cross_checked" in bases:
        return "cross-checked"
    if "contested" in statuses or "contested" in bases:
        return "contested"
    if "verified" in statuses:
        if "verified_primary" in bases:
            return "primary-source"
        return "single-source verified"
    return "unverified"


def _evidence_record_score(record: dict[str, Any]) -> int:
    """Rank evidence records for budget-limited rendering: trust then density."""
    tier = str(record.get("tier", "unknown") or "unknown")
    score = _SOURCE_CONTEXT_TIER_RANK.get(tier, _SOURCE_CONTEXT_TIER_RANK["unknown"])
    score += _VERIFICATION_RANK.get(_record_verification_label(record), 0)
    score += min(30, len(record.get("claims", []) or []) * 5)
    score += min(12, len(record.get("source_passages", []) or []) * 2)
    score += min(4, len(str(record.get("source_snippet", "") or "")) // 120)
    if str(record.get("source_date", "") or "").strip() not in {"", "unknown"}:
        score += 3
    return score


def _render_source_record_block(
    record: dict[str, Any],
    label: str,
    *,
    max_chars: int,
    seen_texts: set[str],
) -> str:
    """Render one EvidenceRecord as a labelled source block.

    ``seen_texts`` is shared across the whole query group so a claim,
    snippet, passage, or summary line is never rendered twice -- this is
    what keeps per-source content substantial instead of repetitive.
    """
    url = _record_primary_url(record)
    title = _bounded_text(record.get("source_title", ""), 200) or url or "Unbenannte Quelle"
    date = str(record.get("source_date", "") or "unknown").strip() or "unknown"
    tier = str(record.get("tier", "unknown") or "unknown")
    verification = _record_verification_label(record)

    # No URL line: the LLM cites the bare [E#] label and the URL is
    # re-attached after synthesis (token saving, fewer citation mistakes).
    header_lines = [f"[{label}] {title}"]
    header_lines.append(
        f"  Datum: {date} | Einstufung: {tier} | Beleglage: {verification}"
    )

    claim_lines: list[str] = []
    evidence_lines: list[str] = []
    for claim in record.get("claims", []) or []:
        claim_text = _full_text(claim.get("claim_text", ""))
        if claim_text and _normalized_text_key(claim_text) not in seen_texts:
            claim_lines.append(claim_text)
            seen_texts.add(_normalized_text_key(claim_text))
        evidence_snippet = _full_text(claim.get("evidence_snippet", ""))
        if evidence_snippet and _normalized_text_key(evidence_snippet) not in seen_texts:
            evidence_lines.append(evidence_snippet)
            seen_texts.add(_normalized_text_key(evidence_snippet))

    source_snippet = _full_text(record.get("source_snippet", ""))
    if source_snippet and _normalized_text_key(source_snippet) not in seen_texts:
        evidence_lines.append(source_snippet)
        seen_texts.add(_normalized_text_key(source_snippet))
    for passage in record.get("source_passages", []) or []:
        passage_text = _full_text(passage.get("text", ""))
        if passage_text and _normalized_text_key(passage_text) not in seen_texts:
            evidence_lines.append(passage_text)
            seen_texts.add(_normalized_text_key(passage_text))

    content_lines: list[str] = []
    if claim_lines:
        content_lines.append("  Aussagen dieser Quelle:")
        content_lines.extend(f"  - {text}" for text in claim_lines)
    if evidence_lines:
        content_lines.append("  Belegausschnitte:")
        content_lines.extend(f"  - {text}" for text in evidence_lines)
    if not content_lines:
        content_lines.append(
            "  Inhalt: dieser Quelle in der Synthese oben zugeordnet "
            "(kein separater Per-Quelle-Auszug verfuegbar)."
        )

    header = "\n".join(header_lines)
    block = "\n".join([header, *content_lines])
    if len(block) <= max_chars:
        return block

    budget = max(0, max_chars - len(header) - 48)
    kept: list[str] = []
    running = 0
    for line in content_lines:
        if running + len(line) + 1 > budget:
            break
        kept.append(line)
        running += len(line) + 1
    kept.append("  [...weitere Belege dieser Quelle wegen Budget gekuerzt]")
    return "\n".join([header, *kept])


def render_evidence_ledger_overview(
    evidence_ledger: list[dict[str, Any]],
    *,
    max_total_chars: int,
    max_record_chars: int,
    label_by_evidence_id: dict[str, str] | None = None,
    query_synthesis: dict[str, dict[str, Any]] | None = None,
) -> EvidenceOverview:
    """Render the EvidenceLedger into the single canonical answer-prompt view.

    Record-driven: every report-eligible EvidenceRecord becomes one
    labelled source block, grouped under the search query that produced
    it so the provider synthesis is shown once instead of repeated per
    source. Verification standing per source is read from the projected
    claim fields, so no bundle or unit list is needed.

    Citation labels (``E1`` ..) are assigned per canonical URL, then
    projected to every EvidenceRecord for that URL. The rendered citation
    allowlist contains only source blocks that are actually visible in the
    prompt. Records that do not fit are counted in
    :attr:`EvidenceOverview.omitted_record_count` rather than dropped
    silently.

    Args:
        evidence_ledger: The run's EvidenceRecords (the primary truth),
            ideally after :func:`project_claim_verification_to_evidence`.
        max_total_chars: Hard budget for the whole rendered view.
        max_record_chars: Per-source-block budget before its evidence
            lines are compacted (the label and metadata are kept).
        label_by_evidence_id: Optional pre-assigned EvidenceRecord label
            map. Existing ``E#`` labels are reused, but collapsed to the
            URL's first label when several EvidenceRecords represent the
            same source.

    Returns:
        An :class:`EvidenceOverview` with the Markdown view, the label/URL
        map for visible source blocks, the visible citation allowlist, the
        stable label map, and rendered/omitted counts.
    """
    total_limit = max(0, int(max_total_chars or 0))
    record_limit = max(400, int(max_record_chars or 400))
    eligible = [
        record
        for record in evidence_ledger
        if record.get("report_eligible") and _record_primary_url(record)
    ]
    if not eligible or total_limit <= 0:
        return EvidenceOverview(markdown="", omitted_record_count=len(eligible))

    label_map, label_by_url = _assign_url_canonical_labels(
        eligible,
        existing_label_by_evidence_id=label_by_evidence_id,
    )
    all_label_urls = {label: url for url, label in label_by_url.items()}

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in eligible:
        group_key = str(record.get("query_id", "") or record.get("query", "") or "")
        groups.setdefault(group_key, []).append(record)

    ranked_groups: list[tuple[int, list[dict[str, Any]]]] = []
    for records in groups.values():
        records_sorted = sorted(records, key=_evidence_record_score, reverse=True)
        group_score = _evidence_record_score(records_sorted[0])
        ranked_groups.append((group_score, records_sorted))
    ranked_groups.sort(key=lambda item: item[0], reverse=True)

    blocks: list[str] = []
    running = 0
    rendered = 0
    omitted = 0
    query_index = 0
    rendered_evidence_ids: list[str] = []
    rendered_labels: set[str] = set()

    def _build_group_block(
        *,
        records: list[dict[str, Any]],
        query_number: int,
        synthesis: dict[str, Any],
        separator: int,
    ) -> tuple[str, list[dict[str, Any]], set[str]] | None:
        """Return the largest visible group block that fits the budget."""
        first = records[0]
        base_header_lines = [f"RECHERCHE-ERGEBNIS R{query_number}"]
        query_text = _bounded_text(first.get("query", ""), 240)
        if query_text:
            base_header_lines.append(f"Suchanfrage: {query_text}")

        for count in range(len(records), 0, -1):
            candidate_records = records[:count]
            visible_url_to_label: dict[str, str] = {}
            for record in candidate_records:
                evidence_id = str(record.get("evidence_id", "") or "")
                url = _record_primary_url(record)
                label = label_map.get(evidence_id)
                if url and label:
                    visible_url_to_label.setdefault(url, label)

            rank_to_url = synthesis.get("citation_urls_by_rank", {}) or {}
            provider_answer = _full_block(
                _remap_synthesis_citations(
                    str(synthesis.get("provider_answer", "") or ""),
                    visible_url_to_label,
                    rank_to_url,
                )
            )
            synthesis_text = provider_answer
            header_lines = list(base_header_lines)
            seen_texts: set[str] = set()
            if synthesis_text:
                header_lines.extend(
                    [
                        "Provider-Synthese "
                        "(Kontext; nicht eigenstaendig verifiziert):",
                        synthesis_text,
                    ]
                )
                seen_texts.add(_normalized_text_key(synthesis_text))

            group_pieces = ["\n".join(header_lines), "Quellen aus dieser Recherche:"]
            labels_in_group: set[str] = set()
            for record in candidate_records:
                evidence_id = str(record.get("evidence_id", "") or "")
                label = label_map.get(evidence_id) or "E?"
                block = _render_source_record_block(
                    record,
                    label,
                    max_chars=record_limit,
                    seen_texts=seen_texts,
                )
                group_pieces.append(block)
                if label != "E?":
                    labels_in_group.add(label)

            block_text = "\n\n".join(group_pieces)
            if running + separator + len(block_text) <= total_limit:
                return block_text, candidate_records, labels_in_group
        return None

    for _, records in ranked_groups:
        query_index += 1
        separator = 2 if blocks else 0
        first = records[0]
        synthesis = (query_synthesis or {}).get(str(first.get("query_id", "") or ""), {})
        candidate = _build_group_block(
            records=records,
            query_number=query_index,
            synthesis=synthesis,
            separator=separator,
        )
        if candidate is None:
            omitted += len(records)
            continue

        block_text, committed_records, labels_in_group = candidate
        blocks.append(block_text)
        running += separator + len(block_text)
        rendered += len(committed_records)
        omitted += len(records) - len(committed_records)
        rendered_labels.update(labels_in_group)
        rendered_evidence_ids.extend(
            str(record.get("evidence_id", "") or "")
            for record in committed_records
            if str(record.get("evidence_id", "") or "")
        )

    if omitted > 0 and blocks:
        blocks.append(
            f"HINWEIS: {omitted} weitere belegfaehige Quellen passten nicht in das "
            "Evidenz-Budget und sind in dieser Uebersicht nicht enthalten."
        )

    visible_label_urls = {
        label: all_label_urls[label]
        for label in sorted(
            rendered_labels,
            key=lambda value: _evidence_label_number(value) or 10**9,
        )
        if label in all_label_urls
    }
    visible_allowed_urls = list(dict.fromkeys(visible_label_urls.values()))

    return EvidenceOverview(
        markdown="\n\n".join(blocks),
        label_urls=visible_label_urls,
        allowed_urls=visible_allowed_urls,
        label_by_evidence_id=label_map,
        rendered_record_count=rendered,
        omitted_record_count=omitted,
        rendered_evidence_ids=rendered_evidence_ids,
    )


def select_section_evidence_records(
    evidence_ledger: list[dict[str, Any]],
    *,
    heading: str,
    required_aspects: list[str],
    used_labels: set[str],
    label_by_evidence_id: dict[str, str],
    max_records: int,
) -> list[dict[str, Any]]:
    """Select a small deterministic record subset relevant to one answer section.

    Uses only general heading-keyword heuristics plus the record's own
    verification standing, content density, and aspect-token overlap --
    no provider- or query-type-specific branches. Records whose citation
    label was already used by an earlier section are de-prioritized so
    source coverage spreads across the whole report instead of clustering.
    """
    eligible = [
        record
        for record in evidence_ledger
        if record.get("report_eligible") and _record_primary_url(record)
    ]
    if not eligible or max_records <= 0:
        return []

    heading_l = heading.lower()
    aspect_tokens = {
        token
        for aspect in required_aspects
        for token in re.findall(r"\w+", str(aspect).lower())
        if len(token) >= 4
    }

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, record in enumerate(eligible):
        verification = _record_verification_label(record)
        score = _VERIFICATION_RANK.get(verification, 10)
        score += _SOURCE_CONTEXT_TIER_RANK.get(
            str(record.get("tier", "unknown") or "unknown"),
            _SOURCE_CONTEXT_TIER_RANK["unknown"],
        ) // 10

        if any(marker in heading_l for marker in ("risik", "unsicher", "offen")):
            if verification in {"contested", "unverified"}:
                score += 40
            elif verification == "source-context":
                score += 12
        elif any(
            marker in heading_l
            for marker in ("analyse", "detail", "hintergrund", "kontext")
        ):
            score += min(12, 2 * len(record.get("source_passages", []) or []))
        elif any(
            marker in heading_l
            for marker in ("summary", "kurzfazit", "fazit", "ausblick")
        ):
            if verification in {
                "cross-checked",
                "primary-source",
                "single-source verified",
                "contested",
            }:
                score += 16

        searchable = " ".join(
            [
                str(record.get("source_title", "") or ""),
                " ".join(
                    str(claim.get("claim_text", "") or "")
                    for claim in record.get("claims", []) or []
                ),
            ]
        ).lower()
        if aspect_tokens and any(token in searchable for token in aspect_tokens):
            score += 8

        label = label_by_evidence_id.get(str(record.get("evidence_id", "") or ""))
        if label and label in used_labels:
            score -= 16

        scored.append((score, -index, record))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored[:max_records]]



def audit_answer_evidence_bindings(
    answer_body: str,
    evidence_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit the answer's cited sources against the EvidenceLedger.

    For every Markdown link in the answer body, resolve the URL to an
    EvidenceRecord and record that source's verification standing. There
    is no separate bundle list: a citation is ``matched`` when it resolves
    to a record carrying a verified or contested claim, ``source_context``
    when it resolves to a claimless or unverified record, and
    ``unknown_citation`` when it does not resolve to any record at all.
    """
    record_by_url: dict[str, dict[str, Any]] = {}
    for record in evidence_ledger:
        primary = _record_primary_url(record)
        if primary and primary not in record_by_url:
            record_by_url[primary] = record
        for citation in record.get("citation_set", []) or []:
            if not isinstance(citation, dict):
                continue
            url = normalize_url(str(citation.get("url", "") or ""))
            if url and url not in record_by_url:
                record_by_url[url] = record

    cited_urls: list[str] = []
    for match in _MARKDOWN_LINK_RE.finditer(answer_body or ""):
        url = normalize_url(match.group(2))
        if url and url not in cited_urls:
            cited_urls.append(url)

    bindings: list[dict[str, Any]] = []
    for url in cited_urls:
        record = record_by_url.get(url)
        if record is None:
            bindings.append(
                {
                    "binding_id": make_record_id("bind_ev", url, "unknown_citation"),
                    "citation_url": url,
                    "evidence_id": "",
                    "verification": "unknown",
                    "binding_status": "unknown_citation",
                }
            )
            continue
        verification = _record_verification_label(record)
        status = (
            "matched"
            if verification
            in {"cross-checked", "primary-source", "single-source verified", "contested"}
            else "source_context"
        )
        bindings.append(
            {
                "binding_id": make_record_id("bind_ev", url, status),
                "citation_url": url,
                "evidence_id": str(record.get("evidence_id", "") or ""),
                "verification": verification,
                "binding_status": status,
            }
        )
    return bindings
