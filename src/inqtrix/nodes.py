"""State machine node functions for the research agent.

Each node takes the state dict plus providers and strategies as keyword
arguments.  The five nodes correspond to the five phases of the research
loop: classify, plan, search, evaluate, answer.

Extracted from ``_original_agent.py`` and adapted to use the provider /
strategy / settings abstractions defined in the ``inqtrix`` package.
"""

from dataclasses import dataclass
import datetime as dt
import inspect
import json
import logging
import re
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ContextManager, NamedTuple

from openai import OpenAIError

from inqtrix.constants import (
    CONTEXT_WINDOW_SAFETY_TOKENS,
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
)
from inqtrix.domains import LANG_NAMES
from inqtrix.evidence import (
    assemble_evidence_records,
    audit_answer_evidence_bindings,
    derive_claim_ledger_from_evidence,
    evidence_id_for_citation,
    merge_evidence_records,
    project_claim_verification_to_evidence,
    render_evidence_ledger_overview,
    select_section_evidence_records,
)
from inqtrix.search_result import GroundedSearchResult
from inqtrix.exceptions import (
    AgentModelCapacityError,
    AgentRateLimited,
    AgentTimeout,
    AnthropicAPIError,
    AzureOpenAIAPIError,
    BedrockAPIError,
)
from inqtrix.i18n import MESSAGES, t
from inqtrix.json_helpers import (
    parse_json_string_list,
    parse_json_string_list_with_status,
)
from inqtrix.prompts import (
    EVALUATE_FORMAT_SUFFIX,
    build_answer_section_system_prompt,
    build_answer_section_user_prompt,
)
from inqtrix.providers.base import (
    LLMResponse,
    ProviderContext,
    _check_deadline,
    get_search_provider_capabilities,
    is_model_capacity_error,
)
from inqtrix.report_profiles import AnswerSectionSpec, ReportProfile
from inqtrix.runtime_logging import (
    format_log_excerpt,
    forensic_enabled,
    make_record_id,
    normalize_source_provenance,
)
from inqtrix.model_routing import describe_resolution, describe_unresolved_resolution
from inqtrix.scoring import append_score_snapshot
from inqtrix.settings import AgentSettings
from inqtrix.state import (
    append_iteration_log,
    check_cancel_event,
    emit_progress,
    emit_run_event,
    track_tokens,
)
from inqtrix.strategies import ClaimExtractionStrategy, ProviderCitationRef, StrategyContext
from inqtrix.text import NEGATION_TOKENS, STOPWORDS, is_none_value, tokenize
from inqtrix.urls import (
    count_allowed_links,
    domain_from_url,
    extract_urls,
    normalize_url,
    sanitize_answer_links,
    today,
)

log = logging.getLogger("inqtrix")

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_ALLOWED_FINISH_REASONS = {"", "stop", "end_turn"}
_SOURCE_TIER_SORT_ORDER = {
    "primary": 0,
    "mainstream": 1,
    "stakeholder": 2,
    "unknown": 3,
    "low": 4,
}
_ADDITIONAL_LINK_LIMIT = 10
_LATER_ROUND_QUERY_MIN = 6
_EVIDENCE_DEPTH_MIN_VERIFIED_BUNDLES = 3

# Single canonical confidence ceiling per evidence-contract status. The answer
# node computes the contract once, stores it on state, and clamps confidence via
# this table -- no per-status if/else scattered across the node. Ordering
# encodes evidence strength: algorithm_failed (synthesis unusable) < source-
# context-only (no claim-grounded sentence) < needs_review (a cited URL has no
# backing record) < clean (uncapped). Statuses absent here ("clean", "unknown")
# impose no cap.
_CONTRACT_CONFIDENCE_CAP: dict[str, int] = {
    "algorithm_failed": 3,
    "source_context_only": 4,
    "needs_review": 6,
}


class AnswerAppendixSections(NamedTuple):
    sections: list[str]
    references: list[dict[str, str]]
    additional_links: list[str]

# Crosscheck-Planner bonus applied to consolidated claims whose
# ``verification_basis == "verified_quality_source"``.
#
# Rationale: that basis is set in ``DefaultClaimConsolidator.consolidate()``
# (`strategies/_claim_consolidation.py`) for claims that are supported by
# *exactly one* citation from a primary- or mainstream-tier domain
# (Reuters / Tagesschau / SEC / ECB / ...). The verification status stays
# ``verified`` -- a single high-tier source is legitimate evidence -- but
# we want the *next* research round to actively try to corroborate it
# with an independent second source. This bonus, stacked on top of the
# generic +5 for any verified claim with ``citation_count < 2`` further
# down in :func:`_select_crosscheck_targets`, gives single-quality-source
# claims a total ranking score of +9, putting them at the front of the
# crosscheck-target queue.
#
# Coupled to: ``verified_quality_source`` branch in ``consolidate()``.
# Changing one without the other will misalign the crosscheck planner
# with the verification rule. See also gotchas.md #47.
_CROSSCHECK_BONUS_QUALITY_SOURCE = 4
_CENTRAL_CLAIM_RE = re.compile(
    r"\b("
    r"benchmark|swe-bench|gpqa|terminal-bench|score|percent|prozent|%"
    r"|jobs?|layoffs?|employment|arbeitsmarkt|regulation|regulierung"
    r"|ai act|kurs|shares?|stock|market|markt|revenue|umsatz"
    r"|mrd|mio|billion|million|euro|usd|\$"
    r")\b",
    re.IGNORECASE,
)
_STORM_SLOT_INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    ("technical", "technical mechanisms, model details, or direct primary technical sources"),
    ("practical", "real-world deployments, user impact, or adoption evidence"),
    ("critical", "limitations, risks, failures, or credible counterarguments"),
    ("comparative", "comparisons with alternatives, competitors, or prior baselines"),
    ("historical", "timeline, prior context, and what changed recently"),
    ("latest", "latest developments, announcements, and dated updates"),
    ("data", "primary statistics, filings, benchmarks, or raw data"),
    ("stakeholder", "affected stakeholders, practitioners, institutions, or markets"),
    ("regulatory", "regulatory, legal, policy, or compliance evidence"),
    ("economic", "market, cost, funding, labor, or business implications"),
)


class SearchOutcome(NamedTuple):
    """One search call's typed result plus an optional non-fatal degrade notice.

    The notice is load-bearing: the search node counts it to report
    "X von Y Suchanfragen fehlgeschlagen" and to log per-query details, so a
    degraded search stays visible (Designprinzip 1) instead of looking like an
    ordinary empty result.
    """

    result: GroundedSearchResult
    notice: str | None


def _domain_filter_for_query_text(
    query: str,
    *,
    base_domain_filter: list[str] | None = None,
) -> list[str] | None:
    """Return provider domain filters implied by one query string."""
    ql = (query or "").lower()
    domains: list[str] = []
    for match in re.finditer(r"(?:^|\s|\()site:([^\s)]+)", ql):
        dom = match.group(1).strip()
        dom = dom.replace("https://", "").replace("http://", "")
        dom = dom.split("/")[0].strip()
        dom = dom.strip(" ,.;:()[]{}<>\\\"'")
        if dom and dom not in domains:
            domains.append(dom)
    if domains:
        return domains
    return list(base_domain_filter or [])


def _target_query_count_for_round(round_index: int, settings: AgentSettings) -> int:
    """Return the planned and executed query width for one search round."""
    first_round = max(1, int(getattr(settings, "first_round_queries", 1) or 1))
    if int(round_index or 0) <= 0:
        return first_round
    return max(_LATER_ROUND_QUERY_MIN, first_round - 2)


def _claim_extraction_text(
    answer_text: str,
    citation_records: list[dict[str, Any]],
    *,
    max_sources: int = 8,
) -> str:
    """Append provider-neutral source provenance to claim extraction text."""
    source_lines: list[str] = []
    seen_urls: set[str] = set()
    for record in citation_records:
        url = normalize_url(str(record.get("canonical_url", "") or record.get("url", "") or ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = " ".join(str(record.get("title", "") or "").split())[:140]
        date = " ".join(str(record.get("source_date", "") or "").split())[:80]
        # Prefix with the provider rank/id so the model can resolve the
        # answer's inline [n] / [web:n] citations to the matching source.
        rank = record.get("rank")
        prefix = f"[{rank}] {url}" if rank else f"- {url}"
        if title:
            prefix += f" | title: {title}"
        if date:
            prefix += f" | date: {date}"
        source_lines.append(prefix)
        if len(seen_urls) >= max_sources:
            break
    if not source_lines:
        return answer_text
    return (
        f"{answer_text}\n\n"
        "Quellenprovenienz aus normalisierten Suchergebnissen:\n"
        + "\n".join(source_lines)
    )


def _claim_provider_refs(citation_records: list[dict[str, Any]]) -> list[ProviderCitationRef]:
    """Return provider-local citation refs for deterministic claim binding."""
    refs: list[ProviderCitationRef] = []
    seen: set[str] = set()
    for record in citation_records:
        rank = str(record.get("rank", "") or "").strip()
        url = normalize_url(str(record.get("canonical_url", "") or record.get("url", "") or ""))
        if not rank or not url:
            continue
        key = f"{rank}|{url}"
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            ProviderCitationRef(
                ref=rank,
                url=url,
                title=str(record.get("title", "") or ""),
            )
        )
    return refs


def _claim_text_needs_targeted_verification(claim_text: str) -> bool:
    """Return whether a claim contains central evidence-depth risk signals."""
    text = claim_text or ""
    return bool(re.search(r"\d", text) or _CENTRAL_CLAIM_RE.search(text))


def _claim_citation_count(claim: dict[str, Any]) -> int:
    """Count distinct citation URLs backing one consolidated claim."""
    urls: list[str] = []
    for citation in claim.get("citation_set", []) or []:
        url = citation.get("url", "") if isinstance(citation, dict) else citation
        normalized = normalize_url(str(url))
        if normalized and normalized not in urls:
            urls.append(normalized)
    for url in claim.get("source_urls", []) or []:
        normalized = normalize_url(str(url))
        if normalized and normalized not in urls:
            urls.append(normalized)
    return len(urls)


def _evidence_depth_gap(s: dict[str, Any]) -> dict[str, Any]:
    """Diagnose whether report evidence is too shallow for early stopping.

    Reads the consolidated claim ledger directly -- verification standing,
    cross-check basis, and per-claim citation counts all live there, so no
    separate report-evidence-bundle list is needed.
    """
    consolidated = list(s.get("consolidated_claims", []) or [])
    verified_claims = [
        claim for claim in consolidated
        if claim.get("status") == "verified"
    ]
    verified_count = len(verified_claims)
    cross_checked_count = sum(
        1 for claim in verified_claims
        if claim.get("verification_basis") == "verified_cross_checked"
    )
    single_source_count = sum(
        1 for claim in verified_claims
        if _claim_citation_count(claim) < 2
    )
    quality_single_count = sum(
        1 for claim in verified_claims
        if claim.get("verification_basis") == "verified_quality_source"
        and _claim_citation_count(claim) < 2
    )
    central_quality_single_count = sum(
        1 for claim in verified_claims
        if claim.get("verification_basis") == "verified_quality_source"
        and _claim_citation_count(claim) < 2
        and _claim_text_needs_targeted_verification(str(claim.get("claim_text", "") or ""))
    )
    single_source_ratio = (
        single_source_count / verified_count
        if verified_count > 0
        else 0.0
    )

    reasons: list[str] = []
    if (
        verified_count >= _EVIDENCE_DEPTH_MIN_VERIFIED_BUNDLES
        and cross_checked_count == 0
    ):
        reasons.append("no_cross_checked_claims")
    if (
        verified_count >= _EVIDENCE_DEPTH_MIN_VERIFIED_BUNDLES
        and single_source_ratio > 0.5
    ):
        reasons.append("majority_single_source_claims")
    if central_quality_single_count > 0:
        reasons.append("central_claim_single_quality_source")

    active = bool(reasons)
    gap = ""
    if active:
        gap = (
            "Report-Evidenz ist noch zu stark single-source; zentrale Aussagen "
            "brauchen unabhaengige Cross-Checks oder Primaerquellen."
        )
    return {
        "active": active,
        "reason": ",".join(reasons),
        "gap": gap,
        "verified_count": verified_count,
        "cross_checked_count": cross_checked_count,
        "single_source_verified_count": single_source_count,
        "single_source_ratio": round(single_source_ratio, 3),
        "verified_quality_source_single_count": quality_single_count,
        "central_single_quality_source_count": central_quality_single_count,
    }


def _select_crosscheck_targets(
    consolidated_claims: list[dict[str, Any]],
    *,
    max_targets: int = 3,
) -> list[dict[str, Any]]:
    """Select a small set of claims that deserve independent verification.

    Reads only the consolidated claim ledger -- there is no separate
    report-evidence-bundle list anymore. A claim's own citation set and
    support counts already carry everything needed to decide whether it
    is still single-sourced and deserves a cross-check query.
    """
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, claim in enumerate(consolidated_claims):
        claim_text = " ".join(str(claim.get("claim_text", "") or "").split())
        if not claim_text:
            continue
        status = str(claim.get("status", "unverified") or "unverified")
        basis = str(claim.get("verification_basis", "") or "")
        needs_primary = bool(claim.get("needs_primary", False))
        support_count = int(claim.get("support_count", 0) or 0)
        independent_count = int(claim.get("independent_support_count", 0) or 0)
        citation_count = len(claim.get("citation_set", []) or [])
        score = 0
        if (
            status == "verified"
            and basis != "verified_cross_checked"
            and citation_count < 2
        ):
            score += 5
        if basis == "verified_quality_source":
            score += _CROSSCHECK_BONUS_QUALITY_SOURCE
        if status == "unverified":
            score += 3
        if needs_primary:
            score += 2
        if support_count < 2 or independent_count < 2:
            score += 2
        if basis in {"missing_primary_source", "weak_evidence", "news_briefing_out_of_window"}:
            score += 2
        if _claim_text_needs_targeted_verification(claim_text):
            score += 2
        if score <= 0:
            continue

        domains: list[str] = []
        for url in claim.get("source_urls", []) or []:
            domain = domain_from_url(str(url))
            if domain and domain not in domains:
                domains.append(domain)
        candidates.append((
            score,
            index,
            {
                "claim_id": str(claim.get("claim_id", "") or claim.get("signature", "") or ""),
                "claim_text": claim_text[:260],
                "status": status,
                "verification_basis": basis,
                "needs_primary": needs_primary,
                "support_count": support_count,
                "independent_support_count": independent_count,
                "citation_count": citation_count,
                "source_domains": domains[:4],
            },
        ))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [target for _, _, target in candidates[: max(0, int(max_targets or 0))]]


def _append_query_slot(
    slots: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    slot_type: str,
    instruction: str,
    **extra: Any,
) -> None:
    """Append a de-duplicated research slot."""
    normalized_instruction = " ".join(instruction.split())
    if not normalized_instruction:
        return
    key = (slot_type, normalized_instruction.lower()[:220])
    if key in seen:
        return
    seen.add(key)
    slot = {"slot_type": slot_type, "instruction": normalized_instruction}
    for name, value in extra.items():
        if value not in (None, "", [], {}):
            slot[name] = value
    slots.append(slot)


def _build_query_slots(
    s: dict[str, Any],
    *,
    target_count: int,
    crosscheck_targets: list[dict[str, Any]],
    evidence_depth_gap: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one concrete research slot for each planned search query."""
    target_count = max(1, int(target_count or 1))
    slots: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    question = str(s.get("question", "") or "").strip()
    round_index = int(s.get("round", 0) or 0)

    for aspect in (s.get("uncovered_aspects", []) or [])[:target_count]:
        _append_query_slot(
            slots,
            seen,
            "gap",
            f"Find direct evidence for the still-uncovered aspect '{aspect}' in relation to: {question}",
            aspect=str(aspect),
        )
        if len(slots) >= target_count:
            return slots[:target_count]

    gap_text = " ".join(str(s.get("gaps", "") or "").split())
    if gap_text and not is_none_value(gap_text):
        _append_query_slot(
            slots,
            seen,
            "gap",
            f"Resolve this explicit evidence gap: {gap_text}. Main question: {question}",
            gap=gap_text[:240],
        )

    max_crosscheck_slots = max(1, target_count // 2) if round_index > 0 else 0
    for target in crosscheck_targets[:max_crosscheck_slots]:
        claim_text = str(target.get("claim_text", "") or "")
        domains = list(target.get("source_domains", []) or [])
        _append_query_slot(
            slots,
            seen,
            "crosscheck",
            "Find an independent source that confirms or refutes this claim "
            f"without relying on the existing domains: {claim_text}",
            target_claim=claim_text,
            avoid_domains=domains,
            claim_id=str(target.get("claim_id", "") or ""),
            verification_basis=str(target.get("verification_basis", "") or ""),
        )
        if len(slots) >= target_count:
            return slots[:target_count]

    if evidence_depth_gap.get("active") and len(slots) < target_count:
        target_claim = (
            str(crosscheck_targets[0].get("claim_text", "") or "")
            if crosscheck_targets
            else question
        )
        _append_query_slot(
            slots,
            seen,
            "primary_source",
            "Find the original primary source or official data behind this central claim "
            f"instead of another summary article: {target_claim}",
            evidence_depth_reason=evidence_depth_gap.get("reason", ""),
        )

    needs_data_slot = (
        _claim_text_needs_targeted_verification(question)
        or any(
            _claim_text_needs_targeted_verification(str(target.get("claim_text", "") or ""))
            for target in crosscheck_targets
        )
    )
    if needs_data_slot and len(slots) < target_count:
        _append_query_slot(
            slots,
            seen,
            "data_verification",
            f"Which primary data, benchmark, filing, or official statistics verify the key numbers for: {question}",
        )

    if round_index > 0 and len(slots) < target_count:
        _append_query_slot(
            slots,
            seen,
            "counterevidence",
            f"What credible evidence challenges or qualifies the strongest emerging answer to: {question}",
        )

    for aspect in (s.get("required_aspects", []) or []):
        if len(slots) >= target_count:
            break
        _append_query_slot(
            slots,
            seen,
            "storm_perspective",
            f"Research the required aspect '{aspect}' with a distinct source path for: {question}",
            aspect=str(aspect),
        )

    perspective_index = 0
    while len(slots) < target_count:
        label, instruction = _STORM_SLOT_INSTRUCTIONS[
            (round_index + perspective_index) % len(_STORM_SLOT_INSTRUCTIONS)
        ]
        _append_query_slot(
            slots,
            seen,
            "storm_perspective",
            f"Use the {label} perspective to find {instruction} for: {question}",
            perspective=label,
        )
        perspective_index += 1
        if perspective_index > len(_STORM_SLOT_INSTRUCTIONS) + target_count:
            break

    return slots[:target_count]


def _fallback_query_for_slot(
    slot: dict[str, Any],
    *,
    question: str,
    search_language: str,
) -> str:
    """Create a complete fallback question for one slot if parsing underfills."""
    slot_type = str(slot.get("slot_type", "") or "")
    claim = str(slot.get("target_claim", "") or "")
    gap = str(slot.get("gap", "") or "")
    aspect = str(slot.get("aspect", "") or "")
    english = (search_language or "").lower().startswith("en")
    if english:
        if slot_type == "crosscheck" and claim:
            return f"Which independent sources confirm or refute the claim that {claim}?"
        if slot_type == "primary_source" and (claim or question):
            return f"What is the original primary source for {claim or question}?"
        if slot_type == "gap" and (gap or aspect):
            return f"Which sources directly answer the open evidence gap {gap or aspect}?"
        if slot_type == "counterevidence":
            return f"What credible evidence challenges the main claims about {question}?"
        if slot_type == "data_verification":
            return f"Which primary data sources verify the key numbers about {question}?"
        return f"What does the {slot.get('perspective', 'specific')} perspective show about {question}?"
    if slot_type == "crosscheck" and claim:
        return f"Welche unabhaengigen Quellen bestaetigen oder widerlegen die Aussage, dass {claim}?"
    if slot_type == "primary_source" and (claim or question):
        return f"Was ist die Primaerquelle fuer {claim or question}?"
    if slot_type == "gap" and (gap or aspect):
        return f"Welche Quellen beantworten die offene Evidenzluecke {gap or aspect} direkt?"
    if slot_type == "counterevidence":
        return f"Welche glaubwuerdige Evidenz widerspricht den wichtigsten Aussagen zu {question}?"
    if slot_type == "data_verification":
        return f"Welche Primaerdaten verifizieren die zentralen Zahlen zu {question}?"
    return f"Was zeigt die Perspektive {slot.get('perspective', 'spezifisch')} zu {question}?"


def _append_forensic_event(
    s: dict[str, Any],
    settings: AgentSettings,
    *,
    event: str,
    node: str,
    payload: dict[str, Any],
) -> None:
    """Append one forensic event through the canonical iteration-log path."""
    if not forensic_enabled(settings):
        return
    append_iteration_log(
        s,
        {
            "event": event,
            "node": node,
            "timestamp": time.time(),
            **payload,
        },
        testing_mode=settings.testing_mode,
    )


def _emit_node_model_resolution_warning(
    s: dict[str, Any],
    settings: AgentSettings,
    *,
    node: str,
    desc: dict[str, str],
    reason: str,
    message: str,
) -> None:
    """Surface a model-resolution fallback on every visible diagnostics channel."""
    log.warning(message)
    emit_progress(s, f"Warnung: {message}", severity="warning")
    append_iteration_log(
        s,
        {
            "event": "node_model_resolution_warning",
            "node": node,
            "timestamp": time.time(),
            "reason": reason,
            **desc,
        },
        testing_mode=settings.testing_mode,
    )


def _remember_node_model_resolution(
    s: dict[str, Any],
    node: str,
    desc: dict[str, str],
) -> None:
    """Store the exact node routing used by this run for result surfaces."""
    resolutions = dict(s.get("node_model_resolutions") or {})
    resolutions[node] = dict(desc)
    s["node_model_resolutions"] = resolutions


def _resolve_node_llm(
    s: dict[str, Any],
    settings: AgentSettings,
    providers: ProviderContext,
    node: str,
) -> tuple[str, str]:
    """Resolve ``(model, reasoning_effort)`` for a call site and surface the choice.

    Single entry point used by every node so model and reasoning selection is
    uniform across the whole graph. Routes through
    :func:`inqtrix.model_routing.describe_resolution` using the per-run
    ``settings.model_tier`` selection, then emits the resolution on the live
    and forensic channels so the model/effort/tier actually used for each node
    is visible (Designprinzip 1/5):

    * ``inqtrix.node.model_resolution`` as a native run-event -> the React live
      view (whenever a run-event sink is wired).
    * ``node_model_resolution`` as a forensic iteration-log event -> the audit
      log (profile-gated).

    Both carry ``model_source``/``effort_source``, so a ``reasoning_model``
    default that grips instead of a tier is explicit, not silent. When the
    provider exposes no ``models`` attribute, or resolution produces an empty
    model, the fallback is also surfaced via warning log, progress event, and a
    ``node_model_resolution_warning`` iteration-log marker.

    Args:
        s: Mutable agent state (receives both events).
        settings: The active :class:`~inqtrix.settings.AgentSettings`; its
            ``model_tier`` field carries an optional per-run tier override.
        providers: The provider context; ``providers.llm.models`` is the
            resolution source.
        node: Call-site name (see
            :data:`inqtrix.model_routing.NODE_TIER_ASSIGNMENT`).

    Returns:
        A ``(model, reasoning_effort)`` tuple. ``reasoning_effort`` is ``""``
        when the tier inherits the provider default. Both are empty strings
        when the provider exposes no ``models`` attribute.
    """
    models = getattr(providers.llm, "models", None)
    requested_tier = (getattr(settings, "model_tier", "") or "").strip() or None
    if models is None:
        desc = describe_unresolved_resolution(node, requested_tier)
        _remember_node_model_resolution(s, node, desc)
        emit_run_event(s, "inqtrix.node.model_resolution", dict(desc))
        _append_forensic_event(
            s,
            settings,
            event="node_model_resolution",
            node=node,
            payload=desc,
        )
        _emit_node_model_resolution_warning(
            s,
            settings,
            node=node,
            desc=desc,
            reason="provider_models_missing",
            message=(
                f"Node '{node}' could not resolve a model: provider "
                f"{type(providers.llm).__name__} exposes no 'models' attribute; "
                "the provider's own default will apply unseen."
            ),
        )
        return "", ""
    desc = describe_resolution(node, models, requested_tier)
    _remember_node_model_resolution(s, node, desc)
    emit_run_event(s, "inqtrix.node.model_resolution", dict(desc))
    _append_forensic_event(
        s,
        settings,
        event="node_model_resolution",
        node=node,
        payload=desc,
    )
    if not desc["model"]:
        _emit_node_model_resolution_warning(
            s,
            settings,
            node=node,
            desc=desc,
            reason="empty_resolved_model",
            message=(
                f"Node '{node}' resolved an empty model "
                f"(source={desc['model_source']}): the provider's own default "
                "will apply unseen."
            ),
        )
    return desc["model"], desc["effort"]


def _resolve_answer_fallback_model(
    settings: AgentSettings,
    providers: ProviderContext,
    primary_answer_model: str,
) -> str | None:
    """Resolve the optional answer fallback model through the central router."""
    models = getattr(providers.llm, "models", None)
    if models is None:
        return None
    requested_tier = (getattr(settings, "model_tier", "") or "").strip() or None
    desc = describe_resolution("evaluate", models, requested_tier)
    fallback_model = desc["model"]
    if not fallback_model or fallback_model == primary_answer_model:
        return None
    return fallback_model


def _claim_extract_accepts_routing(strategy: ClaimExtractionStrategy) -> bool:
    """Return whether *strategy*'s ``extract`` accepts the routing kwargs.

    The search node resolves a model + reasoning effort for ``claim_extract``
    and forwards them to ``extract`` as ``model`` / ``reasoning_effort``. Those
    kwargs were added to the strategy contract after the fact, so a custom
    strategy written against the older signature would raise ``TypeError`` if
    they were passed unconditionally. This guard keeps the Baukasten contract
    backward compatible: the kwargs are a best-effort hint and are only passed
    to extractors whose signature actually accepts them (an explicit ``model``
    parameter, or ``**kwargs``). A pre-existing strategy on the old signature
    keeps working and simply uses its own model.
    """
    try:
        params = inspect.signature(strategy.extract).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "model" in params and "reasoning_effort" in params


def _strict_algorithm_failure_mode(settings: AgentSettings) -> bool:
    """Return whether core algorithm failures must block normal reports."""
    return (
        settings.report_profile is ReportProfile.DEEP
        or str(getattr(settings, "observability_profile", "summary")).lower() == "forensic"
    )


def _record_algorithm_failure(
    s: dict[str, Any],
    settings: AgentSettings,
    *,
    node: str,
    phase: str,
    reason: str,
    message: str,
    blocking: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a visible core-path failure without treating it as a repair path."""
    entry: dict[str, Any] = {
        "phase": phase,
        "reason": reason,
        "message": message,
        "blocking": bool(blocking),
        "round": int(s.get("round", 0) or 0),
        "run_id": str(s.get("_run_id", "") or ""),
    }
    if details:
        entry.update(details)
    failures = s.setdefault("algorithm_failures", [])
    if isinstance(failures, list):
        failures.append(entry)
    log.warning("ALGO-FAIL %s (%s): %s", phase, reason, message)
    _append_forensic_event(
        s,
        settings,
        event="algorithm_failure",
        node=node,
        payload=entry,
    )
    return entry


def _blocking_algorithm_failures(s: dict[str, Any]) -> list[dict[str, Any]]:
    """Return algorithm failures that must prevent normal report synthesis."""
    return [
        failure
        for failure in s.get("algorithm_failures", []) or []
        if isinstance(failure, dict) and bool(failure.get("blocking"))
    ]


def _llm_context_window_tokens(llm: Any) -> int | None:
    """Return a positive configured context window from an LLM provider."""
    value = getattr(llm, "context_window_tokens", None)
    if callable(value):
        value = value()
    if isinstance(value, int) and value > 0:
        return value
    raw = getattr(llm, "_context_window_tokens", None)
    return raw if isinstance(raw, int) and raw > 0 else None


def _llm_default_output_tokens(llm: Any) -> int:
    """Return the provider's configured reasoning output budget."""
    raw = getattr(llm, "_default_max_tokens", None)
    return raw if isinstance(raw, int) and raw > 0 else DEFAULT_LLM_MAX_OUTPUT_TOKENS


def _estimate_required_context_tokens(
    *,
    system: str | None,
    prompt: str,
    requested_output_tokens: int,
) -> tuple[int, int]:
    """Estimate input and total token capacity for one LLM call."""
    char_count = len(system or "") + len(prompt or "")
    estimated_input_tokens = max(1, (char_count + 3) // 4)
    estimated_required_tokens = (
        estimated_input_tokens
        + max(0, int(requested_output_tokens or 0))
        + CONTEXT_WINDOW_SAFETY_TOKENS
    )
    return estimated_input_tokens, estimated_required_tokens


def _check_llm_context_capacity(
    s: dict[str, Any],
    settings: AgentSettings,
    *,
    providers: ProviderContext,
    node: str,
    phase: str,
    model: str,
    system: str | None,
    prompt: str,
    requested_output_tokens: int,
) -> dict[str, int | str | bool | None]:
    """Warn or fail before sending prompts that exceed configured capacity."""
    estimated_input_tokens, estimated_required_tokens = _estimate_required_context_tokens(
        system=system,
        prompt=prompt,
        requested_output_tokens=requested_output_tokens,
    )
    required_floor = int(getattr(settings, "required_context_window_tokens", 0) or 0)
    required_tokens = max(required_floor, estimated_required_tokens)
    context_window_tokens = _llm_context_window_tokens(providers.llm)
    details: dict[str, int | str | bool | None] = {
        "phase": phase,
        "model": model,
        "context_window_tokens": context_window_tokens,
        "required_context_window_tokens": required_floor,
        "estimated_input_tokens": estimated_input_tokens,
        "requested_output_tokens": int(requested_output_tokens or 0),
        "context_window_safety_tokens": CONTEXT_WINDOW_SAFETY_TOKENS,
        "estimated_required_context_tokens": estimated_required_tokens,
    }

    if context_window_tokens is None:
        warning_key = f"_context_window_unknown_warned_{model or 'default'}"
        if not s.get(warning_key):
            s[warning_key] = True
            log.warning(
                "CAPACITY-WARN context_window unknown (model=%s, phase=%s, "
                "estimated_required_context_tokens=%d, required_context_window_tokens=%d)",
                model or "-",
                phase,
                estimated_required_tokens,
                required_floor,
            )
            emit_progress(
                s,
                t(
                    s,
                    "context_window_unknown_warning",
                    model=model or "-",
                    required=required_floor,
                    estimated=estimated_required_tokens,
                ),
            )
            _append_forensic_event(
                s,
                settings,
                event="model_capacity_warning",
                node=node,
                payload={**details, "reason": "context_window_unknown"},
            )
        return details

    if context_window_tokens < required_tokens:
        reason = (
            "below_required_floor"
            if context_window_tokens < required_floor
            else "estimated_requirement_exceeds_window"
        )
        message = (
            f"Model context window {context_window_tokens} tokens is below "
            f"required {required_tokens} tokens for phase {phase}."
        )
        emit_progress(
            s,
            t(
                s,
                "context_window_too_small",
                model=model or "-",
                window=context_window_tokens,
                required=required_tokens,
            ),
        )
        _record_algorithm_failure(
            s,
            settings,
            node=node,
            phase="context_window",
            reason=reason,
            message=message,
            blocking=True,
            details=details,
        )
        raise AgentModelCapacityError(model or "", phase, message)

    return details


def _query_id_for(s: dict[str, Any], *, round_index: int, query_index: int, query: str) -> str:
    return make_record_id(
        "qry",
        s.get("_run_id", ""),
        round_index,
        query_index,
        query,
    )


def _tier_explanations_for_urls(
    urls: list[str],
    strategies: StrategyContext,
) -> dict[str, dict[str, str]]:
    explain = getattr(strategies.source_tiering, "explain_url", None)
    explanations: dict[str, dict[str, str]] = {}
    for url in urls:
        canonical = normalize_url(str(url))
        if not canonical:
            continue
        if callable(explain):
            raw = explain(canonical)
            explanations[canonical] = {
                "tier": str(raw.get("tier", "unknown")),
                "tier_reason": str(raw.get("tier_reason", "")),
            }
        else:
            explanations[canonical] = {
                "tier": strategies.source_tiering.tier_for_url(canonical),
                "tier_reason": "tier_for_url",
            }
    return explanations


_EVIDENCE_LABEL_RE = re.compile(r"\[(E\d+)\](?:\(https?://[^\s)]+\))?")
_ADJACENT_EVIDENCE_LINK_RE = re.compile(
    r"(\[E\d+\]\(https?://[^\s)]+\))(?=\[E\d+\]\()"
)


def _extract_evidence_labels(text: str) -> set[str]:
    """Return per-source evidence labels (``E1`` ..) used in answer text."""
    return {match.group(1) for match in _EVIDENCE_LABEL_RE.finditer(text or "")}


def _expand_bare_evidence_label_links(
    answer_text: str,
    label_urls: dict[str, str],
) -> tuple[str, int]:
    """Convert bare evidence labels such as ``[E12]`` into Markdown links.

    ``label_urls`` is :attr:`EvidenceOverview.label_urls` -- the single
    label/URL map derived from the EvidenceLedger overview.
    """
    if not answer_text:
        return answer_text, 0
    expanded = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal expanded
        label = match.group(1)
        url = label_urls.get(label)
        if not url:
            return f"[{label}: nicht zugeordnet]"
        expanded += 1
        return f"[{label}]({url})"

    linked = re.sub(r"\[(E\d+)\](?!\()", replace, answer_text)
    linked = _ADJACENT_EVIDENCE_LINK_RE.sub(r"\1 ", linked)
    return linked, expanded


def _compact_section_summary(heading: str, text: str, max_chars: int = 900) -> str:
    """Return a deterministic compact summary of one rendered section."""
    compact = " ".join((text or "").split())
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return f"{heading}: {compact}" if compact else ""


def _build_answer_prompt_diagnostics(
    state_data: dict[str, Any],
    s: dict[str, Any],
) -> dict[str, Any]:
    """Return compact counters that explain answer-prompt evidence density.

    All counters describe the single EvidenceLedger overview view -- there
    are no parallel report-evidence / prompt-evidence-unit / rendered-
    context channels to account for anymore.
    """
    evidence_ledger = list(s.get("evidence_ledger", []) or [])
    report_eligible = [r for r in evidence_ledger if r.get("report_eligible")]
    return {
        "evidence_record_count": len(evidence_ledger),
        "report_eligible_evidence_count": len(report_eligible),
        "claimless_evidence_count": sum(
            1 for record in evidence_ledger if not record.get("claims")
        ),
        "rendered_evidence_record_count": int(
            state_data.get("rendered_evidence_record_count", 0) or 0
        ),
        "omitted_evidence_record_count": int(
            state_data.get("omitted_evidence_record_count", 0) or 0
        ),
        "evidence_overview_chars": len(state_data.get("evidence_overview", "") or ""),
        "visible_evidence_label_count": int(
            state_data.get("visible_evidence_label_count", 0) or 0
        ),
        "allowed_citation_count": len(state_data.get("allowed_citations", []) or []),
        "all_citation_count": len(s.get("all_citations", []) or []),
        "consolidated_claim_count": len(s.get("consolidated_claims", []) or []),
    }


@dataclass(slots=True)
class _AnswerCompositionResult:
    answer: str
    finish_reason: str
    section_logs: list[dict[str, Any]]
    composition_aborted: bool = False
    consecutive_empty_at_break: int = 0
    sections_planned: int = 0
    sections_attempted: int = 0


def _llm_complete_with_metadata(
    llm: Any,
    prompt: str,
    **kwargs: Any,
) -> LLMResponse:
    """Return an ``LLMResponse`` even for custom providers without metadata support."""
    state = kwargs.pop("state", None)
    complete_with_metadata = getattr(llm, "complete_with_metadata", None)
    if callable(complete_with_metadata):
        response = complete_with_metadata(prompt, state=None, **kwargs)
        if isinstance(response, LLMResponse):
            normalized = response
        else:
            content = getattr(response, "content", None)
            normalized = LLMResponse(content=str(content if content is not None else response))
    else:
        normalized = LLMResponse(
            content=llm.complete(prompt, **kwargs),
            model=str(kwargs.get("model") or ""),
        )

    if state is not None:
        track_tokens(state, normalized)
    return normalized


def _provider_retry_reason(notice: dict[str, Any]) -> str:
    """Return a compact retry reason safe for progress output."""
    for key in ("error_code", "status_code", "error_type", "reason"):
        value = notice.get(key)
        if value not in (None, ""):
            return str(value)
    return "transient_error"


def _provider_retry_progress_context(
    provider: object,
    s: dict[str, Any],
    *,
    operation_label: str,
    testing_mode: bool = False,
) -> ContextManager[object]:
    """Bind provider retry attempts to live progress events for one call."""
    observer = getattr(provider, "observe_retries", None)
    if not callable(observer):
        return nullcontext()

    def _on_retry(notice: dict[str, Any]) -> None:
        provider_label = str(notice.get("provider") or type(provider).__name__)
        model_label = str(notice.get("model") or "").strip() or "unknown-model"
        attempt = int(notice.get("attempt", 0) or 0)
        max_attempts = int(notice.get("max_attempts", 0) or 0)
        delay = float(notice.get("delay_seconds", 0.0) or 0.0)
        reason = _provider_retry_reason(notice)
        emit_progress(
            s,
            t(
                s,
                "provider_retry_attempt",
                provider=provider_label,
                attempt=attempt,
                max_attempts=max_attempts,
                operation=operation_label,
                model=model_label,
                delay=f"{delay:.1f}",
                reason=reason,
            ),
            severity="warning",
        )
        append_iteration_log(
            s,
            {
                "event": "provider_retry",
                "_provider_retry": True,
                "node": str(s.get("_current_node") or ""),
                "operation": operation_label,
                "provider": provider_label,
                "model": model_label,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay_seconds": delay,
                "reason": reason,
            },
            testing_mode=testing_mode,
        )

    return observer(_on_retry)


def _repair_answer_markdown_tail(answer: str) -> str:
    """Close obviously unbalanced markdown markers before appending diagnostics."""
    repaired = answer.rstrip()
    if not repaired:
        return repaired
    if repaired.count("```") % 2 == 1:
        repaired += "\n```"
    if repaired.count("**") % 2 == 1:
        repaired += "**"
    return repaired


def _finish_reason_indicates_limit(finish_reason: str) -> bool:
    """Return whether the provider reported a non-standard stop condition."""
    return (finish_reason or "").strip().lower() not in _ALLOWED_FINISH_REASONS


def _collect_truncation_signals(
    answer: str,
    *,
    subject_label: str,
) -> list[str]:
    """Collect content-based truncation signals independent of provider metadata."""
    reasons: list[str] = []
    stripped = answer.rstrip()
    if not stripped:
        reasons.append(f"{subject_label} ist leer")
        return reasons

    if stripped.count("**") % 2 == 1:
        reasons.append(f"Unbalancierte Markdown-Fettschrift am Ende {subject_label.lower()}")
    if stripped.count("```") % 2 == 1:
        reasons.append(f"Offener Markdown-Codeblock am Ende {subject_label.lower()}")

    last_line = stripped.splitlines()[-1].strip()
    plain_last_line = re.sub(r"[*_`#>-]", "", last_line).strip()
    if (
        plain_last_line
        and not last_line.startswith(("#", "- ", "* ", ">", "1. "))
        and len(plain_last_line) < 24
        and plain_last_line[-1].isalnum()
        and not re.search(r"[.!?:;\)\]]$", plain_last_line)
    ):
        reasons.append(f"{subject_label} endet mit einem kurzen Fragment ohne Satzschluss")
    return reasons


def _normalize_generated_section(answer: str, heading: str) -> str:
    """Remove a duplicated top-level heading and trim accidental spillover."""
    lines = answer.lstrip().splitlines()
    if lines:
        first_line = re.sub(r"^[#\s]+", "", lines[0]).strip().rstrip(":")
        if first_line.casefold() == heading.casefold():
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]

    kept_lines: list[str] = []
    seen_content = False
    for line in lines:
        if seen_content and line.startswith("## "):
            break
        if line.strip():
            seen_content = True
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _detect_incomplete_section(
    answer: str,
    *,
    finish_reason: str,
) -> tuple[list[str], bool]:
    """Return hard truncation reasons and whether a provider limit was hit."""
    reasons = _collect_truncation_signals(answer, subject_label="Abschnitt")
    limit_hit = _finish_reason_indicates_limit(finish_reason)
    if limit_hit and reasons:
        reasons.insert(0, f"Provider-Stopgrund: {(finish_reason or '').strip().lower()}")

    return _dedupe_reasons(reasons), limit_hit


def _join_rendered_sections(sections: list[str]) -> str:
    return "\n\n---\n\n".join(section.strip() for section in sections if section.strip())


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def _compose_answer_sections(
    s: dict,
    *,
    providers: ProviderContext,
    settings: AgentSettings,
    state_data: dict[str, Any],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> _AnswerCompositionResult:
    """Compose the final answer section by section to avoid hard output truncation.

    Every section is fed a section-scoped slice of the single canonical
    EvidenceLedger overview. Citation labels are stable across sections
    (passed in via ``state_data["evidence_label_by_id"]``), so ``[E12]``
    means the same source everywhere and the running ``used_evidence_labels``
    memory genuinely spreads source coverage across the report.
    """
    completed_headings: list[str] = []
    section_logs: list[dict[str, Any]] = []
    finish_reason = ""
    tuning = settings.report_tuning
    answer_sections = tuple(tuning.answer_sections)
    section_count = max(1, len(answer_sections))
    # The answer node is resolved once in ``answer()``; the model + reasoning
    # effort are passed in here and used for every section. The fallback retry
    # passes its own model with an empty effort so a reasoning-induced failure
    # is not repeated on the retry.
    answer_model = model or ""
    answer_effort = reasoning_effort or ""
    # ``section_focus_record_cap`` only controls the size of the soft
    # ``section_focus_labels`` hint in the user prompt. The LLM still sees
    # the FULL evidence overview in the system prompt -- the hint is a
    # relevance suggestion, not a hard filter.
    section_focus_record_cap = max(6, min(20, int(48 / section_count) or 6))
    consecutive_empty = 0
    _MAX_CONSECUTIVE_EMPTY = 2
    composition_aborted = False
    consecutive_empty_at_break = 0
    sections_attempted = 0
    report_so_far_summary = ""
    used_evidence_labels: set[str] = set()
    evidence_ledger = list(s.get("evidence_ledger", []) or [])
    label_by_evidence_id = dict(state_data.get("evidence_label_by_id", {}) or {})
    visible_evidence_labels = set(state_data.get("visible_evidence_labels", []) or [])
    required_aspects = list(state_data.get("required_aspects", []) or [])
    full_allowed_citation_count = len(state_data.get("allowed_citations", []) or [])

    # Write order vs display order: sections flagged ``write_last`` (Executive
    # Summary in DEEP, Kurzfazit in COMPACT) are rendered AFTER the body so
    # they see the completed body in ``report_so_far_summary`` and
    # ``used_evidence_labels`` and can synthesise it. They still appear at
    # their declared position in the final answer.
    write_plan: list[tuple[int, AnswerSectionSpec]] = [
        (display_index, section_spec)
        for display_index, section_spec in enumerate(answer_sections, 1)
        if not section_spec.write_last
    ] + [
        (display_index, section_spec)
        for display_index, section_spec in enumerate(answer_sections, 1)
        if section_spec.write_last
    ]
    rendered_by_display_index: dict[int, str] = {}

    for index, section_spec in write_plan:
        sections_attempted += 1
        emit_progress(
            s,
            t(
                s,
                "answer_section_start",
                index=index,
                total=len(answer_sections),
                heading=section_spec.heading,
            ),
        )
        provider_output_budget = _llm_default_output_tokens(providers.llm)
        focus_records = select_section_evidence_records(
            evidence_ledger,
            heading=section_spec.heading,
            required_aspects=required_aspects,
            used_labels=used_evidence_labels,
            label_by_evidence_id=label_by_evidence_id,
            max_records=section_focus_record_cap,
        )
        section_focus_labels = sorted(
            {
                label_by_evidence_id[str(record.get("evidence_id", "") or "")]
                for record in focus_records
                if str(record.get("evidence_id", "") or "") in label_by_evidence_id
                and label_by_evidence_id[str(record.get("evidence_id", "") or "")]
                in visible_evidence_labels
            }
        )
        # Each section sees the FULL EvidenceLedger overview via the shared
        # ``state_data`` -- no scoping. The selection above is converted into
        # a soft ``section_focus_labels`` hint in the user prompt so the LLM
        # knows which records the heading heuristic considered most relevant,
        # but the LLM remains free to cite any source from the full overview.
        section_system = build_answer_section_system_prompt(
            state_data,
            heading=section_spec.heading,
            instruction=section_spec.prompt_instruction,
            length_guidance=section_spec.length_guidance,
            section_position=index,
            section_total=len(answer_sections),
        )
        section_prompt = build_answer_section_user_prompt(
            s["question"],
            heading=section_spec.heading,
            instruction=section_spec.prompt_instruction,
            completed_headings=completed_headings,
            report_so_far_summary=report_so_far_summary,
            used_evidence_labels=sorted(used_evidence_labels),
            section_focus_labels=section_focus_labels,
            synthesizing_existing=bool(section_spec.write_last),
        )
        capacity_details = _check_llm_context_capacity(
            s,
            settings,
            providers=providers,
            node="answer",
            phase=f"answer_section:{section_spec.heading}",
            model=answer_model,
            system=section_system,
            prompt=section_prompt,
            requested_output_tokens=provider_output_budget,
        )
        with _provider_retry_progress_context(
            providers.llm,
            s,
            operation_label=(
                f"{t(s, 'retry_operation_answer_synthesis')}: "
                f"{section_spec.heading}"
            ),
            testing_mode=settings.testing_mode,
        ):
            response = _llm_complete_with_metadata(
                providers.llm,
                section_prompt,
                system=section_system,
                deadline=s["deadline"],
                state=s,
                model=answer_model,
                reasoning_effort=answer_effort,
                max_output_tokens=None,
                timeout=settings.reasoning_timeout,
            )
        finish_reason = str(getattr(response, "finish_reason", "") or "")
        section_body = _normalize_generated_section(response.content, section_spec.heading)
        section_body = _repair_answer_markdown_tail(section_body)
        section_reasons, limit_hit = _detect_incomplete_section(
            section_body,
            finish_reason=finish_reason,
        )
        section_prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
        section_completion_tokens = int(getattr(response, "completion_tokens", 0) or 0)
        # Provider-reported effective max_tokens (post-clamp). Falls back to
        # the provider default because report profiles no longer own token
        # budgets.
        section_request_max_tokens = (
            int(getattr(response, "request_max_tokens", 0) or 0)
            or provider_output_budget
        )
        # Utilization against the budget that was actually applied.
        section_token_utilization = (
            round(section_completion_tokens / section_request_max_tokens, 3)
            if section_request_max_tokens
            else 0.0
        )
        # Reflect the per-call reasoning effort actually routed to this section,
        # not the provider's global thinking flag: a graded effort turns thinking
        # on, "none" forces it off, and "" inherits the provider default.
        if answer_effort == "none":
            thinking_likely_active = False
        elif answer_effort:
            thinking_likely_active = True
        else:
            thinking_likely_active = bool(getattr(providers.llm, "_thinking", None))
        visible_tokens_estimate = max(0, round(len(section_body) / 4))
        if thinking_likely_active:
            thinking_tokens_estimate = max(
                0, section_completion_tokens - visible_tokens_estimate
            )
        else:
            thinking_tokens_estimate = 0
        section_log_entry: dict[str, Any] = {
            "heading": section_spec.heading,
            "position": index,
            "request_max_tokens": section_request_max_tokens,
            "required": section_spec.required,
            "model": answer_model,
            "content_length": len(section_body),
            "content_preview": section_body[:240],
            "finish_reason": finish_reason,
            "limit_hit": limit_hit,
            "incomplete": bool(section_reasons),
            "incomplete_reasons": section_reasons,
            "accepted_with_limit": bool(limit_hit and not section_reasons),
            "prompt_tokens": section_prompt_tokens,
            "completion_tokens": section_completion_tokens,
            "thinking_likely_active": thinking_likely_active,
            "visible_tokens_estimate": visible_tokens_estimate,
            "token_utilization": section_token_utilization,
            "system_prompt_chars": len(section_system),
            "user_prompt_chars": len(section_prompt),
            "section_focus_record_count": len(focus_records),
            "section_focus_labels": section_focus_labels[:20],
            "section_allowed_citation_count": full_allowed_citation_count,
            "section_scoped_evidence": True,
            "used_evidence_labels": sorted(used_evidence_labels)[:30],
            "estimated_input_tokens": capacity_details.get("estimated_input_tokens", 0),
            "requested_output_tokens": provider_output_budget,
            "estimated_required_context_tokens": capacity_details.get(
                "estimated_required_context_tokens", 0
            ),
            "context_window_tokens": capacity_details.get("context_window_tokens"),
            "required_context_window_tokens": capacity_details.get(
                "required_context_window_tokens", 0
            ),
        }
        if thinking_likely_active:
            section_log_entry["thinking_tokens_estimate"] = thinking_tokens_estimate
        _append_forensic_event(
            s,
            settings,
            event="answer_section",
            node="answer",
            payload=section_log_entry,
        )
        section_logs.append(section_log_entry)
        if limit_hit:
            emit_progress(
                s,
                t(
                    s,
                    "compose_section_token_limit",
                    heading=section_spec.heading,
                    finish_reason=finish_reason,
                    request_max=section_request_max_tokens,
                    completion=section_completion_tokens,
                ),
            )
            log.warning(
                "TRACE section %d/%d '%s': token limit hit "
                "(finish_reason=%s, request_max_tokens=%d, "
                "completion_tokens=%d, content_length=%d)",
                index,
                len(answer_sections),
                section_spec.heading,
                finish_reason,
                section_request_max_tokens,
                section_completion_tokens,
                len(section_body),
            )
        if section_reasons:
            emit_progress(
                s,
                t(
                    s,
                    "compose_section_truncation",
                    heading=section_spec.heading,
                    reasons=", ".join(section_reasons),
                ),
            )
            log.warning(
                "TRACE section %d/%d '%s': truncation signals detected "
                "(reasons=%s, finish_reason=%s)",
                index,
                len(answer_sections),
                section_spec.heading,
                section_reasons,
                finish_reason,
            )
        if section_body:
            rendered_by_display_index[index] = (
                f"## {section_spec.heading}\n\n{section_body}"
            )
            completed_headings.append(section_spec.heading)
            used_evidence_labels.update(_extract_evidence_labels(section_body))
            section_summary = _compact_section_summary(
                section_spec.heading,
                section_body,
            )
            if section_summary:
                report_so_far_summary = (
                    f"{report_so_far_summary}\n{section_summary}".strip()
                )
                if len(report_so_far_summary) > 2400:
                    report_so_far_summary = "..." + report_so_far_summary[-2400:]
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY:
                composition_aborted = True
                consecutive_empty_at_break = consecutive_empty
                emit_progress(
                    s,
                    t(
                        s,
                        "compose_aborted",
                        n=consecutive_empty,
                        heading=section_spec.heading,
                    ),
                )
                log.warning(
                    "TRACE compose: aborting after %d consecutive empty sections "
                    "(last='%s', position=%d/%d, finish_reason=%s)",
                    consecutive_empty,
                    section_spec.heading,
                    index,
                    len(answer_sections),
                    finish_reason or "unknown",
                )
                break

    # Assemble in DISPLAY order regardless of write order so the final
    # answer reads as planned: Executive Summary / Kurzfazit at the top
    # even though they were written last.
    rendered_sections = [
        rendered_by_display_index[idx]
        for idx in sorted(rendered_by_display_index)
    ]
    answer_text = _join_rendered_sections(rendered_sections)

    return _AnswerCompositionResult(
        answer=answer_text.strip(),
        finish_reason=finish_reason,
        section_logs=section_logs,
        composition_aborted=composition_aborted,
        consecutive_empty_at_break=consecutive_empty_at_break,
        sections_planned=len(answer_sections),
        sections_attempted=sections_attempted,
    )


def _missing_terminal_sections(answer: str, profile: ReportProfile) -> list[str]:
    """Return expected closing sections that are absent from the answer."""
    if profile is ReportProfile.DEEP:
        expected = (
            "## Perspektiven / Positionen",
            "## Risiken / Unsicherheiten",
            "## Fazit / Ausblick",
        )
    else:
        expected = ("## Einordnung / Ausblick",)
    return [section for section in expected if section not in answer]


def _detect_incomplete_answer(
    answer: str,
    *,
    finish_reason: str,
    report_profile: ReportProfile,
) -> list[str]:
    """Return diagnostic reasons when the answer appears truncated."""
    reasons = _collect_truncation_signals(answer, subject_label="Antwort")
    if _finish_reason_indicates_limit(finish_reason) and reasons:
        reasons.insert(0, f"Provider-Stopgrund: {(finish_reason or '').strip().lower()}")

    if reasons:
        missing_sections = _missing_terminal_sections(answer, report_profile)
        if missing_sections:
            reasons.append(
                "Fehlende Abschlussabschnitte: "
                + ", ".join(missing_sections)
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def _extract_used_reference_links(
    answer: str,
    allowed_urls: set[str],
) -> list[tuple[str, str]]:
    """Extract allowed markdown links from the answer in first-seen order."""
    references: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for match in _MARKDOWN_LINK_RE.finditer(answer):
        label = match.group(1).strip() or "Quelle"
        url = normalize_url(match.group(2))
        if url not in allowed_urls or url in seen_urls:
            continue
        seen_urls.add(url)
        references.append((label, url))
    return references


def _format_reference_entries(entries: list[tuple[str, str]]) -> str:
    """Render a deterministic markdown section for report references."""
    lines = ["## Referenzen"]
    for label, url in entries:
        domain = domain_from_url(url)
        suffix = f" — {domain}" if domain else ""
        lines.append(f"- [{label}]({url}){suffix}")
    return "\n".join(lines)


def _format_report_reference_records(
    entries: list[tuple[str, str]],
    strategies: StrategyContext,
    *,
    tier_by_url: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Return the structured twin of the rendered reference appendix."""
    records: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    known_tiers = tier_by_url or {}
    for label, raw_url in entries:
        url = normalize_url(raw_url)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        records.append({
            "label": str(label or "Quelle"),
            "url": url,
            "tier": known_tiers.get(url) or strategies.source_tiering.tier_for_url(url),
        })
    return records


def _reference_tiers_from_evidence_ledger(
    evidence_ledger: list[dict[str, Any]],
) -> dict[str, str]:
    """Map normalized evidence URLs to their ledger source tier."""
    tiers: dict[str, str] = {}
    for record in evidence_ledger:
        tier = str(record.get("tier", "") or "")
        if not tier:
            continue
        canonical_url = normalize_url(record.get("canonical_url", ""))
        if canonical_url:
            tiers.setdefault(canonical_url, tier)
        for citation in record.get("citation_set", []) or []:
            if not isinstance(citation, dict):
                continue
            citation_url = normalize_url(citation.get("url", ""))
            if citation_url:
                tiers.setdefault(citation_url, tier)
    return tiers


def _append_uncited_allowed_references(
    entries: list[tuple[str, str]],
    *,
    allowed_citations: list[str],
    label_urls: dict[str, str],
) -> list[tuple[str, str]]:
    """Append every rendered online evidence source missing from references."""
    merged = list(entries)
    seen_urls = {normalize_url(url) for _, url in merged if normalize_url(url)}
    label_by_url: dict[str, str] = {}
    for label, raw_url in sorted(
        label_urls.items(),
        key=lambda item: int(item[0][1:]) if re.fullmatch(r"E\d+", item[0]) else 10**9,
    ):
        url = normalize_url(raw_url)
        if url:
            label_by_url.setdefault(url, label)

    for index, raw_url in enumerate(allowed_citations, 1):
        url = normalize_url(str(raw_url))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        merged.append((label_by_url.get(url, f"Quelle {index}"), url))
    return merged


def _select_additional_links(
    citations: list[str],
    *,
    excluded_urls: set[str],
    prompt_citation_urls: set[str],
    strategies: StrategyContext,
    limit: int = _ADDITIONAL_LINK_LIMIT,
) -> list[str]:
    """Select a curated set of additional links not used in the report body.

    Note:
        This is intentionally separate from
        :meth:`ClaimConsolidationStrategy.select_answer_citations`. The
        latter ranks by claim score for in-prompt citations; this helper
        ranks raw citations by source tier with strict domain diversity for
        the post-answer "Weiterfuehrende Links" section. They share the
        per-domain diversity idea but produce different orderings on
        purpose — keep them aligned by behaviour, not by code path.
    """
    ranked_candidates: list[tuple[int, int, int, str, str]] = []
    seen_urls: set[str] = set()

    for index, raw_url in enumerate(citations):
        normalized = normalize_url(str(raw_url))
        if not normalized or normalized in excluded_urls or normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        tier = strategies.source_tiering.tier_for_url(normalized)
        if tier == "low":
            continue
        ranked_candidates.append(
            (
                _SOURCE_TIER_SORT_ORDER.get(tier, _SOURCE_TIER_SORT_ORDER["unknown"]),
                0 if normalized in prompt_citation_urls else 1,
                index,
                domain_from_url(normalized),
                normalized,
            )
        )

    ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    selected: list[str] = []
    deferred: list[tuple[int, int, int, str, str]] = []
    seen_domains: set[str] = set()

    for candidate in ranked_candidates:
        _, _, _, domain, url = candidate
        if domain and domain in seen_domains:
            deferred.append(candidate)
            continue
        selected.append(url)
        if domain:
            seen_domains.add(domain)
        if len(selected) >= limit:
            return selected

    for candidate in deferred:
        selected.append(candidate[4])
        if len(selected) >= limit:
            break

    return selected


def _format_additional_links(urls: list[str], strategies: StrategyContext) -> str:
    """Render a deterministic markdown section for curated extra links."""
    lines = ["## Weiterfuehrende Links"]
    for url in urls:
        domain = domain_from_url(url) or url
        tier = strategies.source_tiering.tier_for_url(url)
        lines.append(f"- [{domain}]({url}) — Tier: {tier}")
    return "\n".join(lines)


def _build_answer_appendix_sections(
    answer: str,
    *,
    allowed_citations: list[str],
    label_urls: dict[str, str] | None = None,
    strategies: StrategyContext,
    incomplete_reasons: list[str],
    finish_reason: str,
    answer_contract: str = "general",
    tier_by_url: dict[str, str] | None = None,
) -> AnswerAppendixSections:
    """Build optional post-answer sections without affecting the stats footer.

    ``allowed_citations`` is the EvidenceOverview-derived citation allowlist
    (the union of visible source-block URLs). References are the links the
    answer body actually used; additional links are allowlist sources the
    body left uncited.
    """
    sections: list[str] = []

    allowed_citation_urls = {
        normalize_url(url)
        for url in allowed_citations
        if normalize_url(url)
    }

    if incomplete_reasons:
        lines = [
            "## Hinweis zur Vollständigkeit",
            "- Status: unvollstaendig",
        ]
        if finish_reason:
            lines.append(f"- Provider-Stopgrund: `{finish_reason}`")
        lines.append("- Diagnose: " + "; ".join(incomplete_reasons))
        lines.append(
            "- Hinweis: Bereits sauber abgeschlossene Abschnitte bleiben sichtbar; es wurde keine verdeckende Regeneration ausgefuehrt."
        )
        sections.append("\n".join(lines))

    reference_entries = _extract_used_reference_links(answer, allowed_citation_urls)
    if not reference_entries and incomplete_reasons and allowed_citation_urls:
        reference_entries = [
            (str(index), normalize_url(url))
            for index, url in enumerate(allowed_citations, 1)
            if normalize_url(url)
        ]
    reference_entries = _append_uncited_allowed_references(
        reference_entries,
        allowed_citations=allowed_citations,
        label_urls=label_urls or {},
    )

    reference_records = _format_report_reference_records(
        reference_entries,
        strategies,
        tier_by_url=tier_by_url,
    )
    reference_urls: set[str] = {record["url"] for record in reference_records}

    if reference_entries:
        sections.append(_format_reference_entries(reference_entries))
    else:
        sections.append(
            "## Referenzen\nKeine zitatgebundenen Markdown-Links im Antworttext gefunden."
        )

    additional_links: list[str] = []
    if answer_contract not in {"claim_check", "data_extraction"}:
        additional_links = _select_additional_links(
            allowed_citations,
            excluded_urls=reference_urls,
            prompt_citation_urls=allowed_citation_urls,
            strategies=strategies,
        )
        if additional_links:
            sections.append(_format_additional_links(additional_links, strategies))

    return AnswerAppendixSections(
        sections=sections,
        references=reference_records,
        additional_links=additional_links,
    )


def _answer_segments(answer: str) -> list[str]:
    """Split answer text into compact binding segments."""
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n{2,}", answer or "")
        if segment.strip()
    ]
    return segments or ([answer.strip()] if (answer or "").strip() else [])


def _claim_plausible_in_segment(
    claim: dict[str, Any],
    segment_tokens: set[str],
    normalized_segment: str,
) -> bool:
    """Heuristic: does the segment plausibly carry this consolidated claim?

    Uses the same primitives as :class:`ClaimConsolidationStrategy`: the
    consolidated claim signature is normalised text, and ``tokenize``
    produces lowercase word-tokens. A binding is plausible when either
    (a) the claim signature appears as a substring of the lowercased
    segment, or (b) the claim's content tokens overlap with the
    segment's content tokens by at least three non-stopword,
    non-negation, length>=3 tokens. The thresholds are intentionally
    permissive enough that paraphrased re-uses of the claim still
    match, but strict enough that a citation that merely shares its
    URL with multiple unrelated claims does not promote all of them
    to ``"matched"``.
    """
    signature = str(claim.get("signature", "")).strip().lower()
    if signature and signature in normalized_segment:
        return True
    claim_tokens = {
        t for t in tokenize(str(claim.get("claim_text", "")))
        if len(t) >= 3 and t not in STOPWORDS and t not in NEGATION_TOKENS
    }
    if not claim_tokens:
        return False
    overlap = claim_tokens & segment_tokens
    return len(overlap) >= 3


def _build_answer_claim_bindings(
    answer_body: str,
    *,
    consolidated_claims: list[dict[str, Any]],
    allowed_citations: list[str],
    provider_citation_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map final answer-body links to the claims and sources they support.

    The function scans only ``answer_body`` (the prose produced by the
    LLM, captured **before** any appendix sections such as references
    or further-reading lists were appended). Each markdown link inside
    a body segment is matched against the consolidated claim ledger
    using :func:`_claim_plausible_in_segment`. A claim that shares a
    URL with the citation only becomes ``binding_status="matched"``
    when the segment plausibly carries it; otherwise the row degrades
    to ``"source_only_binding"`` so the URL-claim relationship is
    visible without falsely promoting unrelated claims to ``matched``.
    Citations without any consolidated claim continue to emit
    ``"citation_without_claim"``.

    The optional ``provider_citation_records`` argument lets each
    binding carry the per-query ``citation_id`` for the URL, so log
    consumers can join an answer-side binding straight to the
    forensic ``provider_citation_record`` event without re-deriving
    the citation identity from the URL.
    """
    citation_urls = {normalize_url(url) for url in allowed_citations if normalize_url(url)}

    citation_ids_by_url: dict[str, list[str]] = {}
    for rec in provider_citation_records:
        canonical = str(rec.get("canonical_url", "")).strip()
        if not canonical:
            canonical = normalize_url(str(rec.get("url", "")))
        cid = str(rec.get("citation_id", "")).strip()
        if canonical and cid:
            citation_ids_by_url.setdefault(canonical, []).append(cid)

    claims_by_url: dict[str, list[dict[str, Any]]] = {}
    for claim in consolidated_claims:
        for url in claim.get("source_urls", []) or []:
            canonical = normalize_url(str(url))
            if canonical:
                claims_by_url.setdefault(canonical, []).append(claim)

    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for segment_index, segment in enumerate(_answer_segments(answer_body), start=1):
        normalized_segment = segment.lower()
        segment_preview = format_log_excerpt(segment, limit=500)
        segment_tokens = {
            t for t in tokenize(segment)
            if len(t) >= 3 and t not in STOPWORDS and t not in NEGATION_TOKENS
        }
        for match in _MARKDOWN_LINK_RE.finditer(segment):
            canonical = normalize_url(match.group(2))
            if not canonical or canonical not in citation_urls:
                continue
            source_id = make_record_id("src", canonical)
            citation_id = (citation_ids_by_url.get(canonical) or [""])[0]
            related_claims = claims_by_url.get(canonical, [])

            if not related_claims:
                binding_key = (str(segment_index), canonical, "")
                if binding_key in seen:
                    continue
                seen.add(binding_key)
                bindings.append(
                    {
                        "binding_id": make_record_id(
                            "bind",
                            segment_index,
                            canonical,
                            "unmatched",
                        ),
                        "answer_segment_id": f"segment_{segment_index}",
                        "answer_segment_preview": segment_preview,
                        "citation_url": canonical,
                        "source_id": source_id,
                        "citation_id": citation_id,
                        "claim_id": "",
                        "claim_status": "unmatched",
                        "binding_status": "citation_without_claim",
                    }
                )
                continue

            matched_any = False
            for claim in related_claims:
                claim_id = str(claim.get("claim_id", ""))
                binding_key = (str(segment_index), canonical, claim_id)
                if binding_key in seen:
                    continue
                if not _claim_plausible_in_segment(claim, segment_tokens, normalized_segment):
                    continue
                seen.add(binding_key)
                matched_any = True
                bindings.append(
                    {
                        "binding_id": make_record_id(
                            "bind",
                            segment_index,
                            canonical,
                            claim_id,
                        ),
                        "answer_segment_id": f"segment_{segment_index}",
                        "answer_segment_preview": segment_preview,
                        "citation_url": canonical,
                        "source_id": source_id,
                        "citation_id": citation_id,
                        "claim_id": claim_id,
                        "claim_status": str(claim.get("status", "unverified")),
                        "binding_status": "matched",
                    }
                )

            if not matched_any:
                binding_key = (str(segment_index), canonical, "source_only")
                if binding_key in seen:
                    continue
                seen.add(binding_key)
                bindings.append(
                    {
                        "binding_id": make_record_id(
                            "bind",
                            segment_index,
                            canonical,
                            "source_only",
                        ),
                        "answer_segment_id": f"segment_{segment_index}",
                        "answer_segment_preview": segment_preview,
                        "citation_url": canonical,
                        "source_id": source_id,
                        "citation_id": citation_id,
                        "claim_id": "",
                        "claim_status": "source_only",
                        "binding_status": "source_only_binding",
                    }
                )
    return bindings


# ======================================================================= #
# 1. classify
# ======================================================================= #


def classify(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Analyse the incoming question and seed the initial research state.

    Args:
        s: Mutable AgentState-compatible dict. Reads the question,
            deadline, and history; writes language, query type, risk flags,
            and aspect hints.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for risk scoring and downstream
            claim/context handling.
        settings: Agent behavior settings used for risk scoring and
            timeout handling.

    Returns:
        The mutated state dict with classification results.

    Raises:
        AgentRateLimited: Propagated when the upstream classification
            model hard-fails on rate limiting.

    Example:
        >>> classify(state, providers=providers, strategies=strategies, settings=settings)
        {'query_type': 'general', 'language': 'de', ...}
    """
    check_cancel_event(s)
    emit_progress(s, t(s, "classify_start"))
    _t0 = time.monotonic()
    _classify_fallback: dict[str, Any] = {}

    # Phase 12: drain effort/model-incompatibility warnings collected by the
    # provider at construction time. Each warning is shown to the user via
    # progress feed AND mirrored to log so it's not silently buried in the
    # provider's __init__ log line that runs once before any agent activity.
    _effort_warnings_consumer = getattr(
        providers.llm, "consume_effort_config_warnings", None,
    )
    if callable(_effort_warnings_consumer):
        for _warning in _effort_warnings_consumer() or []:
            emit_progress(s, t(s, "classify_warning_hint", warning=_warning), severity="warning")
            log.warning("CONFIG: %s", _warning)
    s["risk_score"] = strategies.risk_scoring.score(s["question"])
    s["high_risk"] = s["risk_score"] >= settings.high_risk_score_threshold
    classify_model, classify_effort = _resolve_node_llm(s, settings, providers, "classify")

    try:
        _check_deadline(s["deadline"])
        with _provider_retry_progress_context(
            providers.llm,
            s,
            operation_label=t(s, "retry_operation_classify"),
            testing_mode=settings.testing_mode,
        ):
            d = providers.llm.complete(
                f"Heutiges Datum: {today()}\n\n"
                f"Analysiere diese Frage in ZWEI Teilen:\n\n"
                f"=== TEIL 1: KLASSIFIKATION ===\n"
                f"1. Braucht sie eine aktuelle Websuche? "
                f"(Aktuelle Ereignisse, Preise, Statistiken, neue Technologien, "
                f"veraenderliche Fakten → IMMER Suche)\n"
                f"2. In welcher Sprache ist die Frage geschrieben?\n"
                f"3. In welcher Sprache findet man die besten Suchergebnisse? "
                f"(z.B. Programmierung/Tech/Wissenschaft → oft Englisch, "
                f"lokale Themen/Politik/Recht → Sprache der Frage)\n"
                f"4. Wie aktuell muessen die Ergebnisse sein?\n"
                f"   - NONE: Zeitlose Fakten (Mathematik, Geschichte, Definitionen)\n"
                f"   - MONTH: Aktuelle Entwicklungen, neueste Versionen\n"
                f"   - WEEK: Nachrichten der letzten Tage, aktuelle Ereignisse\n"
                f"   - DAY: Breaking News, Live-Daten, heutige Ereignisse\n"
                f"   - HOUR: Echtzeit-Ereignisse, gerade passierende Breaking News\n"
                f"5. Welcher Suchtyp passt am besten?\n"
                f"   - GENERAL: Standard-Websuche\n"
                f"   - ACADEMIC: Wissenschaftliche Fragen, Studien, Papers\n"
                f"   - NEWS: Nachrichten, aktuelle Ereignisse, Meldungen\n\n"
                f"=== TEIL 2: DEKOMPOSITION ===\n"
                f"Zerlege die Frage in 1-3 unabhaengige Teilfragen fuer gezieltere Recherche.\n"
                f"Wenn die Frage einfach genug ist, gib sie unveraendert als einzelne Teilfrage zurueck.\n\n"
                f"ZEITLICHE VERANKERUNG:\n"
                f"- Interpretiere relative Zeitangaben (vor kurzem, neulich, letztens, kuerzlich) "
                f"immer relativ zum heutigen Datum ({today()}).\n"
                f"- 'vor kurzem' = letzte 2-4 Wochen vor dem heutigen Datum.\n"
                f"- FUEGE KEINE konkreten Jahreszahlen ein die du nicht aus der Frage kennst.\n"
                f"- Statt '2025' oder '2026' zu raten, nutze 'recent' oder das Datum.\n\n"
                f"Frage: {s['question']}\n\n"
                f"Antworte EXAKT in diesem Format:\n"
                f"DECISION: SEARCH oder DIRECT\n"
                f"LANGUAGE: Sprachcode der Frage (z.B. de, en, fr)\n"
                f"SEARCH_LANGUAGE: Sprachcode fuer optimale Suche (z.B. en, de)\n"
                f"RECENCY: NONE oder HOUR oder DAY oder WEEK oder MONTH\n"
                f"TYPE: GENERAL oder ACADEMIC oder NEWS\n"
                f"SUB_QUESTIONS: JSON Array von 1-3 Teilfragen als Strings",
                deadline=s["deadline"],
                model=classify_model,
                reasoning_effort=classify_effort,
                state=s,
            )
        s["done"] = bool(re.search(r"DECISION:\s*DIRECT", d, re.IGNORECASE))

        # Extract language — keep the heuristic seed from initial_state when
        # the LLM omits the LANGUAGE field, instead of hard-defaulting to "de".
        m_lang = re.search(r"LANGUAGE:\s*(\w+)", d)
        if m_lang:
            s["language"] = m_lang.group(1).strip().lower()[:2]
        else:
            s["language"] = s.get("language") or "de"

        m_search_lang = re.search(r"SEARCH_LANGUAGE:\s*(\w+)", d)
        s["search_language"] = m_search_lang.group(
            1).strip().lower()[:2] if m_search_lang else s["language"]

        # Extract recency requirement
        m_recency = re.search(r"RECENCY:\s*(\w+)", d)
        recency_raw = m_recency.group(1).strip().upper() if m_recency else "NONE"
        recency_map = {
            "HOUR": "hour",
            "DAY": "day",
            "WEEK": "week",
            "MONTH": "month",
            "NONE": "",
        }
        s["recency"] = recency_map.get(recency_raw, "")

        # Extract query type
        m_type = re.search(r"TYPE:\s*(\w+)", d)
        type_raw = m_type.group(1).strip().upper() if m_type else "GENERAL"
        type_map = {"ACADEMIC": "academic", "NEWS": "news", "GENERAL": "general"}
        s["query_type"] = type_map.get(type_raw, "general")

        # Fallback: keyword-based detection in case LLM misses academic questions
        if s["query_type"] != "academic":
            q_lower = s["question"].lower()
            academic_keywords = (
                "paper", "studie", "study", "preprint", "doi",
                "publikation", "publication", "arxiv", "veroeffentlich",
                "publish", "journal", "conference", "peer-review",
            )
            if any(kw in q_lower for kw in academic_keywords):
                _prev_type = s["query_type"]
                s["query_type"] = "academic"
                log.info("TRACE classify: type override %s->academic (keyword fallback)", _prev_type)

        infer_answer_contract = getattr(strategies.risk_scoring, "infer_answer_contract", None)
        s["answer_contract"] = (
            infer_answer_contract(s["question"])
            if callable(infer_answer_contract)
            else "general"
        )

        # Extract sub-questions (part 2 of the combined call)
        m_sub = re.search(r"SUB_QUESTIONS:\s*(\[.*)", d, re.DOTALL)
        sub_q_text = m_sub.group(1) if m_sub else ""
        s["sub_questions"] = parse_json_string_list(
            sub_q_text, fallback=[s["question"]], max_items=3)

        new_aspects = strategies.risk_scoring.derive_required_aspects(
            s["question"],
            s["query_type"],
            report_profile=settings.report_profile,
        )
        s["required_aspects"] = list(new_aspects)
        s["uncovered_aspects"] = list(new_aspects)
        s["aspect_coverage"] = 0.0

        # Trace logging
        log.info(
            "TRACE classify: decision=%s lang=%s search_lang=%s recency=%s type=%s sub_q=%d risk=%d high_risk=%s model=%s",
            "DIRECT" if s["done"] else "SEARCH",
            s["language"], s["search_language"], s["recency"] or "NONE", s["query_type"],
            len(s["sub_questions"]), s["risk_score"], s["high_risk"], classify_model,
        )

        if s["done"]:
            emit_progress(s, t(s, "classify_direct_answer"))
        else:
            hints: list[str] = []
            if s["search_language"] != s["language"]:
                sl = s["search_language"]
                lang_key = f"lang_name_{sl}"
                lang_name = t(s, lang_key) if lang_key in MESSAGES else sl
                hints.append(t(s, "search_in_lang", lang_name=lang_name))
            if s["recency"]:
                recency_key = f"recency_{s['recency']}"
                label = t(s, recency_key) if recency_key in MESSAGES else s["recency"]
                hints.append(t(s, "recency_label", label=label))
            if s["query_type"] != "general":
                type_key = f"type_{s['query_type']}"
                hints.append(t(s, type_key) if type_key in MESSAGES else s["query_type"])

            hint_str = f" ({', '.join(hints)})" if hints else ""
            emit_progress(s, t(s, "classify_search_required", hints=hint_str))

            if len(s["sub_questions"]) > 1:
                sub_q_display = ", ".join(
                    f"'{q}'" for q in s["sub_questions"][:3]
                )
                emit_progress(
                    s,
                    t(
                        s,
                        "classify_subquestions",
                        n=len(s["sub_questions"]),
                        subs=sub_q_display,
                    ),
                )
    except AgentRateLimited:
        raise
    except (
        OpenAIError,
        AgentTimeout,
        AnthropicAPIError,
        AzureOpenAIAPIError,
        BedrockAPIError,
    ) as exc:
        # Fail-safe: on classification error do NOT fall back to direct answer.
        # Conservatively continue researching with robust defaults — but make
        # the fallback visible in progress, logs and the iteration trace.
        _exc_label = type(exc).__name__
        log.warning(
            "Classify-Fallback aktiviert (%s): %s — nutze deterministische Defaults",
            _exc_label, exc,
        )
        emit_progress(s, t(s, "classify_failed", label=_exc_label), severity="warning")
        s["done"] = False
        s["language"] = s.get("language") or "de"
        s["search_language"] = s.get("search_language") or s["language"]
        s["query_type"] = strategies.risk_scoring.infer_query_type(s["question"])
        infer_answer_contract = getattr(strategies.risk_scoring, "infer_answer_contract", None)
        s["answer_contract"] = (
            infer_answer_contract(s["question"])
            if callable(infer_answer_contract)
            else "general"
        )
        s["recency"] = "month" if s["query_type"] == "news" else ""
        s["sub_questions"] = [s["question"]]
        s["required_aspects"] = strategies.risk_scoring.derive_required_aspects(
            s["question"],
            s["query_type"],
            report_profile=settings.report_profile,
        )
        s["uncovered_aspects"] = list(s["required_aspects"])
        s["aspect_coverage"] = 0.0
        _classify_fallback = {
            "fallback": "classify_default",
            "fallback_reason": _exc_label,
            "fallback_message": str(exc)[:300],
        }

    append_iteration_log(s, {
        "node": "classify",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "decision": "DIRECT" if s["done"] else "SEARCH",
        "question_length": len(s.get("question", "")),
        "history_length": len(s.get("history", "")),
        "lang": s["language"],
        "search_lang": s["search_language"],
        "recency": s["recency"] or "NONE",
        "type": s["query_type"],
        "answer_contract": s.get("answer_contract", "general"),
        "sub_questions": s["sub_questions"],
        "sub_question_count": len(s["sub_questions"]),
        "risk_score": s["risk_score"],
        "high_risk": s["high_risk"],
        "model": classify_model,
        "required_aspects": s.get("required_aspects", []),
        "_classify_parsed": not bool(_classify_fallback),
        "_classify_fallback": bool(_classify_fallback),
        **_classify_fallback,
    }, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 2. plan
# ======================================================================= #


def plan(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Generate the next batch of research queries.

    Args:
        s: Mutable AgentState-compatible dict. Reads the current round,
            required aspects, gaps, and related questions; writes new
            planned queries and related planning metadata.
        providers: Active LLM and search providers.
        strategies: Runtime strategies used to derive quality terms and
            other planning hints.
        settings: Agent behavior settings controlling round limits and
            first-round breadth.

    Returns:
        The mutated state dict with updated query planning.

    Raises:
        AgentRateLimited: Propagated when the planning model hard-fails
            on upstream rate limiting.

    Example:
        >>> plan(state, providers=providers, strategies=strategies, settings=settings)
        {'queries': ['gkv reform 2026', ...], ...}
    """
    check_cancel_event(s)
    emit_progress(
        s,
        t(s, "plan_start", round=s["round"] + 1, max_rounds=settings.max_rounds),
    )
    _t0 = time.monotonic()
    _plan_fallback: dict[str, Any] = {}
    _plan_algorithm_failure: dict[str, Any] = {}
    target_query_count = _target_query_count_for_round(s["round"], settings)
    sub_question_count = len(s.get("sub_questions") or [])
    required_aspect_count = len(s.get("required_aspects") or [])
    uncovered_aspect_count = len(s.get("uncovered_aspects") or [])
    query_generation_mode = "round0_breadth" if s["round"] == 0 else "gap_fill"
    perspective_instruction = ""
    deep_review_instruction = ""
    alternative_instruction = ""
    competing_instruction = ""
    reformulation_instruction = ""
    falsification_instruction = ""
    crosscheck_instruction = ""
    crosscheck_targets: list[dict[str, Any]] = []
    query_slots: list[dict[str, Any]] = []
    evidence_depth_gap = _evidence_depth_gap(s)
    s["evidence_depth_gap"] = evidence_depth_gap
    s["_evidence_depth_gap_active"] = bool(evidence_depth_gap.get("active"))
    try:
        _check_deadline(s["deadline"])
        is_deep = settings.report_profile is ReportProfile.DEEP

        # Build prompt with sub-questions and gap info
        sub_q_info = ""
        if s["sub_questions"]:
            sub_q_info = f"Teilfragen: {json.dumps(s['sub_questions'], ensure_ascii=False)}\n"

        required_info = ""
        if s.get("required_aspects"):
            required_info = (
                f"Pflichtaspekte fuer die Antwort:\n"
                f"{json.dumps(s['required_aspects'], ensure_ascii=False)}\n"
            )

        uncovered_info = ""
        if s.get("uncovered_aspects"):
            uncovered_info = (
                f"Noch NICHT ausreichend abgedeckt:\n"
                f"{json.dumps(s['uncovered_aspects'], ensure_ascii=False)}\n"
                f"Mindestens eine Query MUSS gezielt die offenen Aspekte abdecken.\n"
            )

        gap_info = ""
        if s["gaps"]:
            gap_info = f"Noch fehlende Informationen: {s['gaps']}\n"

        # Determine search language
        search_lang = s.get("search_language", s.get("language", "de"))
        lang_instruction = ""
        if search_lang == "en":
            lang_instruction = (
                "WICHTIG: Formuliere die Suchqueries auf ENGLISCH, "
                "da fuer dieses Thema englische Quellen besser sind.\n"
            )
        elif search_lang != s.get("language", "de"):
            lang_instruction = (
                f"WICHTIG: Formuliere die Suchqueries auf "
                f"{LANG_NAMES.get(search_lang, search_lang)}.\n"
            )

        # Perspective diversity (STORM-inspired)
        perspective_instruction = ""
        if is_deep or s["round"] > 0:
            perspective_instruction = (
                "PERSPEKTIV-DIVERSITAET: Betrachte das Thema aus einer ANDEREN Perspektive "
                "als die bisherigen Queries. Moegliche Perspektiven:\n"
                "- Technisch/Mechanistisch: Wie funktioniert es genau?\n"
                "- Praktisch/Anwendung: Wie wird es eingesetzt?\n"
                "- Kritisch/Limitierungen: Was sind die Grenzen und Probleme?\n"
                "- Vergleichend: Wie steht es im Vergleich zu Alternativen?\n"
                "- Historisch/Kontext: Wie hat es sich entwickelt?\n"
                "- Aktuell/Zukunft: Was sind die neuesten Entwicklungen?\n\n"
            )

        deep_review_instruction = ""
        if is_deep:
            deep_review_instruction = (
                "DEEP-REVIEW-MODUS:\n"
                "Decke die Frage systematisch aus mehreren Perspektiven ab. Suche nicht nur nach der Hauptthese, "
                "sondern auch nach Gegenpositionen, Betroffenen-/Stakeholder-Sichtweisen, Zahlen/Primarquellen "
                "und moeglichen Alternativen oder Vergleichen.\n"
            )
            if s["round"] == 0:
                deep_review_instruction += (
                    "In der ersten Runde sollen die Queries moeglichst unterschiedliche Pflichtaspekte abdecken. "
                    "Vermeide Varianten derselben Suche. Solange Slots verfuegbar sind, sollte mindestens je eine Query "
                    "auf Status quo, Stakeholder/Positionen, Risiken/Gegenargumente, Zahlen/Primarquellen und "
                    "Alternativen/Vergleich zielen.\n\n"
                )
            else:
                deep_review_instruction += (
                    "Nutze spaetere Runden gezielt fuer offene Perspektiven und Evidenzluecken statt fuer Wiederholungen.\n\n"
                )

        # Alternative hypothesis search
        alternative_instruction = ""
        if s["round"] == 1:
            alternative_instruction = (
                "WICHTIG — ALTERNATIVE HYPOTHESEN:\n"
                "Mindestens eine deiner Queries MUSS nach ALTERNATIVEN Ereignissen/Antworten suchen.\n"
                f"Heutiges Datum: {today()}. Die bisherigen Ergebnisse koennten ein AELTERES Ereignis "
                "beschreiben, das nicht das ist was der Nutzer meint.\n"
                "Suche gezielt nach dem AKTUELLSTEN passenden Ereignis — "
                "z.B. 'stock market crash AI February 2026' oder 'latest AI selloff this week'.\n"
                "Wenn die Frage 'vor kurzem/neulich/letztens' sagt, muss mindestens eine Query "
                f"explizit den Zeitraum der letzten 2-4 Wochen vor {today()} abdecken.\n\n"
            )

        # Competing events: force targeted comparison queries
        competing_instruction = ""
        competing = s.get("competing_events", "")
        if competing:
            competing_instruction = (
                f"WICHTIG — KONKURRIERENDE ERKLAERUNGEN:\n"
                f"Die Evaluierung hat folgende moegliche Ereignisse/Antworten identifiziert:\n"
                f"{competing}\n\n"
                f"Deine Queries MUESSEN gezielt klaeren welches Ereignis AKTUELLER und RELEVANTER ist.\n"
                f"Suche nach DIREKTEN Vergleichen, exakten Daten, und spezifischen Details "
                f"die eine eindeutige Zuordnung ermoeglichen.\n"
                f"Mindestens eine Query muss das NEUESTE der konkurrierenden Ereignisse "
                f"mit explizitem Datum/Zeitraum suchen.\n\n"
            )

        # Aggressive reformulation on low confidence after round 1
        reformulation_instruction = ""
        if s["round"] >= 2 and s.get("final_confidence", 5) <= 4:
            reformulation_instruction = (
                "ACHTUNG: Die bisherigen Suchen haben kaum relevante Ergebnisse geliefert "
                f"(Confidence: {s.get('final_confidence', '?')}/10 nach {s['round']} Runden).\n"
                "Du MUSST die Suchstrategie GRUNDLEGEND aendern:\n"
                "1. HINTERFRAGE DIE PRAEMISSE: Vielleicht existiert das Beschriebene gar nicht, "
                "oder der Nutzer verwechselt/vermischt verschiedene Dinge. "
                "Suche stattdessen nach dem was TATSAECHLICH existiert.\n"
                "2. Suche nach dem BREITEREN Thema: z.B. statt 'DeepSeek Paper ueber X' "
                "suche 'DeepSeek neueste Papers 2026 Liste aller Veroeffentlichungen'\n"
                "3. Suche nach den beteiligten Personen/Organisationen und deren neueste Arbeiten\n"
                "4. Formuliere KOMPLETT um — andere Begriffe, Synonyme, uebergeordnete Kategorien\n"
                "5. Suche nach Diskussionen/Nachrichten UEBER das Thema statt nach dem Thema selbst\n\n"
            )

        # Falsification mode (FVA-RAG-inspired)
        falsification_instruction = ""
        if s.get("falsification_triggered", False):
            falsification_query_count = min(2, target_query_count)
            falsification_instruction = (
                "FALSIFIKATIONS-MODUS AKTIV:\n"
                "Die bisherige Recherche hat wiederholt KEINE ueberzeugenden Belege fuer die "
                "Behauptung/Praemisse in der Frage gefunden. Jetzt suchen wir gezielt nach "
                "GEGEN-EVIDENZ um die Praemisse zu testen.\n\n"
                f"Mindestens {falsification_query_count} deiner {target_query_count} "
                "Queries MUESSEN Falsifikations-Queries sein:\n"
                "- '[Behauptung] debunked' oder '[Behauptung] myth'\n"
                "- '[Thema] does not exist' oder '[Thema] never happened'\n"
                "- '[Thema] hoax' oder '[Thema] refuted'\n"
                "- '[Thema] misinformation' oder '[Thema] false claim'\n\n"
                "Eine weitere Query MUSS nach dem TATSAECHLICH existierenden naechstliegenden "
                "Sachverhalt suchen: z.B. statt 'Gemini eingestellt' -> "
                "'Google Gemini aktueller Status 2026' oder 'Was hat [Organisation] "
                "TATSAECHLICH veroeffentlicht?'\n\n"
                "ZIEL: Entweder finden wir Belege dass die Praemisse falsch ist "
                "(-> hochkonfidente Antwort 'existiert nicht'), oder wir finden doch "
                "noch den richtigen Sachverhalt.\n\n"
            )

        answer_contract = s.get("answer_contract", "general")
        contract_instruction = ""
        if answer_contract == "claim_check":
            contract_instruction = (
                "ANTWORT-CONTRACT: Behauptungspruefung.\n"
                "- Plane wenige, zielgerichtete Queries fuer direkte Belege und Gegenbelege.\n"
                "- Priorisiere offizielle/Primaerquellen und unabhaengige Cross-Checks.\n"
                "- Vermeide breite Stakeholder-/Essay-Kontextsuche, wenn sie die Behauptung nicht prueft.\n\n"
            )
        elif answer_contract == "data_extraction":
            contract_instruction = (
                "ANTWORT-CONTRACT: Datenextraktion.\n"
                "- Plane Quellenpfade pro Kennzahl, Zeitraum und Unternehmen.\n"
                "- Suche zuerst nach IR-/Filing-/offiziellen Statistikquellen, danach nach Plausibilitaetsquellen.\n"
                "- Jede harte Zahl braucht eine direkte Quelle; vermeide generische News-Uebersichten.\n\n"
            )
        elif answer_contract == "news_briefing":
            contract_instruction = (
                "ANTWORT-CONTRACT: News-Briefing.\n"
                "- Breite ist erlaubt, aber zentrale Ereignisse brauchen Cross-Check durch mindestens zwei Quellen.\n"
                "- Suche nach Datum, Primaerquelle und etablierter Berichterstattung pro Ereignis.\n\n"
            )

        crosscheck_targets = (
            _select_crosscheck_targets(
                s.get("consolidated_claims", []) or [],
                max_targets=max(1, target_query_count // 2),
            )
            if s["round"] > 0
            else []
        )
        if crosscheck_targets:
            crosscheck_instruction = (
                "CROSS-CHECK-ZIELE:\n"
                "Mindestens eine Query soll eine der folgenden Aussagen mit einer unabhaengigen "
                "Quelle bestaetigen oder widerlegen. Nutze moeglichst andere Domains als die "
                "bisher genannten source_domains und suche nach exakten Zahlen, Daten oder Namen.\n"
                f"{json.dumps(crosscheck_targets, ensure_ascii=False)}\n\n"
            )

        query_slots = _build_query_slots(
            s,
            target_count=target_query_count,
            crosscheck_targets=crosscheck_targets,
            evidence_depth_gap=evidence_depth_gap,
        )
        slot_instruction = ""
        if query_slots:
            slot_instruction = (
                "RECHERCHE-SLOTS:\n"
                f"Erzeuge GENAU {target_query_count} Suchfragen: genau eine individuelle "
                "Suchfrage pro Slot, in derselben Reihenfolge wie die Slots. Jede Query "
                "muss den konkreten Slot-Auftrag abdecken und darf nicht nur eine "
                "Keyword-Kette sein.\n"
                f"{json.dumps(query_slots, ensure_ascii=False)}\n\n"
            )

        round_zero_query_instruction = (
            f"Erzeuge genau {target_query_count} diverse Suchqueries die verschiedene Aspekte, Hypothesen und Perspektiven der Frage breit abdecken"
            if s["round"] == 0
            else f"Erzeuge genau {target_query_count} praezise, spezifische Suchqueries"
        )

        plan_model, plan_effort = _resolve_node_llm(s, settings, providers, "plan")
        with _provider_retry_progress_context(
            providers.llm,
            s,
            operation_label=t(s, "retry_operation_plan"),
            testing_mode=settings.testing_mode,
        ):
            q = providers.llm.complete(
                f"Heutiges Datum: {today()}\n\n"
                f"{round_zero_query_instruction} fuer eine Websuche.\n"
                f"Jede Query sollte 5-15 Woerter lang sein und konkreten Kontext enthalten.\n"
                f"Formuliere jede Query als VOLLSTAENDIGE FRAGE (Was, Welche, Wie, Wann, Wo, Warum). "
                f"Keine reinen Stichwortlisten.\n"
                f"SCHLECHT: 'KI Entwicklung' (Stichwortkette, zu vage)\n"
                f"GUT: 'Welche Durchbrueche bei kuenstlichen Sprachmodellen sind 2025 erschienen?'\n\n"
                f"{reformulation_instruction}"
                f"{falsification_instruction}"
                f"{competing_instruction}"
                f"{alternative_instruction}"
                f"{perspective_instruction}"
                f"{deep_review_instruction}"
                f"{contract_instruction}"
                f"{crosscheck_instruction}"
                f"{slot_instruction}"
                f"{lang_instruction}"
                f"Frage: {s['question']}\n"
                f"{sub_q_info}"
                f"{required_info}"
                f"{uncovered_info}"
                f"{gap_info}"
                f"Bisherige Queries: {s['queries']}\n"
                f"Bisherige Ergebnisse: {len(s.get('evidence_ledger', []) or [])} Evidenz-Records\n\n"
                f"Generiere Queries die NEUE Informationen liefern, nicht schon Bekanntes wiederholen, "
                f"und formuliere sie als Fragen.\n"
                f"Antworte NUR mit einem JSON Array von Strings. Beispiel: [\"query1\", \"query2\"]",
                deadline=s["deadline"],
                model=plan_model,
                reasoning_effort=plan_effort,
                state=s,
            )
    except AgentRateLimited:
        raise
    except (
        OpenAIError,
        AgentTimeout,
        AnthropicAPIError,
        AzureOpenAIAPIError,
        BedrockAPIError,
    ) as exc:
        _exc_label = type(exc).__name__
        _failure_reason = f"Planung fehlgeschlagen ({_exc_label})"
        emit_progress(s, t(s, "plan_failed", label=_exc_label), severity="warning")
        _record_algorithm_failure(
            s,
            settings,
            node="plan",
            phase="plan_query_generation",
            reason=f"provider_{_exc_label}",
            message=f"{_failure_reason}; verwende Fallback-Query",
            blocking=False,
            details={"fallback": "original_question"},
        )
        q = ""
        query_generation_mode = "fallback_provider_error"
        _plan_algorithm_failure = {
            "_plan_algorithm_failure": True,
            "_plan_algorithm_failure_phase": "query_generation",
            "_plan_algorithm_failure_reason": f"provider_{_exc_label}",
        }
        _plan_fallback = {
            "fallback": "plan_default",
            "fallback_reason": _exc_label,
            "fallback_message": str(exc)[:300],
        }

    _max_items = target_query_count
    _fallback_query = [s["question"]]
    new_q, _query_parse_status = parse_json_string_list_with_status(
        q,
        fallback=_fallback_query,
        max_items=_max_items,
    )
    if query_slots and len(new_q) < _max_items:
        fallback_seen = {query.strip() for query in new_q if query.strip()}
        for slot in query_slots[len(new_q):]:
            candidate = _fallback_query_for_slot(
                slot,
                question=s.get("question", ""),
                search_language=s.get("search_language", s.get("language", "de")),
            )
            if candidate and candidate not in fallback_seen:
                new_q.append(candidate)
                fallback_seen.add(candidate)
            if len(new_q) >= _max_items:
                break
    _fell_back_to_original_question = (
        len(new_q) == 1
        and (new_q[0] or "").strip() == (s.get("question", "") or "").strip()
    )
    if not _plan_algorithm_failure and _query_parse_status != "parsed":
        _failure_reason = "Suchquery-Planung lieferte keine valide JSON-Liste"
        emit_progress(
            s,
            t(s, "plan_query_generation_failed", reason=_failure_reason),
            severity="warning",
        )
        _record_algorithm_failure(
            s,
            settings,
            node="plan",
            phase="plan_query_generation",
            reason=_query_parse_status,
            message=f"{_failure_reason}; verwende Fallback-Query",
            blocking=False,
            details={"fallback": "original_question"},
        )
        query_generation_mode = "fallback_invalid_json"
        _plan_algorithm_failure = {
            "_plan_algorithm_failure": True,
            "_plan_algorithm_failure_phase": "query_generation",
            "_plan_algorithm_failure_reason": _query_parse_status,
        }
    elif not _plan_algorithm_failure and _fell_back_to_original_question:
        _failure_reason = "Suchquery-Planung fiel auf die Originalfrage zurueck"
        emit_progress(
            s,
            t(s, "plan_query_generation_failed", reason=_failure_reason),
            severity="warning",
        )
        _record_algorithm_failure(
            s,
            settings,
            node="plan",
            phase="plan_query_generation",
            reason="fallback_original_question",
            message=f"{_failure_reason}; verwende Fallback-Query",
            blocking=False,
            details={"fallback": "original_question"},
        )
        query_generation_mode = "fallback_original_question"
        _plan_algorithm_failure = {
            "_plan_algorithm_failure": True,
            "_plan_algorithm_failure_phase": "query_generation",
            "_plan_algorithm_failure_reason": "fallback_original_question",
        }

    # Deduplicate, preserve order
    seen = set(s["queries"])
    added = 0
    for query in new_q:
        if query not in seen:
            s["queries"].append(query)
            seen.add(query)
            added += 1

    # Inform user about generated queries and active strategies
    strategies_active: list[str] = []
    if deep_review_instruction:
        strategies_active.append("DEEP-Pflichtperspektiven")
    if perspective_instruction:
        strategies_active.append("STORM-Perspektiven")
    if falsification_instruction:
        strategies_active.append("Falsifikation")
    if alternative_instruction:
        strategies_active.append("Alternative Hypothesen")
    if competing_instruction:
        strategies_active.append("Konkurrierende Erklaerungen")
    if reformulation_instruction:
        strategies_active.append("Reformulierung")
    if crosscheck_instruction:
        strategies_active.append("Cross-Check")
    if added > 0:
        analysis_target_label = (
            "Analyseziel" if sub_question_count == 1 else "Analysezielen"
        )
        aspect_label = (
            "Pflichtaspekt" if required_aspect_count == 1 else "Pflichtaspekten"
        )
        basis_hint = (
            f"aus {sub_question_count} {analysis_target_label}, "
            f"{required_aspect_count} {aspect_label}"
        )
        strategy_hint_parts = [basis_hint]
        if strategies_active:
            strategy_hint_parts.append(", ".join(strategies_active))
        strategy_hint = f" ({'; '.join(strategy_hint_parts)})"
        emit_progress(
            s,
            t(s, "plan_new_queries", added=added, strategy_hint=strategy_hint),
        )

    log.info(
        "TRACE plan: round=%d new_queries=%s total=%d",
        s["round"], json.dumps(new_q, ensure_ascii=False), len(s["queries"]),
    )

    # If no new queries: answer directly (prevents infinite loop)
    if added == 0:
        emit_progress(s, t(s, "plan_no_more_queries"))
        log.info("Keine neuen Suchqueries generiert, beende Recherche")
        s["done"] = True

    append_iteration_log(s, {
        "node": "plan",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"],
        "new_queries": new_q,
        "new_query_count": len(new_q),
        "added_queries": added,
        "target_query_count": target_query_count,
        "sub_question_count": sub_question_count,
        "required_aspect_count": required_aspect_count,
        "planning_basis": {
            "sub_question_count": sub_question_count,
            "required_aspect_count": required_aspect_count,
            "uncovered_aspect_count": uncovered_aspect_count,
            "active_strategies": strategies_active,
            "crosscheck_target_count": len(crosscheck_targets),
            "query_slot_count": len(query_slots),
            "query_slot_types": [slot.get("slot_type", "") for slot in query_slots],
            "evidence_depth_gap_active": bool(evidence_depth_gap.get("active")),
        },
        "crosscheck_targets": crosscheck_targets,
        "query_slots": query_slots,
        "query_slot_count": len(query_slots),
        "query_slot_types": [slot.get("slot_type", "") for slot in query_slots],
        "crosscheck_target_count": len(crosscheck_targets),
        "evidence_depth_gap": evidence_depth_gap,
        "query_generation_mode": query_generation_mode,
        "done_no_new_queries": added == 0,
        "_plan_fallback": bool(_plan_fallback),
        "_plan_parse_status": _query_parse_status,
        "_plan_algorithm_failure": bool(_plan_algorithm_failure),
        "_plan_stored_queries": added,
        "total_queries": len(s["queries"]),
        "required_aspects": s.get("required_aspects", []),
        "answer_contract": s.get("answer_contract", "general"),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        **_plan_fallback,
        **_plan_algorithm_failure,
    }, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 3. search
# ======================================================================= #


def search(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Execute the current query batch and merge search evidence into state.

    Args:
        s: Mutable AgentState-compatible dict. Reads queued queries,
            offsets, and deadline; writes context blocks, citations,
            token counters, claims, and round progress.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for claim extraction
            and consolidation.
        settings: Agent behavior settings controlling batch width,
            timeouts, and logging/test instrumentation.

    Returns:
        The mutated state dict after search and claim
        extraction complete or short-circuit.

    Raises:
        AgentRateLimited: Propagated when a provider surfaces a fatal
            rate limit that must abort the run.

    Example:
        >>> search(state, providers=providers, strategies=strategies, settings=settings)
        {'all_citations': ['https://...'], 'context': ['...'], ...}
    """
    check_cancel_event(s)
    _t0 = time.monotonic()
    _batch = _target_query_count_for_round(s["round"], settings)
    offset = s["search_offset"]
    new_q = s["queries"][offset:offset + _batch]
    s["search_offset"] = offset + len(new_q)
    emit_progress(
        s,
        t(
            s,
            "search_start",
            n=len(new_q),
            round=s["round"] + 1,
            max_rounds=settings.max_rounds,
        ),
    )

    if not new_q:
        # No queries left -> go straight to answer
        s["done"] = True
        s["round"] += 1
        append_iteration_log(s, {
            "node": "search",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "round": s["round"] - 1,
            "queries_executed": 0,
            "queries": [],
            "sources_found": 0,
            "total_citations": len(s["all_citations"]),
            "evidence_record_count": len(s.get("evidence_ledger", []) or []),
            "skipped": "no_queries",
        }, testing_mode=settings.testing_mode)
        return s

    try:
        _check_deadline(s["deadline"])
    except AgentTimeout:
        s["done"] = True
        s["round"] += 1
        append_iteration_log(s, {
            "node": "search",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "round": s["round"] - 1,
            "queries_executed": 0,
            "queries": [],
            "sources_found": 0,
            "total_citations": len(s["all_citations"]),
            "evidence_record_count": len(s.get("evidence_ledger", []) or []),
            "skipped": "deadline_exceeded",
        }, testing_mode=settings.testing_mode)
        return s

    search_capabilities = get_search_provider_capabilities(providers.search)

    # Build search parameters from classify results
    search_kwargs: dict[str, Any] = {
        "deadline": s["deadline"],
    }

    if search_capabilities.supports("search_context_size"):
        search_kwargs["search_context_size"] = "high"  # Always max depth

    # Recency filter
    recency = s.get("recency", "")
    if recency and search_capabilities.supports("recency_filter"):
        search_kwargs["recency_filter"] = recency

    # Language filter via API (more reliable than prompt instruction)
    search_lang = s.get("search_language", "")
    if search_lang and search_capabilities.supports("language_filter"):
        search_kwargs["language_filter"] = [search_lang]

    # Query type -> search_mode
    query_type = s.get("query_type", "general")
    if query_type == "academic" and search_capabilities.supports("search_mode"):
        search_kwargs["search_mode"] = "academic"

    # Explicit `site:...` operators still become provider allow-lists. Normal
    # queries intentionally do not carry a default domain blocklist so search
    # breadth is not silently constrained before source-tiering can inspect it.
    _base_domain_filter: list[str] | None = None
    _forensic = forensic_enabled(settings)
    _collect_query_details = settings.testing_mode or log.isEnabledFor(logging.DEBUG) or _forensic

    def _consume_nonfatal_notice(obj: object) -> str | None:
        consumer = getattr(obj, "consume_nonfatal_notice", None)
        if callable(consumer):
            return consumer()
        return None

    def _consume_retry_notices(obj: object) -> list[dict[str, Any]]:
        consumer = getattr(obj, "consume_retry_notices", None)
        if not callable(consumer):
            return []
        notices = consumer()
        if not isinstance(notices, list):
            return []
        return [dict(item) for item in notices if isinstance(item, dict)]

    def _emit_retry_progress(
        notices: list[dict[str, Any]],
        *,
        operation_label: str,
    ) -> None:
        notices = [notice for notice in notices if not notice.get("progress_emitted")]
        if not notices:
            return
        provider_label = str(notices[-1].get("provider") or type(providers.llm).__name__)
        models = sorted({
            str(item.get("model") or "").strip()
            for item in notices
            if str(item.get("model") or "").strip()
        })
        model_label = models[0] if len(models) == 1 else f"{len(models)} models"
        emit_progress(
            s,
            t(
                s,
                "provider_retry_observed",
                provider=provider_label,
                count=len(notices),
                operation=operation_label,
                model=model_label,
            ),
            severity="warning",
        )

    def _domain_filter_for_query(q: str) -> list[str] | None:
        return _domain_filter_for_query_text(q, base_domain_filter=_base_domain_filter)

    # Request related questions in first round
    if s["round"] == 0 and search_capabilities.supports("return_related"):
        search_kwargs["return_related"] = True

    _query_domain_filters = [_domain_filter_for_query(q) for q in new_q]

    # Parallel search
    def _search_one(item: tuple[int, str, list[str] | None]) -> SearchOutcome:
        query_index, q, domain_filter = item
        effective_domain_filter = (
            domain_filter if search_capabilities.supports("domain_filter") else None
        )
        operation_label = f"{t(s, 'retry_operation_search')} {query_index + 1}/{len(new_q)}"
        with _provider_retry_progress_context(
            providers.search,
            s,
            operation_label=operation_label,
            testing_mode=settings.testing_mode,
        ):
            result = providers.search.search(
                q,
                domain_filter=effective_domain_filter,
                **search_kwargs,
            )
        if not isinstance(result, GroundedSearchResult):
            raise TypeError("SearchProvider.search() must return GroundedSearchResult")
        return SearchOutcome(result, _consume_nonfatal_notice(providers.search))

    provider_cap = int(getattr(search_capabilities, "max_concurrency", 0) or 0)
    _n_workers = min(
        len(new_q),
        settings.first_round_queries,
        provider_cap or len(new_q),
    )

    with ThreadPoolExecutor(max_workers=_n_workers) as ex:
        _outcomes = list(
            ex.map(
                _search_one,
                (
                    (query_index, q, domain_filter)
                    for query_index, (q, domain_filter) in enumerate(
                        zip(new_q, _query_domain_filters, strict=True)
                    )
                ),
            )
        )
    results = [outcome.result for outcome in _outcomes]
    notices = [outcome.notice for outcome in _outcomes]

    _round_query_records: list[dict[str, Any]] = []
    _source_records_by_query: dict[int, list[dict[str, Any]]] = {}
    _citation_records_by_query: dict[int, list[dict[str, Any]]] = {}
    _existing_citation_ids = {
        str(record.get("citation_id", ""))
        for record in s.get("provider_citation_records", [])
    }
    _search_provider_label = type(providers.search).__name__
    s.setdefault("query_records", [])
    s.setdefault("source_records", {})
    s.setdefault("provider_citation_records", [])
    s.setdefault("evidence_ledger", [])
    s.setdefault("answer_evidence_bindings", [])

    for _qi, q in enumerate(new_q):
        r = results[_qi]
        query_id = _query_id_for(
            s,
            round_index=s["round"],
            query_index=offset + _qi,
            query=q,
        )
        query_record = {
            "query_id": query_id,
            "round": s["round"],
            "query_index": offset + _qi,
            "query": q,
            "domain_filter": _query_domain_filters[_qi] or [],
            "provider": _search_provider_label,
        }
        _round_query_records.append(query_record)
        s["query_records"].append(query_record)

        _all_source_urls = [src.url for src in r.sources if src.url]
        tier_explanations = _tier_explanations_for_urls(
            _all_source_urls,
            strategies,
        )
        source_records, citation_records = normalize_source_provenance(
            r,
            query_id=query_id,
            provider=_search_provider_label,
            tier_explanations=tier_explanations,
        )
        _source_records_by_query[_qi] = source_records
        _citation_records_by_query[_qi] = citation_records

        source_registry: dict[str, dict[str, Any]] = s["source_records"]
        for source_record in source_records:
            sid = source_record["source_id"]
            if sid in source_registry:
                continue
            source_registry[sid] = source_record
            _append_forensic_event(
                s,
                settings,
                event="source_record",
                node="search",
                payload=source_record,
            )

        for citation_record in citation_records:
            citation_id = str(citation_record.get("citation_id", ""))
            if citation_id and citation_id not in _existing_citation_ids:
                _existing_citation_ids.add(citation_id)
                s["provider_citation_records"].append(citation_record)
            _append_forensic_event(
                s,
                settings,
                event="provider_citation_record",
                node="search",
                payload=citation_record,
            )

        _append_forensic_event(
            s,
            settings,
            event="query_record",
            node="search",
            payload={
                **query_record,
                "source_ids": [record["source_id"] for record in source_records],
                "citation_ids": [record["citation_id"] for record in citation_records],
            },
        )

    _query_details: list[dict[str, Any]] = []
    if _collect_query_details:
        for _qi, q in enumerate(new_q):
            r = results[_qi]
            _detail: dict[str, Any] = {
                "query_id": _round_query_records[_qi]["query_id"],
                "round": s["round"],
                "query_index": offset + _qi,
                "query": q,
                "domain_filter": _query_domain_filters[_qi] or [],
                "provider_notice": notices[_qi] or "",
                "answer_length": len(r.answer),
                "citation_count": len(r.citation_urls),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            if r.citation_urls:
                _detail["urls"] = [normalize_url(u) for u in r.citation_urls[:5]]
            _query_details.append(_detail)

    _search_fallbacks = sum(1 for n in notices if n)
    # "Empty" means the query yielded nothing usable -- neither a synthesized
    # answer NOR citable sources. A source-only result (no prose but real
    # sources) is ingested as evidence, so counting it here would emit a
    # misleading "no results" notice for a query that did return sources. This
    # mirrors the source-only ingestion gate (a result is usable if it carries
    # an answer or sources).
    _empty_without_notice = sum(
        1 for res, n in zip(results, notices)
        if not res.answer and not res.sources and not n
    )
    if _search_fallbacks:
        emit_progress(
            s,
            t(
                s,
                "search_failed_n_of_m",
                failed=_search_fallbacks,
                total=len(new_q),
            ),
            severity="warning",
        )
    if _empty_without_notice:
        emit_progress(
            s,
            t(
                s,
                "search_empty_n_of_m",
                empty=_empty_without_notice,
                total=len(new_q),
            ),
        )

    # Aggregate token usage from Sonar searches
    _search_prompt_tokens = 0
    _search_completion_tokens = 0
    for r in results:
        _search_prompt_tokens += r.prompt_tokens
        _search_completion_tokens += r.completion_tokens

    # Phase 1: structured claim extraction for later consolidation.
    tuning = settings.report_tuning
    _claim_inputs: list[tuple[int, str, list[str], list[ProviderCitationRef]]] = []
    for _qi, r in enumerate(results):
        if r.answer:
            claim_text = _claim_extraction_text(
                r.answer,
                _citation_records_by_query.get(_qi, []),
                max_sources=tuning.claim_citation_cap,
            )
            claim_citations = [
                normalize_url(str(record.get("canonical_url", "") or ""))
                for record in _citation_records_by_query.get(_qi, [])
                if normalize_url(str(record.get("canonical_url", "") or ""))
            ] or list(r.citation_urls)
            _claim_inputs.append((
                _qi,
                claim_text,
                claim_citations,
                _claim_provider_refs(_citation_records_by_query.get(_qi, [])),
            ))

    _claim_results: dict[int, tuple[list[dict[str, Any]], int, int]] = {}
    _claim_metadata: dict[int, dict[str, Any]] = {}
    _claim_fallbacks = 0
    _claim_warnings: dict[int, str] = {}
    _claim_retry_notices: list[dict[str, Any]] = []
    if _claim_inputs:
        # Resolve claim_extract once per round (not per hit) so the model/effort
        # routing is uniform with every other node and emits a single visible
        # resolution event, then thread the result into each worker. The routing
        # kwargs are a best-effort hint: only forward them to a strategy whose
        # extract() accepts them, so a custom strategy on the older signature
        # keeps working (Baukasten backward-compat).
        _ce_model, _ce_effort = _resolve_node_llm(s, settings, providers, "claim_extract")
        _ce_routing: dict[str, Any] = (
            {"model": _ce_model, "reasoning_effort": _ce_effort}
            if _claim_extract_accepts_routing(strategies.claim_extraction)
            else {}
        )

        def _do_claim_extract(
            item: tuple[int, str, list[str], list[ProviderCitationRef]],
        ) -> tuple[
            int,
            tuple[list[dict[str, Any]], int, int],
            str | None,
            dict[str, Any],
            list[dict[str, Any]],
        ]:
            idx, text, citations, provider_refs = item
            with _provider_retry_progress_context(
                providers.llm,
                s,
                operation_label=t(s, "retry_operation_claim_extraction"),
                testing_mode=settings.testing_mode,
            ):
                result_tuple = strategies.claim_extraction.extract(
                    text,
                    citations,
                    s.get("question", ""),
                    deadline=s["deadline"],
                    provider_refs=provider_refs,
                    text_char_limit=tuning.claim_input_char_limit,
                    citation_cap=tuning.claim_citation_cap,
                    max_claims=tuning.claim_max_items,
                    source_url_limit=tuning.claim_source_url_cap,
                    **_ce_routing,
                )
            metadata_consumer = getattr(
                strategies.claim_extraction,
                "consume_extraction_metadata",
                None,
            )
            metadata = metadata_consumer() if callable(metadata_consumer) else {}
            return (
                idx,
                result_tuple,
                _consume_nonfatal_notice(strategies.claim_extraction),
                metadata if isinstance(metadata, dict) else {},
                _consume_retry_notices(providers.llm),
            )

        with ThreadPoolExecutor(max_workers=min(len(_claim_inputs), _n_workers)) as ex:
            for idx, result_tuple, warning, metadata, retry_notices in ex.map(
                _do_claim_extract,
                _claim_inputs,
            ):
                _claim_results[idx] = result_tuple
                _claim_retry_notices.extend(retry_notices)
                if metadata:
                    _claim_metadata[idx] = metadata
                if warning:
                    _claim_fallbacks += 1
                    _claim_warnings[idx] = warning
    s["_claim_extraction_attempts_total"] = (
        int(s.get("_claim_extraction_attempts_total", 0) or 0)
        + len(_claim_inputs)
    )
    s["_claim_extraction_failures_total"] = (
        int(s.get("_claim_extraction_failures_total", 0) or 0)
        + _claim_fallbacks
    )

    _emit_retry_progress(
        _claim_retry_notices,
        operation_label=t(s, "retry_operation_claim_extraction"),
    )
    if _claim_fallbacks:
        emit_progress(
            s,
            t(
                s,
                "claim_extract_failed",
                failed=_claim_fallbacks,
                total=len(_claim_inputs),
            ),
            severity="warning",
        )
        _all_claim_inputs_failed = _claim_inputs and _claim_fallbacks == len(_claim_inputs)
        _record_algorithm_failure(
            s,
            settings,
            node="search",
            phase="claim_extraction",
            reason=(
                "round_all_sources_failed"
                if _all_claim_inputs_failed
                else "round_some_sources_failed"
            ),
            message=(
                "All claim-extraction calls in this search round failed; "
                "the round produced no structured claim path."
                if _all_claim_inputs_failed
                else "Some claim-extraction calls in this search round failed; "
                "the report must not be marked as fully clean."
            ),
            blocking=False,
            details={
                "failed": _claim_fallbacks,
                "total": len(_claim_inputs),
                "claim_notice_samples": list(_claim_warnings.values())[:3],
            },
        )

    _claim_valid_empty = sum(
        1
        for idx, (claims, _, _) in _claim_results.items()
        if idx not in _claim_warnings and not claims
    )
    if _claim_valid_empty:
        log.warning(
            "TRACE search: %d/%d claim extractions returned no structured claims despite search context",
            _claim_valid_empty,
            len(_claim_inputs),
        )
        emit_progress(
            s,
            t(
                s,
                "claim_extract_empty",
                empty=_claim_valid_empty,
                total=len(_claim_inputs),
            ),
        )

    _claim_prompt_tokens = 0
    _claim_completion_tokens = 0
    for _, pt, ct in _claim_results.values():
        _claim_prompt_tokens += pt
        _claim_completion_tokens += ct
    _claim_mode_counts: dict[str, int] = {}
    _claim_unknown_provider_ref_count = 0
    _claim_unbound_count = 0
    for metadata in _claim_metadata.values():
        mode = str(metadata.get("claim_extraction_mode") or "unknown")
        _claim_mode_counts[mode] = _claim_mode_counts.get(mode, 0) + 1
        _claim_unknown_provider_ref_count += int(
            metadata.get("unknown_provider_ref_count", 0) or 0
        )
        _claim_unbound_count += int(metadata.get("unbound_claim_count", 0) or 0)

    # Surface claim-binding health once per round when claims could not be
    # bound to a source. This is the visible counterpart to the forensic
    # `_claim_unbound_claims` / `_claim_unknown_provider_refs` markers, and it
    # explains a later `needs_review` / `source_context_only` contract
    # downgrade while the report is still being built.
    if _claim_unbound_count or _claim_unknown_provider_ref_count:
        emit_progress(
            s,
            t(
                s,
                "search_claim_binding_issues",
                unbound=_claim_unbound_count,
                unknown=_claim_unknown_provider_ref_count,
            ),
            severity="warning",
        )

    # Phase 2: Sequential context assembly (state access, not parallelisable)
    focus_stems = strategies.claim_consolidation.focus_stems_from_question(
        s.get("question", ""))
    if s.get("answer_contract") == "news_briefing":
        focus_stems = set()

    sources_found = 0
    _ledger_dropped_total = 0
    _kept_claim_ids: list[str] = []
    _sources_summary: list[dict[str, Any]] = []
    for _qi, r in enumerate(results):
        # Ingest a result that carries EITHER a synthesized answer OR cited
        # sources. GroundedSearchResult.sources is independent of .answer, so a
        # provider may return citable sources without prose; those become
        # source-context evidence records (claim extraction simply yields none
        # without answer text). Skip only when both are empty.
        if not r.answer and not r.sources:
            continue

        if r.citation_urls:
            # Normalise URLs and collect globally (deduplicated)
            for url in r.citation_urls:
                normalized = normalize_url(url)
                if normalized not in s["all_citations"]:
                    s["all_citations"].append(normalized)
        sources_found += 1

        # Fill claim ledger with structured assertions
        extracted_claims = _claim_results.get(_qi, ([], 0, 0))[0]
        kept_claims = 0
        query_id = _round_query_records[_qi]["query_id"]
        source_id_by_url = {
            record["canonical_url"]: record["source_id"]
            for record in _source_records_by_query.get(_qi, [])
        }
        citation_ids_by_url: dict[str, list[str]] = {}
        evidence_ids_by_url: dict[str, str] = {}
        for record in _citation_records_by_query.get(_qi, []):
            citation_ids_by_url.setdefault(record["canonical_url"], []).append(
                record["citation_id"]
            )
            evidence_ids_by_url[record["canonical_url"]] = evidence_id_for_citation(
                query_id, record
            )
        query_claim_entries: list[dict[str, Any]] = []
        for claim in extracted_claims:
            claim_text = str(claim.get("claim_text", "")).strip()
            if len(claim_text) < 12:
                continue
            if not strategies.claim_consolidation.claim_matches_focus_stems(claim_text, focus_stems):
                continue
            signature = str(claim.get("signature", "")).strip(
            ) or strategies.claim_consolidation.claim_signature(claim_text)
            if not signature:
                continue
            source_urls = [
                normalize_url(u)
                for u in claim.get("source_urls", [])
                if u
            ][: tuning.claim_source_url_cap]
            source_ids = [
                source_id_by_url[url]
                for url in source_urls
                if url in source_id_by_url
            ]
            citation_ids: list[str] = []
            for url in source_urls:
                citation_ids.extend(citation_ids_by_url.get(url, []))
            evidence_ids = [
                evidence_ids_by_url[url]
                for url in source_urls
                if url in evidence_ids_by_url
            ]
            raw_claim_id = make_record_id(
                "raw_claim",
                s.get("_run_id", ""),
                s["round"],
                query_id,
                signature,
                kept_claims,
            )
            entry = {
                "raw_claim_id": raw_claim_id,
                "claim_text": claim_text,
                "evidence_snippet": str(claim.get("evidence_snippet", "")).strip(),
                "claim_type": str(claim.get("claim_type", "fact")),
                "polarity": str(claim.get("polarity", "affirmed")),
                "needs_primary": bool(claim.get("needs_primary", False)),
                "provider_refs": list(claim.get("provider_refs", []) or []),
                "source_urls": source_urls,
                "source_ids": source_ids,
                "citation_ids": citation_ids,
                "evidence_ids": evidence_ids,
                "binding_status": str(claim.get("binding_status", "unbound")),
                "verification_status": "unverified",
                "verification_basis": "",
                "supporting_evidence_ids": [],
                "supporting_domain_count": 0,
                "published_date": str(claim.get("published_date", "unknown")),
                "signature": signature,
                "round": s["round"],
                "query": new_q[_qi] if _qi < len(new_q) else "",
                "query_id": query_id,
            }
            query_claim_entries.append(entry)
            kept_claims += 1
            _kept_claim_ids.append(raw_claim_id)
            _append_forensic_event(
                s,
                settings,
                event="claim_record",
                node="search",
                payload={
                    "raw_claim_id": raw_claim_id,
                    "query_id": query_id,
                    "signature": signature,
                    "claim_text": claim_text,
                    "evidence_snippet": entry["evidence_snippet"],
                    "claim_type": entry["claim_type"],
                    "polarity": entry["polarity"],
                    "needs_primary": entry["needs_primary"],
                    "source_ids": source_ids,
                    "citation_ids": citation_ids,
                    "provider_refs": entry["provider_refs"],
                    "binding_status": entry["binding_status"],
                    "evidence_ids": evidence_ids,
                    "source_urls": source_urls,
                    "published_date": entry["published_date"],
                    "round": s["round"],
                },
            )

        s.setdefault("query_synthesis", {})[query_id] = {
            "query": new_q[_qi] if _qi < len(new_q) else "",
            "round": s["round"],
            "provider_answer": r.answer,
            "citation_urls_by_rank": {
                str(src.rank): src.url
                for src in r.sources
                if src.rank and src.url
            },
        }
        _append_forensic_event(
            s,
            settings,
            event="query_synthesis",
            node="search",
            payload={
                "query_id": query_id,
                **s["query_synthesis"][query_id],
            },
        )

        evidence_records = assemble_evidence_records(
            query_id=query_id,
            query=new_q[_qi] if _qi < len(new_q) else "",
            provider=_search_provider_label,
            source_records=_source_records_by_query.get(_qi, []),
            citation_records=_citation_records_by_query.get(_qi, []),
            claim_entries=query_claim_entries,
        )
        s["evidence_ledger"] = merge_evidence_records(
            s.get("evidence_ledger", []),
            evidence_records,
        )
        for evidence_record in evidence_records:
            _append_forensic_event(
                s,
                settings,
                event="evidence_record",
                node="search",
                payload=evidence_record,
            )

        if _collect_query_details:
            _entry = dict(_query_details[_qi]) if _qi < len(_query_details) else {
                "query": new_q[_qi] if _qi < len(new_q) else "?",
            }
            _entry["claims_extracted"] = len(extracted_claims)
            _entry["claims_kept"] = kept_claims
            _entry["claim_extraction_valid_empty"] = (
                _qi in _claim_results and _qi not in _claim_warnings and not extracted_claims
            )
            if _qi in _claim_metadata:
                _metadata = _claim_metadata[_qi]
                _entry["claim_extraction_mode"] = str(
                    _metadata.get("claim_extraction_mode") or ""
                )
                _entry["claim_extraction_schema"] = str(
                    _metadata.get("claim_extraction_schema") or ""
                )
                _entry["claim_extraction_structured_supported"] = bool(
                    _metadata.get("claim_extraction_structured_supported")
                )
                for _count_key in (
                    "claim_extraction_raw_claim_count",
                    "claim_extraction_normalized_claim_count",
                    "claim_extraction_filtered_claim_count",
                    "unknown_provider_ref_count",
                    "unbound_claim_count",
                ):
                    if _count_key in _metadata:
                        _entry[_count_key] = int(_metadata.get(_count_key, 0) or 0)
            _entry["evidence_record_count"] = len(evidence_records)
            _entry["evidence_context_source_count"] = sum(
                len(record.get("citation_set", []) or [])
                for record in evidence_records
            )
            _entry["source_ids"] = [
                record["source_id"] for record in _source_records_by_query.get(_qi, [])
            ]
            _entry["citation_ids"] = [
                record["citation_id"] for record in _citation_records_by_query.get(_qi, [])
            ]
            if _qi in _claim_warnings:
                _entry["claim_notice"] = _claim_warnings[_qi]
            if extracted_claims:
                _entry["claims_sample"] = [
                    str(c.get("claim_text", "")).strip()
                    for c in extracted_claims[:3]
                    if str(c.get("claim_text", "")).strip()
                ]
            _sources_summary.append(_entry)
            _append_forensic_event(
                s,
                settings,
                event="query_summary",
                node="search",
                payload=_entry,
            )
    claim_ledger = derive_claim_ledger_from_evidence(s.get("evidence_ledger", []) or [])
    if len(claim_ledger) > tuning.claim_ledger_cap:
        _ledger_dropped_total += len(claim_ledger) - tuning.claim_ledger_cap
        claim_ledger = claim_ledger[-tuning.claim_ledger_cap:]

    emit_progress(
        s,
        t(
            s,
            "search_sources_processed",
            n=sources_found,
            citations=len(s["all_citations"]),
            evidence=len(s.get("evidence_ledger", []) or []),
        ),
    )

    # Update source quality and aspect coverage
    tier_counts, quality_score = strategies.source_tiering.quality_from_urls(s["all_citations"])
    s["source_tier_counts"] = tier_counts
    s["source_quality_score"] = quality_score
    consolidated_claims_all = strategies.claim_consolidation.consolidate(claim_ledger)
    consolidated_claims_all = apply_answer_contract_claim_gates(
        consolidated_claims_all,
        answer_contract=s.get("answer_contract", "general"),
    )
    consolidated_claims = strategies.claim_consolidation.materialize(
        consolidated_claims_all,
        max_total=tuning.materialize_max_total,
        max_unverified=tuning.materialize_max_unverified,
    )
    s["consolidated_claims"] = consolidated_claims
    claim_counts, claim_quality, np_total, np_verified = strategies.claim_consolidation.quality_metrics(
        consolidated_claims)
    s["claim_status_counts"] = claim_counts
    s["claim_quality_score"] = claim_quality
    s["claim_needs_primary_total"] = np_total
    s["claim_needs_primary_verified"] = np_verified
    s["evidence_ledger"] = project_claim_verification_to_evidence(
        s.get("evidence_ledger", []),
        consolidated_claims,
    )
    # Verification standing counts read directly from the consolidated claim
    # ledger -- the run no longer maintains a parallel report-evidence-bundle
    # list. The EvidenceLedger plus its projected verification fields is the
    # single source of truth for both the research loop and the final answer.
    _verified_claim_count = sum(
        1 for claim in consolidated_claims if claim.get("status") == "verified"
    )
    _contested_claim_count = sum(
        1 for claim in consolidated_claims if claim.get("status") == "contested"
    )
    _cross_checked_claim_count = sum(
        1 for claim in consolidated_claims
        if claim.get("verification_basis") == "verified_cross_checked"
    )
    _primary_supported_claim_count = sum(
        1 for claim in consolidated_claims
        if claim.get("status") == "verified"
        and str(claim.get("verification_basis", "")).startswith("verified_primary")
    )
    _single_source_verified_count = sum(
        1 for claim in consolidated_claims
        if claim.get("status") == "verified" and _claim_citation_count(claim) < 2
    )
    _report_eligible_count = sum(
        1 for record in s.get("evidence_ledger", []) or []
        if record.get("report_eligible")
    )
    evidence_depth_gap = _evidence_depth_gap(s)
    s["evidence_depth_gap"] = evidence_depth_gap
    s["_evidence_depth_gap_active"] = bool(evidence_depth_gap.get("active"))
    _append_forensic_event(
        s,
        settings,
        event="evidence_verification_projection",
        node="search",
        payload={
            "evidence_count": len(s.get("evidence_ledger", [])),
            "verified_claim_supports": sum(
                1
                for record in s.get("evidence_ledger", [])
                for claim in record.get("claims", []) or []
                if claim.get("verification_status") == "verified"
            ),
            "contested_claim_supports": sum(
                1
                for record in s.get("evidence_ledger", [])
                for claim in record.get("claims", []) or []
                if claim.get("verification_status") == "contested"
            ),
            "unverified_claim_supports": sum(
                1
                for record in s.get("evidence_ledger", [])
                for claim in record.get("claims", []) or []
                if claim.get("verification_status") == "unverified"
            ),
            "supporting_evidence_link_count": sum(
                len(claim.get("supporting_evidence_ids", []) or [])
                for record in s.get("evidence_ledger", [])
                for claim in record.get("claims", []) or []
            ),
        },
    )
    _append_forensic_event(
        s,
        settings,
        event="evidence_selection",
        node="search",
        payload={
            "consolidated_claim_count": len(consolidated_claims),
            "verified_claim_count": _verified_claim_count,
            "primary_supported_claim_count": _primary_supported_claim_count,
            "contested_claim_count": _contested_claim_count,
            "cross_checked_claim_count": _cross_checked_claim_count,
            "single_source_verified_count": _single_source_verified_count,
            "report_eligible_evidence_count": _report_eligible_count,
            "evidence_depth_gap": evidence_depth_gap,
        },
    )
    for claim in consolidated_claims:
        _append_forensic_event(
            s,
            settings,
            event="claim_merge",
            node="search",
            payload={
                "claim_id": claim.get("claim_id", ""),
                "signature": claim.get("signature", ""),
                "member_claim_ids": claim.get("member_claim_ids", []),
                "status": claim.get("status", ""),
                "status_reason": claim.get("status_reason", ""),
                "verification_basis": claim.get("verification_basis", ""),
                "evidence_snippets": claim.get("evidence_snippets", []),
                "support_count": claim.get("support_count", 0),
                "supporting_evidence_ids": claim.get("supporting_evidence_ids", []),
                "supporting_domain_count": claim.get("supporting_domain_count", 0),
                "contradicting_evidence_ids": claim.get("contradicting_evidence_ids", []),
                "contradict_count": claim.get("contradict_count", 0),
                "source_ids": claim.get("source_ids", []),
                "citation_ids": claim.get("citation_ids", []),
                "source_urls": claim.get("source_urls", []),
                "round_first_seen": claim.get("round_first_seen", 0),
                "round_last_updated": claim.get("round_last_updated", 0),
            },
        )

    log.info(
        "TRACE search: round=%d queries=%s sources_found=%d total_citations=%d "
        "evidence_records=%d claims=%d claim_quality=%.2f",
        s["round"], json.dumps(new_q, ensure_ascii=False),
        sources_found, len(s["all_citations"]), len(s.get("evidence_ledger", []) or []),
        len(consolidated_claims), claim_quality,
    )

    # Aspect coverage is estimated directly from the EvidenceLedger: each
    # report-eligible record's claims and snippet, plus the provider
    # synthesis, form the text corpus. There is no separate prunable context
    # channel anymore -- the ledger keeps every record, and the answer-time
    # overview renderer applies the only budget.
    aspect_coverage_blocks = [
        " ".join(
            part
            for part in (
                str(record.get("source_snippet", "") or ""),
                " ".join(
                    str(claim.get("claim_text", "") or "")
                    for claim in record.get("claims", []) or []
                ),
            )
            if part
        )
        for record in s.get("evidence_ledger", []) or []
        if record.get("report_eligible")
    ]
    aspect_coverage_blocks.extend(
        str(synth.get("provider_answer", "") or "")
        for synth in (s.get("query_synthesis", {}) or {}).values()
    )
    uncovered, coverage = strategies.risk_scoring.estimate_aspect_coverage(
        s.get("required_aspects", []),
        aspect_coverage_blocks,
    )
    s["uncovered_aspects"] = uncovered
    s["aspect_coverage"] = coverage
    s["round"] += 1

    emit_progress(
        s,
        t(
            s,
            "search_source_quality",
            primary=int(tier_counts.get("primary", 0) or 0),
            mainstream=int(tier_counts.get("mainstream", 0) or 0),
            stakeholder=int(tier_counts.get("stakeholder", 0) or 0),
            unknown=int(tier_counts.get("unknown", 0) or 0),
            low=int(tier_counts.get("low", 0) or 0),
            quality=f"{quality_score:.2f}",
        ),
    )
    emit_progress(
        s,
        t(
            s,
            "search_quality_summary",
            verified_claims=_verified_claim_count,
            unverified_claims=max(
                0,
                len(consolidated_claims)
                - _verified_claim_count
                - _contested_claim_count,
            ),
            cross_checked_claims=_cross_checked_claim_count,
            coverage=int(coverage * 100),
        ),
    )
    _search_score_snapshot = append_score_snapshot(s, phase="search")
    _append_forensic_event(
        s,
        settings,
        event="score_snapshot",
        node="search",
        payload=_search_score_snapshot,
    )

    # Aggregate token usage from search provider + claim extraction.
    s["total_prompt_tokens"] += _search_prompt_tokens + _claim_prompt_tokens
    s["total_completion_tokens"] += (
        _search_completion_tokens + _claim_completion_tokens
    )

    append_iteration_log(s, {
        "node": "search",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"] - 1,
        "worker_count": _n_workers,
        "queries_executed": len(new_q),
        "target_query_count": _batch,
        "queries": new_q,
        "query_record_ids": [record["query_id"] for record in _round_query_records],
        "search_parameters": {
            "search_context_size": search_kwargs.get("search_context_size") or "",
            "recency_filter": search_kwargs.get("recency_filter") or "",
            "language_filter": search_kwargs.get("language_filter", []),
            "search_mode": search_kwargs.get("search_mode") or "",
            "return_related": bool(search_kwargs.get("return_related")),
            "supported_parameters": sorted(search_capabilities.supported_parameters),
        },
        "search_fallbacks": _search_fallbacks,
        "claim_fallbacks": _claim_fallbacks,
        "_claim_extraction_fallback": _claim_fallbacks > 0,
        "claim_extraction_modes": _claim_mode_counts,
        "claim_extraction_attempts_total": s.get("_claim_extraction_attempts_total", 0),
        "claim_extraction_failures_total": s.get("_claim_extraction_failures_total", 0),
        "unknown_provider_ref_count": _claim_unknown_provider_ref_count,
        "unbound_claim_count": _claim_unbound_count,
        "_claim_unknown_provider_refs": _claim_unknown_provider_ref_count > 0,
        "_claim_unbound_claims": _claim_unbound_count > 0,
        "claim_valid_empty": _claim_valid_empty,
        "_claim_extraction_empty": _claim_valid_empty > 0,
        "algorithm_failure_count": len(s.get("algorithm_failures", []) or []),
        "blocking_algorithm_failure_count": len(_blocking_algorithm_failures(s)),
        "algorithm_failures": (s.get("algorithm_failures", []) or [])[-10:],
        "sources_found": sources_found,
        "total_citations": len(s["all_citations"]),
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "source_record_ids": [
            record["source_id"]
            for records in _source_records_by_query.values()
            for record in records
        ],
        "provider_citation_record_ids": [
            record["citation_id"]
            for records in _citation_records_by_query.values()
            for record in records
        ],
        "evidence_record_count": len(s.get("evidence_ledger", [])),
        "report_eligible_evidence_count": _report_eligible_count,
        "verified_claim_count": _verified_claim_count,
        "cross_checked_claim_count": _cross_checked_claim_count,
        "primary_supported_claim_count": _primary_supported_claim_count,
        "single_source_verified_count": _single_source_verified_count,
        "evidence_depth_gap": evidence_depth_gap,
        "claim_ledger_size": len(claim_ledger),
        "claim_record_ids": _kept_claim_ids,
        "claim_ledger_dropped": _ledger_dropped_total,
        "consolidated_claims_count": len(s.get("consolidated_claims", [])),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "aspect_coverage": s.get("aspect_coverage", 0.0),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "sources_summary": _sources_summary,
    }, testing_mode=settings.testing_mode)
    return s


@dataclass(slots=True)
class ConfidenceGuardrailResult:
    """Outcome of the post-evaluation confidence guardrails.

    Attributes:
        confidence: Clamped confidence after all guardrails ran.
        gap_suggestion: First gap text proposed by a guardrail, or
            ``None`` if no guardrail proposed one. The caller only
            applies it when ``s["gaps"]`` is still empty.
        reasons: Human-readable trace of each guardrail that actually
            changed the confidence value. Used for telemetry and log
            diagnostics.
    """

    confidence: int
    gap_suggestion: str | None
    reasons: list[str]


def apply_answer_contract_claim_gates(
    consolidated_claims: list[dict[str, Any]],
    *,
    answer_contract: str,
) -> list[dict[str, Any]]:
    """Downgrade claims that are not strong enough for the answer contract."""
    if answer_contract != "news_briefing":
        return consolidated_claims

    gated: list[dict[str, Any]] = []
    for claim in consolidated_claims:
        updated = dict(claim)
        if updated.get("status") == "verified" and _claim_outside_news_window(updated):
            updated["status"] = "unverified"
            updated["status_reason"] = "outside requested news window"
            updated["verification_basis"] = "news_briefing_out_of_window"
        gated.append(updated)
    return gated


_NEWS_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
    "januar": 1,
    "februar": 2,
    "märz": 3,
    "maerz": 3,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "oktober": 10,
    "dezember": 12,
}


def _extract_claim_dates(text: str) -> list[dt.date]:
    dates: list[dt.date] = []
    seen: set[dt.date] = set()

    def add_date(year: int, month: int, day: int) -> None:
        try:
            value = dt.date(year, month, day)
        except ValueError:
            return
        if value not in seen:
            seen.add(value)
            dates.append(value)

    for match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text or ""):
        add_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    for match in re.finditer(
        r"\b([A-Za-zäöüÄÖÜ]+)\s+(\d{1,2}),?\s+(20\d{2})\b",
        text or "",
        re.IGNORECASE,
    ):
        month = _NEWS_MONTHS.get(match.group(1).lower())
        if month:
            add_date(int(match.group(3)), month, int(match.group(2)))
    for match in re.finditer(
        r"\b(\d{1,2})\.\s*([A-Za-zäöüÄÖÜ]+)\s+(20\d{2})\b",
        text or "",
        re.IGNORECASE,
    ):
        month = _NEWS_MONTHS.get(match.group(2).lower())
        if month:
            add_date(int(match.group(3)), month, int(match.group(1)))
    return dates


def _claim_outside_news_window(claim: dict[str, Any]) -> bool:
    dates = _extract_claim_dates(
        " ".join(
            [
                str(claim.get("published_date", "") or ""),
                str(claim.get("claim_text", "") or ""),
            ]
        )
    )
    if not dates:
        return False
    today_date = dt.date.today()
    start = today_date - dt.timedelta(days=7)
    return all(date < start or date > today_date for date in dates)


def apply_confidence_guardrails(
    conf: int,
    *,
    has_citations: bool,
    primary_n: int,
    mainstream_n: int,
    low_n: int,
    uncovered_aspects: list[str],
    contested_claims: int,
    needs_primary: bool,
    existing_gap: str,
    has_evidence_records: bool = False,
    has_claims: bool = True,
    has_report_bundles: bool = True,
) -> ConfidenceGuardrailResult:
    """Couple confidence to source quality and aspect coverage.

    Centralizes the five LLM-independent guardrails that previously
    lived inline in ``evaluate``. The function is pure: it neither
    reads nor writes state, which makes the interaction between the
    individual clamps explicit and unit-testable.

    The gap suggestion follows the same first-writer-wins semantics as
    the legacy inline code: if the caller already has a gap message
    stored, no suggestion is emitted; otherwise the first guardrail
    whose condition fires proposes the gap text.

    Args:
        conf: Baseline confidence after LLM evaluation and prior
            stop-criteria strategies.
        has_citations: ``True`` when at least one citation URL is
            present in state.
        has_evidence_records: ``True`` when the search node created any
            structured EvidenceRecord rows.
        has_claims: ``True`` when claim consolidation produced at
            least one answer-facing claim.
        has_report_bundles: ``True`` when verified/contested report
            evidence is available for final answer grounding.
        primary_n: Count of primary-tier citations.
        mainstream_n: Count of mainstream-tier citations.
        low_n: Count of low-tier citations.
        uncovered_aspects: Required aspects still uncovered after the
            latest round.
        contested_claims: Count of consolidated claims with
            ``status == "contested"``.
        needs_primary: ``True`` when the question keywords indicate
            that a primary source is required.
        existing_gap: Current ``state["gaps"]`` value. When truthy, no
            gap suggestion is emitted.

    Returns:
        ConfidenceGuardrailResult: Clamped confidence, optional gap
        suggestion, and a list of reason tags for each guardrail that
        changed the confidence.
    """
    reasons: list[str] = []
    gap_suggestion: str | None = None

    def propose(text: str) -> None:
        nonlocal gap_suggestion
        if not existing_gap and gap_suggestion is None:
            gap_suggestion = text

    if not has_citations:
        new_conf = min(conf, 6)
        if new_conf != conf:
            reasons.append(f"no_citations:conf {conf}->{new_conf}")
            conf = new_conf
        propose("Keine belastbaren Quellen gefunden.")
    if has_citations and has_evidence_records and not has_claims:
        new_conf = min(conf, 5)
        if new_conf != conf:
            reasons.append(f"no_structured_evidence:conf {conf}->{new_conf}")
            conf = new_conf
        propose("Evidence-Pipeline konnte keine strukturierten Claims aus den Suchergebnissen ableiten.")
    elif has_citations and has_claims and not has_report_bundles:
        new_conf = min(conf, 6)
        if new_conf != conf:
            reasons.append(f"no_report_bundles:conf {conf}->{new_conf}")
            conf = new_conf
        propose("Claims vorhanden, aber keine verifizierten Report-Evidence-Bundles.")
    if low_n > (primary_n + mainstream_n) and conf > 7:
        reasons.append(f"low_quality_majority:conf {conf}->7")
        conf = 7
    if needs_primary and primary_n == 0 and conf > 8:
        reasons.append(f"needs_primary_missing:conf {conf}->8")
        conf = 8
        propose("Zentrale Zahlen/Regelungen nicht mit Primaerquelle belegt.")
    if len(uncovered_aspects) > 0 and conf > 8:
        reasons.append(f"aspects_uncovered:conf {conf}->8")
        conf = 8
        propose(f"Pflichtaspekte offen: {', '.join(uncovered_aspects[:2])}")
    if contested_claims >= 2 and conf > 7:
        reasons.append(f"contested_claims>=2:conf {conf}->7")
        conf = 7
        propose("Mehrere zentrale Aussagen sind zwischen Quellen umstritten.")

    return ConfidenceGuardrailResult(
        confidence=conf,
        gap_suggestion=gap_suggestion,
        reasons=reasons,
    )


# ======================================================================= #
# 4. evaluate
# ======================================================================= #


def evaluate(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Evaluate evidence quality, stop criteria, and remaining gaps.

    Args:
        s: Mutable AgentState-compatible dict. Reads accumulated
        evidence, claims, and prior confidence; writes quality
            metrics, stop decisions, and gap information for later rounds.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for source quality, claim
            consolidation, risk coverage, and stop criteria.
        settings: Agent behavior settings controlling risk scoring and
            stopping thresholds.

    Returns:
        The mutated state dict with refreshed quality metrics and stop
        status.

    Raises:
        AgentRateLimited: Propagated when the evaluation model hard-fails
            on upstream rate limiting.

    Example:
        >>> evaluate(state, providers=providers, strategies=strategies, settings=settings)
        {'final_confidence': 8, 'done': True, ...}
    """
    check_cancel_event(s)
    emit_progress(
        s,
        t(s, "evaluate_start", round=s["round"], max_rounds=settings.max_rounds),
    )
    _t0 = time.monotonic()
    _stagnation_detected = False
    _evaluate_fallback: dict[str, Any] = {}
    _confidence_parsed = True
    if s.get("done"):
        s["_stop_reason"] = s.get("_stop_reason") or "already_done"
        append_iteration_log(s, {
            "node": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "confidence": s.get("final_confidence", 0),
            "skipped": "already_done",
            "_stop_reason": s.get("_stop_reason", "already_done"),
        }, testing_mode=settings.testing_mode)
        return s
    try:
        _check_deadline(s["deadline"])
    except AgentTimeout:
        s["done"] = True
        s["_stop_reason"] = "deadline_exceeded"
        append_iteration_log(s, {
            "node": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "confidence": s.get("final_confidence", 0),
            "skipped": "deadline_exceeded",
            "_stop_reason": s.get("_stop_reason", "deadline_exceeded"),
        }, testing_mode=settings.testing_mode)
        return s

    # Read metrics that the search node already computed for this round.
    # Re-running quality_from_urls / consolidate / materialize / quality_metrics
    # here would duplicate work without changing the result: nothing between
    # the end of search and the start of evaluate mutates citations, the
    # claim_ledger, or the materialize tuning.
    tuning = settings.report_tuning
    tier_counts: dict[str, int] = s.get("source_tier_counts", {}) or {}
    quality_score = float(s.get("source_quality_score", 0.0) or 0.0)
    consolidated_claims = s.get("consolidated_claims", []) or []
    claim_counts: dict[str, int] = s.get("claim_status_counts", {}) or {}
    claim_quality = float(s.get("claim_quality_score", 0.0) or 0.0)
    claim_np_total = int(s.get("claim_needs_primary_total", 0) or 0)
    claim_np_verified = int(s.get("claim_needs_primary_verified", 0) or 0)
    evidence_depth_gap = _evidence_depth_gap(s)
    s["evidence_depth_gap"] = evidence_depth_gap
    s["_evidence_depth_gap_active"] = bool(evidence_depth_gap.get("active"))
    report_eligible_evidence_count = sum(
        1
        for record in s.get("evidence_ledger", []) or []
        if record.get("report_eligible")
    )
    evaluate_evidence_overview = render_evidence_ledger_overview(
        s.get("evidence_ledger", []) or [],
        max_total_chars=min(24000, tuning.prompt_evidence_total_char_budget),
        max_record_chars=tuning.prompt_evidence_record_char_limit,
        query_synthesis=s.get("query_synthesis", {}),
    ).markdown

    evaluate_model, evaluate_effort = _resolve_node_llm(s, settings, providers, "evaluate")

    # Previous-round context block (Issue 1 — confidence stability):
    # Surfaces the evaluator's own prior CONFIDENCE / GAPS to the LLM so the
    # next bewertung is a delta against the last run, not an isolated fresh
    # rating. The bestehende EVALUATE_FORMAT_SUFFIX trains the model to keep
    # the value monotone unless new contradictions or competing events
    # surface; this block provides the comparison anchor.
    previous_round_hint = ""
    if s["round"] >= 1:
        _prev_conf_for_hint = int(s.get("final_confidence", 0) or 0)
        if _prev_conf_for_hint > 0:
            _prev_gaps_for_hint = (s.get("gaps", "") or "").strip() or "Keine"
            _new_citations_for_hint = max(
                0,
                len(s.get("all_citations", []))
                - int(s.get("prev_citation_count", 0) or 0),
            )
            previous_round_hint = (
                "\n\nVORRUNDEN-KONTEXT:\n"
                f"VORRUNDE (Runde {s['round'] - 1}): "
                f"CONFIDENCE={_prev_conf_for_hint}, GAPS=\"{_prev_gaps_for_hint}\".\n"
                f"In dieser Runde sind {_new_citations_for_hint} neue Quellen "
                f"hinzugekommen.\n"
            )

    # Hint for negative evidence
    negative_evidence_hint = ""
    if s["round"] >= 2:
        _prev_conf = s.get("final_confidence", 0)
        _n_citations = len(s.get("all_citations", []))
        negative_evidence_hint = (
            "\n\nWICHTIG — NEGATIVE EVIDENZ:\n"
            f"Es wurden bereits {s['round']} Suchrunden mit {len(s['queries'])} Queries durchgefuehrt "
            f"und {_n_citations} Quellen gesammelt.\n"
        )
        if _prev_conf > 0 and _prev_conf <= 4:
            negative_evidence_hint += (
                f"Die Confidence war in der vorherigen Runde ebenfalls nur {_prev_conf}/10.\n"
                f"Wenn sich trotz {_n_citations} durchsuchter Quellen nichts Substantielles "
                f"verbessert hat, ist das ein STARKES Signal: "
                f"Die Praemisse der Frage ist wahrscheinlich FALSCH.\n"
                f"Setze in diesem Fall CONFIDENCE auf 7-9 — 'Es existiert nicht' ist eine "
                f"hochkonfidente Erkenntnis nach umfangreicher Recherche.\n"
            )
        negative_evidence_hint += (
            "Wenn die Recherche KONSISTENT keine Belege fuer die Behauptung/Annahme in der Frage findet, "
            "dann IST das ein Ergebnis. 'Es existiert nicht' oder 'Die Annahme ist falsch' "
            "sind valide Antworten mit HOHER Confidence (7-9).\n"
            "Bewerte also nicht nur ob du gefunden hast WAS gefragt wurde, "
            "sondern auch ob du genug gesucht hast um sicher zu sagen dass es NICHT existiert.\n"
        )

    # --- LLM call: evaluate information quality ---
    _eval_raw = ""
    quality_hint = (
        "\nQUELLENQUALITAET:\n"
        f"- primary={tier_counts.get('primary', 0)}, "
        f"mainstream={tier_counts.get('mainstream', 0)}, "
        f"stakeholder={tier_counts.get('stakeholder', 0)}, "
        f"unknown={tier_counts.get('unknown', 0)}, "
        f"low={tier_counts.get('low', 0)}.\n"
        f"- Gesamt-Qualitaetsscore: {quality_score:.2f} (0-1).\n"
        "- Wenn zentrale Aussagen nur durch Stakeholder- oder Low-Quality-Quellen belegt sind, "
        "reduziere CONFIDENCE und setze GAPS entsprechend.\n"
        "- Trenne strikt zwischen neutralem Fakt und Akteursbehauptung "
        "(z.B. Parteiverband, Branchenverband, Lobbyorganisation).\n\n"
    )
    aspect_hint = ""
    if s.get("required_aspects"):
        aspect_hint = (
            "ASPEKTABDECKUNG:\n"
            f"- Pflichtaspekte: {json.dumps(s['required_aspects'], ensure_ascii=False)}\n"
            f"- Noch offen: {json.dumps(s['uncovered_aspects'], ensure_ascii=False)}\n"
            f"- Coverage: {int(s['aspect_coverage'] * 100)}%.\n"
            "- Wenn Pflichtaspekte offen sind, kann STATUS nicht SUFFICIENT sein.\n\n"
        )
    claim_hint = (
        "CLAIM-LEDGER:\n"
        f"- verified={claim_counts.get('verified', 0)}, "
        f"contested={claim_counts.get('contested', 0)}, "
        f"unverified={claim_counts.get('unverified', 0)}.\n"
        f"- Claim-Qualitaetsscore: {claim_quality:.2f} (0-1).\n"
        f"- Primaerpflichtige Claims verifiziert: {claim_np_verified}/{claim_np_total}.\n"
        "- Falls zentrale Claims contested/unverified sind, reduziere CONFIDENCE und "
        "setze STATUS auf INSUFFICIENT.\n"
        "- Nutze die folgende konsolidierte Claim-Liste fuer die Bewertung:\n"
        + strategies.claim_consolidation.claims_prompt_view(
            consolidated_claims,
            max_items=tuning.answer_claim_prompt_items,
        )
        + "\n\n"
    )
    evidence_depth_hint = ""
    if evidence_depth_gap.get("active"):
        evidence_depth_hint = (
            "EVIDENCE-DEPTH-GAP:\n"
            f"- Diagnose: {evidence_depth_gap.get('reason', '')}.\n"
            f"- Verified Claims: {evidence_depth_gap.get('verified_count', 0)}, "
            f"Cross-checked Claims: {evidence_depth_gap.get('cross_checked_count', 0)}, "
            f"Single-source verified: {evidence_depth_gap.get('single_source_verified_count', 0)}.\n"
            "- Wenn zentrale Aussagen ueberwiegend single-source sind, ist STATUS nur "
            "SUFFICIENT, wenn die verbleibende Unsicherheit explizit gering ist. Sonst setze "
            "GAPS auf unabhaengige Cross-Checks oder Primaerquellen.\n\n"
        )
    try:
        eval_prompt = (
            f"Heutiges Datum: {today()}\n\n"
            f"Bewerte ob die recherchierten Informationen ausreichen, "
            f"um die Frage vollstaendig und korrekt zu beantworten.\n\n"
            f"ZEITLICHE KONSISTENZ:\n"
            f"- Wenn die Frage relative Zeitangaben enthaelt ('vor kurzem', 'neulich', 'letztens'), "
            f"pruefe ob die gefundenen Ereignisse zeitlich zum heutigen Datum ({today()}) passen.\n"
            f"- Ein Ereignis von vor 12+ Monaten ist NICHT 'vor kurzem'.\n"
            f"- Wenn die gefundenen Ereignisse zeitlich nicht passen, setze GAPS "
            f"auf 'Zeitlich aktuelleres Ereignis nicht gefunden' und CONFIDENCE maximal 5.\n\n"
            f"MEHRERE PASSENDE EREIGNISSE:\n"
            f"- Wenn mehrere Ereignisse auf die Beschreibung passen koennten, "
            f"setze GAPS auf 'Moeglicherweise ein anderes/aktuelleres Ereignis gemeint' "
            f"und reduziere CONFIDENCE.\n\n"
            f"KONKURRIERENDE ERKLAERUNGEN:\n"
            f"- Wenn du in den Recherche-Ergebnissen VERSCHIEDENE moegliche Ereignisse/Antworten findest, "
            f"die auf die Frage passen koennten, liste sie auf.\n"
            f"- Antworte in der Zeile COMPETING_EVENTS mit einer kurzen Auflistung: "
            f"'Event A (Datum) vs Event B (Datum)' oder 'Keine'.\n\n"
            + quality_hint
            + aspect_hint
            + claim_hint
            + evidence_depth_hint
            + f"Frage: {s['question']}\n\n"
            + "Recherche-Ergebnisse (Evidenz-Uebersicht):\n"
            + (evaluate_evidence_overview or "(noch keine belegfaehige Evidenz vorhanden)")
            + "\n"
            + previous_round_hint
            + negative_evidence_hint
            + EVALUATE_FORMAT_SUFFIX
        )
        with _provider_retry_progress_context(
            providers.llm,
            s,
            operation_label=t(s, "retry_operation_evaluate"),
            testing_mode=settings.testing_mode,
        ):
            a = providers.llm.complete(
                eval_prompt,
                deadline=s["deadline"],
                model=evaluate_model,
                reasoning_effort=evaluate_effort,
                state=s,
            )
        _eval_raw = a

        # --- Parse base values from LLM response ---
        m_conf = re.search(r"CONFIDENCE:\s*(\d+)", a)
        if m_conf:
            conf = int(m_conf.group(1))
        else:
            conf = 5
            _confidence_parsed = False
            log.warning(
                "Evaluate-Parse-Warnung: CONFIDENCE-Feld fehlt in LLM-Antwort "
                "(round=%d, model=%s) -> Default 5",
                s["round"], evaluate_model,
            )
            emit_progress(s, t(s, "evaluate_confidence_missing"), severity="warning")

        m_gaps = re.search(r"GAPS:\s*(.+?)(?:\n|$)", a)
        gaps = m_gaps.group(1).strip() if m_gaps else ""
        s["gaps"] = "" if is_none_value(gaps) else gaps
        if evidence_depth_gap.get("active") and not s.get("gaps"):
            s["gaps"] = str(evidence_depth_gap.get("gap", "") or "")

        # --- Apply heuristics ---
        conf = strategies.stop_criteria.check_contradictions(s, a, conf)
        conf = strategies.stop_criteria.extract_competing_events(s, a, conf)
        conf = strategies.stop_criteria.extract_evidence_scores(s, a, conf)

    except AgentRateLimited:
        raise
    except (
        OpenAIError,
        AgentTimeout,
        AnthropicAPIError,
        AzureOpenAIAPIError,
        BedrockAPIError,
    ) as exc:
        # No fail-open: on evaluate error stay conservative.
        _exc_label = type(exc).__name__
        log.warning(
            "Evaluate-Fallback aktiviert (%s, round=%d, model=%s): %s",
            _exc_label, s["round"], evaluate_model, exc,
        )
        emit_progress(s, t(s, "evaluate_failed", label=_exc_label), severity="warning")
        conf = min(max(s.get("final_confidence", 0), 5), settings.confidence_stop - 2)
        _confidence_parsed = False
        if not s.get("gaps"):
            s["gaps"] = "Automatische Qualitaetsbewertung unvollstaendig; Antwort vorsichtig formulieren."
        _evaluate_fallback = {
            "fallback": "evaluate_default",
            "fallback_reason": _exc_label,
            "fallback_message": str(exc)[:300],
        }

    # Guardrails: couple confidence to source quality and aspect coverage.
    q_lower = s["question"].lower()
    needs_primary = bool(
        re.search(r"\b(prozent|mrd|mio|euro|gesetz|regel|politik|beitrag|kosten)\b", q_lower))
    primary_n = int(tier_counts.get("primary", 0))
    mainstream_n = int(tier_counts.get("mainstream", 0))
    low_n = int(tier_counts.get("low", 0))
    verified_claims = int(claim_counts.get("verified", 0))
    contested_claims = int(claim_counts.get("contested", 0))
    unverified_claims = int(claim_counts.get("unverified", 0))

    _guardrail_result = apply_confidence_guardrails(
        conf,
        has_citations=bool(s.get("all_citations")),
        has_evidence_records=bool(s.get("evidence_ledger")),
        has_claims=bool(consolidated_claims),
        has_report_bundles=any(
            claim.get("status") in {"verified", "contested"}
            for claim in consolidated_claims
        ),
        primary_n=primary_n,
        mainstream_n=mainstream_n,
        low_n=low_n,
        uncovered_aspects=list(s.get("uncovered_aspects", [])),
        contested_claims=contested_claims,
        needs_primary=needs_primary,
        existing_gap=s.get("gaps", "") or "",
    )
    conf = _guardrail_result.confidence
    if _guardrail_result.gap_suggestion and not s.get("gaps"):
        s["gaps"] = _guardrail_result.gap_suggestion
    if _guardrail_result.reasons:
        log.info(
            "TRACE evaluate: guardrail_reasons=%s",
            _guardrail_result.reasons,
        )
    if (
        bool(s.get("all_citations"))
        and bool(s.get("evidence_ledger"))
        and not consolidated_claims
    ):
        s["evidence_sufficiency"] = min(int(s.get("evidence_sufficiency", 0) or 0), 3)

    # --- Post-LLM stop heuristics ---
    _prev_conf = s.get("final_confidence", 0)
    _n_citations = len(s.get("all_citations", []))

    _falsification_just_triggered = strategies.stop_criteria.check_falsification(
        s, conf, _prev_conf)
    conf, _stagnation_detected = strategies.stop_criteria.check_stagnation(
        s, conf, _prev_conf, _n_citations, _falsification_just_triggered)
    _utility, _utility_stop = strategies.stop_criteria.compute_utility(
        s, conf, _prev_conf, _n_citations)
    _done_after_utility = bool(s.get("done"))

    s["final_confidence"] = conf

    # Diagnostic snapshot for `confidence_unjustified_drop` (Issue 1):
    # capture prev_competing_events BEFORE check_plateau rotates it. The marker
    # surfaces when the LLM lowered CONFIDENCE without naming a new
    # CONTRADICTION or COMPETING_EVENT to justify it — a sign that the
    # EVALUATE_FORMAT_SUFFIX stability instruction was not honoured by the
    # current model.
    _competing_unchanged_for_marker = (
        (s.get("competing_events") or "").strip()
        == (s.get("prev_competing_events") or "").strip()
    )

    _plateau_stop = strategies.stop_criteria.check_plateau(
        s, conf, _prev_conf, _stagnation_detected)
    _done_after_plateau = bool(s.get("done"))

    _m_contra_for_marker = re.search(
        r"CONTRADICTIONS:\s*([^\n]+)", _eval_raw or "", flags=re.IGNORECASE,
    )
    _contra_lead_token_for_marker = ""
    if _m_contra_for_marker:
        _contra_text = _m_contra_for_marker.group(1).strip()
        if _contra_text:
            _contra_lead_token_for_marker = (
                _contra_text.split()[0].rstrip(".:,;").lower()
            )
    _contradictions_present_for_marker = _contra_lead_token_for_marker not in {
        "", "nein", "no", "keine", "none", "k.a.", "n/a",
    }
    _confidence_unjustified_drop = bool(
        not _evaluate_fallback
        and int(_prev_conf or 0) > 0
        and int(conf) < int(_prev_conf)
        and _competing_unchanged_for_marker
        and not _contradictions_present_for_marker
    )
    if _confidence_unjustified_drop:
        log.warning(
            "TRACE evaluate: confidence drop without new contradictions/"
            "competing-events (prev=%d, curr=%d, round=%d)",
            int(_prev_conf), int(conf), s["round"],
        )

    _stop_cascade: dict[str, Any] = {
        "confidence": conf,
        "confidence_stop_target": settings.confidence_stop,
        "round": s["round"],
        "max_rounds": settings.max_rounds,
        "min_rounds": max(1, int(getattr(settings, "min_rounds", 1) or 1)),
        "utility_score": _utility,
        "utility_stop": _utility_stop,
        "plateau_stop": _plateau_stop,
        "stagnation_detected": _stagnation_detected,
        "falsification_triggered": bool(s.get("falsification_triggered", False)),
        "done_after_utility": _done_after_utility,
        "done_after_plateau": _done_after_plateau,
        "confidence_stop": conf >= settings.confidence_stop,
        "round_limit": s["round"] >= settings.max_rounds,
        "evidence_depth_gap_active": bool(evidence_depth_gap.get("active")),
        "evidence_depth_gap": evidence_depth_gap,
        "report_eligible_evidence_count": report_eligible_evidence_count,
        "min_report_eligible_evidence": tuning.min_report_eligible_evidence,
    }

    def _resolve_stop_reason() -> str:
        if _utility_stop:
            return "utility_stop"
        if _plateau_stop:
            return "plateau_stop"
        if conf >= settings.confidence_stop:
            return "confidence_stop"
        if s["round"] >= settings.max_rounds:
            return "round_limit"
        if s.get("done"):
            return "strategy_done"
        return ""

    # --- Final stop logic ---
    log.info(
        "TRACE evaluate: round=%d confidence=%d/%d gaps='%s' evidence_records=%d "
        "quality=%.2f claim_quality=%.2f claims(v/c/u)=%d/%d/%d model=%s done=%s",
        s["round"], conf, settings.confidence_stop,
        format_log_excerpt(s.get("gaps", ""), limit=300),
        len(s.get("evidence_ledger", []) or []),
        quality_score, claim_quality,
        verified_claims, contested_claims, unverified_claims,
        evaluate_model,
        conf >= settings.confidence_stop or s["round"] >= settings.max_rounds or s["done"],
    )
    emit_progress(
        s,
        t(
            s,
            "evaluate_quality_summary",
            source_quality=f"{quality_score:.2f}",
            claim_quality=f"{claim_quality:.2f}",
            evidence_records=len(s.get("evidence_ledger", []) or []),
            open_aspects=len(s.get("uncovered_aspects", []) or []),
        ),
    )

    _resolved_stop_reason = _resolve_stop_reason()
    if s["done"] or conf >= settings.confidence_stop or s["round"] >= settings.max_rounds:
        s["done"] = True
        s["_stop_reason"] = _resolved_stop_reason or "done"
        emit_progress(
            s,
            t(s, "research_finished", conf=conf, round=s["round"]),
        )
    else:
        s["_stop_reason"] = ""
        emit_progress(s, t(s, "confidence_continue", conf=conf))

    # min_rounds enforcement: if any earlier stop heuristic flipped done=True
    # but the configured min_rounds floor is not reached yet AND we are still
    # below max_rounds, suppress the stop. ``max_rounds`` always wins so a
    # mis-configured ``min_rounds > max_rounds`` cannot extend the loop
    # beyond the user-specified hard cap. Greift einheitlich für alle
    # Stop-Heuristiken (confidence, plateau, utility, stagnation,
    # falsification) — kein per-Heuristik-Sondercode noetig.
    _min_rounds = max(1, int(getattr(settings, "min_rounds", 1) or 1))
    if (
        s["done"]
        and s["round"] < _min_rounds
        and s["round"] < settings.max_rounds
    ):
        log.info(
            "TRACE evaluate: stop suppressed by min_rounds (round=%d < %d, max=%d)",
            s["round"], _min_rounds, settings.max_rounds,
        )
        emit_progress(
            s,
            t(
                s,
                "min_rounds_continue",
                min_rounds=_min_rounds,
                round=s["round"],
            ),
        )
        s["done"] = False
        _stop_cascade["suppressed_by_min_rounds"] = True
        _stop_cascade["suppressed_stop_reason"] = s.get("_stop_reason", "")
        s["_stop_reason"] = ""
    else:
        _stop_cascade["suppressed_by_min_rounds"] = False
    # Do not stop early while the EvidenceLedger still holds too few
    # report-eligible records to render a substantial report. This is a
    # general minimum-evidence guardrail, not a damping layer over a
    # derived list -- it checks the primary truth directly.
    report_evidence_too_thin = (
        bool(s.get("evidence_ledger"))
        and report_eligible_evidence_count < tuning.min_report_eligible_evidence
    )
    if (
        s["done"]
        and report_evidence_too_thin
        and s["round"] < settings.max_rounds
    ):
        log.info(
            "TRACE evaluate: stop suppressed by report-eligible evidence "
            "(records=%d/%d round=%d/%d)",
            report_eligible_evidence_count,
            tuning.min_report_eligible_evidence,
            s["round"],
            settings.max_rounds,
        )
        emit_progress(
            s,
            (
                "Evidenzlage noch zu duenn "
                f"({report_eligible_evidence_count}/{tuning.min_report_eligible_evidence} "
                "report-faehige Records); weitere Recherche."
            ),
        )
        _stop_cascade["suppressed_by_report_evidence"] = True
        _stop_cascade["suppressed_stop_reason"] = s.get("_stop_reason", "")
        s["done"] = False
        s["_stop_reason"] = ""
    else:
        _stop_cascade["suppressed_by_report_evidence"] = False
    _stop_cascade["final_stop_reason"] = s.get("_stop_reason", "")
    _stop_cascade["final_done"] = bool(s.get("done"))

    _append_forensic_event(
        s,
        settings,
        event="stop_cascade",
        node="evaluate",
        payload=_stop_cascade,
    )
    _evaluate_score_snapshot = append_score_snapshot(
        s,
        phase="evaluate",
        extra={
            "llm_confidence": conf,
            "final_confidence": s.get("final_confidence", conf),
            "utility_score": _utility,
            "stop_reason": s.get("_stop_reason", ""),
            "evidence_depth_gap_active": bool(evidence_depth_gap.get("active")),
            "report_eligible_evidence_count": report_eligible_evidence_count,
            "confidence_unjustified_drop": _confidence_unjustified_drop,
        },
    )
    _append_forensic_event(
        s,
        settings,
        event="score_snapshot",
        node="evaluate",
        payload=_evaluate_score_snapshot,
    )

    _eval_log_entry: dict[str, Any] = {
        "node": "evaluate",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"],
        "confidence": conf,
        "final_confidence": conf,
        "prev_conf": _prev_conf,
        "confidence_parsed": _confidence_parsed,
        "_confidence_parsed": _confidence_parsed,
        "confidence_unjustified_drop": _confidence_unjustified_drop,
        "confidence_stop_target": settings.confidence_stop,
        "gaps": s.get("gaps", ""),
        "competing_events": s.get("competing_events", ""),
        "stagnation_detected": _stagnation_detected,
        "falsification_triggered": s.get("falsification_triggered", False),
        "evidence_consistency": s.get("evidence_consistency", 0),
        "evidence_sufficiency": s.get("evidence_sufficiency", 0),
        "evidence_consistency_parsed": s.get("_evidence_consistency_parsed", True),
        "evidence_sufficiency_parsed": s.get("_evidence_sufficiency_parsed", True),
        "_evidence_consistency_parsed": s.get("_evidence_consistency_parsed", True),
        "_evidence_sufficiency_parsed": s.get("_evidence_sufficiency_parsed", True),
        "verified_claims": verified_claims,
        "contested_claims": contested_claims,
        "unverified_claims": unverified_claims,
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "consolidated_claims_count": len(s.get("consolidated_claims", [])),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "evidence_depth_gap": evidence_depth_gap,
        "evidence_depth_gap_active": bool(evidence_depth_gap.get("active")),
        "report_eligible_evidence_count": report_eligible_evidence_count,
        "aspect_coverage": s.get("aspect_coverage", 0.0),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "model": evaluate_model,
        "utility_score": _utility,
        "utility_stop": _utility_stop,
        "plateau_stop": _plateau_stop,
        "evidence_record_count": len(s.get("evidence_ledger", []) or []),
        "stop_cascade": _stop_cascade,
        "_stop_reason": s.get("_stop_reason", ""),
        "stop_by_confidence": conf >= settings.confidence_stop,
        "stop_by_round_limit": s["round"] >= settings.max_rounds,
        "stop_by_existing_done": _done_after_utility or _done_after_plateau,
        "done": s["done"],
        "guardrail_reasons": list(_guardrail_result.reasons or []),
        "_evaluate_fallback": bool(_evaluate_fallback),
        **_evaluate_fallback,
    }
    if settings.testing_mode and _eval_raw:
        _eval_log_entry["reasoning"] = _eval_raw
    append_iteration_log(s, _eval_log_entry, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 5. answer
# ======================================================================= #


def _build_algorithm_failure_report(
    s: dict[str, Any],
    *,
    failures: list[dict[str, Any]],
    evidence_record_count: int,
    citation_count: int,
) -> str:
    """Render a short diagnostic instead of a normal report after core failure."""
    lines = [
        "## ALGO-FAIL: Kein auditierbarer Report erzeugt",
        "",
        (
            "Der normale Abschlussbericht wurde blockiert, weil ein Kernschritt "
            "des Evidence-Pfads fehlgeschlagen ist. Ein quellenbasierter "
            "Hard-Fact-Report waere in diesem Zustand nicht belastbar."
        ),
        "",
        "## Diagnose",
        "",
        f"- Run-ID: `{s.get('_run_id', '')}`",
        f"- Rechercherunden: `{s.get('round', 0)}`",
        f"- Referenzen gesammelt: `{citation_count}`",
        f"- Evidenz-Records im Ledger: `{evidence_record_count}`",
        f"- Claim-Extraction-Aufrufe: `{s.get('_claim_extraction_attempts_total', 0)}`",
        f"- Claim-Extraction-Fehler: `{s.get('_claim_extraction_failures_total', 0)}`",
        "",
        "## Blockierende Kernfehler",
        "",
    ]
    for failure in failures:
        phase = str(failure.get("phase", "unknown"))
        reason = str(failure.get("reason", "unknown"))
        round_index = failure.get("round", "")
        message = str(failure.get("message", "") or "")
        lines.append(f"- `{phase}` in Runde `{round_index}`: `{reason}`")
        if message:
            lines.append(f"  {message}")
    lines.extend(
        [
            "",
            "## Naechster Debug-Schritt",
            "",
            (
                "Bitte das Forensic-Log mit "
                "`uv run python scripts/debug_research_log.py <logfile>` auswerten. "
                "Der finale Bericht wurde absichtlich nicht formuliert, damit ein "
                "Algorithmusfehler nicht wie ein erfolgreicher degradierter Normalpfad wirkt."
            ),
        ]
    )
    return "\n".join(lines)


def answer(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Formulate the final user-facing answer from the collected evidence.

    Args:
        s: Mutable AgentState-compatible dict. Reads context, claim
            metrics, language, citations, and deadline; writes the final
            answer text and answer-node runtime metadata.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for answer citation selection and
            claim formatting.
        settings: Agent behavior settings controlling citation limits and
            fallback behavior.

    Returns:
        The mutated state dict with the final answer text populated.

    Raises:
        AgentRateLimited: Propagated when final answer generation hits a
            fatal upstream rate limit.

    Example:
        >>> answer(state, providers=providers, strategies=strategies, settings=settings)
        {'answer': 'Aktueller Stand ...', ...}
    """
    check_cancel_event(s)
    n_rounds = s.get("round", 0)
    round_label = t(
        s,
        "answer_round_singular" if n_rounds == 1 else "answer_round_plural",
    )
    emit_progress(
        s,
        t(s, "answer_start", n=n_rounds, round_label=round_label),
    )
    _t0 = time.monotonic()
    tuning = settings.report_tuning

    # Determine answer language
    lang = s.get("language", "de")
    answer_lang = LANG_NAMES.get(lang, lang)

    # Single canonical evidence view: the answer prompt consumes exactly one
    # record-driven Markdown overview rendered straight from the EvidenceLedger.
    # There are no parallel report-evidence-bundle / prompt-evidence-unit /
    # rendered-context channels and no separate answer-citation lists -- the
    # citation allowlist is the union of visible source-block URLs.
    consolidated_claims = s.get("consolidated_claims", [])
    evidence_ledger = s.get("evidence_ledger", []) or []
    evidence_overview = render_evidence_ledger_overview(
        evidence_ledger,
        max_total_chars=tuning.prompt_evidence_total_char_budget,
        max_record_chars=tuning.prompt_evidence_record_char_limit,
        query_synthesis=s.get("query_synthesis", {}),
    )
    evidence_overview_markdown = evidence_overview.markdown
    allowed_citations = list(evidence_overview.allowed_urls)
    has_ledger_context = bool(evidence_ledger)
    strict_algorithm_mode = _strict_algorithm_failure_mode(settings)
    if strict_algorithm_mode:
        claim_attempts_total = int(s.get("_claim_extraction_attempts_total", 0) or 0)
        claim_failures_total = int(s.get("_claim_extraction_failures_total", 0) or 0)
        if claim_attempts_total and claim_failures_total >= claim_attempts_total:
            _record_algorithm_failure(
                s,
                settings,
                node="answer",
                phase="claim_extraction",
                reason="run_all_sources_failed",
                message=(
                    "All claim-extraction calls across the run failed; "
                    "normal report synthesis would rely on unverified source context only."
                ),
                blocking=True,
                details={
                    "attempts": claim_attempts_total,
                    "failed": claim_failures_total,
                    "total": claim_attempts_total,
                },
            )
    if has_ledger_context and evidence_overview.rendered_record_count == 0:
        _record_algorithm_failure(
            s,
            settings,
            node="answer",
            phase="evidence_overview",
            reason="no_rendered_evidence_records",
            message=(
                "The EvidenceLedger held records but the evidence overview "
                "rendered no source records for the final composer."
            ),
            blocking=strict_algorithm_mode,
            details={
                "evidence_record_count": len(evidence_ledger),
                "omitted_record_count": evidence_overview.omitted_record_count,
            },
        )
    blocking_algorithm_failures = _blocking_algorithm_failures(s)
    algorithm_report_blocked = bool(blocking_algorithm_failures)

    # Assemble shared prompt state for the section-wise composer.
    state_data: dict[str, Any] = {
        "today_str": today(),
        "answer_lang": answer_lang,
        "evidence_overview": evidence_overview_markdown,
        "evidence_label_urls": dict(evidence_overview.label_urls),
        "evidence_label_by_id": dict(evidence_overview.label_by_evidence_id),
        "visible_evidence_labels": sorted(evidence_overview.label_urls),
        "visible_evidence_label_count": len(evidence_overview.label_urls),
        "rendered_evidence_ids": list(evidence_overview.rendered_evidence_ids),
        "allowed_citations": allowed_citations,
        "rendered_evidence_record_count": evidence_overview.rendered_record_count,
        "omitted_evidence_record_count": evidence_overview.omitted_record_count,
        # The evidence-depth gap is the only signal that knows how many
        # verified claims rest on a single source. It is read by the
        # prompt builder to trigger the EVIDENZTIEFE block + the existing
        # TRANSPARENZPFLICHT block when cross-check coverage is thin.
        "evidence_depth_gap": dict(s.get("evidence_depth_gap", {}) or {}),
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "required_aspects": s.get("required_aspects"),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "competing_events": s.get("competing_events", ""),
        "history": s.get("history", ""),
        "report_profile": str(settings.report_profile),
    }
    s.update(
        {
            "evidence_label_urls": dict(evidence_overview.label_urls),
            "evidence_label_by_id": dict(evidence_overview.label_by_evidence_id),
            "visible_evidence_labels": sorted(evidence_overview.label_urls),
            "visible_evidence_label_count": len(evidence_overview.label_urls),
            "rendered_evidence_ids": list(evidence_overview.rendered_evidence_ids),
            "allowed_citations": allowed_citations,
            "rendered_evidence_record_count": evidence_overview.rendered_record_count,
            "omitted_evidence_record_count": evidence_overview.omitted_record_count,
        }
    )
    if settings.testing_mode or forensic_enabled(settings):
        diagnostics = _build_answer_prompt_diagnostics(state_data, s)
        append_iteration_log(
            s,
            {
                "event": "answer_prompt_diagnostics",
                "node": "answer",
                "timestamp": time.time(),
                **diagnostics,
            },
            testing_mode=settings.testing_mode,
        )
    # Resolve the answer node once (emits the single model_resolution event) and
    # reuse the result for the diagnostics and both composition calls, so every
    # answer-node diagnostic names the model/effort actually routed -- not the
    # provider's reasoning_model default.
    answer_model, answer_effort = _resolve_node_llm(s, settings, providers, "answer")
    _append_forensic_event(
        s,
        settings,
        event="answer_prompt_inputs",
        node="answer",
        payload={
            "report_profile": str(settings.report_profile),
            "model": answer_model,
            "evidence_record_count": len(evidence_ledger),
            "rendered_evidence_record_count": evidence_overview.rendered_record_count,
            "omitted_evidence_record_count": evidence_overview.omitted_record_count,
            "visible_evidence_label_count": len(evidence_overview.label_urls),
            "evidence_overview_chars": len(evidence_overview_markdown),
            "allowed_citation_count": len(allowed_citations),
            "consolidated_claim_count": len(consolidated_claims),
            "algorithm_failure_count": len(s.get("algorithm_failures", []) or []),
            "blocking_algorithm_failure_count": len(blocking_algorithm_failures),
            "evidence_overview": evidence_overview_markdown,
            "section_plan": [
                {
                    "heading": section.heading,
                    "required": section.required,
                }
                for section in settings.report_tuning.answer_sections
            ],
        },
    )
    fallback_model = _resolve_answer_fallback_model(settings, providers, answer_model)
    fallback_attempted = False
    fallback_succeeded = False
    answer_fallback_kind = ""
    answer_fallback_reason = ""
    composition_result = _AnswerCompositionResult(answer="", finish_reason="", section_logs=[])

    def _answer_warning_header(reason: str) -> str:
        """Render a visible warning header for fallback answers.

        The header is the first thing the operator and any downstream
        consumer (UI, parity tooling, exported markdown) sees, so a
        silent degradation cannot pass unnoticed even when the iteration
        log is not consulted.
        """
        return (
            "> [!WARNING] Antwort-Synthese-Fallback aktiv\n"
            f"> {reason}\n"
            "> Die folgende Antwort enthaelt **keine LLM-synthetisierte** "
            "Zusammenfassung, sondern den bisherigen Recherche-Kontext "
            "in Rohform. Iteration-Marker `_answer_fallback=true`."
        )

    def _build_fallback_answer(reason: str, *, no_context_message: str) -> str:
        if evidence_overview_markdown:
            return f"{_answer_warning_header(reason)}\n\n{evidence_overview_markdown}"
        return f"{_answer_warning_header(reason)}\n\n{no_context_message}"

    def _answer_failure_phase(exc: BaseException) -> str:
        """Classify provider failures for fail-closed answer diagnostics."""
        if is_model_capacity_error(exc):
            return "model_capacity"
        error_text = str(exc).lower()
        if any(
            marker in error_text
            for marker in (
                "usage limit",
                "usage limits",
                "rate limit",
                "quota",
                "context limit",
                "token limit",
                "max token",
                "maximum token",
            )
        ):
            return "provider_limit"
        return "answer_synthesis"

    def _build_answer_algorithm_failure(
        *,
        exc: BaseException,
        phase: str,
        reason: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> _AnswerCompositionResult:
        _record_algorithm_failure(
            s,
            settings,
            node="answer",
            phase=phase,
            reason=reason or type(exc).__name__,
            message=message or str(exc),
            blocking=True,
            details=details,
        )
        failures = _blocking_algorithm_failures(s)
        emit_progress(
            s,
            t(s, "algorithm_failure_report_blocked", count=len(failures)),
        )
        s["final_confidence"] = min(int(s.get("final_confidence", 0) or 0) or 3, 3)
        s["answer"] = _build_algorithm_failure_report(
            s,
            failures=failures,
            evidence_record_count=len(evidence_ledger),
            citation_count=len(s.get("source_records", {}) or {}),
        )
        return _AnswerCompositionResult(
            answer=s["answer"],
            finish_reason=phase,
            section_logs=[],
            composition_aborted=True,
            sections_planned=len(settings.report_tuning.answer_sections),
            sections_attempted=0,
        )

    if algorithm_report_blocked:
        emit_progress(
            s,
            t(s, "algorithm_failure_report_blocked", count=len(blocking_algorithm_failures)),
        )
        s["final_confidence"] = min(int(s.get("final_confidence", 0) or 0) or 3, 3)
        s["answer"] = _build_algorithm_failure_report(
            s,
            failures=blocking_algorithm_failures,
            evidence_record_count=len(evidence_ledger),
            citation_count=len(s.get("source_records", {}) or {}),
        )
        composition_result = _AnswerCompositionResult(
            answer=s["answer"],
            finish_reason="algorithm_failure",
            section_logs=[],
            composition_aborted=True,
            sections_planned=len(settings.report_tuning.answer_sections),
            sections_attempted=0,
        )
    else:
        try:
            composition_result = _compose_answer_sections(
                s,
                providers=providers,
                settings=settings,
                state_data=state_data,
                model=answer_model,
                reasoning_effort=answer_effort,
            )
            s["answer"] = composition_result.answer
        except AgentModelCapacityError as exc_capacity:
            _record_algorithm_failure(
                s,
                settings,
                node="answer",
                phase="model_capacity",
                reason=type(exc_capacity).__name__,
                message=str(exc_capacity),
                blocking=True,
                details={
                    "model": getattr(exc_capacity, "model", ""),
                    "capacity_phase": getattr(exc_capacity, "phase", ""),
                },
            )
            blocking_algorithm_failures = _blocking_algorithm_failures(s)
            emit_progress(
                s,
                t(s, "algorithm_failure_report_blocked", count=len(blocking_algorithm_failures)),
            )
            s["final_confidence"] = min(int(s.get("final_confidence", 0) or 0) or 3, 3)
            s["answer"] = _build_algorithm_failure_report(
                s,
                failures=blocking_algorithm_failures,
                evidence_record_count=len(evidence_ledger),
                citation_count=len(s.get("source_records", {}) or {}),
            )
            composition_result = _AnswerCompositionResult(
                answer=s["answer"],
                finish_reason="model_capacity",
                section_logs=[],
                composition_aborted=True,
                sections_planned=len(settings.report_tuning.answer_sections),
                sections_attempted=0,
            )
        except AgentTimeout as exc_timeout:
            if strict_algorithm_mode:
                composition_result = _build_answer_algorithm_failure(
                    exc=exc_timeout,
                    phase="answer_synthesis_timeout",
                    message=(
                        "Answer synthesis exceeded the configured timeout; "
                        "strict mode will not emit a raw-context fallback report."
                    ),
                    details={"error": str(exc_timeout)[:300]},
                )
                s["answer"] = composition_result.answer
                answer_fallback_kind = ""
                answer_fallback_reason = ""
                finish_reason = str(composition_result.finish_reason or "")
                section_logs = list(composition_result.section_logs)
                # Continue with common answer bookkeeping below.
            else:
                answer_fallback_kind = "timeout"
                answer_fallback_reason = (
                    "Die Antwort-Synthese hat das LLM-Timeout ueberschritten."
                )
                log.warning("TRACE answer fallback (timeout): %s", exc_timeout)
                emit_progress(s, t(s, "answer_timeout_fallback"), severity="warning")
                s["answer"] = _build_fallback_answer(
                    answer_fallback_reason,
                    no_context_message=(
                        "Es liegen noch keine Kontextdaten vor. Bitte erneut versuchen, "
                        "ggf. `REASONING_TIMEOUT` erhoehen oder `MAX_TOTAL_SECONDS` anpassen."
                    ),
                )
        except (
            OpenAIError,
            AgentRateLimited,
            AnthropicAPIError,
            AzureOpenAIAPIError,
            BedrockAPIError,
        ) as e:
            provider_error_label = f"{type(e).__name__}: {e}"[:240]
            log.error("Finale Antwort fehlgeschlagen: %s", e)
            if strict_algorithm_mode:
                phase = _answer_failure_phase(e)
                composition_result = _build_answer_algorithm_failure(
                    exc=e,
                    phase=phase,
                    reason=type(e).__name__,
                    message=(
                        "Answer synthesis provider call failed; strict mode will "
                        "not emit a raw-context fallback report."
                    ),
                    details={
                        "error": str(e)[:300],
                        "error_type": type(e).__name__,
                    },
                )
                s["answer"] = composition_result.answer
            elif fallback_model:
                try:
                    fallback_attempted = True
                    emit_progress(
                        s,
                        t(s, "answer_fallback_model", model=fallback_model),
                        severity="warning",
                    )
                    composition_result = _compose_answer_sections(
                        s,
                        providers=providers,
                        settings=settings,
                        state_data=state_data,
                        model=fallback_model,
                        reasoning_effort="",
                    )
                    s["answer"] = composition_result.answer
                    fallback_succeeded = True
                except (
                    OpenAIError,
                    AgentTimeout,
                    AgentRateLimited,
                    AnthropicAPIError,
                    AzureOpenAIAPIError,
                    BedrockAPIError,
                ) as e2:
                    fallback_error_label = f"{type(e2).__name__}: {e2}"[:240]
                    answer_fallback_kind = "fallback_model_failed"
                    answer_fallback_reason = (
                        f"Primaeres Antwort-Modell fehlgeschlagen ({provider_error_label}); "
                        f"Fallback-Modell `{fallback_model}` ebenfalls fehlgeschlagen "
                        f"({fallback_error_label})."
                    )
                    log.warning(
                        "TRACE answer fallback (fallback_model_failed model=%s): %s",
                        fallback_model,
                        e2,
                    )
                    emit_progress(
                        s,
                        t(s, "answer_fallback_model_failed", error=fallback_error_label),
                        severity="warning",
                    )
                    s["answer"] = _build_fallback_answer(
                        answer_fallback_reason,
                        no_context_message=(
                            "Es liegen noch keine Kontextdaten vor. Bitte erneut versuchen "
                            "und Provider-Verbindung pruefen."
                        ),
                    )
            else:
                answer_fallback_kind = "no_fallback_model"
                answer_fallback_reason = (
                    f"Antwort-Synthese fehlgeschlagen ({provider_error_label}); "
                    "kein Fallback-Modell konfiguriert."
                )
                log.warning(
                    "TRACE answer fallback (no_fallback_model): %s",
                    e,
                )
                emit_progress(
                    s,
                    t(
                        s,
                        "answer_provider_failed_no_fallback",
                        error=provider_error_label,
                    ),
                    severity="warning",
                )
                s["answer"] = _build_fallback_answer(
                    answer_fallback_reason,
                    no_context_message=(
                        "Es liegen noch keine Kontextdaten vor. Bitte erneut versuchen "
                        "und Provider-Verbindung pruefen."
                    ),
                )

    algorithm_report_blocked = bool(_blocking_algorithm_failures(s))
    finish_reason = str(composition_result.finish_reason or "")
    section_logs = list(composition_result.section_logs)
    s["answer_finish_reason"] = finish_reason
    incomplete_reasons = _detect_incomplete_answer(
        s.get("answer", ""),
        finish_reason=finish_reason,
        report_profile=settings.report_profile,
    )
    s["answer_incomplete"] = bool(incomplete_reasons)
    s["answer_incomplete_reasons"] = list(incomplete_reasons)
    if incomplete_reasons:
        emit_progress(s, t(s, "answer_incomplete_detected"), severity="warning")
        log.warning(
            "TRACE answer: incomplete answer detected (finish_reason=%s, reasons=%s)",
            finish_reason or "unknown",
            incomplete_reasons,
        )

    s["answer"], expanded_evidence_label_links = _expand_bare_evidence_label_links(
        s.get("answer", ""),
        state_data.get("evidence_label_urls", {}) or {},
    )

    # Quick citation guardrail: remove links that are not in the
    # EvidenceOverview-derived allowlist (the union of visible source-block URLs).
    removed_link_count = 0
    allowed_citation_urls = {
        normalize_url(url)
        for url in allowed_citations
        if normalize_url(url)
    }
    appended_sources_footer = False
    allowed_link_count = 0
    if allowed_citation_urls and s.get("answer"):
        s["answer"], removed_link_count = sanitize_answer_links(
            s["answer"], allowed_citation_urls)
        if removed_link_count:
            log.info("TRACE answer: removed %d non-allowed links", removed_link_count)
            emit_progress(s, t(s, "answer_links_removed", n=removed_link_count))
        allowed_link_count = count_allowed_links(s["answer"], allowed_citation_urls)

    if incomplete_reasons:
        s["answer"] = _repair_answer_markdown_tail(s.get("answer", ""))
    if algorithm_report_blocked:
        appendix_sections = []
        reference_link_count = 0
        additional_link_count = 0
        s["report_references"] = []
    else:
        appendix = _build_answer_appendix_sections(
            s.get("answer", ""),
            allowed_citations=allowed_citations,
            label_urls=state_data.get("evidence_label_urls", {}) or {},
            strategies=strategies,
            incomplete_reasons=incomplete_reasons,
            finish_reason=finish_reason,
            answer_contract=s.get("answer_contract", "general"),
            tier_by_url=_reference_tiers_from_evidence_ledger(evidence_ledger),
        )
        appendix_sections = appendix.sections
        reference_link_count = len(appendix.references)
        additional_link_count = len(appendix.additional_links)
        s["report_references"] = appendix.references
    answer_body = s.get("answer", "")
    if appendix_sections:
        s["answer"] = answer_body.rstrip() + "\n\n---\n\n" + "\n\n---\n\n".join(appendix_sections)
        appended_sources_footer = True

    answer_claim_bindings = _build_answer_claim_bindings(
        answer_body,
        consolidated_claims=consolidated_claims,
        allowed_citations=allowed_citations,
        provider_citation_records=s.get("provider_citation_records", []) or [],
    )
    matched_claim_ids = {
        str(binding.get("claim_id", ""))
        for binding in answer_claim_bindings
        if binding.get("claim_id") and binding.get("binding_status") == "matched"
    }
    for claim in consolidated_claims:
        claim["used_in_answer"] = str(claim.get("claim_id", "")) in matched_claim_ids
    s["answer_claim_bindings"] = answer_claim_bindings
    for binding in answer_claim_bindings:
        _append_forensic_event(
            s,
            settings,
            event="answer_claim_binding",
            node="answer",
            payload=binding,
        )
    answer_evidence_bindings = audit_answer_evidence_bindings(
        answer_body,
        evidence_ledger,
    )
    s["answer_evidence_bindings"] = answer_evidence_bindings
    _unknown_citation_count = sum(
        1
        for binding in answer_evidence_bindings
        if binding.get("binding_status") == "unknown_citation"
    )
    _matched_evidence_count = sum(
        1
        for binding in answer_evidence_bindings
        if binding.get("binding_status") == "matched"
    )
    # The contract is decided by the claim-level binding (does a cited answer
    # sentence plausibly carry a verified/contested consolidated claim?), not by
    # the coarser URL-level evidence audit (does the URL resolve to a record
    # with any verified claim?).
    _matched_claim_count = sum(
        1
        for binding in answer_claim_bindings
        if binding.get("binding_status") == "matched"
    )
    # A cited body sentence whose URL carries NO consolidated claim at all -- the
    # claim-level twin of `unknown_citation` (a cited link with no backing
    # record). Both mean the answer points at something the evidence base does
    # not substantiate, so mixed with a genuine match the run is not clean.
    _citation_without_claim_count = sum(
        1
        for binding in answer_claim_bindings
        if binding.get("binding_status") == "citation_without_claim"
    )
    # Tiers ordered strictly by severity, monotonic to _CONTRACT_CONFIDENCE_CAP
    # (algorithm_failed 3 < source_context_only 4 < needs_review 6 < clean):
    #   no citable evidence base at all              -> unknown (audit N/A, uncapped)
    #   no claim-bound sentence at all               -> source_context_only
    #   bound, but a cited target is unsubstantiated -> needs_review
    #     (unknown_citation: no record; citation_without_claim: record w/o claim)
    #   bound and nothing unsubstantiated            -> clean
    # `unknown` keys on whether the body HAD citable evidence to ground in
    # (allowed_citations), not on whether it happened to cite any. Keying it on
    # empty bindings let an answer that ignores real gathered evidence and cites
    # nothing escape to uncapped `unknown` -- while the appendix still appended
    # references, making the report look sourced. Such an ungrounded body now
    # falls to source_context_only (cap 4) like any other claimless answer; only
    # a run with no citable evidence at all (direct-LLM / empty search) is
    # `unknown`. A `source_only_binding` (background citation whose claim sits in
    # the ledger) is deliberately tolerated and never downgrades the tier.
    _evidence_contract_status = (
        "algorithm_failed"
        if algorithm_report_blocked
        else "unknown"
        if not allowed_citations
        else "source_context_only"
        if not _matched_claim_count
        else "needs_review"
        if (_unknown_citation_count or _citation_without_claim_count)
        else "clean"
    )
    if s.get("algorithm_failures") and _evidence_contract_status == "clean":
        _evidence_contract_status = "needs_review"
    # A run where almost every "verified" claim rests on a single quality
    # source is not a clean hard-fact report -- downgrade so the
    # Evidenzhinweis appendix + confidence cap fire even when the URL audit
    # itself looks fine.
    if (
        _evidence_contract_status == "clean"
        and (s.get("evidence_depth_gap") or {}).get("active")
    ):
        _evidence_contract_status = "needs_review"

    # Single source of truth: persist the canonical contract so result.py and
    # scoring.py read it instead of each recomputing a divergent copy.
    s["evidence_contract_status"] = _evidence_contract_status

    # One confidence ceiling, keyed on the contract (see _CONTRACT_CONFIDENCE_CAP).
    _confidence_cap = _CONTRACT_CONFIDENCE_CAP.get(_evidence_contract_status)
    if _confidence_cap is not None:
        s["final_confidence"] = min(
            int(s.get("final_confidence", 0) or 0), _confidence_cap
        )

    if _evidence_contract_status in {"needs_review", "source_context_only"}:
        if _evidence_contract_status == "needs_review":
            emit_progress(s, t(s, "evidence_contract_needs_review"))
        else:
            emit_progress(s, t(s, "evidence_contract_source_context_only"))
        if s.get("answer"):
            depth_gap_active_now = bool(
                (s.get("evidence_depth_gap") or {}).get("active")
            )
            if _evidence_contract_status == "source_context_only":
                evidence_notice = (
                    "Die Antwort stuetzt sich auf quellenbasierten Evidence-Ledger-"
                    "Kontext, aber keine der zitierten Quellen traegt eine verifizierte "
                    "oder strittige Aussage. Sie ist daher eine quellenbasierte Synthese, "
                    "kein verifizierter Hard-Fact-Report."
                )
            elif depth_gap_active_now:
                gap = s.get("evidence_depth_gap") or {}
                gap_ratio = float(gap.get("single_source_ratio", 0.0) or 0.0)
                evidence_notice = (
                    "Die Evidenzlage ist ueberwiegend single-source: "
                    f"{int(gap.get('single_source_verified_count', 0) or 0)} von "
                    f"{int(gap.get('verified_count', 0) or 0)} verifizierten Aussagen "
                    f"({int(gap_ratio * 100)} %) ruhen auf einer einzigen Quelle, "
                    f"nur {int(gap.get('cross_checked_count', 0) or 0)} sind cross-checked. "
                    "Aussagen ohne cross-checked oder primary-source-Basis sind als "
                    "Einzelquellenangaben zu lesen; die in einzelnen Saetzen angegebene "
                    "Quelle traegt die jeweilige Behauptung allein."
                )
            else:
                evidence_notice = (
                    "Einige zitierte Links liessen sich nicht auf einen Evidence-Record "
                    "zurueckfuehren. Die betroffenen Punkte sind als nicht ausreichend "
                    "belegt zu behandeln."
                )
            s["answer"] = (
                s["answer"].rstrip()
                + "\n\n---\n\n"
                + "## Evidenzhinweis\n\n"
                + evidence_notice
            )

    for binding in answer_evidence_bindings:
        _append_forensic_event(
            s,
            settings,
            event="answer_sentence_audit",
            node="answer",
            payload=binding,
        )
    _answer_score_snapshot = append_score_snapshot(
        s,
        phase="answer",
        extra={
            # Same claim-grounded count the contract uses; the URL-level audit
            # feeds the separate matched_evidence_binding_count diagnostic.
            "answer_bound_claims_count": _matched_claim_count,
            "unbound_answer_citations_count": _unknown_citation_count,
            "evidence_contract_status": _evidence_contract_status,
            "final_confidence": s.get("final_confidence", 0),
        },
    )
    _append_forensic_event(
        s,
        settings,
        event="score_snapshot",
        node="answer",
        payload=_answer_score_snapshot,
    )
    _append_forensic_event(
        s,
        settings,
        event="citation_selection",
        node="answer",
        payload={
            "allowed_citations": allowed_citations[:60],
            "allowed_citation_count": len(allowed_citations),
            "allowed_link_count": allowed_link_count,
            "removed_non_allowed_links": removed_link_count,
            "answer_claim_binding_count": len(answer_claim_bindings),
            "answer_evidence_binding_count": len(answer_evidence_bindings),
            "matched_evidence_binding_count": _matched_evidence_count,
            "unknown_citation_count": _unknown_citation_count,
            "expanded_evidence_label_links": expanded_evidence_label_links,
        },
    )

    # Append stats footer
    elapsed = time.monotonic() - s.get("start_time", time.monotonic())
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    time_str = f"{minutes}:{seconds:02d} min" if minutes > 0 else f"{seconds}s"
    conf = s.get("final_confidence", 0)
    n_sources = len(s.get("source_records", {}) or {})
    n_rounds = s.get("round", 0)
    n_queries = len(s.get("queries", []))
    rendered_record_count = evidence_overview.rendered_record_count
    omitted_record_count = evidence_overview.omitted_record_count
    verified_claim_count = sum(
        1 for claim in consolidated_claims if claim.get("status") == "verified"
    )
    cross_checked_claim_count = sum(
        1 for claim in consolidated_claims
        if claim.get("verification_basis") == "verified_cross_checked"
    )

    stats_parts = []
    if n_sources:
        stats_parts.append(f"{n_sources} Quellen recherchiert")
    if rendered_record_count:
        stats_parts.append(f"{rendered_record_count} Quellen im Evidence-Prompt")
    if allowed_link_count:
        stats_parts.append(f"{allowed_link_count} Quellen im Report verlinkt")
    if verified_claim_count:
        stats_parts.append(f"{verified_claim_count} verifizierte Aussagen")
    if cross_checked_claim_count:
        stats_parts.append(f"{cross_checked_claim_count} cross-checked Aussagen")
    # When the evidence-depth gap is active, surface the single-source share
    # so the footer makes the imbalance visible at a glance instead of
    # hiding it behind a generic "verified" count.
    _depth_gap_footer = s.get("evidence_depth_gap") or {}
    if _depth_gap_footer.get("active") and verified_claim_count:
        single_source_n = int(_depth_gap_footer.get("single_source_verified_count", 0) or 0)
        ratio_pct = int(
            float(_depth_gap_footer.get("single_source_ratio", 0.0) or 0.0) * 100
        )
        stats_parts.append(
            f"{single_source_n}/{verified_claim_count} single-source ({ratio_pct}%)"
        )
    if omitted_record_count:
        stats_parts.append(f"{omitted_record_count} Quellen wegen Budget ausgelassen")
    if _evidence_contract_status != "unknown":
        stats_parts.append(f"Evidence-Contract: {_evidence_contract_status}")
    if s.get("algorithm_failures"):
        stats_parts.append(
            f"ALGO-FAIL: {len(s.get('algorithm_failures', []) or [])}"
        )
    if n_queries:
        stats_parts.append(f"{n_queries} Suchen")
    if n_rounds:
        stats_parts.append(f"{n_rounds} {'Runde' if n_rounds == 1 else 'Runden'}")
    stats_parts.append(time_str)
    if conf:
        stats_parts.append(f"Confidence {conf}/10")

    stats_line = " · ".join(stats_parts)
    s["answer"] += f"\n\n---\n*{stats_line}*"

    log.info(
        "TRACE answer: length=%d sources=%d rendered_records=%d omitted_records=%d "
        "linked=%d refs=%d extra_links=%d sections=%d rounds=%d elapsed=%.1fs "
        "confidence=%d finish_reason=%s incomplete=%s",
        len(s["answer"]), n_sources, rendered_record_count, omitted_record_count,
        allowed_link_count, reference_link_count, additional_link_count, len(section_logs),
        n_rounds, elapsed, conf, finish_reason or "", bool(incomplete_reasons),
    )
    log.debug("ANSWER text:\n%s", s["answer"])

    append_iteration_log(s, {
        "node": "answer",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "answer_length": len(s["answer"]),
        "evidence_record_count": len(evidence_ledger),
        "rendered_evidence_record_count": rendered_record_count,
        "omitted_evidence_record_count": omitted_record_count,
        "evidence_overview_chars": len(evidence_overview_markdown),
        "visible_evidence_label_count": len(evidence_overview.label_urls),
        "allowed_citation_count": len(allowed_citations),
        "allowed_citations": allowed_citations[:10],
        "_answer_fallback": bool(answer_fallback_kind) or (fallback_attempted and not fallback_succeeded),
        "_answer_fallback_kind": answer_fallback_kind,
        "_answer_fallback_reason": answer_fallback_reason,
        "_answer_links_sanitized": removed_link_count,
        "removed_non_allowed_links": removed_link_count,
        "expanded_evidence_label_links": expanded_evidence_label_links,
        "allowed_link_count": allowed_link_count,
        "reference_link_count": reference_link_count,
        "additional_link_count": additional_link_count,
        "answer_claim_binding_count": len(answer_claim_bindings),
        "answer_claim_bindings": answer_claim_bindings[:20],
        "answer_evidence_binding_count": len(answer_evidence_bindings),
        "answer_evidence_bindings": answer_evidence_bindings[:20],
        "evidence_contract_status": _evidence_contract_status,
        "algorithm_report_blocked": algorithm_report_blocked,
        "algorithm_failure_count": len(s.get("algorithm_failures", []) or []),
        "blocking_algorithm_failure_count": len(blocking_algorithm_failures),
        "algorithm_failures": (s.get("algorithm_failures", []) or [])[-10:],
        "section_logs": section_logs,
        "composition_aborted": composition_result.composition_aborted,
        "consecutive_empty_at_break": composition_result.consecutive_empty_at_break,
        "sections_planned": composition_result.sections_planned,
        "sections_attempted": composition_result.sections_attempted,
        "sections_rendered": len(section_logs),
        "answer_finish_reason": finish_reason,
        "answer_incomplete": bool(incomplete_reasons),
        "answer_incomplete_reasons": incomplete_reasons,
        "appended_sources_footer": appended_sources_footer,
        "fallback_model": fallback_model or "",
        "fallback_attempted": fallback_attempted,
        "fallback_succeeded": fallback_succeeded,
        "stats_line": stats_line,
        "rounds": n_rounds,
        "elapsed_total_s": round(elapsed, 1),
        "confidence": conf,
    }, testing_mode=settings.testing_mode)

    emit_progress(s, "done")
    return s
