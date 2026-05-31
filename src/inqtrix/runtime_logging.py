"""Helpers for structured runtime logging.

Keeps provider/model metadata extraction and iteration-entry formatting in
one place so normal runtime logs and testing-mode iteration logs stay
semantically aligned.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from inqtrix.search_result import GroundedSearchResult
from inqtrix.settings import AgentSettings
from inqtrix.urls import domain_from_url, normalize_url, sanitize_error

log = logging.getLogger("inqtrix")

_SENSITIVE_EVENT_KEYS = frozenset({
    "access_token",
    "api_key",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "env",
    "headers",
    "password",
    "raw",
    "raw_response",
    "request_body",
    "request_headers",
    "request_kwargs",
    "secret",
    "session_token",
    "x_api_key",
    "x_key",
})

# Generous bound for evidence-content fields in the forensic log. State keeps
# the full text; this only prevents a single log line from growing unbounded.
_LOG_CONTENT_CAP = 20000

_TEXT_CAPS = {
    "answer_segment_preview": 500,
    "answer_preview": 800,
    "claim_prompt_view": 4000,
    "report_evidence_prompt": 6000,
    "unverified_evidence_prompt": 4000,
    "title": 180,
    "content_preview": 500,
    "context_preview": 1200,
    "query": 240,
    "status_reason": 180,
    # Evidence content is uncapped in state; the forensic log mirrors it with a
    # generous bound so logged records are not misleadingly thinner than state.
    "evidence_snippet": _LOG_CONTENT_CAP,
    "snippet": _LOG_CONTENT_CAP,
    "claim_text": _LOG_CONTENT_CAP,
    "summary": _LOG_CONTENT_CAP,
    "provider_answer": _LOG_CONTENT_CAP,
    "text": _LOG_CONTENT_CAP,
    "as_reported": _LOG_CONTENT_CAP,
}

# Common envelope fields added to every event by the iteration-log wrappers
# (see :func:`inqtrix.state.append_iteration_log`) and by
# :func:`emit_runtime_event` itself. They are always allowed in addition to
# the per-event allowlist.
_COMMON_EVENT_KEYS = frozenset({
    "event",
    "event_seq",
    "node",
    "run_id",
    "timestamp",
})

# Per-event allowlists for forensic events. Events without a schema entry
# fall back to the recursive drop-list + URL/secret redaction in
# :func:`_sanitize_value` (hybrid policy: strict allowlist for the
# forensic-lineage events, drop-list for ``iteration_summary`` headers).
_EVENT_SCHEMAS: dict[str, frozenset[str]] = {
    "run_start": frozenset({
        "run_id",
        "run_mode",
        "question_length",
        "history_length",
        "llm",
        "search",
        "settings",
    }),
    "run_end": frozenset({
        "run_id",
        "run_mode",
        "status",
        "reason",
        "elapsed_s",
        "round",
        "done",
        "cancelled",
        "final_confidence",
        "total_citations",
        "evidence_record_count",
        "consolidated_claims_count",
        "algorithm_failure_count",
        "blocking_algorithm_failure_count",
        "total_prompt_tokens",
        "total_completion_tokens",
    }),
    "algorithm_failure": frozenset({
        "phase",
        "reason",
        "message",
        "blocking",
        "round",
        "failed",
        "total",
        "attempts",
        "candidate_count",
        "claim_notice_samples",
        "has_citations",
        "has_ledger_context",
        "model",
        "capacity_phase",
        "context_window_tokens",
        "required_context_window_tokens",
        "estimated_input_tokens",
        "requested_output_tokens",
        "context_window_safety_tokens",
        "estimated_required_context_tokens",
    }),
    "node_model_resolution": frozenset({
        "node",
        "model",
        "tier",
        "effort",
        "model_source",
        "effort_source",
        "requested_tier",
    }),
    "node_model_resolution_warning": frozenset({
        "node",
        "model",
        "tier",
        "effort",
        "model_source",
        "effort_source",
        "requested_tier",
        "reason",
    }),
    "query_record": frozenset({
        "query_id",
        "round",
        "query_index",
        "query",
        "domain_filter",
        "provider",
        "source_ids",
        "citation_ids",
    }),
    "source_record": frozenset({
        "source_id",
        "url",
        "canonical_url",
        "domain",
        "provider",
        "first_seen_query_id",
        "first_seen_rank",
        "origin",
        "tier",
        "tier_reason",
        "access_status",
    }),
    "provider_citation_record": frozenset({
        "citation_id",
        "query_id",
        "source_id",
        "url",
        "canonical_url",
        "rank",
        "origin",
        "provider",
        "title",
        "snippet",
        "source_date",
        "last_updated",
        "source",
        "annotation_start",
        "annotation_end",
    }),
    "claim_record": frozenset({
        "raw_claim_id",
        "query_id",
        "evidence_ids",
        "signature",
        "claim_text",
        "evidence_snippet",
        "claim_type",
        "polarity",
        "needs_primary",
        "source_ids",
        "citation_ids",
        "provider_refs",
        "source_urls",
        "published_date",
        "round",
    }),
    "query_synthesis": frozenset({
        "query_id",
        "query",
        "round",
        "provider_answer",
        "citation_urls_by_rank",
    }),
    "claim_merge": frozenset({
        "claim_id",
        "signature",
        "member_claim_ids",
        "status",
        "status_reason",
        "verification_basis",
        "evidence_snippets",
        "support_count",
        "supporting_evidence_ids",
        "supporting_domain_count",
        "contradicting_evidence_ids",
        "contradict_count",
        "source_ids",
        "citation_ids",
        "source_urls",
        "round_first_seen",
        "round_last_updated",
    }),
    "evidence_record": frozenset({
        "evidence_id",
        "query_id",
        "query",
        "source_id",
        "citation_id",
        "canonical_url",
        "domain",
        "tier",
        "tier_reason",
        "provider",
        "source_title",
        "source_snippet",
        "source_date",
        "last_updated",
        "source_passages",
        "claims",
        "record_type",
        "report_eligible",
        "citation_set",
    }),
    "evidence_verification_projection": frozenset({
        "evidence_count",
        "verified_claim_supports",
        "contested_claim_supports",
        "unverified_claim_supports",
        "supporting_evidence_link_count",
    }),
    "evidence_selection": frozenset({
        "consolidated_claim_count",
        "verified_claim_count",
        "primary_supported_claim_count",
        "contested_claim_count",
        "cross_checked_claim_count",
        "single_source_verified_count",
        "report_eligible_evidence_count",
        "evidence_depth_gap",
    }),
    "score_snapshot": frozenset({
        "round",
        "phase",
        "source",
        "evidence",
        "claims",
        "coverage",
        "evaluate",
        "stop",
        "answer",
    }),
    "query_summary": frozenset({
        "query_id",
        "round",
        "query_index",
        "query",
        "answer_length",
        "claims_extracted",
        "claims_kept",
        "claims_sample",
        "claim_extraction_mode",
        "claim_extraction_schema",
        "claim_extraction_structured_supported",
        "claim_extraction_valid_empty",
        "claim_extraction_raw_claim_count",
        "claim_extraction_normalized_claim_count",
        "claim_extraction_filtered_claim_count",
        "evidence_record_count",
        "evidence_context_source_count",
        "source_ids",
        "citation_ids",
        "urls",
        "prompt_tokens",
        "completion_tokens",
        "provider_notice",
        "claim_notice",
        "domain_filter",
        "related_question_count",
    }),
    "stop_cascade": frozenset({
        "confidence",
        "confidence_stop_target",
        "round",
        "max_rounds",
        "min_rounds",
        "utility_score",
        "utility_stop",
        "plateau_stop",
        "stagnation_detected",
        "falsification_triggered",
        "done_after_utility",
        "done_after_plateau",
        "confidence_stop",
        "round_limit",
        "evidence_depth_gap_active",
        "evidence_depth_gap",
        "report_eligible_evidence_count",
        "min_report_eligible_evidence",
        "suppressed_by_min_rounds",
        "suppressed_by_report_evidence",
        "suppressed_stop_reason",
        "final_stop_reason",
        "final_done",
    }),
    "citation_selection": frozenset({
        "allowed_citations",
        "allowed_citation_count",
        "allowed_link_count",
        "removed_non_allowed_links",
        "answer_claim_binding_count",
        "answer_evidence_binding_count",
        "matched_evidence_binding_count",
        "unknown_citation_count",
        "expanded_evidence_label_links",
    }),
    "answer_prompt_inputs": frozenset({
        "report_profile",
        "model",
        "evidence_record_count",
        "rendered_evidence_record_count",
        "omitted_evidence_record_count",
        "evidence_overview_chars",
        "visible_evidence_label_count",
        "allowed_citation_count",
        "consolidated_claim_count",
        "algorithm_failure_count",
        "blocking_algorithm_failure_count",
        "evidence_overview",
        "section_plan",
    }),
    "answer_section": frozenset({
        "heading",
        "position",
        "model",
        "content_length",
        "content_preview",
        "finish_reason",
        "limit_hit",
        "incomplete",
        "incomplete_reasons",
        "prompt_tokens",
        "completion_tokens",
        "request_max_tokens",
        "max_output_tokens",
        "estimated_input_tokens",
        "requested_output_tokens",
        "estimated_required_context_tokens",
        "context_window_tokens",
        "required_context_window_tokens",
        "system_prompt_chars",
        "user_prompt_chars",
        "section_focus_record_count",
        "section_focus_labels",
        "section_allowed_citation_count",
        "section_scoped_evidence",
        "used_evidence_labels",
        "token_utilization",
        "thinking_likely_active",
        "visible_tokens_estimate",
        "thinking_tokens_estimate",
    }),
    "answer_claim_binding": frozenset({
        "binding_id",
        "answer_segment_id",
        "answer_segment_preview",
        "citation_url",
        "source_id",
        "citation_id",
        "claim_id",
        "claim_status",
        "binding_status",
    }),
    "answer_sentence_audit": frozenset({
        "binding_id",
        "citation_url",
        "evidence_id",
        "verification",
        "binding_status",
    }),
}


@dataclass(frozen=True, slots=True)
class ProviderCitationRecord:
    """Normalized provider-side citation provenance.

    These records capture the provider-neutral link between a query and a
    source URL without retaining raw SDK payloads, request headers, or
    credentials. Providers supply :class:`GroundedSource` rows on the
    typed :class:`GroundedSearchResult`; those rows are normalized into
    this stable event shape by :func:`normalize_source_provenance`.
    """

    citation_id: str
    query_id: str
    source_id: str
    url: str
    canonical_url: str
    rank: int
    origin: str
    provider: str
    title: str = ""
    snippet: str = ""
    source_date: str = ""
    last_updated: str = ""
    source: str = ""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Run-local source registry record.

    The registry de-duplicates URLs by canonical URL and stores only
    allowlisted forensic metadata: domain, tier, tier reason, provider,
    and first-seen query. It intentionally does not include provider
    request bodies, SDK response objects, headers, or credential-bearing
    configuration.
    """

    source_id: str
    url: str
    canonical_url: str
    domain: str
    provider: str
    first_seen_query_id: str
    first_seen_rank: int
    origin: str
    tier: str = "unknown"
    tier_reason: str = ""
    access_status: str = "answer"


def _serialize_payload(payload: dict[str, Any]) -> str:
    """Render a pre-sanitized payload as stable JSON for debug logs.

    The caller is responsible for sanitising the payload (via
    :func:`sanitize_event_payload`) before invoking this helper so that the
    ``_RedactSecretsFilter`` in ``logging_config.py`` does not corrupt the
    JSON structure.  The filter's ``sanitize_error`` regex
    ``https?://[^\\s]+`` can consume trailing quotes, turning
    ``"url": "https://x"`` into ``"url": "[URL]`` (missing closing quote).
    Pre-sanitising replaces URLs while the value is still a plain string,
    keeping JSON delimiters intact.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def new_run_id() -> str:
    """Return an opaque, log-safe identifier for one research run."""
    return f"run_{uuid.uuid4().hex}"


def make_record_id(prefix: str, *parts: Any) -> str:
    """Build a stable short id from semantic record parts."""
    raw = "|".join(str(part) for part in parts)
    digest = sha1(raw.encode("utf-8")).hexdigest()[:14]
    clean_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", prefix).strip("_") or "id"
    return f"{clean_prefix}_{digest}"


def _is_sensitive_event_key(key: object) -> bool:
    """Return whether a payload key is unsafe for structured logs."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return normalized in _SENSITIVE_EVENT_KEYS


def _cap_text(key: object, text: str) -> str:
    cap = _TEXT_CAPS.get(str(key), 0)
    if cap <= 0 or len(text) <= cap:
        return text
    return f"{text[:cap].rstrip()}..."


def format_log_excerpt(text: Any, *, limit: int = 300) -> str:
    """Return a compact, word-boundary excerpt for human-facing log lines.

    Structured event payloads keep the longer sanitized value in the same
    logfile. This helper is only for INFO-level trace lines where a hard
    slice would otherwise cut German words in the middle and hide that the
    text was abbreviated.
    """
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    if limit <= 4:
        return normalized[:limit]
    boundary = normalized.rfind(" ", 0, limit - 4)
    if boundary < max(20, limit // 2):
        boundary = limit - 4
    return normalized[:boundary].rstrip() + " ..."


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize values before JSON serialization."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, val in value.items():
            if _is_sensitive_event_key(key):
                continue
            sanitized[str(key)] = _sanitize_value(_cap_text(key, val) if isinstance(val, str) else val)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, (str, Exception)):
        return sanitize_error(value)
    return value


def sanitize_event_payload(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized copy of *payload* ready for logs and exports.

    Two-layer policy:

    1. If *event* has a registered allowlist in :data:`_EVENT_SCHEMAS`, the
       top-level keys are filtered down to that allowlist (plus the common
       envelope fields ``event``/``event_seq``/``node``/``run_id``/
       ``timestamp``). Unknown keys, including credential-bearing
       provider fields like ``request_kwargs`` or ``raw_response``, are
       dropped before any value is inspected.
    2. The result is then passed through :func:`_sanitize_value`, which
       recursively drops sensitive keys, redacts URL query parameters
       such as ``api_key``/``token``/``sig``, and applies length caps to
       known free-text fields.

    Events without a schema entry (notably ``iteration_summary``) skip
    step 1 and rely solely on step 2. This is the hybrid policy: strict
    schemas for forensic-lineage events, drop-list defense-in-depth
    everywhere else.
    """
    schema = _EVENT_SCHEMAS.get(event)
    if schema is not None:
        allowed = schema | _COMMON_EVENT_KEYS
        payload = {k: v for k, v in payload.items() if k in allowed}
    sanitized = _sanitize_value(payload)
    return sanitized if isinstance(sanitized, dict) else {}


def emit_runtime_event(
    event: str,
    payload: dict[str, Any],
    *,
    prefix: str | None = None,
    level: int = logging.DEBUG,
) -> None:
    """Emit one sanitized runtime event through the existing logger.

    The function is the central structured-log path. It does not install
    handlers or create a parallel sink; it only formats an allowlisted,
    recursively sanitized JSON payload and lets ``logging_config`` apply
    the existing handler-level redaction once more.
    """
    if not log.isEnabledFor(level):
        return
    event_payload = dict(payload)
    event_payload.setdefault("event", event)
    sanitized = sanitize_event_payload(event, event_payload)
    message_prefix = prefix or f"EVENT {event}"
    log.log(level, "%s: %s", message_prefix, _serialize_payload(sanitized))


def normalize_source_provenance(
    result: GroundedSearchResult,
    *,
    query_id: str,
    provider: str,
    tier_explanations: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build provider-neutral source and citation records for a search result.

    Args:
        result: A :class:`~inqtrix.search_result.GroundedSearchResult`. Each
            :class:`~inqtrix.search_result.GroundedSource` becomes one
            citation record and, de-duplicated by canonical URL, one source
            record. Per-source snippets are stored in full -- no cap.
        query_id: Stable ID of the query that produced the result.
        provider: Provider class label used for operator-facing lineage.
        tier_explanations: Optional mapping from canonical URL to
            ``{"tier": ..., "tier_reason": ...}``.

    Returns:
        Tuple of ``(source_records, citation_records)`` represented as
        plain dicts ready for state storage and structured logging.
    """
    grounded = result
    tier_explanations = tier_explanations or {}
    source_records: list[dict[str, Any]] = []
    citation_records: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    access_status = "answer" if grounded.answer else "empty_answer"

    for index, source in enumerate(grounded.sources, start=1):
        url = str(source.url or "").strip()
        canonical_url = normalize_url(url)
        if not canonical_url:
            continue
        rank = int(source.rank or index)
        origin = str(source.origin or "provider_source")
        source_id = make_record_id("src", canonical_url)
        citation_id = make_record_id("cit", query_id, rank, canonical_url, origin)
        explanation = tier_explanations.get(canonical_url, {})

        if source_id not in seen_sources:
            seen_sources.add(source_id)
            source_records.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "canonical_url": canonical_url,
                    "domain": domain_from_url(canonical_url),
                    "provider": provider,
                    "first_seen_query_id": query_id,
                    "first_seen_rank": rank,
                    "origin": origin,
                    "tier": explanation.get("tier", "unknown"),
                    "tier_reason": explanation.get("tier_reason", ""),
                    "access_status": access_status,
                }
            )

        citation_records.append(
            {
                "citation_id": citation_id,
                "query_id": query_id,
                "source_id": source_id,
                "url": url,
                "canonical_url": canonical_url,
                "rank": rank,
                "origin": origin,
                "provider": provider,
                "title": source.title,
                "snippet": source.snippet,
                "source_date": str(source.date or "")[:80],
                "last_updated": str(source.last_updated or "")[:80],
            }
        )

    return source_records, citation_records


def forensic_enabled(settings: AgentSettings) -> bool:
    """Return whether detailed lineage events should be emitted."""
    return str(getattr(settings, "observability_profile", "summary")).lower() == "forensic"


def _unwrap_provider(provider: object) -> object:
    """Return the wrapped provider for ConfiguredLLMProvider-like adapters."""
    if type(provider).__name__ == "ConfiguredLLMProvider" and hasattr(provider, "_provider"):
        return getattr(provider, "_provider")
    return provider


def _clean_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    return value


def describe_llm_provider(provider: object) -> dict[str, Any]:
    """Extract human-readable runtime metadata for the active LLM provider."""
    resolved = _unwrap_provider(provider)
    metadata: dict[str, Any] = {
        "provider": type(resolved).__name__,
    }

    models = getattr(provider, "models", None)
    if models is not None:
        metadata["reasoning_model"] = str(getattr(models, "reasoning_model", "") or "")
        metadata["classify_model"] = str(
            getattr(models, "effective_classify_model", "")
            or getattr(models, "classify_model", "")
            or getattr(models, "reasoning_model", "")
            or ""
        )
        metadata["claim_extract_model"] = str(
            getattr(models, "effective_claim_extract_model", "")
            or getattr(models, "claim_extract_model", "")
            or getattr(models, "reasoning_model", "")
            or ""
        )
        metadata["evaluate_model"] = str(
            getattr(models, "effective_evaluate_model", "")
            or getattr(models, "evaluate_model", "")
            or getattr(models, "reasoning_model", "")
            or ""
        )

    for attr_name in (
        "_default_max_tokens",
        "_context_window_tokens",
        "_token_budget_parameter",
        "_temperature",
    ):
        cleaned = _clean_value(getattr(resolved, attr_name, None))
        if cleaned is not None:
            metadata[attr_name.lstrip("_")] = cleaned

    thinking = getattr(resolved, "_thinking", None)
    if isinstance(thinking, dict) and thinking:
        metadata["thinking"] = dict(thinking)

    return metadata


def describe_search_provider(provider: object) -> dict[str, Any]:
    """Extract human-readable runtime metadata for the active search provider."""
    resolved = _unwrap_provider(provider)
    metadata: dict[str, Any] = {
        "provider": type(resolved).__name__,
    }

    search_model = _clean_value(getattr(resolved, "search_model", None))
    model = _clean_value(getattr(resolved, "_model", None))
    agent_name = _clean_value(getattr(resolved, "_agent_name", None))
    agent_version = _clean_value(getattr(resolved, "_agent_version", None))
    agent_id = _clean_value(getattr(resolved, "_agent_id", None))

    if search_model is not None:
        metadata["engine"] = str(search_model)
    elif model is not None:
        metadata["engine"] = str(model)
    elif agent_name is not None:
        engine = str(agent_name)
        if agent_version is not None:
            engine = f"{engine}@{agent_version}"
        metadata["engine"] = engine
        metadata["agent_name"] = str(agent_name)
        if agent_version is not None:
            metadata["agent_version"] = str(agent_version)
    elif agent_id is not None:
        metadata["engine"] = str(agent_id)
        metadata["agent_id"] = str(agent_id)

    return metadata


def build_run_metadata(
    *,
    question: str,
    history: str,
    providers: Any,
    settings: AgentSettings,
    run_mode: str = "run",
) -> dict[str, Any]:
    """Build a structured payload for the start of a research run."""
    return {
        "event": "run_start",
        "run_mode": run_mode,
        "question_length": len(question or ""),
        "history_length": len(history or ""),
        "llm": describe_llm_provider(getattr(providers, "llm", None)),
        "search": describe_search_provider(getattr(providers, "search", None)),
        "settings": {
            "report_profile": str(settings.report_profile),
            "observability_profile": str(getattr(settings, "observability_profile", "summary")),
            "max_rounds": settings.max_rounds,
            "confidence_stop": settings.confidence_stop,
            "required_context_window_tokens": settings.required_context_window_tokens,
            "max_total_seconds": settings.max_total_seconds,
            "testing_mode": settings.testing_mode,
        },
    }


def log_run_start(
    *,
    question: str,
    history: str,
    providers: Any,
    settings: AgentSettings,
    run_mode: str = "run",
    run_id: str | None = None,
) -> None:
    """Write a compact start banner plus structured debug metadata."""
    metadata = build_run_metadata(
        question=question,
        history=history,
        providers=providers,
        settings=settings,
        run_mode=run_mode,
    )
    if run_id:
        metadata["run_id"] = run_id

    llm = metadata["llm"]
    search = metadata["search"]
    run_settings = metadata["settings"]

    log.info(
        "RUN start: id=%s mode=%s profile=%s observability=%s llm=%s reasoning=%s classify=%s evaluate=%s claim_extract=%s default_max_tokens=%s context_window_tokens=%s required_context_window_tokens=%s search=%s engine=%s max_rounds=%d confidence_stop=%d max_total_seconds=%d testing_mode=%s question_len=%d history_len=%d",
        metadata.get("run_id") or "-",
        run_mode,
        run_settings.get("report_profile") or "compact",
        run_settings.get("observability_profile") or "summary",
        llm.get("provider") or "unknown",
        llm.get("reasoning_model") or "-",
        llm.get("classify_model") or "-",
        llm.get("evaluate_model") or "-",
        llm.get("claim_extract_model") or "-",
        llm.get("default_max_tokens") if llm.get("default_max_tokens") is not None else "-",
        llm.get("context_window_tokens") if llm.get("context_window_tokens") is not None else "-",
        run_settings.get("required_context_window_tokens") or "-",
        search.get("provider") or "unknown",
        search.get("engine") or "-",
        run_settings["max_rounds"],
        run_settings["confidence_stop"],
        run_settings["max_total_seconds"],
        run_settings["testing_mode"],
        metadata["question_length"],
        metadata["history_length"],
    )
    emit_runtime_event("run_start", metadata, prefix="RUN metadata")


def log_run_end(
    *,
    run_id: str | None,
    run_mode: str,
    status: str,
    elapsed_s: float,
    state: dict[str, Any],
    reason: str = "",
) -> None:
    """Write a normalized run-end event through the existing logger."""
    payload = {
        "event": "run_end",
        "run_id": run_id or state.get("_run_id", ""),
        "run_mode": run_mode,
        "status": status,
        "reason": reason or state.get("_stop_reason", ""),
        "elapsed_s": round(elapsed_s, 3),
        "round": state.get("round", 0),
        "done": bool(state.get("done", False)),
        "cancelled": bool(state.get("cancelled", False)),
        "final_confidence": state.get("final_confidence", 0),
        "total_citations": len(state.get("all_citations", []) or []),
        "evidence_record_count": len(state.get("evidence_ledger", []) or []),
        "consolidated_claims_count": len(state.get("consolidated_claims", []) or []),
        "algorithm_failure_count": len(state.get("algorithm_failures", []) or []),
        "blocking_algorithm_failure_count": sum(
            1
            for failure in state.get("algorithm_failures", []) or []
            if isinstance(failure, dict) and bool(failure.get("blocking"))
        ),
        "total_prompt_tokens": state.get("total_prompt_tokens", 0),
        "total_completion_tokens": state.get("total_completion_tokens", 0),
    }
    log.info(
        "RUN end: id=%s mode=%s status=%s reason=%s elapsed=%.1fs round=%s confidence=%s citations=%s",
        payload["run_id"] or "-",
        run_mode,
        status,
        payload["reason"] or "-",
        elapsed_s,
        payload["round"],
        payload["final_confidence"],
        payload["total_citations"],
    )
    emit_runtime_event("run_end", payload, prefix="RUN metadata")


def log_iteration_entry(entry: dict[str, Any]) -> None:
    """Write a structured iteration payload to DEBUG logs."""
    node = str(entry.get("node", "unknown"))
    emit_runtime_event(
        str(entry.get("event", "iteration_summary")),
        entry,
        prefix=f"ITERATION {node}",
    )
