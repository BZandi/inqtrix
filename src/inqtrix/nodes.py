"""State machine node functions for the research agent.

Each node takes the state dict plus providers and strategies as keyword
arguments.  The five nodes correspond to the five phases of the research
loop: classify, plan, search, evaluate, answer.

Extracted from ``_original_agent.py`` and adapted to use the provider /
strategy / settings abstractions defined in the ``inqtrix`` package.
"""

from dataclasses import dataclass
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAIError

from inqtrix.domains import LANG_NAMES, LOW_QUALITY_DOMAINS, is_de_policy_question
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AnthropicAPIError, BedrockAPIError
from inqtrix.json_helpers import parse_json_object, parse_json_string_list
from inqtrix.prompts import (
    EVALUATE_FORMAT_SUFFIX,
    SUMMARIZE_PROMPT_DEEP,
    build_answer_section_system_prompt,
    build_answer_section_user_prompt,
)
from inqtrix.providers.base import (
    LLMResponse,
    ProviderContext,
    SummarizeOptions,
    _check_deadline,
    get_search_provider_capabilities,
)
from inqtrix.report_profiles import ReportProfile
from inqtrix.settings import AgentSettings
from inqtrix.state import append_iteration_log, check_cancel_event, emit_progress, track_tokens
from inqtrix.strategies import StrategyContext
from inqtrix.text import is_none_value, tokenize
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


@dataclass(slots=True)
class _AnswerCompositionResult:
    answer: str
    finish_reason: str
    section_logs: list[dict[str, Any]]
    composition_aborted: bool = False
    consecutive_empty_at_break: int = 0
    sections_planned: int = 0
    sections_attempted: int = 0


def _limit_prompt_citations_by_char_budget(
    citations: list[str],
    *,
    char_budget: int | None,
) -> list[str]:
    """Trim the numbered prompt citation map to a bounded character budget."""
    if char_budget is None or char_budget <= 0 or len(citations) <= 1:
        return citations

    selected: list[str] = []
    used_chars = 0
    for idx, url in enumerate(citations, 1):
        line = f"[{idx}]: {url}\n"
        remaining = len(citations) - idx
        overflow_line = (
            f"... ({remaining} weitere recherchierte Quellen nicht im Prompt-Set)\n"
            if remaining > 0
            else ""
        )
        projected = used_chars + len(line) + len(overflow_line)
        if selected and projected > char_budget:
            break
        selected.append(url)
        used_chars += len(line)

    return selected or citations[:1]


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
) -> _AnswerCompositionResult:
    """Compose the final answer section by section to avoid hard output truncation."""
    rendered_sections: list[str] = []
    completed_headings: list[str] = []
    section_logs: list[dict[str, Any]] = []
    finish_reason = ""
    answer_sections = tuple(settings.report_tuning.answer_sections)
    consecutive_empty = 0
    _MAX_CONSECUTIVE_EMPTY = 2
    composition_aborted = False
    consecutive_empty_at_break = 0
    sections_attempted = 0

    for index, section_spec in enumerate(answer_sections, 1):
        sections_attempted = index
        section_token_limit = section_spec.max_output_tokens
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
        )
        response = _llm_complete_with_metadata(
            providers.llm,
            section_prompt,
            system=section_system,
            deadline=s["deadline"],
            state=s,
            model=model,
            max_output_tokens=section_token_limit,
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
        # Provider-reported effective max_tokens (post-clamp). For Anthropic /
        # Bedrock with thinking this is the auto-raised value (>= 16k), not
        # the smaller limit the section spec asked for. Falls back to the
        # spec when the provider does not report it.
        section_request_max_tokens = (
            int(getattr(response, "request_max_tokens", 0) or 0) or section_token_limit
        )
        # Utilization against the budget that was actually applied.
        section_token_utilization = (
            round(section_completion_tokens / section_request_max_tokens, 3)
            if section_request_max_tokens
            else 0.0
        )
        # Rough visible-vs-thinking decomposition. Only meaningful when the
        # provider actually ran a thinking pass — detected by comparing the
        # post-clamp ``request_max_tokens`` against the caller-supplied
        # ``section_token_limit``. If the provider auto-raised the budget
        # (Anthropic / Bedrock with thinking → 16k floor), thinking tokens
        # are likely embedded in ``completion_tokens`` and the visible/think
        # split is informative. Otherwise the difference is just tokenizer
        # overhead and the estimate would mislead.
        thinking_likely_active = section_request_max_tokens > section_token_limit
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
            "max_output_tokens": section_token_limit,
            "request_max_tokens": section_request_max_tokens,
            "required": section_spec.required,
            "model": str(model or ""),
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
        }
        if thinking_likely_active:
            section_log_entry["thinking_tokens_estimate"] = thinking_tokens_estimate
        section_logs.append(section_log_entry)
        if limit_hit:
            emit_progress(
                s,
                f"Warnung: Abschnitt '{section_spec.heading}' hat Token-Limit "
                f"erreicht (finish_reason={finish_reason}, request_max_tokens={section_request_max_tokens}, "
                f"completion_tokens={section_completion_tokens})",
            )
            log.warning(
                "TRACE section %d/%d '%s': token limit hit "
                "(finish_reason=%s, request_max_tokens=%d, max_output_tokens=%d, "
                "completion_tokens=%d, content_length=%d)",
                index,
                len(answer_sections),
                section_spec.heading,
                finish_reason,
                section_request_max_tokens,
                section_token_limit,
                section_completion_tokens,
                len(section_body),
            )
        if section_reasons:
            emit_progress(
                s,
                f"Warnung: Abschnitt '{section_spec.heading}' zeigt "
                f"Trunkierungsanzeichen: {', '.join(section_reasons)}",
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
            rendered_sections.append(f"## {section_spec.heading}\n\n{section_body}")
            completed_headings.append(section_spec.heading)
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY:
                composition_aborted = True
                consecutive_empty_at_break = consecutive_empty
                emit_progress(
                    s,
                    f"Antwort-Synthese abgebrochen: {consecutive_empty} aufeinanderfolgende leere "
                    f"Sections (zuletzt '{section_spec.heading}')",
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
    prompt_citations: list[str],
    all_citations: list[str],
    strategies: StrategyContext,
    incomplete_reasons: list[str],
    finish_reason: str,
) -> tuple[list[str], int, int]:
    """Build optional post-answer sections without affecting the stats footer."""
    sections: list[str] = []

    prompt_citation_urls = {
        normalize_url(url)
        for url in prompt_citations
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

    reference_entries = _extract_used_reference_links(answer, prompt_citation_urls)
    if not reference_entries and incomplete_reasons and prompt_citation_urls:
        reference_entries = [
            (str(index), normalize_url(url))
            for index, url in enumerate(prompt_citations, 1)
            if normalize_url(url)
        ]

    reference_urls: set[str] = {url for _, url in reference_entries}

    if reference_entries:
        sections.append(_format_reference_entries(reference_entries))
    else:
        sections.append(
            "## Referenzen\nKeine zitatgebundenen Markdown-Links im Antworttext gefunden."
        )

    additional_links = _select_additional_links(
        all_citations,
        excluded_urls=reference_urls,
        prompt_citation_urls=prompt_citation_urls,
        strategies=strategies,
    )
    if additional_links:
        sections.append(_format_additional_links(additional_links, strategies))

    return sections, len(reference_entries), len(additional_links)


@dataclass(slots=True)
class FollowupResolution:
    """Decision how to treat the FOLLOWUP marker emitted by classify.

    Attributes:
        mode: One of ``"deepening"`` (reuse seeded data, merge new
            aspects), ``"new_topic"`` (clear seeded data, fresh
            aspects), or ``"fresh"`` (no session was seeded).
        merge_aspects: True when classify should add new aspects to
            the existing list instead of overwriting.
        reset_citations: True when classify should clear the seeded
            research data (citations, context, claims, queries).
    """

    mode: str
    merge_aspects: bool
    reset_citations: bool


def resolve_followup_reset(s: dict, classify_response: str) -> FollowupResolution:
    """Decide how a follow-up marker should reshape the initial state.

    This function is pure: it reads the current follow-up flag and the
    LLM's raw classify response and returns a resolution describing
    whether the next steps of ``classify`` should merge aspects into a
    seeded session, reset the seeded data entirely, or proceed as a
    fresh first-time run.

    Args:
        s: The current agent state dict. Only ``_is_followup`` is read.
        classify_response: The raw text response emitted by the
            classify LLM call. Parsed for the ``FOLLOWUP: YES|NO``
            marker (German ``JA|NEIN`` is also accepted).

    Returns:
        FollowupResolution: Deterministic instructions for the
        downstream apply step.
    """
    if not s.get("_is_followup"):
        return FollowupResolution(mode="fresh", merge_aspects=False, reset_citations=False)
    m_followup = re.search(r"FOLLOWUP:\s*(YES|NO|JA|NEIN)", classify_response, re.IGNORECASE)
    is_followup = bool(m_followup and m_followup.group(1).upper() in ("YES", "JA"))
    if is_followup:
        return FollowupResolution(mode="deepening", merge_aspects=True, reset_citations=False)
    return FollowupResolution(mode="new_topic", merge_aspects=False, reset_citations=True)


def _reset_seeded_research(s: dict) -> None:
    """Clear session-seeded research data when a new topic is detected."""
    s["all_citations"] = []
    s["context"] = []
    s["consolidated_claims"] = []
    s["claim_ledger"] = []
    s["queries"] = []
    s["source_tier_counts"] = {
        "primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0}
    s["source_quality_score"] = 0.0
    s["claim_status_counts"] = {"verified": 0, "contested": 0, "unverified": 0}
    s["claim_quality_score"] = 0.0
    s["_prev_question"] = ""
    s["_prev_answer"] = ""


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
            follow-up markers, and deadline; writes language, query type,
            risk flags, aspect hints, and reset markers for new topics.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for risk scoring and downstream
            claim/context handling.
        settings: Agent behavior settings used for risk escalation and
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
    emit_progress(s, "Analysiere Frage...")
    _t0 = time.monotonic()
    _followup_seeded = bool(s.get("_is_followup"))
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
            emit_progress(s, f"Hinweis: {_warning}")
            log.warning("CONFIG: %s", _warning)
    s["risk_score"] = strategies.risk_scoring.score(s["question"])
    s["high_risk"] = s["risk_score"] >= settings.high_risk_score_threshold
    classify_model = (
        providers.llm.models.reasoning_model
        if (s["high_risk"] and settings.high_risk_classify_escalate)
        else providers.llm.models.effective_classify_model
    )

    # Follow-up context for the classify prompt
    _followup_prompt_ctx = ""
    if s.get("_is_followup") and s.get("_prev_question"):
        _followup_prompt_ctx = (
            f"=== KONTEXT: VORHERIGE RECHERCHE ===\n"
            f"Vorherige Frage: {s['_prev_question']}\n"
            f"Vorherige Antwort (Auszug): {s['_prev_answer'][:300]}\n"
            f"Der Nutzer stellt jetzt eine neue Frage. Bestimme ob sie sich "
            f"thematisch auf die vorherige Recherche bezieht (Follow-up/Vertiefung) "
            f"oder ein komplett neues Thema ist.\n\n"
        )

    try:
        _check_deadline(s["deadline"])
        d = providers.llm.complete(
            f"Heutiges Datum: {today()}\n\n"
            f"{_followup_prompt_ctx}"
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
            + (f"FOLLOWUP: YES oder NO (bezieht sich die Frage auf die vorherige Recherche?)\n"
               if s.get("_is_followup") else "")
            + f"SUB_QUESTIONS: JSON Array von 1-3 Teilfragen als Strings",
            deadline=s["deadline"],
            model=classify_model,
            state=s,
        )
        s["done"] = bool(re.search(r"DECISION:\s*DIRECT", d, re.IGNORECASE))

        # Extract language
        m_lang = re.search(r"LANGUAGE:\s*(\w+)", d)
        s["language"] = m_lang.group(1).strip().lower()[:2] if m_lang else "de"

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
        followup_resolution = resolve_followup_reset(s, d)

        if followup_resolution.mode == "deepening":
            log.info("TRACE classify: follow-up detected, keeping seeded research data")
            emit_progress(s, "Vertiefungsfrage erkannt — nutze bisherige Recherche")
            existing = set(s.get("required_aspects", []))
            for asp in new_aspects:
                if asp not in existing:
                    s["required_aspects"].append(asp)
                    s["uncovered_aspects"].append(asp)
            if s["required_aspects"]:
                covered = len(s["required_aspects"]) - len(s["uncovered_aspects"])
                s["aspect_coverage"] = max(0.0, covered / len(s["required_aspects"]))
        else:
            if followup_resolution.reset_citations:
                log.info("TRACE classify: new topic detected, clearing seeded data")
                emit_progress(s, "Neues Thema erkannt — starte frische Recherche")
                _reset_seeded_research(s)
            s["required_aspects"] = list(new_aspects)
            s["uncovered_aspects"] = list(new_aspects)
            s["aspect_coverage"] = 0.0

        if s.get("_is_followup"):
            # One-shot flag -- reset after resolution is applied
            s["_is_followup"] = False

        # Trace logging
        log.info(
            "TRACE classify: decision=%s lang=%s search_lang=%s recency=%s type=%s sub_q=%d risk=%d high_risk=%s model=%s",
            "DIRECT" if s["done"] else "SEARCH",
            s["language"], s["search_language"], s["recency"] or "NONE", s["query_type"],
            len(s["sub_questions"]), s["risk_score"], s["high_risk"], classify_model,
        )

        if s["done"]:
            emit_progress(s, "Direktantwort ohne Websuche")
        else:
            hints: list[str] = []
            if s["search_language"] != s["language"]:
                hints.append(
                    f"Suche auf {LANG_NAMES.get(s['search_language'], s['search_language'])}")
            if s["recency"]:
                recency_labels = {
                    "hour": "letzte Stunde",
                    "day": "heute",
                    "week": "diese Woche",
                    "month": "diesen Monat",
                }
                hints.append(f"Aktualitaet: {recency_labels.get(s['recency'], s['recency'])}")
            if s["query_type"] != "general":
                type_labels = {"academic": "Akademisch", "news": "Nachrichten"}
                hints.append(type_labels.get(s["query_type"], s["query_type"]))

            hint_str = f" ({', '.join(hints)})" if hints else ""
            emit_progress(s, f"Websuche erforderlich{hint_str}")

            if len(s["sub_questions"]) > 1:
                sub_q_display = ", ".join(
                    f"'{q}'" for q in s["sub_questions"][:3]
                )
                emit_progress(
                    s,
                    f"Frage in {len(s['sub_questions'])} Teilfragen zerlegt: "
                    f"{sub_q_display}",
                )
    except AgentRateLimited:
        raise
    except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError) as exc:
        # Fail-safe: on classification error do NOT fall back to direct answer.
        # Conservatively continue researching with robust defaults — but make
        # the fallback visible in progress, logs and the iteration trace.
        _exc_label = type(exc).__name__
        log.warning(
            "Classify-Fallback aktiviert (%s): %s — nutze deterministische Defaults",
            _exc_label, exc,
        )
        emit_progress(
            s,
            f"Klassifikation fehlgeschlagen ({_exc_label}) — nutze konservative Defaults",
        )
        s["done"] = False
        s["language"] = s.get("language") or "de"
        s["search_language"] = s.get("search_language") or s["language"]
        s["query_type"] = strategies.risk_scoring.infer_query_type(s["question"])
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
        "sub_questions": s["sub_questions"],
        "sub_question_count": len(s["sub_questions"]),
        "risk_score": s["risk_score"],
        "high_risk": s["high_risk"],
        "followup_seeded": _followup_seeded,
        "model": classify_model,
        "required_aspects": s.get("required_aspects", []),
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
    emit_progress(s, f"Plane Suchanfragen (Runde {s['round'] + 1}/{settings.max_rounds})...")
    _t0 = time.monotonic()
    _plan_fallback: dict[str, Any] = {}
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

        # Use related questions from Perplexity as inspiration
        related_info = ""
        if s["related_questions"]:
            related_info = (
                f"Verwandte Fragen (von der Suchmaschine vorgeschlagen):\n"
                f"{json.dumps(s['related_questions'][:5], ensure_ascii=False)}\n"
                f"Nutze diese als Inspiration, aber kopiere sie nicht woertlich.\n"
            )

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
            falsification_instruction = (
                "FALSIFIKATIONS-MODUS AKTIV:\n"
                "Die bisherige Recherche hat wiederholt KEINE ueberzeugenden Belege fuer die "
                "Behauptung/Praemisse in der Frage gefunden. Jetzt suchen wir gezielt nach "
                "GEGEN-EVIDENZ um die Praemisse zu testen.\n\n"
                "Mindestens 2 deiner 3 Queries MUESSEN Falsifikations-Queries sein:\n"
                "- '[Behauptung] debunked' oder '[Behauptung] myth'\n"
                "- '[Thema] does not exist' oder '[Thema] never happened'\n"
                "- '[Thema] hoax' oder '[Thema] refuted'\n"
                "- '[Thema] misinformation' oder '[Thema] false claim'\n\n"
                "Die dritte Query MUSS nach dem TATSAECHLICH existierenden naechstliegenden "
                "Sachverhalt suchen: z.B. statt 'Gemini eingestellt' -> "
                "'Google Gemini aktueller Status 2026' oder 'Was hat [Organisation] "
                "TATSAECHLICH veroeffentlicht?'\n\n"
                "ZIEL: Entweder finden wir Belege dass die Praemisse falsch ist "
                "(-> hochkonfidente Antwort 'existiert nicht'), oder wir finden doch "
                "noch den richtigen Sachverhalt.\n\n"
            )

        # Follow-up context
        followup_plan_ctx = ""
        if s.get("_prev_question"):
            followup_plan_ctx = (
                f"KONTEXT: Dies ist eine Vertiefungsfrage zu: {s['_prev_question']}\n"
                f"Bisherige Recherche enthielt {len(s['context'])} Informationsbloecke "
                f"und {len(s['all_citations'])} Quellen.\n"
                f"Generiere Queries die GEZIELT die neue Frage beantworten, "
                f"nicht die bereits recherchierten Aspekte wiederholen.\n\n"
            )

        round_zero_query_instruction = (
            f"Erzeuge {max(2, settings.first_round_queries - 1)}-{settings.first_round_queries} diverse Suchqueries die verschiedene Aspekte, Hypothesen und Perspektiven der Frage breit abdecken"
            if s["round"] == 0
            else "Erzeuge 2-3 praezise, spezifische Suchqueries"
        )

        q = providers.llm.complete(
            f"Heutiges Datum: {today()}\n\n"
            f"{followup_plan_ctx}"
            f"{round_zero_query_instruction} fuer eine Websuche.\n"
            f"Jede Query sollte 5-15 Woerter lang sein und konkreten Kontext enthalten.\n"
            f"SCHLECHT: 'KI Entwicklung' (zu vage)\n"
            f"GUT: 'neueste Durchbrueche kuenstliche Intelligenz 2025 Sprachmodelle' (spezifisch)\n\n"
            f"{reformulation_instruction}"
            f"{falsification_instruction}"
            f"{competing_instruction}"
            f"{alternative_instruction}"
            f"{perspective_instruction}"
            f"{deep_review_instruction}"
            f"{lang_instruction}"
            f"Frage: {s['question']}\n"
            f"{sub_q_info}"
            f"{required_info}"
            f"{uncovered_info}"
            f"{gap_info}"
            f"{related_info}"
            f"Bisherige Queries: {s['queries']}\n"
            f"Bisherige Ergebnisse: {len(s['context'])} Informationsbloecke\n\n"
            f"Generiere Queries die NEUE Informationen liefern, nicht schon Bekanntes wiederholen.\n"
            f"Antworte NUR mit einem JSON Array von Strings. Beispiel: [\"query1\", \"query2\"]",
            deadline=s["deadline"],
            state=s,
        )
    except AgentRateLimited:
        raise
    except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError) as exc:
        _exc_label = type(exc).__name__
        log.warning(
            "Plan-Fallback aktiviert (%s, round=%d): %s — keine LLM-Queries verfuegbar",
            _exc_label, s["round"], exc,
        )
        emit_progress(
            s,
            f"Planung fehlgeschlagen ({_exc_label}) — verwende Original-Frage als Fallback-Query",
        )
        q = ""
        _plan_fallback = {
            "fallback": "plan_default",
            "fallback_reason": _exc_label,
            "fallback_message": str(exc)[:300],
        }

    _max_items = settings.first_round_queries if s["round"] == 0 else 3
    new_q = parse_json_string_list(
        q,
        fallback=[s["question"]],
        max_items=_max_items,
    )

    # Quality-chase: force at least one primary and mainstream query for
    # policy / news questions (DE). Centralised behind enable_de_policy_bias
    # so deployments outside German health/social policy can opt out.
    search_lang = s.get("search_language", s.get("language", "de"))
    is_policyish = (
        getattr(settings, "enable_de_policy_bias", True)
        and is_de_policy_question(s.get("question", "") or "")
    )
    if is_policyish and (search_lang or "").lower() == "de":
        tiers = s.get("source_tier_counts", {}) or {}
        need_primary = (
            s["round"] == 0
            or int(tiers.get("primary", 0)) == 0
            or int(s.get("claim_needs_primary_total", 0)) > int(s.get("claim_needs_primary_verified", 0))
        )
        need_mainstream = (
            s["round"] == 0
            or int(tiers.get("mainstream", 0)) == 0
            or float(s.get("claim_quality_score", 0.0) or 0.0) < 0.35
        )
        new_q = strategies.risk_scoring.inject_quality_site_queries(
            new_q,
            search_lang=search_lang,
            question=s.get("question", ""),
            query_type=s.get("query_type", "general"),
            need_primary=need_primary,
            need_mainstream=need_mainstream,
            max_items=_max_items,
        )

    # Deduplicate, preserve order
    seen = set(s["queries"])
    added = 0
    for query in new_q:
        if query not in seen:
            s["queries"].append(query)
            seen.add(query)
            added += 1

    # Inform user about generated queries and active strategies
    if added > 0:
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
        strategy_hint = f" ({', '.join(strategies_active)})" if strategies_active else ""
        emit_progress(s, f"{added} neue Suchanfragen generiert{strategy_hint}")

    log.info(
        "TRACE plan: round=%d new_queries=%s total=%d",
        s["round"], json.dumps(new_q, ensure_ascii=False), len(s["queries"]),
    )

    # If no new queries: answer directly (prevents infinite loop)
    if added == 0:
        emit_progress(s, "Keine neuen Suchanfragen moeglich \u2014 beende Recherche")
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
        "done_no_new_queries": added == 0,
        "quality_site_queries": [q for q in new_q if (q or "").lower().startswith("site:")],
        "total_queries": len(s["queries"]),
        "required_aspects": s.get("required_aspects", []),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        **_plan_fallback,
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
        strategies: Runtime strategies for claim extraction,
            consolidation, and pruning.
        settings: Agent behavior settings controlling batch width,
            timeouts, and logging/test instrumentation.

    Returns:
        The mutated state dict after search, summarization, and claim
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
    # Dynamic batch size: broader first round for better coverage
    _batch = settings.first_round_queries if s["round"] == 0 else 3
    offset = s["search_offset"]
    new_q = s["queries"][offset:offset + _batch]
    s["search_offset"] = offset + len(new_q)
    emit_progress(
        s,
        f"Durchsuche {len(new_q)} Quellen (Runde {s['round'] + 1}/{settings.max_rounds})...",
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
            "context_blocks": len(s["context"]),
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
            "context_blocks": len(s["context"]),
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

    # Default: exclude low-quality domains.
    # When the query contains an explicit `site:...`, we use the
    # Perplexity domain filter as an allow-list for that domain.
    _base_domain_filter = LOW_QUALITY_DOMAINS
    _collect_query_details = settings.testing_mode or log.isEnabledFor(logging.DEBUG)

    def _consume_nonfatal_notice(obj: object) -> str | None:
        consumer = getattr(obj, "consume_nonfatal_notice", None)
        if callable(consumer):
            return consumer()
        return None

    def _domain_filter_for_query(q: str) -> list[str] | None:
        ql = (q or "").lower()
        m = re.search(r"(?:^|\s)site:([^\s]+)", ql)
        if not m:
            return _base_domain_filter
        dom = m.group(1).strip()
        dom = dom.replace("https://", "").replace("http://", "")
        dom = dom.split("/")[0].strip()
        dom = dom.strip(" ,.;:()[]{}<>\\\"'")
        if not dom:
            return _base_domain_filter
        return [dom]

    # Request related questions in first round
    if s["round"] == 0 and search_capabilities.supports("return_related"):
        search_kwargs["return_related"] = True

    _query_domain_filters = [_domain_filter_for_query(q) for q in new_q]

    # Parallel search
    def _search_one(item: tuple[str, list[str] | None]) -> dict[str, Any]:
        q, domain_filter = item
        effective_domain_filter = (
            domain_filter if search_capabilities.supports("domain_filter") else None
        )
        result = providers.search.search(
            q,
            domain_filter=effective_domain_filter,
            **search_kwargs,
        )
        warning = _consume_nonfatal_notice(providers.search)
        if warning:
            result = dict(result)
            result["_nonfatal_notice"] = warning
        return result

    _n_workers = min(len(new_q), settings.first_round_queries)

    with ThreadPoolExecutor(max_workers=_n_workers) as ex:
        results = list(ex.map(_search_one, zip(new_q, _query_domain_filters)))

    _query_details: list[dict[str, Any]] = []
    if _collect_query_details:
        for _qi, q in enumerate(new_q):
            r = results[_qi]
            _detail: dict[str, Any] = {
                "query": q,
                "domain_filter": _query_domain_filters[_qi] or [],
                "provider_notice": r.get("_nonfatal_notice") or "",
                "answer_length": len(r.get("answer", "") or ""),
                "citation_count": len(r.get("citations", []) or []),
                "related_question_count": len(r.get("related_questions", []) or []),
                "prompt_tokens": r.get("_prompt_tokens", 0),
                "completion_tokens": r.get("_completion_tokens", 0),
            }
            if r.get("citations"):
                _detail["urls"] = [normalize_url(u) for u in r.get("citations", [])[:5]]
            _query_details.append(_detail)

    _search_fallbacks = sum(1 for r in results if r.get("_nonfatal_notice"))
    _empty_without_notice = sum(
        1 for r in results
        if not r.get("answer") and not r.get("_nonfatal_notice")
    )
    if _search_fallbacks:
        emit_progress(
            s,
            f"{_search_fallbacks} von {len(new_q)} Suchanfragen fehlgeschlagen "
            f"(leere Ergebnisse zurueckgegeben)",
        )
    if _empty_without_notice:
        emit_progress(
            s,
            f"{_empty_without_notice} von {len(new_q)} Suchanfragen lieferten "
            f"keine Ergebnisse",
        )

    # Aggregate token usage from Sonar searches
    _search_prompt_tokens = 0
    _search_completion_tokens = 0
    for r in results:
        _search_prompt_tokens += r.get("_prompt_tokens", 0)
        _search_completion_tokens += r.get("_completion_tokens", 0)

    # Phase 1: Parallel summarize -- all independent Claude calls concurrently
    tuning = settings.report_tuning
    result_citation_cap = max(1, int(tuning.result_citation_cap or 8))
    summarize_options = SummarizeOptions(
        prompt_template=(
            SUMMARIZE_PROMPT_DEEP if settings.report_profile is ReportProfile.DEEP else None
        ),
        input_char_limit=tuning.summarize_input_char_limit,
        fallback_char_limit=tuning.summarize_fallback_char_limit,
        max_output_tokens=tuning.summarize_max_output_tokens,
    )
    _summarize_inputs: list[tuple[int, str]] = []
    for _qi, r in enumerate(results):
        if r["answer"]:
            _summarize_inputs.append((_qi, r["answer"]))

    _summarize_results: dict[int, tuple[str, int, int]] = {}
    _summarize_fallbacks = 0
    _summarize_warnings: dict[int, str] = {}
    if _summarize_inputs:
        def _do_summarize(item: tuple[int, str]) -> tuple[int, tuple[str, int, int], str | None]:
            idx, text = item
            result_tuple = providers.llm.summarize_parallel(
                text,
                deadline=s["deadline"],
                options=summarize_options,
            )
            return (idx, result_tuple, _consume_nonfatal_notice(providers.llm))

        with ThreadPoolExecutor(max_workers=min(len(_summarize_inputs), _n_workers)) as ex:
            for idx, result_tuple, warning in ex.map(_do_summarize, _summarize_inputs):
                _summarize_results[idx] = result_tuple
                if warning:
                    _summarize_fallbacks += 1
                    _summarize_warnings[idx] = warning

    if _summarize_fallbacks:
        emit_progress(
            s,
            f"{_summarize_fallbacks} von {len(_summarize_inputs)} Zusammenfassungen "
            f"fehlgeschlagen (Rohtext verwendet)",
        )

    # Aggregate summarize tokens
    _sum_prompt_tokens = 0
    _sum_completion_tokens = 0
    for facts, pt, ct in _summarize_results.values():
        _sum_prompt_tokens += pt
        _sum_completion_tokens += ct

    # Phase 1b: Parallel claim extraction for later consolidation
    _claim_inputs: list[tuple[int, str, list[str]]] = []
    for _qi, r in enumerate(results):
        if r["answer"]:
            _claim_inputs.append((_qi, r["answer"], r.get("citations", [])))

    _claim_results: dict[int, tuple[list[dict[str, Any]], int, int]] = {}
    _claim_fallbacks = 0
    _claim_warnings: dict[int, str] = {}
    if _claim_inputs:
        def _do_claim_extract(
            item: tuple[int, str, list[str]],
        ) -> tuple[int, tuple[list[dict[str, Any]], int, int], str | None]:
            idx, text, citations = item
            result_tuple = strategies.claim_extraction.extract(
                text,
                citations,
                s.get("question", ""),
                deadline=s["deadline"],
                text_char_limit=tuning.claim_input_char_limit,
                citation_cap=tuning.claim_citation_cap,
                max_claims=tuning.claim_max_items,
                source_url_limit=tuning.claim_source_url_cap,
            )
            return (idx, result_tuple, _consume_nonfatal_notice(strategies.claim_extraction))

        with ThreadPoolExecutor(max_workers=min(len(_claim_inputs), _n_workers)) as ex:
            for idx, result_tuple, warning in ex.map(_do_claim_extract, _claim_inputs):
                _claim_results[idx] = result_tuple
                if warning:
                    _claim_fallbacks += 1
                    _claim_warnings[idx] = warning

    if _claim_fallbacks:
        emit_progress(
            s,
            f"{_claim_fallbacks} von {len(_claim_inputs)} Claim-Extraktionen fehlgeschlagen "
            f"(uebersprungen)",
        )

    _claim_prompt_tokens = 0
    _claim_completion_tokens = 0
    for _, pt, ct in _claim_results.values():
        _claim_prompt_tokens += pt
        _claim_completion_tokens += ct

    # Phase 2: Sequential context assembly (state access, not parallelisable)
    focus_stems = strategies.claim_consolidation.focus_stems_from_question(
        s.get("question", ""))
    sources_found = 0
    _sources_summary: list[dict[str, Any]] = []
    for _qi, r in enumerate(results):
        if not r["answer"]:
            continue

        # Get summarize result from phase 1
        if _qi in _summarize_results:
            facts = _summarize_results[_qi][0]
        else:
            facts = r["answer"][:tuning.summarize_fallback_char_limit]

        block = facts
        if r["citations"]:
            # Normalise URLs and collect globally (deduplicated)
            for url in r["citations"][:result_citation_cap]:
                normalized = normalize_url(url)
                if normalized not in s["all_citations"]:
                    s["all_citations"].append(normalized)
            sources = "\n".join(
                f"- {normalize_url(url)}" for url in r["citations"][:result_citation_cap]
            )
            block += f"\n\nQuellen:\n{sources}"
        s["context"].append(block)
        sources_found += 1

        # Fill claim ledger with structured assertions
        extracted_claims = _claim_results.get(_qi, ([], 0, 0))[0]
        kept_claims = 0
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
            entry = {
                "claim_text": claim_text,
                "claim_type": str(claim.get("claim_type", "fact")),
                "polarity": str(claim.get("polarity", "affirmed")),
                "needs_primary": bool(claim.get("needs_primary", False)),
                "source_urls": [
                    normalize_url(u)
                    for u in claim.get("source_urls", [])
                    if u
                ][: tuning.claim_source_url_cap],
                "published_date": str(claim.get("published_date", "unknown")),
                "signature": signature,
                "round": s["round"],
                "query": new_q[_qi] if _qi < len(new_q) else "",
            }
            s["claim_ledger"].append(entry)
            kept_claims += 1

        # Cap ledger size to keep prompt and RAM stable
        if len(s["claim_ledger"]) > tuning.claim_ledger_cap:
            s["claim_ledger"] = s["claim_ledger"][-tuning.claim_ledger_cap:]

        if _collect_query_details:
            _entry = dict(_query_details[_qi]) if _qi < len(_query_details) else {
                "query": new_q[_qi] if _qi < len(new_q) else "?",
            }
            _entry["summary"] = facts
            _entry["claims_extracted"] = len(extracted_claims)
            _entry["claims_kept"] = kept_claims
            if _qi in _summarize_warnings:
                _entry["summarize_notice"] = _summarize_warnings[_qi]
            if _qi in _claim_warnings:
                _entry["claim_notice"] = _claim_warnings[_qi]
            if extracted_claims:
                _entry["claims_sample"] = [
                    str(c.get("claim_text", "")).strip()
                    for c in extracted_claims[:3]
                    if str(c.get("claim_text", "")).strip()
                ]
            _sources_summary.append(_entry)

        # Collect related questions
        for rq in r.get("related_questions", []):
            if rq not in s["related_questions"]:
                s["related_questions"].append(rq)

    emit_progress(
        s, f"{sources_found} Quellen verarbeitet, {len(s['all_citations'])} Referenzen gesammelt")

    if s["round"] == 0 and s["related_questions"]:
        emit_progress(
            s,
            f"{len(s['related_questions'])} verwandte Fragen aus Suchergebnissen erkannt",
        )

    # Update source quality and aspect coverage
    tier_counts, quality_score = strategies.source_tiering.quality_from_urls(s["all_citations"])
    s["source_tier_counts"] = tier_counts
    s["source_quality_score"] = quality_score
    consolidated_claims_all = strategies.claim_consolidation.consolidate(
        s.get("claim_ledger", []))
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

    log.info(
        "TRACE search: round=%d queries=%s sources_found=%d total_citations=%d context_blocks=%d "
        "claims=%d claim_quality=%.2f",
        s["round"], json.dumps(new_q, ensure_ascii=False),
        sources_found, len(s["all_citations"]), len(s["context"]),
        len(consolidated_claims), claim_quality,
    )

    # Relevance-based context pruning instead of FIFO
    s["context"] = strategies.context_pruning.prune(
        s["context"],
        question=s["question"],
        sub_questions=s.get("sub_questions", []),
        max_blocks=settings.max_context,
        n_new=sources_found,
        required_aspects=(
            s.get("required_aspects", [])
            if settings.report_profile is ReportProfile.DEEP
            else None
        ),
    )
    uncovered, coverage = strategies.risk_scoring.estimate_aspect_coverage(
        s.get("required_aspects", []),
        s["context"],
    )
    s["uncovered_aspects"] = uncovered
    s["aspect_coverage"] = coverage
    s["round"] += 1

    emit_progress(
        s,
        f"Quellenqualitaet {quality_score:.2f}, Claim-Qualitaet {claim_quality:.2f}, "
        f"Aspektabdeckung {int(coverage * 100)}%",
    )

    # Aggregate token usage from Sonar + Summarize + Claim extraction
    s["total_prompt_tokens"] += _search_prompt_tokens + \
        _sum_prompt_tokens + _claim_prompt_tokens
    s["total_completion_tokens"] += (
        _search_completion_tokens + _sum_completion_tokens + _claim_completion_tokens
    )

    append_iteration_log(s, {
        "node": "search",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"] - 1,
        "worker_count": _n_workers,
        "queries_executed": len(new_q),
        "queries": new_q,
        "search_parameters": {
            "search_context_size": search_kwargs.get("search_context_size") or "",
            "recency_filter": search_kwargs.get("recency_filter") or "",
            "language_filter": search_kwargs.get("language_filter", []),
            "search_mode": search_kwargs.get("search_mode") or "",
            "return_related": bool(search_kwargs.get("return_related")),
            "supported_parameters": sorted(search_capabilities.supported_parameters),
        },
        "search_fallbacks": _search_fallbacks,
        "summarize_fallbacks": _summarize_fallbacks,
        "claim_fallbacks": _claim_fallbacks,
        "sources_found": sources_found,
        "total_citations": len(s["all_citations"]),
        "context_blocks": len(s["context"]),
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_ledger_size": len(s.get("claim_ledger", [])),
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
            metrics, stop decisions, and follow-up gap information.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for source quality, claim
            consolidation, risk coverage, and stop criteria.
        settings: Agent behavior settings controlling escalation and
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
        f"Bewerte Informationsqualitaet (nach Runde {s['round']}/{settings.max_rounds})...",
    )
    _t0 = time.monotonic()
    _stagnation_detected = False
    _evaluate_fallback: dict[str, Any] = {}
    _confidence_parsed = True
    if s.get("done"):
        append_iteration_log(s, {
            "node": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "confidence": s.get("final_confidence", 0),
            "skipped": "already_done",
        }, testing_mode=settings.testing_mode)
        return s
    try:
        _check_deadline(s["deadline"])
    except AgentTimeout:
        s["done"] = True
        append_iteration_log(s, {
            "node": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "confidence": s.get("final_confidence", 0),
            "skipped": "deadline_exceeded",
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

    evaluate_model = (
        providers.llm.models.reasoning_model
        if (s.get("high_risk", False) and settings.high_risk_evaluate_escalate)
        else providers.llm.models.effective_evaluate_model
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
            max_items=(24 if settings.report_profile is ReportProfile.DEEP else 14),
        )
        + "\n\n"
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
            + f"Frage: {s['question']}\n\n"
            + f"Recherche-Ergebnisse (nummeriert):\n"
            + "\n".join(f"[Block {i + 1}]:\n{block}" for i, block in enumerate(s["context"]))
            + negative_evidence_hint
            + EVALUATE_FORMAT_SUFFIX
        )
        a = providers.llm.complete(
            eval_prompt,
            deadline=s["deadline"],
            model=evaluate_model,
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
            emit_progress(
                s,
                "Bewertung unvollstaendig (CONFIDENCE-Feld fehlt) — nutze Default 5",
            )

        m_gaps = re.search(r"GAPS:\s*(.+?)(?:\n|$)", a)
        gaps = m_gaps.group(1).strip() if m_gaps else ""
        s["gaps"] = "" if is_none_value(gaps) else gaps

        # --- Apply heuristics ---
        conf = strategies.stop_criteria.check_contradictions(s, a, conf)
        strategies.stop_criteria.filter_irrelevant_blocks(s, a)
        conf = strategies.stop_criteria.extract_competing_events(s, a, conf)
        conf = strategies.stop_criteria.extract_evidence_scores(s, a, conf)

    except AgentRateLimited:
        raise
    except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError) as exc:
        # No fail-open: on evaluate error stay conservative.
        _exc_label = type(exc).__name__
        log.warning(
            "Evaluate-Fallback aktiviert (%s, round=%d, model=%s): %s",
            _exc_label, s["round"], evaluate_model, exc,
        )
        emit_progress(
            s,
            f"Qualitaetsbewertung fehlgeschlagen ({_exc_label}) — konservative Confidence-Begrenzung",
        )
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

    # --- Post-LLM stop heuristics ---
    _prev_conf = s.get("final_confidence", 0)
    _n_citations = len(s.get("all_citations", []))

    _falsification_just_triggered = strategies.stop_criteria.check_falsification(
        s, conf, _prev_conf)
    conf, _stagnation_detected = strategies.stop_criteria.check_stagnation(
        s, conf, _prev_conf, _n_citations, _falsification_just_triggered)
    _utility, _utility_stop = strategies.stop_criteria.compute_utility(
        s, conf, _prev_conf, _n_citations)

    s["final_confidence"] = conf

    _plateau_stop = strategies.stop_criteria.check_plateau(
        s, conf, _prev_conf, _stagnation_detected)

    # --- Final stop logic ---
    log.info(
        "TRACE evaluate: round=%d confidence=%d/%d gaps='%s' context_blocks=%d "
        "quality=%.2f claim_quality=%.2f claims(v/c/u)=%d/%d/%d model=%s done=%s",
        s["round"], conf, settings.confidence_stop,
        s.get("gaps", "")[:100], len(s["context"]),
        quality_score, claim_quality,
        verified_claims, contested_claims, unverified_claims,
        evaluate_model,
        conf >= settings.confidence_stop or s["round"] >= settings.max_rounds or s["done"],
    )

    if s["done"] or conf >= settings.confidence_stop or s["round"] >= settings.max_rounds:
        s["done"] = True
        emit_progress(
            s, f"Recherche abgeschlossen (Confidence: {conf}/10, Runden: {s['round']})")
    else:
        emit_progress(s, f"Confidence {conf}/10 — weitere Recherche noetig")

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
            f"min_rounds={_min_rounds} noch nicht erreicht (aktuell {s['round']}); "
            f"setze Recherche fort",
        )
        s["done"] = False

    _eval_log_entry: dict[str, Any] = {
        "node": "evaluate",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"],
        "confidence": conf,
        "confidence_parsed": _confidence_parsed,
        "confidence_stop_target": settings.confidence_stop,
        "gaps": s.get("gaps", ""),
        "competing_events": s.get("competing_events", ""),
        "stagnation_detected": _stagnation_detected,
        "falsification_triggered": s.get("falsification_triggered", False),
        "evidence_consistency": s.get("evidence_consistency", 0),
        "evidence_sufficiency": s.get("evidence_sufficiency", 0),
        "evidence_consistency_parsed": s.get("_evidence_consistency_parsed", True),
        "evidence_sufficiency_parsed": s.get("_evidence_sufficiency_parsed", True),
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
        "aspect_coverage": s.get("aspect_coverage", 0.0),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "model": evaluate_model,
        "utility_score": _utility,
        "utility_stop": _utility_stop,
        "plateau_stop": _plateau_stop,
        "context_blocks": len(s["context"]),
        "stop_by_confidence": conf >= settings.confidence_stop,
        "stop_by_round_limit": s["round"] >= settings.max_rounds,
        "stop_by_existing_done": bool(s.get("done")),
        "done": s["done"],
        "guardrail_reasons": list(_guardrail_result.reasons or []),
        **_evaluate_fallback,
    }
    if settings.testing_mode and _eval_raw:
        _eval_log_entry["reasoning"] = _eval_raw
    append_iteration_log(s, _eval_log_entry, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 5. answer
# ======================================================================= #


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
    round_label = "Runde" if n_rounds == 1 else "Runden"
    emit_progress(s, f"Formuliere Antwort (nach {n_rounds} {round_label})...")
    _t0 = time.monotonic()
    tuning = settings.report_tuning
    ctx = "\n\n---\n\n".join(s["context"]) if s["context"] else ""

    # Determine answer language
    lang = s.get("language", "de")
    answer_lang = LANG_NAMES.get(lang, lang)

    # Build numbered source list (fast-path: claim-bound selection)
    consolidated_claims = s.get("consolidated_claims", [])
    citations = s.get("all_citations", [])
    selected_prompt_citations = strategies.claim_consolidation.select_answer_citations(
        consolidated_claims,
        citations,
        max_items=settings.answer_prompt_citations_max,
        source_tiering=strategies.source_tiering,
    )
    prompt_citations_used_fallback = False
    if not selected_prompt_citations and citations:
        fallback_n = (
            len(citations)
            if settings.report_profile is ReportProfile.DEEP
            else min(25, settings.answer_prompt_citations_max)
        )
        selected_prompt_citations = citations[:fallback_n]
        prompt_citations_used_fallback = True
    body_citation_cap = max(
        1,
        int(
            tuning.answer_body_citation_cap
            or len(selected_prompt_citations)
            or 1
        ),
    )
    body_prompt_citations = selected_prompt_citations[:body_citation_cap]
    prompt_citations_trimmed_for_body = max(
        0,
        len(selected_prompt_citations) - len(body_prompt_citations),
    )
    body_prompt_citations_before_budget = len(body_prompt_citations)
    body_prompt_citations = _limit_prompt_citations_by_char_budget(
        body_prompt_citations,
        char_budget=tuning.answer_citation_block_char_budget,
    )
    prompt_citations_trimmed_by_budget = max(
        0,
        body_prompt_citations_before_budget - len(body_prompt_citations),
    )

    # Assemble shared prompt state for the section-wise composer.
    state_data: dict[str, Any] = {
        "today_str": today(),
        "answer_lang": answer_lang,
        "context": ctx,
        "prompt_citations": body_prompt_citations,
        "all_citations": citations,
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "consolidated_claims": consolidated_claims,
        "claims_prompt_view_fn": strategies.claim_consolidation.claims_prompt_view,
        "claim_prompt_max_items": tuning.answer_claim_prompt_items,
        "required_aspects": s.get("required_aspects"),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "competing_events": s.get("competing_events", ""),
        "history": s.get("history", ""),
        "_prev_question": s.get("_prev_question", ""),
        "_prev_answer": s.get("_prev_answer", ""),
        "report_profile": str(settings.report_profile),
    }
    fallback_model = (
        providers.llm.models.effective_evaluate_model
        if (
            providers.llm.models.effective_evaluate_model
            and providers.llm.models.effective_evaluate_model != providers.llm.models.reasoning_model
        )
        else None
    )
    fallback_attempted = False
    fallback_succeeded = False
    composition_result = _AnswerCompositionResult(answer="", finish_reason="", section_logs=[])

    try:
        composition_result = _compose_answer_sections(
            s,
            providers=providers,
            settings=settings,
            state_data=state_data,
        )
        s["answer"] = composition_result.answer
    except AgentTimeout:
        # On timeout: use context directly as answer
        if ctx:
            s["answer"] = (
                "Die Recherche konnte aus Zeitgruenden nicht vollstaendig abgeschlossen werden. "
                f"Hier die bisherigen Ergebnisse:\n\n{ctx}"
            )
        else:
            s["answer"] = "Die Anfrage konnte aufgrund eines Zeitlimits nicht bearbeitet werden. Bitte erneut versuchen."
    except (OpenAIError, AnthropicAPIError, BedrockAPIError) as e:
        log.error("Finale Antwort fehlgeschlagen: %s", e)
        if fallback_model:
            try:
                fallback_attempted = True
                emit_progress(
                    s, f"Finale Antwort fehlgeschlagen — Fallback-Modell {fallback_model}")
                composition_result = _compose_answer_sections(
                    s,
                    providers=providers,
                    settings=settings,
                    state_data=state_data,
                    model=fallback_model,
                )
                s["answer"] = composition_result.answer
                fallback_succeeded = True
            except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError) as e2:
                log.error("Finale Antwort-Fallback fehlgeschlagen (%s): %s", fallback_model, e2)
                if ctx:
                    s["answer"] = (
                        "Bei der Formulierung der Antwort ist ein Fehler aufgetreten. "
                        f"Hier die Recherche-Ergebnisse:\n\n{ctx}"
                    )
                else:
                    s["answer"] = "Bei der Verarbeitung ist ein Fehler aufgetreten. Bitte erneut versuchen."
        elif ctx:
            s["answer"] = (
                "Bei der Formulierung der Antwort ist ein Fehler aufgetreten. "
                f"Hier die Recherche-Ergebnisse:\n\n{ctx}"
            )
        else:
            s["answer"] = "Bei der Verarbeitung ist ein Fehler aufgetreten. Bitte erneut versuchen."

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
        emit_progress(s, "Finale Antwort als unvollstaendig erkannt")
        log.warning(
            "TRACE answer: incomplete answer detected (finish_reason=%s, reasons=%s)",
            finish_reason or "unknown",
            incomplete_reasons,
        )

    # Quick citation guardrail: remove non-allowed links.
    removed_link_count = 0
    allowed_citation_urls = {
        normalize_url(url)
        for url in body_prompt_citations
        if normalize_url(url)
    }
    appended_sources_footer = False
    allowed_link_count = 0
    if allowed_citation_urls and s.get("answer"):
        s["answer"], removed_link_count = sanitize_answer_links(
            s["answer"], allowed_citation_urls)
        if removed_link_count:
            log.info("TRACE answer: removed %d non-allowed links", removed_link_count)
            emit_progress(s, f"{removed_link_count} nicht-zugelassene Links entfernt")
        allowed_link_count = count_allowed_links(s["answer"], allowed_citation_urls)

    if incomplete_reasons:
        s["answer"] = _repair_answer_markdown_tail(s.get("answer", ""))
    appendix_sections, reference_link_count, additional_link_count = _build_answer_appendix_sections(
        s.get("answer", ""),
        prompt_citations=body_prompt_citations,
        all_citations=citations,
        strategies=strategies,
        incomplete_reasons=incomplete_reasons,
        finish_reason=finish_reason,
    )
    if appendix_sections:
        s["answer"] = s["answer"].rstrip() + "\n\n---\n\n" + "\n\n---\n\n".join(appendix_sections)
        appended_sources_footer = True

    # Append stats footer
    elapsed = time.monotonic() - s.get("start_time", time.monotonic())
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    time_str = f"{minutes}:{seconds:02d} min" if minutes > 0 else f"{seconds}s"
    conf = s.get("final_confidence", 0)
    n_sources = len(s.get("all_citations", []))
    n_rounds = s.get("round", 0)
    n_queries = len(s.get("queries", []))

    stats_parts = []
    if n_sources:
        if 0 < allowed_link_count < n_sources:
            stats_parts.append(f"{n_sources} Quellen ({allowed_link_count} verlinkt)")
        else:
            stats_parts.append(f"{n_sources} Quellen")
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
        "TRACE answer: length=%d citations=%d prompt_citations=%d body_prompt_citations=%d linked=%d refs=%d extra_links=%d sections=%d rounds=%d elapsed=%.1fs confidence=%d finish_reason=%s incomplete=%s",
        len(s["answer"]), n_sources, len(selected_prompt_citations), len(body_prompt_citations),
        allowed_link_count, reference_link_count, additional_link_count, len(section_logs),
        n_rounds, elapsed, conf, finish_reason or "", bool(incomplete_reasons),
    )
    log.debug("ANSWER text:\n%s", s["answer"])

    append_iteration_log(s, {
        "node": "answer",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "answer_length": len(s["answer"]),
        "citation_count": n_sources,
        "prompt_citation_count": len(selected_prompt_citations),
        "body_prompt_citation_count": len(body_prompt_citations),
        "prompt_citations": selected_prompt_citations[:10],
        "body_prompt_citations": body_prompt_citations[:10],
        "prompt_citations_used_fallback": prompt_citations_used_fallback,
        "prompt_citations_trimmed_for_body": prompt_citations_trimmed_for_body,
        "prompt_citations_trimmed_by_budget": prompt_citations_trimmed_by_budget,
        "removed_non_allowed_links": removed_link_count,
        "allowed_link_count": allowed_link_count,
        "reference_link_count": reference_link_count,
        "additional_link_count": additional_link_count,
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
