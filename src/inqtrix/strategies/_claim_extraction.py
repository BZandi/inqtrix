"""Claim extraction strategy — extract structured claims from search text."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAIError

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AnthropicAPIError,
    AzureFoundryBingAPIError,
    AzureFoundryWebSearchAPIError,
    AzureOpenAIAPIError,
    AzureOpenAIWebSearchAPIError,
    BedrockAPIError,
)
from inqtrix.json_helpers import parse_json_object
from inqtrix.prompts import build_claim_extraction_prompt
from inqtrix.providers.base import LLMProvider, _NonFatalNoticeMixin, _bounded_timeout, _check_deadline
from inqtrix.urls import normalize_url

log = logging.getLogger("inqtrix")

_CLAIM_TYPES: set[str] = {"fact", "actor_claim", "forecast"}
_CLAIM_POLARITY: set[str] = {"affirmed", "negated"}

_CLAIM_PRIMARY_HINT_RE = re.compile(
    r"(\b\d{1,3}(?:[.,]\d+)?\s*(?:%|prozent|mrd|mio|million(?:en)?|milliard(?:en)?|euro)\b"
    r"|\b(gesetz|verordnung|richtlinie)\b|\b\u00a7\s*\d+\b|\bart\.?\s*\d+\b)",
    re.IGNORECASE,
)

_CLAIM_ACTOR_VERB_RE = re.compile(
    r"\b(sagte|sagt|warnte|warnt|forderte|fordert|lehnte|lehnt|schloss|schliesst|"
    r"erkl[a\u00e4]rte|erkl[a\u00e4]rt|kuendigte|k\u00fcndigte|kritisiert|kritisierte|nannte|"
    r"bezeichnete|wies|zurueck|zur\u00fcck)\b",
    re.IGNORECASE,
)


class ClaimExtractionStrategy(ABC):
    """Contract for extracting structured claims from raw search text.

    A claim extractor turns the textual output of one search hit into
    a bounded list of normalised claim dicts that the consolidation
    strategy can later deduplicate and verify. Implementations must
    return safe-to-consume defaults on failure (empty claims list,
    zero token counts) rather than raising — only hard rate-limit /
    timeout conditions should propagate.
    """

    @abstractmethod
    def extract(
        self,
        text: str,
        citations: list[str],
        question: str,
        *,
        deadline: float | None = None,
        text_char_limit: int = 7000,
        citation_cap: int = 8,
        max_claims: int = 8,
        source_url_limit: int = 4,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Extract normalised claims from one search result.

        Args:
            text: Raw search-result text to analyse. Empty / whitespace
                input must short-circuit to an empty result; large
                inputs must be trimmed by the implementation
                (``text_char_limit``).
            citations: Citation URLs that are allowed to appear in the
                extracted claim payload's ``source_urls`` field. Acts
                as an allow-list — URLs absent here must be discarded
                silently to prevent the LLM from inventing sources.
            question: Original user question. Forwarded into the
                extraction prompt so the model can focus on
                question-relevant claims rather than every fact in
                the source text.
            deadline: Optional absolute monotonic deadline for the
                whole agent run. Implementations should clamp any
                per-call timeout to the remaining budget and raise
                :class:`AgentTimeout` when the budget is exhausted.
            text_char_limit: Maximum number of characters from
                ``text`` forwarded to the LLM. Defaults to ``7000``
                (COMPACT profile); raise via the report-profile
                tuning bundle. Capped to ``>= 1000`` to prevent
                pathological tiny inputs.
            citation_cap: Maximum number of entries from ``citations``
                included in the prompt's source list. Bounds prompt
                size when a single search returns very many URLs.
            max_claims: Maximum number of claims returned. The
                implementation may over-extract internally and then
                truncate to this cap. Capped to ``>= 1``.
            source_url_limit: Maximum number of source URLs attached
                to a single claim. Bounds the per-claim payload size
                when many citations support the same statement.

        Returns:
            Tuple ``(claims, prompt_tokens, completion_tokens)``.
            ``claims`` is a list of dicts each with keys
            ``claim_text`` (string), ``claim_type`` (one of
            ``fact``/``actor_claim``/``forecast``), ``polarity`` (one
            of ``affirmed``/``negated``), ``needs_primary`` (bool),
            ``source_urls`` (list of allow-listed URLs),
            ``published_date`` (string or ``"unknown"``). Token counts
            come from the underlying LLM call; ``0`` when no call ran.

        Raises:
            AgentRateLimited: When the LLM provider escalates a fatal
                rate limit. Other failure modes (timeouts, parse
                errors, provider API errors) must NOT raise — return
                an empty claim list and surface the cause via the
                non-fatal notice mixin instead.
        """


class LLMClaimExtractor(_NonFatalNoticeMixin, ClaimExtractionStrategy):
    """LLM-backed claim extractor used during the search node.

    Converts free-form search-result text into a bounded list of
    structured claims by prompting the configured ``summarize_model``.
    Validates each claim against the schema (allowed types, polarity),
    downgrades obvious actor attributions from ``fact`` to
    ``actor_claim``, infers ``needs_primary`` from regex hints when
    the model omitted it, and restricts ``source_urls`` to the
    known-citations allow-list to prevent hallucinated sources.

    Failure handling is non-fatal: parse errors, timeouts and
    provider API errors all yield an empty claim list plus a
    non-fatal notice that the search node can surface to the user.
    Only :class:`AgentRateLimited` propagates so the agent can abort
    consistently on hard rate-limit conditions.
    """

    def __init__(
        self,
        llm: LLMProvider | None,
        summarize_model: str,
        summarize_timeout: int = 60,
    ) -> None:
        """Bind the underlying LLM and per-call defaults to the extractor.

        Args:
            llm: LLM provider that will perform the extraction calls.
                When ``None``, every :meth:`extract` call short-
                circuits to an empty result and stores a non-fatal
                notice — used by tests and by the env-driven path
                when no LLM provider could be auto-created.
            summarize_model: Model identifier passed to the provider's
                ``complete_with_metadata`` call. In env-driven mode
                this is the resolved value of
                ``ModelSettings.effective_summarize_model``; in
                Baukasten mode the caller picks an explicit model
                name. Empty string is accepted but typically results
                in the provider falling back to its own default.
            summarize_timeout: Per-call timeout (seconds) before
                deadline clamping. Default ``60`` mirrors the
                ``AgentSettings.summarize_timeout`` default; tighten
                for faster failure on slow upstreams, raise for
                models with long warmup.

        Example:
            >>> from inqtrix.strategies import LLMClaimExtractor
            >>> extractor = LLMClaimExtractor(llm=None, summarize_model="")
            >>> extractor.extract("", [], "question")
            ([], 0, 0)
        """
        self._llm = llm
        self._summarize_model = summarize_model
        self._summarize_timeout = summarize_timeout

    # ------------------------------------------------------------------ #
    # extract
    # ------------------------------------------------------------------ #
    def extract(
        self,
        text: str,
        citations: list[str],
        question: str,
        *,
        deadline: float | None = None,
        text_char_limit: int = 7000,
        citation_cap: int = 8,
        max_claims: int = 8,
        source_url_limit: int = 4,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Run claim extraction for one search result.

        The method validates claim types and polarity, downgrades obvious
        speaker attributions from ``fact`` to ``actor_claim``, infers
        ``needs_primary`` when the model omitted it, and keeps only URLs from
        the provided citation allow-list.

        Returns:
            Tuple of ``(claims, prompt_tokens, completion_tokens)``.

        Raises:
            AgentRateLimited: Propagated so the agent can abort consistently on
                hard rate-limit conditions.
            AgentTimeout: Raised before the call when the absolute deadline is
                already exhausted.
        """
        self._clear_nonfatal_notice()
        if not text.strip():
            return [], 0, 0
        if self._llm is None:
            self._set_nonfatal_notice("Claim-Extraktion uebersprungen — kein LLM konfiguriert")
            return [], 0, 0
        if deadline is not None:
            _check_deadline(deadline)

        text_char_limit = max(1000, int(text_char_limit or 7000))
        citation_cap = max(1, int(citation_cap or 8))
        max_claims = max(1, int(max_claims or 8))
        source_url_limit = max(1, int(source_url_limit or 4))

        normalized_citations = [normalize_url(u) for u in (citations or []) if u]
        known_urls = set(normalized_citations)

        prompt = (
            f"{build_claim_extraction_prompt(max_claims=max_claims)}\n"
            f"Frage:\n{(question or '').strip()}\n\n"
            f"Quellenliste:\n{json.dumps(normalized_citations[:citation_cap], ensure_ascii=False)}\n\n"
            f"Text:\n{text[:text_char_limit]}"
        )

        try:
            without_thinking = getattr(self._llm, "without_thinking", None)
            if callable(without_thinking):
                with without_thinking():
                    response = self._llm.complete_with_metadata(
                        prompt,
                        model=self._summarize_model,
                        timeout=_bounded_timeout(self._summarize_timeout, deadline),
                        deadline=deadline,
                    )
            else:
                response = self._llm.complete_with_metadata(
                    prompt,
                    model=self._summarize_model,
                    timeout=_bounded_timeout(self._summarize_timeout, deadline),
                    deadline=deadline,
                )
            raw = response.content or ""
            parsed = parse_json_object(raw, fallback={"claims": []})
            raw_claims = parsed.get("claims", [])
            claims: list[dict[str, Any]] = []

            if isinstance(raw_claims, list):
                for item in raw_claims[: max(max_claims * 2, max_claims)]:
                    if not isinstance(item, dict):
                        continue
                    claim_text = str(item.get("claim_text", "")).strip()
                    if len(claim_text) < 12:
                        continue

                    claim_type = str(item.get("claim_type", "fact")).strip().lower()
                    if claim_type not in _CLAIM_TYPES:
                        claim_type = "fact"
                    if claim_type == "fact" and _CLAIM_ACTOR_VERB_RE.search(claim_text):
                        claim_type = "actor_claim"

                    polarity = str(item.get("polarity", "affirmed")).strip().lower()
                    if polarity not in _CLAIM_POLARITY:
                        polarity = "affirmed"

                    raw_needs_primary = item.get("needs_primary", None)
                    if claim_type != "fact":
                        needs_primary = False
                    elif isinstance(raw_needs_primary, bool):
                        needs_primary = raw_needs_primary
                    else:
                        needs_primary = bool(_CLAIM_PRIMARY_HINT_RE.search(claim_text))

                    source_urls: list[str] = []
                    raw_urls = item.get("source_urls", [])
                    if isinstance(raw_urls, list):
                        for u in raw_urls:
                            n = normalize_url(str(u))
                            if not n:
                                continue
                            if known_urls and n not in known_urls:
                                continue
                            if n not in source_urls:
                                source_urls.append(n)

                    claims.append({
                        "claim_text": claim_text,
                        "claim_type": claim_type,
                        "polarity": polarity,
                        "needs_primary": needs_primary,
                        "source_urls": source_urls[:source_url_limit],
                        "published_date": str(
                            item.get("published_date", "unknown"),
                        ).strip() or "unknown",
                    })

            return claims[:max_claims], response.prompt_tokens, response.completion_tokens

        except AgentRateLimited:
            raise
        except (
            OpenAIError,
            AgentTimeout,
            AnthropicAPIError,
            BedrockAPIError,
            AzureOpenAIAPIError,
            AzureFoundryBingAPIError,
            AzureFoundryWebSearchAPIError,
            AzureOpenAIWebSearchAPIError,
        ) as exc:
            # Phase-2-style visibility: capture the actual error so the
            # operator sees WHY claim extraction failed (model rejected
            # parameter, rate limit, timeout, ...), not just "failed".
            exc_label = type(exc).__name__
            exc_message = str(exc)[:200]
            log.warning(
                "Claim-Extraktion fehlgeschlagen (model=%s, %s): %s",
                self._summarize_model,
                exc_label,
                exc_message,
            )
            self._set_nonfatal_notice(
                f"Claim-Extraktion via {self._summarize_model} fehlgeschlagen "
                f"({exc_label}: {exc_message}); Quelle wird ohne Claims weiterverwendet."
            )
            return [], 0, 0
