"""Claim extraction strategy — extract structured claims from search text."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAIError

from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AnthropicAPIError, BedrockAPIError
from inqtrix.json_helpers import parse_json_object
from inqtrix.prompts import CLAIM_EXTRACTION_PROMPT
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
    """Extract structured claims from search text."""

    @abstractmethod
    def extract(
        self,
        text: str,
        citations: list[str],
        question: str,
        *,
        deadline: float | None = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Extract normalized claims from a search result.

        Args:
            text: Raw search-result text to analyze.
            citations: Citation URLs that are allowed to appear in the
                extracted claim payload.
            question: Original user question used to focus extraction.
            deadline: Optional absolute deadline for the extraction call.

        Returns:
            Tuple of ``(claims, prompt_tokens, completion_tokens)``.
        """


class LLMClaimExtractor(_NonFatalNoticeMixin, ClaimExtractionStrategy):
    """LLM-backed claim extractor used during the search node.

    The extractor converts free-form result text into a bounded list of
    structured claims, normalizes schema fields, restricts source URLs to the
    citations already known for that result, and degrades non-fatally when the
    summarize model fails.
    """

    def __init__(
        self,
        llm: LLMProvider | None,
        summarize_model: str,
        summarize_timeout: int = 60,
    ) -> None:
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
            return [], 0, 0
        if deadline is not None:
            _check_deadline(deadline)

        normalized_citations = [normalize_url(u) for u in (citations or []) if u]
        known_urls = set(normalized_citations)

        prompt = (
            f"{CLAIM_EXTRACTION_PROMPT}\n"
            f"Frage:\n{(question or '').strip()}\n\n"
            f"Quellenliste:\n{json.dumps(normalized_citations[:8], ensure_ascii=False)}\n\n"
            f"Text:\n{text[:7000]}"
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
                for item in raw_claims[:12]:
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
                        "source_urls": source_urls[:4],
                        "published_date": str(
                            item.get("published_date", "unknown"),
                        ).strip() or "unknown",
                    })

            return claims[:8], response.prompt_tokens, response.completion_tokens

        except AgentRateLimited:
            raise
        except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError):
            self._set_nonfatal_notice(
                f"Claim-Extraktion via {self._summarize_model} fehlgeschlagen; Quelle wird ohne Claims weiterverwendet."
            )
            return [], 0, 0
