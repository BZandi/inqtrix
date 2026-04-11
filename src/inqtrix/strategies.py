"""Pluggable strategy ABCs and their default implementations.

Each strategy encapsulates a single algorithmic concern that nodes can
depend on through the ``StrategyContext`` dataclass.  Default
implementations reproduce the exact behaviour of the original monolithic
``_original_agent.py`` so existing node call-sites remain compatible.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAIError

from inqtrix.domains import (
    GENERIC_QUERY_TERMS_DE,
    LOW_QUALITY_DOMAINS,
    MAINSTREAM_SOURCE_DOMAINS,
    PRIMARY_SOURCE_DOMAINS,
    QUALITY_MAINSTREAM_SITES_DE,
    QUALITY_PRIMARY_SITES_DE,
    SOURCE_TIER_WEIGHTS,
    STAKEHOLDER_SOURCE_DOMAINS,
)
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AnthropicAPIError
from inqtrix.prompts import CLAIM_EXTRACTION_PROMPT, default_claims_prompt_view
from inqtrix.providers.base import LLMProvider, _NonFatalNoticeMixin, _bounded_timeout, _check_deadline
from inqtrix.json_helpers import parse_json_object
from inqtrix.settings import AgentSettings
from inqtrix.text import (
    NEGATION_TOKENS,
    STOPWORDS,
    is_none_value,
    norm_match_token,
    tokenize,
)
from inqtrix.urls import domain_from_url, domain_matches, normalize_url

log = logging.getLogger("inqtrix")


# ===================================================================== #
# 1. SourceTieringStrategy
# ===================================================================== #

class SourceTieringStrategy(ABC):
    """Classify URLs into quality tiers."""

    @abstractmethod
    def tier_for_url(self, url: str) -> str:
        """Return tier name for *url*: primary/mainstream/stakeholder/unknown/low."""

    @abstractmethod
    def quality_from_urls(self, urls: list[str]) -> tuple[dict[str, int], float]:
        """Return ``(tier_counts, quality_score)`` for a list of URLs."""


class DefaultSourceTiering(SourceTieringStrategy):
    """Reproduce ``_source_tier_for_url`` / ``_source_quality_from_urls``."""

    # -------------------------------------------------------------- #
    # tier_for_url
    # -------------------------------------------------------------- #
    def tier_for_url(self, url: str) -> str:
        domain = domain_from_url(url)
        if not domain:
            return "unknown"

        low_domains = {d.lstrip("-").lower().strip() for d in LOW_QUALITY_DOMAINS}
        if domain_matches(domain, low_domains):
            return "low"
        if domain_matches(domain, PRIMARY_SOURCE_DOMAINS):
            return "primary"
        if domain_matches(domain, MAINSTREAM_SOURCE_DOMAINS):
            return "mainstream"
        if domain_matches(domain, STAKEHOLDER_SOURCE_DOMAINS):
            return "stakeholder"
        return "unknown"

    # -------------------------------------------------------------- #
    # quality_from_urls
    # -------------------------------------------------------------- #
    def quality_from_urls(self, urls: list[str]) -> tuple[dict[str, int], float]:
        counts: dict[str, int] = {
            "primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0,
        }
        if not urls:
            return counts, 0.0

        score = 0.0
        for url in urls:
            tier = self.tier_for_url(url)
            counts[tier] = counts.get(tier, 0) + 1
            score += SOURCE_TIER_WEIGHTS.get(tier, 0.35)

        return counts, round(score / len(urls), 3)


# ===================================================================== #
# 2. ClaimExtractionStrategy
# ===================================================================== #

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
        """Return ``(claims, prompt_tokens, completion_tokens)``."""


class LLMClaimExtractor(_NonFatalNoticeMixin, ClaimExtractionStrategy):
    """Reproduce ``_extract_claims_parallel`` using an LLM provider."""

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
        except (OpenAIError, AgentTimeout, AnthropicAPIError):
            self._set_nonfatal_notice(
                f"Claim-Extraktion via {self._summarize_model} fehlgeschlagen; Quelle wird ohne Claims weiterverwendet."
            )
            return [], 0, 0


# ===================================================================== #
# 3. ClaimConsolidationStrategy
# ===================================================================== #

class ClaimConsolidationStrategy(ABC):
    """Consolidate, filter, and format claim ledgers."""

    @abstractmethod
    def focus_stems_from_question(self, question: str) -> set[str]: ...

    @abstractmethod
    def claim_matches_focus_stems(self, claim_text: str, focus_stems: set[str]) -> bool: ...

    @abstractmethod
    def claim_signature(self, text: str) -> str: ...

    @abstractmethod
    def consolidate(self, claim_ledger: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    @abstractmethod
    def materialize(
        self,
        consolidated: list[dict[str, Any]],
        *,
        max_total: int = 24,
        max_unverified: int = 8,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def quality_metrics(
        self,
        consolidated: list[dict[str, Any]],
    ) -> tuple[dict[str, int], float, int, int]: ...

    @abstractmethod
    def claims_prompt_view(
        self,
        consolidated: list[dict[str, Any]],
        max_items: int = 16,
    ) -> str: ...

    @abstractmethod
    def select_answer_citations(
        self,
        consolidated: list[dict[str, Any]],
        all_citations: list[str],
        *,
        max_items: int,
    ) -> list[str]: ...


class DefaultClaimConsolidator(ClaimConsolidationStrategy):
    """Reproduce the original claim consolidation logic."""

    def __init__(self, source_tiering: SourceTieringStrategy) -> None:
        self._tiering = source_tiering

    # ------------------------------------------------------------------ #
    # focus_stems_from_question
    # ------------------------------------------------------------------ #
    def focus_stems_from_question(self, question: str) -> set[str]:
        stems: set[str] = set()
        q_norm = norm_match_token(question)

        for tok in tokenize(question):
            nt = norm_match_token(tok)
            if len(nt) < 4:
                continue
            if nt in STOPWORDS:
                continue
            if nt in GENERIC_QUERY_TERMS_DE:
                continue
            stems.add(nt[:6] if len(nt) > 6 else nt)

        if "zahn" in q_norm:
            stems.update({"zahn", "zahnar", "zahnaer", "zahnbe", "zahnlei"})
        if "privatis" in q_norm or "privat" in q_norm:
            stems.update({"privat", "privati", "privatis"})
        if "gkv" in q_norm or "krankenkass" in q_norm:
            stems.update({"gkv", "krankenk"})

        return {s for s in stems if len(s) >= 4}

    # ------------------------------------------------------------------ #
    # claim_matches_focus_stems
    # ------------------------------------------------------------------ #
    def claim_matches_focus_stems(self, claim_text: str, focus_stems: set[str]) -> bool:
        if not focus_stems:
            return True
        for tok in tokenize(claim_text):
            nt = norm_match_token(tok)
            if len(nt) < 4:
                continue
            for st in focus_stems:
                if nt.startswith(st):
                    return True
        return False

    # ------------------------------------------------------------------ #
    # claim_signature
    # ------------------------------------------------------------------ #
    def claim_signature(self, text: str) -> str:
        tokens = tokenize(text)
        cleaned = [
            t for t in tokens
            if len(t) > 2 and t not in NEGATION_TOKENS and t not in STOPWORDS
        ]
        if not cleaned:
            cleaned = [t for t in tokens if t not in NEGATION_TOKENS]
        return " ".join(cleaned[:16]).strip()

    # ------------------------------------------------------------------ #
    # consolidate
    # ------------------------------------------------------------------ #
    def consolidate(self, claim_ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim in claim_ledger:
            sig = (
                str(claim.get("signature", "")).strip()
                or self.claim_signature(str(claim.get("claim_text", "")))
            )
            if not sig:
                continue
            groups[sig].append(claim)

        consolidated: list[dict[str, Any]] = []
        for sig, entries in groups.items():
            claim_type = Counter(
                str(e.get("claim_type", "fact")) for e in entries
            ).most_common(1)[0][0]
            raw_needs_primary = any(bool(e.get("needs_primary", False)) for e in entries)
            needs_primary = raw_needs_primary and claim_type == "fact"
            support_count = sum(1 for e in entries if e.get("polarity") == "affirmed")
            contradict_count = sum(1 for e in entries if e.get("polarity") == "negated")

            source_urls: list[str] = []
            for e in entries:
                for u in e.get("source_urls", []):
                    n = normalize_url(str(u))
                    if n and n not in source_urls:
                        source_urls.append(n)

            tier_counts, _ = self._tiering.quality_from_urls(source_urls)
            has_primary = tier_counts.get("primary", 0) > 0
            has_mainstream = tier_counts.get("mainstream", 0) > 0
            has_stakeholder = tier_counts.get("stakeholder", 0) > 0
            support_like = max(support_count, contradict_count)

            if support_count > 0 and contradict_count > 0:
                status = "contested"
                reason = "widerspruechliche Evidenz"
            elif needs_primary and not has_primary:
                status = "unverified"
                reason = "primaerbeleg fehlt"
            elif support_like >= 2 and (has_primary or has_mainstream or has_stakeholder):
                status = "verified"
                reason = "mehrfach belegt"
            elif support_like >= 1 and (has_primary or has_mainstream):
                status = "verified"
                reason = "belegt durch hochwertige Quelle"
            else:
                status = "unverified"
                reason = "Evidenzlage zu schwach"

            representative = max(
                (str(e.get("claim_text", "")).strip() for e in entries),
                key=len,
                default="",
            )
            consolidated.append({
                "signature": sig,
                "claim_text": representative,
                "claim_type": claim_type,
                "needs_primary": needs_primary,
                "support_count": support_count,
                "contradict_count": contradict_count,
                "status": status,
                "status_reason": reason,
                "source_urls": source_urls[:8],
                "source_tier_counts": tier_counts,
            })

        order = {"verified": 0, "contested": 1, "unverified": 2}
        consolidated.sort(
            key=lambda c: (
                order.get(str(c.get("status", "unverified")), 3),
                -int(c.get("support_count", 0)),
                int(c.get("contradict_count", 0)),
            ),
        )
        return consolidated

    # ------------------------------------------------------------------ #
    # materialize
    # ------------------------------------------------------------------ #
    def materialize(
        self,
        consolidated: list[dict[str, Any]],
        *,
        max_total: int = 24,
        max_unverified: int = 8,
    ) -> list[dict[str, Any]]:
        if not consolidated:
            return []
        max_total = int(max_total or 0)
        max_unverified = int(max_unverified or 0)
        if max_total <= 0:
            return []

        verified: list[dict[str, Any]] = []
        contested: list[dict[str, Any]] = []
        unverified: list[dict[str, Any]] = []
        for c in consolidated:
            st = str(c.get("status", "unverified"))
            if st == "verified":
                verified.append(c)
            elif st == "contested":
                contested.append(c)
            else:
                unverified.append(c)

        if max_unverified <= 0:
            out = verified + contested
            return out[:max_total]

        unverified_high_tier: list[dict[str, Any]] = []
        unverified_unknown_only: list[dict[str, Any]] = []
        for c in unverified:
            tiers = c.get("source_tier_counts", {}) or {}
            has_any_quality = any(
                int(tiers.get(k, 0) or 0) > 0
                for k in ("primary", "mainstream", "stakeholder")
            )
            if has_any_quality:
                unverified_high_tier.append(c)
            else:
                unverified_unknown_only.append(c)

        kept_unverified = (unverified_high_tier + unverified_unknown_only)[:max_unverified]
        out = verified + contested + kept_unverified
        return out[:max_total]

    # ------------------------------------------------------------------ #
    # quality_metrics
    # ------------------------------------------------------------------ #
    def quality_metrics(
        self,
        consolidated: list[dict[str, Any]],
    ) -> tuple[dict[str, int], float, int, int]:
        counts: dict[str, int] = {"verified": 0, "contested": 0, "unverified": 0}
        needs_primary_total = 0
        needs_primary_verified = 0
        for claim in consolidated:
            status = str(claim.get("status", "unverified"))
            if status not in counts:
                status = "unverified"
            counts[status] += 1
            if bool(claim.get("needs_primary", False)):
                needs_primary_total += 1
                if status == "verified":
                    needs_primary_verified += 1

        total = len(consolidated)
        if total == 0:
            return counts, 0.0, needs_primary_total, needs_primary_verified
        score = (counts["verified"] + 0.5 * counts["contested"]) / total
        return counts, round(score, 3), needs_primary_total, needs_primary_verified

    # ------------------------------------------------------------------ #
    # claims_prompt_view
    # ------------------------------------------------------------------ #
    def claims_prompt_view(
        self,
        consolidated: list[dict[str, Any]],
        max_items: int = 16,
    ) -> str:
        return default_claims_prompt_view(consolidated, max_items=max_items)

    # ------------------------------------------------------------------ #
    # select_answer_citations
    # ------------------------------------------------------------------ #
    def select_answer_citations(
        self,
        consolidated: list[dict[str, Any]],
        all_citations: list[str],
        *,
        max_items: int,
    ) -> list[str]:
        max_items = max(0, int(max_items or 0))
        if max_items == 0:
            return []

        selected: list[str] = []
        seen: set[str] = set()

        def _add(url: str) -> None:
            n = normalize_url(str(url))
            if not n or n in seen:
                return
            seen.add(n)
            selected.append(n)

        if consolidated:
            status_order = {"verified": 0, "contested": 1, "unverified": 2}
            ranked_claims = sorted(
                list(consolidated),
                key=lambda c: (
                    status_order.get(str(c.get("status", "unverified")), 3),
                    -int(c.get("support_count", 0) or 0),
                    int(c.get("contradict_count", 0) or 0),
                ),
            )

            for claim in ranked_claims:
                status = str(claim.get("status", "unverified"))
                if status not in ("verified", "contested"):
                    continue
                for u in claim.get("source_urls", [])[:4]:
                    _add(str(u))
                    if len(selected) >= max_items:
                        return selected[:max_items]

            min_target = min(20, max_items)
            if len(selected) < min_target:
                for claim in ranked_claims:
                    if str(claim.get("status", "unverified")) != "unverified":
                        continue
                    tiers = claim.get("source_tier_counts", {}) or {}
                    has_quality = (
                        int(tiers.get("primary", 0) or 0) > 0
                        or int(tiers.get("mainstream", 0) or 0) > 0
                    )
                    if not has_quality:
                        continue
                    for u in claim.get("source_urls", [])[:3]:
                        _add(str(u))
                        if len(selected) >= max_items:
                            return selected[:max_items]

        if len(selected) < min(15, max_items):
            for u in all_citations:
                _add(str(u))
                if len(selected) >= max_items:
                    break

        return selected[:max_items]


# ===================================================================== #
# 4. ContextPruningStrategy
# ===================================================================== #

class ContextPruningStrategy(ABC):
    """Prune context blocks to stay within budget."""

    @abstractmethod
    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
    ) -> list[str]:
        """Return pruned context list."""


class RelevanceBasedPruning(ContextPruningStrategy):
    """Reproduce ``_prune_context`` -- relevance-based instead of FIFO."""

    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
    ) -> list[str]:
        if len(context) <= max_blocks:
            return context

        # Reference words from question + sub-questions
        ref_text = question.lower()
        for sq in sub_questions:
            ref_text += " " + sq.lower()
        ref_words = {w for w in ref_text.split() if len(w) > 2 and w not in STOPWORDS}

        if not ref_words:
            # Fallback: FIFO
            return context[-max_blocks:]

        # Separate protected (new) and candidate (old) blocks
        n_protected = min(n_new, len(context))
        protected = context[-n_protected:]
        candidates = context[:-n_protected] if n_protected > 0 else context[:]

        # How many old blocks to keep?
        keep_old = max_blocks - len(protected)
        if keep_old <= 0:
            return protected[-max_blocks:]
        if len(candidates) <= keep_old:
            return candidates + protected

        # Score: fraction of reference words present in block
        scored: list[tuple[float, int, str]] = []
        for idx, block in enumerate(candidates):
            block_lower = block.lower()
            block_words = set(block_lower.split())
            overlap = len(ref_words & block_words)
            score = overlap / len(ref_words)
            scored.append((score, idx, block))

        # Sort by score descending, then by index ascending (oldest first out)
        scored.sort(key=lambda x: (-x[0], x[1]))
        kept = [block for _, _, block in scored[:keep_old]]

        return kept + protected


# ===================================================================== #
# 5. RiskScoringStrategy
# ===================================================================== #

class RiskScoringStrategy(ABC):
    """Score risk, infer query type, derive aspects, inject quality queries."""

    @abstractmethod
    def score(self, question: str) -> int: ...

    @abstractmethod
    def infer_query_type(self, question: str) -> str: ...

    @abstractmethod
    def derive_required_aspects(self, question: str, query_type: str) -> list[str]: ...

    @abstractmethod
    def estimate_aspect_coverage(
        self,
        aspects: list[str],
        context_blocks: list[str],
    ) -> tuple[list[str], float]: ...

    @abstractmethod
    def quality_terms_for_question(self, question: str, query_type: str) -> list[str]: ...

    @abstractmethod
    def inject_quality_site_queries(
        self,
        queries: list[str],
        *,
        search_lang: str,
        question: str,
        query_type: str,
        need_primary: bool,
        need_mainstream: bool,
        max_items: int,
    ) -> list[str]: ...


class KeywordRiskScorer(RiskScoringStrategy):
    """Reproduce ``_compute_risk_score`` and related helpers."""

    # ------------------------------------------------------------------ #
    # score
    # ------------------------------------------------------------------ #
    def score(self, question: str) -> int:
        q = (question or "").lower()
        score = 0

        if re.search(
            r"\b(gesetz\w*|recht\w*|verordnung\w*|regulier\w*|politik\w*|koalition\w*"
            r"|gkv|beitrag\w*|haushalt\w*|privatis\w*)\b",
            q,
        ):
            score += 2
        if re.search(r"\b(aktuell|heute|neueste|zuletzt|diskussion|trend|ausblick|prognose)\b", q):
            score += 1
        if re.search(r"\b(prozent|mrd|mio|euro|\d+[%\u20ac]?)\b", q):
            score += 1
        if re.search(r"\b(soll|sollen|geplant|durchsetzbar|realistisch)\b", q):
            score += 1
        if len(question) > 220:
            score += 1

        return min(score, 10)

    # ------------------------------------------------------------------ #
    # infer_query_type
    # ------------------------------------------------------------------ #
    def infer_query_type(self, question: str) -> str:
        q = (question or "").lower()
        if re.search(r"\b(paper|studie|study|doi|arxiv|journal|conference|peer-review)\b", q):
            return "academic"
        if re.search(r"\b(heute|aktuell|news|nachricht|meld|diskussion|debatte)\b", q):
            return "news"
        return "general"

    # ------------------------------------------------------------------ #
    # derive_required_aspects
    # ------------------------------------------------------------------ #
    def derive_required_aspects(self, question: str, query_type: str) -> list[str]:
        q = (question or "").lower()
        aspects: list[str] = []

        if query_type in ("news", "general"):
            aspects.extend([
                "Status quo mit konkretem Datum",
                "Positionen zentraler Akteure",
                "Richtung der laufenden Diskussion",
            ])
        if re.search(r"\b(soll|sollen|privatis|reform|gesetz|regel|politik)\b", q):
            aspects.extend([
                "Politische Umsetzbarkeit und Mehrheitslage",
                "Abgrenzung zwischen Vorschlag und beschlossener Regel",
            ])
        if re.search(r"\b(prozent|mrd|mio|euro|kosten|beitrag|ausgaben)\b", q):
            aspects.append("Zahlenbasis mit Primärbeleg oder expliziter Unsicherheit")
        if query_type == "academic":
            aspects.extend([
                "Primaerpublikation und Kernaussage",
                "Methodik und Limitationen",
            ])

        dedup: list[str] = []
        for aspect in aspects:
            if aspect not in dedup:
                dedup.append(aspect)
        return dedup[:6]

    # ------------------------------------------------------------------ #
    # estimate_aspect_coverage
    # ------------------------------------------------------------------ #
    def estimate_aspect_coverage(
        self,
        aspects: list[str],
        context_blocks: list[str],
    ) -> tuple[list[str], float]:
        if not aspects:
            return [], 1.0
        text = " ".join(context_blocks).lower()
        if not text.strip():
            return list(aspects), 0.0

        uncovered: list[str] = []
        for aspect in aspects:
            aspect_l = aspect.lower()
            tokens = [t for t in tokenize(aspect) if len(t) > 3]
            if "status quo" in aspect_l:
                tokens.extend(["status", "stand", "aktuell", "derzeit"])
            if "datum" in aspect_l:
                tokens.extend(["datum", "stand", "heute"])
            if "position" in aspect_l and "akteur" in aspect_l:
                tokens.extend(["position", "regierung", "partei", "verband", "akteur"])
            if "richtung" in aspect_l or "diskussion" in aspect_l:
                tokens.extend(["richtung", "trend", "debatte", "diskussion", "entwicklung"])
            if "mehrheitslage" in aspect_l or "umsetzbarkeit" in aspect_l:
                tokens.extend(["mehrheit", "mehrheitsfaehig", "durchsetzbar", "umsetzbar"])

            # Deduplicate (preserve order)
            dedup_tokens: list[str] = []
            for tok in tokens:
                if tok not in dedup_tokens:
                    dedup_tokens.append(tok)
            tokens = dedup_tokens

            if not tokens:
                continue
            hits = sum(1 for t in tokens if t in text)
            if hits == 0:
                uncovered.append(aspect)

        covered = len(aspects) - len(uncovered)
        return uncovered, round(covered / len(aspects), 3)

    # ------------------------------------------------------------------ #
    # quality_terms_for_question
    # ------------------------------------------------------------------ #
    def quality_terms_for_question(self, question: str, query_type: str) -> list[str]:
        q = (question or "").strip()
        ql = q.lower()
        terms: list[str] = []

        if re.search(r"\bzahn", ql):
            terms.extend(["zahnbehandlung", "zahnleistungen"])
        if re.search(r"privatis", ql):
            terms.append("privatisierung")
        if query_type in ("news", "general") and not re.search(r"\bgkv\b|krankenkass", ql):
            terms.append("gkv")

        for tok in tokenize(q):
            if len(tok) < 4:
                continue
            if tok in STOPWORDS:
                continue
            if tok in GENERIC_QUERY_TERMS_DE:
                continue
            if tok not in terms:
                terms.append(tok)
            if len(terms) >= 7:
                break

        return terms[:7]

    # ------------------------------------------------------------------ #
    # inject_quality_site_queries
    # ------------------------------------------------------------------ #
    def inject_quality_site_queries(
        self,
        queries: list[str],
        *,
        search_lang: str,
        question: str,
        query_type: str,
        need_primary: bool,
        need_mainstream: bool,
        max_items: int,
    ) -> list[str]:
        if (search_lang or "").lower() != "de":
            return queries[:max_items]
        if max_items <= 0:
            return []

        base_terms = self.quality_terms_for_question(question, query_type)
        if not base_terms:
            return queries[:max_items]

        def _already_has_site(domains: list[str]) -> bool:
            for q in queries:
                ql_inner = (q or "").lower()
                if "site:" in ql_inner:
                    for d in domains:
                        if f"site:{d}" in ql_inner:
                            return True
                for d in domains:
                    if d in ql_inner:
                        return True
            return False

        inject: list[str] = []
        if need_primary and not _already_has_site(QUALITY_PRIMARY_SITES_DE):
            inject.append(f"site:{QUALITY_PRIMARY_SITES_DE[0]} " + " ".join(base_terms[:6]))
        if need_mainstream and not _already_has_site(QUALITY_MAINSTREAM_SITES_DE):
            inject.append(f"site:{QUALITY_MAINSTREAM_SITES_DE[0]} " + " ".join(base_terms[:6]))

        out: list[str] = []
        seen: set[str] = set()
        for q in inject + list(queries or []):
            qq = (q or "").strip()
            if not qq or qq in seen:
                continue
            out.append(qq)
            seen.add(qq)
            if len(out) >= max_items:
                break
        return out


# ===================================================================== #
# 6. StopCriteriaStrategy
# ===================================================================== #

class StopCriteriaStrategy(ABC):
    """Evaluate whether the research loop should stop."""

    @abstractmethod
    def check_contradictions(self, s: dict, eval_text: str, conf: int) -> int: ...

    @abstractmethod
    def filter_irrelevant_blocks(self, s: dict, eval_text: str) -> None: ...

    @abstractmethod
    def extract_competing_events(self, s: dict, eval_text: str, conf: int) -> int: ...

    @abstractmethod
    def extract_evidence_scores(self, s: dict, eval_text: str, conf: int) -> int: ...

    @abstractmethod
    def check_falsification(self, s: dict, conf: int, prev_conf: int) -> bool: ...

    @abstractmethod
    def check_stagnation(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        n_citations: int,
        falsification_just_triggered: bool,
    ) -> tuple[int, bool]: ...

    @abstractmethod
    def should_suppress_utility_stop(self, s: dict) -> bool: ...

    @abstractmethod
    def compute_utility(
        self, s: dict, conf: int, prev_conf: int, n_citations: int,
    ) -> tuple[float, bool]: ...

    @abstractmethod
    def check_plateau(
        self, s: dict, conf: int, prev_conf: int, stagnation_detected: bool,
    ) -> bool: ...

    @abstractmethod
    def should_stop(self, state: dict) -> tuple[bool, str]: ...


def _emit_progress(s: dict, message: str) -> None:
    """Send a progress update to the stream (thin helper)."""
    q = s.get("progress")
    if q is not None:
        q.put(("progress", message))


class MultiSignalStopCriteria(StopCriteriaStrategy):
    """Reproduce the ``_check_*`` / ``_compute_*`` family from the original agent."""

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings

    @property
    def _confidence_stop(self) -> int:
        return self._settings.confidence_stop

    @property
    def _max_rounds(self) -> int:
        return self._settings.max_rounds

    # ------------------------------------------------------------------ #
    # check_contradictions
    # ------------------------------------------------------------------ #
    def check_contradictions(self, s: dict, eval_text: str, conf: int) -> int:
        m = re.search(r"CONTRADICTIONS:\s*(.+?)(?:\n|$)", eval_text)
        if not m or "ja" not in m.group(1).lower():
            return conf
        contradiction_text = m.group(1).lower()
        severe_keywords = (
            "grundlegend", "fundamental", "gegenteil",
            "widerspricht", "unvereinbar", "komplett",
            "voellig", "falsch", "inkorrekt", "gegensaetzlich",
        )
        if any(kw in contradiction_text for kw in severe_keywords):
            _emit_progress(s, "Schwere Widersprueche erkannt \u2014 Confidence stark begrenzt")
            return min(conf, self._confidence_stop - 2)
        _emit_progress(s, "Leichte Widersprueche erkannt (z.B. Datumsabweichungen)")
        return min(conf, self._confidence_stop - 1)

    # ------------------------------------------------------------------ #
    # filter_irrelevant_blocks
    # ------------------------------------------------------------------ #
    def filter_irrelevant_blocks(self, s: dict, eval_text: str) -> None:
        m = re.search(r"IRRELEVANT:\s*(.+?)(?:\n|$)", eval_text)
        if not m:
            return
        irr_text = m.group(1).strip()
        if is_none_value(irr_text):
            return
        try:
            drop_indices = {
                int(x.strip()) - 1 for x in irr_text.split(",") if x.strip().isdigit()
            }
            if drop_indices and len(drop_indices) < len(s["context"]):
                filtered = [b for i, b in enumerate(s["context"]) if i not in drop_indices]
                dropped = len(s["context"]) - len(filtered)
                s["context"] = filtered
                if dropped:
                    _emit_progress(s, f"{dropped} irrelevante Quellen gefiltert")
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    # extract_competing_events
    # ------------------------------------------------------------------ #
    def extract_competing_events(self, s: dict, eval_text: str, conf: int) -> int:
        m = re.search(r"COMPETING_EVENTS:\s*(.+?)(?:\n|$)", eval_text)
        if not m:
            return conf
        comp_text = m.group(1).strip()
        if is_none_value(comp_text):
            s["competing_events"] = ""
            log.info(
                "TRACE evaluate: competing_events=None (parsed '%s')",
                comp_text.lower()[:60],
            )
            return conf

        s["competing_events"] = comp_text
        log.info("TRACE evaluate: competing_events='%s'", comp_text[:200])
        _emit_progress(s, "Mehrere moegliche Erklaerungen erkannt")

        _prev_comp = s.get("prev_competing_events", "")
        _comp_is_new = (not _prev_comp) or (comp_text != _prev_comp)
        if conf >= self._confidence_stop and (_comp_is_new or s["round"] < 3):
            conf = self._confidence_stop - 1
            log.info(
                "TRACE evaluate: competing_events cap applied (is_new=%s, round=%d)",
                _comp_is_new, s["round"],
            )
        elif conf >= self._confidence_stop:
            log.info(
                "TRACE evaluate: competing_events cap SKIPPED "
                "(same text for 2+ rounds, round=%d)",
                s["round"],
            )
        return conf

    # ------------------------------------------------------------------ #
    # extract_evidence_scores
    # ------------------------------------------------------------------ #
    def extract_evidence_scores(self, s: dict, eval_text: str, conf: int) -> int:
        m_consistency = re.search(r"EVIDENCE_CONSISTENCY:\s*(\d+)", eval_text)
        s["evidence_consistency"] = int(m_consistency.group(1)) if m_consistency else 5

        m_sufficiency = re.search(r"EVIDENCE_SUFFICIENCY:\s*(\d+)", eval_text)
        s["evidence_sufficiency"] = int(m_sufficiency.group(1)) if m_sufficiency else 5

        if (
            s["evidence_consistency"] == 0
            and s["evidence_sufficiency"] == 0
            and conf >= self._confidence_stop
        ):
            log.warning(
                "TRACE evaluate: evidence sanity check failed "
                "(consistency=0, sufficiency=0, conf=%d -> %d)",
                conf, self._confidence_stop - 1,
            )
            conf = self._confidence_stop - 1

        log.info(
            "TRACE evaluate: evidence_consistency=%d evidence_sufficiency=%d",
            s["evidence_consistency"], s["evidence_sufficiency"],
        )
        return conf

    # ------------------------------------------------------------------ #
    # check_falsification
    # ------------------------------------------------------------------ #
    def check_falsification(self, s: dict, conf: int, prev_conf: int) -> bool:
        if (
            s["round"] >= 2
            and prev_conf > 0
            and prev_conf <= 4
            and conf <= 4
            and not s.get("falsification_triggered", False)
        ):
            s["falsification_triggered"] = True
            log.info(
                "TRACE evaluate: falsification triggered (prev=%d, curr=%d, round=%d)",
                prev_conf, conf, s["round"],
            )
            _emit_progress(s, "Niedrige Evidenz \u2014 starte Falsifikations-Recherche")
            return True
        return False

    # ------------------------------------------------------------------ #
    # check_stagnation
    # ------------------------------------------------------------------ #
    def check_stagnation(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        n_citations: int,
        falsification_just_triggered: bool,
    ) -> tuple[int, bool]:
        if (
            s["round"] >= 2
            and prev_conf > 0
            and prev_conf <= 4
            and conf <= 4
            and abs(conf - prev_conf) <= 1
            and not falsification_just_triggered
            and (n_citations >= 30 or s.get("falsification_triggered", False))
        ):
            log.info(
                "TRACE evaluate: stagnation detected "
                "(prev=%d, curr=%d, citations=%d, falsified=%s) -> forcing stop",
                prev_conf, conf, n_citations, s.get("falsification_triggered", False),
            )
            _emit_progress(
                s,
                "Umfangreiche Recherche abgeschlossen \u2014 "
                "Praemisse der Frage wahrscheinlich falsch",
            )
            return self._confidence_stop, True
        return conf, False

    # ------------------------------------------------------------------ #
    # should_suppress_utility_stop
    # ------------------------------------------------------------------ #
    def should_suppress_utility_stop(self, s: dict) -> bool:
        if int(s.get("round", 0) or 0) >= self._max_rounds:
            return False
        if str(s.get("query_type", "general")) == "academic":
            return False

        ql = (s.get("question", "") or "").lower()
        is_policyish = bool(
            re.search(
                r"\b(privatis\w*|gkv|krankenkass\w*|gesetz\w*|recht\w*|verordnung\w*"
                r"|regulier\w*|politik\w*|beitrag\w*|kosten|haushalt\w*)\b",
                ql,
            )
        )
        if not is_policyish:
            return False

        uncovered_n = len(s.get("uncovered_aspects", []) or [])
        if uncovered_n > 0:
            return True

        tiers = s.get("source_tier_counts", {}) or {}
        primary_n = int(tiers.get("primary", 0) or 0)
        mainstream_n = int(tiers.get("mainstream", 0) or 0)

        claim_quality = float(s.get("claim_quality_score", 0.0) or 0.0)
        claim_counts = s.get("claim_status_counts", {}) or {}
        verified = int(claim_counts.get("verified", 0) or 0)
        unverified = int(claim_counts.get("unverified", 0) or 0)

        np_total = int(s.get("claim_needs_primary_total", 0) or 0)
        np_verified = int(s.get("claim_needs_primary_verified", 0) or 0)
        if np_total > 0 and np_verified < np_total:
            return True

        if (primary_n + mainstream_n) == 0 and (unverified > verified or claim_quality < 0.35):
            return True

        return False

    # ------------------------------------------------------------------ #
    # compute_utility
    # ------------------------------------------------------------------ #
    def compute_utility(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        n_citations: int,
    ) -> tuple[float, bool]:
        _delta_conf = (conf - prev_conf) / 10.0 if prev_conf > 0 else 0.5
        _new_cit = n_citations - s.get("prev_citation_count", 0)
        _delta_cit_norm = min(1.0, _new_cit / 10.0)
        _sufficiency_norm = s.get("evidence_sufficiency", 5) / 10.0
        utility = round(
            0.4 * _delta_conf + 0.3 * _delta_cit_norm + 0.3 * _sufficiency_norm, 4,
        )
        s["utility_scores"].append(utility)
        s["prev_citation_count"] = n_citations

        log.info(
            "TRACE evaluate: utility=%.4f (delta_conf=%.2f delta_cit=%.2f suff=%.2f) scores=%s",
            utility, _delta_conf, _delta_cit_norm, _sufficiency_norm,
            [round(u, 3) for u in s["utility_scores"]],
        )

        utility_stop = False
        if len(s["utility_scores"]) >= 2 and not s["done"]:
            _last = s["utility_scores"][-1]
            _prev_u = s["utility_scores"][-2]
            if _last < 0.15 and _prev_u < 0.15:
                if self.should_suppress_utility_stop(s):
                    log.info(
                        "TRACE evaluate: utility stop SUPPRESSED "
                        "(evidence still weak) scores=%s",
                        [round(u, 3) for u in s["utility_scores"][-2:]],
                    )
                    _emit_progress(
                        s,
                        "Informationsgewinn stagniert, aber Evidenzlage "
                        "noch zu schwach \u2014 suche weiter",
                    )
                else:
                    utility_stop = True
                    s["done"] = True
                    log.info(
                        "TRACE evaluate: utility stop triggered (scores=%s)",
                        [round(u, 3) for u in s["utility_scores"][-2:]],
                    )
                    _emit_progress(s, "Informationsgewinn stagniert \u2014 beende Recherche")
        return utility, utility_stop

    # ------------------------------------------------------------------ #
    # check_plateau
    # ------------------------------------------------------------------ #
    def check_plateau(
        self,
        s: dict,
        conf: int,
        prev_conf: int,
        stagnation_detected: bool,
    ) -> bool:
        _prev_comp_for_plateau = s.get("prev_competing_events", "")
        _current_competing = s.get("competing_events", "")
        _competing_active_and_changing = (
            bool(_current_competing)
            and _current_competing != _prev_comp_for_plateau
        )

        _conf_stable_rounds = s.get("_conf_stable_rounds", 0)
        if prev_conf > 0 and conf == prev_conf:
            _conf_stable_rounds += 1
        else:
            _conf_stable_rounds = 0
        s["_conf_stable_rounds"] = _conf_stable_rounds

        if _competing_active_and_changing and _conf_stable_rounds >= 2:
            log.info(
                "TRACE evaluate: competing events override \u2014 conf stable for %d rounds, "
                "treating as non-changing for plateau check",
                _conf_stable_rounds,
            )
            _competing_active_and_changing = False

        plateau_stop = False
        if (
            s["round"] >= 2
            and prev_conf > 0
            and conf == prev_conf
            and conf >= 6
            and not stagnation_detected
            and not _competing_active_and_changing
            and not s["done"]
        ):
            plateau_stop = True
            s["done"] = True
            log.info(
                "TRACE evaluate: plateau stop triggered "
                "(conf=%d stable for 2 rounds, round=%d)",
                conf, s["round"],
            )
            _emit_progress(s, f"Confidence {conf}/10 stabil \u2014 Recherche abgeschlossen")
        elif _competing_active_and_changing and not s["done"]:
            log.info(
                "TRACE evaluate: plateau stop SUPPRESSED "
                "(competing events changed, round=%d)",
                s["round"],
            )

        s["prev_competing_events"] = _current_competing
        return plateau_stop

    # ------------------------------------------------------------------ #
    # should_stop  (combined convenience method)
    # ------------------------------------------------------------------ #
    def should_stop(self, state: dict) -> tuple[bool, str]:
        """High-level stop check combining all signals.

        Returns ``(should_stop, reason)`` where *reason* is a short tag
        describing which signal triggered the stop (empty when not stopping).

        This method is intentionally a thin wrapper.  The evaluate node
        typically calls the individual ``check_*`` / ``compute_*`` methods
        directly so it can thread confidence values between them; this
        combined entry-point is provided for simpler call-sites that only
        need a boolean answer.
        """
        if state.get("done"):
            return True, "already_done"

        conf = int(state.get("final_confidence", 0))
        if conf >= self._confidence_stop:
            return True, "confidence"

        if int(state.get("round", 0)) >= self._max_rounds:
            return True, "max_rounds"

        return False, ""


# ===================================================================== #
# 7. StrategyContext + factory
# ===================================================================== #

@dataclass
class StrategyContext:
    """Bundle of all pluggable strategies available to nodes."""

    source_tiering: SourceTieringStrategy
    claim_extraction: ClaimExtractionStrategy
    claim_consolidation: ClaimConsolidationStrategy
    context_pruning: ContextPruningStrategy
    risk_scoring: RiskScoringStrategy
    stop_criteria: StopCriteriaStrategy


def create_default_strategies(
    settings: AgentSettings,
    *,
    llm: LLMProvider | None = None,
    summarize_model: str = "",
    summarize_timeout: int = 60,
) -> StrategyContext:
    """Create a :class:`StrategyContext` with all default implementations.

    Parameters
    ----------
    settings:
        Agent-level configuration (thresholds, timeouts, etc.).
    llm:
        LLM provider used for default claim extraction. Custom providers
        can participate without exposing any private client internals.
    summarize_model:
        Model identifier used for claim extraction.
    summarize_timeout:
        Per-call timeout for the summarize model.
    """
    tiering = DefaultSourceTiering()
    return StrategyContext(
        source_tiering=tiering,
        claim_extraction=LLMClaimExtractor(
            llm=llm,
            summarize_model=summarize_model,
            summarize_timeout=summarize_timeout,
        ),
        claim_consolidation=DefaultClaimConsolidator(source_tiering=tiering),
        context_pruning=RelevanceBasedPruning(),
        risk_scoring=KeywordRiskScorer(),
        stop_criteria=MultiSignalStopCriteria(settings=settings),
    )
