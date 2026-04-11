"""Risk scoring strategy — score risk, infer query type, derive aspects."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from inqtrix.domains import GENERIC_QUERY_TERMS_DE, QUALITY_MAINSTREAM_SITES_DE, QUALITY_PRIMARY_SITES_DE
from inqtrix.text import STOPWORDS, tokenize

log = logging.getLogger("inqtrix")


class RiskScoringStrategy(ABC):
    """Score risk, infer query type, derive aspects, inject quality queries."""

    @abstractmethod
    def score(self, question: str) -> int:
        """Return a bounded deterministic risk score for *question*."""
        ...

    @abstractmethod
    def infer_query_type(self, question: str) -> str:
        """Infer the coarse query type such as ``general``, ``news``, or ``academic``."""
        ...

    @abstractmethod
    def derive_required_aspects(self, question: str, query_type: str) -> list[str]:
        """Derive the aspect checklist the loop should eventually cover."""
        ...

    @abstractmethod
    def estimate_aspect_coverage(
        self,
        aspects: list[str],
        context_blocks: list[str],
    ) -> tuple[list[str], float]:
        """Estimate uncovered aspects and the aggregate coverage ratio."""
        ...

    @abstractmethod
    def quality_terms_for_question(self, question: str, query_type: str) -> list[str]:
        """Return search terms suitable for quality-site query injection."""
        ...

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
    ) -> list[str]:
        """Prepend site-scoped quality queries while preserving uniqueness and caps."""
        ...


class KeywordRiskScorer(RiskScoringStrategy):
    """Deterministic heuristics for risk, aspects, and quality-query injection.

    The scorer deliberately avoids LLM calls. It uses regex families and token
    matching so classify, plan, and search can share stable heuristics that are
    cheap to evaluate and easy to regression-test.
    """

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
        """Extract domain terms that sharpen quality-site queries.

        The helper keeps question-specific tokens, removes generic filler, and
        injects a few domain synonyms for the German health-policy use cases
        that the default quality-site lists are tuned for.
        """
        q = (question or "").strip()
        ql = q.lower()
        terms: list[str] = []
        is_health_topic = bool(
            re.search(
                r"\b(gkv|krankenkass\w*|krankenversicher\w*|gesundheits\w*|zahn\w*)\b",
                ql,
            )
        )

        if re.search(r"\bzahn", ql):
            terms.extend(["zahnbehandlung", "zahnleistungen"])
        if re.search(r"privatis", ql):
            terms.append("privatisierung")
        if (
            query_type in ("news", "general")
            and is_health_topic
            and not re.search(r"\bgkv\b|krankenkass", ql)
        ):
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
        """Inject site-filtered German quality queries ahead of generic queries.

        The method is intentionally conservative: it only activates for German
        search, respects ``max_items``, preserves uniqueness, and leaves the
        original query list untouched for non-German searches.
        """
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
