"""Risk scoring strategy — score risk, infer query type, derive aspects."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from inqtrix.domains import GENERIC_QUERY_TERMS_DE, QUALITY_MAINSTREAM_SITES_DE, QUALITY_PRIMARY_SITES_DE
from inqtrix.report_profiles import ReportProfile
from inqtrix.text import STOPWORDS, aspect_synonyms, tokenize

log = logging.getLogger("inqtrix")


class RiskScoringStrategy(ABC):
    """Contract for question-level risk scoring and aspect derivation.

    Risk scoring is one of the cheapest and most useful early-loop
    signals: it gates whether classify and evaluate escalate to the
    reasoning model (via ``high_risk_*_escalate``) and seeds the
    ``required_aspects`` checklist that drives plan diversification
    and stop-cascade aspect coverage.

    Implementations are expected to be deterministic and side-effect-
    free so the same question always produces the same risk score and
    aspect list. LLM-backed scorers are allowed but discouraged for
    cost reasons; the default :class:`KeywordRiskScorer` uses regex
    families and token matching exclusively.
    """

    @abstractmethod
    def score(self, question: str) -> int:
        """Return a bounded deterministic risk score for ``question``.

        Args:
            question: User question to score. Must tolerate empty
                input and return ``0`` rather than raising.

        Returns:
            Integer in ``[0, 10]``. Values ``>= AgentSettings.high_risk_score_threshold``
            (default ``4``) trigger the reasoning-model escalation
            path in classify and evaluate.
        """
        ...

    @abstractmethod
    def infer_query_type(self, question: str) -> str:
        """Infer a coarse query-type label that the planner specialises on.

        Args:
            question: User question. Tolerate empty input.

        Returns:
            One of ``"general"`` (default), ``"news"`` (recency-biased
            phrasing), or ``"academic"`` (paper / DOI / journal
            language). Implementations may add backend-specific
            labels but should keep the three core values stable so
            the planner's existing branches keep working.
        """
        ...

    @abstractmethod
    def derive_required_aspects(
        self,
        question: str,
        query_type: str,
        report_profile: ReportProfile | str = ReportProfile.COMPACT,
    ) -> list[str]:
        """Derive the aspect checklist the loop should eventually cover.

        Args:
            question: User question. Aspects should reflect the
                question's domain and risk profile.
            query_type: Output of :meth:`infer_query_type`.
            report_profile: Active report profile. DEEP profiles
                typically request more aspects (background,
                stakeholders, counter-arguments) than COMPACT.

        Returns:
            Ordered list of human-readable aspect labels (German by
            default). Capped per implementation policy (the default
            uses ``6`` for COMPACT, ``11`` for DEEP). The evaluator
            and stop cascade compare context coverage against this
            list.
        """
        ...

    @abstractmethod
    def estimate_aspect_coverage(
        self,
        aspects: list[str],
        context_blocks: list[str],
    ) -> tuple[list[str], float]:
        """Estimate which required aspects are still uncovered.

        Args:
            aspects: Output of :meth:`derive_required_aspects`. Empty
                list short-circuits to "fully covered".
            context_blocks: Current context blocks accumulated by the
                search node. Empty list short-circuits to "nothing
                covered".

        Returns:
            Tuple ``(uncovered_aspects, coverage_ratio)`` where
            ``coverage_ratio`` is in ``[0.0, 1.0]`` rounded to three
            decimal places. Used directly by the stop cascade and
            surfaced as ``ResearchMetrics.aspect_coverage``.
        """
        ...

    @abstractmethod
    def quality_terms_for_question(self, question: str, query_type: str) -> list[str]:
        """Return search terms suitable for quality-site query injection.

        Args:
            question: User question.
            query_type: Output of :meth:`infer_query_type`.

        Returns:
            Ordered list of lowercase German tokens (or backend-
            equivalent) capped at ~7 entries; combined with
            ``site:`` operators by :meth:`inject_quality_site_queries`.
            Empty list when the question yields no usable
            quality-anchor tokens.
        """
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
        """Prepend site-scoped quality queries to the planner's query list.

        Args:
            queries: Original planned queries (preserved relative
                order; never mutated).
            search_lang: ISO-639 language code chosen for this round.
                Implementations may no-op for non-target languages
                (the default scorer only activates for ``"de"``).
            question: Original user question, used for term extraction.
            query_type: Output of :meth:`infer_query_type`.
            need_primary: Whether the loop has signalled it is short
                on primary-tier evidence (drives ``site:`` injection
                against the primary-quality domain table).
            need_mainstream: Same for mainstream-tier evidence.
            max_items: Hard cap on the returned list length.

        Returns:
            New list with injected quality queries first (when active)
            followed by the original queries, deduplicated and capped
            at ``max_items``. Returns ``[]`` when ``max_items <= 0``.
        """
        ...


class KeywordRiskScorer(RiskScoringStrategy):
    """Deterministic regex-based default risk scorer.

    No LLM calls — uses regex families and token matching so classify,
    plan and search share stable heuristics that are cheap to
    evaluate and easy to regression-test. Tuned for German health-
    and social-policy topics; activates the ``enable_de_policy_bias``
    branches via :meth:`quality_terms_for_question` and
    :meth:`inject_quality_site_queries`.

    Suitable for production as the default; subclass to replace the
    keyword tables, or implement :class:`RiskScoringStrategy`
    from scratch for non-German domains.
    """

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        """Return a copy of ``items`` with duplicates removed, order preserved.

        Args:
            items: Input list. Not mutated.

        Returns:
            New list containing the first occurrence of each value.
        """
        deduped: list[str] = []
        for item in items:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @classmethod
    def _aspect_tokens(cls, aspect: str) -> list[str]:
        """Return aspect tokens for substring scan: raw tokens + shared synonyms."""
        tokens = [t for t in tokenize(aspect) if len(t) > 3]
        tokens.extend(aspect_synonyms(aspect))
        return cls._dedupe_keep_order(tokens)

    # ------------------------------------------------------------------ #
    # score
    # ------------------------------------------------------------------ #
    def score(self, question: str) -> int:
        """Compute a regex-based risk score capped at 10.

        Adds:
        - ``+2`` for political / regulatory vocabulary
          (``gesetz``, ``recht``, ``verordnung``, ``politik``, ``gkv``,
          ``haushalt``, ``privatis``).
        - ``+1`` for current-affairs markers (``aktuell``, ``heute``,
          ``trend``, ``ausblick``).
        - ``+1`` for numeric / monetary mentions
          (``prozent``, ``mrd``, ``euro``, ``\\d+%``).
        - ``+1`` for normative phrasing
          (``soll``, ``geplant``, ``durchsetzbar``).
        - ``+1`` for very long questions (> 220 chars).

        Implements the contract from
        :meth:`RiskScoringStrategy.score`.

        Args:
            question: User question. Empty / ``None`` returns ``0``.

        Returns:
            Integer in ``[0, 10]``. ``10`` is the hard cap regardless
            of how many regex families fire.
        """
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
        """Classify the question as ``academic``, ``news`` or ``general``.

        Implements the contract from
        :meth:`RiskScoringStrategy.infer_query_type`. Precedence:
        academic markers (``paper``, ``studie``, ``doi``, ``arxiv``,
        ``journal``, ``peer-review``) win over news markers
        (``heute``, ``aktuell``, ``debatte``); everything else maps
        to ``general``.

        Args:
            question: User question. Empty / ``None`` returns
                ``"general"``.

        Returns:
            One of ``"academic"``, ``"news"``, or ``"general"``.
        """
        q = (question or "").lower()
        if re.search(r"\b(paper|studie|study|doi|arxiv|journal|conference|peer-review)\b", q):
            return "academic"
        if re.search(r"\b(heute|aktuell|news|nachricht|meld|diskussion|debatte)\b", q):
            return "news"
        return "general"

    # ------------------------------------------------------------------ #
    # derive_required_aspects
    # ------------------------------------------------------------------ #
    def derive_required_aspects(
        self,
        question: str,
        query_type: str,
        report_profile: ReportProfile | str = ReportProfile.COMPACT,
    ) -> list[str]:
        """Derive a German-language aspect checklist for the loop to cover.

        Implements the contract from
        :meth:`RiskScoringStrategy.derive_required_aspects`. Three
        layers are merged and deduplicated in order:

        1. Base aspects per ``query_type`` (``news`` / ``general`` get
           status-quo + actor-positions + discussion-direction;
           ``academic`` gets primary publication + methodology).
        2. Topic-specific extensions (regulatory / political phrasing
           triggers Mehrheitslage and Vorschlag-vs-beschlossen;
           numeric phrasing triggers Zahlenbasis-mit-Primärbeleg).
        3. DEEP profile extensions (background, drivers, stakeholders,
           counter-arguments, comparisons).

        Args:
            question: User question.
            query_type: Output of :meth:`infer_query_type`.
            report_profile: Active report profile. ``COMPACT`` caps
                the result at ``6`` aspects; ``DEEP`` at ``11``.

        Returns:
            Ordered, deduplicated list of German aspect labels.
            Empty when ``question`` is empty AND ``query_type`` is
            unknown (defensive — production calls always populate at
            least the base layer).
        """
        q = (question or "").lower()
        aspects: list[str] = []
        try:
            normalized_profile = ReportProfile(report_profile)
        except ValueError:
            normalized_profile = ReportProfile.COMPACT

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

        if normalized_profile is ReportProfile.DEEP:
            aspects.extend([
                "Hintergrund und Ausgangslage",
                "Wesentliche Treiber und Mechanismen",
                "Stakeholder- und Betroffenenperspektiven",
                "Gegenargumente, Risiken und Limitationen",
                "Vergleich mit Alternativen oder Gegenmodellen",
            ])

        dedup = self._dedupe_keep_order(aspects)
        max_items = 11 if normalized_profile is ReportProfile.DEEP else 6
        return dedup[:max_items]

    # ------------------------------------------------------------------ #
    # estimate_aspect_coverage
    # ------------------------------------------------------------------ #
    def estimate_aspect_coverage(
        self,
        aspects: list[str],
        context_blocks: list[str],
    ) -> tuple[list[str], float]:
        """Compute the uncovered-aspect list and coverage ratio.

        Implements the contract from
        :meth:`RiskScoringStrategy.estimate_aspect_coverage`. An
        aspect is considered covered when at least one of its
        substring tokens (raw + synonyms via
        :func:`~inqtrix.text.aspect_synonyms`) appears anywhere in
        the joined lowercase context text. Substring matching is
        used (not whole-word) so morphological variants count.

        Args:
            aspects: Output of :meth:`derive_required_aspects`. Empty
                list returns ``([], 1.0)`` (degenerate "fully
                covered").
            context_blocks: Current context blocks. Empty / blank
                joined text returns ``(list(aspects), 0.0)``.

        Returns:
            Tuple ``(uncovered_aspects, coverage_ratio)``. Ratio is
            in ``[0.0, 1.0]`` rounded to three decimals.
        """
        if not aspects:
            return [], 1.0
        text = " ".join(context_blocks).lower()
        if not text.strip():
            return list(aspects), 0.0

        uncovered: list[str] = []
        for aspect in aspects:
            tokens = self._aspect_tokens(aspect)

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
