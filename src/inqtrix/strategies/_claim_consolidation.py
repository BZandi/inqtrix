"""Claim consolidation strategy — consolidate, filter, and format claim ledgers."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any

from inqtrix.domains import GENERIC_QUERY_TERMS_DE, SOURCE_TIER_WEIGHTS
from inqtrix.prompts import default_claims_prompt_view
from inqtrix.strategies._source_tiering import SourceTieringStrategy
from inqtrix.text import NEGATION_TOKENS, STOPWORDS, is_none_value, norm_match_token, tokenize
from inqtrix.urls import domain_from_url, normalize_url

_STATUS_WEIGHTS: dict[str, float] = {
    "verified": 1.0,
    "contested": 0.7,
    "unverified": 0.35,
}
_DEFAULT_TIER_WEIGHT = SOURCE_TIER_WEIGHTS.get("unknown", 0.35)
_SUPPORT_BONUS_SCALE = math.log1p(10)
_PHASE2_MAX_PER_DOMAIN = 3

log = logging.getLogger("inqtrix")


class ClaimConsolidationStrategy(ABC):
    """Consolidate, filter, and format claim ledgers."""

    @abstractmethod
    def focus_stems_from_question(self, question: str) -> set[str]:
        """Build normalized focus stems from the user question.

        Args:
            question: Original user question.

        Returns:
            A set of normalized stems used to keep claims on-topic.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.focus_stems_from_question("Was kostet die GKV-Reform?")
            {"gkv", "reform", ...}
        """
        ...

    @abstractmethod
    def claim_matches_focus_stems(
        self,
        claim_text: str,
        focus_stems: set[str],
    ) -> bool:
        """Decide whether a claim is relevant to the question focus.

        Args:
            claim_text: Candidate claim text.
            focus_stems: Normalized stems produced from the question.

        Returns:
            ``True`` when the claim should remain in scope.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.claim_matches_focus_stems("Die Reform kostet mehr", {"reform"})
            True
        """
        ...

    @abstractmethod
    def claim_signature(self, text: str) -> str:
        """Create a stable signature for deduplicating claim text.

        Args:
            text: Raw claim text.

        Returns:
            A normalized signature used for grouping semantically similar
            claims.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.claim_signature("Der Beitrag steigt um 5 Prozent")
            'beitrag steigt 5 prozent'
        """
        ...

    @abstractmethod
    def consolidate(self, claim_ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge a raw claim ledger into consolidated claim groups.

        Args:
            claim_ledger: Raw claim entries gathered across rounds.

        Returns:
            Consolidated claim records with support/contradiction counts.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.consolidate([{"claim_text": "..."}])
            [{"claim_text": "...", "support_count": 1, ...}]
        """
        ...

    @abstractmethod
    def materialize(
        self,
        consolidated: list[dict[str, Any]],
        *,
        max_total: int = 24,
        max_unverified: int = 8,
    ) -> list[dict[str, Any]]:
        """Select the answer-facing subset of consolidated claims.

        Args:
            consolidated: Consolidated claim groups.
            max_total: Maximum number of claims to keep.
            max_unverified: Maximum number of unverified claims to keep.

        Returns:
            Materialized claims ready for prompts and exports.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.materialize(consolidated_claims, max_total=10)
            [{"claim_text": "..."}, ...]
        """
        ...

    @abstractmethod
    def quality_metrics(
        self,
        consolidated: list[dict[str, Any]],
    ) -> tuple[dict[str, int], float, int, int]:
        """Summarize quality metrics for consolidated claims.

        Args:
            consolidated: Consolidated claim groups.

        Returns:
            Tuple of ``(status_counts, quality_score,
            needs_primary_total, needs_primary_verified)``.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.quality_metrics(consolidated_claims)
            ({"verified": 2}, 0.8, 1, 1)
        """
        ...

    @abstractmethod
    def claims_prompt_view(
        self,
        consolidated: list[dict[str, Any]],
        max_items: int = 16,
    ) -> str:
        """Render consolidated claims into prompt-friendly text.

        Args:
            consolidated: Consolidated claim groups.
            max_items: Maximum number of claims to render.

        Returns:
            A textual summary that can be embedded into prompts.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.claims_prompt_view(consolidated_claims, max_items=5)
            '- Claim 1\\n- Claim 2'
        """
        ...

    @abstractmethod
    def select_answer_citations(
        self,
        consolidated: list[dict[str, Any]],
        all_citations: list[str],
        *,
        max_items: int,
        source_tiering: SourceTieringStrategy | None = None,
    ) -> list[str]:
        """Choose the citations that should be forwarded to answer synthesis.

        Args:
            consolidated: Consolidated claim groups.
            all_citations: Full citation list collected during research.
            max_items: Hard cap for citations included in the answer prompt.
            source_tiering: Optional tiering strategy used to bias ranking
                toward higher-quality domains. When omitted, the claim's
                pre-computed tier counts are used as fallback.

        Returns:
            Citation URLs ordered for answer synthesis.

        Raises:
            NotImplementedError: Implementations must provide the logic.

        Example:
            >>> strategy.select_answer_citations(consolidated_claims, citations, max_items=8)
            ['https://example.com/report']
        """
        ...


class DefaultClaimConsolidator(ClaimConsolidationStrategy):
    """Group claims by signature and derive answer-facing claim quality.

    The consolidator is the bridge between raw extracted claims and the
    smaller, quality-scored claim set used in evaluate and answer. It keeps
    polarity-aware support counts, classifies each group as verified,
    contested, or unverified, and then materializes a bounded subset for
    prompts and exports.
    """

    def __init__(self, source_tiering: SourceTieringStrategy) -> None:
        """Bind a source-tiering strategy to the consolidator.

        Args:
            source_tiering: Strategy used to compute per-claim
                source-tier counts (consumed inside :meth:`consolidate`
                to derive the verified / contested / unverified
                status). Reused as-is; the consolidator does not
                wrap or shadow it.
        """
        self._tiering = source_tiering

    # ------------------------------------------------------------------ #
    # focus_stems_from_question
    # ------------------------------------------------------------------ #
    def focus_stems_from_question(self, question: str) -> set[str]:
        """Derive prefix stems from the question for relevance filtering.

        Implements the contract from
        :meth:`ClaimConsolidationStrategy.focus_stems_from_question`.
        Removes generic / stopword tokens, then truncates remaining
        tokens to a 6-character prefix so morphological variants
        ("Zahnarzt" / "Zahnaerzte" / "Zahnbehandlung") collapse to a
        single stem. Adds a hand-tuned stem set for German health-
        policy keywords (``zahn*``, ``privat*``, ``gkv``,
        ``krankenkass*``) where the auto-stemming would otherwise
        miss obvious matches.

        Args:
            question: User question to analyse. May contain mixed
                case and punctuation; tokenised internally.

        Returns:
            Set of lower-cased stem strings, each at least 4
            characters. Empty set when the question has no usable
            tokens (very short / all stopwords).
        """
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
        """Test whether at least one claim token starts with a focus stem.

        Implements the contract from
        :meth:`ClaimConsolidationStrategy.claim_matches_focus_stems`.
        Checks each claim token (length ``>= 4`` after normalisation)
        for a prefix match against the focus stem set.

        Args:
            claim_text: Claim text to test. Tokenised and normalised
                internally.
            focus_stems: Set of stems from
                :meth:`focus_stems_from_question`. An empty set is
                treated as "no filter" (every claim matches).

        Returns:
            ``True`` when at least one normalised claim token starts
            with a focus stem, or when ``focus_stems`` is empty.
            ``False`` otherwise.
        """
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
        """Compute a stable dedup signature for one claim string.

        Implements the contract from
        :meth:`ClaimConsolidationStrategy.claim_signature`. Drops
        stopwords, negation tokens and very short tokens, then keeps
        the first 16 informative tokens joined by spaces. Negation
        tokens are dropped intentionally so an affirmed and a
        contradicted version of the same claim share the signature
        and are tracked together via the polarity field instead.

        Args:
            text: Claim text to fingerprint. May contain mixed case
                and punctuation; tokenised internally.

        Returns:
            Lower-cased space-joined token string suitable as a dict
            key. Empty string when ``text`` has no usable tokens.
        """
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
        """Collapse ledger entries with the same normalized claim signature.

        Claims are first bucketed by signature, then merged into a single
        representative record with source-tier metadata and a derived status.
        Verification intentionally depends on affirmative support counts;
        purely negated evidence is kept visible via ``contradict_count`` but
        does not mark the claim as verified.
        """
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

            if support_count > 0 and contradict_count > 0:
                status = "contested"
                reason = "widerspruechliche Evidenz"
            elif needs_primary and not has_primary:
                status = "unverified"
                reason = "primaerbeleg fehlt"
            elif support_count >= 2 and (has_primary or has_mainstream or has_stakeholder):
                status = "verified"
                reason = "mehrfach belegt"
            elif support_count >= 1 and (has_primary or has_mainstream):
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
        """Select the bounded claim subset forwarded to later nodes.

        Verified and contested claims are always preferred. Unverified claims
        are only kept up to ``max_unverified`` and are biased toward groups
        that already have at least some quality-source backing.
        """
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
        """Compute claim-status counts and the aggregate claim quality score.

        The score follows the architecture contract:
        ``(verified + 0.5 * contested) / total``.
        """
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
        """Render consolidated claims into the answer-prompt block.

        Thin delegate to
        :func:`~inqtrix.strategies._claim_consolidation.default_claims_prompt_view`
        — kept here so that custom subclasses can override the rendering
        without re-implementing dedup/consolidation. Implements the
        contract from
        :meth:`ClaimConsolidationStrategy.claims_prompt_view`.

        Args:
            consolidated: Output of :meth:`consolidate`. Caller is
                responsible for trimming to the desired claim
                population (e.g. via :meth:`materialize`) before
                rendering.
            max_items: Maximum number of claims rendered into the
                returned string. Defaults to ``16``; the answer
                composer typically passes
                ``ReportProfileTuning.answer_claim_prompt_items``.

        Returns:
            Multi-line string suitable for inlining into the answer
            prompt's ``Claims:`` section. Empty string when
            ``consolidated`` is empty.
        """
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
        source_tiering: SourceTieringStrategy | None = None,
    ) -> list[str]:
        """Pick citation URLs for final answer synthesis.

        Ranking combines claim status (verified/contested/unverified) with
        source-tier weights so high-trust claims backed by mainstream or
        stakeholder domains can rank ahead of unverified claims that happen
        to cite a primary domain. Verified claims still dominate because
        ``status_weight`` is the primary multiplier; tier only re-orders
        within each status class. ``support_count`` adds a log-scaled bonus,
        ``contradict_count`` a bounded penalty.
        """
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

        def _tier_weight_for_url(url: str) -> float:
            if source_tiering is None:
                return _DEFAULT_TIER_WEIGHT
            tier = source_tiering.tier_for_url(url)
            return SOURCE_TIER_WEIGHTS.get(tier, _DEFAULT_TIER_WEIGHT)

        def _best_tier_weight(claim: dict[str, Any]) -> float:
            urls = [str(u) for u in (claim.get("source_urls") or []) if u]
            if source_tiering is not None and urls:
                return max(
                    (_tier_weight_for_url(u) for u in urls),
                    default=_DEFAULT_TIER_WEIGHT,
                )
            counts = claim.get("source_tier_counts") or {}
            best = 0.0
            for tier, count in counts.items():
                if int(count or 0) > 0:
                    w = SOURCE_TIER_WEIGHTS.get(str(tier), 0.0)
                    if w > best:
                        best = w
            return best or _DEFAULT_TIER_WEIGHT

        def _combined_score(claim: dict[str, Any]) -> float:
            status = str(claim.get("status", "unverified"))
            status_weight = _STATUS_WEIGHTS.get(status, _STATUS_WEIGHTS["unverified"])
            tier_weight = _best_tier_weight(claim)
            support = max(0, int(claim.get("support_count", 0) or 0))
            contradict = max(0, int(claim.get("contradict_count", 0) or 0))
            support_bonus = math.log1p(support) / _SUPPORT_BONUS_SCALE
            contradict_malus = min(0.3, contradict * 0.1)
            return (status_weight * tier_weight) + support_bonus - contradict_malus

        def _sorted_urls(claim: dict[str, Any]) -> list[str]:
            urls = [str(u) for u in (claim.get("source_urls") or []) if u]
            if source_tiering is None:
                return urls
            return sorted(urls, key=lambda u: -_tier_weight_for_url(u))

        seen_domains: set[str] = set()
        domain_counts: Counter[str] = Counter()

        def _add_with_domain(url: str) -> bool:
            n = normalize_url(str(url))
            if not n or n in seen:
                return False
            d = domain_from_url(n)
            if d and d in seen_domains:
                return False
            seen.add(n)
            if d:
                seen_domains.add(d)
                domain_counts[d] += 1
            selected.append(n)
            return True

        def _add_with_domain_cap(url: str) -> bool:
            n = normalize_url(str(url))
            if not n or n in seen:
                return False
            d = domain_from_url(n)
            if d and domain_counts[d] >= _PHASE2_MAX_PER_DOMAIN:
                return False
            seen.add(n)
            if d:
                domain_counts[d] += 1
            selected.append(n)
            return True

        if consolidated:
            ranked_claims = sorted(
                list(consolidated),
                key=lambda c: (
                    -_combined_score(c),
                    -int(c.get("support_count", 0) or 0),
                    int(c.get("contradict_count", 0) or 0),
                ),
            )

            # Phase 1 — Domain-Diversity: at most one URL per claim, and only
            # if the URL's domain has not been seen yet. Status-neutral; the
            # ranking via _combined_score already handles verified/unverified
            # priority (status_weight 1.0 / 0.7 / 0.35).
            for claim in ranked_claims:
                for u in _sorted_urls(claim):
                    if _add_with_domain(u):
                        if len(selected) >= max_items:
                            return selected[:max_items]
                        break

            # Phase 2 — Fill: top up with additional URLs from already-seen
            # domains, in claim-score order, but cap per-domain contribution
            # at _PHASE2_MAX_PER_DOMAIN to preserve diversity under heavy
            # concentration on a single authoritative source.
            for claim in ranked_claims:
                for u in _sorted_urls(claim):
                    _add_with_domain_cap(u)
                    if len(selected) >= max_items:
                        return selected[:max_items]

        # Phase 3 — Fallback: if the claim ledger was too sparse, supplement
        # from the raw all_citations pool. Honour the per-domain cap so
        # fallback cannot reintroduce dominance of a single domain.
        if len(selected) < min(15, max_items):
            for u in all_citations:
                _add_with_domain_cap(str(u))
                if len(selected) >= max_items:
                    break

        return selected[:max_items]
