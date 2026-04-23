"""Source tiering strategy — classify URLs into quality tiers.

The source-tiering strategy is the primary signal Inqtrix uses to weight
evidence quality. It assigns each citation URL a tier (``primary``,
``mainstream``, ``stakeholder``, ``unknown``, ``low``) and aggregates a
0.0–1.0 quality score over a batch of URLs. Implementations should be
pure functions of the URL — no I/O, no per-call state — so they can be
called freely inside hot loops (search-result post-processing, answer
synthesis, metric aggregation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from inqtrix.domains import (
    LOW_QUALITY_DOMAINS,
    MAINSTREAM_SOURCE_DOMAINS,
    PRIMARY_SOURCE_DOMAINS,
    SOURCE_TIER_WEIGHTS,
    STAKEHOLDER_SOURCE_DOMAINS,
)
from inqtrix.urls import domain_from_url, domain_matches


class SourceTieringStrategy(ABC):
    """Contract for classifying citation URLs into source-quality tiers.

    The strategy is consumed in three places: the search node tags each
    citation as it arrives, the evaluate node uses the aggregate
    quality score as a stop-cascade input, and the answer node uses
    the per-URL tier to order references. Implementations must be
    deterministic for a given URL and side-effect-free; the same URL
    must always classify to the same tier within one run.
    """

    @abstractmethod
    def tier_for_url(self, url: str) -> str:
        """Return the quality tier for a single URL.

        Args:
            url: Citation URL to classify. Implementations are
                expected to handle URLs that lack a scheme or that
                point at well-known aggregator hosts; malformed input
                must not raise — return ``"unknown"`` instead.

        Returns:
            One of ``"primary"`` (peer-reviewed / official), ``"mainstream"``
            (established media), ``"stakeholder"`` (industry / NGO /
            association), ``"unknown"`` (no classification), or
            ``"low"`` (known low-quality domains).
        """

    @abstractmethod
    def quality_from_urls(self, urls: list[str]) -> tuple[dict[str, int], float]:
        """Aggregate tier counts and a weighted quality score over URLs.

        Args:
            urls: Citation URLs to score. May be empty; implementations
                must return zero counts and ``0.0`` for an empty list.

        Returns:
            Tuple ``(tier_counts, quality_score)``.
            ``tier_counts`` is a dict that contains all five tier keys
            (zero for absent tiers) for stable downstream consumption.
            ``quality_score`` is in ``[0.0, 1.0]`` — typically the
            weighted mean over per-URL tier weights from
            :data:`~inqtrix.domains.SOURCE_TIER_WEIGHTS` (``primary=1.0``,
            ``mainstream=0.8``, ``stakeholder=0.45``, ``unknown=0.35``,
            ``low=0.1`` by default).
        """


class DefaultSourceTiering(SourceTieringStrategy):
    """Default domain-table-driven source-quality classifier.

    Maps the URL's normalised domain (lower-cased eTLD+1) onto the
    static allow-lists in :mod:`inqtrix.domains`. Match precedence is
    ``low`` > ``primary`` > ``mainstream`` > ``stakeholder`` > ``unknown``
    so that explicitly bad domains can never be re-promoted by a
    later table membership.

    No I/O, no caching layer, no per-instance state — instances are
    cheap to construct and safe to share across runs.
    """

    # -------------------------------------------------------------- #
    # tier_for_url
    # -------------------------------------------------------------- #
    def tier_for_url(self, url: str) -> str:
        """Classify ``url`` by matching its domain against configured allow-lists.

        Args:
            url: Citation URL to classify. Empty strings, whitespace,
                or URLs with an unrecognisable host all produce
                ``"unknown"`` rather than raising.

        Returns:
            The first matching tier in the precedence order ``low`` >
            ``primary`` > ``mainstream`` > ``stakeholder``, or
            ``"unknown"`` when no allow-list matches.
        """
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
        """Count tiers and compute the mean weighted source-quality score.

        Implements the heuristic described in
        :meth:`SourceTieringStrategy.quality_from_urls` using
        :data:`~inqtrix.domains.SOURCE_TIER_WEIGHTS` as per-tier
        weights and ``0.35`` as the fallback weight for tiers that
        somehow fall outside the table (defensive — the default
        weights table contains all five known tiers).

        Args:
            urls: Citation URLs to score. May be empty.

        Returns:
            Tuple ``(tier_counts, quality_score)``. ``tier_counts``
            always contains the five tier keys (zero for absent
            tiers). ``quality_score`` is rounded to three decimal
            places; ``0.0`` for an empty input list.
        """
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
