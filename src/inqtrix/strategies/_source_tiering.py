"""Source tiering strategy — classify URLs into quality tiers."""

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
