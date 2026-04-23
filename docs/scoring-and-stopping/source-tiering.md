# Source tiering

> Files: `strategies/_source_tiering.py` (`DefaultSourceTiering`), `domains.py`

## Scope

How Inqtrix classifies URLs into quality tiers, how the aggregate source-quality score is computed, and how to extend or replace the default tiering for a different domain landscape.

## Tier classification

| Tier | Weight | Examples |
|------|--------|----------|
| **primary** | 1.0 | `bundesregierung.de`, `bundestag.de`, `ec.europa.eu`, `who.int`, `oecd.org` |
| **mainstream** | 0.8 | `spiegel.de`, `tagesschau.de`, `reuters.com`, `nature.com`, `arxiv.org` |
| **stakeholder** | 0.45 | `kzbv.de`, `vdek.com`, `spd.de`, `gruene.de` (parties, associations, trade unions) |
| **unknown** | 0.35 | Unrecognised domains |
| **low** | 0.1 | `pinterest.com`, `reddit.com`, `medium.com`, `gutefrage.net` |

The domain-to-tier mapping is held in [`src/inqtrix/domains.py`](../../src/inqtrix/domains.py). Lookups are case-insensitive and match suffixes only (so `www.spiegel.de/panorama/...` resolves to `spiegel.de`).

## Quality score

```
q_source = sum(weight[tier(url)] for url in citations) / len(citations)
```

Range: 0.1 (all low-tier) to 1.0 (all primary). An empty citation list yields `0.0` by convention and does **not** divide by zero.

The score is recomputed in:

- `search` after each round (informs ongoing context pruning),
- `evaluate` (feeds into [Stop criteria](stop-criteria.md)),
- `answer` (drives citation ranking via `select_answer_citations`).

## When a URL contributes

A URL contributes to the aggregate score each time it appears in `all_citations`. Duplicates are deduplicated earlier in `search` via `normalize_url()`, so the score reflects distinct sources.

## Extending the tiering

Two extension points exist:

### 1. Add domains to the default tier lists

Append to the lists in `domains.py` and keep the default `DefaultSourceTiering`. Domains are grouped by tier, ordered alphabetically by convention. No code change in nodes or strategies is needed.

### 2. Replace the strategy

Implement `SourceTieringStrategy` and pass it to `AgentConfig`:

```python
from inqtrix import AgentConfig, ResearchAgent, SourceTieringStrategy


class InternalWikiTiering(SourceTieringStrategy):
    def tier_for_url(self, url: str) -> str:
        if "internal-wiki.example.com" in url:
            return "primary"
        return "unknown"

    def quality_from_urls(self, urls: list[str]) -> tuple[dict[str, int], float]:
        counts = {"primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0}
        for url in urls:
            counts[self.tier_for_url(url)] += 1
        total = len(urls) or 1
        weights = {"primary": 1.0, "mainstream": 0.8, "stakeholder": 0.45, "unknown": 0.35, "low": 0.1}
        score = sum(weights[self.tier_for_url(u)] for u in urls) / total
        return counts, score


agent = ResearchAgent(AgentConfig(source_tiering=InternalWikiTiering()))
```

The node code only calls `tier_for_url` and `quality_from_urls`; you can change the tier list, weights, or counting logic freely as long as the return shape holds.

## Interaction with stop criteria

Source tiering is consumed by several guardrail caps in [Stop criteria](stop-criteria.md):

- **No-citation cap** — caps confidence at 6 if zero citations accumulated.
- **Low >> high cap** — caps at 7 if low-tier sources dominate over primary/mainstream.
- **Missing-primary cap** — caps at 8 if claims flagged `needs_primary` have no primary source.

## Related docs

- [Claims](claims.md)
- [Aspect coverage](aspect-coverage.md)
- [Stop criteria](stop-criteria.md)
- [Strategies](../architecture/strategies.md)
