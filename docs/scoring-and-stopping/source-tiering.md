# Source tiering

> Files: `strategies/_source_tiering.py` (`DefaultSourceTiering`), `domains.py`

## Scope

How Inqtrix classifies URLs into discovery-quality tiers, how the aggregate source-quality score is computed, and how to extend or replace the default tiering for a different domain landscape. Tiering is an observability, ranking, and search-control signal after retrieval; normal searches are not pre-filtered by the tier table. A tier never verifies a claim, and an `unknown` tier never removes the provider-grounded search result from synthesis.

## Tier classification

| Tier | Weight | Domains in list | Examples |
|------|--------|---:|----------|
| **primary** | 1.0 | 199 | `bundesregierung.de`, `sec.gov`, `ec.europa.eu`, `ecb.europa.eu`, `bundesbank.de`, `bafin.de`, `bankofcanada.ca`, `frbsf.org`, `stlouisfed.org`, `who.int`, `oecd.org`, `cdc.gov`, `nih.gov`, `bls.gov`, `cbo.gov`, `gao.gov`, `nature.com`, `science.org`, `arxiv.org`, `nejm.org`, `thelancet.com`, `bmj.com`, `cell.com`, `pnas.org`, `plos.org`, `ieeexplore.ieee.org`, `dl.acm.org`, `sciencedirect.com`, `link.springer.com`, `onlinelibrary.wiley.com`, `stanford.edu`, `mit.edu`, `harvard.edu`, `ethz.ch`, `mpg.de`, `openai.com`, `anthropic.com`, `ai.meta.com`, ... |
| **mainstream** | 0.8 | 263 | DE: `tagesschau.de`, `spiegel.de`, `faz.net`, `sueddeutsche.de`, `zeit.de`, `welt.de`, `handelsblatt.com`, `manager-magazin.de`, `wiwo.de`, `heise.de`, `golem.de`, `t3n.de`; intl: `reuters.com`, `apnews.com`, `bloomberg.com`, `ft.com`, `wsj.com`, `nytimes.com`, `washingtonpost.com`, `theguardian.com`, `bbc.com`, `economist.com`, `nikkei.com`, `scmp.com`, `thehindu.com`, `lemonde.fr`, `elpais.com`, `nzz.ch`, `aljazeera.com`, ... |
| **stakeholder** | 0.45 | 108 | DE health: `kzbv.de`, `vdek.com`; parties: `cdu.de`, `spd.de`, `gruene.de`; think tanks: `brookings.edu`, `rand.org`, `cfr.org`, `chathamhouse.org`, `bruegel.org`, `ifo.de`, `diw.de`, `bertelsmann-stiftung.de`, `aei.org`, `bipartisanpolicy.org`, `adalovelaceinstitute.org`; consulting: `mckinsey.com`, `deloitte.com`, `kpmg.com`, `pwc.com`, `bcg.com`; banks/asset managers: `jpmorgan.com`, `goldmansachs.com`, `blackrock.com`, `vanguard.com`, `fidelity.com`; ratings: `spglobal.com`, `moodys.com`, `fitchratings.com`; foundations: `gatesfoundation.org`, `fordfoundation.org`; NGOs: `amnesty.org`, `hrw.org`, `transparency.org`, ... |
| **unknown** | 0.35 | — | Any domain not in the lists above. |
| **low** | 0.1 | 17 | `pinterest.com`, `reddit.com`, `medium.com`, `quora.com`, `tiktok.com`, `youtube.com` (community/UGC), plus a maintained list of confirmed AI-content farms. |

The domain-to-tier mapping is held in
[`src/inqtrix/domains.py`](../../src/inqtrix/domains.py). The four named
sets are:

- `PRIMARY_REGULATOR_DOMAINS` (90) -- governments, regulators, central
  banks, statistical offices, international organisations.
- `PRIMARY_OFFICIAL_COMPANY_DOMAINS` (28) -- official corporate IR / press
  / research channels.
- `PRIMARY_ACADEMIC_INSTITUTION_DOMAINS` (50) -- top US/UK/EU universities
  and major research institutions (MPG, Fraunhofer, Helmholtz, INRIA, …).
- `PRIMARY_ACADEMIC_PUBLISHER_DOMAINS` (28) -- top peer-reviewed journals
  and journal-database subdomains (NEJM, Lancet, JAMA, BMJ, Cell, PNAS,
  PLOS, plus `ieeexplore.ieee.org`, `dl.acm.org`, `sciencedirect.com`,
  `link.springer.com`, `onlinelibrary.wiley.com`, …).

These are unioned into `PRIMARY_SOURCE_DOMAINS` (199 total) together with a
small ad-hoc tail (`nature.com`, `science.org`, `arxiv.org`).
`MAINSTREAM_SOURCE_DOMAINS` (263) covers German, European, Asian, MENA,
African, and Latin-American major outlets plus tech / finance trade press.
`STAKEHOLDER_SOURCE_DOMAINS` (108) covers think tanks, consulting firms,
big banks, foundations, NGOs, and German health/party stakeholders.

Lookups are case-insensitive and match suffixes via
`urls.py::domain_matches()` -- `www.spiegel.de/panorama/...` matches
`spiegel.de`, and `ieeexplore.ieee.org/document/...` matches
`ieeexplore.ieee.org` but **not** the broader `ieee.org` (which is not in
PRIMARY -- this is intentional, so the journalistic
`spectrum.ieee.org` stays in MAINSTREAM rather than getting tiered up to
PRIMARY by an apex match).

`LOW_QUALITY_DOMAINS` still marks known weak domains as `low`, but it is
not sent as a default search blocklist. Explicit `site:` queries still
become provider domain allowlists.

## Quality score

```
q_source = sum(weight[tier(url)] for url in citations) / len(citations)
```

Range: 0.1 (all low-tier) to 1.0 (all primary). An empty citation list yields `0.0` by convention and does **not** divide by zero.

The score is computed in `search` after each round. `evaluate` reads the stored value for [Stop criteria](stop-criteria.md), and `answer` reads the same stored metrics for result reporting and citation context.

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
- **Missing-primary cap** — caps at 8 if a claim flagged `needs_primary` has no primary-tier provider-grounded support.

## Related docs

- [Claims](claims.md)
- [Aspect coverage](aspect-coverage.md)
- [Stop criteria](stop-criteria.md)
- [Strategies](../architecture/strategies.md)
- [Calculation overview](calculation-overview.md) -- every score and threshold in one place.
