# Evidence pipeline

## Scope

How Inqtrix carries provider-grounded search results into the final report
without losing the exact query, the coherent provider answer, or the links
returned with it. This page covers the Research graph's working evidence
ledger, the durable web-search ledger used by the Agent, the answer-prompt
projection, child-to-parent handoff, and post-answer audit binding. Inqtrix
does not fetch the linked webpages. This page does not describe provider
authentication or individual stop heuristics (see
[stop-criteria.md](../scoring-and-stopping/stop-criteria.md),
[confidence.md](../scoring-and-stopping/confidence.md), and
[calculation-overview.md](../scoring-and-stopping/calculation-overview.md)).

## Flow

This diagram answers: "How does a provider result become a cited answer
sentence?" The full provider result remains available through synthesis. A
domain tier may influence ordering and confidence, but it cannot discard an
`unknown` result or require a second page read.

```mermaid
flowchart TD
    Azure{{"provider Azure Web Search"}} --> QuerySynthesis[("data AgentState.query_synthesis<br/>one answer per query")]
    Azure --> SourceEvidence[("data source EvidenceRecord")]
    Perplexity{{"provider Perplexity search_results"}} --> QuerySynthesis
    Perplexity --> SourceEvidence
    Legacy{{"provider URL-only result"}} --> CitationInventory[("data citation inventory")]
    SourceEvidence --> EvidenceLedger[("data AgentState.evidence_ledger<br/>single evidence truth")]
    CitationInventory --> EvidenceLedger
    EvidenceLedger --> ClaimLedger["fn derive_claim_ledger_from_evidence()<br/>(local, not persisted)"]
    ClaimLedger --> ClaimIndex[("data consolidated_claims<br/>research-loop claim view")]
    ClaimIndex --> Verification["fn project_claim_verification_to_evidence()"]
    Verification --> EvidenceLedger
    QuerySynthesis --> WebLedger[("data web_search_ledger<br/>query, provider answer, citations")]
    CitationInventory --> WebLedger
    EvidenceLedger --> Overview["fn render_evidence_ledger_overview()<br/>single canonical answer-prompt view"]
    Overview --> AnswerPrompt{{"LLM call: answer prompt"}}
    AnswerPrompt --> AnswerAudit[("data AnswerEvidenceBinding<br/>audit view")]
    AnswerPrompt --> Parent["kernel or mission parent"]
    WebLedger --> Parent
    WebLedger --> Canvas["Evidence Canvas<br/>query, answer, mapping, links"]
```

Key transitions:

- Provider-specific result shapes are normalized into `EvidenceRecord` rows
  without filtering records merely because a domain is unknown.
- `EvidenceRecord.claims[]` are projected into a **local** raw claim ledger
  (`derive_claim_ledger_from_evidence`, recomputed in `search()`, not a state
  field) so the claim consolidator can operate.
- Verification status is projected **back onto the EvidenceLedger** by
  `project_claim_verification_to_evidence`, so each record carries its own
  `verification_status` / `verification_basis` -- this is what makes the ledger
  self-describing and removes the need for a separate bundle list.
- The answer prompt sees exactly one view: the Markdown overview produced by
  `render_evidence_ledger_overview` directly from the EvidenceLedger. It is
  record-driven (eligible records become labelled source blocks under their
  search query so the rich query synthesis shows once), and the citation
  allowlist is the union of visible source-block URLs. There
  is no parallel report-evidence-bundle / prompt-evidence-unit / rendered-
  context channel and no separate numbered prompt-citation map.
- The durable `web_search_ledger` retains the exact query, provider identity,
  coherent provider answer, every returned citation, timing, usage, status,
  and non-fatal notice. It is audit state, not a second search pipeline.
- A child research report reaches its parent together with this ledger. The
  parent may evaluate, consolidate, or ask another question, but it does not
  require a linked webpage to be fetched before it may use the provider result.

The graph shows that provider output is not forced into one shape. Azure's
synthesized web-search answer is stored once in `query_synthesis`; its cited
URLs still become source records so the final report has concrete anchors.
Per-source snippets remain associated with their URLs. URL-only providers
contribute citation inventory. In every case the user can open the returned
link, while the Canvas separately shows what Azure actually returned to
Inqtrix.

One coherent provider answer can synthesize several links. Therefore every
ledger citation carries an honest mapping status:

- `provider_answer_context`: the provider associated supporting answer context
  with this citation;
- `provider_snippet`: the provider returned URL-specific snippet metadata;
- `source_only`: the provider listed the URL but supplied no exclusive passage.

The Canvas shows the complete coherent answer for the query and never claims
that a sentence came from exactly one URL unless the provider supplied that
mapping. Credential-bearing URLs are redacted before provider output,
metadata, prompts, checkpoints, or UI state can observe them. Provider prose
is kept under a visible persistence bound; truncation includes an explicit
marker rather than silently masquerading as a complete result.

There is no LLM-backed grouping step in `search()` and no LLM step in the
answer-prompt view: claim consolidation and the evidence overview are
deterministic projections over the EvidenceLedger. The only LLM call after the
research loop is the per-section answer composer (see
[answer-composition.md](answer-composition.md) for the section-by-section flow,
prompt templates, and write-last summary mechanism).

Claim extraction is the one structured helper call inside `search()`. The
default `LLMClaimExtractor` first asks the active `LLMProvider` whether the
claim-extraction model supports native JSON-schema output. Built-in Bedrock,
Anthropic, and Azure OpenAI adapters answer this through
`supports_structured_output(model=...)` and use their provider-native schema
mechanism (`outputConfig.textFormat`, `output_config.format`, or
`response_format`). Providers that do not opt in keep the legacy prompt-based
JSON path. Parse failures, schema-shape failures, API failures, and token-limit
stops all remain visible as `ALGO-FAIL claim_extraction`; a valid
`{"claims": []}` response is logged separately as a valid-empty extraction.

## Primary truth vs views

`evidence_ledger` is the Research graph's canonical **working** truth. It joins
the query, source, citation, provider snippets/passages, raw claims, and --
after `project_claim_verification_to_evidence` -- each claim's verification
standing. The Agent's `web_search_ledger` is the durable projection used
across parent/child and Canvas boundaries. It is derived from the provider
result and does not introduce another search, ranker, filter, or prompt
renderer.

Every ledger search retains the stable invocation/query id, exact query,
provider and bounded parameters, start/end timestamps, provider-reported token
usage, status, non-fatal notice, coherent provider answer, and citation list.
This protected run audit distinguishes a bad search result from later
extraction or synthesis errors without copying private queries into ordinary
application logs.

The other evidence-related fields are kept deliberately minimal:

- `consolidated_claims`: the one persisted claims view, used by the research
  loop only (`plan` picks cross-check targets, `evaluate` runs stop heuristics
  and the evidence-depth-gap check). Recomputed each round in `search()` from
  a local `claim_ledger = derive_claim_ledger_from_evidence(evidence_ledger)`.
- `all_citations`: flat URL inventory for source-tier scoring; not an
  answer-prompt citation source.
- `evidence_depth_gap`: diagnostic over `consolidated_claims` that flags
  shallow single-source evidence for the next planning round. See
  [stop-criteria.md](../scoring-and-stopping/stop-criteria.md).
- `score_ledger`: chronological diagnostic snapshots built from the
  EvidenceLedger, consolidated claims, evaluator output, and answer audit.
  See [score-ledger.md](../scoring-and-stopping/score-ledger.md).

There is no `context`, `claim_ledger`, `report_evidence_bundles`,
`selected_report_evidence`, `unverified_evidence_notes`, or
`prompt_evidence_units` state field anymore. Claimless sources are **not**
degraded: a report-eligible `EvidenceRecord` with `claims=[]` still carries
`source_snippet`, `source_passages`, and `citation_set`, while
the query-level synthesis lives once in `state["query_synthesis"]`; it is
rendered as a `source-context` block in the
overview -- usable for attributed, source-backed statements, just not as a
cross-checked hard fact.

Terminology used below:

| Term | Role | Stored where | Read by |
|---|---|---|---|
| **Working truth** | Complete graph-local representation of search-derived evidence and projected verification | `AgentState.evidence_ledger` | claim projection, evidence overview, answer audit |
| **Durable web-search contract** | Exact queries, coherent provider answers, citation metadata, mapping precision, timing, usage, and status | `web_search_ledger` inside the existing evidence artifact | audit, resume, kernel/mission handoff, Canvas |
| **Parent handoff** | Complete child report plus its translated references and web-search ledger | child tool result and evidence artifact | kernel or mission parent |
| **Claims view** | The one persisted claims projection used by the research loop | `consolidated_claims` | `plan`, `evaluate`, stop diagnostics, observability |
| **Answer-prompt view** | The single record-driven Markdown overview the answer composer consumes | rendered on demand by `render_evidence_ledger_overview()` (not persisted) | `answer()`, section composer |
| **Audit view** | Post-answer check that cited URLs resolve to EvidenceRecords | `answer_evidence_bindings` | metrics, debugging |

## EvidenceRecord

One evidence row. `record_type` is currently always `source`; query-level
provider synthesis is not copied onto each record, but stored once in
`state["query_synthesis"][query_id]`.

```python
EvidenceRecord = {
    "evidence_id": "ev_9f3c4f2a1b22d0",
    "record_type": "source",
    "report_eligible": True,
    "query_id": "qry_3546f9a7bd7465",
    "query": "Meta 2026 AI capital expenditures official filing",
    "source_id": "src_704e45c2c8b896",
    "citation_id": "cit_d5ebe5322bd948",
    "canonical_url": "https://investor.atmeta.com/investor-news/...",
    "domain": "investor.atmeta.com",
    "tier": "primary",
    "tier_reason": "matched_official_company_domain",
    "provider": "PerplexitySearch",
    "source_title": "Meta Reports First Quarter 2026 Results",
    "source_snippet": "Meta reports revenue and 2026 capital expenditure guidance...",
    "source_date": "2026-04-30",
    "last_updated": "2026-05-01",
    "source_passages": [
        {
            "passage_id": "passage_ev_9f3c4f2a1b22d0_1",
            "origin": "source_snippet",
            "text": "Meta expects full-year 2026 capital expenditures of 115-135 USD billion.",
            "char_count": 86,
        }
    ],
    "claims": [
        {
            "raw_claim_id": "raw_claim_5f587376c11924",
            "claim_id": "claim_62ff008895f616",
            "claim_text": "Meta plans 2026 AI capital expenditures of 115-135 USD billion.",
            "claim_type": "fact",
            "polarity": "affirmed",
            "needs_primary": True,
            "evidence_snippet": "Meta expects full-year 2026 capital expenditures...",
            "verification_status": "verified",
            "verification_basis": "verified_primary",
            "supporting_evidence_ids": ["ev_9f3c4f2a1b22d0", "ev_a4417c19cf8842"],
            "supporting_domain_count": 2,
        }
    ],
}
```

The ledger stores source records plus extracted source passages and claims. For
Azure, the full synthesized answer lives in `query_synthesis.provider_answer`;
Azure citation URLs become report-eligible source records without duplicating
the same answer text into every URL row.

## RawClaim

`RawClaim` rows are claim-support rows derived from `EvidenceRecord.claims[]`.
They are a compatibility view for the existing consolidation strategy, not a
separate truth source. One raw claim may point back to one or more evidence
records.

```python
RawClaim = {
    "raw_claim_id": "raw_claim_5f587376c11924",
    "claim_text": "Meta plans 2026 AI capital expenditures of 115-135 USD billion.",
    "evidence_snippet": "Meta expects full-year 2026 capital expenditures...",
    "claim_type": "fact",
    "polarity": "affirmed",
    "needs_primary": True,
    "source_urls": ["https://investor.atmeta.com/investor-news/..."],
    "source_ids": ["src_704e45c2c8b896"],
    "citation_ids": ["cit_d5ebe5322bd948"],
    "evidence_ids": ["ev_9f3c4f2a1b22d0"],
    "published_date": "2026-04-30",
    "signature": "meta plans 2026 ai capital expenditures",
    "round": 1,
    "query": "Meta 2026 AI capital expenditures official filing",
    "query_id": "qry_3546f9a7bd7465",
}
```

`search()` derives this view after assembling evidence records. If `evidence_ledger` has
no claims, the local raw claim list stays empty and evaluation/answer must
treat hard facts conservatively. The raw view is never persisted on
`AgentState` -- only `consolidated_claims` is.

## ConsolidatedClaim

`ConsolidatedClaim` is the verification view produced by
`ClaimConsolidationStrategy.consolidate()` (default implementation in
`src/inqtrix/strategies/_claim_consolidation.py`). It groups raw claims by
signature and decides whether the claim is `verified`, `contested`, or
`unverified`, along with a more specific `verification_basis`. See
[claims.md](../scoring-and-stopping/claims.md) for the full branch rules; the
possible bases are `verified_cross_checked`, `verified_primary`,
`verified_quality_source`, `contested`, `missing_primary_source`, and
`weak_evidence`. Static source tiers remain ranking and confidence metadata.
They do not filter provider-grounded content out of the answer path.

```python
ConsolidatedClaim = {
    "claim_id": "claim_62ff008895f616",
    "signature": "meta plans 2026 ai capital expenditures",
    "claim_text": "Meta plans 2026 AI capital expenditures of 115-135 USD billion.",
    "claim_type": "fact",
    "needs_primary": True,
    "support_count": 2,
    "supporting_evidence_ids": ["ev_9f3c4f2a1b22d0", "ev_a4417c19cf8842"],
    "contradicting_evidence_ids": [],
    "status": "verified",
    "status_reason": "mehrfach belegt",
    "verification_basis": "verified_cross_checked",
    "source_urls": ["https://investor.atmeta.com/...", "https://www.bloomberg.com/..."],
    "citation_set": [{"label": "E1", "url": "https://investor.atmeta.com/..."}],
    "member_claim_ids": ["raw_claim_5f587376c11924", "raw_claim_a8..."],
    "member_claim_texts": ["Meta plans 2026 AI capital expenditures ..."],
    "round_first_seen": 1,
    "round_last_updated": 1,
    "source_tier_counts": {"primary": 1, "mainstream": 1, "unknown": 0},
}
```

After consolidation, `project_claim_verification_to_evidence()` writes the
`status` / `verification_basis` / `supporting_evidence_ids` /
`supporting_domain_count` fields back onto every matching
`EvidenceRecord.claims[n]`, so the EvidenceLedger is self-describing for the
overview renderer.

## Rendering: EvidenceLedger → Markdown

The answer composer never reads the EvidenceLedger directly; it reads exactly
one rendered Markdown block produced by
`render_evidence_ledger_overview()` in `src/inqtrix/evidence.py`. This is the
**single canonical answer-prompt view** -- there is no parallel bundle list,
no prompt-evidence-unit list, no rendered-context channel.

### Signature

```python
def render_evidence_ledger_overview(
    evidence_ledger: list[dict[str, Any]],
    *,
    max_total_chars: int,
    max_record_chars: int,
    label_by_evidence_id: dict[str, str] | None = None,
    query_synthesis: dict[str, dict[str, Any]] | None = None,
) -> EvidenceOverview:
```

### EvidenceOverview dataclass

```python
@dataclass(slots=True)
class EvidenceOverview:
    markdown: str                           # the Markdown block handed to the LLM
    label_urls: dict[str, str]              # visible E1 -> canonical URL
    allowed_urls: list[str]                 # URLs of visible source blocks
    label_by_evidence_id: dict[str, str]    # EvidenceRecord-id -> URL-canonical label
    rendered_record_count: int              # records that made it into markdown
    omitted_record_count: int               # records dropped purely because of budget
    rendered_evidence_ids: list[str]        # visible EvidenceRecord ids
```

### Filter

Only records with `report_eligible=True` AND a non-empty primary URL
(`canonical_url` or first citation URL) are eligible for rendering. Everything
else is skipped silently (it is not budget loss, it is "not report material").

### Order

Eligible records are first **labeled by canonical URL** in trust-then-density
order, *before* the budget cut is applied, so the same source URL has one
`E1`/`E2` label even when several EvidenceRecords reference it. The label order
comes from `_evidence_record_score()`:

```python
score  = _SOURCE_CONTEXT_TIER_RANK[tier]            # tier weight
score += _VERIFICATION_RANK[verification_label]     # how strongly the record's claims are verified
score += min(30, len(claims) * 5)                   # up to 6 claims
score += min(12, len(source_passages) * 2)          # up to 6 passages
score += min(4, len(source_snippet) // 120)
if source_date not in ("", "unknown"): score += 3
```

with the two ranking tables exactly:

```python
_SOURCE_CONTEXT_TIER_RANK = {
    "primary":     60,
    "mainstream":  50,
    "stakeholder": 40,
    "unknown":     25,
    "low":        -100,
}
_VERIFICATION_RANK = {
    "cross-checked":          50,
    "primary-source":         42,
    "contested":              30,
    "single-source verified": 24,
    "source-context":         12,
    "unverified":              8,
}
```

After per-record labels are assigned, the renderer groups records by their
`query_id` (the search query that produced them), ranks groups by the score of
their best record, and renders top-to-bottom under "RECHERCHE-ERGEBNIS R1",
"RECHERCHE-ERGEBNIS R2", … blocks. Each group may show the provider synthesis
once; individual records contribute their claims, snippets, passages, and data
points without repeating query-level context.

### Budget

Two budgets apply:

| Budget | DEEP | COMPACT | Effect |
|---|---|---|---|
| `max_total_chars` (`prompt_evidence_total_char_budget`) | 180 000 | 30 000 | Hard cap on the entire rendered Markdown. Once the running length would exceed it, the next record (and remaining records in its group) are counted in `omitted_record_count`. |
| `max_record_chars` (`prompt_evidence_record_char_limit`) | 2 600 | 2 200 | Per-source-block budget. When a block exceeds it, evidence lines are trimmed with a visible `[...weitere Belege dieser Quelle wegen Budget gekuerzt]` marker; header / label / metadata are always kept. |

When records are dropped because of the total budget, the renderer appends a
visible footer line:

```
HINWEIS: 12 weitere belegfaehige Quellen passten nicht in das Evidenz-Budget
und sind in dieser Uebersicht nicht enthalten.
```

so budget loss is **never silent** -- the answer-LLM sees that some report-
eligible sources were left out.

The query-level fields fed into each `RECHERCHE-ERGEBNIS` header come from
`state["query_synthesis"]` and are included once per query group. They are
governed by the same visible overview budget as the source blocks.

### Stable labels across section prompts

The answer composer renders the full EvidenceOverview once, stores the
`label_by_evidence_id` map on state, and reuses it for every per-section prompt.
That means every EvidenceRecord for the same Reuters URL resolves to the same
`E12` label. Section hints are limited to labels whose source blocks were
visible in the full overview.

### Example output (synthetic, ~25 lines)

```text
RECHERCHE-ERGEBNIS R1
Suchanfrage: Meta 2026 AI capital expenditures official filing
Provider-Synthese (Kontext; nicht eigenstaendig verifiziert):
Meta gave FY2026 capex guidance of 115-135 USD billion (Q1 2026 release), with
both the official IR page and Bloomberg corroborating the range. Analysts
flagged elevated AI-infrastructure spend; no source contradicted the range.

Quellen aus dieser Recherche:
[E1] Meta Reports First Quarter 2026 Results
  Datum: 2026-04-30 | Einstufung: primary | Beleglage: cross-checked
  Aussagen dieser Quelle:
  - Meta plans 2026 AI capital expenditures of 115-135 USD billion.
  Belegausschnitte:
  - Meta expects full-year 2026 capital expenditures of 115-135 USD billion.

[E2] Meta lifts 2026 capex range on AI buildout
  Datum: 2026-04-30 | Einstufung: mainstream | Beleglage: cross-checked
  Aussagen dieser Quelle:
  - Meta plans 2026 AI capital expenditures of 115-135 USD billion.
  Belegausschnitte:
  - Meta raised its 2026 capex range to $115B-$135B, citing AI infrastructure ...
```

## Verification labels per record

`_record_verification_label()` (in `src/inqtrix/evidence.py`) computes the
verification label shown in each rendered source block ("Beleglage: …"). It
reads the verification fields projected onto every `record.claims[n]` by
`project_claim_verification_to_evidence()` and picks the strongest across all
claims of that record. Possible values:

| Label | Trigger (across all claims of the record) | Allowed report use |
|---|---|---|
| `cross-checked` | At least one claim has `verification_basis == "verified_cross_checked"` | Hard fact, cite with the label. |
| `primary-source` | At least one claim is supported by a provider-cited primary-tier source (`verification_basis == "verified_primary"`) | Hard fact within that scope; cite with the label. |
| `contested` | Some claim is `contested` (in status or basis) | Show both sides, attribute. |
| `single-source verified` | At least one `verified` claim with no `verified_primary` / `verified_cross_checked` (one provider-grounded source) | Allowed, but must be inline-attributed ("laut [E12] …"). |
| `source-context` | The record has no claims at all (claimless background) | Allowed for context / source-backed statements; not a cross-checked hard fact. |
| `unverified` | All claims have `status="unverified"` | Discuss as uncertainty, never as a confirmed fact. |

The labels feed the `_VERIFICATION_RANK` bonus in the record-score formula
above, so cross-checked records rise to the top of the overview within their
tier band.

## Final answer system prompt

The system prompt of every per-section answer LLM call is built by
`_build_answer_system_prompt_with_style()` in `src/inqtrix/prompts.py` (lines
~405-702). It is **identical across all section calls** of a single answer --
only the section-specific style block (heading / instruction / position /
length guidance, see [answer-composition.md](answer-composition.md)) changes.
The block order below reflects the actual emission order in code.

### 1. Header (always emitted)

Role, language directive, plus six guardrail blocks emitted unconditionally:

- **SICHERHEIT / PROMPT-INJECTION**: search content is treated as untrusted;
  embedded instructions are ignored.
- **SELBST-VERIFIKATION**: each claim must be checked against its source;
  unconfirmed claims must be marked.
- **PRAEZISION**: precise rules for legal / regulatory citations.
- **ZEITLICHE PRAEZISION**: temporal language for ongoing processes.
- **FORMATIERUNGS-REGELN**: Markdown structure rules.

### 2. CLAIM-KALIBRIERUNG (conditional)

Triggers when `source_counts` or `claim_counts` are non-empty. Shows the
source-tier mix, the quality score, and the verified / contested / unverified
breakdown, plus the primary-source obligation count. Lets the LLM calibrate
the language strength.

### 3. ABDECKUNGSREGEL (conditional)

Triggers when `required_aspects` is non-empty. Lists required and uncovered
aspects; instructs to mark uncovered aspects inline as `(unbestaetigt)`.

### 4. EVIDENZTIEFE (conditional on `evidence_depth_gap.active == True`)

Emitted when the depth-gap diagnostic in
[stop-criteria.md](../scoring-and-stopping/stop-criteria.md) is active --
i.e., 3+ verified claims but most ride on a single source, or a central
numeric claim only has one quality source backing it. Tells the LLM:

```text
EVIDENZTIEFE -- WICHTIG FUER DEN TON DES REPORTS:
- Von {N} verifizierten Aussagen sind nur {k} cross-checked; {m} ({p}%) ruhen
  auf einer einzigen Quelle.
- Behandle single-source verified Aussagen mit inline-Attribution ("laut [E12]
  …") und nicht als gesichert.
- Erwaehne im Bericht explizit, dass die Belegtiefe begrenzt ist.
```

### 5. TRANSPARENZPFLICHT (conditional on `evidence_is_weak`)

Triggers when **any** of:
- `unverified_count > verified_count`, or
- `claim_needs_primary_verified < claim_needs_primary_total` (primary
  obligations unmet), or
- `evidence_depth_gap.active == True`.

In section mode the block also checks if the current section heading allows a
transparency sub-section; if so, it instructs the LLM to add a
"Unsicherheiten / Offene Punkte" sub-section.

### 6. EVIDENZ-UEBERSICHT (always)

The single canonical evidence view. The LLM is told the structure (query
groups, per-source labels, verification labels, citation rule) and then
receives the embedded `evidence_overview.markdown` produced by
`render_evidence_ledger_overview()`.

### 7. ZITATIONS-REGELN (always)

Quoted from `prompts.py`:

```text
ZITATIONS-REGELN:
- Verwende inline Markdown-Links der Form [E12](URL). Die Labels und URLs
  stammen ausschliesslich aus der Evidenz-Uebersicht oben.
- Erfinde keine Labels und keine URLs. Wenn keine Quelle passt, nenne sie
  nicht.
- Jeder substanzielle Satz braucht mindestens eine Citation.
- Keine separate "Quellen"-Liste am Ende -- die wird systemweit generiert.
```

### Embedded evidence block

After the rules, the system prompt embeds `evidence_overview.markdown` (the
output of the renderer above) verbatim. Because the same system prompt is
sent to every section LLM call, every section sees the **full** evidence
overview, not a section-scoped subset (`section_focus_labels` in the user
prompt is only a soft hint -- see
[answer-composition.md](answer-composition.md)).

## AnswerEvidenceBinding

After answer generation, the audit checks whether the cited URLs match
EvidenceRecords.

```python
AnswerEvidenceBinding = {
    "binding_id": "bind_ev_2aa4f188ac31e0",
    "citation_url": "https://investor.atmeta.com/...",
    "evidence_id": "ev_9f3c4f2a1b22d0",
    "verification": "primary-source",
    "binding_status": "matched",
}
```

This URL-level audit resolves every cited link to an EvidenceRecord:

- `matched`: the URL resolves to a record carrying a verified or contested
  claim.
- `source_context`: the URL resolves to a claimless or unverified record.
- `unknown_citation`: the URL resolves to no record at all.

The `evidence_contract_status` (`clean` / `needs_review` /
`source_context_only` / `algorithm_failed` / `unknown`) is decided by the
**claim-level** binding (`_build_answer_claim_bindings`: does a cited sentence
plausibly carry a consolidated claim?), not by this coarser URL audit. The URL
audit contributes the `unknown_citation` signal that forces `needs_review`. See
[score-ledger.md](../scoring-and-stopping/score-ledger.md).

## Mini example: one fact through the pipeline

1. `SearchProvider.search()` returns a Bloomberg + Meta-IR result for the
   query "Meta 2026 AI capital expenditures official filing" and records the
   exact query, invocation id, timing, usage, provider parameters, provider
   synthesis, snippets, and URLs.
2. `search()` stores the provider answer once in `query_synthesis[query_id]`
   and normalizes both cited URLs into `record_type="source"` records. The full
   Azure answer and all returned links also enter `web_search_ledger`.
3. `derive_claim_ledger_from_evidence()` projects the nested
   `EvidenceRecord.claims[]` into a flat list of `RawClaim`s (not persisted on
   state).
4. `DefaultClaimConsolidator.consolidate()` groups the two claims by
   signature "meta plans 2026 ai capital expenditures", finds 2 affirmed
   supports from 2 different non-low domains (`primary` + `mainstream`), sets
   `status="verified"` and `verification_basis="verified_cross_checked"`.
5. `project_claim_verification_to_evidence()` writes that
   `verification_status` / `verification_basis` /
   `supporting_evidence_ids=["ev_9f3c4f2a1b22d0", "ev_a4417c19cf8842"]` back
   onto every matching `record.claims[n]`.
6. The child passes its complete report and `web_search_ledger` to the parent.
   No URL is removed because its tier is `unknown`, and no page-read contract
   is required.
7. `render_evidence_ledger_overview(...)` ranks records by
   `_evidence_record_score()`, assigns one label per canonical URL (`E1` for
   Meta IR, `E2` for Bloomberg), groups them under their query, renders the
   Markdown block, and returns an `EvidenceOverview` with
   `rendered_record_count=2`, `omitted_record_count=0`,
   `allowed_urls=["https://investor.atmeta.com/...", "https://www.bloomberg.com/..."]`.
8. `_build_answer_system_prompt_with_style()` assembles the final system
   prompt (Header → CLAIM-KALIBRIERUNG → EVIDENZ-UEBERSICHT → ZITATIONS-REGELN
   → embedded `evidence_overview.markdown`); the answer composer runs one LLM
   call per section with this system prompt and a section-specific user
   prompt (see [answer-composition.md](answer-composition.md)).
9. `audit_answer_evidence_bindings()` records that the generated answer cited
   both `https://investor.atmeta.com/...` and
   `https://www.bloomberg.com/...` near the capex sentence → binding status
   `matched`.

The important maintenance rule: graph-local evidence fields originate in
`EvidenceRecord`; durable web-search fields originate in the provider result
and are deterministically projected into `web_search_ledger`. Neither
consumer may silently invent or drop a claim, query, provider answer, or link
to make the two representations agree.

## Related docs

- [Answer composition](answer-composition.md) -- per-section LLM call flow, prompt templates, write-last summary.
- [Knowledge retrieval](knowledge-retrieval.md) -- the parallel evidence and grounding flow for `mode=knowledge` (`[K#]` instead of `[E#]`).
- [Nodes](nodes.md) -- the five LangGraph nodes that own the pipeline.
- [State and iteration](state-and-iteration.md) -- AgentState fields and per-node read/write.
- [Score ledger](../scoring-and-stopping/score-ledger.md) -- diagnostic snapshots over this pipeline.
- [Calculation overview](../scoring-and-stopping/calculation-overview.md) -- every score and metric, LLM-vs-deterministic, with formulas.
- [Worked example](../reference/worked-example.md) -- one concrete run end-to-end.
