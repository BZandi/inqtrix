# Glossary

Domain-specific terms that appear throughout the codebase and the docs. Terms are listed alphabetically. Cross-references link to the page where the concept is explained in depth.

## A

**AgentConfig** — Pydantic `BaseModel` that holds everything `ResearchAgent` needs at construction time: providers, strategies, timeouts, cache settings, report profile, and per-call tunings. Server-only deployment settings live on `ServerSettings` instead. See [Agent config](../configuration/agent-config.md).

**AgentState** — Internal `TypedDict` with 60+ fields that LangGraph threads through all five nodes. It holds runtime input, planning state, evidence ledgers, claim views, stop signals, answer audit bindings, and diagnostics. Runtime-only keys may be added to the underlying dict even when they are not declared on the `TypedDict`. See [State and iteration](../architecture/state-and-iteration.md).

**Aspect coverage** — Deterministic token-match score (0.0–1.0) of how many required aspects are demonstrably present in the accumulated context. Caps confidence at 8 while aspects remain uncovered. See [Aspect coverage](../scoring-and-stopping/aspect-coverage.md).

## B

**Baukasten** — German for "construction kit". Internal shorthand for the pluggable-provider, pluggable-strategy design: a caller assembles providers and strategies as independent building blocks instead of configuring a monolith. Reflected in the Constructor-First principle.

## C

**Chunk** — One retrieval unit of an ingested document (`kch_` id). Carries the immutable `source_text`, an optional generated `retrieval_context`, and the exact UTF-8 span that locates it in the canonical document. The chunker splits on paragraph boundaries up to ~2000 characters and uses no overlap. See [Knowledge engine](../knowledge/overview.md#data-model).

**Claim (raw)** — Structured atomic statement extracted from a search result and attached to its `EvidenceRecord.claims[]` row (the primary truth in `state["evidence_ledger"]`). The raw claim list is derived per round by `derive_claim_ledger_from_evidence()` as a local variable in `search()`, **not** persisted on `AgentState`. After consolidation, the resulting `consolidated_claims` is the one persisted claim view. See [Claims](../scoring-and-stopping/claims.md).

**Claim status** — One of `verified`, `contested`, or `unverified`. Determined from provider-grounded supports, source tiers, contradictions, and the `needs_primary` flag. An unknown tier never removes evidence. See [Claims](../scoring-and-stopping/claims.md).

**Competing events** — Multiple conflicting explanations of the same event in the collected context (e.g. two different dates for the same policy vote). Detected by the evaluate LLM; can cap confidence to `confidence_stop - 1` when confidence is already at or above the stop threshold and events are newly detected. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

**Confidence stop** — Threshold (default 8) at which the loop terminates once `final_confidence` reaches it. See [Confidence](../scoring-and-stopping/confidence.md).

**Constructor first** — Design principle: providers never read environment variables directly. All values are passed as constructor arguments. `.env` translation happens only in example scripts and the `Settings` bridge layer.

**Contextual retrieval / `retrieval_context`** — Optional ingestion stage (`INQTRIX_KNOWLEDGE_CONTEXTUALIZE`, default `off`) in which a fast-tier LLM writes one or two situating sentences per chunk. The prefix is indexed together with the chunk body but stored separately, and it never reaches an answer, a citation, or the quote check — generated text may help *find* a chunk, never serve as evidence. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#what-contextualization-changes-and-what-it-must-never-change).

## D

**Deadline** — Monotonic wall-clock timestamp computed at run start; every node entry and every provider call respects it. See [Timeouts and errors](../observability/timeouts-and-errors.md).

**Dense embedding** — A learned fixed-size vector representation of a text, produced by the collection's immutable embedding model. Nearby vectors mean related meaning, which is what makes paraphrase and cross-language matches work — and why exact identifiers can be lost in the compression. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#why-two-searches-instead-of-one).

## E

**Effective model** — The model name a specific role (classify, claim extraction, evaluate) resolves to, after fallback to `reasoning_model`. Exposed via `LLMProvider.models.effective_*_model` and `resolve_claim_extract_model(llm, fallback)` (constructor-first).

**Effort / Thinking** — Opt-in reasoning budget exposed by Anthropic (`thinking`) and Azure/OpenAI (`effort`). Not all models accept all values; Inqtrix surfaces rejection warnings via `consume_effort_config_warnings`.

**Evidence consistency / sufficiency** — Two numeric signals the evaluate LLM is asked to produce alongside the confidence score. If parsing fails, both default to 5 and set parse flags to false; a dedicated sanity cap applies only when both parsed values are exactly 0 while confidence is already high. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

**EvidenceRecord** — One row in `state["evidence_ledger"]` (`AgentState` key, `list[dict[str, Any]]`). It joins query, source, citation, bounded passages, data points, and raw claim supports. This is the primary internal truth for search-derived evidence. See [Evidence pipeline](../architecture/evidence-pipeline.md).

## F

**Falsification mode** — One-shot switch that adds debunk-style and nearest-explanation slots to the plan node after two low-confidence rounds. See [Falsification](../scoring-and-stopping/falsification.md).

**`final_k`** — How many evidence chunks reach the answer prompt as `[K#]` entries. Derived as `min(top_k × profile.final_k_factor, EVIDENCE_K_MAX)` unless `knowledge_filters.final_k` pins it; only `tief` raises the factor above 1.0. It can legitimately exceed `top_k`, because decomposition, reranking, and gate rounds all widen the candidate pool. See [Retrieval profiles](../configuration/knowledge-profiles.md).

## G

**Generation** — One immutable collection-wide index build (`gen_` id) with its revision manifest. Exactly one generation is active; a rebuild writes a shadow generation and publication switches a single Postgres pointer, leaving the predecessor rollbackable. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#storage-topology-postgres-canonical-qdrant-derived).

**Grounding** — The fail-closed quote check on a `mode=knowledge` answer. The model must emit a `ZITATE:` block of labelled verbatim quotes before `ANTWORT:`, and each quote is verified by **deterministic code** — a formatting-normalized substring match against the labelled chunk's `source_text` — with no second LLM call. An unverifiable quote allows at most one visible regeneration; a malformed response terminates immediately. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#retrieval-pipeline-question-to-cited-answer).

## H

**Hybrid retrieval** — Running the dense and BM25 branches over the same collection scope and fusing their rankings with RRF. The two branches fail in opposite directions, so the fusion covers each one's blind spot; requires `INQTRIX_VECTOR_BACKEND=qdrant`. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#why-two-searches-instead-of-one).

## I

**Iteration log** — Structured per-round record stored in `state["iteration_logs"]` (`AgentState` key, `list[dict[str, Any]]`) in testing mode and exported by test/parity entry points as top-level `iteration_logs`. It is not part of `ResearchResult`. See [Iteration log](../observability/iteration-log.md).

## L

**LLMProvider** — ABC for language-model backends. Required methods are `complete` and `is_available`. Optional `complete_with_metadata` and `complete_structured` add token counts and schema validation. See [Providers overview](../providers/overview.md).

## P

**Plateau stop** — Stop heuristic that triggers when confidence has been stable at ≥ 6 across two or more rounds and competing events are unchanged. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

**Primary-need / `needs_primary`** — Evidence-depth flag on a factual claim that drives primary-source and corroboration checks. It does not discard the provider answer. See [Claims](../scoring-and-stopping/claims.md).

**Progress event** — Short human-readable message emitted by `emit_progress(...)` at node boundaries. Surfaced via `agent.stream(...)` and SSE on the HTTP endpoint. See [Progress events](../observability/progress-events.md).

**EvidenceOverview** — The single rendered Markdown view of the EvidenceLedger that the answer composer reads. Produced by `render_evidence_ledger_overview()` from `evidence_ledger`; carries `markdown`, `allowed_urls` (visible source-block citation allowlist), `label_urls` (visible `E# -> URL` map), `label_by_evidence_id` (EvidenceRecord ids projected to URL-canonical labels), and `rendered_record_count` / `omitted_record_count` for budget visibility. Replaces the legacy `PromptEvidenceUnit` / `ReportEvidenceBundle` channels. See [Evidence pipeline](../architecture/evidence-pipeline.md#rendering-evidenceledger--markdown) and [Answer composition](../architecture/answer-composition.md).

**ProviderContext** — Runtime `@dataclass` containing the active `LLMProvider` and `SearchProvider`. Built before the graph runs and injected into every node. See [Architecture overview](../architecture/overview.md).

## R

**Report profile** — Enum `compact` or `deep`. Controls evidence context density, claim breadth, and answer length without changing provider wiring. See [Report profiles](../configuration/report-profiles.md).

**Reranker** — Optional stage that re-scores the fused candidate pool down to `final_k`. Unlike retrieval it sees question and passage *together*, so it can distinguish "about this topic" from "answers this question" — a distinction the ingestion-time vector could not encode. `cohere` calls a rerank-schema endpoint; `llm` is a listwise fallback through the deployment's own LLM, capped at 20 candidates. Default `none`. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#why-a-reranker-is-still-worth-it-after-retrieval).

**Retrieval profile** — The per-request switch bundling the optional `mode=knowledge` stages: `schnell`, `standard`, `gruendlich`, `tief`, or `auto`. It selects *within* the operator ceiling and can never enable what the environment forbids; every clamp appears as `degraded_stages`. See [Retrieval profiles](../configuration/knowledge-profiles.md).

**RRF (Reciprocal Rank Fusion)** — Rank-based fusion of the dense and BM25 result lists, computed server-side by Qdrant. It scores a chunk by the sum of `1 / (k + rank)` across branches, using only positions because cosine and BM25 scores are not on a comparable scale. Not a model and not an embedding. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#what-rrf-does-with-the-two-rankings).

**Verification basis** — Per-consolidated-claim label that records *why* a claim is verified or unverified. Possible values: `verified_cross_checked`, `verified_primary`, `verified_quality_source`, `contested`, `missing_primary_source`, and `weak_evidence`. Computed deterministically by `DefaultClaimConsolidator.consolidate()` from provider-grounded source records and projected back onto each `EvidenceRecord.claims[n]`. See [Claims](../scoring-and-stopping/claims.md#status-determination).

**Risk score** — Deterministic regex-based integer 0–10 computed from the question text. Values ≥ `HIGH_RISK_SCORE_THRESHOLD` (default 4) flag the question as `high_risk`. This is an observability signal only (forensic events, `/health`, follow-up preservation); it does not change model selection — use the model tiers or a per-node model override for a stronger model. See [Nodes](../architecture/nodes.md).

## S

**SearchProvider** — ABC for search backends. One method: `search(query, ...) -> GroundedSearchResult`; sources are carried as typed `GroundedSource` rows. Has a standardised `search_model` property for `/health` display. See [Providers overview](../providers/overview.md).

**Sparse vector (BM25)** — The lexical retrieval branch, stored as a vector but computed by an algorithm rather than a neural model: `Qdrant/bm25` through `fastembed` emits one weight per occurring token and zero everywhere else, with the rarity (IDF) term applied server-side. It is language-bound (`bm25_german` today), so cross-language keyword matching is structurally impossible except for shared tokens such as names and codes. See [Knowledge retrieval](../architecture/knowledge-retrieval.md#why-two-searches-instead-of-one).

**Sufficiency gate** — One fast-tier LLM call that judges whether the retrieved evidence carries the question and may propose a rewritten query for another retrieval round. The verdict is three-way coverage: `full` answers normally, `partial` answers with the gaps named, `none` yields the honest no-evidence answer. An unparseable response fails *open* with a visible `_knowledge_gate_fallback` marker. See [Retrieval profiles](../configuration/knowledge-profiles.md).

**Source tier** — One of `primary` (weight 1.0), `mainstream` (0.8), `stakeholder` (0.45), `unknown` (0.35), or `low` (0.1). Drives the source-quality score. See [Source tiering](../scoring-and-stopping/source-tiering.md).

**ScoreSnapshot** — One chronological scoring record appended after search, evaluate, and answer phases. It groups source, evidence, claim, coverage, stop, and answer-audit metrics. See [Score ledger](../scoring-and-stopping/score-ledger.md).

**STORM** — Multi-perspective search planning technique from Stanford OVAL. Inqtrix uses the perspective-diversity pattern in the plan node (round 1+) to rotate stakeholder viewpoints. See [Research foundations](research-foundations.md).

**Stop criteria strategy** — ABC whose methods implement the heuristic cascade used by `evaluate`; a global `min_rounds` suppression is applied afterwards in `nodes.py`. Default strategy is `MultiSignalStopCriteria`. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

**StrategyContext** — Runtime dataclass containing the five active strategy objects. Nodes call these strategies for tiering, claim extraction/consolidation, risk/aspect policy, and stop heuristics. See [Strategies](../architecture/strategies.md).

## T

**Tier** — See *source tier*.

**Testing mode** — Boolean flag (`TESTING_MODE=true`) that exposes the `/v1/test/run` endpoint, populates `state["iteration_logs"]`, and enables additional structured output in test/parity payloads. Off in production. See [Settings and env](../configuration/settings-and-env.md).

**`top_k`** — The per-(sub-)query retrieval width for `mode=knowledge` (`INQTRIX_KNOWLEDGE_TOP_K`, default 8, range 1–50; per-request override `knowledge_filters.top_k`). It bounds each individual `retrieve()` call, not the evidence that reaches the answer — that is `final_k`. See [Retrieval profiles](../configuration/knowledge-profiles.md).

## U

**Utility** — Marginal-gain signal computed in `compute_utility`: `0.3·Δconf + 0.2·Δcit + 0.2·sufficiency_norm + 0.3·evidence_gain`. Two consecutive rounds with utility < 0.15 trigger the utility plateau stop (unless policy suppression is active). See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

## V

**Vocabulary bridge** — A variant of the gate's rewrite prompt that translates everyday phrasing into the documents' technical or official vocabulary ("wie lange behalten" to "Aufbewahrungsfrist"). It costs no extra LLM call because it reuses the rewrite that already runs, and it is enabled in `gruendlich` and `tief`. See [Retrieval profiles](../configuration/knowledge-profiles.md).

## Related docs

- [Architecture overview](../architecture/overview.md)
- [Nodes](../architecture/nodes.md)
- [Strategies](../architecture/strategies.md)
- [FAQ](faq.md)
