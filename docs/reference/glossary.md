# Glossary

Domain-specific terms that appear throughout the codebase and the docs. Terms are listed alphabetically. Cross-references link to the page where the concept is explained in depth.

## A

**AgentConfig** — Pydantic `BaseModel` that holds everything `ResearchAgent` needs at construction time: providers, strategies, timeouts, cache settings, report profile, and per-call tunings. Server-only deployment settings live on `ServerSettings` instead. See [Agent config](../configuration/agent-config.md).

**AgentState** — `TypedDict` with ~48 fields that LangGraph threads through all five nodes. Subsets carry over into follow-up turns via the session system. See [State and iteration](../architecture/state-and-iteration.md).

**Aspect coverage** — Deterministic token-match score (0.0–1.0) of how many required aspects are demonstrably present in the accumulated context. Caps confidence at 8 while aspects remain uncovered. See [Aspect coverage](../scoring-and-stopping/aspect-coverage.md).

## B

**Baukasten** — German for "construction kit". Internal shorthand for the pluggable-provider, pluggable-strategy design: a caller assembles providers and strategies as independent building blocks instead of configuring a monolith. Reflected in the Constructor-First principle.

## C

**Claim (ledger)** — Structured atomic statement extracted from a search result, stored in `state["claim_ledger"]`. Each entry has a signature, polarity, and source URLs. See [Claims](../scoring-and-stopping/claims.md).

**Claim status** — One of `verified`, `contested`, or `unverified`. Determined by support/contradict counts, source tier, and the `needs_primary` flag. See [Claims](../scoring-and-stopping/claims.md).

**Competing events** — Multiple conflicting explanations of the same event in the collected context (e.g. two different dates for the same policy vote). Detected by the evaluate LLM; caps confidence to `confidence_stop - 1` on first appearance. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

**Confidence stop** — Threshold (default 8) at which the loop terminates once `final_confidence` reaches it. See [Confidence](../scoring-and-stopping/confidence.md).

**Constructor first** — Design principle: providers never read environment variables directly. All values are passed as constructor arguments. `.env` translation happens only in example scripts and the `Settings` bridge layer.

## D

**DE-policy bias** — Regex that detects German policy topics (`privatis*|gkv|krankenkass*|gesetz*|verordnung*|beitrag*|kosten|haushalt*`). Used by the plan node to inject `site:`-prefixed quality-source queries and by the stop criteria to suppress utility-based stopping on policy-critical questions.

**Deadline** — Monotonic wall-clock timestamp computed at run start; every node entry and every provider call respects it. See [Timeouts and errors](../observability/timeouts-and-errors.md).

## E

**Effective model** — The model name a specific role (classify, summarize, evaluate) resolves to, after fallback to `reasoning_model`. Exposed via `LLMProvider.models.effective_*_model` and `resolve_summarize_model(llm, fallback)` (constructor-first).

**Effort / Thinking** — Opt-in reasoning budget exposed by Anthropic (`thinking`) and Azure/OpenAI (`effort`). Not all models accept all values; Inqtrix surfaces rejection warnings via `consume_effort_config_warnings`.

**Evidence consistency / sufficiency** — Two numeric signals the evaluate LLM is asked to produce alongside the confidence score. Both feed into the heuristic cascade; both default to 0 if unparseable, which triggers a sanity cap. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

## F

**Falsification mode** — One-shot switch that changes the plan node's query distribution to "2/3 debunk, 1/3 nearest-explanation" after two low-confidence rounds. See [Falsification](../scoring-and-stopping/falsification.md).

## I

**Iteration log** — Structured per-round record (a list of dicts) that every node appends to. Available on `ResearchResult` in testing mode. See [Iteration log](../observability/iteration-log.md).

## L

**LLMProvider** — ABC for language-model backends. Three methods: `complete`, `summarize_parallel`, `is_available`. Optional `complete_with_metadata` adds token counts. See [Providers overview](../providers/overview.md).

## P

**Plateau stop** — Stop heuristic that triggers when confidence has been stable at ≥ 6 across two or more rounds and competing events are unchanged. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

**Primary-need / `needs_primary`** — Flag on a claim that requires a primary-tier source for verification. If not satisfied, the claim stays `unverified` even when support exists on mainstream sources. See [Claims](../scoring-and-stopping/claims.md).

**Progress event** — Short human-readable message emitted by `emit_progress(...)` at node boundaries. Surfaced via `agent.stream(...)` and SSE on the HTTP endpoint. See [Progress events](../observability/progress-events.md).

## R

**Report profile** — Enum `compact` or `deep`. Controls summarize/context/answer budgets and answer length without changing provider wiring. See [Report profiles](../configuration/report-profiles.md).

**Risk score** — Deterministic regex-based integer 0–10 computed from the question text. Values ≥ `HIGH_RISK_SCORE_THRESHOLD` (default 4) escalate classify and evaluate to the stronger reasoning model. See [Nodes](../architecture/nodes.md).

## S

**SearchProvider** — ABC for search backends. One method: `search(query, ...) -> dict` with fixed return shape (`answer`, `citations`, `related_questions`, `_prompt_tokens`, `_completion_tokens`). Has a standardised `search_model` property for `/health` display. See [Providers overview](../providers/overview.md).

**Source tier** — One of `primary` (weight 1.0), `mainstream` (0.8), `stakeholder` (0.45), `unknown` (0.35), or `low` (0.1). Drives the source-quality score. See [Source tiering](../scoring-and-stopping/source-tiering.md).

**STORM** — Multi-perspective search planning technique from Stanford OVAL. Inqtrix uses the perspective-diversity pattern in the plan node (round 1+) to rotate stakeholder viewpoints. See [Research foundations](research-foundations.md).

**Stop criteria strategy** — ABC with ten methods that together form the nine-step heuristic cascade. Default is `MultiSignalStopCriteria`. See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

## T

**Tier** — See *source tier*.

**Testing mode** — Boolean flag (`TESTING_MODE=true`) that exposes the `/v1/test/run` endpoint, populates the iteration log on `ResearchResult`, and enables additional structured output. Off in production. See [Settings and env](../configuration/settings-and-env.md).

## U

**Utility** — Marginal-gain signal computed in `compute_utility`: `0.4 * delta_conf + 0.3 * delta_cit + 0.3 * sufficiency`. Two consecutive rounds with utility < 0.15 trigger the utility plateau stop (unless policy suppression is active). See [Stop criteria](../scoring-and-stopping/stop-criteria.md).

## Related docs

- [Architecture overview](../architecture/overview.md)
- [Nodes](../architecture/nodes.md)
- [Strategies](../architecture/strategies.md)
- [FAQ](faq.md)
