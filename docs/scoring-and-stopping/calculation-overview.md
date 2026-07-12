# Calculation overview

## Scope

Every internal score, count, threshold, and decision label in one place,
with its **provenance** (LLM-parsed or deterministic), its formula, the
inputs it reads, and a worked example. Use this page as the single
transparent reference when reviewing a run, debugging a stop decision, or
deciding whether a behaviour change belongs in this codebase or upstream
in a model prompt.

The page is intentionally an index: each row in the table links to the
dedicated detail page that explains the rule fully. The aim is that a
reader can scan one table and find which file in `src/inqtrix/` to open
next, and that no metric exists in the code without a row here.

## How to read the table

- **Type = LLM-parsed**: the value is extracted from a model response by a
  regex parser. The five LLM-parsed values come from one call -- the
  evaluator -- and a parse failure emits a `_…_parsed=False` marker plus a
  conservative default.
- **Type = Deterministic**: pure Python code, no model involvement after
  consolidation. Inputs are listed; the formula is shown either inline
  here or on the detail page.
- **Effect**: what decision the value gates. Surfaces hidden coupling --
  a metric used by three different decisions is harder to reason about
  than one used by a single decision.

## Master metric table

### Group 1 -- Risk and classification

| Metric / label | Where (`src/inqtrix/…`) | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `risk_score` | `strategies/_risk_scoring.py::KeywordRiskScorer.score()` | Deterministic | Question text | Regex keyword weights: +2 politics, +1 current-affairs, +1 numeric, +1 normative, +1 if >220 chars; capped at 10 | 0-10 | [confidence.md](confidence.md) |
| `high_risk` | `nodes.py::classify` | Deterministic | `risk_score`, `high_risk_score_threshold` | `risk_score >= threshold` | bool | [confidence.md](confidence.md) |
| `high_risk_score_threshold` | `settings.py` (`AgentSettings`) | Config | -- | Threshold for the `high_risk` flag; observability signal only (no model selection, no query/answer heuristic) | 4 | [confidence.md](confidence.md) |
| `required_aspects` | `strategies/_risk_scoring.py::derive_required_aspects()` | Deterministic | Question, `query_type`, profile | Query-type base aspects + topic extensions + profile additions; dedup; cap 6 COMPACT / 11 DEEP | -- | [aspect-coverage.md](aspect-coverage.md) |

### Group 2 -- Coverage

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `aspect_coverage` | `strategies/_risk_scoring.py::estimate_aspect_coverage()` | Deterministic | `required_aspects` + accumulated `context` | `covered / total`, 3 decimals; covered iff any augmented token of the aspect appears in context | 0.0-1.0 | [aspect-coverage.md](aspect-coverage.md) |
| `uncovered_aspects` | same function | Deterministic | Same | The aspects with zero token hits | List | [aspect-coverage.md](aspect-coverage.md) |

### Group 3 -- Source tier

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `tier_for_url` | `strategies/_source_tiering.py::tier_for_url()` | Deterministic | URL | Suffix-match against `PRIMARY_*` / `MAINSTREAM_*` / `STAKEHOLDER_*` / `LOW_QUALITY_*` sets in `domains.py` | `primary` / `mainstream` / `stakeholder` / `unknown` / `low` | [source-tiering.md](source-tiering.md) |
| `source_tier_counts` | `nodes.py::evaluate` via `quality_from_urls()` | Deterministic | `all_citations` | Per-tier count dict | dict[str,int] | [source-tiering.md](source-tiering.md) |
| `source_quality_score` | `strategies/_source_tiering.py::quality_from_urls()` | Deterministic | `all_citations` | Weighted mean: primary=1.0, mainstream=0.8, stakeholder=0.45, unknown=0.35, low=0.1 | 0.0-1.0 | [source-tiering.md](source-tiering.md) |
| `SOURCE_TIER_WEIGHTS` | `domains.py` | Constant | -- | `{primary:1.0, mainstream:0.8, stakeholder:0.45, unknown:0.35, low:0.1}` | -- | [source-tiering.md](source-tiering.md) |

### Group 4 -- Claim level (per consolidated claim)

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `verification_basis` | `strategies/_claim_consolidation.py::consolidate()` | Deterministic | `support_count`, `contradict_count`, tier booleans, `needs_primary` | 8-branch decision table | One of `verified_cross_checked` / `verified_primary` / `verified_quality_source` / `contested` / `missing_primary_source` / `weak_evidence` | [claims.md](claims.md#status-determination) |
| `status` | same | Deterministic | Same | Maps from `verification_basis` | `verified` / `contested` / `unverified` | [claims.md](claims.md) |
| `support_count` | same | Deterministic | Affirmed evidence rows | Count of affirmed evidence | int | [claims.md](claims.md) |
| `contradict_count` | same | Deterministic | Negated evidence rows | Count of negated evidence | int | [claims.md](claims.md) |
| `independent_support_count` | same | Deterministic | Affirmed evidence rows | Count grouped by distinct domain | int | [claims.md](claims.md) |
| `supporting_non_low_domain_count` | same | Deterministic | Domains backing the claim | Count where tier != `low` | int | [claims.md](claims.md) |
| `quality_domain_count` | same | Deterministic | Domains backing the claim | Count where tier in (`primary`, `mainstream`) | int | [claims.md](claims.md) |
| `has_primary` / `has_mainstream` / `has_stakeholder` | same | Deterministic | Tier set of backing domains | Boolean per tier | bool | [claims.md](claims.md) |

### Group 5 -- Claim aggregates

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `claim_status_counts` | `strategies/_claim_consolidation.py::quality_metrics()` | Deterministic | `consolidated_claims` | `{verified: n, contested: n, unverified: n}` | dict | [claims.md](claims.md) |
| `claim_quality_score` | same | Deterministic | `consolidated_claims` | Weighted average: cross_checked 1.0, primary 0.9, quality_source 0.7, fallback verified 0.8, contested 0.5, unverified 0.0 | 0.0-1.0 | [claims.md](claims.md#claim-quality-score) |
| `claim_needs_primary_total` | same | Deterministic | Claims flagged `needs_primary` | Count | int | [claims.md](claims.md) |
| `claim_needs_primary_verified` | same | Deterministic | `needs_primary` claims with `status=verified` | Count | int | [claims.md](claims.md) |

### Group 6 -- Evidence rendering

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `_record_verification_label` | `evidence.py` | Deterministic | A record's `claims[]` verification fields | `cross-checked` / `primary-source` / `single-source verified` / `contested` / `source-context` / `unverified` | -- | [evidence-pipeline.md](../architecture/evidence-pipeline.md#verification-labels-per-record) |
| `_evidence_record_score` | `evidence.py` (~628) | Deterministic | tier, verification label, claims/passages counts, snippet length, `source_date` | Tier rank + verification rank + capped contributions; see formula in evidence-pipeline.md | int | [evidence-pipeline.md](../architecture/evidence-pipeline.md#rendering-evidenceledger-markdown) |
| `_SOURCE_CONTEXT_TIER_RANK` | `evidence.py` (~25) | Constant | -- | `{primary:60, mainstream:50, stakeholder:40, unknown:25, low:-100}` | -- | [evidence-pipeline.md](../architecture/evidence-pipeline.md) |
| `_VERIFICATION_RANK` | `evidence.py` (~545) | Constant | -- | `{cross-checked:50, primary-source:42, contested:30, single-source verified:24, source-context:12, unverified:8}` | -- | [evidence-pipeline.md](../architecture/evidence-pipeline.md) |
| `rendered_record_count` | `evidence.py::render_evidence_ledger_overview()` | Deterministic | Records that fit the char budget | Counter | int | [evidence-pipeline.md](../architecture/evidence-pipeline.md) |
| `omitted_record_count` | same | Deterministic | Report-eligible records dropped by budget | Counter | int (visible in overview's "HINWEIS: …" footer) | [evidence-pipeline.md](../architecture/evidence-pipeline.md) |
| `allowed_urls` | same | Deterministic | URLs whose source blocks rendered visibly | Citation allowlist for answer composer | list | [evidence-pipeline.md](../architecture/evidence-pipeline.md) |
| `label_by_evidence_id` | same | Deterministic | `_evidence_record_score` ranking plus canonical URL dedupe | Stable EvidenceRecord-to-`E#` projection; records with the same URL share one label | dict | [evidence-pipeline.md](../architecture/evidence-pipeline.md) |

### Group 7 -- Evidence depth gap

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `evidence_depth_gap.active` | `nodes.py::_evidence_depth_gap()` (~207) | Deterministic | `consolidated_claims` | True iff `no_cross_checked_claims` OR `majority_single_source_claims` OR `central_claim_single_quality_source` | bool | [stop-criteria.md](stop-criteria.md#evidence-depth-gap) |
| `verified_count` / `cross_checked_count` / `single_source_verified_count` | same | Deterministic | Verified claims | Counts | int | [stop-criteria.md](stop-criteria.md) |
| `single_source_ratio` | same | Deterministic | Same | `single_source_verified_count / verified_count` | 0.0-1.0 | [stop-criteria.md](stop-criteria.md) |
| `verified_quality_source_single_count` | same | Deterministic | Claims with `verified_quality_source` and 1 URL | Count | int | [stop-criteria.md](stop-criteria.md) |
| `central_quality_source_single_count` | same | Deterministic | `verified_quality_source_single` claims whose text matches central-claim regex | Count | int | [stop-criteria.md](stop-criteria.md) |
| `_EVIDENCE_DEPTH_MIN_VERIFIED_BUNDLES` | `nodes.py` | Constant | -- | 3 (threshold for two of the three triggers) | -- | [stop-criteria.md](stop-criteria.md) |

### Group 8 -- Crosscheck planner

| Metric / label | Where | Type | Inputs | Formula / trigger | Default | Detail |
|---|---|---|---|---|---|---|
| `_select_crosscheck_targets` score | `nodes.py` (~279-346) | Deterministic | Per consolidated claim: `status`, `verification_basis`, `support_count`, `independent_support_count`, `citation_count`, `needs_primary`, claim text | Point system: `verified` non-cross_checked + citation_count<2 = +5; `verified_quality_source` = +4; `unverified` = +3; `needs_primary` = +2; `support<2` OR `independent<2` = +2; basis in weak set = +2; central-claim regex = +2 | Score per claim; top 3 selected | [stop-criteria.md](stop-criteria.md) |

### Group 9 -- Confidence pipeline

See [confidence.md](confidence.md) for the full 9-stage pipeline.

| Metric / label | Where | Type | Notes |
|---|---|---|---|
| `llm_confidence` | `nodes.py::evaluate` | **LLM-parsed** | Forensic raw value before any cap. |
| `final_confidence` | `nodes.py::evaluate` | Deterministic over LLM input | Value after 9 stages. Range 0-10. |
| `prev_conf` | function-local | Deterministic | The previous round's `final_confidence`. |
| `_conf_stable_rounds` | `s["_conf_stable_rounds"]` | Deterministic | Counter, increments when `conf == prev_conf`, resets otherwise. Triggers plateau at 2+. |
| `_confidence_parsed` | iteration log | Deterministic | `False` when the LLM `CONFIDENCE: N` regex did not match (default 5). |
| `_evaluate_fallback` | iteration log | Deterministic | `True` when the evaluator call itself raised or timed out. |
| `_confidence_unjustified_drop` | log warning only | Deterministic | Fires when LLM produced `conf < prev_conf` without new contradictions / competing events. **Not exposed in iteration log or score ledger** -- see [Surfaced complexity](#surfaced-complexity-candidates-for-code-simplification). |

### Group 10 -- Evaluator LLM signals

All five are **LLM-parsed** from the same evaluator call as `CONFIDENCE`.
The prompt + parser live in `prompts.py::EVALUATE_FORMAT_SUFFIX` (~lines
83-96) and `nodes.py::evaluate()` (~lines 4057-4080).

| Metric / label | Where parsed | Range | Effect | Detail |
|---|---|---|---|---|
| `evidence_consistency` | `_stop_criteria.py::extract_evidence_scores()` | 0-10 | Defaults to 5 if parse fails (`_evidence_consistency_parsed=False`). Feeds Stage 2c sanity cap. | [confidence.md](confidence.md#stage-2c--evidence-score-sanity-cap) |
| `evidence_sufficiency` | same | 0-10 | Defaults to 5 if parse fails. Lowered to ≤ 3 by Stage 3 "Empty claim ledger" guardrail. Feeds Stage 2c and `utility_score`. | [confidence.md](confidence.md) |
| `competing_events` | `_stop_criteria.py::extract_competing_events()` | text | Compared to `prev_competing_events` for novelty. Triggers Stage 2b cap when `conf >= confidence_stop` AND (new OR `round < 3`). | [confidence.md](confidence.md#stage-2b--competing-events-cap) |
| `contradictions_detected` | `_stop_criteria.py::check_contradictions()` | "ja"/"nein" + severity | Severity keywords (`grundlegend`, `fundamental`, `gegenteil`, `widerspricht`, `unvereinbar`) → cap `confidence_stop - 2`; otherwise → `confidence_stop - 1`. | [confidence.md](confidence.md#stage-2a--contradictions-cap) |
| `gaps` | `nodes.py::evaluate` | text | Free-text gap report. Read by next `plan()` round for query generation. | [stop-criteria.md](stop-criteria.md) |

### Group 11 -- Stop signals

| Metric / label | Where | Type | Trigger | Effect | Detail |
|---|---|---|---|---|---|
| `utility_score` | `_stop_criteria.py::compute_utility()` | Deterministic | `0.3·Δconf + 0.2·Δcit + 0.2·suff_norm + 0.3·evidence_gain` | Two consecutive rounds < 0.15 → `done=True` (unless suppressed) | [stop-criteria.md](stop-criteria.md) |
| `falsification_triggered` | `_stop_criteria.py::check_falsification()` | Deterministic | `round ≥ 2` AND `0 < prev_conf ≤ 4` AND `conf ≤ 4`; release when `conf ≥ confidence_stop - 2` | Next `plan()` injects "debunked" queries | [falsification.md](falsification.md) |
| `stagnation_detected` | `_stop_criteria.py::check_stagnation()` | Deterministic | `round ≥ 2` AND `prev_conf, conf ≤ 4` AND `|Δconf| ≤ 1` AND (`citations ≥ 30` OR `falsification_triggered`) | `done=True`, `_stop_reason="stagnation_low_evidence"` | [stop-criteria.md](stop-criteria.md) |
| `plateau_detected` | `_stop_criteria.py::check_plateau()` | Deterministic | `round ≥ 2` AND `conf == prev_conf` AND `conf ≥ 6` AND competing events not changing AND `!stagnation` AND `!evidence_depth_gap.active` | `done=True`, `_stop_reason="plateau_stop"` | [confidence.md](confidence.md#stage-4d--plateau-check) |
| `evidence_contract_status` | `nodes.py::answer` | Deterministic | Branch on `answer_evidence_bindings` + `algorithm_failures` + `depth_gap` | `clean` / `needs_review` / `source_context_only` / `algorithm_failed` / `unknown` | [evidence-pipeline.md](../architecture/evidence-pipeline.md#answerevidencebinding) |

### Group 12 -- Loop control

| Setting / metric | Where | Type | Effect | Default |
|---|---|---|---|---|
| `round` | `s["round"]` | int counter | Loop counter incremented by `search` | 0..max_rounds |
| `max_rounds` | `settings.py` | Config | Hard upper bound | 2 COMPACT / 4 DEEP |
| `min_rounds` | `settings.py` | Config | Suppresses stop until reached | 1 COMPACT / 2 DEEP |
| `confidence_stop` | `settings.py` | Config | Final stop threshold | 8 |
| `first_round_queries` | `settings.py` | Config | Round-0 query count | 6 COMPACT / 8 DEEP |
| `_LATER_ROUND_QUERY_MIN` | `nodes.py` | Constant | Min queries in later research rounds | 6 |

### Group 13 -- Materialization caps

See [report-profiles.md](../configuration/report-profiles.md).

| Knob | COMPACT | DEEP | Effect |
|---|---|---|---|
| `materialize_max_total` | 24 | 48 | `consolidated_claims` list size cap |
| `materialize_max_unverified` | 8 | 48 | Unverified sub-cap |
| `answer_prompt_citations_max` | 60 | 500 | Hard-cap on citations to the answer composer |
| `prompt_evidence_total_char_budget` | 30 000 | 180 000 | Total evidence-overview char budget |
| `prompt_evidence_record_char_limit` | 2 200 | 2 600 | Per-source-block char cap |
| `min_report_eligible_evidence` | 3 | 8 | Report-evidence floor for stop suppression |

### Group 14 -- Token and timeout budgets

| Knob | Where | Default | Effect |
|---|---|---|---|
| `DEFAULT_LLM_MAX_OUTPUT_TOKENS` | `constants.py` | 64 000 | Default output tokens for provider constructors |
| `max_total_seconds` | `settings.py` | 3600 | Active research-run wall-clock deadline |
| `reasoning_timeout` | `settings.py` | 600 | One reasoning operation, including retries |
| `search_timeout` | `settings.py` | 600 | One search operation, including retries |
| `claim_extract_timeout` | `settings.py` | 600 | One claim-extraction operation, including retries |

### Group 15 -- Answer audit

| Metric / label | Where | Type | Notes | Detail |
|---|---|---|---|---|
| `binding_status` | `evidence.py::audit_answer_evidence_bindings()` | Deterministic | `matched` / `source_context` / `unknown_citation` | [evidence-pipeline.md](../architecture/evidence-pipeline.md#answerevidencebinding) |
| `matched_evidence_count` | `nodes.py::answer` | Deterministic | Count of bindings with `matched` | [score-ledger.md](score-ledger.md) |
| `unknown_citation_count` | `nodes.py::answer` | Deterministic | Count of bindings with `unknown_citation` | [score-ledger.md](score-ledger.md) |
| `_CENTRAL_CLAIM_RE` | `nodes.py` | Constant regex | Pattern for numeric / benchmark / monetary / regulatory keywords; used by both depth-gap and crosscheck-planner | [stop-criteria.md](stop-criteria.md) |

## LLM-driven values

The codebase makes **five LLM-parsed values** depend on a model call. All
five come from the **same** evaluator LLM call (one call per `evaluate`
node round). The prompt template is in `prompts.py::EVALUATE_FORMAT_SUFFIX`
(~lines 83-96); the call is made in `nodes.py::evaluate()` (~line 3905);
the parser regex set is in `_stop_criteria.py` (`check_contradictions`,
`extract_competing_events`, `extract_evidence_scores`) and inline in
`nodes.py` for `CONFIDENCE`.

```text
STATUS: SUFFICIENT oder INSUFFICIENT
CONFIDENCE: 1-10                       -> llm_confidence / final_confidence
- Vergleiche bewusst mit der Vorrunde, falls vorhanden.
- Wenn neue Evidenz hinzugekommen ist UND keine neuen Widersprueche oder
  konkurrierenden Ereignisse auftauchen, sollte CONFIDENCE NICHT unter den
  Vorrunden-Wert sinken. Wenn doch, begruende kurz.
GAPS: kurze Liste                       -> gaps
CONTRADICTIONS: ja|nein, plus Severity  -> contradictions_detected
COMPETING_EVENTS: kurze Liste           -> competing_events
EVIDENCE_CONSISTENCY: 0-10              -> evidence_consistency
EVIDENCE_SUFFICIENCY: 0-10              -> evidence_sufficiency
```

Context passed to the call: question, sub-questions, required aspects,
the evidence overview excerpt, the previous round's confidence (so the
"should NOT decrease" rule can be evaluated against it), and the gap
hints from the prior round. Source: `nodes.py` (~lines 3850-3930).

Silent-degradation guarantees: parse failures emit
`_confidence_parsed=False`, `_evidence_consistency_parsed=False`,
`_evidence_sufficiency_parsed=False` markers in the iteration log; a full
evaluator failure emits `_evaluate_fallback=True`. None of the five LLM
values can fall back without emitting a marker.

Every other value in the master table above is purely deterministic over
state, regex, or config.

## Decision-condition trees

### `evaluate()` → `done=True`

```text
Stop accepted (Group A) when ANY of:
- compute_utility() sets done=True  (utility_score < 0.15 twice, unless suppressed)
- check_plateau() sets done=True    (conf >= 6, stable 2+ rounds, no depth_gap)
- check_stagnation() sets done=True (low conf + 30 citations or falsification)
- final_confidence >= confidence_stop
- round >= max_rounds

Then suppressed (Group B) when ANY of:
- round < min_rounds AND round < max_rounds
- report_eligible_evidence_count < min_report_eligible_evidence
    AND round < max_rounds
    AND evidence_ledger non-empty

Otherwise: done remains True.
```

### Confidence cap chain

See [confidence.md](confidence.md) for the full nine-stage flow.
Short form:

```text
parse(LLM CONFIDENCE)
  -> Stage 2a: contradictions cap
  -> Stage 2b: competing-events cap
  -> Stage 2c: evidence-sanity cap
  -> Stage 3:  7 deterministic guardrails (no-citations, no-claims,
                empty-ledger, low-tier-majority, needs-primary-missing,
                uncovered-aspects, contested>=2)
  -> Stage 4:  falsification/stagnation/utility/plateau (may set done=True,
                only stagnation and plateau set done directly; falsification
                only signals, utility computes score)
  -> Stage 5:  s["final_confidence"] = conf
  -> Stage 6:  _confidence_unjustified_drop diagnostic (log only)
```

### Prompt-block triggers (answer node)

| Block | Trigger |
|---|---|
| `CLAIM-KALIBRIERUNG` | `source_counts` or `claim_counts` non-empty |
| `ABDECKUNGSREGEL` | `required_aspects` non-empty |
| `EVIDENZTIEFE` | `evidence_depth_gap["active"] == True` |
| `TRANSPARENZPFLICHT` | `unverified_count > verified_count` OR `claim_needs_primary_verified < claim_needs_primary_total` OR `evidence_depth_gap["active"]` |
| `EVIDENZ-UEBERSICHT` | always |
| `ZITATIONS-REGELN` | always |

## Surfaced complexity (candidates for code simplification)

A side effect of writing this page was finding places where the rules are
correct but harder to explain than they should be, plus a small amount of
dead code. The table below records them honestly. **Status = documented**
means the rule is intentional and now visible; **status = simplification
candidate** means it is worth a follow-up code PR to either remove or
expose the value properly.

| Item | Lokation | Beschreibung | Status |
|---|---|---|---|
| Dead log-field references after refactor | `nodes.py::evaluate()` ~line 4009-4010 | Iteration-log reads `verified_bundle_count` / `cross_checked_bundle_count` from the depth-gap dict, but those keys are not written anymore (the dict carries `verified_count` / `cross_checked_count` after the refactor). The two reads silently default to 0. | **simplification candidate** |
| Magic `+4` bonus for `verified_quality_source` in crosscheck planner | `nodes.py::_select_crosscheck_targets()` | `verified_quality_source` claims get +4 ranking points without a code comment explaining why. The rule is intentional (lax single-quality-source verification needs cross-check), but the magic number deserves to be a named constant. | **documented** in [claims.md](claims.md) and Group 8 above |
| Confidence cap chain spread across 3 files | `nodes.py` + `apply_confidence_guardrails()` + `_stop_criteria.py` | 9 stages in 3 files; before this doc PR no single place explained the order and interactions. | **documented** in [confidence.md](confidence.md) |
| `_confidence_unjustified_drop` not exposed in structured output | `nodes.py` ~4187-4199 | Computed every round but only emitted as `log.warning()`; missing from `iteration_log`, `score_ledger`, and `ResearchResult`. A downstream observer cannot see when the rule fires. | **simplification candidate** (narrow scope) -- expose in `iteration_log` and `score_snapshot` so the observation actually accumulates. Whether the drop should also be auto-corrected (revert to `prev_conf`, raise a new stop-heuristic, change utility weighting, or keep the status quo) is a **separate** open design question; surfacing the marker is the prerequisite for deciding it. |
| `_evidence_record_score` weights without rationale | `evidence.py` ~628 | 7 weighted components and 6 cap values, no code comments justifying the numbers. | **documented** in [evidence-pipeline.md](../architecture/evidence-pipeline.md#rendering-evidenceledger-markdown) and Group 6 above |
| Plateau suppression hidden inside `check_plateau` | `_stop_criteria.py::check_plateau()` | `_evidence_depth_gap_active` suppresses the plateau stop; the link is not obvious without reading the code. | **documented** in [stop-criteria.md](stop-criteria.md) and [confidence.md](confidence.md#stage-4d--plateau-check) |
| First-writer-wins for guardrail gap hints | `nodes.py::apply_confidence_guardrails()` | Multiple guardrails may fire in one round, but only the first one publishes the gap hint. | **documented** in [confidence.md](confidence.md#hidden-coupling-----first-writer-wins-for-gap-hints) |
| Empty-claim-ledger special case | guardrails | Instead of capping `conf`, lowers `evidence_sufficiency` to ≤ 3, which feeds the next round's Stage 2c. | **documented** in [confidence.md](confidence.md) |
| Evidence overview not per-section scoped | `_compose_answer_sections` + section system prompt | Every section LLM call sees the full evidence overview; `section_focus_labels` is only an advisory bullet in the user prompt. | **documented** in [answer-composition.md](../architecture/answer-composition.md) (intentional design) |
| `used_evidence_labels` double effect | `nodes.py` + `evidence.py::select_section_evidence_records` | Same set acts as a soft hint in the user prompt AND a hard `-16` penalty in the focus ranker. | **documented** in [answer-composition.md](../architecture/answer-composition.md) |
| `report_so_far_summary` is unstructured text | `nodes.py::_compact_section_summary` | The summary handed to the write-last section is `Heading: 900-char body`, no claim IDs, no argument categories. The write-last section can drift. | **simplification candidate** -- a structured summary with claim IDs would tighten the Executive Summary |
| Write order ≠ display order | `_compose_answer_sections` write-plan split | Executive Summary is written last but appears first in the final answer. | **documented** in [answer-composition.md](../architecture/answer-composition.md) and [nodes.md](../architecture/nodes.md) |

The combination of "documented" rows above is the audit trail of this doc
PR; the "simplification candidate" rows are queued for follow-up code
work and are not changed in this docs-only PR.

## Related docs

- [Confidence](confidence.md) -- the most-cited page from this overview.
- [Stop criteria](stop-criteria.md)
- [Claims](claims.md)
- [Aspect coverage](aspect-coverage.md)
- [Falsification](falsification.md)
- [Source tiering](source-tiering.md)
- [Score ledger](score-ledger.md) -- chronological diagnostic record built from the values in this page.
- [Evidence pipeline](../architecture/evidence-pipeline.md)
- [Answer composition](../architecture/answer-composition.md)
- [Worked example](../reference/worked-example.md) -- one concrete run that exercises most of these values end-to-end.
