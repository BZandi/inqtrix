# Worked example

This is an illustrative end-to-end walkthrough. It matches the current heuristics and control flow, shows the sub-methods called in order, and carries realistic (but synthesised) LLM outputs. The point is to anchor the abstract node descriptions in one concrete run.

## Question

```text
Sollen zahnaerztliche Leistungen privatisiert werden, wie laeuft aktuell die Diskussion und was wuerde das fuer den GKV-Beitrag bedeuten?
```

## Step 1: `classify()` — risk scoring, classification, aspect derivation

**Method call sequence:**

1. `emit_progress(s, "Analysiere Frage...")` — stream progress.
2. `strategies.risk_scoring.score(question)` — keyword-regex scoring.

Score breakdown:

- `privatisiert` or `GKV-Beitrag` → +2 (policy/regulation)
- `aktuell` → +1 (recency)
- `sollen` → +1 (normative)

```text
risk_score = 4
high_risk = true    (>= HIGH_RISK_SCORE_THRESHOLD)
```

3. `providers.llm.complete(prompt, ...)` — single LLM call for classification plus decomposition. Model is `reasoning_model` because `high_risk=true` and `HIGH_RISK_CLASSIFY_ESCALATE=true`.

Illustrative parsed output:

```json
{
  "decision": "SEARCH",
  "language": "de",
  "search_language": "de",
  "recency": "month",
  "query_type": "general"
}
```

4. `parse_json_string_list(sub_q_text, max_items=3)` — parse the SUB_QUESTIONS JSON array:

```json
[
  "Wie ist der aktuelle politische Status von Vorschlaegen zur Privatisierung zahnaerztlicher Leistungen?",
  "Welche Akteure unterstuetzen oder kritisieren den Vorschlag?",
  "Welche belastbaren Hinweise gibt es auf Auswirkungen fuer den GKV-Beitrag?"
]
```

5. `strategies.risk_scoring.derive_required_aspects(question, query_type)`:

```json
[
  "Aktueller Stand: Gesetzgebung oder Beschluesse",
  "Positionen relevanter Akteure",
  "Quantitative Auswirkung auf den GKV-Beitrag",
  "Zeitlicher Rahmen und politischer Kontext",
  "Wissenschaftliche oder fachliche Einordnung",
  "Zahlenbasis mit Primaerbeleg oder expliziter Unsicherheit"
]
```

6. `append_iteration_log(s, {...})` — record metrics (testing mode).

**State written:** `risk_score=4`, `high_risk=true`, `done=false`, `language="de"`, `search_language="de"`, `recency="month"`, `query_type="general"`, `sub_questions=[...]`, `required_aspects=[...]`, `uncovered_aspects=[...]`, `aspect_coverage=0.0`.

## Step 2: `plan()` — query generation (round 0)

**Method call sequence:**

1. `emit_progress(s, "Plane Suchanfragen (Runde 1/4)...")`.
2. `_check_deadline(s["deadline"])`.
3. `providers.llm.complete(prompt, ...)` — generate search queries. Round 0 → "5-6 diverse queries".
4. `parse_json_string_list(q, max_items=FIRST_ROUND_QUERIES)` (max 6).

Illustrative queries:

```json
[
  "privatisierung zahnaerztliche leistungen gkv aktuelle diskussion",
  "zahnaerztliche leistungen privatisierung positionen parteien verbaende",
  "gkv beitrag auswirkung privatisierung zahnaerztliche leistungen",
  "aktueller status reform zahnaerztliche leistungen gesetz krankenkasse"
]
```

5. Policy detection: regex `\b(privatis\w*|gkv|krankenkass\w*|gesetz\w*|...)\b` matches → `is_policyish=True`.
6. `strategies.risk_scoring.inject_quality_site_queries(new_q, ...)`:
   - `quality_terms_for_question(question, query_type)` filters generic tokens.
   - Adds for example `"site:bundesgesundheitsministerium.de zahnarzt privat gkv"` and `"site:tagesschau.de zahnarzt privatisierung gkv diskussion"`.
7. Deduplicate against `s["queries"]`.
8. `append_iteration_log(s, {...})`.

**State written:** `queries` appended with 6 unique queries.

## Step 3: `search()` — parallel search, summarise, claim extraction

**Method call sequence:**

1. `emit_progress(s, "Durchsuche 6 Quellen...")`.
2. `_check_deadline(s["deadline"])`.

**Phase 1 — parallel Perplexity calls:**

3. `providers.search.search(q, ...)` × 6 in parallel via `ThreadPoolExecutor`.
   - Cache lookup by SHA-256 of `query + params`.
   - Domain filter: `site:` queries get allowlist, normal queries get `LOW_QUALITY_DOMAINS` blocklist.
   - Sends request to Perplexity Sonar via the search provider.
   - Round 0 → `return_related=True`.

**Phase 2 — parallel LLM post-processing:**

4. `providers.llm.summarize_parallel(text, deadline)` × M in parallel.
5. `strategies.claim_extraction.extract(text, citations, question, deadline)` × M in parallel:
   - Parses JSON claims.
   - Validates types (`fact`, `actor_claim`, `forecast`) and polarity (`affirmed`, `negated`).
   - Applies actor-verb regex to reclassify `"fact"` → `"actor_claim"` for speech verbs.
   - Applies primary-hint regex for `needs_primary`.
   - Normalises and allow-lists `source_urls` against the search result's citations.
   - Degrades non-fatally on provider errors.
   - Keeps at most 8 claims per result.

**Phase 3 — sequential assembly:**

6. `strategies.claim_consolidation.focus_stems_from_question(question)`.
7. Per result (sequential):
   - `normalize_url(url)`.
   - Append to `s["context"]` with source block + citations (max 8 per block).
   - `claim_matches_focus_stems(claim_text, focus_stems)`.
   - `claim_signature(claim_text)` for dedup.
   - Append matching claims to `s["claim_ledger"]`.
8. `strategies.source_tiering.quality_from_urls(all_citations)` — tier counts + quality score.
9. `strategies.claim_consolidation.consolidate(claim_ledger)`.
10. `strategies.claim_consolidation.materialize(consolidated)`.
11. `strategies.claim_consolidation.quality_metrics(consolidated_claims)`.
12. `strategies.context_pruning.prune(context, question, sub_questions, MAX_CONTEXT, n_new)`.
13. `strategies.risk_scoring.estimate_aspect_coverage(required_aspects, context)`.
14. Claim-ledger cap: keep last 400 if exceeded.
15. `append_iteration_log(s, {...})`.

Illustrative state after round 0:

```text
round = 1 (incremented)
context = [6 blocks]
claim_ledger = [14 raw claims]
consolidated_claims = {verified: 3, contested: 1, unverified: 5}
source_quality_score = 0.63
aspect_coverage = 0.67
```

## Step 4: `evaluate()` — quality assessment and stop decision

**Method call sequence:**

1. `emit_progress(s, "Bewerte Informationsqualitaet...")`.
2. Early return if `s["done"]`.
3. `_check_deadline(s["deadline"])`.
4. `strategies.source_tiering.quality_from_urls(all_citations)`.
5. `strategies.risk_scoring.estimate_aspect_coverage(required_aspects, context)`.
6. `strategies.claim_consolidation.consolidate(claim_ledger)`.
7. `strategies.claim_consolidation.materialize(consolidated)`.
8. `strategies.claim_consolidation.quality_metrics(consolidated_claims)`.
9. `strategies.claim_consolidation.claims_prompt_view(consolidated_claims, max_items=14)`.
10. `providers.llm.complete(eval_prompt, ...)` — LLM evaluation. Model = `reasoning_model` because `high_risk=true` and `HIGH_RISK_EVALUATE_ESCALATE=true`. Round < 2, so negative-evidence hint is **not** injected.

Illustrative output:

```text
CONFIDENCE: 6
GAPS: Quantitative Belege fuer GKV-Beitragsauswirkung fehlen.
CONTRADICTIONS: none
IRRELEVANT: none
COMPETING_EVENTS: none
EVIDENCE_CONSISTENCY: 7
EVIDENCE_SUFFICIENCY: 5
```

**Group A — LLM-parse heuristics:**

11. `check_contradictions(s, answer_text, conf=6)` — no cap.
12. `filter_irrelevant_blocks(s, answer_text)` — none removed.
13. `extract_competing_events(s, answer_text, conf=6)` — none.
14. `extract_evidence_scores(s, answer_text, conf=6)` — both > 0, no cap.

**Group B — guardrail caps:**

15. No-citation cap: not triggered.
16. Low-quality cap: not triggered.
17. Needs-primary cap: possibly triggered → conf stays ≤ 8 (already 6).
18. Uncovered-aspects cap: possibly triggered → conf stays ≤ 8.
19. Contested-claims cap: 1 contested < 2 → not triggered.

**Group C — post-LLM stop heuristics:**

20. `check_falsification(s, conf=6, prev_conf=0)` — `prev_conf=0`, skipped.
21. `check_stagnation(...)` — `prev_conf=0`, skipped.
22. `compute_utility(...)` — first round, utility treated as 0.5.
23. `check_plateau(...)` — delta too large, no plateau.
24. Final stop: `conf=6 < 8` and `round=1 < 4` → `done=false`.

**Result:** loop continues.

## Step 5: second round (`plan` at round 1, `evaluate` at round 2)

`plan()` differences in round 1:

- Prompt asks for "2-3 precise queries".
- STORM perspective-diversity instruction appended.
- Temporal-recency instruction added (round 1 specific).
- Alternative-hypothesis instruction added.
- `parse_json_string_list(q, max_items=3)`.
- `inject_quality_site_queries()` may add additional `site:` queries if primary ratio still low.

`search()` differences in round 1:

- Batch size = 3 (not `FIRST_ROUND_QUERIES`).
- `return_related = false`.
- Same Phase 1 → 2 → 3 pipeline.

`evaluate()` differences in round 2:

- `search()` incremented `s["round"]` from 1 to 2, so `evaluate()` now injects the negative-evidence hint.
- `check_falsification()` and `check_stagnation()` are active; they do not fire here because confidence remains above 4.
- `should_suppress_utility_stop()` may suppress utility-based stopping when the question matches the policy regex and uncovered aspects remain.

Illustrative state after round 1:

```text
round = 2
confidence = 6
claim_quality_score = 0.42
source_quality_score = 0.63
uncovered_aspects = ["Zahlenbasis mit Primaerbeleg oder expliziter Unsicherheit"]
done = false
```

## Step 6: third round — possible final stop

`plan()` differences in round 2:

- `round >= 2` AND `final_confidence <= 4` would trigger aggressive reformulation (not triggered here, conf=6).
- If `competing_events` were set, a comparison instruction would be added.

`evaluate()` after `search()` increments to round 3:

- Negative-evidence hint remains active.
- Falsification remains armed; does not trigger (confidence well above 4).
- Stagnation remains armed; does not trigger (pre-condition no longer met).
- Plateau: confidence changed between rounds, no plateau.

Illustrative state after round 2 (assuming stronger sources):

```text
round = 3
confidence = 8
aspect_coverage = 1.0
claim_needs_primary_verified = claim_needs_primary_total
done = true   (conf >= CONFIDENCE_STOP)
```

## Step 7: `answer()` — synthesis

**Method call sequence:**

1. `emit_progress(s, "Formuliere Antwort...")`.
2. `strategies.claim_consolidation.select_answer_citations(consolidated_claims, all_citations, max_items=ANSWER_PROMPT_CITATIONS_MAX)` — claim-ranked citation selection.
3. `strategies.claim_consolidation.claims_prompt_view(consolidated_claims, max_items=20)`.
4. `build_answer_system_prompt(state_data)`:
   - Structured response template (Kurzfazit / Kernaussagen / Detailanalyse / Einordnung).
   - Claim calibration rules.
   - Uncovered-aspects → "Unsicherheiten / Offene Punkte" section.
   - Competing-events disambiguation (if any).
   - Citation rules with allowed-URL set.
   - Research content marked as untrusted (prompt-injection guard).
5. `providers.llm.complete(question, system=system_prompt, ...)`.
6. `sanitize_answer_links(answer, allowed_citation_urls)`.
7. `count_allowed_links(answer, allowed_citation_urls)`:
   - If 0 links survive → append fallback source bar `**Quellen:** [1](url1) | [2](url2) | ...` with top 5 prompt citations.
8. Append stats footer:

```text
---
*18 Quellen · 9 Suchen · 3 Runden · 45s · Confidence 8/10*
```

9. `append_iteration_log(s, {...})`.
10. `emit_progress(s, "done")`.

**State written:** `answer` (final answer text including stats footer).

## Related docs

- [Nodes](../architecture/nodes.md)
- [Stop criteria](../scoring-and-stopping/stop-criteria.md)
- [Claims](../scoring-and-stopping/claims.md)
- [Iteration log](../observability/iteration-log.md)
