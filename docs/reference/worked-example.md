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

3. `providers.llm.complete(prompt, ..., model=..., reasoning_effort=...)` — single LLM call for classification plus decomposition. Model + effort come from the central tier router (`_resolve_node_llm`); classify maps to the **fast** tier by default. (`high_risk` no longer changes the model — see [LLM calls](../architecture/llm-calls.md).)

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

5. Deduplicate against `s["queries"]`.
6. `append_iteration_log(s, {...})`.

**State written:** `queries` appended with 6 unique queries.

## Step 3: `search()` — parallel search and claim extraction

**Method call sequence:**

1. `emit_progress(s, "Durchsuche 6 Quellen...")`.
2. `_check_deadline(s["deadline"])`.

**Phase 1 — parallel Perplexity calls:**

3. `providers.search.search(q, ...)` × 6 in parallel via `ThreadPoolExecutor`.
   - Cache lookup by SHA-256 of `query + params`.
   - Domain filter: `site:` queries get an allowlist; normal queries do not get a default blocklist.
   - Sends request to Perplexity Sonar via the search provider.
   - Round 0 → `return_related=True`.

**Phase 2 — parallel claim extraction:**

4. `strategies.claim_extraction.extract(text, citations, question, deadline)` × M in parallel:
   - Parses JSON claims.
   - Validates types (`fact`, `actor_claim`, `forecast`) and polarity (`affirmed`, `negated`).
   - Applies actor-verb regex to reclassify `"fact"` → `"actor_claim"` for speech verbs.
   - Applies primary-hint regex for `needs_primary`.
   - Normalises and allow-lists `source_urls` against the search result's citations.
   - Degrades non-fatally on provider errors.
   - Keeps at most 8 claims per result.

**Phase 3 — sequential assembly:**

6. `assemble_evidence_records(...)` produces 6-12 `EvidenceRecord` rows
   (`evidence.py`): each carries `evidence_id`, `record_type`,
   `report_eligible`, `query_id`, `canonical_url`, `domain`, `tier`,
   `source_title`, `source_snippet`, `source_date`,
   `source_passages[]`, `claims[]`, and `citation_set[]`. See
   [Evidence pipeline -- EvidenceRecord](../architecture/evidence-pipeline.md#evidencerecord)
   for the full schema.
7. `derive_claim_ledger_from_evidence(evidence_ledger)` -- **local
   variable**, not persisted on state. Flattens `EvidenceRecord.claims[]`
   into raw claim rows for the consolidator.
8. `strategies.source_tiering.quality_from_urls(all_citations)` -- tier counts + quality score (see [source-tiering.md](../scoring-and-stopping/source-tiering.md)).
9. `strategies.claim_consolidation.consolidate(local_raw_claims)` -- deterministic verification from provider-grounded supports, source tiers, and contradictions per [claims.md](../scoring-and-stopping/claims.md#status-determination). Unknown tiers stay in the evidence path.
10. `strategies.claim_consolidation.materialize(consolidated)` with profile caps `materialize_max_total` (24 COMPACT / 48 DEEP).
11. `strategies.claim_consolidation.quality_metrics(consolidated_claims)` -- weighted average per [claims.md](../scoring-and-stopping/claims.md#claim-quality-score).
12. `project_claim_verification_to_evidence(evidence_ledger, consolidated_claims)` -- writes `verification_status` / `verification_basis` / `supporting_evidence_ids` back onto each `EvidenceRecord.claims[n]` so the ledger is self-describing.
13. `_evidence_depth_gap(s)` -- diagnostic over the consolidated claims; sets `s["evidence_depth_gap"]` (see [stop-criteria.md](../scoring-and-stopping/stop-criteria.md#evidence-depth-gap)).
14. `strategies.risk_scoring.estimate_aspect_coverage(required_aspects, accumulated_text)`.
15. `append_iteration_log(s, {...})`.

Illustrative state after round 0:

```text
round = 1 (incremented)
evidence_ledger = [12 EvidenceRecords; 8 report-eligible]
consolidated_claims = [
  {claim_id: "...", status: "verified", verification_basis: "verified_cross_checked", support_count: 3},
  {claim_id: "...", status: "verified", verification_basis: "verified_primary", support_count: 1, source_urls: ["https://issuer.example/filing"]},
  {claim_id: "...", status: "contested", verification_basis: "contested", support_count: 2, contradict_count: 1},
  {claim_id: "...", status: "unverified", verification_basis: "weak_evidence"},
  ...
]
claim_status_counts = {verified: 3, contested: 1, unverified: 5}
claim_quality_score = 0.42      # weighted average per claims.md
source_quality_score = 0.63
aspect_coverage = 0.67
evidence_depth_gap = {
  "active": True,
  "verified_count": 3, "cross_checked_count": 1,
  "single_source_verified_count": 2, "single_source_ratio": 0.667,
  "reason": "majority_single_source_claims",
  ...
}
```

## Step 4: `evaluate()` — quality assessment and stop decision

**Method call sequence:**

1. `emit_progress(s, "Bewerte Informationsqualitaet...")`.
2. Early return if `s["done"]`.
3. `_check_deadline(s["deadline"])`.
4. `strategies.source_tiering.quality_from_urls(all_citations)`.
5. `strategies.risk_scoring.estimate_aspect_coverage(required_aspects, context)`.
6. `strategies.claim_consolidation.consolidate(local_raw_claims)`.
7. `strategies.claim_consolidation.materialize(consolidated)`.
8. `strategies.claim_consolidation.quality_metrics(consolidated_claims)`.
9. `strategies.claim_consolidation.claims_prompt_view(consolidated_claims, max_items=14)`.
10. `providers.llm.complete(eval_prompt, ..., model=..., reasoning_effort=...)` — LLM evaluation. Model + effort come from the tier router; evaluate maps to the **mid** tier by default. Round < 2, so negative-evidence hint is **not** injected.

Illustrative output:

```text
CONFIDENCE: 6
GAPS: Quantitative Belege fuer GKV-Beitragsauswirkung fehlen.
CONTRADICTIONS: none
COMPETING_EVENTS: none
EVIDENCE_CONSISTENCY: 7
EVIDENCE_SUFFICIENCY: 5
```

**Group A — LLM-parse heuristics:**

11. `check_contradictions(s, answer_text, conf=6)` — no cap.
12. `extract_competing_events(s, answer_text, conf=6)` — none.
13. `extract_evidence_scores(s, answer_text, conf=6)` — both > 0, no cap.

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

- Prompt builds one concrete research slot per later-round query.
- With the default COMPACT profile, round 1 asks for 6 precise questions; DEEP starts with 8.
- STORM perspectives fill remaining slots after gaps, cross-checks, primary-source, counterevidence, and data-verification slots.
- Temporal-recency instruction added (round 1 specific).
- Alternative-hypothesis instruction added.
- `parse_json_string_list(q, max_items=target_query_count)`.

`search()` differences in round 1:

- Batch size uses the same target-query helper as `plan()`, so COMPACT executes 6 later-round searches and DEEP executes 8.
- `return_related = false`.
- Same Phase 1 → 2 → 3 pipeline.

`evaluate()` differences in round 2:

- `search()` incremented `s["round"]` from 1 to 2, so `evaluate()` now injects the negative-evidence hint.
- `check_falsification()` and `check_stagnation()` are active; they do not fire here because confidence remains above 4.

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

`answer()` is split into six observable phases. The whole flow is
documented in [Answer composition](../architecture/answer-composition.md);
this section shows the same run end-to-end with concrete data.

Assume the loop has stopped at round 3 with the state shown above
(extended): `evidence_ledger` has 18 report-eligible records,
`evidence_depth_gap.active=True`, claim status mix `5 verified / 1
contested / 8 unverified`, `final_confidence=8`.

### Phase 5a — Render the evidence overview

```python
overview = render_evidence_ledger_overview(
    s["evidence_ledger"],
    max_total_chars=180_000,     # DEEP profile
    max_record_chars=2_600,
    query_synthesis=s["query_synthesis"],
)
```

Returns an `EvidenceOverview` dataclass:

```python
overview.markdown               # ~38 KB; 12 records rendered, 6 omitted
overview.rendered_record_count  # 12
overview.omitted_record_count   # 6  -> visible "HINWEIS: ..." footer in markdown
overview.allowed_urls           # visible source-block URLs only
overview.label_by_evidence_id   # {'ev_...': 'E1', 'ev_...': 'E2', ...}; same URL shares one label
overview.rendered_evidence_ids  # ['ev_...', ...]
```

### Phase 5b — The rendered Markdown block (excerpt)

```text
RECHERCHE-ERGEBNIS R1
Suchanfrage: privatisierung zahnaerztliche leistungen gkv aktuelle diskussion
Provider-Synthese (Kontext; nicht eigenstaendig verifiziert):
Die GKV-Reformkommission hat im April 2026 erstmals einen Vorschlag zur
Privatisierung bestimmter Zahnersatzleistungen formuliert. Reuters und
Tagesschau berichten parallel; die KZBV widerspricht in einer
Pressemitteilung. Quantitative Auswirkungen auf den GKV-Beitrag sind
weiterhin nur in Sekundaerquellen geschaetzt; ein BMG-Primaerbeleg liegt
nicht vor.

Quellen aus dieser Recherche:
[E1] Reuters: GKV-Reformkommission diskutiert Privatisierung von Zahnleistungen
  Datum: 2026-04-12 | Einstufung: mainstream | Beleglage: cross-checked
  Aussagen dieser Quelle:
  - Die GKV-Reformkommission diskutiert die Privatisierung von Zahnersatzleistungen.
  Belegausschnitte:
  - "Die heute vom Bundesgesundheitsministerium eingesetzte Reformkommission..."

[E2] Tagesschau: Streit um Privatisierung der Zahnleistungen
  Datum: 2026-04-13 | Einstufung: mainstream | Beleglage: cross-checked
  Aussagen dieser Quelle:
  - Die GKV-Reformkommission diskutiert die Privatisierung von Zahnersatzleistungen.
  Belegausschnitte:
  - "Patientenverbaende warnen vor der Mehrbelastung..."

[E3] KZBV-Pressemitteilung
  Datum: 2026-04-14 | Einstufung: stakeholder | Beleglage: single-source verified
  Aussagen dieser Quelle:
  - Die Kassenzahnaerztliche Bundesvereinigung lehnt die Privatisierungsplaene ab.

...

HINWEIS: 6 weitere belegfaehige Quellen passten nicht in das Evidenz-Budget
und sind in dieser Uebersicht nicht enthalten.
```

### Phase 5c — Build the per-section system prompt

`build_answer_section_system_prompt(state_data, heading=..., instruction=..., section_position=..., section_total=...)` produces the system message for each section call. The structure is identical across sections (only the inner style block changes). For this run, the LLM receives, in order:

1. **Header + guardrails** (SICHERHEIT / SELBST-VERIFIKATION / PRAEZISION / ZEITLICHE PRAEZISION / FORMATIERUNGS-REGELN) -- always.
2. **CLAIM-KALIBRIERUNG** -- fires (source/claim counts non-empty). Shows the tier mix and the `5 verified / 1 contested / 8 unverified` breakdown.
3. **ABDECKUNGSREGEL** -- fires (`required_aspects` non-empty). Lists the 6 required aspects and the still-uncovered ones.
4. **EVIDENZTIEFE** -- fires (`evidence_depth_gap.active=True`). Quoted verbatim from `prompts.py`:

   ```text
   EVIDENZTIEFE -- WICHTIG FUER DEN TON DES REPORTS:
   - Von 5 verifizierten Aussagen sind nur 2 cross-checked; 3 (60%) ruhen
     auf einer einzigen Quelle.
   - Behandle single-source verified Aussagen mit inline-Attribution
     ("laut [E12] ...") und nicht als gesichert.
   - Erwaehne im Bericht explizit, dass die Belegtiefe begrenzt ist.
   ```
5. **TRANSPARENZPFLICHT** -- fires (`unverified_count=8 > verified_count=5`, depth_gap active). Tells the LLM to add a "Unsicherheiten / Offene Punkte" sub-section in the Risiken section.
6. **Section style block** (`_build_section_answer_style`) -- the only part that changes per call. Example for section 3 of 6 (Analyse): `"Aktueller Abschnitt: 3/6 -- **Analyse** ..."`.
7. **EVIDENZ-UEBERSICHT** -- the `overview.markdown` from Phase 5b embedded verbatim.
8. **ZITATIONS-REGELN** -- inline Markdown links labelled `E1` whose targets
   are in the allowlist, no separate Quellen list.

### Phase 5d — Section-by-section LLM calls

DEEP profile = 6 sections. Write order: 2 -> 3 -> 4 -> 5 -> 6 -> 1
(Executive Summary is `write_last`). Display order: 1 -> 2 -> 3 -> 4 -> 5 -> 6.

**Call 1: section 2/6 = Hintergrund / Kontext**

User prompt (built by `build_answer_section_user_prompt`):
```text
Nutzerfrage:
Sollen zahnaerztliche Leistungen privatisiert werden, wie laeuft aktuell
die Diskussion und was wuerde das fuer den GKV-Beitrag bedeuten?

Schreibe jetzt nur den Abschnitt 'Hintergrund / Kontext'.
Abschnittsfokus: Erklaere den Ausgangspunkt der Debatte und den
gesetzlichen Rahmen.

Fuer diesen Abschnitt besonders relevante Quellen (weiche Empfehlung,
du darfst auch andere Quellen aus der Evidenz-Uebersicht zitieren):
E3, E7, E12

Gib nur den Abschnittsinhalt ohne die Hauptueberschrift '## Hintergrund / Kontext' zurueck.
```

LLM output (synthetic, ~3-4 sentences):
```text
Die GKV-Reformkommission hat im April 2026 erstmals einen Vorschlag zur
Privatisierung bestimmter Zahnersatzleistungen formuliert
[E3](https://www.reuters.com/...). Der politische Anstoss kam aus der
laufenden Beitragsdebatte: zuletzt war der durchschnittliche
Zusatzbeitrag auf 1,9 % angestiegen [E12](https://www.tagesschau.de/...).
```

After this section: `used_evidence_labels = {E3, E12}`, `report_so_far_summary = "Hintergrund / Kontext: Die GKV-Reformkommission hat im April 2026 ..."`.

**Calls 2-5: sections 3-6 = Analyse / Perspektiven / Risiken / Fazit**

Each call inherits the accumulated `completed_headings`,
`report_so_far_summary`, `used_evidence_labels`, plus a fresh
`section_focus_labels` from `select_section_evidence_records()` -- where
the labels in `used_evidence_labels` get a `-16` rank penalty so coverage
spreads.

By the end of call 5 (Fazit / Ausblick):
```text
used_evidence_labels = {E1, E3, E4, E7, E8, E12, E15, E20}
report_so_far_summary (rolling 2400 chars) =
"Hintergrund / Kontext: Die GKV-Reformkommission hat im April 2026 ...
Analyse: Drei zentrale Treiber wurden identifiziert: ...
Perspektiven / Positionen: KZBV widerspricht (E3, E7), waehrend ...
Risiken / Unsicherheiten: Die Belegtiefe ist begrenzt; ...
Fazit / Ausblick: ..."
```

**Call 6: section 1/6 = Executive Summary** (`write_last=True`,
`synthesizing_existing=True`)

User prompt now carries the **full** `report_so_far_summary` and the
prompt-wording switch:

```text
...
Bisherige Report-Zusammenfassung:
Hintergrund / Kontext: Die GKV-Reformkommission hat im April 2026 ...
Analyse: Drei zentrale Treiber wurden identifiziert: ...
Perspektiven / Positionen: KZBV widerspricht (E3, E7), waehrend ...
Risiken / Unsicherheiten: Die Belegtiefe ist begrenzt; ...
Fazit / Ausblick: ...

Bereits verwendete Evidence-Labels:
E1, E3, E4, E7, E8, E12, E15, E20

Stuetze deine Verdichtung auf diese Labels, wenn du Aussagen
aus den geschriebenen Abschnitten zusammenfasst.
```

LLM output (synthetic, ~600 chars): a compact summary citing primarily
already-used labels.

### Phase 5e — Final assembly

```python
rendered_by_display_index = {
    1: "...Executive Summary body...",
    2: "...Hintergrund / Kontext body...",
    3: "...Analyse body...",
    4: "...Perspektiven / Positionen body...",
    5: "...Risiken / Unsicherheiten body...",
    6: "...Fazit / Ausblick body...",
}
answer_text = "\n\n".join(rendered_by_display_index[i] for i in sorted(rendered_by_display_index))
```

So even though "Executive Summary" was the **last** LLM call, it appears
**first** in the final Markdown.

### Phase 5f — Citation sanitization + answer audit

- `sanitize_answer_links(answer, allowed_urls)` strips any Markdown link
  whose URL is not in the visible source-block allowlist.
- `audit_answer_evidence_bindings()` walks the answer body, finds each
  `[E\d+]` reference, and records:

  ```python
  AnswerEvidenceBinding(
      binding_id="bind_...",
      answer_segment_id="claim_62ff008895f616",
      answer_segment_preview="Die GKV-Reformkommission ...",
      evidence_ids=["ev_..."],
      citation_urls=["https://www.reuters.com/...", "https://www.tagesschau.de/..."],
      matched_citation_urls=["https://www.reuters.com/...", "https://www.tagesschau.de/..."],
      binding_status="matched",
  )
  ```

- `evidence_contract_status` is set: with `depth_gap.active=True`, even a
  "clean" audit downgrades to `needs_review` so the surface flag in the
  result reflects the depth concern.

### Stats footer

```text
---
*18 Quellen . 9 Suchen . 3 Runden . 45s . Confidence 8/10*
```

**State written:** `answer` (final answer text including stats footer),
visible citation allowlist / label fields (`allowed_citations`,
`evidence_label_urls`, `evidence_label_by_id`, `visible_evidence_labels`,
`rendered_evidence_ids`), `answer_claim_bindings`,
`answer_evidence_bindings`, `answer_finish_reason`,
`answer_incomplete=False`, iteration-log entries, and token totals. No
bundles, no prompt units, no unverified-notes -- the state is the rendered
answer, visible citation basis, and audit.

## Where to find these fields in the code

| Step | File | Function / region |
|---|---|---|
| `classify` LLM call + risk score | `nodes.py` | `classify()`; `strategies/_risk_scoring.py::KeywordRiskScorer.score()` |
| `plan` LLM call | `nodes.py` | `plan()` |
| `assemble_evidence_records` | `evidence.py` | `assemble_evidence_records()` |
| `derive_claim_ledger_from_evidence` | `evidence.py` | `derive_claim_ledger_from_evidence()` |
| `consolidate` / `quality_metrics` | `strategies/_claim_consolidation.py` | `DefaultClaimConsolidator.consolidate()` and `.quality_metrics()` |
| `project_claim_verification_to_evidence` | `evidence.py` | same name (~line 727 region) |
| `_evidence_depth_gap` | `nodes.py` | ~line 207 |
| `apply_confidence_guardrails` | `nodes.py` | ~lines 3705-3806 |
| Stop hooks (contradictions / competing / sufficiency / falsification / stagnation / utility / plateau) | `strategies/_stop_criteria.py` | `MultiSignalStopCriteria` methods |
| `render_evidence_ledger_overview` + `EvidenceOverview` | `evidence.py` | ~lines 555-902 |
| Section composer | `nodes.py` | `_compose_answer_sections()` (~970-1277) |
| Section system prompt | `prompts.py` | `_build_answer_system_prompt_with_style()` (~405-702) |
| Section user prompt | `prompts.py` | `build_answer_section_user_prompt()` (~204-274) |
| `_compact_section_summary` / `_extract_evidence_labels` | `nodes.py` | ~798-803 / ~768-770 |
| `select_section_evidence_records` | `evidence.py` | ~905 |
| `AnswerSectionSpec`, `_COMPACT_ANSWER_SECTIONS`, `_DEEP_ANSWER_SECTIONS` | `report_profiles.py` | 28-65, 155, 194 |
| `audit_answer_evidence_bindings` | `evidence.py` | same name |

## Related docs

- [Nodes](../architecture/nodes.md)
- [Evidence pipeline](../architecture/evidence-pipeline.md)
- [Answer composition](../architecture/answer-composition.md)
- [Confidence](../scoring-and-stopping/confidence.md)
- [Stop criteria](../scoring-and-stopping/stop-criteria.md)
- [Claims](../scoring-and-stopping/claims.md)
- [Calculation overview](../scoring-and-stopping/calculation-overview.md)
- [Iteration log](../observability/iteration-log.md)
