# Retrieval profiles (mode=knowledge)

> Files: `src/inqtrix/knowledge/profiles.py`, `src/inqtrix/knowledge/algorithm.py`, `src/inqtrix/knowledge/decompose.py`, `src/inqtrix/providers/rerankers.py`

## Scope

A retrieval profile is the per-request switch for how much machinery a
knowledge question runs through. The request names ONE profile in
`knowledge_filters.profile`; the pipeline stages (rerank, sufficiency
gate, vocabulary-bridge rewrite, query decomposition, report-form
synthesis) are bundled behind it. The architecture and data flow of those stages are in [Knowledge retrieval](../architecture/knowledge-retrieval.md). A request without a profile behaves
exactly like the pre-profile pipeline (`standard`).

```
Question
  │  (profile "auto": zero-cost heuristic picks schnell/standard/gruendlich)
  ▼
Resolve plan  =  requested profile ∩ operator ceiling (env)
  ▼
┌──────────────────── the agentic loop ────────────────────┐
│ retrieve (dense + BM25 hybrid) ─► [rerank] ─► evidence   │
│      ▲                                  │                │
│ rewrite the query           [gate: sufficient?]          │
│ (vocabulary bridge)              │ no, rounds left       │
│      └───────────────────────────┘                       │
└────────────────│ yes ────────────────────────────────────┘
                 ▼
   answer (quotes first), verify quotes deterministically
   `tief` additionally: decompose into sub-queries before the loop,
   interleave the per-aspect results, answer as a sectioned report.
```

## The profile matrix

| Stage | `schnell` | `standard` (default) | `gruendlich` | `tief` |
|---|---|---|---|---|
| Rerank | off | on (if configured) | on, candidate depth ×1.5 | on, ×2.0 |
| Gate rewrite rounds | 0 (no gate call) | 1 | up to 2 | up to `INQTRIX_KNOWLEDGE_GATE_MAX_ROUNDS` |
| Vocabulary bridge | – | off | ON | ON |
| Query decomposition | – | – | – | ON (one fast-tier call) |
| Answer form | compact | compact | compact | sectioned report |
| Grounding (quote check) | ON | ON | ON | ON |
| Typical LLM calls | 1 | 2 | 2–4 | 4–7 |

Grounding stays on everywhere: verification is deterministic (no extra LLM
call) and fail-closed. A malformed quote block or one unverifiable labelled
quote terminates the answer with a typed, visible cause; it never becomes a
plain-answer fallback. `schnell` saves the gate call, the rerank roundtrip,
and any second retrieval pass — that is the entire latency win.

`auto` routes per question with zero-cost heuristics (strong
enumeration markers, multiple question marks, length) and NEVER picks
`tief`; the chosen profile and the reason travel in the
`inqtrix.knowledge.profile.resolved` event as the telemetry on which a
later LLM-escalation decision will be grounded.

## The two-level rule

Environment switches are the OPERATOR CEILING; the profile selects
within it:

* `INQTRIX_KNOWLEDGE_GATE=off` removes the gate from every profile.
* `INQTRIX_KNOWLEDGE_GROUNDING=off` removes quote verification
  everywhere.
* No reranker configured (`INQTRIX_RERANKER_PROVIDER=none`) removes
  the rerank stage everywhere.
* `INQTRIX_KNOWLEDGE_GATE_MAX_ROUNDS` (default 3, 1–5) caps the
  rewrite rounds of every profile; `tief` requests up to this cap.

A profile never enables what the ceiling forbids. Every clamp is
visible — `degraded_stages` appears in the `profile.resolved` event,
in `result_state.knowledge_profile`, and per profile in the
`/v1/capabilities` manifest (`knowledge.profiles[].degraded`). The UI
renders what would actually run; nothing degrades silently.

## Transport and validation

```json
POST /v1/runs   (or /v1/chat/completions)
{
  "mode": "knowledge",
  "question": "…",
  "knowledge_filters": {
    "collection_ids": ["kc_…"],
    "profile": "gruendlich",
    "top_k": 8,
    "final_k": 12
  }
}
```

Valid `profile` values: `schnell`, `standard`, `gruendlich`, `tief`,
`auto`. An unknown value fails the request with HTTP 400 naming the
valid set — a typo never silently runs a different profile.

`top_k` (per-query retrieval width) and `final_k` (the number of
evidence chunks that reach the answer) are orthogonal overrides under
every profile. Both are validated at the one resolver chokepoint and
fail with HTTP 400 when not a plain integer in range — `top_k` in
`1..50` (mirrors `INQTRIX_KNOWLEDGE_TOP_K`), `final_k` in
`1..EVIDENCE_K_MAX`. Without `final_k` the surfaced-evidence count is
`min(top_k * profile.final_k_factor, EVIDENCE_K_MAX)` (only `tief`
raises the factor above `1.0`); an explicit `final_k` pins it directly,
overriding the factor. The ceiling `EVIDENCE_K_MAX` and every profile's
`final_k_factor` are published in the `/v1/capabilities` manifest
(`knowledge.evidence_k_max`, `knowledge.profiles[].final_k_factor`) so a
client can render the effective `final_k` and bound its override.

Note: `final_k` can exceed `top_k`. The wider candidate pool comes from
the breadth of the run, not from `top_k` — query decomposition (`tief`)
fans out into sub-queries each retrieving `top_k`, the reranker selects
`final_k` from a far deeper candidate pool, and gate rewrite rounds add
more. For a profile without decomposition, the single retrieval is run
at `final_k` directly (so `top_k` only governs per-sub-query width).

## Stage notes

* **Vocabulary bridge** — the gate rewrite prompt variant that
  translates everyday phrasing into the documents' technical/official
  vocabulary (the measured DORA d20 failure class). It lives INSIDE
  the gate rewrite: there is exactly one place in the pipeline where
  queries are rewritten. Enabled in `gruendlich`/`tief`; promotion
  into `standard` is a deliberate re-baselining decision after the
  DORA answer eval shows gains without a false-refusal regression.
* **Query decomposition** (`knowledge/decompose.py`) — splits
  multi-aspect questions into 2–4 self-contained sub-queries (it
  splits, never reformulates — that distinction keeps it out of the
  rewrite location). Results merge round-robin so every aspect
  contributes to the top-k (the BSI b30 aggregation fix). Unparseable
  responses degrade to a no-op with the loud
  `_knowledge_decompose_fallback` marker.
* **LLM reranker** (`INQTRIX_RERANKER_PROVIDER=llm`) — listwise
  ranking through the deployment's own LLM for installations without
  a rerank API contract. Roughly an order of magnitude costlier and
  slower than a cross-encoder; candidates are hard-capped at 20 per
  query (visibly logged). A fallback, not a Cohere replacement. New
  routing nodes: `knowledge_rerank`, `knowledge_decompose` (both
  fast tier).

## Evaluation

The answer eval (the only eval that exercises the gate) parametrizes
over golden tiers and profiles:

```bash
INQTRIX_EVAL_GOLDEN_SET=dora INQTRIX_EVAL_KNOWLEDGE_PROFILE=gruendlich \
  uv run --env-file .env pytest tests/eval/test_answer_eval.py -v

# Standard pip/plain-Python environment:
python -m pip install -e ".[dev]"
set -a
. ./.env
set +a
INQTRIX_EVAL_GOLDEN_SET=dora INQTRIX_EVAL_KNOWLEDGE_PROFILE=gruendlich \
  python -m pytest tests/eval/test_answer_eval.py -v
```

Baselines are keyed `(model, tier, profile)`; tiers without
`no_evidence` queries report `abstention_rate: null` and skip that
floor. The `dora_holdout` tier is never tuned against and runs only as
a held-out overfitting-regression gate.
