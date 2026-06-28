# Report profiles

> Files: `src/inqtrix/report_profiles.py`, `src/inqtrix/agent.py`

## Scope

`ReportProfile` is the public switch for answer depth. It controls evidence context density, claim breadth, and answer section layout without changing provider wiring or provider token defaults. Two profiles ship out of the box.

## The two profiles

| Profile | Enum | Optimised for | Typical answer length |
|---------|------|---------------|-----------------------|
| Compact | `ReportProfile.COMPACT` | Fast Q&A, chat UIs, cost control | 400–700 words |
| Deep | `ReportProfile.DEEP` | Review-style reports, research briefings | 900–1500 words |

### Knobs per profile

The profile sets the following knobs on a round-by-round basis. The values
below mirror `_COMPACT_TUNING` and `_DEEP_TUNING` in `report_profiles.py`:

| Knob | COMPACT | DEEP | Effect |
|---|---|---|---|
| `max_rounds` | 2 | 4 | Hard cap on research rounds. |
| `min_rounds` | 1 | 2 | Minimum completed rounds before early-stop is allowed. |
| `confidence_stop` | 7 | 8 | Evaluator-confidence threshold for the stop cascade. |
| `first_round_queries` | 6 | 10 | Broad search fan-out in Round 0. |
| `context_block_max_len` | profile default | profile default | Per-block truncation for ingested source passages. |
| `answer_prompt_citations_max` | 60 | 500 | Hard cap on citations passed to the answer composer. Can also be set directly via `AgentConfig`. |
| `claim_max_items` | profile default | profile default | Maximum claims extracted per source by the claim-extraction LLM. |
| `materialize_max_total` | 24 | 48 | Maximum `consolidated_claims` list size (research-loop view; **not** report breadth). |
| `materialize_max_unverified` | 8 | 48 | Sub-cap on unverified claims inside `materialize_max_total`. |
| `prompt_evidence_total_char_budget` | 30 000 | 180 000 | Total character budget for the rendered evidence overview that ends up in every section's system prompt. Overflow becomes `EvidenceOverview.omitted_record_count` with a visible "HINWEIS: …" footer in the overview -- never a silent cascade. |
| `prompt_evidence_record_char_limit` | 2 200 | 2 600 | Per-source-block budget inside the overview. Records that exceed it have their evidence lines trimmed in inhaltsstärkste-zuerst order (claims, snippets, passages, data points), with a visible `[...weitere Belege ... gekuerzt]` marker. |
| `min_report_eligible_evidence` | 3 | 8 | Minimum report-eligible records required before `evaluate()` accepts early-stop. **Hidden veto** in the stop cascade: applies even when utility-stop, plateau, or confidence-threshold has already fired. See [stop-criteria.md](../scoring-and-stopping/stop-criteria.md). |

Section list per profile is defined inline:

- COMPACT: `_COMPACT_ANSWER_SECTIONS` (report_profiles.py:155). Four
  sections: **Kurzfazit** (`write_last=True`), **Kernaussagen**,
  **Detailanalyse**, **Einordnung / Ausblick**.
- DEEP: `_DEEP_ANSWER_SECTIONS` (report_profiles.py:194). Six sections:
  **Executive Summary** (`write_last=True`), **Hintergrund / Kontext**,
  **Analyse**, **Perspektiven / Positionen**, **Risiken / Unsicherheiten**,
  **Fazit / Ausblick**.

### `AnswerSectionSpec.write_last`

A section flagged `write_last=True` is rendered **after** all body sections,
so it can see them via `report_so_far_summary` and `used_evidence_labels`.
Display order is preserved at assembly time -- the Executive Summary still
appears at the top of the final answer, even though its LLM call ran last.
See [Answer composition](../architecture/answer-composition.md) for the
exact mechanism, including the write-plan split and how the rolling body
summary is built.

### Operator guidance

Operators typically change the profile **only**; the derived evidence and
prompt budgets should not be tuned individually because they were calibrated
together. LLM output-token defaults live on provider constructors
(`default_max_tokens`) or provider-specific model configuration, not in
report profiles. Citations
are restricted to `EvidenceOverview.allowed_urls` -- the canonical URLs whose
source blocks are visible in the final evidence overview -- and the link
sanitizer strips any other URL.
There is no parallel numbered citation map.

In `DEEP`, claim extraction keeps more claims per source, the unverified
materialisation sub-cap is set equal to the total materialisation cap, and
`min_rounds` defaults to `2`. This keeps uncertain but potentially useful
evidence available for evaluation and cross-check planning instead of
hiding it behind a smaller secondary cap, while forcing at least one
additional research round before high-confidence early stopping can end a deep run.

## Selecting a profile

### Library mode

```python
from inqtrix import AgentConfig, ReportProfile, ResearchAgent

agent = ResearchAgent(AgentConfig(report_profile=ReportProfile.DEEP))
```

Or via environment:

```bash
export REPORT_PROFILE=deep
```

The env variable is read by `Settings` and flows into `AgentConfig` when the library uses the auto-creation path.

### Server mode

On the HTTP `/v1/chat/completions` endpoint, callers can flip the profile per request:

```json
{
  "model": "research-agent",
  "messages": [{"role": "user", "content": "..."}],
  "agent_overrides": {"report_profile": "deep"}
}
```

`"compact"` and `"deep"` are the two accepted values. Unknown values return HTTP 400. See [Web server mode](../deployment/webserver-mode.md) for the full override whitelist.

## Profile-switch semantics

`apply_overrides()` treats the profile and any caller/operator-provided knob as explicit: when a request or an operator sets `report_profile`, profile-derived defaults fill only fields that neither side set directly. This keeps surgical overrides (e.g. `max_rounds=2` together with `report_profile=deep`) working the way the operator expects, while a pure profile switch can still move from `deep` back to `compact`. The pinned scenarios live in `tests/test_server_overrides.py`.

## Writing a custom profile

Report profiles are plain dataclasses in `report_profiles.py`. Adding a new profile requires:

1. Define the new `ReportProfile` enum value.
2. Register its budget dataclass in `report_profiles.py` alongside `COMPACT` and `DEEP`.
3. Update `with_report_profile_defaults(...)` if your profile adds new derived fields.
4. Add a test scenario in `tests/test_server_overrides.py` that pins the new profile together with at least one other overridden field.

Keep profiles additive; do not remove or rename the existing ones without a deprecation cycle.

## Related docs

- [Agent config](agent-config.md)
- [Settings and env](settings-and-env.md)
- [Web server mode](../deployment/webserver-mode.md)
- [Answer composition](../architecture/answer-composition.md) -- how the section list and char budgets feed the per-section LLM calls.
- [Evidence pipeline](../architecture/evidence-pipeline.md) -- where `prompt_evidence_total_char_budget` and `prompt_evidence_record_char_limit` are consumed.
- [Calculation overview](../scoring-and-stopping/calculation-overview.md) -- every budget and threshold in one place.
