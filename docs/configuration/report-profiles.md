# Report profiles

> Files: `src/inqtrix/report_profiles.py`, `src/inqtrix/agent.py`

## Scope

`ReportProfile` is the public switch for answer depth. It controls summarise-, context-, and answer-prompt budgets without changing provider wiring. Two profiles ship out of the box.

## The two profiles

| Profile | Enum | Optimised for | Typical answer length |
|---------|------|---------------|-----------------------|
| Compact | `ReportProfile.COMPACT` | Fast Q&A, chat UIs, cost control | 400–700 words |
| Deep | `ReportProfile.DEEP` | Review-style reports, research briefings | 900–1500 words |

The profile sets the following knobs on a round-by-round basis (see `report_profiles.py` for the exact values):

- `summarize_max_tokens` — per-source summary budget.
- `answer_max_tokens` — final answer token budget.
- `context_block_max_len` — per-block context truncation.
- `answer_prompt_citations_max` — upper bound on URLs in the answer prompt (can also be set directly via `AgentConfig`).

Operators typically change the profile **only**; the derived budgets should be tuned together because otherwise the answer can exceed its token budget before it reaches the citation section.

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

`apply_overrides()` treats the profile as an explicit field: when a request or an operator sets `report_profile`, the derived budgets are **not** overwritten with the profile's default. This keeps surgical overrides (e.g. `max_rounds=2` together with `report_profile=deep`) working the way the operator expects. The pinned scenarios live in `tests/test_server_overrides.py`.

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
