# Model Cards

## Scope

How Inqtrix describes selectable LLMs to the UI: the curated **model-card
catalogue** (`src/inqtrix/model_cards.py`), how a provider offers concrete
models for direct selection (`selectable_models`), how a per-run model/effort
override reaches the wire, and how the catalogue is exposed to and rendered by
the React app (the model picker and the composer token meter). It does **not**
cover the algorithm's internal high/mid/fast tier routing — that stays in
[`../architecture/llm-calls.md`](../architecture/llm-calls.md); model cards are
an additive, optional layer on top.

## What a model card is

A *model card* is provider-neutral metadata for one logical model. One card per
model, regardless of how many providers serve it. The card carries the
UI-facing facts plus a flat `aliases` list of every provider-specific identifier
string that should resolve to it, so the same logical model resolves to one card
whether called via the Anthropic API (`claude-opus-4-8`), Amazon Bedrock
(`anthropic.claude-opus-4-8`, optionally region-prefixed `eu.`/`us.` and
version-suffixed `-v1:0`), or an Azure deployment.

| Field | Behaviour |
|---|---|
| `id`, `display_name`, `vendor` | Identity. The UI always shows `display_name`, never a raw versioned id. |
| `category` (`high`/`mid`/`fast`) | Display grouping in the picker. Independent of latency and of node-tier routing. |
| `speed` | TEMPO tile label (`langsam`/`mittel`/`schnell`); decoupled from `category`. |
| `context_window_tokens`, `max_output_tokens` | Capacity. Drive the token meter: usable input ≈ window − reserved output − safety. |
| `reasoning_levels` | Effort tokens the model accepts (incl. `none` only if reasoning can be disabled). Empty ⇒ no effort control (e.g. Haiku), so the picker hides the reasoning selector. |
| `capabilities` | Chips in the hover-card (`reasoning`/`code`/`tool_use`/`vision`). |
| `pricing` | List $/MTok; the picker derives a `$`..`$$$$` tier + exact value. |
| `aliases` | Every identifier that resolves to this card (matched after Bedrock region/version normalisation). |
| `source_url`, `last_verified` | Provenance — facts are curated, not guessed (Designprinzip 1). |

## Offering models for direct selection (optional)

Every LLM provider accepts an **optional** `selectable_models` list. Empty (the
default) keeps the UI on the high/mid/fast tier picker — a web-search-only
library user needs no model configuration at all. When set, the UI offers those
concrete models, grouped by each card's `category`.

```python
from inqtrix import AnthropicLLM

# Anthropic / OpenAI-direct: ship the latest models as defaults.
llm = AnthropicLLM(
    api_key=ANTHROPIC_API_KEY,
    default_model="claude-opus-4-8",
    tier_high_model="claude-opus-4-8",
    tier_mid_model="claude-sonnet-4-6",
    tier_fast_model="claude-haiku-4-5",
    context_window_tokens=1_000_000,
    selectable_models=["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
)
```

Bedrock uses region-prefixed ids; they normalise onto the same cards:

```python
BedrockLLM(
    region_name="eu-central-1",
    default_model="eu.anthropic.claude-opus-4-8-v1:0",
    selectable_models=[
        "eu.anthropic.claude-opus-4-8-v1:0",
        "eu.anthropic.claude-sonnet-4-6-v1:0",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
```

Azure deployment names are operator-chosen — **name each deployment after the
canonical model id** so the catalogue resolves it (see
[`../../examples/webserver_stacks/azure_foundry_web_search.py`](../../examples/webserver_stacks/azure_foundry_web_search.py)):

```python
AzureOpenAILLM(
    azure_endpoint=AZURE_ENDPOINT,
    default_model="gpt-5.4",
    selectable_models=["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
)
```

## Per-run model + effort override

A request may pick a concrete model and reasoning effort via
`agent_overrides.model` and `agent_overrides.effort` (HTTP), mirrored by
`AgentSettings.model`/`effort` and `AgentConfig.model`/`effort`. An explicit
model short-circuits tier resolution in `describe_resolution` with
`model_source="explicit_request"` (visible in the forensic log). The override is
**scoped to the direct-chat answer and the editor-assist endpoints**; research
runs keep tier routing for every node, because one model cannot stand in for the
classify/plan/evaluate/answer split.

## How the catalogue reaches the UI

`/health` and `/v1/stacks` return `models_catalog` (one `{model_id, card}` entry
per selectable model; `card` is `null` for an unknown id) and a top-level
`context_window_tokens` (provider-level, `null` when unknown). The React picker
renders models grouped by `category` with an `i` hover-card (KONTEXT/TEMPO/KOSTEN
tiles + capability chips) and an adaptive `No think`/`Think`/`Think hard`
reasoning selector derived from the card's `reasoning_levels`. The composer token
meter uses the selected card's `context_window_tokens` as its capacity.

## Unknown model (no card on file)

If a selectable model has no matching card, nothing is fabricated (Designprinzip
1): `resolve_model_card` returns `None`, the `/health` entry has `card: null`,
the picker shows the raw id in an "Ohne Kategorie" group with a muted "Keine
Karte" badge (no hover stats, no reasoning selector), and the token meter keeps
estimating but reports "context window unknown" with no percentage. Sending is
still allowed. The fix is one entry in `MODEL_CARDS` — no code change.

## Adding a model

Append one `ModelCard` to `MODEL_CARDS` in `src/inqtrix/model_cards.py` with its
`aliases` (include every provider id), `category`, `pricing`, capacities,
`reasoning_levels`, and `source_url`/`last_verified`. No other code changes are
needed. Claude facts come from the Anthropic models overview; OpenAI/Google from
the respective provider docs.

## Related docs

- [LLM calls](../architecture/llm-calls.md) — node/tier routing the cards layer on top of.
- [Agent config](agent-config.md) — `model_tier`, `model`, `effort` fields.
- [Webserver mode](../deployment/webserver-mode.md) — `/health` and `/v1/stacks` fields, `agent_overrides`.
- [React UI](../deployment/react-ui.md) — the model picker and token meter.
