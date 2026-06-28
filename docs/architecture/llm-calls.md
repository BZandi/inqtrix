# LLM calls, model tiers, and reasoning effort

This page is the single source of truth for **which LLM call happens where in
the algorithm, what each call does, and how a model and reasoning effort are
chosen for it**. It is written so you can decide, per call site, which model
tier is appropriate.

Every call site resolves its model and reasoning effort through one central
router, [`inqtrix.model_routing`](../../src/inqtrix/model_routing.py). There are
no per-node special cases.

---

## 1. What each LLM call does, and what it sees

All prompt templates are German (see
[`prompts.py`](../../src/inqtrix/prompts.py)); the descriptions below are in
English.

| Call site | Concrete task | Prompt / template | Most important context fed in | Output + parsing | Volume |
|---|---|---|---|---|---|
| **classify** | Decides SEARCH vs. DIRECT; detects input language, optimal search language, recency (NONE/HOUR/DAY/WEEK/MONTH) and type (GENERAL/ACADEMIC/NEWS); **decomposes** the question into 1-3 sub-questions. | Inline prompt in [`nodes.py`](../../src/inqtrix/nodes.py) (no separate system prompt). | `question`, today's date. | `KEY: VALUE` lines (DECISION/LANGUAGE/SEARCH_LANGUAGE/RECENCY/TYPE) + a `SUB_QUESTIONS` JSON array. Regex + JSON parse with visible fallbacks. | 1× per run |
| **plan** | Generates N new search queries for the next round; chooses strategy (breadth vs. depth vs. falsification vs. cross-check). | Inline prompt + **conditional strategy fragments** (see below). | `question`, `sub_questions`, `required_aspects`, `uncovered_aspects`, `gaps` (previous round), prior `queries`, `round`, `search_language`, `competing_events`, `final_confidence`, `falsification_triggered`, evidence-ledger size. | Strict JSON array of strings. JSON parse with status + slot fallback. | 1× per round |
| **evaluate** | Scores confidence (1-10), names gaps, detects contradictions / competing events, estimates evidence consistency + sufficiency; drives the stop decision. | Inline prompt + `EVALUATE_FORMAT_SUFFIX`. | `question`, the **rendered evidence-ledger overview**, `required`/`uncovered_aspects` + coverage, `source_tier_counts`, consolidated claims + status counts, evidence-depth gap, previous-round `final_confidence`/`gaps`. | `KEY: VALUE` (STATUS/CONFIDENCE/GAPS/CONTRADICTIONS/COMPETING_EVENTS/EVIDENCE_*). Regex + stop-strategy heuristics + guardrails. | 1× per round |
| **answer** (per section) | Writes one markdown section of the final report with `[E*]` citations; iterates over 3-6 sections, each seeing the running report. | `build_answer_section_system_prompt(...)` + `build_answer_section_user_prompt(...)`; full system prompt with style/safety/citation rules. | `question`, the **full evidence overview** + allowed citations + label map, source/claim metrics, `required`/`uncovered_aspects`, `competing_events`, conversation `history`, completed headings, report-so-far summary, used evidence labels, the section spec. | Free markdown (`##`/`###`, `[E12]`). Normalisation + markdown repair + incompleteness detection. | 3-6× per run |
| **claim_extract** | Extracts checkable single claims per search hit (`claim_text`, `evidence_snippet`, `claim_type`, `polarity`, `needs_primary`, `provider_refs`, `published_date`). | `build_claim_extraction_prompt(max_claims)`. | `question`, the **search hit full text** (capped ~16k chars), normalised citations, source-ref map, `max_claims`. | JSON schema via `complete_structured` (with a JSON fallback). Reasoning follows the **fast** tier's effort (`tier_fast_effort`; unset/`none` by default → no reasoning). | 1× per search hit (5-50+) |
| **direct_chat** | Answers directly without research when classify returns DIRECT (the conversational path). | `_DIRECT_CHAT_SYSTEM_PROMPT` + a formatted prompt in [`graph.py`](../../src/inqtrix/graph.py). | `question`, conversation `history`. | Free text. | 1× per run (DIRECT only) |
| **knowledge_contextualize** | Rewrites a Knowledge Ask follow-up into a standalone retrieval query. | `build_knowledge_followup_context_prompt(question, history)`. | Current user question plus formatted conversation history; history is context only, not evidence. | Strict JSON object with `question`; provider/parse failures fall back loudly to the original question with `_knowledge_query_context_fallback`. | 0-1× per `mode=knowledge` run |
| **knowledge_decompose** | Splits a deep-profile knowledge retrieval query into independent sub-queries. | `build_knowledge_decompose_prompt(question)`. | Standalone retrieval query. | Strict JSON array; parse failure degrades to no split with a visible marker. | 0-1× per `mode=knowledge` run |
| **knowledge_gate** | Judges whether retrieved `[K#]` evidence is sufficient and may propose a rewritten query. | `build_knowledge_gate_prompt(...)`. | Standalone retrieval query and rendered evidence block. | Strict JSON object; parse failure fails open with `_knowledge_gate_fallback`. | 0-N× per `mode=knowledge` run |
| **knowledge_answer** | Synthesises the cited Knowledge answer. | `build_knowledge_answer_prompt(...)`. | Original user question, optional conversation history, and current `[K#]` evidence excerpts. | Markdown answer, optionally preceded by a quote block verified by `grounding.py`. | 0-1× per `mode=knowledge` run |

**plan — conditional strategy fragments** (appended depending on state):
perspective-diversity (STORM), deep-review, alternative-hypothesis (round 1),
competing-events, reformulation (round ≥ 2 and confidence ≤ 4), falsification,
answer-contract, cross-check, query-slots, language-directive. The *strategy*
lives largely in the prompt, not the model — which is why a capable
non-thinking model is enough for `plan` (see the tier rationale below).

> Not part of the research flow: the parity diagnostic tool
> ([`parity/_analysis.py`](../../src/inqtrix/parity/_analysis.py)) uses
> `reasoning_model` directly and is out of scope for the tier system.

---

## 2. The three tiers and how nodes map to them

A node is mapped to a tier by the hard-coded
`NODE_TIER_ASSIGNMENT` constant. Per-node model overrides cover edge cases.

| Call site | Default tier | Recommended effort | Rationale |
|---|---|---|---|
| **answer** | **high** | reasoning **on** | Final synthesis, largest quality lever, large context. The one path where extended thinking clearly pays off. |
| **plan** | **mid** | none | Strategy is scaffolded by the prompt; a capable non-thinking model suffices. Thinking would add latency to every round for marginal gain. |
| **evaluate** | **mid** | none | Confidence calibration; mid is capable enough and heuristics/guardrails carry much of the work. |
| **classify** | **fast** | none | Classification + decomposition; small models are reliable here; 1× per run. |
| **claim_extract** | **fast** | none | Highest volume (1× per hit) — the biggest cost lever; strict-schema extraction where reasoning hurts. |
| **direct_chat** | **mid** (request-selectable) | none | Single conversational call; mid balances quality and latency. Selectable per request via `model_tier`. |
| **knowledge_contextualize** | **fast** | none | Cheap query rewrite for follow-up resolution; the answer still relies on retrieved evidence. |
| **knowledge_decompose** | **fast** | none | Small structured rewrite task; deep-profile only. |
| **knowledge_gate** | **fast** | none | Repeated structured sufficiency checks; cost-sensitive and prompt-scaffolded. |
| **knowledge_answer** | **high** | reasoning **on** | Final cited synthesis over retrieved evidence, largest quality lever in Knowledge mode. |

Default tier effort is **unset** for every tier, so out of the box tiers differ
only by *model* and nothing gains implicit reasoning. Turn reasoning on
deliberately, e.g. `TIER_HIGH_EFFORT=medium` for `answer`.

---

## 3. Reasoning effort (provider-neutral)

`reasoning_effort` is a provider-neutral token: `""` (unset), `none`, `minimal`,
`low`, `medium`, `high`, `xhigh`. `""` means *inherit the provider's constructor
default*; an explicit `none` means *force reasoning off for this call*. That
sentinel difference is what keeps the change backward compatible.

| Provider | Mapping | Notes |
|---|---|---|
| **Anthropic / Bedrock** | `none` → no thinking. A graded level → `thinking={"type":"adaptive"}` + `output_config.effort=<level>`. | For Opus 4.7 adaptive thinking is the on-switch and `effort` an orthogonal cap. Haiku class: adaptive only (no `effort`), with a visible warning. `minimal` is mapped to the nearest supported level. |
| **Azure (OpenAI)** | `none`/unknown → omit `reasoning_effort` (force off, 400-safe). A graded level → top-level `reasoning_effort=<level>` and `temperature` is dropped (Azure rejects both together). | A graded level on a non-reasoning deployment surfaces loudly as an Azure HTTP 400. |
| **LiteLLM** | Accepted but **not mapped** yet (deferred). A real effort logs a one-time warning. | Tier *model* routing still applies. Use Anthropic/Bedrock/Azure for reasoning control. |

Unsupported levels are downgraded to `none` with a visible warning (no silent
fallbacks).

---

## 4. Configuration — three layers, simple to fine-grained

**Layer 1 — one model for everything** (unchanged historical default):

```
REASONING_MODEL=claude-sonnet-4-6
```

**Layer 2 — three tiers** (constructor; the headline feature):

```python
# Anthropic — thinking only on the high tier
AnthropicLLM(
    api_key="sk-ant-...",
    tier_high_model="claude-opus-4-7",   tier_high_effort="medium",
    tier_mid_model="claude-sonnet-4-6",  tier_mid_effort="none",
    tier_fast_model="claude-haiku-4-5",  tier_fast_effort="none",
)

# Azure / OpenAI (GPT-5.4 family)
AzureOpenAILLM(
    azure_endpoint="https://...",
    tier_high_model="gpt-5.4",       tier_high_effort="medium",
    tier_mid_model="gpt-5.4",        tier_mid_effort="none",
    tier_fast_model="gpt-5.4-mini",  tier_fast_effort="none",
)
```

**Layer 3 — per-node model override** (beats the tier for one node):

```python
AnthropicLLM(
    api_key="sk-ant-...",
    tier_high_model="claude-opus-4-7", tier_mid_model="claude-sonnet-4-6",
    tier_fast_model="claude-haiku-4-5", tier_high_effort="medium",
    plan_model="claude-opus-4-7",  # pin plan to opus regardless of its tier
)
```

**Server mode (env):**

```
TIER_HIGH_MODEL=claude-opus-4-7
TIER_HIGH_EFFORT=medium
TIER_MID_MODEL=claude-sonnet-4-6
TIER_FAST_MODEL=claude-haiku-4-5
# optional per-node model override:
ANSWER_MODEL=claude-opus-4-7
```

### Resolution order

```
model(node):  <node>_model  ->  tier_<requested_tier OR default_tier>_model  ->  reasoning_model
effort(node): tier_<requested_tier OR default_tier>_effort  ->  "" (inherit provider default)
```

`requested_tier` comes from the per-run `model_tier` setting (see below). With
nothing configured, every node uses `reasoning_model` and inherits the
provider's default effort — byte-for-byte the historical behaviour.

Reasoning effort is configured **per tier**, not per node. Per-node granularity
exists for the *model* only.

---

## 5. Per-run tier selection (`model_tier`) and the chat path

`AgentSettings.model_tier` (env `MODEL_TIER`, or the per-request override
`model_tier`) selects a tier for the whole run, replacing the default per-node
assignment. An explicit per-node model override still wins.

The HTTP override whitelist accepts `model_tier` (`"high"`/`"mid"`/`"fast"`).
Combined with `skip_search=true`, this lets a caller pick the **model class for
a direct-chat answer**:

```jsonc
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "..."}],
  "agent_overrides": {"skip_search": true, "model_tier": "fast"}
}
```

### React chat model switcher

The React Research Desk uses the same per-request hook for the Chat mode model
picker. Discovery surfaces expose a `chat_model_options` array with the
`direct_chat` resolution for each selectable tier (`high`, `mid`, `fast`):

```json
{
  "tier": "mid",
  "node": "direct_chat",
  "model": "claude-sonnet-4-6",
  "effort": "none",
  "model_source": "tier:mid",
  "effort_source": "tier:mid",
  "requested_tier": "mid"
}
```

The UI persists the selected tier as project UI state and sends
`agent_overrides.model_tier` on each `/v1/chat/completions` request. Specific
model names remain operator-owned configuration; browser clients can only choose
among the advertised tiers. Discovery is stack-scoped: on multi-stack servers
the UI must read the selected stack's `models.chat_model_options` and must not
silently fall back to `/health`, because `/health` describes the default stack.

---

## 6. Observability

Every resolved node emits `inqtrix.node.model_resolution` into the native run
event stream (and therefore the React live view when a sink is attached). With
the forensic observability profile active, the same descriptor is also written
as a `node_model_resolution` iteration-log event:

```json
{
  "event": "node_model_resolution",
  "node": "answer",
  "model": "claude-opus-4-7",
  "tier": "high",
  "effort": "medium",
  "model_source": "tier:high",
  "effort_source": "tier:high",
  "requested_tier": ""
}
```

If provider model metadata is missing, or resolution produces an empty model, the
runtime additionally emits a warning progress event and a
`node_model_resolution_warning` iteration-log marker. The latest descriptor for
each node is also kept in `state["node_model_resolutions"]` so terminal result
surfaces can report exactly what a run used.

`/health` and `/v1/stacks` expose resolver descriptors under `node_models` and
the Chat picker subset under `chat_model_options`. For OpenAI-compatible direct
chat, `/v1/chat/completions` adds `inqtrix.model_resolution` to non-streaming
responses, and the streaming path emits a metadata chunk with the same
`inqtrix.model_resolution` before answer tokens. The OpenAI-compatible top-level
`model` field remains `research-agent`; consumers that need the real provider
model should read the additive `inqtrix` block.
