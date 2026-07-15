# Public API layer

> Files: `agent.py`, `result.py`, `__init__.py`

## Scope

This page covers the typed library entry point and the cross-resource HTTP
contracts that applications must treat consistently. The library surface is
lazy-initialised and backwards-compatible. The v0.2 platform HTTP contract is
a hard cut: clients must use canonical UUID identities, direct shares, and
integer revisions rather than legacy subject or cached-grant fields.

## `ResearchAgent`

The main entry point. Wraps the internal `graph.run()` machinery behind a clean interface.

```python
from inqtrix import ResearchAgent, AgentConfig

agent = ResearchAgent(AgentConfig(max_rounds=3))
result = agent.research("Question")
```

### Lifecycle

This diagram answers: "What does the public object do before the internal graph
starts?" Rectangles are methods, cylinder-style nodes are constructed runtime
objects, and the last node is the public Pydantic result.

Conventional flowchart
```mermaid
flowchart TD
    A["ResearchAgent(config)"] --> B{"First .research() call?"}
    B -->|Yes| C["_ensure_initialised()"]
    C --> D["Build AgentSettings from AgentConfig"]
    C --> E["Create Providers<br/>(auto or custom)"]
    C --> F["Create Strategies<br/>(defaults + overrides)"]
    B -->|No| G["Reuse cached providers/strategies"]
    G --> H["graph.run(question, ...)"]
    C --> H
    H --> I["ResearchResult.from_raw(raw)"]
    I --> J["Return typed ResearchResult"]
```

Typed flowchart
```mermaid
flowchart TD
    A["fn ResearchAgent(config)"] --> B{"router: first .research() call?"}
    B -->|Yes| C["fn _ensure_initialised()"]
    C --> D[("data AgentSettings")]
    C --> E[("data ProviderContext")]
    C --> F[("data StrategyContext")]
    B -->|No| G[("data cached runtime objects")]
    G --> H["fn graph.run(question, ...)"]
    C --> H
    H --> Raw[("data raw result_state")]
    Raw --> I["fn ResearchResult.from_raw(raw)"]
    I --> J[("data ResearchResult")]
```

The public API hides the mutable `AgentState`. `graph.run()` returns a raw dict
for internal/parity use, then `ResearchResult.from_raw()` projects selected
fields into typed public models.

The agent is reusable across runs. A typical web server keeps a single `ResearchAgent` instance for the lifetime of the process (see [Web server mode](../deployment/webserver-mode.md)).

### Public methods

| Method | Purpose |
|--------|---------|
| `research(question, history=None, deadline=None)` | Blocking run; returns a typed `ResearchResult`. |
| `stream(question, *, include_progress=True, history=None, deadline=None)` | Generator that yields progress messages (optional) followed by answer chunks. Used for CLIs, SSE servers, and browser UIs. |

Both methods are thread-safe as long as a single agent instance is not invoked concurrently against the same cancel event. The HTTP server uses a semaphore for concurrency (see [Web server mode](../deployment/webserver-mode.md)).

## `AgentConfig`

Pydantic `BaseModel` holding all `ResearchAgent`-relevant configuration. It covers agent behaviour, model selection via provider constructors, timeouts, cache settings, and provider connection settings. Server-only deployment settings remain in `ServerSettings`.

```python
AgentConfig(
    llm=MyCustomLLM(),           # Optional: custom LLM
    search=CustomWebSearch(),    # Optional: custom search
    stop_criteria=FastStop(),    # Optional: custom strategy
    max_rounds=2,
    report_profile=ReportProfile.DEEP,
)
```

Fields set to `None` (providers, strategies) are auto-created from defaults on first use. Model names live on the provider constructors, not on `AgentConfig`. See [AgentConfig reference](../configuration/agent-config.md) for every field.

The current `AgentConfig` parametrizes the default iterative research
procedure. It does not yet expose `algorithm="..."` or a graph-topology field.

## `ResearchResult`

Pydantic model returned by `research()`:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Markdown-formatted answer |
| `metrics` | `ResearchMetrics` | Aggregated quality and performance metrics |
| `top_sources` | `list[Source]` | Sources ordered by answer-linked URLs first, then prompt-selected evidence URLs, then remaining discovered citations; tiers prefer run source records |
| `references` | `list[ReportReference]` | Exact source list rendered in the report's `## Referenzen` appendix, including label, URL, and tier |
| `top_claims` | `list[Claim]` | Key claims with verification status, evidence counts, primary-need flag, and source-tier breakdown |
| `execution` | `AgentExecution \| None` | Effective Agent Desk route, model/effort, source policy, consent reason, and actual source-tool counts; absent for ordinary research and legacy results |

See [Result schema](result-schema.md) for the full field list and the export helper (`to_export_payload`). `ResearchResult.from_raw()` bridges the internal state dict to the typed Pydantic model.

Typical library consumption:

```python
from inqtrix import AgentConfig, LiteLLM, PerplexitySearch, ResearchAgent

agent = ResearchAgent(AgentConfig(
    llm=LiteLLM(api_key="...", default_model="gpt-4o"),
    search=PerplexitySearch(api_key="...", model="sonar-pro"),
    max_rounds=3,
))

result = agent.research("What changed in EU AI Act implementation this month?")

print(result.answer)
print(result.metrics.confidence)
print(result.metrics.evidence_contract_status)
print([source.url for source in result.top_sources[:3]])
```

Use the public result when building applications. Inspect raw state only in
debugging or parity tooling, because internal ledger shapes can evolve more
quickly than the public model.

## Platform HTTP contract (v0.2)

### Identity and access annotations

`GET /api/auth/session` exposes the current local user as
`user.id` (`UUID` string). `GET /v1/users/search` returns that same `id` plus
display name and email. Admin, workspace-member, quota, sharing, session, PAT,
audit, and actor contracts all refer to this UUID; external OIDC/LDAP issuer
and subject values never cross into resource APIs.

Regular lists for runs, knowledge collections, prompt templates, and skills
combine owned and accepted-shared resources. Each record carries one
authoritative access annotation:

```json
{"access":{"mode":"shared","permission":"edit"}}
```

`mode` is `unscoped`, `owner`, or `shared`. `permission` is present only for a
shared record and is `view` or `edit`. The annotation explains the response;
it is not a reusable authorization token. The server re-evaluates access on
every subsequent read or mutation. Missing or unauthorized resources use the
same 404 response to avoid existence disclosure.

### Direct shares

`resource_shares` is a direct user-to-resource lifecycle for `run`,
`knowledge_collection`, `prompt_template`, and `skill_template`. Pending
shares grant no access. `POST /v1/shares` validates the complete invitee batch
before writing; duplicate or malformed recipients return 400 with zero writes,
and an already-active direct share returns 409. A recipient accepts with
`POST /v1/shares/{share_id}/accept`; the owner revokes, or the recipient
declines/leaves, with `DELETE /v1/shares/{share_id}`. A later re-share creates
a new id and requires new consent.

Only the resource owner manages shares. An accepted editor may perform the
resource-specific edit operations but may not delete or re-share the resource.
`GET /v1/shares/inbox` and `GET /v1/shares/mine` are lifecycle views; there are
no `/shared-with-me` or `/outgoing` resource-list endpoints. See
[Authentication modes](../deployment/auth-modes.md#canonical-identity-and-direct-sharing)
for request examples and the endpoint matrix.

### Revisions and imports

Share permission updates, prompt updates, and skill updates use mandatory
integer optimistic concurrency. The client sends `expected_revision`; the
server performs a compare-and-swap, increments `revision` on success, and
returns HTTP 409 with `current_revision` when another editor won. There is no
force-overwrite or timestamp fallback.

`POST /v1/runs/import` requires `source_run_id`, but that value is provenance
and an owner-scoped idempotency key only. The server always generates the
public `run_id`. While an imported run exists, re-importing the same
`source_run_id` for the same owner is idempotent; after retention removes it,
a later import receives a new server id. A historical share therefore cannot
attach to a newly imported report through client-controlled id reuse.

### Collection and file boundary

Sharing a knowledge collection grants access to its metadata and extracted,
indexed text according to `view`/`edit`. It does not share the uploader's
original binary. Files remain owner-bound and there is no generic file-share
contract. A shared editor may ingest extracted content into the collection;
that content becomes part of the collection even if the editor later leaves,
while the source binary remains owned by its uploader.

## Related docs

- [Configuration overview](../configuration/agent-config.md)
- [Result schema](result-schema.md)
- [Strategies](strategies.md)
- [Providers overview](../providers/overview.md)
