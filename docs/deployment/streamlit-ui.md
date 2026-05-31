# Streamlit UI

> Files: `webapp.py`, `inqtrix_webapp/client.py`, `inqtrix_webapp/translations.py`

## Scope

How to run and operate the bundled Streamlit chat UI. The UI is a prototype frontend for the HTTP server: it is useful for local operation, demos, and integration testing, but it is not a hardened multi-user product surface.

## Architecture

`webapp.py` is a pure HTTP consumer. It does not import the `inqtrix` package and does not read provider credentials. All model, search, security, and stack choices live on the server process.
This diagram answers: "Which process owns UI state and which process owns the
research algorithm?" The Streamlit process only sends HTTP requests; the server
owns providers, strategies, native runs, and graph execution.

```mermaid
flowchart LR
    UI["process: webapp.py<br/>Streamlit UI"] --> Client["fn inqtrix_webapp.client"]
    Client --> Health["HTTP GET /health"]
    Client --> Stacks["HTTP GET /v1/stacks<br/>multi-stack only"]
    Client --> Chat["HTTP POST /v1/chat/completions<br/>SSE or blocking"]
    Chat --> Server["process: Inqtrix HTTP server<br/>runs graph + providers"]
```

The UI discovers server capabilities, lets the user pick a stack when `/v1/stacks` is available, and streams progress plus answer chunks from `/v1/chat/completions`.

## Start it

Start a server first. The multi-stack example is the most convenient companion because the UI can discover all active stacks:

```bash
# Terminal 1
uv run python examples/webserver_stacks/multi_stack.py
```

Then install the UI extra and run Streamlit:

```bash
# Terminal 2
uv sync --extra ui
INQTRIX_WEBAPP_BASE_URL=http://localhost:5100 \
  uv run streamlit run webapp.py
```

`INQTRIX_WEBAPP_BASE_URL` is the primary startup path. Use an IP endpoint when the server runs on another host, for example:

```bash
INQTRIX_WEBAPP_BASE_URL=http://192.168.1.42:5100 \
  uv run streamlit run webapp.py
```

If the server has `INQTRIX_SERVER_API_KEY` set, pass the same token to the UI:

```bash
INQTRIX_WEBAPP_BASE_URL=http://localhost:5100 \
INQTRIX_WEBAPP_API_KEY=dev-secret-xxxxx \
  uv run streamlit run webapp.py
```

If `INQTRIX_WEBAPP_BASE_URL` is not set, the UI does not probe `localhost` implicitly. Instead, open the sidebar and enter a server URL manually, then click **Apply**. The **Load env** button restores the env value.

## Request controls

The composer controls map to the server-side `agent_overrides` whitelist:

| UI control | Request field | Behaviour |
|------------|---------------|-----------|
| Research mode | `report_profile` | `compact` or `deep`. |
| Effort | `max_rounds`, `min_rounds` | `Auto` omits both so server defaults apply; other values send a fixed pair. |
| Confidence | `confidence_stop` | Stop threshold from 1 to 10. |
| Time budget | `max_total_seconds` | Wall-clock deadline in seconds. |
| First-round breadth | `first_round_queries` | Number of broad queries in round 0. |
| Web search | `skip_search` | When web search is off, the UI sends `skip_search=true`; the server answers directly through the LLM without citations. |

Example body sent by the UI:

```json
{
  "model": "research-agent",
  "messages": [{"role": "user", "content": "Was ist der Stand der GKV-Reform?"}],
  "stream": true,
  "include_progress": true,
  "stack": "anthropic_perplexity",
  "agent_overrides": {
    "report_profile": "deep",
    "max_rounds": 4,
    "min_rounds": 2,
    "confidence_stop": 8,
    "max_total_seconds": 300,
    "first_round_queries": 6
  }
}
```

When the web-search toggle is off, the UI adds `"skip_search": true` to the same object.

## Operational notes

- `/health` and `/v1/models` are always queried without authentication; `/v1/stacks` is also open in multi-stack apps so the UI can render stack selection before asking for a token.
- Discovery calls are sent only after a server URL is configured (env or manual sidebar input); no implicit localhost scan/probe happens.
- Chat requests include `Authorization: Bearer ...` only when `INQTRIX_WEBAPP_API_KEY` is set or the user enters a token in the UI.
- Streamlit Stop closes the active request. The server cancels at the next LangGraph node boundary; an in-flight provider call may still complete before the backend run stops.
- The UI stores conversation state in Streamlit's page state. Restarting Streamlit clears the local chat view; the HTTP server does not retain research snapshots.

## Related docs

- [Web server mode](webserver-mode.md)
- [Security hardening](security-hardening.md)
- [Agent config](../configuration/agent-config.md)
- [Progress events](../observability/progress-events.md)
