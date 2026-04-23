# First research run

## Scope

The shortest viable path from a freshly installed repo to a real research answer. Two paths: library and HTTP. Pick one based on how you intend to integrate Inqtrix.

## Prerequisites

- Editable install working (see [Installation](installation.md)).
- A local `.env` with at least one LLM key and one search key. Copy the template:

  ```bash
  cp .env.example .env
  # edit .env
  ```

- For production-style provider combinations (Azure, Bedrock), see [Providers overview](../providers/overview.md) first.

## Path A: Library, env-based

```dotenv
# .env
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
```

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")

print(result.answer)
print(f"Confidence: {result.metrics.confidence}/10")
print(f"Sources: {result.metrics.total_citations}")
print(f"Rounds: {result.metrics.rounds}")
```

Run with:

```bash
uv run python main.py
```

See [Library mode](../deployment/library-mode.md) for the explicit-providers variant and streaming.

## Path B: HTTP server

```bash
cp .env.example .env
# edit .env with provider credentials

uv run python -m inqtrix
```

In a second shell:

```bash
curl -N http://localhost:5100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "research-agent",
        "messages": [
            {"role": "user", "content": "Was ist der aktuelle Stand der GKV-Reform?"}
        ],
        "stream": true
    }'
```

The response is an OpenAI-compatible SSE stream: progress chunks first, then answer chunks, terminated by `data: [DONE]`. Pass `"include_progress": false` for answer-only SSE.

See [Web server mode](../deployment/webserver-mode.md) for authentication, per-request overrides, multi-stack serving, and SSE details.

## What a good answer looks like

A healthy run prints the final Markdown answer followed by a stats footer:

```
---
*18 Quellen · 9 Suchen · 3 Runden · 45s · Confidence 8/10*
```

If you see confidence stuck at 6–8 with several uncovered aspects, that is the aspect-coverage cap (see [Aspect coverage](../scoring-and-stopping/aspect-coverage.md)). If confidence stays at 1–4 across rounds, the loop will eventually trigger falsification or stagnation (see [Falsification](../scoring-and-stopping/falsification.md)).

## What to do when the answer looks wrong

- Turn on logging: `INQTRIX_LOG_ENABLED=true`, `INQTRIX_LOG_FILE=./logs/inqtrix.log`, `INQTRIX_LOG_LEVEL=INFO`. See [Logging](../observability/logging.md).
- Read the iteration log for the run (testing mode) — the markers explain every decision. See [Iteration log](../observability/iteration-log.md).
- Look for provider errors in the log (`AnthropicAPIError`, `AzureOpenAIAPIError`, `PerplexityAPIError`). See [Debugging runs](../observability/debugging-runs.md).
- Re-run with a different stack (for example `examples/webserver_stacks/bedrock_perplexity.py`) to isolate whether the problem is provider-specific.

## Next steps

- [Library mode](../deployment/library-mode.md) — the three library entry paths.
- [Web server mode](../deployment/webserver-mode.md) — the OpenAI-compatible HTTP surface.
- [Architecture overview](../architecture/overview.md) — understand what the agent is doing internally.

## Related docs

- [Overview](overview.md)
- [Installation](installation.md)
- [Providers overview](../providers/overview.md)
