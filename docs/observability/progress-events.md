# Progress events

> Files: `src/inqtrix/runtime_logging.py` (`emit_progress`), `src/inqtrix/server/streaming.py`

## Scope

How Inqtrix surfaces per-step progress messages to callers — both the library (`agent.stream(...)`) and the HTTP server (Server-Sent Events). Text progress events are a UX feature for humans and a debug aid for operators; they are not a structured analytics channel (see [Iteration log](iteration-log.md) for audit detail and [Run events](run-events.md) for native UI state snapshots).

## Mechanism

Every node calls `emit_progress(state, message)` at meaningful boundaries:

- `"Analysiere Frage..."` — start of classify.
- `"Analyseziele erkannt: N Teilfragen: ..."` — classify found high-level
  analysis targets; these are not a one-to-one list of planned search
  queries.
- `"Plane Suchanfragen (Runde X/Y)..."` — start of plan.
- `"N neue Suchanfragen generiert (aus X Analysezielen, Y Pflichtaspekten; ...)"` —
  plan generated concrete search queries from the question decomposition,
  required aspects, and active diversification strategies.
- `"ALGO-FAIL plan_query_generation: ...; verwende Fallback-Query"` — query
  planning could not produce a usable JSON query list and the original
  question is used as a visible fallback query.
- `"Durchsuche N Suchanfragen (Runde X/Y)..."` — start of search.
- `"Bewerte Informationsqualitaet (nach Runde X/Y)..."` — start of evaluate.
- `"Formuliere Antwort (nach N Runden)..."` — start of answer.
- `"done"` — terminal marker at the end of `answer`.

The function writes a queue entry for the legacy streaming generator and, when a native run event sink is attached, an `inqtrix.progress.message` event with a compact `snapshot`. The string is German by default because the user-facing UI is German-first (see the conventions note on user-facing strings).

## Library streaming

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
for chunk in agent.stream("Meine Frage"):
    print(chunk, end="", flush=True)
```

By default the stream yields progress messages first, then the answer chunks. Pass `include_progress=False` to get only the answer text chunks — useful when another program (not a human) consumes the stream.

```python
for chunk in agent.stream("Meine Frage", include_progress=False):
    print(chunk, end="", flush=True)
```

## HTTP streaming

When a client POSTs to `/v1/chat/completions` with `"stream": true`, the server wraps the same queue into SSE chunks in the OpenAI-compatible `data: {...}` format:

1. Progress chunks prefixed with `> Research Step: ...`.
2. Separator line `---`.
3. Answer chunks word-by-word.
4. Terminal `data: [DONE]`.

Include the flag `"include_progress": false` in the body to get only answer chunks; the separator and progress prefixes are then omitted. This flag affects the OpenAI-compatible chat stream only. Native `/v1/runs/{run_id}/events` still emits structured progress events because UI state depends on them. See [Web server mode](../deployment/webserver-mode.md) for the full API contract.

Progress blockquotes appear for the graph modes (`research`, `direct_llm`), which push coarse messages onto the progress queue. `knowledge` emits its detailed gate and grounding steps as structured native-run events and therefore streams the **answer only** on the chat-completions surface. This is intentional: streaming dispatches every mode through the `AlgorithmRegistry`, while the chat surface renders coarse progress only from the queue used by graph modes.

## Cancel interaction

The streaming generator on the server side races `progress_queue.get(timeout=0.3)` against a watcher task that calls `await request.receive()`. When the client disconnects, the watcher sets `cancel_event`, the generator exits cleanly, and the next node boundary raises `AgentCancelled`. Result: progress messages stop arriving within roughly one second of disconnect; the active provider call continues until its natural completion (see [Web server mode](../deployment/webserver-mode.md)).

## Extending progress messages

Adding a new message in a custom node is trivial:

```python
from inqtrix.runtime_logging import emit_progress


def fact_check_node(s: dict, *, providers, strategies, settings) -> dict:
    emit_progress(s, "Pruefe Faktenlage...")
    ...
```

Keep messages short, in the target UI language, and avoid embedding confidence numbers or partial answers — those belong in the iteration log and the final answer respectively.

## Related docs

- [Logging](logging.md)
- [Run events](run-events.md)
- [Iteration log](iteration-log.md)
- [Web server mode](../deployment/webserver-mode.md)
