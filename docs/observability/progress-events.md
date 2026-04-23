# Progress events

> Files: `src/inqtrix/runtime_logging.py` (`emit_progress`), `src/inqtrix/server/streaming.py`

## Scope

How Inqtrix surfaces per-step progress messages to callers — both the library (`agent.stream(...)`) and the HTTP server (Server-Sent Events). Progress events are a UX feature for humans and a debug aid for operators; they are not a structured analytics channel (see [Iteration log](iteration-log.md) for that).

## Mechanism

Every node calls `emit_progress(state, message)` at meaningful boundaries:

- `"Analysiere Frage..."` — start of classify.
- `"Plane Suchanfragen (Runde X/Y)..."` — start of plan.
- `"Durchsuche N Quellen (Runde X/Y)..."` — start of search.
- `"Bewerte Informationsqualitaet (nach Runde X/Y)..."` — start of evaluate.
- `"Formuliere Antwort (nach N Runden)..."` — start of answer.
- `"done"` — terminal marker at the end of `answer`.

The function writes both a log line (for operators) and a queue entry (for the streaming generator); the string is German by default because the user-facing UI is German-first (see the conventions note on product strings).

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

Include the flag `"include_progress": false` in the body to get only answer chunks; the separator and progress prefixes are then omitted. See [Web server mode](../deployment/webserver-mode.md) for the full API contract.

## Cancel interaction

The streaming generator on the server side races `progress_queue.get(timeout=0.3)` against a watcher task that calls `await request.receive()`. When the client disconnects, the watcher sets `cancel_event`, the generator exits cleanly, and the next node boundary raises `AgentCancelled`. Result: progress messages stop arriving within roughly one second of disconnect; the active provider call continues until its natural completion (see Gotcha #18 in the internal notes and [Web server mode](../deployment/webserver-mode.md)).

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
- [Iteration log](iteration-log.md)
- [Web server mode](../deployment/webserver-mode.md)
