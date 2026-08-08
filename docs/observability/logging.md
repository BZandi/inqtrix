# Logging

> Files: `src/inqtrix/logging_config.py`, `src/inqtrix/runtime_logging.py`, `src/inqtrix/server/app.py`

## Scope

How Inqtrix configures the `inqtrix` logger, which environment variables control it, how secrets are redacted, and how uvicorn and FastAPI logs are mirrored into the same file when running the HTTP server.

## Logger topology

The library writes to `logging.getLogger("inqtrix")`. `configure_logging(...)` builds a dedicated handler pipeline that does **not** propagate to the root logger. Tests that want to capture Inqtrix logs must attach `caplog.handler` directly to the `inqtrix` logger.

## Configuration helpers

Two public helpers live in `inqtrix.logging_config`:

- `configure_logging(*, enabled, level, log_dir="logs", console=False, force=True, json_format=False) -> Path | None` — configures the `inqtrix` logger: optional file handler, optional console handler, secret-redaction filter. Pass `force=False` to skip reconfiguration when another caller already set up handlers. With `json_format=True` the console handler carries the full `level` (stdout is the canonical machine-readable sink); in text mode it stays at `WARNING`.
- `build_uvicorn_log_config(log_file: Path | str | None, web_level: str = "INFO", json_format: bool = False) -> dict` — produces a `logging.config.dictConfig`-compatible dict that mirrors uvicorn's default stderr/stdout setup and additionally writes `uvicorn.error` and `uvicorn.access` into the same `log_file` as Inqtrix. Pass this to `uvicorn.run(app, log_config=...)`. **Pass the SAME `json_format` value as `configure_logging`** — uvicorn builds its handlers from its own dictConfig, so a mismatch mixes JSON and text lines on one stream.

Structured runtime events are emitted by `inqtrix.runtime_logging.emit_runtime_event(...)`. It is not a second logging system and it does not create a separate JSON file. Iteration events first enter the protected run audit; the same logger receives a fail-closed operational projection containing only identifiers, lifecycle/status fields, models, counters, usage and timings. Exact queries, provider prose, snippets, claim/evidence text, prompt views and URLs are not mirrored into the timestamped `logs/inqtrix_*.log` file.

Minimal library setup:

```python
from inqtrix import configure_logging

log_file = configure_logging(
    enabled=True,
    level="INFO",
    log_dir="logs",
    console=False,
)
```

Server bootstrap with uvicorn mirroring:

```python
from pathlib import Path
import uvicorn

from inqtrix.logging_config import build_uvicorn_log_config, configure_logging
from inqtrix.server.app import create_app

log_file = configure_logging(enabled=True, level="INFO", console=False, log_dir="logs")

uvicorn.run(
    create_app(),
    host="0.0.0.0",
    port=5100,
    log_config=build_uvicorn_log_config(log_file),
)
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `INQTRIX_LOG_ENABLED` | `false` | Master switch; when `false` the library only attaches a `NullHandler`. |
| `INQTRIX_LOG_LEVEL` | `INFO` | Any of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `INQTRIX_LOG_CONSOLE` | `false` | Mirror WARNING+ records from the `inqtrix` logger to stderr in addition to any file sink. |
| `INQTRIX_LOG_WEB_LEVEL` | `INFO` | Used by `build_uvicorn_log_config` for uvicorn/FastAPI logs. |
| `INQTRIX_LOG_INCLUDE_WEB` | `true` | Opt-out for uvicorn mirroring. |
| `OBSERVABILITY_PROFILE` | `summary` | `summary`, `debug`, or `forensic`. `forensic` produces source/citation/claim/answer lineage in the protected run audit and an operational, content-minimized log projection; `debug` is currently reserved for future mid-level detail. |
| `INQTRIX_LOG_FORMAT` | `text` | Line shape. `text` keeps the historical pipe format byte-identical; `json` renders one machine-readable object per line — for the `inqtrix` logger AND the mirrored uvicorn/FastAPI loggers. Authoritative definition: [Settings and environment variables](../configuration/settings-and-env.md#process-level-variables-outside-settings). |

The example webserver scripts (`examples/webserver_stacks/*.py`) read these variables, call `configure_logging(...)` once at startup, and pass `log_config=build_uvicorn_log_config(...)` to `uvicorn.run` so the timestamped file under `logs/` holds Inqtrix lifecycle lines, uvicorn startup/shutdown, and `uvicorn.access` request lines in one place.

## JSON log lines and correlation fields

With `INQTRIX_LOG_FORMAT=json`, every record becomes one JSON object with a
fixed envelope: `ts` (RFC3339 UTC), `level`, `logger`, `event`, `message`,
`thread`, plus `exc` for exceptions and the correlation fields bound by the
request/worker context: `request_id` (accepted from a valid incoming
`X-Request-ID` or generated, and echoed on every response), `run_id`, `user`
(the STABLE pseudonym `usr_<hex16>` — see `INQTRIX_PSEUDONYM_PEPPER`),
`workspace`, and `tenant`. Once tracing is enabled, `trace_id`/`span_id`
join the envelope so a log line and its trace correlate directly.

Redaction is format-independent: the same `_RedactSecretsFilter` runs in
front of both formatters, and structured field values are additionally
scrubbed. Follow one user across api and worker containers with:

```bash
docker compose logs api worker | jq -c 'select(.user == "usr_ab12cd34ef56aa77")'
```

In JSON mode the console sink carries the FULL `INQTRIX_LOG_LEVEL` (stdout is
the canonical machine-readable sink for container runtimes), so this works
without a file sink. In text mode the console keeps its historical
`WARNING`-and-above mirror.

## How the switches interact

There are three separate concerns:

- `INQTRIX_LOG_ENABLED` decides whether a persistent file under `logs/` is created.
- `INQTRIX_LOG_LEVEL` decides which records reach that file.
- `OBSERVABILITY_PROFILE` decides whether detailed forensic lineage events are produced at all. Only `forensic` enables `query_record`, `query_summary`, `source_record`, `provider_citation_record`, `evidence_record`, `claim_record`, `claim_merge`, `evidence_verification_projection`, `evidence_selection`, `answer_prompt_inputs`, `answer_section`, `answer_claim_binding`, `answer_sentence_audit`, and related events.

**Lineage events live on the trace, not in the log.** They are attached to the
span of the step that produced them, so the waterfall shows them in place
instead of in a separate stream that has to be joined by hand. Recording them
therefore needs a trace sink:

```bash
OBSERVABILITY_PROFILE=forensic       # how deep
INQTRIX_TRACING=file                 # where to  (or otlp)
```

`forensic` with `INQTRIX_TRACING=off` or `local`, or with a sample rate below
`1.0`, records only part of what was asked for — the process warns about both
combinations at startup. See [Tracing legend](tracing-legend.md) for what an
event carries and [Debugging runs](debugging-runs.md) for the drill-down path.

The log keeps what it has always been good at: lifecycle lines, warnings,
fallback markers and the operational per-round trace, each carrying the
correlation fields so a log line and its trace join on `trace_id`.

Do not treat the container/file log as the durable evidence audit. It is intentionally insufficient to reconstruct private content. Authorized run artifacts (including `web_search_ledger`, result/artifact records, or `iteration_logs` in testing/parity mode) carry the reconstructable, redacted chain; compact INFO/DEBUG lines are only operational signposts that can be joined through stable IDs.

`TESTING_MODE=true` is separate from file logging. It exposes the HTTP `/v1/test/run` endpoint and attaches the sanitized `iteration_logs` list to test/parity results. With `OBSERVABILITY_PROFILE=forensic`, those exported iteration logs include the same forensic events even when file logging is disabled; never enable testing mode in production.

Common recipes:

```bash
# Persistent summary log: lifecycle, warnings, fallback markers.
INQTRIX_LOG_ENABLED=true
INQTRIX_LOG_LEVEL=INFO
```

```bash
# Algorithm debug log: per-round trace, no forensic lineage.
INQTRIX_LOG_ENABLED=true
INQTRIX_LOG_LEVEL=DEBUG
```

```bash
# Forensic lineage (on the trace, see above) plus a persistent log file.
INQTRIX_LOG_ENABLED=true
INQTRIX_LOG_LEVEL=DEBUG
OBSERVABILITY_PROFILE=forensic
```

```bash
# Export sanitized iteration_logs for parity/testing tools.
TESTING_MODE=true
OBSERVABILITY_PROFILE=forensic
```

## Secret redaction

Every handler attached by `configure_logging(...)` includes `_RedactSecretsFilter` from `logging_config.py`. Structured events also pass through an allowlist-style serializer before the handler filter sees them. Together they provide two layers of protection:

- Do not emit blocked structured fields such as `headers`, `authorization`, `request_kwargs`, `request_body`, `raw_response`, `api_key`, `client_secret`, `password`, `secret`, or credential objects.
- Redact credential-bearing URL query parameters such as `api_key`, `token`, `sig`, `signature`, `client_secret`, and `password`. Iteration-event console projections omit every URL, including benign and private URLs.
- Redact bearer tokens, `sk-*`/`pplx-*` API-key-like strings, and AWS access/session tokens in nested dict/list payloads.
- Keep raw provider request bodies, headers, SDK responses, and client configuration out of the standard and forensic logs.
- Authorization-denial warnings never contain raw user, tenant, recipient, or
  resource identifiers. They expose only bounded categorical fields and
  domain-separated HMAC references (`actor_ref`, `tenant_ref`,
  `resource_ref`). With `INQTRIX_PSEUDONYM_PEPPER` configured, references are
  stable across processes and restarts; without it they deliberately fall
  back to a process-local key and startup warns once.
- Authentication warnings that need to correlate one browser session use the
  same helper with the `ses` namespace. A full session identifier, or any
  prefix sliced from it, is still bearer-derived credential material and must
  never be logged.
- Durable `auth.logout` audit rows use the same `ses_<hex16>` namespace at
  their domain writer. The revoked browser credential itself belongs in
  neither `audit_log` nor its admin list and CSV/NDJSON exports. Historical
  rows are sanitized set-wise inside PostgreSQL with a domain-separated
  SHA-256 derivation; migration output and errors expose only counts.

The same filter is reused by the uvicorn mirror so access logs do not leak tokens.

## Evidence lineage events

Forensic mode persists the following evidence-specific events in the protected audit. Their ordinary log projection contains only non-content operational fields:

- `evidence_record`: one source/citation-level EvidenceRecord with source
  passages, source snippets, and raw claim supports.
- `query_summary`: per-query summary with extracted/kept claim counts,
  `claim_extraction_valid_empty`, and rendered evidence-context size. This is
  the fastest way to distinguish valid empty claims from parse/API `ALGO-FAIL`.
- `algorithm_failure`: visible core-path failure record. In forensic or deep
  report runs, blocking failures prevent normal final-report synthesis and
  produce a diagnostic answer instead.
- `evidence_verification_projection`: aggregate count of claim verification projected back onto EvidenceRecords.
- `evidence_overview_render`: EvidenceOverview render statistics for answer
  synthesis (`rendered_record_count`, `omitted_record_count`, visible
  `allowed_urls` size, visible label count, `label_by_evidence_id` size,
  verification-label mix), plus the `evidence_depth_gap` diagnostic.
- `answer_prompt_diagnostics`: compact answer-prompt density counters emitted in
  testing mode or forensic observability. Use it to compare EvidenceRecords,
  report-eligible records, rendered vs omitted records, evidence overview char
  length, visible-label counts, and allowed-URL counts without dumping the full
  prompt in normal logs.
- `answer_sentence_audit`: answer-side audit row that marks `matched`, `source_context`, or `unknown_citation`.

All protected audit payloads go through `sanitize_event_payload(...)`. Source passages and provider excerpts are bounded and redacted there; the subsequent console projection removes all such content before logging. Raw provider responses, headers, request bodies and credentials enter neither sink.

For live provider triage after a run, use either:

```bash
# uv
uv run python scripts/debug_research_log.py logs/inqtrix_YYYYMMDD_HHMMSS.log

# standard Python/pip
python -m pip install -e .
python scripts/debug_research_log.py logs/inqtrix_YYYYMMDD_HHMMSS.log
```

The script prints provider, source, EvidenceRecord, claim-extraction,
`ALGO-FAIL`, bundle, answer input, prompt evidence, appendix, and
final-confidence counts without echoing raw URLs or secrets.

## Fallback markers visibility

"No Silent Fallbacks" (internal Design Principle 1) requires every fallback path to emit all three of: a `log.warning(...)` line, a progress event on the user-visible queue, and a structured iteration-log marker. The following markers are the ones operators rely on to reconstruct a run:

- `_classify_fallback`, `_plan_fallback`, `_evaluate_fallback`
- `_confidence_parsed`, `_evidence_consistency_parsed`, `_evidence_sufficiency_parsed`
- `_claim_extraction_fallback` — claim extraction failed for at least one source. A full-round failure also emits an `algorithm_failure` event with `phase=claim_extraction`.
- `algorithm_report_blocked` — answer synthesis was blocked because a core evidence-path failure would otherwise produce a normal-looking report from unaudited source context.
- `_answer_fallback` plus `_answer_fallback_kind` / `_answer_fallback_reason` — answer synthesis took a fallback path (`timeout`, `no_fallback_model`, or `fallback_model_failed`). The answer body starts with a visible `> [!WARNING] Antwort-Synthese-Fallback aktiv` block so the degradation is also surfaced in the rendered output, not only in operator logs.
- `_stop_reason`
- `Run cancelled by client disconnect`

See [Iteration log](iteration-log.md) for the structured marker view and [Debugging runs](debugging-runs.md) for the typical recovery flows.

## The `force=False` rule for the server

`create_app(...)` and `create_multi_stack_app(...)` call `configure_logging(..., force=False)`. This preserves the example-script configuration when a user starts uvicorn from a script that already configured the `inqtrix` file handler. If you build your own bootstrap path, preserve that invariant so your file handler is not replaced by a later server default.

## Related

- [Tracing legend](tracing-legend.md) — which span/observation type a step becomes, and which attributes are mandatory docs

- [Progress events](progress-events.md)
- [Iteration log](iteration-log.md)
- [Evidence pipeline](../architecture/evidence-pipeline.md)
- [Debugging runs](debugging-runs.md)
- [Forensic cookbook](forensic-cookbook.md)
- [Web server mode](../deployment/webserver-mode.md)
