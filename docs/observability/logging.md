# Logging

> Files: `src/inqtrix/logging_config.py`, `src/inqtrix/runtime_logging.py`, `src/inqtrix/server/app.py`

## Scope

How Inqtrix configures the `inqtrix` logger, which environment variables control it, how secrets are redacted, and how uvicorn and FastAPI logs are mirrored into the same file when running the HTTP server.

## Logger topology

The library writes to `logging.getLogger("inqtrix")`. `configure_logging(...)` builds a dedicated handler pipeline that does **not** propagate to the root logger (see Gotcha #14 in the internal notes). Tests that want to capture Inqtrix logs must attach `caplog.handler` directly to the `inqtrix` logger.

## Configuration helpers

Two public helpers live in `inqtrix.logging_config`:

- `configure_logging(*, enabled, level, console, file_path=None, force=True) -> bool` — configures the `inqtrix` logger: optional file handler (rotating), optional stderr console handler, secret-redaction filter. Returns `True` if a new configuration was applied. Pass `force=False` to skip reconfiguration when another caller already set up handlers (see ADR-WS-9).
- `build_uvicorn_log_config(log_file: Path, web_level: str = "INFO") -> dict` — produces a `logging.config.dictConfig`-compatible dict that mirrors uvicorn's default stderr setup and additionally writes `uvicorn.error` and `uvicorn.access` into the same `log_file` as Inqtrix (see ADR-WS-10). Pass this to `uvicorn.run(app, log_config=...)`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `INQTRIX_LOG_ENABLED` | `false` | Master switch; when `false` the library only attaches a `NullHandler`. |
| `INQTRIX_LOG_LEVEL` | `INFO` | Any of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `INQTRIX_LOG_CONSOLE` | `true` | Mirror to stderr in addition to the file sink. |
| `INQTRIX_LOG_FILE` | *(unset)* | Full path for the file sink; if unset, logs are stderr-only. |
| `INQTRIX_LOG_WEB_LEVEL` | `INFO` | Used by `build_uvicorn_log_config` for uvicorn/FastAPI logs. |
| `INQTRIX_LOG_INCLUDE_WEB` | `true` | Opt-out for uvicorn mirroring. |

The example webserver scripts (`examples/webserver_stacks/*.py`) read these variables, call `configure_logging(...)` once at startup, and pass `log_config=build_uvicorn_log_config(...)` to `uvicorn.run` so the resulting log file holds Inqtrix lifecycle lines, uvicorn startup/shutdown, and `uvicorn.access` request lines in one place.

## Secret redaction

Every handler attached by `configure_logging(...)` includes `_RedactSecretsFilter` from `runtime_logging.py`. The filter:

- Substitutes `Authorization: Bearer ...` headers with `Authorization: Bearer <redacted>`.
- Redacts `api_key=`, `api-key:`, and `key=` values in URLs and JSON bodies.
- Uses `sanitize_log_message(message)` from `runtime_logging.py` as the single source of truth; call that helper whenever you manually build a log line that may contain a secret.

The same filter is reused by the uvicorn mirror so access logs do not leak tokens.

## Fallback markers visibility

"No Silent Fallbacks" (internal Design Principle 1) requires every fallback path to emit both a `log.warning(...)` and an iteration-log marker. The following markers are the ones operators rely on to reconstruct a run:

- `_classify_fallback`, `_plan_fallback`, `_evaluate_fallback`
- `_confidence_parsed`, `_evidence_consistency_parsed`, `_evidence_sufficiency_parsed`
- `_claim_extraction_fallback`
- `Run cancelled by client disconnect`

See [Iteration log](iteration-log.md) for the structured marker view and [Debugging runs](debugging-runs.md) for the typical recovery flows.

## The `force=False` rule for the server

`create_app(...)` and `create_multi_stack_app(...)` call `configure_logging(..., force=False)`. This preserves the example-script configuration when a user starts uvicorn from a script that already set `INQTRIX_LOG_FILE`. If you build your own bootstrap path, preserve that invariant — see Gotcha #22 in the internal notes and [ADR-WS-9].

## Related docs

- [Progress events](progress-events.md)
- [Iteration log](iteration-log.md)
- [Debugging runs](debugging-runs.md)
- [Web server mode](../deployment/webserver-mode.md)
