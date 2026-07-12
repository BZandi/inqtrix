# Metrics (`/metrics`)

Inqtrix can expose a Prometheus scrape endpoint for the API process. It is
**off by default** and mounts only when both are true:

- `INQTRIX_METRICS_ENABLED=true`, and
- the image was built with the optional `metrics` extra
  (`prometheus-client`). The bundled stack image
  ([`deploy/docker/Dockerfile.api`](../../deploy/docker/Dockerfile.api)) bakes
  it in; a custom build needs `--extra metrics`.

If the flag is set but the extra is missing, the server logs a WARNING and
leaves `/metrics` unmounted rather than failing to start.

## Access

When `INQTRIX_SERVER_API_KEY` is set, `/metrics` requires that same Bearer
token (constant-time compare, like the other gated routes). With no API key it
is unauthenticated — it is never exposed through the web ingress, so keep the
scrape path cluster-internal (a `NetworkPolicy`/ingress allowlist, or a
`PodMonitor`/`ServiceMonitor` scraping the pod directly).

The Helm chart wires this with one toggle:

```yaml
metrics:
  enabled: true          # sets INQTRIX_METRICS_ENABLED
  podAnnotations: true   # emits prometheus.io/{scrape,port,path} on the api pods
```

Set `podAnnotations: false` when you scrape via a `PodMonitor`/`ServiceMonitor`
CR instead of the classic annotation config.

## Series

All series are **bounded cardinality** — there are no run-id, subject, or
session labels, and HTTP paths are the matched route *template*
(`/v1/runs/{run_id}`), never the expanded id.

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `inqtrix_run_queue_depth` | gauge | — | Native runs waiting in the queue (`status=queued`). Read from the run store at scrape time. |
| `inqtrix_run_active` | gauge | — | Native runs currently executing (`status=running`). |
| `inqtrix_run_capacity` | gauge | — | In-process concurrent run capacity. Emitted **only** for the in-memory run store; with the durable (Postgres + worker) backend the worker fleet owns the slots, so this series is absent. |
| `inqtrix_run_admission_rejected_total` | counter | `reason` | Run/quota admissions rejected before entering the queue. `reason` is one of `queue_full` (global queue full), `per_user_limit` (per-user fairness cap), `quota` (monthly quota exhausted). |
| `inqtrix_http_requests_total` | counter | `method`, `handler`, `status` | HTTP requests by route template and status. |
| `inqtrix_http_request_duration_seconds` | histogram | `method`, `handler` | Request latency by route template. |

The admission counter is the direct signal for the scaling ceilings: a rising
`reason="queue_full"` means the global queue (`RUN_QUEUE_MAX_SIZE`) is the
bottleneck, `reason="per_user_limit"` means a single user is hitting
`RUN_MAX_CONCURRENT_PER_USER`, and `reason="quota"` means monthly budgets are
biting. Pair the gauges (`queue_depth` vs `active`/`capacity`) with these to
tell "saturated" apart from "throttling one noisy tenant".

## Readiness vs. liveness

`/metrics` is for scraping, not health checks. Kubernetes probes use:

- `/health` — liveness/startup (process up; unauthenticated).
- `/readyz` — readiness: database `SELECT 1` and the queue `PING` must pass
  (503 when either is down); a dead vector store degrades to `200` with a
  `degraded` body, since research/chat/files still work.
