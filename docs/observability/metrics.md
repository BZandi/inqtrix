# Metrics (`/metrics`)

## Scope

This page describes the API Prometheus endpoint, the per-process worker
exporter, and the private metrics exposed by the optional editor
collaboration service. It covers scrape boundaries and bounded-cardinality
series, not alert thresholds or a complete monitoring stack.

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
  workerPort: 9091       # sets INQTRIX_WORKER_METRICS_PORT + worker pod annotations (0 = off)
```

Set `podAnnotations: false` when you scrape via a `PodMonitor`/`ServiceMonitor`
CR instead of the classic annotation config.

Importable Grafana dashboards for all series live under
[`deploy/grafana/`](../../deploy/grafana/README.md).

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

### Call metrics (API and worker)

The shared call-metric series are fed at single chokepoints — the LLM/search
provider wrappers, the retrieval pipeline, and the worker loops — so every
feature (research, kernel, knowledge, chat, editor, indexing) reports through
the same definitions. `model` is normalized through the model-card catalog
and `feature` comes from the request context, never from user input.

Two synthetic `model` values exist by design: `unknown` when a provider
reports no model id, and `other` once more than 50 DIFFERENT non-catalog
model names appear in one process. The second is the cardinality guard —
per-request model overrides are free-form strings, so without a cap one
client could mint unlimited series. It logs a WARNING when it engages;
`other` dominating a panel means either an unregistered model in regular
use (add a model card) or override traffic worth investigating.

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `inqtrix_llm_requests_total` | counter | `provider`, `model`, `operation`, `outcome` | Every LLM call. `operation` is `chat`, `text_completion` (also covers structured calls), or `embeddings`; `outcome` is one of `success`, `timeout`, `cancelled`, `error`. |
| `inqtrix_llm_request_duration_seconds` | histogram | `provider`, `model`, `operation` | LLM call latency (buckets up to 600 s — the configured per-call ceiling). |
| `inqtrix_llm_tokens_total` | counter | `model`, `feature`, `token_type` | Prompt/completion tokens per feature — the aggregate twin of the per-user usage ledger. |
| `inqtrix_search_requests_total` / `inqtrix_search_request_duration_seconds` | counter / histogram | `provider`, `engine`, (`outcome`) | Web-search provider calls. |
| `inqtrix_retrieval_duration_seconds` | histogram | `step` | Knowledge retrieval stages (`hybrid_search`, `rerank`). |
| `inqtrix_run_duration_seconds` | histogram | `mode`, `outcome` | Worker execution **segments** per run mode — parked runs resume as fresh segments, so one deep run contributes several samples. Outcomes: `completed`, `failed`, `cancelled`, `parked`. Fenced-out attempts are not recorded (the winning attempt records the segment). Edge case: a park that the store resolves as an immediately-cancelled run still counts as `parked` — the segment did run up to the park attempt. |
| `inqtrix_run_queue_wait_seconds` | histogram | — | Time from enqueue to worker claim, native **runs only** and only for first deliveries — redelivered messages keep their original enqueue timestamp and would fold the prior attempt's runtime into the wait. |
| `inqtrix_worker_jobs_total` | counter | `loop`, `outcome` | Worker job terminations per loop (`runs`, `indexing`, `uploads`) and outcome (`terminal`, `parked`, `fenced`, `finalization_failed`); the uploads loop currently emits only `finalization_failed`. |
| `inqtrix_indexing_documents_total` | counter | `outcome` | Documents finishing an indexing pass (`completed`, `failed` — pauses/cancellations are not failures and stay uncounted). |

### Worker exporter

Worker processes serve their **own** registry (no pushgateway): set
`INQTRIX_WORKER_METRICS_PORT` to a port >0 together with
`INQTRIX_METRICS_ENABLED=true` and each worker exposes
`http://0.0.0.0:<port>/metrics` for that process. Port `0` (default) keeps
the worker exporter off; a missing `metrics` extra or an unbindable port
logs one WARNING/ERROR and stays off — never a worker crash. Scrape every
worker replica as its own target.

Unlike the FastAPI endpoint, the worker port has **no bearer gate of its
own** — `INQTRIX_SERVER_API_KEY` does not apply here. Keep it strictly
cluster-internal (NetworkPolicy/scrape allowlist), like the collaboration
service's private endpoint below.

The admission counter is the direct signal for the scaling ceilings: a rising
`reason="queue_full"` means the global queue (`RUN_QUEUE_MAX_SIZE`) is the
bottleneck, `reason="per_user_limit"` means a single user is hitting
`RUN_MAX_CONCURRENT_PER_USER`, and `reason="quota"` means monthly budgets are
biting. Pair the gauges (`queue_depth` vs `active`/`capacity`) with these to
tell "saturated" apart from "throttling one noisy tenant".

## Collaboration service metrics

The optional Node service always exposes Prometheus text at its private
`GET /metrics` endpoint. It is not routed through the public Python web
Service and has no bearer gate of its own, so restrict scraping to the private
service network. `INQTRIX_METRICS_ENABLED` controls the FastAPI endpoint only;
it does not mount or unmount the sidecar endpoint.

| Metric family | Meaning |
|---|---|
| `inqtrix_collaboration_active_connections`, `inqtrix_collaboration_rooms` | Current live transport and loaded-room counts. |
| `inqtrix_collaboration_document_queue_depth` | Serialized work waiting for one document/generation. |
| `inqtrix_collaboration_update_validation_seconds`, `inqtrix_collaboration_update_persistence_seconds`, `inqtrix_collaboration_durable_ack_seconds` | Validation, database commit, and end-to-end durable acknowledgement latency. |
| `inqtrix_collaboration_rejections_total`, `inqtrix_collaboration_websocket_rejections_total`, `inqtrix_collaboration_http_rejections_total` | Bounded rejection reasons for document, transport, and internal HTTP policy. |
| `inqtrix_collaboration_instance_ready`, `inqtrix_collaboration_instance_epoch`, `inqtrix_collaboration_instance_renew_failures_total` | Single-writer fencing state. Readiness becomes false when the active lease expires. |
| `inqtrix_collaboration_internal_api_seconds`, `inqtrix_collaboration_internal_api_errors_total`, `inqtrix_collaboration_http_request_seconds`, `inqtrix_collaboration_http_requests_total` | Node-to-FastAPI and FastAPI-to-Node operation health. |
| `inqtrix_collaboration_awareness_scratch_states_removed_total`, `inqtrix_collaboration_awareness_dropped_total` | Hocuspocus scratch-state normalization and intentionally sampled presence traffic. A drop in scratch-state removals after a dependency update is the signal to revalidate and remove the compatibility adapter. |
| `inqtrix_collaboration_policy_poll_seconds`, `inqtrix_collaboration_policy_poll_errors_total`, `inqtrix_collaboration_policy_revalidations_total`, `inqtrix_collaboration_policy_revalidation_timeouts_total`, `inqtrix_collaboration_policy_resets_total` | Revocation-feed polling and connection revalidation. Alert on polling failures separately because readiness currently keys on the fencing lease. |
| `inqtrix_collaboration_snapshots_total`, `inqtrix_collaboration_snapshot_errors_total`, `inqtrix_collaboration_compaction_runs_total`, `inqtrix_collaboration_compaction_pruned_total` | Snapshot and retention maintenance outcomes. |

No series labels a user, room, document, token, comment, or update body. Use
document metadata (`persisted_sequence` and `projection_sequence`) for targeted
projection-lag diagnosis instead of adding document IDs to Prometheus.

## Readiness vs. liveness

`/metrics` is for scraping, not health checks. Kubernetes probes use:

- `/health` — liveness/startup (process up; unauthenticated).
- `/readyz` — readiness: database `SELECT 1` and the queue `PING` must pass
  (503 when either is down); a dead vector store degrades to `200` with a
  `degraded` body, since research/chat/files still work.

The collaboration process has separate private probes: `/health/live` means
the Node process is running, while `/health/ready` additionally requires its
active single-writer fencing lease.

## Related docs

- [Tracing legend](tracing-legend.md) — which span/observation type a step becomes, and which attributes are mandatory

- [Deploy editor collaboration](../deployment/editor-collaboration.md)
- [Logging](logging.md)
- [Security hardening](../deployment/security-hardening.md)
- [Runbooks](../deployment/runbooks.md)
