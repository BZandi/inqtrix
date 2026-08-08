# Tracing legend (what is recorded, where, and as what)

## Scope

This page is the binding contract for **new instrumentation**. It says which
Langfuse observation type a step becomes, which attributes are mandatory,
what only exists under the forensic profile, and where truncation must be
reported.

**It is meant to grow.** When a step does not fit any category, when the
granularity turns out too coarse to debug a real incident, when a new engine
or observation type arrives, or when a rule here proves wrong in practice —
extend this page FIRST, then instrument to it. Note why a category was added
so the next reader can judge whether their case belongs to it. Instrumenting
first and leaving the legend behind is exactly how a trace becomes
heterogeneous again.

It does not replace [Logging](logging.md) (log envelope), [Metrics](metrics.md)
(aggregates), or the audit trail (security record). Those are separate signals
with separate audiences; see [the observability overview](debugging-runs.md).

## 1. The three levels

Every trace has exactly this shape. A new engine adds levels 2 and 3 — never a
second level 1. Level 0 exists only for HTTP-initiated work; worker-initiated
runs start at level 1.

| Level | What | Langfuse type | Span name | Opened at |
|---|---|---|---|---|
| **0 Request** | One HTTP request | `SPAN` (kind SERVER) | `<METHOD> <path>` | `RequestContextMiddleware` — adopts an incoming `traceparent`, so an upstream caller's trace continues. Probe/scrape paths are excluded |
| **1 Root** | One service start (run, chat turn, indexing job) | `AGENT` | `inqtrix.run`, `inqtrix.chat`, `inqtrix.indexing` | The execution boundary — worker loop, durable no-queue store, in-memory store, or the executor-thread wrapper for chat/editor |
| **2 Step** | One engine phase | `SPAN` | The phase name (`classify`, `plan`, `search`, `evaluate`, `answer`, `contextualize`, `decompose`, `retrieve`, `gate`, `rerank`, `grounding`) | Inside the engine, around the phase |
| **3 Call** | One provider round-trip | `GENERATION` (chat/completion), `EMBEDDING` (embeddings), `SPAN` (web search), `TOOL` (kernel tool) | `text_completion`, `chat`, `embeddings`, `web_search`, the tool name | The provider wrappers — never by hand |

Entry points that ARE a service start but run as a plain synchronous
function (the editor assist cores) use `operation_root_span` so their
provider spans do not become orphan roots — `editor:suggest`,
`editor:instruct`.

The observation type is derived from `gen_ai.operation.name`; Langfuse maps
`chat`/`text_completion` → GENERATION, `embeddings` → EMBEDDING,
`execute_tool` → TOOL, `invoke_agent` → AGENT. Set it through
[`semconv.py`](../../src/inqtrix/observability/semconv.py), never as a literal
string.

**How the chain holds.** The request span is what makes level 1 a child
rather than a disconnected root: the run payload persisted at submit carries
the W3C `traceparent` only while a span is active, and the worker parents its
`inqtrix.run` span on it. Without a request span the injection writes nothing
and every worker run opens its own trace — a failure that is invisible unless
a test drives the REAL ASGI stack (hand-opening a span around the store call
proves only that the helper works).

**Root-span rule (Langfuse specific).** Trace-level fields are taken from the
ROOT span only, and our run span is usually a CHILD (of the request span, or
of a parent run for delegated child runs). Therefore `langfuse.trace.name`,
`langfuse.user.id` and `langfuse.session.id` are always set EXPLICITLY on the
run span via the `langfuse.*` keys — never left to the generic
`user.id`/`session.id` fallbacks.

## 2. Mandatory attributes

On every span: `inqtrix.run_id`, `inqtrix.tenant`, `inqtrix.user`
(pseudonym `usr_<hex16>` — never a raw id), `inqtrix.workspace`,
`service.name`, `service.version`.

Per level:

- **Root**: `langfuse.trace.name` (`run:<mode>` / `chat:<mode>`), the mode,
  and on completion the outcome plus `gen_ai.usage.input_tokens` /
  `output_tokens`. Failures mark the span ERROR with the stable failure code.
- **Step**: `inqtrix.node` + `inqtrix.round` (research), the step-specific
  counters (candidate counts, sub-query counts, gate decision + reason). The
  knowledge grounding step additionally carries its verdict
  (`inqtrix.grounding.status`, `quotes_total`, `quotes_verified`,
  `format_repaired`, `failure_code`) — a rejected answer must be explainable
  from the span alone.
- **Call**: `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.request.model` + response model, token usage, finish reason,
  request parameters, duration, error class.

**Failure marking.** Every execution boundary marks its span on failure —
run, chat and indexing alike (`mark_current_span_error` + an outcome
attribute). A failed job that renders as a clean span in the waterfall is the
defect this rule exists to prevent.

## 3. What only forensic captures

`OBSERVABILITY_PROFILE=forensic` (or `INQTRIX_TRACE_CONTENT=on`) additionally
records CONTENT: `gen_ai.system_instructions`, `gen_ai.input.messages`,
`gen_ai.output.messages`, the redacted raw provider payload, kernel tool
arguments, search queries, raw provider answers, and the per-source search
records (`inqtrix.search.sources`: url, title, snippet, date, rank) — without
those the provenance of a bad report cannot be reconstructed from the trace.

Everything on that path goes through
[`content.py`](../../src/inqtrix/observability/content.py) — redaction plus the
`INQTRIX_TRACE_MAX_ATTR_BYTES` cap. There is no second way to set a content
attribute.

**Forensic needs a recording sink.** Depth and destination are separate knobs,
so `forensic` with `INQTRIX_TRACING=off` or `local`, or with a sample rate
below 1.0, records nothing or only some runs. Settings warns loudly about both
combinations; for complete lineage use `file` or `otlp` at rate `1.0`.

**Known content limits** (visible, never silent):

- The bare-text `complete()` path cannot expose the provider payload, so
  extended-thinking output produced there is not captured. Such spans carry
  `inqtrix.raw_unavailable=true` so an absent raw response is distinguishable
  from an empty one. Callers that need the payload use
  `complete_with_metadata`.
- That same path learns its token usage only from the state accumulator.
  With an accumulator it exports the identical numbers the ledger and
  Prometheus book; without one it carries `inqtrix.usage_unavailable=true`
  rather than a fabricated zero. Absent usage attributes are not neutral:
  the trace backend then infers token counts from the message text, and
  those inferred numbers disagree with the ledger.
- Diagnostics emitted through a plain `log.warning(...)` have no span
  counterpart — only `emit_runtime_event` bridges to span events.

## 4. Truncation is always reported

Every cap raises `inqtrix.truncation` with `limit_name`, `original_size`,
`capped_size`:

| Cap | Applies to | Default |
|---|---|---|
| `INQTRIX_TRACE_MAX_ATTR_BYTES` | Content attributes (prompts, responses, tool args) | 1 MiB |
| `_NESTED_ATTR_MAX_BYTES` | Nested values flattened into span EVENTS | 256 KiB |
| `SpanLimits(max_events)` | Events per span — OTel evicts the OLDEST first, i.e. the head of a run's lineage. Exhaustion logs a WARNING | 8192 |
| Answer-section limits | `limit_hit`, `finish_reason`, `token_utilization` on the answer step | per profile |

A new cap without a truncation event is a defect, not a detail.

**Caps are backstops, not routine limits.** They exist to stop a pathological
payload from breaking the export — never to trim normal traffic. A cap that
fires during ordinary operation is a defect in the CAP, not a fact of life:
raise it. Neither OpenTelemetry (no attribute-length limit) nor Langfuse
(warns past 16 MB per span) forces the small values that were used before; the
real constraints are the reverse proxy body size and storage volume, both of
which sit far above a normal prompt. If you see `inqtrix.truncation` in
day-to-day traces, treat it as a bug report.

## 5. Adding instrumentation — the checklist

1. Which level is it? Reuse the level-1 root, do not open a second one.
2. Name from `semconv.py`; add the constant there if it is missing, and never
   write the attribute string inline.
3. Mandatory attributes present? Pseudonyms, never raw identities.
4. Is any value content? Then it goes through `content.py` and only under the
   forensic policy.
5. Does anything get capped? Then emit `inqtrix.truncation`.
6. Executor thread involved? Bind the correlation context INSIDE the thread
   (`traced_thread_call` / `bound_thread_call`) — contextvars do not cross
   thread boundaries, and a pool submit needs `contextvars.copy_context()` per
   item.
7. Authentication diagnostics correlate browser sessions only through the
   central domain-separated `ses_<hex16>` HMAC reference. Full opaque session
   ids and raw prefixes are credential material and belong in neither spans
   nor logs. Durable logout audit rows use the same reference contract; the
   credential must not reappear through the admin read model or exports.
8. A resource mutation that changes another session's visible state uses the
   transactional resource-effect contract below; do not model cache
   invalidation as a trace event or publish it after a durable commit.
9. Add a line to this page.

## 6. Transactional resource-mutation effects

Audit rows and user invalidations are separate signals from traces, but their
atomicity is part of the instrumentation contract. A PostgreSQL mutation that
creates, revokes, or deletes a user-visible resource calls
`append_resource_effects(...)` inside the same tenant transaction as the
state change. A volatile repository may use `ResourceInvalidator` only when
its matching atomic-effects capability is false; the service must not publish
the fallback for an atomic repository. Repositories whose every mutation
implements this contract expose `atomic_resource_effects`; Editor currently
exposes the narrower `atomic_delete_resource_effects` because its Markdown
create/update paths have a different synchronization contract.

An Editor Markdown deletion uses action `editor_document.deleted`, resource
type `editor_document`, and invalidation scope `editor_documents`. It targets
the owner and every recipient that had to be revoked, contains no document
body or title, and rolls back the physical deletion if either audit or
invalidation cannot be recorded. This category exists because persistence,
cross-session cache correctness, and the security audit must describe one
outcome: committing any subset can leave authenticated sessions exposing
deleted content or leave the audit claiming a mutation that did not commit.

Personal access tokens use the same atomicity rule without a user
invalidation: PostgreSQL writes `pat.created`, sampled `pat.used`, and
`pat.revoked` in the token mutation transaction. The `pat.used` row is sampled
by the same guarded five-minute update as `last_used_at`, so repeated requests
inside the interval produce neither another timestamp write nor another audit
row. If that audit insert fails, the timestamp update rolls back and
authentication continues with a warning; create and revoke fail with their
transaction. A store exposes `atomic_audit_effects` so the service and verifier
use the canonical audit sink only for a non-atomic fallback and never duplicate
durable rows. These records contain the public token id and actor pseudonym,
but never the token name, plaintext, secret HMAC, email address, or request
payload.

## 7. Worker database-contract availability

The durable worker distinguishes a proven unsafe runtime contract from a
temporarily unreachable database. An unsafe role, permission, tenant policy,
schema, or migration revision is latched and process-fatal. A known DNS,
connection, or timeout failure instead opens the shared claim circuit:
`worker.database_contract_unavailable` is logged once per actual bounded
probe, no new queue item may cross the durable claim boundary, and the
container remains alive. The first fully successful contract probe closes the
circuit and logs `worker.database_contract_recovered` once.

These are process-level operational logs, not run spans: an unavailable
database may prevent a durable run from being claimed, so no run identity or
trace exists yet. Both events contain only the retry interval and exception
class, never a database URL, host, credential, tenant, queue payload, or job
identifier. This category exists because treating reachability like permanent
schema drift caused rapid container restarts, repeated tracing
initialization, and noisy recovery during ordinary database failover.

## 8. Related

- [Logging](logging.md) — envelope and correlation fields
- [Metrics](metrics.md) — aggregates and label cardinality
- [Debugging runs](debugging-runs.md) — the run_id → trace_id → export path
- [Settings](../configuration/settings-and-env.md) — every variable named here
