"""THE one mapping point for trace attribute names (GenAI + Langfuse).

The OTel GenAI semantic conventions are still status *Development*
(pinned here against semantic-conventions v1.37+, verified 2026-07-23),
and Langfuse maps a fixed keyword set with documented priorities. Both
contracts live in THIS module only — when either evolves, this file is
the single edit point; instrumentation code never spells attribute
strings itself.

Langfuse contract essentials (verified against the Langfuse ingestion
processor, server v3.224):

* ``langfuse.*`` keys always win over generic ones. Trace-level fields
  (name/user/session/tags) are taken ONLY from the root span or from
  explicit ``langfuse.*`` keys — and Inqtrix's run span is a CHILD of
  the API request context, so user/session/name MUST use the explicit
  keys (:data:`LANGFUSE_USER_ID`, :data:`LANGFUSE_SESSION_ID`,
  :data:`LANGFUSE_TRACE_NAME`).
* The observation type is derived from ``gen_ai.operation.name``
  (``chat``/``text_completion`` → generation, ``embeddings`` →
  embedding, ``invoke_agent`` → agent, ``execute_tool`` → tool);
  spans without a GenAI operation stay plain spans.
* Content belongs in the ``gen_ai.input.messages`` /
  ``gen_ai.output.messages`` / ``gen_ai.system_instructions`` SPAN
  attributes (JSON strings), never in span events.
* Langfuse applies NO server-side truncation or masking in the OSS
  tier — the caller-side policy in
  :mod:`inqtrix.observability.content` is the only guard.
"""

from __future__ import annotations

# --- OTel GenAI semantic conventions (Development; v1.37+ shapes) ------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

# Operation values (drive the Langfuse observation type).
OPERATION_CHAT = "chat"
OPERATION_TEXT_COMPLETION = "text_completion"
OPERATION_EMBEDDINGS = "embeddings"
OPERATION_INVOKE_AGENT = "invoke_agent"
OPERATION_EXECUTE_TOOL = "execute_tool"

# --- Langfuse first-class keys (highest mapping priority) --------------
LANGFUSE_TRACE_NAME = "langfuse.trace.name"
LANGFUSE_USER_ID = "langfuse.user.id"
LANGFUSE_SESSION_ID = "langfuse.session.id"

# --- Inqtrix namespace (own attributes, stable across backends) --------
INQTRIX_RUN_ID = "inqtrix.run_id"
INQTRIX_TENANT = "inqtrix.tenant"
INQTRIX_WORKSPACE = "inqtrix.workspace"
INQTRIX_ATTEMPT = "inqtrix.attempt"
HTTP_REQUEST_METHOD = "http.request.method"
URL_PATH = "url.path"
INQTRIX_REQUEST_ID = "inqtrix.request_id"
INQTRIX_NODE = "inqtrix.node"
INQTRIX_RAW_UNAVAILABLE = "inqtrix.raw_unavailable"
"""Set when a call path structurally cannot expose the provider's
raw payload (bare-text complete()) — so an absent raw response is
distinguishable from a provider that returned nothing."""
INQTRIX_USAGE_UNAVAILABLE = "inqtrix.usage_unavailable"
"""Set when a bare-text complete() runs without a state accumulator, the
only channel through which that path learns its token usage. Absent usage
attributes let the trace backend infer token counts from the message text,
which then disagrees with the ledger; this marker keeps the gap visible
instead of exporting a fabricated zero."""
INQTRIX_ROUND = "inqtrix.round"
INQTRIX_REASONING_EFFORT = "inqtrix.request.reasoning_effort"
INQTRIX_SCHEMA_NAME = "inqtrix.request.schema_name"
INQTRIX_RESPONSE_RAW = "inqtrix.response.raw"
INQTRIX_TOOL_CALL_COUNT = "inqtrix.response.tool_call_count"
INQTRIX_SEARCH_PROVIDER = "inqtrix.search.provider"
INQTRIX_SEARCH_ENGINE = "inqtrix.search.engine"
INQTRIX_SEARCH_QUERY = "inqtrix.search.query"
INQTRIX_SEARCH_MODE = "inqtrix.search.mode"
INQTRIX_SEARCH_RECENCY = "inqtrix.search.recency_filter"
INQTRIX_SEARCH_DOMAIN_FILTER_COUNT = "inqtrix.search.domain_filter_count"
INQTRIX_SEARCH_SOURCE_COUNT = "inqtrix.search.source_count"
INQTRIX_SEARCH_ANSWER_LENGTH = "inqtrix.search.answer_length"
INQTRIX_SEARCH_SOURCES = "inqtrix.search.sources"
INQTRIX_SEARCH_ANSWER = "inqtrix.search.answer"
INQTRIX_SEARCH_INPUT_TOKENS = "inqtrix.search.input_tokens"
INQTRIX_SEARCH_OUTPUT_TOKENS = "inqtrix.search.output_tokens"
INQTRIX_EMBED_TEXT_COUNT = "inqtrix.embeddings.text_count"

# Truncation visibility (§ Erfassungskatalog E): every capped value adds
# this span event so accidental caps are systematically findable.
TRUNCATION_EVENT = "inqtrix.truncation"
TRUNCATION_ORIGINAL_SIZE = "original_size"
TRUNCATION_CAPPED_SIZE = "capped_size"
TRUNCATION_LIMIT_NAME = "limit_name"
