"""Correlation context carried into every log line.

``contextvars`` keep the per-request / per-run correlation fields
(request id, run id, subject pseudonym, workspace, tenant) flowing
through async call chains without threading parameters through every
function. The JSON formatter reads :func:`current_log_context` for each
record, so one ``bind_log_context`` at a boundary (request middleware,
worker job claim, principal resolution) stamps every log line inside
that boundary.

Two usage patterns:

* **Boundaries that outlive their task** (the ASGI middleware, worker
  job loops): keep the returned tokens and call
  :func:`reset_log_context` in ``finally`` — those code paths reuse one
  task/thread for many logical units, so values MUST NOT leak.
* **Request-scoped enrichment** (auth dependencies): binding without a
  reset is safe because every request runs in its own task context and
  the values die with it.

The user field always carries the stable pseudonym, never a raw id —
the same reference that appears in audit correlation, so log lines and
audit rows join without exposing identities in the log stream.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Mapping

from inqtrix.auth.log_redaction import stable_pseudonym

if TYPE_CHECKING:
    from inqtrix.auth.principal import Principal

log = logging.getLogger("inqtrix")

_REQUEST_ID: ContextVar[str] = ContextVar("inqtrix_request_id", default="")
_RUN_ID: ContextVar[str] = ContextVar("inqtrix_run_id", default="")
_USER_REF: ContextVar[str] = ContextVar("inqtrix_user_ref", default="")
_WORKSPACE_ID: ContextVar[str] = ContextVar(
    "inqtrix_workspace_id", default=""
)
_TENANT_ID: ContextVar[str] = ContextVar("inqtrix_tenant_id", default="")

# Field name -> ContextVar. The names are the PUBLIC log-field names;
# an unknown keyword in bind_log_context is a programming error and
# raises instead of silently vanishing from the logs.
_FIELDS: Mapping[str, ContextVar[str]] = {
    "request_id": _REQUEST_ID,
    "run_id": _RUN_ID,
    "user": _USER_REF,
    "workspace": _WORKSPACE_ID,
    "tenant": _TENANT_ID,
}


def bind_log_context(**fields: object) -> dict[str, Token[str]]:
    """Bind correlation fields for the current context.

    Args:
        **fields: Any of ``request_id``, ``run_id``, ``user``,
            ``workspace``, ``tenant``. Values are stringified; ``None``
            binds the empty string (which :func:`current_log_context`
            omits).

    Returns:
        Tokens for :func:`reset_log_context`. Boundaries that reuse a
        task/thread for many logical units must reset in ``finally``.

    Raises:
        KeyError: On a field name that is not a known log field.
    """
    tokens: dict[str, Token[str]] = {}
    for name, value in fields.items():
        var = _FIELDS[name]
        tokens[name] = var.set("" if value is None else str(value))
    return tokens


def reset_log_context(tokens: Mapping[str, Token[str]]) -> None:
    """Undo one :func:`bind_log_context` call (pass its return value)."""
    for name, token in tokens.items():
        _FIELDS[name].reset(token)


def current_log_context() -> dict[str, str]:
    """The non-empty correlation fields for the current context."""
    context: dict[str, str] = {}
    for name, var in _FIELDS.items():
        value = var.get()
        if value:
            context[name] = value
    return context


def bind_principal_context(principal: "Principal | Any") -> None:
    """Stamp the resolved principal onto the log context.

    Called by the auth dependencies after successful resolution; the
    request task owns the values, so no reset is needed. The user field
    carries the stable pseudonym (``usr_<hex16>``), never the raw id.
    Unscoped principals (anonymous/static) bind only the tenant.
    """
    user_id = getattr(principal, "user_id", None)
    tenant_id = getattr(principal, "tenant_id", "")
    fields: dict[str, object] = {}
    if tenant_id:
        fields["tenant"] = tenant_id
    if user_id is not None:
        fields["user"] = stable_pseudonym("usr", user_id)
    if fields:
        bind_log_context(**fields)
    # Audit origin: record HOW this request authenticated (principal
    # kind). Lives in the request-origin vars, not the log context —
    # only audit writers read it.
    from inqtrix.auth.request_origin import bind_auth_method

    bind_auth_method(principal)


# ---- product feature (metrics label, NOT log context) -------------- #

_FEATURE: "ContextVar[str]" = ContextVar("inqtrix_feature", default="")

_MODE_TO_FEATURE = {
    "research": "research",
    "direct_llm": "research",
    "knowledge": "knowledge",
    "agent_kernel": "kernel",
    "workspace_agent": "kernel",
    "chat": "chat",
    "editor": "editor",
    "indexing": "indexing",
}


def bind_feature(mode_or_feature: str) -> "Token[str]":
    """Bind the product feature for metrics labels.

    Accepts either a run mode (mapped onto the bounded feature
    vocabulary) or a feature name directly; anything unknown collapses
    to ``other`` — the label set must stay bounded. Deliberately NOT a
    log-context field: it exists for the token counters, not for every
    log line. IMPORTANT: executor threads copy no contextvars — set it
    INSIDE the thread callable (the chat/editor gotcha).
    """
    value = str(mode_or_feature or "").strip()
    feature = _MODE_TO_FEATURE.get(value, value if value in
                                   _MODE_TO_FEATURE.values() else "other")
    return _FEATURE.set(feature)


_reset_feature_misuse_warned = False


def reset_feature(token: "Token[str]") -> None:
    global _reset_feature_misuse_warned
    try:
        _FEATURE.reset(token)
    except ValueError:
        # Token from a different context = a caller violated the
        # bind-inside-the-thread contract. Repair the label, but never
        # silently: the misuse is defect evidence.
        if not _reset_feature_misuse_warned:
            _reset_feature_misuse_warned = True
            log.warning(
                "reset_feature: Token stammt aus einem anderen Context "
                "(bind/reset muessen im selben Thread-Callable laufen) - "
                "Feature-Label wird stattdessen geleert."
            )
        _FEATURE.set("")


def with_feature(feature: str):
    """Decorator: bind the metrics feature for the wrapped SYNC call.

    For executor-thread cores (editor suggest/instruct) whose threads
    are reused — bind on entry, reset on exit, exception-safe.
    """

    def decorate(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = bind_feature(feature)
            try:
                return fn(*args, **kwargs)
            finally:
                reset_feature(token)

        return wrapper

    return decorate


def clear_feature() -> None:
    """Unconditional reset for reused executor/claim threads.

    The loops call this in their ``finally`` so a segment's feature can
    never leak into the next job on the same thread.
    """
    _FEATURE.set("")


def current_feature() -> str:
    """The bound feature, or ``other`` when nothing was bound."""
    return _FEATURE.get() or "other"


# --------------------------------------------------------------------- #
# Usage-ledger subject
#
# The provider wrappers need the RAW booking identity (tenant + user
# UUID + workspace) to write llm_usage rows — the log-context ``user``
# field deliberately carries only the one-way pseudonym and cannot be
# used. Bound at the same boundaries that bind the feature label; the
# same executor-thread rule applies (bind INSIDE the thread callable).
# --------------------------------------------------------------------- #

_USAGE_SUBJECT: ContextVar["UsageSubjectContext | None"] = ContextVar(
    "inqtrix_usage_subject", default=None
)


class UsageSubjectContext:
    """Raw booking identity for the usage ledger (never logged)."""

    __slots__ = ("tenant_id", "user_id", "workspace_id")

    def __init__(
        self,
        tenant_id: str,
        user_id: "Any",
        workspace_id: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.workspace_id = workspace_id


def bind_usage_subject(
    tenant_id: str | None,
    user_id: "Any",
    workspace_id: str | None = None,
) -> "Token[UsageSubjectContext | None]":
    """Bind the ledger subject; missing identity binds None (unmetered)."""
    if not tenant_id or user_id is None:
        return _USAGE_SUBJECT.set(None)
    return _USAGE_SUBJECT.set(
        UsageSubjectContext(str(tenant_id), user_id, workspace_id)
    )


def clear_usage_subject() -> None:
    """Unconditional reset for reused executor/claim threads."""
    _USAGE_SUBJECT.set(None)


def bound_thread_call(
    fn: Any,
    *,
    feature: str | None = None,
    usage_subject: tuple[Any, Any, Any] | None = None,
) -> Any:
    """Wrap an executor callable so feature/ledger context binds INSIDE
    the thread (run_in_executor copies no contextvars) and always
    clears — the span-less sibling of ``traced_thread_call``."""

    # Snapshot the caller's correlation context while we are still in
    # the request task; the pool thread inherits no contextvars.
    log_context = {
        key: value for key, value in current_log_context().items() if value
    }

    def runner() -> Any:
        feature_token = bind_feature(feature) if feature else None
        if usage_subject is not None:
            bind_usage_subject(*usage_subject)
        log_tokens = bind_log_context(**log_context)
        try:
            return fn()
        finally:
            if feature_token is not None:
                reset_feature(feature_token)
            if usage_subject is not None:
                clear_usage_subject()
            reset_log_context(log_tokens)

    return runner


def current_usage_subject() -> "UsageSubjectContext | None":
    return _USAGE_SUBJECT.get()
