"""Log-context contextvars (bind/reset/current + principal stamping)."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from inqtrix.auth import log_redaction
from inqtrix.auth.log_redaction import (
    configure_stable_pseudonyms,
    stable_pseudonym,
)
from inqtrix.observability.context import (
    bind_log_context,
    bind_principal_context,
    current_log_context,
    reset_log_context,
)


@pytest.fixture(autouse=True)
def _reset_pseudonym_state():
    saved_key = log_redaction._stable_key
    saved_warned = log_redaction._fallback_warned
    yield
    log_redaction._stable_key = saved_key
    log_redaction._fallback_warned = saved_warned


def test_bind_current_reset_roundtrip():
    tokens = bind_log_context(request_id="req-1", run_id="run_abc")
    try:
        context = current_log_context()
        assert context["request_id"] == "req-1"
        assert context["run_id"] == "run_abc"
    finally:
        reset_log_context(tokens)
    assert "request_id" not in current_log_context()
    assert "run_id" not in current_log_context()


def test_unknown_field_raises_loudly():
    with pytest.raises(KeyError):
        bind_log_context(reqeust_id="typo")


def test_none_binds_empty_and_is_omitted():
    tokens = bind_log_context(run_id=None)
    try:
        assert "run_id" not in current_log_context()
    finally:
        reset_log_context(tokens)


def test_context_is_isolated_between_tasks():
    async def scenario():
        async def task_a():
            bind_log_context(run_id="run_a")
            await asyncio.sleep(0)
            return current_log_context().get("run_id")

        async def task_b():
            await asyncio.sleep(0)
            return current_log_context().get("run_id")

        return await asyncio.gather(task_a(), task_b())

    seen_a, seen_b = asyncio.run(scenario())
    assert seen_a == "run_a"
    assert seen_b is None
    assert current_log_context().get("run_id") is None


def test_principal_binding_uses_stable_pseudonym():
    configure_stable_pseudonyms("context-test-pepper")
    user_id = uuid.uuid4()
    principal = SimpleNamespace(user_id=user_id, tenant_id="default")
    tokens = bind_log_context(user="", tenant="")
    try:
        bind_principal_context(principal)
        context = current_log_context()
        assert context["user"] == stable_pseudonym("usr", user_id)
        assert str(user_id) not in context["user"]
        assert context["tenant"] == "default"
    finally:
        reset_log_context(tokens)


def test_unscoped_principal_binds_no_user():
    principal = SimpleNamespace(user_id=None, tenant_id="default")
    tokens = bind_log_context(user="", tenant="")
    try:
        bind_principal_context(principal)
        assert "user" not in current_log_context()
        assert current_log_context()["tenant"] == "default"
    finally:
        reset_log_context(tokens)


def test_bound_thread_call_carries_log_context():
    """Executor threads inherit no contextvars, so
    the wrapper must re-bind the correlation context — otherwise every
    log line of the chat/editor path loses its request_id."""
    from inqtrix.observability.context import (
        bind_log_context,
        bound_thread_call,
        current_log_context,
        reset_log_context,
    )

    seen: dict = {}

    def _inner():
        seen.update(current_log_context())

    tokens = bind_log_context(request_id="req-42", user="usr_abc")
    try:
        runner = bound_thread_call(_inner, feature="chat")
    finally:
        reset_log_context(tokens)
    # Run it where the caller's context is already gone (like a pool
    # thread would).
    assert current_log_context() == {}
    runner()
    assert seen["request_id"] == "req-42"
    assert seen["user"] == "usr_abc"
    assert current_log_context() == {}  # cleared again


def test_audit_entry_derives_actor_pseudonym():
    """The read model shows the actor ONLY as
    actor_pseudonym, so AuditEntry derives it for every writer."""
    import uuid

    from inqtrix.auth.permissions import AuditEntry

    entry = AuditEntry(
        tenant_id="default",
        actor_user_id=uuid.uuid4(),
        action="admin.thing",
        resource_type="run",
        resource_id="run_1",
    )
    assert entry.actor_pseudonym and entry.actor_pseudonym.startswith("usr_")
    anonymous = AuditEntry(
        tenant_id="default",
        actor_user_id=None,
        action="auth.login_failed",
        resource_type="user",
        resource_id="x@example.com",
    )
    assert anonymous.actor_pseudonym is None
