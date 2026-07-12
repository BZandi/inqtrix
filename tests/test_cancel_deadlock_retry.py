"""P4: the synchronous cancel path retries a Postgres deadlock (40P01).

The cancel cascade locks parent->child while a concurrent child terminal
write locks child->parent; the opposite orders can deadlock and Postgres
aborts one side. The terminal side is redelivered by the queue, but cancel
is a synchronous request with no redelivery, so ``PostgresRunStore.cancel``
retries the whole transaction a bounded number of times on 40P01 and
surfaces anything else immediately.

Driver realism matters here: this deployment uses **asyncpg**, and the
asyncpg dialect maps a ``PostgresError`` (the base of
``DeadlockDetectedError``) to the bare DBAPI ``Error``, so SQLAlchemy wraps a
real 40P01 as :class:`sqlalchemy.exc.DBAPIError` — NOT ``OperationalError``.
The deadlock cases below therefore use ``DBAPIError`` (what the driver
actually emits); an ``OperationalError``-only catch would let them through as
a 500. A psycopg-shaped ``OperationalError`` case guards the driver-agnostic
path (``OperationalError`` is a ``DBAPIError`` subclass).

These are pure control-flow tests: ``_cancel_db``/``_call`` are stubbed, so
no database, engine, or background loop is needed. Only the retry contract
is under test — the actual deadlock is a Postgres runtime behavior.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from inqtrix.runs.postgres_store import _CANCEL_DEADLOCK_RETRIES, PostgresRunStore


def _make_store() -> PostgresRunStore:
    """A PostgresRunStore with only the attributes ``cancel`` touches.

    Bypasses ``__init__`` (which would spawn a loop thread and adopt an
    engine); ``_cancel_db`` and ``_call`` are stubbed per test.
    """
    store = object.__new__(PostgresRunStore)
    store._lock = threading.RLock()
    store._local = {}
    store._swept_waiting = SimpleQueue()
    store._parents_to_wake = SimpleQueue()
    return store


def _db_error(cls: type[DBAPIError] = DBAPIError, sqlstate: str = "40P01"):
    """A SQLAlchemy DB error carrying ``orig.sqlstate`` like the real driver.

    Default class is the bare :class:`DBAPIError` — the exact class an asyncpg
    40P01 is wrapped as; pass ``OperationalError`` for the psycopg shape.
    """
    orig = Exception("deadlock detected")
    orig.sqlstate = sqlstate
    return cls("cancel", {}, orig)


def test_cancel_retries_on_asyncpg_deadlock_then_succeeds():
    store = _make_store()
    summary = {"status": "cancelled"}
    store._cancel_db = Mock(return_value=object())  # placeholder passed to _call
    # asyncpg wraps a 40P01 as a bare DBAPIError (regression guard: the old
    # `except OperationalError` would NOT have caught this).
    store._call = Mock(side_effect=[_db_error(), (summary, [])])

    result = store.cancel("run-1")

    assert result is summary
    assert store._call.call_count == 2  # first aborted, second committed


def test_cancel_retries_on_operationalerror_deadlock_too():
    """Driver-agnostic: a psycopg-shaped OperationalError 40P01 also retries."""
    store = _make_store()
    summary = {"status": "cancelled"}
    store._cancel_db = Mock(return_value=object())
    store._call = Mock(
        side_effect=[_db_error(cls=OperationalError), (summary, [])]
    )

    result = store.cancel("run-1")

    assert result is summary
    assert store._call.call_count == 2


def test_cancel_reraises_non_deadlock_immediately():
    store = _make_store()
    store._cancel_db = Mock(return_value=object())
    # 55P03 lock_not_available is NOT a deadlock — must surface at once.
    store._call = Mock(side_effect=_db_error(sqlstate="55P03"))

    with pytest.raises(DBAPIError):
        store.cancel("run-1")

    assert store._call.call_count == 1


def test_cancel_gives_up_after_max_deadlock_retries():
    store = _make_store()
    store._cancel_db = Mock(return_value=object())
    store._call = Mock(
        side_effect=[_db_error() for _ in range(_CANCEL_DEADLOCK_RETRIES)]
    )

    with pytest.raises(DBAPIError):
        store.cancel("run-1")

    assert store._call.call_count == _CANCEL_DEADLOCK_RETRIES


def test_parallel_parent_wake_drains_each_committed_handoff_once() -> None:
    """Concurrent terminal callers cannot check-then-pop the same wake."""
    store = _make_store()
    calls: list[str] = []
    calls_lock = threading.Lock()

    class DispatchQueue:
        def enqueue(self, *, run_id: str, tenant_id: str) -> None:
            assert tenant_id == "default"
            with calls_lock:
                calls.append(run_id)

    store._dispatch_queue = DispatchQueue()
    expected = [f"parent-{index}" for index in range(200)]
    for run_id in expected:
        store._parents_to_wake.put(run_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _index: store._dispatch_woken_parents(), range(8)))

    assert sorted(calls) == sorted(expected)
