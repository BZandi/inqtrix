"""Unit tests for the shared run-event expansion and store selection.

``expand_run_event`` is the single implementation of the snapshot-
companion rule both store backends emit through; the selection bridge
``build_run_store`` is the startup guard that rejects contradictory
backend combinations loudly.
"""

from __future__ import annotations

import pytest

from inqtrix.runs.shared import access_permits_edit, expand_run_event
from inqtrix.server.container import build_run_store
from inqtrix.server.runs import RunStore
from inqtrix.settings import (
    QueueSettings,
    ServerSettings,
    Settings,
    StorageSettings,
)


def test_plain_event_passes_through_unexpanded():
    new_snapshot, events = expand_run_event(
        "inqtrix.run.queued",
        {"status": "queued", "queue_position": 1},
        status="queued",
    )
    assert new_snapshot is None
    assert events == [
        ("inqtrix.run.queued", {"status": "queued", "queue_position": 1})
    ]


def test_snapshot_payload_emits_companion_before_carrier():
    snapshot = {"current_node": "answer", "done": True}
    new_snapshot, events = expand_run_event(
        "inqtrix.run.completed",
        {"status": "completed", "snapshot": snapshot},
        status="completed",
    )
    assert new_snapshot == snapshot
    assert [event_type for event_type, _ in events] == [
        "inqtrix.run.snapshot",
        "inqtrix.run.completed",
    ]
    assert events[0][1] == {"status": "completed", "snapshot": snapshot}


def test_snapshot_event_itself_gets_no_companion():
    _, events = expand_run_event(
        "inqtrix.run.snapshot",
        {"status": "running", "snapshot": {"done": False}},
        status="running",
    )
    assert [event_type for event_type, _ in events] == ["inqtrix.run.snapshot"]


# ------------------------------------------------------------------ #
# access_permits_edit — the run-mutation gate predicate
# ------------------------------------------------------------------ #


def test_access_permits_edit_requires_explicit_access_annotation():
    """A missing annotation is malformed and must fail closed."""
    assert access_permits_edit(None) is False


@pytest.mark.parametrize("mode", ["owner", "unscoped"])
def test_access_permits_edit_allows_owner_modes(mode):
    assert access_permits_edit({"mode": mode}) is True


@pytest.mark.parametrize(
    "permission, allowed",
    [
        ("view", False),
        ("edit", True),
    ],
)
def test_access_permits_edit_respects_ordered_permission(permission, allowed):
    """A shared-in grant permits mutation only for ``edit``."""
    access = {"mode": "shared", "permission": permission}
    assert access_permits_edit(access) is allowed


def test_access_permits_edit_unknown_permission_denies():
    """An unrecognized permission string is treated as no access, not a crash."""
    assert (
        access_permits_edit({"mode": "shared", "permission": "bogus"})
        is False
    )


# ------------------------------------------------------------------ #
# build_run_store selection and startup guards
# ------------------------------------------------------------------ #


def test_default_settings_select_the_memory_store():
    store = build_run_store(Settings())
    assert isinstance(store, RunStore)


def test_valkey_queue_without_postgres_fails_loudly():
    settings = Settings(
        queue=QueueSettings(backend="valkey", valkey_url="redis://x:1/0")
    )
    with pytest.raises(RuntimeError, match="INQTRIX_STORAGE_BACKEND=postgres"):
        build_run_store(settings)


def test_valkey_queue_without_url_fails_loudly():
    settings = Settings(
        storage=StorageSettings(
            backend="postgres",
            database_url="postgresql+asyncpg://x:y@127.0.0.1:1/inqtrix",
        ),
        queue=QueueSettings(backend="valkey"),
    )
    with pytest.raises(RuntimeError, match="INQTRIX_VALKEY_URL"):
        build_run_store(settings)


def test_postgres_storage_without_url_fails_loudly():
    settings = Settings(storage=StorageSettings(backend="postgres"))
    with pytest.raises(RuntimeError, match="INQTRIX_DATABASE_URL"):
        build_run_store(settings)


def test_postgres_storage_selects_the_durable_store():
    from inqtrix.runs.postgres_store import PostgresRunStore

    settings = Settings(
        storage=StorageSettings(
            backend="postgres",
            database_url="postgresql+asyncpg://x:y@127.0.0.1:1/inqtrix",
        )
    )
    store = build_run_store(settings)
    assert isinstance(store, PostgresRunStore)


def test_postgres_store_uses_durable_retention_not_replay_ttl():
    """The durable store must keep terminal runs for the durable retention
    window, not the in-memory store's short replay TTL; otherwise completed
    research reports are pruned minutes after finishing.

    Distinct values pin the wiring: a revert to ``run_completed_ttl_seconds``
    would make this go red.
    """
    from inqtrix.runs.postgres_store import PostgresRunStore

    settings = Settings(
        server=ServerSettings(
            run_completed_ttl_seconds=300,
            run_durable_retention_seconds=7_776_000,
        ),
        storage=StorageSettings(
            backend="postgres",
            database_url="postgresql+asyncpg://x:y@127.0.0.1:1/inqtrix",
        ),
    )
    store = build_run_store(settings)
    assert isinstance(store, PostgresRunStore)
    assert store._completed_ttl_seconds == 7_776_000


def test_run_durable_retention_defaults_to_ninety_days():
    """The out-of-the-box durable retention is 90 days (operators can extend)."""
    assert ServerSettings().run_durable_retention_seconds == 90 * 86_400
