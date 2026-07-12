"""/readyz readiness contract (2.3).

Liveness (``/health``) is provider-only; readiness keys traffic on the
STATEFUL dependencies: database or queue down => 503 (drain the pod),
vector store down => degraded but ready (knowledge fails loudly
per-request, everything else serves). Memory backends are trivially
ready so the zero-infrastructure default stays green.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inqtrix.services.system_runtime import readiness_payload
from inqtrix.settings import Settings


def _container(
    *,
    storage_backend: str = "memory",
    queue_backend: str = "memory",
    session_factory=None,
    knowledge_service=None,
) -> SimpleNamespace:
    settings = Settings()
    settings.storage.backend = storage_backend  # type: ignore[assignment]
    settings.queue.backend = queue_backend  # type: ignore[assignment]
    return SimpleNamespace(
        settings=settings,
        session_factory=session_factory,
        knowledge_service=knowledge_service,
    )


@pytest.mark.asyncio
async def test_memory_backends_are_trivially_ready() -> None:
    status_code, payload = await readiness_payload(_container())
    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["checks"] == {
        "database": "skipped",
        "queue": "skipped",
        "vector_store": "skipped",
    }


@pytest.mark.asyncio
async def test_postgres_without_reachable_database_is_not_ready() -> None:
    class _FailingSession:
        async def __aenter__(self):
            raise ConnectionError("db down")

        async def __aexit__(self, *exc):
            return False

    status_code, payload = await readiness_payload(
        _container(storage_backend="postgres", session_factory=_FailingSession)
    )
    assert status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["checks"]["database"] == "unavailable"


@pytest.mark.asyncio
async def test_postgres_declared_without_factory_is_a_visible_miswire() -> None:
    status_code, payload = await readiness_payload(
        _container(storage_backend="postgres", session_factory=None)
    )
    assert status_code == 503
    assert payload["checks"]["database"] == "unavailable"


@pytest.mark.asyncio
async def test_dead_vector_store_degrades_but_stays_ready() -> None:
    knowledge = SimpleNamespace(
        knowledge=SimpleNamespace(
            store=SimpleNamespace(is_available=lambda: False)
        )
    )
    status_code, payload = await readiness_payload(
        _container(knowledge_service=knowledge)
    )
    assert status_code == 200
    assert payload["status"] == "degraded"
    assert payload["checks"]["vector_store"] == "unavailable"
