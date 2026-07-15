"""User invalidations remain bounded, scoped, and content-free."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from inqtrix.auth.principal import Principal
from inqtrix.server.routers.user_events import (
    _invalidation_frame,
    _sse_frame,
    build_router,
)
from inqtrix.user_events import (
    MemoryUserEventStore,
    UserEventPage,
    UserInvalidation,
)


@pytest.mark.asyncio
async def test_memory_user_events_replay_only_the_target_users_frames() -> None:
    store = MemoryUserEventStore()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    await store.append(
        tenant_id="default",
        target_user_id=user_a,
        scope="runs",
        resource_type="run",
        resource_id="run_a",
    )
    await store.append(
        tenant_id="default",
        target_user_id=user_b,
        scope="prompts",
        resource_type="prompt_template",
        resource_id="pt_b",
    )

    page = await store.page_after(
        tenant_id="default", target_user_id=user_a, cursor=0
    )

    assert page.reset_required is False
    assert [(event.scope, event.resource_id) for event in page.events] == [
        ("runs", "run_a")
    ]
    assert page.current_cursor == 2


@pytest.mark.asyncio
async def test_memory_user_events_request_reset_after_bounded_eviction() -> None:
    store = MemoryUserEventStore(max_events=1)
    user_id = uuid.uuid4()
    await store.append(
        tenant_id="default", target_user_id=user_id, scope="runs"
    )
    await store.append(
        tenant_id="default", target_user_id=user_id, scope="skills"
    )

    page = await store.page_after(
        tenant_id="default", target_user_id=user_id, cursor=1
    )

    assert page.reset_required is True
    assert page.events == ()
    assert page.current_cursor == 2


@pytest.mark.asyncio
async def test_memory_user_events_reset_a_cursor_from_before_restart() -> None:
    store = MemoryUserEventStore()
    user_id = uuid.uuid4()
    await store.append(
        tenant_id="default", target_user_id=user_id, scope="runs"
    )

    page = await store.page_after(
        tenant_id="default", target_user_id=user_id, cursor=500
    )

    assert page.reset_required is True
    assert page.events == ()
    assert page.current_cursor == 1


def test_ready_and_reset_sse_frames_have_the_public_contract() -> None:
    user_id = uuid.uuid4()

    ready = _sse_frame(
        "ready", {"user_id": str(user_id), "cursor": "17"}
    )
    reset = _sse_frame("reset", {})

    assert "event: ready" in ready
    assert json.loads(ready.split("data: ", 1)[1]) == {
        "user_id": str(user_id),
        "cursor": "17",
    }
    assert "data: {}" in reset


@pytest.mark.asyncio
async def test_invalidation_frame_never_serializes_an_unrelated_payload() -> None:
    store = MemoryUserEventStore()
    event = await store.append(
        tenant_id="default",
        target_user_id=uuid.uuid4(),
        scope="skills",
        resource_type="skill_template",
        resource_id="skill_1",
    )

    frame = _invalidation_frame(event)

    assert "id: 1" in frame
    assert '"scope":"skills"' in frame
    assert '"resource_id":"skill_1"' in frame
    assert "title" not in frame
    assert "content" not in frame
    assert "permission" not in frame


class _ConnectedRequest:
    def __init__(self, last_event_id: str = "0") -> None:
        self.headers = {"last-event-id": last_event_id}

    async def is_disconnected(self) -> bool:
        return False


class _ReplayStore:
    def __init__(self, event: UserInvalidation) -> None:
        self.event = event

    async def current_cursor(self, *, tenant_id: str) -> int:
        del tenant_id
        return self.event.id

    async def page_after(self, **kwargs) -> UserEventPage:
        cursor = int(kwargs["cursor"])
        return UserEventPage(
            (self.event,) if cursor < self.event.id else (),
            self.event.id,
        )

    async def wait_for_change(self, **kwargs) -> None:
        del kwargs


def _event_endpoint(container) -> object:
    router = build_router(container)
    return next(
        route.endpoint
        for route in router.routes
        if route.path == "/v1/user/events"
    )


@pytest.mark.asyncio
async def test_user_stream_sends_ready_before_replayed_invalidation() -> None:
    user_id = uuid.uuid4()
    principal = Principal(user_id=user_id, kind="oidc_session")
    event = UserInvalidation(
        id=4,
        tenant_id="default",
        target_user_id=user_id,
        scope="runs",
        resource_type="run",
        resource_id="run_4",
    )

    async def resolve(_request):
        return principal

    endpoint = _event_endpoint(
        SimpleNamespace(
            user_event_store=_ReplayStore(event),
            principal_dependency=resolve,
        )
    )
    response = await endpoint(_ConnectedRequest("3"), principal)
    iterator = response.body_iterator

    ready = await anext(iterator)
    invalidation = await anext(iterator)
    await iterator.aclose()

    assert "event: ready" in ready
    assert f'"user_id":"{user_id}"' in ready
    assert "id: 4" in invalidation
    assert "event: invalidate" in invalidation


@pytest.mark.asyncio
async def test_user_stream_rechecks_identity_before_each_data_frame() -> None:
    user_id = uuid.uuid4()
    principal = Principal(user_id=user_id, kind="oidc_session")
    event = UserInvalidation(
        id=2,
        tenant_id="default",
        target_user_id=user_id,
        scope="skills",
    )
    calls = 0

    async def resolve(_request):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise HTTPException(status_code=401)
        return principal

    endpoint = _event_endpoint(
        SimpleNamespace(
            user_event_store=_ReplayStore(event),
            principal_dependency=resolve,
        )
    )
    response = await endpoint(_ConnectedRequest("1"), principal)
    iterator = response.body_iterator

    ready = await anext(iterator)

    assert "event: ready" in ready
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_user_stream_rechecks_identity_before_quiet_keepalive() -> None:
    user_id = uuid.uuid4()
    principal = Principal(user_id=user_id, kind="oidc_session")
    event = UserInvalidation(
        id=2,
        tenant_id="default",
        target_user_id=user_id,
        scope="skills",
    )
    calls = 0

    async def resolve(_request):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise HTTPException(status_code=401)
        return principal

    endpoint = _event_endpoint(
        SimpleNamespace(
            user_event_store=_ReplayStore(event),
            principal_dependency=resolve,
        )
    )
    response = await endpoint(_ConnectedRequest("2"), principal)
    iterator = response.body_iterator

    ready = await anext(iterator)

    assert "event: ready" in ready
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)
