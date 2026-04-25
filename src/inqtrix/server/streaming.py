"""SSE streaming utilities for OpenAI-compatible chat completion responses."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from functools import partial
from queue import Empty, Queue
from typing import Any, AsyncIterator

from fastapi import Request

from inqtrix.graph import run as agent_run
from inqtrix.i18n import detect_ui_language, t
from inqtrix.providers.base import ProviderContext
from inqtrix.server.session import SessionStore, prospective_session_id
from inqtrix.settings import AgentSettings
from inqtrix.strategies import StrategyContext
from inqtrix.text import iter_word_chunks
from inqtrix.urls import sanitize_error

log = logging.getLogger("inqtrix")

MODEL_NAME = "research-agent"


def make_chunk(
    chat_id: str,
    content: str,
    finish_reason: str | None = None,
) -> str:
    """Build a single SSE chunk in the OpenAI streaming format."""
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _watch_disconnect(
    request: Request,
    cancel_event: threading.Event,
) -> None:
    """Background task: signal *cancel_event* on the next ``http.disconnect``.

    Why this is a dedicated task rather than a poll inside the streaming
    loop: ``Request.is_disconnected()`` is a non-blocking probe that uses
    an ``anyio.CancelScope().cancel()`` trick to read the ASGI receive
    channel without blocking. uvicorn does not deliver ``http.disconnect``
    via that path during an active streaming response (its ASGI receive
    side stays idle while we only write SSE chunks), so the probe never
    flips. A blocking ``await request.receive()`` in a parallel task is
    the only reliable way to surface the disconnect — verified live in
    the Azure stack test 2026-04-19 where polling missed the disconnect
    entirely and a 138s deep-3-round run completed despite a 3s
    ``curl --max-time`` abort.

    Args:
        request: The Starlette/FastAPI request whose receive channel we
            listen on. Must be the live request object the route handler
            saw — passing a stale or replayed request silently no-ops.
        cancel_event: Threading event the LangGraph node-boundary probe
            (``inqtrix.state.check_cancel_event``) consults. We set it
            on the first ``http.disconnect`` message; subsequent messages
            are ignored. Setting is thread-safe (used by both the asyncio
            loop here and the agent ThreadPoolExecutor that reads it).

    Notes:
        * The task exits as soon as the disconnect is observed. The
          calling generator is responsible for cancelling this task on
          normal completion (``task.cancel()`` + ``await task`` with
          ``CancelledError`` swallowed) so it does not leak.
        * Any exception inside ``request.receive()`` (e.g. underlying
          transport torn down by uvicorn) also signals the cancel — we
          treat unexpected receive errors as "client gone" rather than
          continuing without the disconnect signal.
    """
    try:
        while not cancel_event.is_set():
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                cancel_event.set()
                log.info("Run cancelled by client disconnect")
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — disconnect watcher must not crash the stream
        log.debug(
            "Disconnect watcher exiting after receive() error: %s",
            sanitize_error(exc),
        )
        cancel_event.set()


async def stream_response(
    question: str,
    history: str,
    prev_session: dict[str, Any] | None,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
    session_store: SessionStore,
    messages: list[dict] | None = None,
    include_progress: bool = True,
    request: Request | None = None,
    cancel_event: threading.Event | None = None,
    stack_name: str = "",
) -> AsyncIterator[str]:
    """Execute the agent and yield progress updates + answer as SSE chunks.

    When ``request`` is supplied (server route path), a background
    :func:`_watch_disconnect` task is spawned. It blocks on
    ``await request.receive()`` and sets ``cancel_event`` as soon as
    uvicorn delivers ``http.disconnect`` — the only ASGI signal that
    actually fires when an SSE client closes the socket mid-stream.
    The streaming loop only checks ``cancel_event`` (no own polling)
    and exits the generator when set.

    ``cancel_event`` defaults to a fresh :class:`threading.Event` when
    not supplied so library tests can still inspect the sequence
    without wiring a Request object. Backwards compatibility: tests
    that supply a Request mock with ``is_disconnected()`` continue to
    pass because the watcher only depends on ``await request.receive()``
    — they just need to expose ``receive()`` as well.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    request_deadline = time.monotonic() + settings.max_total_seconds + 30
    if cancel_event is None:
        cancel_event = threading.Event()

    # The agent runs in a worker thread, so its mutated state["language"] is
    # not visible from this generator. Pre-compute a UI-language pseudo-state
    # for SSE error chunks emitted from this side: prefer a carry-over from
    # the previous session (follow-up), otherwise fall back to the same
    # heuristic state.initial_state uses for the first progress event.
    _prev_lang = (prev_session or {}).get("language")
    _ui_lang = "de" if _prev_lang == "de" else detect_ui_language(question)
    ui_state: dict[str, Any] = {"language": _ui_lang}

    # OpenAI-compatible first chunk: role announcement
    role_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    # Progress queue for live updates from the agent
    progress_queue: Queue | None = Queue() if include_progress else None
    loop = asyncio.get_running_loop()

    # Start agent in a separate thread
    agent_future = loop.run_in_executor(
        None,
        partial(
            agent_run,
            question,
            history=history,
            progress_queue=progress_queue,
            prev_session=prev_session,
            providers=providers,
            strategies=strategies,
            settings=settings,
            cancel_event=cancel_event,
        ),
    )

    # Spawn the disconnect watcher when we have a real request. Tests
    # that pass request=None (library smoke tests) bypass this path.
    disconnect_watcher: asyncio.Task | None = None
    if request is not None and hasattr(request, "receive"):
        disconnect_watcher = asyncio.create_task(
            _watch_disconnect(request, cancel_event),
        )

    async def _shutdown_watcher() -> None:
        """Cancel and await the watcher task, swallowing the expected CancelledError."""
        if disconnect_watcher is None or disconnect_watcher.done():
            return
        disconnect_watcher.cancel()
        try:
            await disconnect_watcher
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — cleanup-only
            pass

    # Read progress updates and stream them as SSE chunks
    while include_progress and progress_queue is not None and not agent_future.done():
        if time.monotonic() >= request_deadline:
            await _shutdown_watcher()
            yield make_chunk(chat_id, t(ui_state, "sse_request_timeout"))
            yield make_chunk(chat_id, "", finish_reason="stop")
            yield "data: [DONE]\n\n"
            return
        if cancel_event.is_set():
            await _shutdown_watcher()
            return
        try:
            msg_type, msg_content = await loop.run_in_executor(
                None, partial(progress_queue.get, True, 0.3),
            )
            if msg_type == "progress" and msg_content != "done":
                yield make_chunk(chat_id, f"> `{msg_content}`\n>\n")
        except Empty:
            continue
        except Exception as exc:
            log.warning(
                "Progress-Streaming deaktiviert nach unerwartetem Fehler: %s",
                sanitize_error(exc),
            )
            break

    # Agent finished -- drain remaining queue messages
    while include_progress and progress_queue is not None and not progress_queue.empty():
        try:
            msg_type, msg_content = progress_queue.get_nowait()
            if msg_type == "progress" and msg_content != "done":
                yield make_chunk(chat_id, f"> `{msg_content}`\n>\n")
        except Empty:
            break
        except Exception as exc:
            log.warning(
                "Restliche Progress-Meldungen konnten nicht serialisiert werden: %s",
                sanitize_error(exc),
            )
            break

    # Get the result
    try:
        remaining = max(0.0, request_deadline - time.monotonic())
        if remaining <= 0.0:
            raise asyncio.TimeoutError
        result = await asyncio.wait_for(agent_future, timeout=remaining)
        answer_text = result["answer"]
    except asyncio.TimeoutError:
        await _shutdown_watcher()
        yield make_chunk(chat_id, t(ui_state, "sse_request_timeout"))
        yield make_chunk(chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        await _shutdown_watcher()
        log.error("Agent-Fehler: %s", e)
        yield make_chunk(
            chat_id,
            t(ui_state, "sse_agent_error", err=sanitize_error(e)),
        )
        yield make_chunk(chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Cancel-on-disconnect path: graph.run returns cancelled=True with an
    # empty answer. Skip session save and stop emitting (the client is gone).
    # Defense-in-depth: also short-circuit when the cancel_event is set
    # but the result-state lacks the cancelled marker (e.g. an agent that
    # finished one tick before the cancel probe could raise). This covers
    # the race where the watcher task fires after the agent already
    # returned naturally — semantically the client is still gone, so
    # streaming the answer would be wasted bandwidth.
    if (
        result.get("result_state", {}).get("cancelled")
        or cancel_event.is_set()
    ):
        await _shutdown_watcher()
        log.info("Run finished in cancelled state; skipping session save.")
        return

    # Save session snapshot for future follow-up questions; tag with the
    # resolved stack name so a multi-stack server keeps sessions isolated
    # per stack (ADR-MS-4). single-stack default stack_name="" reproduces
    # the historic hash exactly.
    if result.get("result_state") and messages is not None:
        try:
            save_id = prospective_session_id(messages, answer_text, stack_name=stack_name)
            if save_id:
                session_store.save(
                    save_id, result["result_state"], question, answer_text
                )
                log.info(
                    "Session %s gespeichert (%d Quellen, %d Context-Bloecke)",
                    save_id,
                    len(result["result_state"].get("all_citations", [])),
                    len(result["result_state"].get("context", [])),
                )
        except Exception:
            log.debug("Session-Save fehlgeschlagen (non-critical)", exc_info=True)

    # Separator between progress and answer
    if include_progress:
        yield make_chunk(chat_id, "\n\n---\n\n")

    # Stream answer word-by-word for better UX
    for token in iter_word_chunks(answer_text):
        yield make_chunk(chat_id, token)

    yield make_chunk(chat_id, "", finish_reason="stop")
    yield "data: [DONE]\n\n"
    # Normal completion: tear down the disconnect watcher so the
    # background task does not leak past the request lifecycle.
    await _shutdown_watcher()


async def guarded_stream(
    question: str,
    history: str,
    prev_session: dict[str, Any] | None,
    sem: asyncio.Semaphore,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
    session_store: SessionStore,
    messages: list[dict] | None = None,
    include_progress: bool = True,
    request: Request | None = None,
    cancel_event: threading.Event | None = None,
    stack_name: str = "",
) -> AsyncIterator[str]:
    """Stream with semaphore guard for correct concurrency limiting.

    The semaphore is held INSIDE the generator so it is only released
    after the streaming is complete. ``request`` and ``cancel_event``
    are forwarded to :func:`stream_response` so the disconnect probe
    and the implicit cancel pathway take effect. ``stack_name`` is
    propagated so the session snapshot saved at the end is keyed under
    the correct stack-isolated id.
    """
    async with sem:
        async for chunk in stream_response(
            question,
            history,
            prev_session,
            providers=providers,
            strategies=strategies,
            settings=settings,
            session_store=session_store,
            messages=messages,
            include_progress=include_progress,
            request=request,
            cancel_event=cancel_event,
            stack_name=stack_name,
        ):
            yield chunk
