"""SSE streaming utilities for OpenAI-compatible chat completion responses."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from functools import partial
from queue import Empty, Queue
from typing import Any, AsyncIterator

from inqtrix.graph import run as agent_run
from inqtrix.providers.base import ProviderContext
from inqtrix.session import SessionStore, prospective_session_id
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
) -> AsyncIterator[str]:
    """Execute the agent and yield progress updates + answer as SSE chunks."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    request_deadline = time.monotonic() + settings.max_total_seconds + 30

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
        ),
    )

    # Read progress updates and stream them as SSE chunks
    while include_progress and progress_queue is not None and not agent_future.done():
        if time.monotonic() >= request_deadline:
            yield make_chunk(
                chat_id,
                "\n---\n\n⚠️ **Fehler bei der Recherche:** Request-Timeout erreicht",
            )
            yield make_chunk(chat_id, "", finish_reason="stop")
            yield "data: [DONE]\n\n"
            return
        try:
            msg_type, msg_content = await loop.run_in_executor(
                None, partial(progress_queue.get, True, 0.3),
            )
            if msg_type == "progress" and msg_content != "done":
                yield make_chunk(chat_id, f"> `{msg_content}`\n>\n")
        except Empty:
            continue
        except Exception:
            break

    # Agent finished -- drain remaining queue messages
    while include_progress and progress_queue is not None and not progress_queue.empty():
        try:
            msg_type, msg_content = progress_queue.get_nowait()
            if msg_type == "progress" and msg_content != "done":
                yield make_chunk(chat_id, f"> `{msg_content}`\n>\n")
        except Empty:
            break

    # Get the result
    try:
        remaining = max(0.0, request_deadline - time.monotonic())
        if remaining <= 0.0:
            raise asyncio.TimeoutError
        result = await asyncio.wait_for(agent_future, timeout=remaining)
        answer_text = result["answer"]
    except asyncio.TimeoutError:
        yield make_chunk(
            chat_id,
            "\n---\n\n⚠️ **Fehler bei der Recherche:** Request-Timeout erreicht",
        )
        yield make_chunk(chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        log.error("Agent-Fehler: %s", e)
        yield make_chunk(
            chat_id,
            f"\n---\n\n⚠️ **Fehler bei der Recherche:** {sanitize_error(e)}",
        )
        yield make_chunk(chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Save session snapshot for future follow-up questions
    if result.get("result_state") and messages is not None:
        try:
            save_id = prospective_session_id(messages, answer_text)
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
) -> AsyncIterator[str]:
    """Stream with semaphore guard for correct concurrency limiting.

    The semaphore is held INSIDE the generator so it is only released
    after the streaming is complete.
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
        ):
            yield chunk
