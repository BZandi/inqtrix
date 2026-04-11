"""FastAPI route definitions (health, models, chat completions, test)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from functools import partial
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inqtrix.graph import run as agent_run, run_test as agent_run_test
from inqtrix.providers.base import ProviderContext
from inqtrix.server.session import SessionStore, derive_session_id, prospective_session_id
from inqtrix.settings import AgentSettings, ServerSettings, Settings
from inqtrix.strategies import StrategyContext
from inqtrix.server.streaming import guarded_stream, MODEL_NAME
from inqtrix.urls import sanitize_error

log = logging.getLogger("inqtrix")


def create_router() -> APIRouter:
    """Create a fresh APIRouter instance (avoids module-level reuse)."""
    return APIRouter()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _format_history(
    messages: list[dict], max_messages: int = 20
) -> str:
    """Format message history for agent context."""
    if len(messages) <= 1:
        return ""
    history_msgs = messages[:-1][-max_messages:]
    parts: list[str] = []
    for msg in history_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if content:
            label = {"user": "Nutzer", "assistant": "Assistent",
                     "system": "System"}.get(role, role)
            parts.append(f"{label}: {content[:500]}")
    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Route factory
# ------------------------------------------------------------------ #


def register_routes(
    _router: APIRouter,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: Settings,
    session_store: SessionStore,
    semaphore_factory: Any,
) -> None:
    """Bind all routes to *_router* with dependency injection.

    *semaphore_factory* is a callable that returns the
    :class:`asyncio.Semaphore` for concurrency limiting (lazy init
    because the event loop may not exist at import time).
    """

    # -- /health -------------------------------------------------------

    def _provider_label(provider: object) -> str:
        wrapped = getattr(provider, "_provider", None)
        if wrapped is not None:
            return type(wrapped).__name__
        return type(provider).__name__

    def _provider_available(provider: object, *, label: str) -> bool:
        try:
            checker = getattr(provider, "is_available", None)
            return bool(checker()) if callable(checker) else False
        except Exception as exc:
            log.warning("Health-Check fuer %s fehlgeschlagen: %s", label, sanitize_error(exc))
            return False

    def _request_timeout_seconds() -> int:
        return settings.agent.max_total_seconds + 30

    @_router.get("/health")
    def health():
        llm_label = _provider_label(providers.llm)
        search_label = _provider_label(providers.search)
        llm_available = _provider_available(providers.llm, label=llm_label)
        search_available = _provider_available(providers.search, label=search_label)
        status_code = 200 if llm_available and search_available else 503
        payload = {
            "status": "ok" if status_code == 200 else "degraded",
            "llm": {
                "provider": llm_label,
                "status": "connected" if llm_available else "unavailable",
            },
            "search": {
                "provider": search_label,
                "status": "connected" if search_available else "unavailable",
            },
            "testing_mode": settings.agent.testing_mode,
            "reasoning_model": settings.models.reasoning_model,
            "search_model": settings.models.search_model,
            "classify_model": settings.models.effective_classify_model,
            "summarize_model": settings.models.effective_summarize_model,
            "evaluate_model": settings.models.effective_evaluate_model,
            "high_risk_score_threshold": settings.agent.high_risk_score_threshold,
            "high_risk_classify_escalate": settings.agent.high_risk_classify_escalate,
            "high_risk_evaluate_escalate": settings.agent.high_risk_evaluate_escalate,
        }
        if status_code == 200:
            return {
                **payload,
            }
        return JSONResponse(status_code=status_code, content=payload)

    # -- /v1/models ----------------------------------------------------

    @_router.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": 0,
                    "owned_by": "inqtrix",
                }
            ],
        }

    # -- /v1/test/run --------------------------------------------------

    @_router.post("/v1/test/run")
    async def test_run(req: Request):
        """Run a single test question and return structured metrics."""
        if not settings.agent.testing_mode:
            return JSONResponse(
                status_code=404,
                content={"error": {
                    "message": "Test-Endpoint nur im Testing-Modus verfuegbar",
                    "type": "not_found",
                }},
            )

        try:
            body = await req.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "Ungueltiger JSON-Body",
                    "type": "invalid_request_error",
                }},
            )

        question = body.get("question", "")
        if not question or not isinstance(question, str):
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "Feld 'question' (String) ist erforderlich",
                    "type": "invalid_request_error",
                }},
            )

        if len(question) > settings.agent.max_question_length:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": (
                        f"Frage zu lang ({len(question)} Zeichen, "
                        f"max. {settings.agent.max_question_length})"
                    ),
                    "type": "invalid_request_error",
                }},
            )

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(
                        agent_run_test,
                        question,
                        providers=providers,
                        strategies=strategies,
                        settings=settings.agent,
                    ),
                ),
                timeout=_request_timeout_seconds(),
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": {
                    "message": "Test-Durchlauf Timeout",
                    "type": "timeout_error",
                }},
            )
        except Exception as e:
            log.error("Test-Durchlauf Fehler: %s", e)
            return JSONResponse(
                status_code=502,
                content={"error": {
                    "message": f"Agent-Fehler: {sanitize_error(e)}",
                    "type": "server_error",
                }},
            )

        return result

    # -- /v1/chat/completions ------------------------------------------

    @_router.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        try:
            body = await req.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "Ungueltiger JSON-Body",
                    "type": "invalid_request_error",
                }},
            )

        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "messages darf nicht leer sein",
                    "type": "invalid_request_error",
                }},
            )

        question = messages[-1].get("content", "")
        if isinstance(question, list):
            question = " ".join(
                p.get("text", "") for p in question
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "Letzte Nachricht hat keinen Inhalt",
                    "type": "invalid_request_error",
                }},
            )

        if len(question) > settings.agent.max_question_length:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": (
                        f"Frage zu lang ({len(question)} Zeichen, "
                        f"max. {settings.agent.max_question_length})"
                    ),
                    "type": "invalid_request_error",
                }},
            )

        history = _format_history(
            messages, max_messages=settings.server.max_messages_history
        )

        # Session management for follow-up questions
        session_id = derive_session_id(messages)
        prev_session: dict[str, Any] | None = None
        if session_id:
            snap = session_store.get(session_id)
            if snap is not None:
                prev_session = snap.__dict__
                log.info("Session %s gefunden -- Follow-up-Modus aktiv", session_id)

        stream = body.get("stream", False)
        include_progress_raw = body.get("include_progress", True)
        include_progress = (
            include_progress_raw
            if isinstance(include_progress_raw, bool)
            else True
        )

        # Concurrency check
        sem = semaphore_factory()
        if sem.locked():
            return JSONResponse(
                status_code=429,
                content={"error": {
                    "message": "Zu viele gleichzeitige Anfragen. Bitte warten.",
                    "type": "rate_limit_error",
                }},
            )

        if stream:
            return StreamingResponse(
                guarded_stream(
                    question,
                    history,
                    prev_session,
                    sem,
                    providers=providers,
                    strategies=strategies,
                    settings=settings.agent,
                    session_store=session_store,
                    messages=messages,
                    include_progress=include_progress,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            agent_run,
                            question,
                            history=history,
                            prev_session=prev_session,
                            providers=providers,
                            strategies=strategies,
                            settings=settings.agent,
                        ),
                    ),
                    timeout=_request_timeout_seconds(),
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"error": {
                        "message": "Recherche-Request Timeout",
                        "type": "timeout_error",
                    }},
                )
            except Exception as e:
                log.error("Agent-Fehler: %s", e)
                return JSONResponse(
                    status_code=502,
                    content={"error": {
                        "message": f"Agent-Fehler: {sanitize_error(e)}",
                        "type": "server_error",
                    }},
                )

            # Save session snapshot for future follow-ups
            if result.get("result_state"):
                try:
                    save_id = prospective_session_id(messages, result["answer"])
                    if save_id:
                        session_store.save(
                            save_id,
                            result["result_state"],
                            question,
                            result["answer"],
                        )
                        log.info(
                            "Session %s gespeichert (%d Quellen, %d Context-Bloecke)",
                            save_id,
                            len(result["result_state"].get("all_citations", [])),
                            len(result["result_state"].get("context", [])),
                        )
                except Exception:
                    log.debug(
                        "Session-Save fehlgeschlagen (non-critical)", exc_info=True
                    )

            usage = result.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result["answer"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                },
            }
