"""FastAPI route definitions (health, models, chat completions, test)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from functools import partial
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inqtrix.graph import run as agent_run, run_test as agent_run_test
from inqtrix.providers.base import ProviderContext
from inqtrix.server.overrides import apply_overrides, parse_overrides_payload
from inqtrix.server.session import SessionStore, derive_session_id, prospective_session_id
from inqtrix.settings import AgentSettings, ServerSettings, Settings
from inqtrix.strategies import StrategyContext
from inqtrix.server.streaming import guarded_stream, MODEL_NAME
from inqtrix.urls import sanitize_error

log = logging.getLogger("inqtrix")


def create_router() -> APIRouter:
    """Create a fresh APIRouter instance (avoids module-level reuse)."""
    return APIRouter()


class _StackResolutionError(Exception):
    """Raised when the multi-stack registry cannot resolve body['stack']."""

    def __init__(self, message: str, available: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.available = available or []


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
    api_key_dependency: Any | None = None,
    stacks: dict[str, Any] | None = None,
    default_stack: str = "",
) -> None:
    """Bind all routes to *_router* with dependency injection.

    *semaphore_factory* is a callable that returns the
    :class:`asyncio.Semaphore` for concurrency limiting (lazy init
    because the event loop may not exist at import time).

    *api_key_dependency* is an optional FastAPI dependency callable
    (typically built by
    :func:`inqtrix.server.security.make_api_key_dependency`). When
    supplied, it is attached to ``/v1/chat/completions`` and
    ``/v1/test/run`` to enforce a Bearer-API-key gate. ``/health``
    and ``/v1/models`` deliberately remain unauthenticated so
    Kubernetes probes and model discovery clients keep working
    without credentials.

    *stacks* and *default_stack* are the multi-stack registry from
    :func:`inqtrix.server.stacks.create_multi_stack_app`. When
    *stacks* is non-None the routes resolve the per-request
    ``body["stack"]`` field, override providers/strategies/settings
    with that bundle, and tag the session id with the stack name so
    cross-stack follow-up cannot collide. When *stacks* is None the
    legacy single-stack path stays in effect (the *providers* /
    *strategies* / *settings* args are used as-is).
    """
    from fastapi import Depends

    auth_deps = [Depends(api_key_dependency)] if api_key_dependency is not None else []
    stacks_registry = stacks or {}

    def _resolve_request_stack(body: dict[str, Any]) -> tuple[str, Any | None]:
        """Pick the stack bundle for this request; raise via JSONResponse caller on miss.

        Returns ``(stack_name, bundle)`` where ``bundle`` is the
        ``StackBundle`` to use. ``stack_name`` is empty when no
        multi-stack registry was supplied (single-stack mode).
        """
        if not stacks_registry:
            return "", None
        requested = body.get("stack")
        if requested is None:
            return default_stack, stacks_registry[default_stack]
        if not isinstance(requested, str):
            raise _StackResolutionError(
                f"Field 'stack' must be a string, got {type(requested).__name__}"
            )
        if requested not in stacks_registry:
            raise _StackResolutionError(
                f"Unknown stack {requested!r}",
                available=sorted(stacks_registry.keys()),
            )
        return requested, stacks_registry[requested]

    # -- /health -------------------------------------------------------

    def _provider_label(provider: object) -> str:
        wrapped = getattr(provider, "_provider", None)
        if wrapped is not None:
            return type(wrapped).__name__
        return type(provider).__name__

    def _provider_ready(provider: object, *, label: str) -> bool:
        try:
            checker = getattr(provider, "is_available", None)
            return bool(checker()) if callable(checker) else False
        except Exception as exc:
            log.warning("Health-Check fuer %s fehlgeschlagen: %s", label, sanitize_error(exc))
            return False

    def _resolve_search_model(search_provider: object) -> str:
        """Read the standardized ``search_model`` property off the provider.

        Every search provider in :mod:`inqtrix.providers` overrides
        ``SearchProvider.search_model`` to return its operator-facing
        identifier (e.g. ``"sonar-pro"`` for Perplexity,
        ``"foundry-bing:my-agent@v1"`` for Foundry Bing). The default
        ABC implementation returns ``"<ClassName>(unknown)"`` so a
        custom subclass that forgets the override is loud rather than
        silently leaking the global ``Settings.models.search_model``.
        Falling back to ``settings.models.search_model`` is therefore
        a defensive last resort only when ``getattr`` finds nothing
        (older third-party SearchProvider subclasses pre-dating ADR-WS-12).
        """
        value = getattr(search_provider, "search_model", "")
        if isinstance(value, str) and value:
            return value
        return settings.models.search_model

    def _resolve_health_models(
        llm_provider: object,
        search_provider: object,
    ) -> dict[str, str]:
        """Return the effective per-role model names for the /health payload.

        Constructor-First (Designprinzip 6): every model name shown to
        operators must reflect what the provider was *actually* built
        with, not what the global ``settings.models`` block defaults to.
        Falling back to ``settings.models.*`` was the source of the
        ``claude-opus-4.6-agent`` confusion observed in the live test
        on Anthropic / Bedrock — the global default leaked into the
        health payload even though every real call used the provider's
        own model identifiers.
        """
        provider_models = getattr(llm_provider, "models", None)

        def _from(obj: object | None, attr: str, fallback: str) -> str:
            if obj is None:
                return fallback
            value = getattr(obj, attr, "")
            return value if value else fallback

        return {
            "reasoning_model": _from(
                provider_models, "reasoning_model", settings.models.reasoning_model
            ),
            "search_model": _resolve_search_model(search_provider),
            "classify_model": _from(
                provider_models, "effective_classify_model",
                settings.models.effective_classify_model,
            ),
            "summarize_model": _from(
                provider_models, "effective_summarize_model",
                settings.models.effective_summarize_model,
            ),
            "evaluate_model": _from(
                provider_models, "effective_evaluate_model",
                settings.models.effective_evaluate_model,
            ),
        }

    def _request_timeout_seconds(agent_settings: AgentSettings | None = None) -> int:
        active = agent_settings if agent_settings is not None else settings.agent
        return active.max_total_seconds + 30

    @_router.get("/health")
    def health():
        llm_label = _provider_label(providers.llm)
        search_label = _provider_label(providers.search)
        llm_ready = _provider_ready(providers.llm, label=llm_label)
        search_ready = _provider_ready(providers.search, label=search_label)
        status_code = 200 if llm_ready and search_ready else 503
        models_payload = _resolve_health_models(providers.llm, providers.search)
        payload = {
            "status": "ok" if status_code == 200 else "degraded",
            "llm": {
                "provider": llm_label,
                "status": "ready" if llm_ready else "unavailable",
            },
            "search": {
                "provider": search_label,
                "status": "ready" if search_ready else "unavailable",
            },
            "testing_mode": settings.agent.testing_mode,
            "report_profile": str(settings.agent.report_profile),
            **models_payload,
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

    @_router.post("/v1/test/run", dependencies=auth_deps)
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

        # Multi-stack resolution (no-op when stacks_registry is empty)
        try:
            _stack_name, stack_bundle = _resolve_request_stack(body)
        except _StackResolutionError as exc:
            content = {"error": {
                "message": exc.message,
                "type": "invalid_request_error",
            }}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        active_providers = stack_bundle.providers if stack_bundle is not None else providers
        active_strategies = stack_bundle.strategies if stack_bundle is not None else strategies
        active_agent_settings = (
            stack_bundle.agent_settings
            if stack_bundle is not None and stack_bundle.agent_settings is not None
            else settings.agent
        )

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(
                        agent_run_test,
                        question,
                        providers=active_providers,
                        strategies=active_strategies,
                        settings=active_agent_settings,
                    ),
                ),
                timeout=_request_timeout_seconds(active_agent_settings),
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

    @_router.post("/v1/chat/completions", dependencies=auth_deps)
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

        # Multi-stack resolution (ADR-MS-1/MS-2) — no-op in single-stack mode
        try:
            stack_name, stack_bundle = _resolve_request_stack(body)
        except _StackResolutionError as exc:
            content = {"error": {
                "message": exc.message,
                "type": "invalid_request_error",
            }}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        active_providers = stack_bundle.providers if stack_bundle is not None else providers
        active_strategies = stack_bundle.strategies if stack_bundle is not None else strategies
        base_agent_settings = (
            stack_bundle.agent_settings
            if stack_bundle is not None and stack_bundle.agent_settings is not None
            else settings.agent
        )

        # Per-request overrides (ADR-WS-6) — whitelist + range validated
        # in inqtrix.server.overrides; layered on top of the stack-resolved
        # AgentSettings so per-stack tuning + per-request UI sliders compose.
        try:
            overrides = parse_overrides_payload(body.get("agent_overrides"))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        per_request_agent_settings = apply_overrides(base_agent_settings, overrides)

        history = _format_history(
            messages, max_messages=settings.server.max_messages_history
        )

        # Session management for follow-up questions; stack name is part
        # of the hash so cross-stack follow-ups are isolated (ADR-MS-4).
        session_id = derive_session_id(messages, stack_name=stack_name)
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
            cancel_event = threading.Event()
            return StreamingResponse(
                guarded_stream(
                    question,
                    history,
                    prev_session,
                    sem,
                    providers=active_providers,
                    strategies=active_strategies,
                    settings=per_request_agent_settings,
                    session_store=session_store,
                    messages=messages,
                    include_progress=include_progress,
                    request=req,
                    cancel_event=cancel_event,
                    stack_name=stack_name,
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
                            providers=active_providers,
                            strategies=active_strategies,
                            settings=per_request_agent_settings,
                        ),
                    ),
                    timeout=_request_timeout_seconds(per_request_agent_settings),
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

            # Save session snapshot for future follow-ups; tag it with
            # the resolved stack name so the next turn against the
            # same stack reuses it (ADR-MS-4).
            if result.get("result_state"):
                try:
                    save_id = prospective_session_id(
                        messages, result["answer"], stack_name=stack_name
                    )
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
