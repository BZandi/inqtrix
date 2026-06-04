"""FastAPI route definitions (health, models, chat completions, test)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from functools import partial
from queue import Empty
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inqtrix.exceptions import AgentStructuredOutputError
from inqtrix.graph import run as agent_run, run_test as agent_run_test
from inqtrix.legal import legal_metadata
from inqtrix.model_routing import (
    describe_chat_model_options,
    describe_node_resolutions,
    resolve_effort,
    resolve_model,
)
from inqtrix.providers.base import ProviderContext
from inqtrix.result import ResearchResult
from inqtrix.server.editor_suggestions import (
    EDITOR_SUGGEST_SCHEMA,
    EDITOR_SUGGEST_SCHEMA_NAME,
    EditorSuggestError,
    build_editor_suggest_prompt,
    clamp_background,
    parse_editor_suggest_payload,
    parse_editor_suggest_response,
    result_from_parsed,
    validate_editor_suggest_result,
    warnings_for_validation_issues,
)
from inqtrix.server.editor_instructions import (
    EDITOR_INSTRUCT_SCHEMA,
    EDITOR_INSTRUCT_SCHEMA_NAME,
    EditorInstructError,
    build_editor_instruct_prompt,
    parse_editor_instruct_payload,
    parse_editor_instruct_response,
    result_from_parsed as result_from_instruction_parsed,
    validate_editor_instruct_result,
)
from inqtrix.server.reference_documents import (
    clamp_reference_documents,
    render_reference_documents,
)
from inqtrix.server.overrides import (
    AgentOverridesRequest,
    apply_overrides,
    parse_overrides_payload,
)
from inqtrix.server.runs import (
    RunHandle,
    RunNotFound,
    RunQueueFull,
    RunStore,
    format_sse_event,
)
from inqtrix.settings import AgentSettings, Settings
from inqtrix.strategies import StrategyContext
from inqtrix.server.streaming import guarded_stream, MODEL_NAME
from inqtrix.server.text_improvements import (
    TextImprovementError,
    build_text_improvement_prompt,
    parse_text_improvement_payload,
    parse_text_improvement_response,
)
from inqtrix.state import build_run_snapshot
from inqtrix.urls import sanitize_error

log = logging.getLogger("inqtrix")

RunMode = Literal["research", "direct_llm"]
_WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_WORKSPACE_ID_HEADER = "x-inqtrix-workspace-id"
_EDITOR_INSTRUCT_CONTEXT_FLOOR_TOKENS = 128_000
_EDITOR_INSTRUCT_RESERVED_PROMPT_TOKENS = 6_000


def create_router() -> APIRouter:
    """Create a fresh APIRouter instance (avoids module-level reuse)."""
    return APIRouter()


class _StackResolutionError(Exception):
    """Raised when the multi-stack registry cannot resolve body['stack']."""

    def __init__(self, message: str, available: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.available = available or []


@dataclass(frozen=True)
class _ResolvedAgentContext:
    """Stack, settings, and run-mode resolved from one HTTP request."""

    stack_name: str
    providers: ProviderContext
    strategies: StrategyContext
    agent_settings: AgentSettings
    agent_overrides: dict[str, Any]
    mode: RunMode


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


def _workspace_id_from_request(
    req: Request,
    body: dict[str, Any] | None = None,
) -> str | None:
    """Resolve and validate the optional browser/project workspace namespace."""
    body_workspace_id = body.get("workspace_id") if isinstance(body, dict) else None
    raw = (
        req.headers.get(_WORKSPACE_ID_HEADER)
        or req.query_params.get("workspace_id")
        or body_workspace_id
    )
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not _WORKSPACE_ID_PATTERN.fullmatch(raw):
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "Invalid workspace_id.",
                "type": "invalid_request_error",
            }},
        )
    return raw


# ------------------------------------------------------------------ #
# Route factory
# ------------------------------------------------------------------ #


def register_routes(
    _router: APIRouter,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: Settings,
    semaphore_factory: Any,
    api_key_dependency: Any | None = None,
    stacks: dict[str, Any] | None = None,
    default_stack: str = "",
    run_store: RunStore | None = None,
) -> None:
    """Bind all routes to *_router* with dependency injection.

    *semaphore_factory* is a callable that returns the
    :class:`asyncio.Semaphore` for concurrency limiting (lazy init
    because the event loop may not exist at import time).

    *api_key_dependency* is an optional FastAPI dependency callable
    (typically built by
    :func:`inqtrix.server.security.make_api_key_dependency`). When
    supplied, it is attached to ``/v1/chat/completions``,
    ``/v1/text/improvements``, ``/v1/test/run``, and native run routes
    to enforce a Bearer-API-key gate. ``/health`` and ``/v1/models``
    deliberately remain unauthenticated so
    Kubernetes probes and model discovery clients keep working
    without credentials.

    *stacks* and *default_stack* are the multi-stack registry from
    :func:`inqtrix.server.stacks.create_multi_stack_app`. When
    *stacks* is non-None the routes resolve the per-request
    ``body["stack"]`` field, override providers/strategies/settings
    with that bundle. When *stacks* is None the single-stack path stays
    in effect (the *providers* / *strategies* / *settings* args are
    used as-is).
    """
    from fastapi import Depends

    auth_deps = [Depends(api_key_dependency)] if api_key_dependency is not None else []
    stacks_registry = stacks or {}
    active_run_store = run_store or RunStore.from_settings(settings.server)

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
        ``"foundry-web:my-agent@v1"`` for Foundry web search). The default
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
        agent_settings: AgentSettings,
    ) -> dict[str, Any]:
        """Return the effective per-role + per-node model names for /health.

        Constructor-First (Designprinzip 6): every model name shown to
        operators must reflect what the provider was *actually* built
        with, not what the global ``settings.models`` block defaults to.
        Falling back to ``settings.models.*`` was the source of the
        ``claude-opus-4.6-agent`` confusion observed in the live test
        on Anthropic / Bedrock — the global default leaked into the
        health payload even though every real call used the provider's
        own model identifiers.

        The ``node_models`` block reports, per call site, the model and
        reasoning effort the graph would actually route to (with
        ``model_source`` / ``effort_source`` provenance), so an operator can
        see e.g. ``answer -> opus`` vs ``classify -> haiku`` and tell when a
        ``reasoning_model`` default grips instead of a tier — the same
        resolution used at runtime (Designprinzip 4/5).
        """
        provider_models = getattr(llm_provider, "models", None)
        requested_tier = (agent_settings.model_tier or "").strip() or None
        node_models = describe_node_resolutions(provider_models, requested_tier)

        def _from_node(node: str) -> str:
            return node_models.get(node, {}).get("model", "")

        return {
            "reasoning_model": (
                getattr(provider_models, "reasoning_model", "")
                if provider_models is not None
                else ""
            ),
            "search_model": _resolve_search_model(search_provider),
            "classify_model": _from_node("classify"),
            "claim_extract_model": _from_node("claim_extract"),
            "evaluate_model": _from_node("evaluate"),
            "node_models": node_models,
            "chat_model_options": describe_chat_model_options(provider_models),
        }

    def _health_agent_settings() -> AgentSettings:
        if stacks_registry and default_stack in stacks_registry:
            stack_bundle = stacks_registry[default_stack]
            stack_settings = getattr(stack_bundle, "agent_settings", None)
            if stack_settings is not None:
                return stack_settings
        return settings.agent

    def _request_timeout_seconds(agent_settings: AgentSettings | None = None) -> int:
        active = agent_settings if agent_settings is not None else settings.agent
        return active.max_total_seconds + 30

    def _error_response(
        status_code: int,
        message: str,
        error_type: str,
        **extra: Any,
    ) -> JSONResponse:
        """Return the existing OpenAI-style error envelope."""
        error = {"message": message, "type": error_type}
        error.update(extra)
        return JSONResponse(status_code=status_code, content={"error": error})

    def _text_from_content(content: Any) -> str:
        """Extract user-visible text from OpenAI chat content."""
        if isinstance(content, list):
            return " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return str(content or "")

    def _enforce_payload_caps(question: str, messages: list[dict[str, Any]]) -> None:
        """Reject oversize ``messages[]`` arrays with HTTP 413.

        Two caps from :class:`ServerSettings`:

        * ``max_message_count`` rejects array-bomb payloads that would
          stretch the agent's token bookkeeping for no real-user reason.
        * ``max_total_input_tokens`` rejects bodies whose combined
          *question + messages content* approximate-token count
          (4 chars per token) exceeds the operator-configured limit.
          Defends against pathological 1-message-with-megabytes payloads.

        The defaults are deliberately generous (200 messages, 500k
        tokens) so realistic multi-turn flows never trip them. The
        check runs after content normalization so it sees what the
        agent will actually consume.
        """
        max_count = settings.server.max_message_count
        if len(messages) > max_count:
            raise HTTPException(
                status_code=413,
                detail={"error": {
                    "message": (
                        f"messages array exceeds limit ({len(messages)} > "
                        f"{max_count})"
                    ),
                    "type": "payload_too_large",
                }},
            )
        total_chars = sum(
            len(_text_from_content(msg.get("content", ""))) for msg in messages
        ) + len(question)
        approx_tokens = total_chars // 4
        max_tokens = settings.server.max_total_input_tokens
        if approx_tokens > max_tokens:
            raise HTTPException(
                status_code=413,
                detail={"error": {
                    "message": (
                        f"input size ~{approx_tokens} tokens exceeds limit "
                        f"({max_tokens})"
                    ),
                    "type": "payload_too_large",
                }},
            )

    def _question_and_messages(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        """Resolve either native ``question`` or OpenAI ``messages`` input."""
        raw_question = body.get("question")
        messages = body.get("messages", [])
        if isinstance(raw_question, str) and raw_question.strip():
            question = raw_question.strip()
            normalized_messages = (
                messages
                if (
                    isinstance(messages, list)
                    and messages
                    and all(isinstance(msg, dict) for msg in messages)
                )
                else [{"role": "user", "content": question}]
            )
            _enforce_payload_caps(question, normalized_messages)
            return question, normalized_messages
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400,
                detail={"error": {
                    "message": "Feld 'question' oder nicht-leere 'messages' ist erforderlich",
                    "type": "invalid_request_error",
                }},
            )
        if not all(isinstance(msg, dict) for msg in messages):
            raise HTTPException(
                status_code=400,
                detail={"error": {
                    "message": "messages muss eine Liste von Objekten sein",
                    "type": "invalid_request_error",
                }},
            )
        last = messages[-1]
        question = _text_from_content(last.get("content", "")).strip()
        if not question:
            raise HTTPException(
                status_code=400,
                detail={"error": {
                    "message": "Letzte Nachricht hat keinen Inhalt",
                    "type": "invalid_request_error",
                }},
            )
        _enforce_payload_caps(question, messages)
        return question, messages

    def _parse_mode_payload(body: dict[str, Any]) -> RunMode | None:
        """Validate the optional top-level run mode field."""
        raw_mode = body.get("mode")
        if raw_mode is None:
            return None
        if raw_mode == "research" or raw_mode == "direct_llm":
            return raw_mode
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "mode muss 'research' oder 'direct_llm' sein",
                "type": "invalid_request_error",
            }},
        )

    def _resolve_mode_settings(
        *,
        agent_settings: AgentSettings,
        overrides: AgentOverridesRequest | None,
        requested_mode: RunMode | None,
    ) -> tuple[AgentSettings, RunMode]:
        """Apply explicit mode semantics on top of resolved agent settings."""
        if requested_mode is None:
            return (
                agent_settings,
                "direct_llm" if agent_settings.skip_search else "research",
            )

        has_skip_override = (
            overrides is not None
            and "skip_search" in getattr(overrides, "model_fields_set", set())
            and overrides.skip_search is not None
        )
        if has_skip_override:
            if requested_mode == "direct_llm" and overrides.skip_search is False:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "message": (
                            "mode='direct_llm' widerspricht "
                            "agent_overrides.skip_search=false"
                        ),
                        "type": "invalid_request_error",
                    }},
                )
            if requested_mode == "research" and overrides.skip_search is True:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "message": (
                            "mode='research' widerspricht "
                            "agent_overrides.skip_search=true"
                        ),
                        "type": "invalid_request_error",
                    }},
                )

        skip_search = requested_mode == "direct_llm"
        if agent_settings.skip_search is skip_search:
            return agent_settings, requested_mode
        return agent_settings.model_copy(update={"skip_search": skip_search}), requested_mode

    def _resolve_agent_context(body: dict[str, Any]) -> _ResolvedAgentContext:
        """Resolve stack, per-request overrides, and explicit run mode."""
        stack_name, stack_bundle = _resolve_request_stack(body)
        active_providers = stack_bundle.providers if stack_bundle is not None else providers
        active_strategies = stack_bundle.strategies if stack_bundle is not None else strategies
        base_agent_settings = (
            stack_bundle.agent_settings
            if stack_bundle is not None and stack_bundle.agent_settings is not None
            else settings.agent
        )
        overrides = parse_overrides_payload(body.get("agent_overrides"))
        agent_overrides = (
            overrides.model_dump(mode="json", exclude_none=True)
            if overrides is not None
            else {}
        )
        requested_mode = _parse_mode_payload(body)
        agent_settings = apply_overrides(base_agent_settings, overrides)
        agent_settings, mode = _resolve_mode_settings(
            agent_settings=agent_settings,
            overrides=overrides,
            requested_mode=requested_mode,
        )
        return _ResolvedAgentContext(
            stack_name=stack_name,
            providers=active_providers,
            strategies=active_strategies,
            agent_settings=agent_settings,
            agent_overrides=agent_overrides,
            mode=mode,
        )

    def _chat_settings_for_question(
        agent_settings: AgentSettings,
        question: str,
    ) -> AgentSettings:
        """Return request-local settings for direct chat with large attachments.

        ``max_question_length`` is a character guard for research questions.
        Direct chat already passed the route-level aggregate token cap, and the
        chat composer may inline attached documents into the final user
        message. In that mode, raise only the local character guard so the
        central graph check does not reject otherwise accepted payloads.

        Args:
            agent_settings: Resolved settings for this request after stack,
                override, and mode handling.
            question: The normalized current user message that will be passed
                to the agent entry point.

        Returns:
            The original settings for normal research or already-short direct
            chat requests; otherwise a request-local copy with
            ``max_question_length`` raised to the current message length.
        """
        if (
            not agent_settings.skip_search
            or len(question) <= agent_settings.max_question_length
        ):
            return agent_settings
        return agent_settings.model_copy(
            update={"max_question_length": len(question)}
        )

    @_router.get("/health")
    def health():
        llm_label = _provider_label(providers.llm)
        search_label = _provider_label(providers.search)
        llm_ready = _provider_ready(providers.llm, label=llm_label)
        search_ready = _provider_ready(providers.search, label=search_label)
        status_code = 200 if llm_ready and search_ready else 503
        active_agent_settings = _health_agent_settings()
        models_payload = _resolve_health_models(
            providers.llm, providers.search, active_agent_settings
        )
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
            "testing_mode": active_agent_settings.testing_mode,
            "report_profile": str(active_agent_settings.report_profile),
            **models_payload,
            "high_risk_score_threshold": active_agent_settings.high_risk_score_threshold,
            "model_tier": active_agent_settings.model_tier,
            "auth_required": api_key_dependency is not None,
            "legal": legal_metadata(),
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

    # -- /v1/text/improvements ----------------------------------------

    @_router.post("/v1/text/improvements", dependencies=auth_deps)
    async def improve_text(req: Request):
        """Improve one browser text field without creating an agent run."""
        try:
            body = await req.json()
        except Exception:
            return _error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            _workspace_id_from_request(req, body)
            resolved = _resolve_agent_context(body)
            improvement_request = parse_text_improvement_payload(
                body,
                max_text_chars=resolved.agent_settings.max_question_length,
            )
        except ValueError as exc:
            return _error_response(400, str(exc), "invalid_request_error")
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except _StackResolutionError as exc:
            content = {"error": {
                "message": exc.message,
                "type": "invalid_request_error",
            }}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        sem = semaphore_factory()
        if sem.locked():
            return _error_response(
                429,
                "Zu viele gleichzeitige Anfragen. Bitte warten.",
                "rate_limit_error",
            )

        prompt = build_text_improvement_prompt(improvement_request)
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                raw_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            resolved.providers.llm.complete,
                            prompt,
                            max_output_tokens=2500,
                            timeout=resolved.agent_settings.claim_extract_timeout,
                        ),
                    ),
                    timeout=min(
                        _request_timeout_seconds(resolved.agent_settings),
                        resolved.agent_settings.claim_extract_timeout + 30,
                    ),
                )
                result = parse_text_improvement_response(raw_response)
            except asyncio.TimeoutError:
                return _error_response(
                    504,
                    "Textverbesserung Timeout",
                    "timeout_error",
                )
            except TextImprovementError as exc:
                log.warning("Textverbesserung konnte nicht geparst werden: %s", exc)
                return _error_response(502, str(exc), "server_error")
            except Exception as exc:
                log.error("Textverbesserung Fehler: %s", exc)
                return _error_response(
                    502,
                    f"Textverbesserung Fehler: {sanitize_error(exc)}",
                    "server_error",
                )

        return result.to_payload()

    # -- /v1/editor/suggest -------------------------------------------
    @_router.post("/v1/editor/suggest", dependencies=auth_deps)
    async def editor_suggest(req: Request):
        """Rewrite one editor paragraph with the LLM (no agent run).

        Serves the editor's Direkt single-paragraph edit and each per-paragraph
        call of a Sammeln global run. Unlike ``/v1/text/improvements`` it routes
        by the composer-selected tier (``agent_overrides.model_tier`` via the
        ``direct_chat`` node) and prefers native structured output, with a
        visible prompt-JSON fallback.
        """
        try:
            body = await req.json()
        except Exception:
            return _error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            _workspace_id_from_request(req, body)
            resolved = _resolve_agent_context(body)
            suggest_request = parse_editor_suggest_payload(
                body,
                max_text_chars=resolved.agent_settings.max_question_length,
                max_background_chars=400_000,
            )
        except ValueError as exc:
            return _error_response(400, str(exc), "invalid_request_error")
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except _StackResolutionError as exc:
            content = {"error": {"message": exc.message, "type": "invalid_request_error"}}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        sem = semaphore_factory()
        if sem.locked():
            return _error_response(
                429,
                "Zu viele gleichzeitige Anfragen. Bitte warten.",
                "rate_limit_error",
            )

        llm = resolved.providers.llm
        provider_models = getattr(llm, "models", None)
        requested_tier = resolved.agent_settings.model_tier or None
        warnings: list[str] = []
        if provider_models is not None:
            model = resolve_model("direct_chat", provider_models, requested_tier) or None
            effort = resolve_effort("direct_chat", provider_models, requested_tier) or None
        else:
            # Custom provider without published model metadata: use the
            # provider's own default model and surface that (Designprinzip 1).
            model = None
            effort = None
            warnings.append("Modellauswahl ueber Provider-Default (keine Tier-Metadaten).")

        context_window = getattr(llm, "context_window_tokens", None)
        budget_tokens = max(2000, (context_window or 16000) - 6000)
        background, truncated = clamp_background(
            suggest_request.background,
            suggest_request.block_text,
            max_chars=budget_tokens * 3,
        )
        if truncated:
            warnings.append("Bericht-Kontext gekuerzt (Umgebungsausschnitt).")
        suggest_request = replace(suggest_request, background=background)

        reference_budget = max(0, budget_tokens * 3 - len(background))
        reference_docs, reference_truncated = clamp_reference_documents(
            suggest_request.reference_documents,
            max_chars=reference_budget,
        )
        warnings.extend(suggest_request.reference_warnings)
        if reference_truncated:
            warnings.append("Reference documents truncated (context budget).")
            log.warning("Editor suggestion reference documents truncated to fit the context budget.")
        reference_block = render_reference_documents(reference_docs)

        timeout = resolved.agent_settings.claim_extract_timeout
        wait_timeout = min(_request_timeout_seconds(resolved.agent_settings), timeout + 30)
        async with sem:
            loop = asyncio.get_running_loop()

            async def complete_editor_prompt(
                prompt: str,
                prompt_warnings: list[str],
            ):
                if llm.supports_structured_output(model=model):
                    structured = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(
                                llm.complete_structured,
                                prompt,
                                schema=EDITOR_SUGGEST_SCHEMA,
                                schema_name=EDITOR_SUGGEST_SCHEMA_NAME,
                                model=model,
                                reasoning_effort=effort,
                                max_output_tokens=4000,
                                timeout=timeout,
                            ),
                        ),
                        timeout=wait_timeout,
                    )
                    return result_from_parsed(structured.parsed, warnings=prompt_warnings)
                raw_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            llm.complete,
                            prompt,
                            model=model,
                            reasoning_effort=effort,
                            max_output_tokens=4000,
                            timeout=timeout,
                        ),
                    ),
                    timeout=wait_timeout,
                )
                return parse_editor_suggest_response(raw_response, warnings=prompt_warnings)

            try:
                prompt = build_editor_suggest_prompt(
                    suggest_request,
                    reference_block=reference_block,
                )
                result = await complete_editor_prompt(prompt, warnings)
                validation_issues = validate_editor_suggest_result(suggest_request, result)
                if validation_issues:
                    log.warning(
                        "Editor-Vorschlag verletzt Editiervertrag: %s",
                        ", ".join(issue.code for issue in validation_issues),
                    )
                    retry_prompt = build_editor_suggest_prompt(
                        suggest_request,
                        previous_result=result.rewritten_text,
                        validation_issues=validation_issues,
                        reference_block=reference_block,
                    )
                    result = await complete_editor_prompt(retry_prompt, warnings)
                    validation_issues = validate_editor_suggest_result(suggest_request, result)
                    if validation_issues:
                        result = replace(
                            result,
                            warnings=[
                                *result.warnings,
                                *warnings_for_validation_issues(
                                    validation_issues,
                                    locale=suggest_request.locale,
                                ),
                            ],
                        )
            except asyncio.TimeoutError:
                return _error_response(504, "Editor-Vorschlag Timeout", "timeout_error")
            except (EditorSuggestError, AgentStructuredOutputError) as exc:
                log.warning("Editor-Vorschlag konnte nicht geparst werden: %s", exc)
                return _error_response(502, str(exc), "server_error")
            except Exception as exc:
                log.error("Editor-Vorschlag Fehler: %s", exc)
                return _error_response(
                    502,
                    f"Editor-Vorschlag Fehler: {sanitize_error(exc)}",
                    "server_error",
                )

        return result.to_payload()

    # -- /v1/editor/instruct ------------------------------------------
    @_router.post("/v1/editor/instruct", dependencies=auth_deps)
    async def editor_instruct(req: Request):
        """Turn one free-form editor instruction into anchored document edits."""
        try:
            body = await req.json()
        except Exception:
            return _error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            _workspace_id_from_request(req, body)
            resolved = _resolve_agent_context(body)
            instruct_request = parse_editor_instruct_payload(
                body,
                max_instruction_chars=resolved.agent_settings.max_question_length,
                max_document_chars=400_000,
            )
        except ValueError as exc:
            return _error_response(400, str(exc), "invalid_request_error")
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except _StackResolutionError as exc:
            content = {"error": {"message": exc.message, "type": "invalid_request_error"}}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        sem = semaphore_factory()
        if sem.locked():
            return _error_response(
                429,
                "Zu viele gleichzeitige Anfragen. Bitte warten.",
                "rate_limit_error",
            )

        llm = resolved.providers.llm
        context_window = getattr(llm, "context_window_tokens", None)
        if not isinstance(context_window, int) or context_window <= 0:
            context_window = None
        effective_context_window = max(
            context_window or 0,
            _EDITOR_INSTRUCT_CONTEXT_FLOOR_TOKENS,
        )
        budget_chars = max(
            2000,
            effective_context_window - _EDITOR_INSTRUCT_RESERVED_PROMPT_TOKENS,
        ) * 3
        if len(instruct_request.document_markdown) > budget_chars:
            return _error_response(
                400,
                "Dokument zu groß für eine Komplettüberarbeitung.",
                "invalid_request_error",
            )

        provider_models = getattr(llm, "models", None)
        requested_tier = resolved.agent_settings.model_tier or None
        warnings: list[str] = []
        if provider_models is not None:
            model = resolve_model("direct_chat", provider_models, requested_tier) or None
            effort = resolve_effort("direct_chat", provider_models, requested_tier) or None
        else:
            model = None
            effort = None
            warnings.append("Modellauswahl ueber Provider-Default (keine Tier-Metadaten).")

        reference_budget = max(0, budget_chars - len(instruct_request.document_markdown))
        reference_docs, reference_truncated = clamp_reference_documents(
            instruct_request.reference_documents,
            max_chars=reference_budget,
        )
        warnings.extend(instruct_request.reference_warnings)
        if reference_truncated:
            warnings.append("Reference documents truncated (context budget).")
            log.warning("Editor instruction reference documents truncated to fit the context budget.")
        reference_block = render_reference_documents(reference_docs)

        timeout = resolved.agent_settings.claim_extract_timeout
        wait_timeout = min(_request_timeout_seconds(resolved.agent_settings), timeout + 30)
        async with sem:
            loop = asyncio.get_running_loop()

            async def complete_instruction_prompt(prompt: str):
                if llm.supports_structured_output(model=model):
                    structured = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(
                                llm.complete_structured,
                                prompt,
                                schema=EDITOR_INSTRUCT_SCHEMA,
                                schema_name=EDITOR_INSTRUCT_SCHEMA_NAME,
                                model=model,
                                reasoning_effort=effort,
                                max_output_tokens=8000,
                                timeout=timeout,
                            ),
                        ),
                        timeout=wait_timeout,
                    )
                    return result_from_instruction_parsed(structured.parsed)
                raw_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            llm.complete,
                            prompt,
                            model=model,
                            reasoning_effort=effort,
                            max_output_tokens=8000,
                            timeout=timeout,
                        ),
                    ),
                    timeout=wait_timeout,
                )
                return parse_editor_instruct_response(raw_response)

            try:
                prompt = build_editor_instruct_prompt(
                    instruct_request,
                    reference_block=reference_block,
                )
                result = await complete_instruction_prompt(prompt)
                if warnings:
                    result = replace(result, warnings=[*warnings, *result.warnings])
                result = validate_editor_instruct_result(instruct_request, result)
            except asyncio.TimeoutError:
                return _error_response(504, "Editor-Anweisung Timeout", "timeout_error")
            except (EditorInstructError, AgentStructuredOutputError) as exc:
                log.warning("Editor-Anweisung konnte nicht geparst werden: %s", exc)
                return _error_response(502, str(exc), "server_error")
            except Exception as exc:
                log.error("Editor-Anweisung Fehler: %s", exc)
                return _error_response(
                    502,
                    f"Editor-Anweisung Fehler: {sanitize_error(exc)}",
                    "server_error",
                )

        return result.to_payload()

    # -- /v1/runs ------------------------------------------------------

    @_router.post("/v1/runs", dependencies=auth_deps)
    async def create_run(req: Request):
        """Create a queued native research run for browser UI clients."""
        try:
            body = await req.json()
        except Exception:
            return _error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            workspace_id = _workspace_id_from_request(req, body)
            question, messages = _question_and_messages(body)
            resolved = _resolve_agent_context(body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except _StackResolutionError as exc:
            content = {"error": {
                "message": exc.message,
                "type": "invalid_request_error",
            }}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        if len(question) > resolved.agent_settings.max_question_length:
            return _error_response(
                400,
                (
                    f"Frage zu lang ({len(question)} Zeichen, "
                    f"max. {resolved.agent_settings.max_question_length})"
                ),
                "invalid_request_error",
            )

        history = _format_history(
            messages, max_messages=settings.server.max_messages_history
        )

        def _work(handle: RunHandle) -> None:
            t0 = time.monotonic()

            def _event_sink(event_type: str, payload: dict[str, Any]) -> None:
                handle.emit(event_type, payload)

            result = agent_run(
                question,
                history=history,
                providers=resolved.providers,
                strategies=resolved.strategies,
                settings=resolved.agent_settings,
                cancel_event=handle.cancel_event,
                run_id=handle.run_id,
                run_event_sink=_event_sink,
            )
            result_state = result.get("result_state", {}) or {}
            if handle.cancel_event.is_set() or result_state.get("cancelled"):
                handle.cancel("client_requested_cancel")
                return

            research_result = ResearchResult.from_raw(result)
            research_result.metrics.elapsed_seconds = round(time.monotonic() - t0, 2)
            answer = research_result.answer
            payload = research_result.to_export_payload()
            usage = result.get("usage", {})
            payload["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": (
                    usage.get("prompt_tokens", 0)
                    + usage.get("completion_tokens", 0)
                ),
            }
            handle.emit_answer(answer)
            current_node = "direct_llm" if resolved.mode == "direct_llm" else "answer"
            handle.complete(
                payload,
                snapshot=build_run_snapshot(
                    result_state,
                    current_node=current_node,
                    last_message="completed",
                ),
            )

        try:
            summary = active_run_store.submit(
                question=question,
                stack_name=resolved.stack_name or "default",
                work=_work,
                agent_overrides=resolved.agent_overrides,
                mode=resolved.mode,
                workspace_id=workspace_id,
            )
        except RunQueueFull:
            return _error_response(
                429,
                "Zu viele wartende Recherche-Auftraege. Bitte warten.",
                "rate_limit_error",
            )
        return JSONResponse(status_code=202, content=summary)

    @_router.get("/v1/runs", dependencies=auth_deps)
    def list_runs(req: Request):
        """List all queued, running, and short-lived terminal native runs."""
        try:
            workspace_id = _workspace_id_from_request(req)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return {
            "object": "list",
            "data": active_run_store.list(workspace_id=workspace_id),
        }

    @_router.get("/v1/runs/{run_id}", dependencies=auth_deps)
    def get_run(run_id: str, req: Request):
        """Return the current public summary for one native run."""
        try:
            workspace_id = _workspace_id_from_request(req)
            return active_run_store.get(run_id, workspace_id=workspace_id)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return _error_response(404, "Run nicht gefunden", "not_found")

    @_router.get("/v1/runs/{run_id}/result", dependencies=auth_deps)
    def get_run_result(run_id: str, req: Request):
        """Return the final report payload for a completed native run."""
        try:
            workspace_id = _workspace_id_from_request(req)
            summary = active_run_store.get(run_id, workspace_id=workspace_id)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return _error_response(404, "Run nicht gefunden", "not_found")
        if summary["status"] != "completed":
            return _error_response(
                409,
                "Run ist noch nicht abgeschlossen",
                "run_not_completed",
                status=summary["status"],
            )
        try:
            return active_run_store.result(run_id, workspace_id=workspace_id)
        except RunNotFound:
            return _error_response(404, "Run-Ergebnis nicht gefunden", "not_found")

    @_router.post("/v1/runs/{run_id}/cancel", dependencies=auth_deps)
    def cancel_run(run_id: str, req: Request):
        """Request cancellation for a queued or running native run."""
        try:
            workspace_id = _workspace_id_from_request(req)
            return active_run_store.cancel(run_id, workspace_id=workspace_id)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return _error_response(404, "Run nicht gefunden", "not_found")

    @_router.get("/v1/runs/{run_id}/events", dependencies=auth_deps)
    async def run_events(run_id: str, req: Request):
        """Stream buffered and live native run events as SSE."""
        try:
            workspace_id = _workspace_id_from_request(req)
            subscription = active_run_store.subscribe(run_id, workspace_id=workspace_id)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return _error_response(404, "Run nicht gefunden", "not_found")

        terminal_events = {
            "inqtrix.run.completed",
            "inqtrix.run.failed",
            "inqtrix.run.cancelled",
        }

        async def _event_generator():
            try:
                terminal_replayed = False
                for event in subscription.replay:
                    yield format_sse_event(event)
                    terminal_replayed = event.get("type") in terminal_events
                if terminal_replayed:
                    return
                while True:
                    if await req.is_disconnected():
                        return
                    try:
                        event = await asyncio.to_thread(
                            subscription.queue.get,
                            True,
                            0.5,
                        )
                    except Empty:
                        continue
                    yield format_sse_event(event)
                    if event.get("type") in terminal_events:
                        return
            finally:
                subscription.close()

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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

        try:
            workspace_id = _workspace_id_from_request(req, body)
            question, messages = _question_and_messages(body)
            resolved = _resolve_agent_context(body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except _StackResolutionError as exc:
            content = {"error": {
                "message": exc.message,
                "type": "invalid_request_error",
            }}
            if exc.available:
                content["error"]["available_stacks"] = exc.available
            return JSONResponse(status_code=400, content=content)

        if (
            not resolved.agent_settings.skip_search
            and len(question) > resolved.agent_settings.max_question_length
        ):
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": (
                        f"Frage zu lang ({len(question)} Zeichen, "
                        f"max. {resolved.agent_settings.max_question_length})"
                    ),
                    "type": "invalid_request_error",
                }},
            )
        chat_agent_settings = _chat_settings_for_question(
            resolved.agent_settings,
            question,
        )

        history = _format_history(
            messages, max_messages=settings.server.max_messages_history
        )

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
                    sem,
                    providers=resolved.providers,
                    strategies=resolved.strategies,
                    settings=chat_agent_settings,
                    messages=messages,
                    include_progress=include_progress,
                    request=req,
                    cancel_event=cancel_event,
                    stack_name=resolved.stack_name,
                    workspace_id=workspace_id or "",
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
                            providers=resolved.providers,
                            strategies=resolved.strategies,
                            settings=chat_agent_settings,
                        ),
                    ),
                    timeout=_request_timeout_seconds(chat_agent_settings),
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

            usage = result.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            result_state = result.get("result_state", {}) or {}
            model_resolution = (
                result_state
                .get("node_model_resolutions", {})
                .get("direct_chat")
            )
            payload = {
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
            if isinstance(model_resolution, dict):
                payload["inqtrix"] = {"model_resolution": model_resolution}
            return payload
