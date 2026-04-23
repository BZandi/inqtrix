"""Multi-stack hosting for the Inqtrix HTTP server (ADR-MS-1).

Lets a single FastAPI process host multiple ``(providers, strategies,
agent_settings)`` triples side by side. UIs pick one per request via a
new ``body["stack"]`` top-level field, and a ``GET /v1/stacks``
discovery endpoint exposes the available bundles plus a cached health
flag so a Streamlit/React app can render a selection box without
DDoSing the upstream providers.

Single-stack ``create_app(...)`` is unaffected — multi-stack lives in
this module behind its own factory ``create_multi_stack_app``. The
two factories share the routes, lifespan, security helpers and
session store.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inqtrix.providers.base import ProviderContext
from inqtrix.server.routes import create_router, register_routes
from inqtrix.server.security import (
    make_api_key_dependency,
    make_cors_middleware_kwargs,
)
from inqtrix.server.session import SessionStore
from inqtrix.settings import AgentSettings, Settings
from inqtrix.strategies import StrategyContext, create_default_strategies, resolve_summarize_model

log = logging.getLogger("inqtrix")


_STACK_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
_DISCOVERY_CACHE_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class StackBundle:
    """A named bundle of providers / strategies / settings for multi-stack hosting.

    Attributes:
        providers: The :class:`ProviderContext` for this stack — the
            LLM and search providers that requests routed to this
            stack will use.
        strategies: Optional :class:`StrategyContext`. ``None`` (the
            default) makes the multi-stack factory derive defaults
            from ``providers.llm`` via
            :func:`inqtrix.strategies.create_default_strategies`.
        agent_settings: Optional per-stack :class:`AgentSettings` that
            overrides the global ``settings.agent`` for requests
            routed to this stack only. ``None`` falls back to the
            shared global.
        description: Free-text label shown in the ``/v1/stacks``
            discovery payload. Should be operator-friendly
            (e.g. ``"Bedrock Opus 4.6 + Perplexity Sonar Pro"``).
    """

    providers: ProviderContext
    strategies: StrategyContext | None = None
    agent_settings: AgentSettings | None = None
    description: str = ""


def _validate_stacks(
    stacks: dict[str, StackBundle], default_stack: str
) -> None:
    """Validate the stack registry. Raises ValueError on any inconsistency."""
    if not stacks:
        raise ValueError("create_multi_stack_app requires at least one stack")
    for name in stacks:
        if not _STACK_NAME_PATTERN.match(name):
            raise ValueError(
                f"Stack name {name!r} must match ^[a-z0-9_]+$ "
                "(lowercase letters, digits, underscore)."
            )
    if default_stack not in stacks:
        raise ValueError(
            f"default_stack {default_stack!r} is not in stacks "
            f"({sorted(stacks.keys())})"
        )


def _provider_label(provider: object) -> str:
    """Mirror the helper in app.py — extract the public class name."""
    wrapped = getattr(provider, "_provider", None)
    if wrapped is not None:
        return type(wrapped).__name__
    return type(provider).__name__


def _provider_ready(provider: object) -> bool:
    """Probe the provider's is_available without raising."""
    try:
        checker = getattr(provider, "is_available", None)
        return bool(checker()) if callable(checker) else False
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Stack-Discovery-Health-Probe fuer %s fehlgeschlagen: %s",
            _provider_label(provider),
            exc,
        )
        return False


class _DiscoveryCache:
    """Time-bound cache for the ``/v1/stacks`` payload.

    ``is_available()`` may touch the network; callers (Streamlit
    polling every second) must not be allowed to fan out into a
    provider-call storm. The cache holds the rendered payload for
    ``_DISCOVERY_CACHE_TTL_SECONDS`` and re-renders thereafter.
    """

    def __init__(self) -> None:
        self._payload: dict[str, Any] | None = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    def get(
        self,
        *,
        stacks: dict[str, StackBundle],
        default_stack: str,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if (
                self._payload is not None
                and (now - self._fetched_at) < _DISCOVERY_CACHE_TTL_SECONDS
            ):
                return self._payload
            self._payload = self._render(stacks, default_stack)
            self._fetched_at = now
            return self._payload

    @staticmethod
    def _render(
        stacks: dict[str, StackBundle], default_stack: str
    ) -> dict[str, Any]:
        rendered = []
        for name, bundle in stacks.items():
            rendered.append(
                {
                    "name": name,
                    "llm": _provider_label(bundle.providers.llm),
                    "search": _provider_label(bundle.providers.search),
                    "ready": _provider_ready(bundle.providers.llm)
                    and _provider_ready(bundle.providers.search),
                    "description": bundle.description,
                    "models": _stack_models_payload(bundle),
                }
            )
        return {"default": default_stack, "stacks": rendered}


def _stack_models_payload(bundle: StackBundle) -> dict[str, str]:
    """Per-stack model identifiers for ``GET /v1/stacks`` discovery output.

    Constructor-First (Designprinzip 6): each entry reports what the
    bundle's LLM and search providers were *built* with, not what the
    process-global ``Settings`` defaults claim. UIs use this payload to
    render an honest stack-selection box (model chips next to the stack
    name) without making an extra call against ``/health``.

    Empty strings are returned when a provider exposes no public model
    attribute; consumers should treat that as "unknown / provider-default"
    and not as an error.
    """
    llm_models = getattr(bundle.providers.llm, "models", None)

    def _from_llm(attr: str) -> str:
        if llm_models is None:
            return ""
        value = getattr(llm_models, attr, "")
        return value if isinstance(value, str) else ""

    # ADR-WS-12: read the standardized SearchProvider.search_model
    # property; the ABC default returns "<ClassName>(unknown)" for
    # subclasses that forget to override, which is intentionally loud.
    search_provider = bundle.providers.search
    search_model_value = getattr(search_provider, "search_model", "")
    search_model = (
        search_model_value if isinstance(search_model_value, str) and search_model_value
        else ""
    )

    return {
        "reasoning_model": _from_llm("reasoning_model"),
        "summarize_model": _from_llm("effective_summarize_model"),
        "classify_model": _from_llm("effective_classify_model"),
        "evaluate_model": _from_llm("effective_evaluate_model"),
        "search_model": search_model,
    }


def create_multi_stack_app(
    *,
    settings: Settings,
    stacks: dict[str, StackBundle],
    default_stack: str,
) -> FastAPI:
    """Build a FastAPI app that hosts multiple Baukasten stacks (ADR-MS-1).

    Validates that ``stacks`` is non-empty, ``default_stack`` is a key
    in it, and every stack name matches ``^[a-z0-9_]+$``.

    The resulting app exposes the unauthenticated discovery endpoint
    ``GET /v1/stacks`` on top of the standard surface (``/health``,
    ``/v1/models``, ``/v1/chat/completions``, ``/v1/test/run``). Each
    chat-completions request resolves the stack via
    ``body["stack"]``; missing → ``default_stack``; unknown →
    ``400 invalid_request_error`` with ``available_stacks`` hint.

    The opt-in security layers (TLS, Bearer-API-key, CORS) come from
    ``settings.server`` exactly as in single-stack ``create_app``;
    discovery stays unauthenticated by design (Streamlit/React UIs
    need to read the stack list before they have the API key form
    rendered).

    Args:
        settings: Resolved Inqtrix :class:`Settings`. Provides the
            global session-store sizing, server-side concurrency
            limits and security configuration.
        stacks: Mapping ``{stack_name: StackBundle}``. Keys must
            match ``^[a-z0-9_]+$``.
        default_stack: The key used when a request omits the
            ``"stack"`` body field. Must exist in ``stacks``.

    Returns:
        A fully wired FastAPI instance with multi-stack routing in
        place. Lifespan logs the discovery defaults and per-stack
        readiness on startup.
    """
    _validate_stacks(stacks, default_stack)

    # Ensure every stack has a usable strategies bundle. Defaults are
    # derived from the LLM provider via the same factory the single-
    # stack path uses.
    resolved_stacks: dict[str, StackBundle] = {}
    for name, bundle in stacks.items():
        if bundle.strategies is None:
            agent_for_defaults = bundle.agent_settings or settings.agent
            strategies = create_default_strategies(
                agent_for_defaults,
                llm=bundle.providers.llm,
                summarize_model=resolve_summarize_model(
                    bundle.providers.llm,
                    fallback=settings.models.effective_summarize_model,
                ),
                summarize_timeout=agent_for_defaults.summarize_timeout,
            )
            resolved_stacks[name] = StackBundle(
                providers=bundle.providers,
                strategies=strategies,
                agent_settings=bundle.agent_settings,
                description=bundle.description,
            )
        else:
            resolved_stacks[name] = bundle

    # Configure logging as a last-resort default — see create_app for
    # the rationale behind ``force=False``. A multi-stack example that
    # set up its own ``configure_logging(...)`` keeps its handlers; the
    # silent default is only installed when nothing was configured yet.
    from inqtrix.logging_config import configure_logging
    configure_logging(
        enabled=settings.agent.testing_mode,
        level="DEBUG" if settings.agent.testing_mode else "WARNING",
        console=True,
        force=False,
    )

    # Session store with the same DEEP-tuning hardening as single-stack.
    tuning = settings.agent.report_tuning
    session_store = SessionStore(
        ttl_seconds=settings.server.session_ttl_seconds,
        max_count=settings.server.session_max_count,
        max_context_blocks=max(
            settings.server.session_max_context_blocks,
            tuning.session_max_context_blocks,
        ),
        max_claim_ledger=max(
            settings.server.session_max_claim_ledger,
            tuning.session_max_claim_ledger,
        ),
        max_answer_chars=tuning.session_max_answer_chars,
    )

    _semaphore: asyncio.Semaphore | None = None

    def semaphore_factory() -> asyncio.Semaphore:
        nonlocal _semaphore
        if _semaphore is None:
            _semaphore = asyncio.Semaphore(settings.server.max_concurrent)
        return _semaphore

    api_key_dependency = make_api_key_dependency(settings.server)
    cors_kwargs = make_cors_middleware_kwargs(settings.server)
    api_key_active = api_key_dependency is not None
    cors_active = cors_kwargs is not None

    discovery_cache = _DiscoveryCache()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "Inqtrix multi-stack server starting | stacks=%d | default=%s "
            "| max_concurrent=%d | api_key_gate=%s | cors=%s",
            len(resolved_stacks),
            default_stack,
            settings.server.max_concurrent,
            "on" if api_key_active else "off",
            "on" if cors_active else "off",
        )
        for name, bundle in resolved_stacks.items():
            llm_label = _provider_label(bundle.providers.llm)
            search_label = _provider_label(bundle.providers.search)
            log.info(
                "  stack=%s | llm=%s | search=%s | description=%s",
                name,
                llm_label,
                search_label,
                bundle.description or "(none)",
            )
        try:
            yield
        finally:
            log.info(
                "Inqtrix multi-stack server stopping | stacks=%d",
                len(resolved_stacks),
            )

    app_router = create_router()

    # The default-stack providers/strategies/settings drive the legacy
    # single-stack code path inside register_routes; the actual
    # multi-stack resolution happens per-request via stacks_lookup.
    default_bundle = resolved_stacks[default_stack]
    default_agent_settings = (
        default_bundle.agent_settings
        if default_bundle.agent_settings is not None
        else settings.agent
    )

    register_routes(
        app_router,
        providers=default_bundle.providers,
        strategies=default_bundle.strategies or create_default_strategies(
            default_agent_settings,
            llm=default_bundle.providers.llm,
            summarize_model=resolve_summarize_model(
                default_bundle.providers.llm,
                fallback=settings.models.effective_summarize_model,
            ),
            summarize_timeout=default_agent_settings.summarize_timeout,
        ),
        settings=settings,
        session_store=session_store,
        semaphore_factory=semaphore_factory,
        api_key_dependency=api_key_dependency,
        stacks=resolved_stacks,
        default_stack=default_stack,
    )

    # Discovery route — unauthenticated by design (ADR-MS-3).
    @app_router.get("/v1/stacks")
    def list_stacks() -> dict[str, Any]:
        return discovery_cache.get(stacks=resolved_stacks, default_stack=default_stack)

    app = FastAPI(
        title="Inqtrix Research Agent (multi-stack)",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    if cors_kwargs is not None:
        app.add_middleware(CORSMiddleware, **cors_kwargs)
    app.include_router(app_router)

    return app
