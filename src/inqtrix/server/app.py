"""FastAPI application factory with optional Baukasten injection and lifespan."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inqtrix.providers import create_providers
from inqtrix.providers.base import ProviderContext
from inqtrix.server.routes import create_router, register_routes
from inqtrix.server.security import (
    make_api_key_dependency,
    make_cors_middleware_kwargs,
)
from inqtrix.server.runs import RunStore
from inqtrix.settings import Settings
from inqtrix.strategies import StrategyContext, create_default_strategies, resolve_claim_extract_model

log = logging.getLogger("inqtrix")


def _provider_label(provider: object) -> str:
    """Return the public class name of a provider, unwrapping adapter shells."""
    wrapped = getattr(provider, "_provider", None)
    if wrapped is not None:
        return type(wrapped).__name__
    return type(provider).__name__


def _provider_ready(provider: object) -> bool:
    """Probe the provider's ``is_available`` hook without raising."""
    try:
        checker = getattr(provider, "is_available", None)
        return bool(checker()) if callable(checker) else False
    except Exception as exc:  # noqa: BLE001 — keep startup probe non-fatal
        log.warning(
            "Startup-Health-Probe fuer %s fehlgeschlagen: %s",
            _provider_label(provider),
            exc,
        )
        return False


def create_app(
    *,
    settings: Settings | None = None,
    providers: ProviderContext | None = None,
    strategies: StrategyContext | None = None,
) -> FastAPI:
    """Build the Inqtrix FastAPI app with optional Baukasten injection.

    Resolution order (precedence high → low):

    1. ``providers`` injected (Baukasten mode) — when supplied,
       ``create_providers(...)`` is not called for the LLM/search slot,
       and the caller is fully responsible for provider construction.
       This is the path used by the ``examples/webserver_stacks/*.py``
       scripts.
    2. Explicit ``settings`` without ``providers`` — env-var mode against
       the supplied settings.
    3. Pure env-var configuration via ``Settings()`` as the final fallback.

    ``strategies`` is optional in every mode; when ``None``,
    :func:`inqtrix.strategies.create_default_strategies` is invoked with
    the resolved LLM provider so existing default heuristics stay intact.

    Args:
        settings: Pre-built :class:`Settings`. When ``None``, env
            resolution via :class:`Settings` applies.
        providers: Pre-built :class:`ProviderContext`. When supplied,
            the injected providers are used verbatim (Baukasten injection).
        strategies: Pre-built :class:`StrategyContext`. When ``None``,
            defaults are derived from the resolved LLM provider.

    Returns:
        A fully-wired :class:`FastAPI` instance with all routes
        registered, an ASGI ``lifespan`` context attached for startup
        health-probe and shutdown logging, and the OpenAPI schema
        intentionally disabled (``docs_url=None``).

    Example:
        Library / env-only mode (current default ``python -m inqtrix``)::

            from inqtrix.server import create_app
            app = create_app()

        Baukasten injection from a webserver-stack example::

            from inqtrix import LiteLLM, PerplexitySearch
            from inqtrix.providers.base import ProviderContext
            from inqtrix.server import create_app
            from inqtrix.settings import Settings

            providers = ProviderContext(
                llm=LiteLLM(api_key="...", default_model="gpt-4o"),
                search=PerplexitySearch(api_key="..."),
            )
            app = create_app(settings=Settings(), providers=providers)
    """
    if providers is not None:
        if settings is None:
            settings = Settings()
    elif settings is not None:
        pass
    else:
        settings = Settings()

    # Configure logging as a last-resort default. The webserver-stack
    # examples (and any caller that already ran ``configure_logging``
    # before invoking ``create_app``) keep their handlers — passing
    # ``force=False`` makes this call a no-op when a real handler is
    # already attached (see ``logging_config.is_configured``). Without
    # this guard the previous unconditional reset silently dropped every
    # INFO-level marker (``_classify_fallback``, ``Round 1``, ...) that
    # Designprinzip 1 relies on for "No Silent Fallbacks" visibility.
    from inqtrix.logging_config import configure_logging
    configure_logging(
        enabled=settings.agent.testing_mode,
        level="DEBUG" if settings.agent.testing_mode else "WARNING",
        console=True,
        force=False,
    )

    # Resolve providers — injected wins over env-driven defaults.
    if providers is None:
        providers = create_providers(settings)

    # Resolve strategies — injected wins, otherwise defaults from LLM.
    # The claim_extract_model is resolved Constructor-First from the provider's
    # own models attribute (Designprinzip 6); the global settings.models
    # serves only as a last-resort fallback for providers without one.
    if strategies is None:
        strategies = create_default_strategies(
            settings.agent,
            llm=providers.llm,
            claim_extract_model=resolve_claim_extract_model(
                providers.llm,
                fallback=settings.models.effective_claim_extract_model,
            ),
            claim_extract_timeout=settings.agent.claim_extract_timeout,
        )

    # Lazy semaphore (event loop may not exist yet)
    _semaphore: asyncio.Semaphore | None = None

    def semaphore_factory() -> asyncio.Semaphore:
        nonlocal _semaphore
        if _semaphore is None:
            _semaphore = asyncio.Semaphore(settings.server.max_concurrent)
        return _semaphore

    run_store = RunStore.from_settings(settings.server)

    # Resolve opt-in security layers (all default to None / disabled).
    api_key_dependency = make_api_key_dependency(settings.server)
    cors_kwargs = make_cors_middleware_kwargs(settings.server)
    api_key_active = api_key_dependency is not None
    cors_active = cors_kwargs is not None

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Startup — log resolved provider identity, reachability and the
        # behavioural envelope so operators can confirm the deployment
        # took effect without grepping a separate log file.
        # Also emit a terminal-visible banner about the active logging
        # configuration so operators see at a glance whether file
        # logging is enabled and where the log file lives. Printed
        # directly to stderr (not through the inqtrix logger) so the
        # banner stays visible even when file logging is disabled or
        # the logger is silent.
        from inqtrix.logging_config import print_logging_banner
        logging_state = print_logging_banner()
        log.info(
            "Logging status | file_enabled=%s | log_file=%s | level=%s "
            "| console=%s | web_mirrored=%s",
            logging_state["file_enabled"],
            logging_state["file_path"] or "-",
            logging_state["level"],
            logging_state["console_enabled"],
            logging_state["web_mirrored"],
        )

        llm_label = _provider_label(providers.llm)
        search_label = _provider_label(providers.search)
        llm_ready = _provider_ready(providers.llm)
        search_ready = _provider_ready(providers.search)
        log.info(
            "Inqtrix server starting | llm=%s ready=%s | search=%s ready=%s "
            "| report_profile=%s | max_concurrent=%d | run_max_concurrent=%d "
            "| run_queue_max_size=%d | run_completed_ttl_seconds=%d "
            "| api_key_gate=%s | cors=%s",
            llm_label,
            llm_ready,
            search_label,
            search_ready,
            settings.agent.report_profile,
            settings.server.max_concurrent,
            settings.server.run_max_concurrent or settings.server.max_concurrent,
            settings.server.run_queue_max_size,
            settings.server.run_completed_ttl_seconds,
            "on" if api_key_active else "off",
            "on" if cors_active else "off",
        )
        try:
            yield
        finally:
            log.info(
                "Inqtrix server stopping | llm=%s | search=%s",
                llm_label,
                search_label,
            )

    # Fresh router per create_app() call to avoid duplicate route handlers
    app_router = create_router()

    register_routes(
        app_router,
        providers=providers,
        strategies=strategies,
        settings=settings,
        semaphore_factory=semaphore_factory,
        api_key_dependency=api_key_dependency,
        run_store=run_store,
    )

    app = FastAPI(
        title="Inqtrix Research Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    if cors_kwargs is not None:
        app.add_middleware(CORSMiddleware, **cors_kwargs)
    app.include_router(app_router)

    return app
