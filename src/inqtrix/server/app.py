"""FastAPI application factory with optional Baukasten injection and lifespan."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inqtrix.auth.api_key import build_auth_provider
from inqtrix.providers import create_providers
from inqtrix.providers.base import ProviderContext
from inqtrix.server.routes import create_router, register_routes
from inqtrix.server.security import make_cors_middleware_kwargs
from inqtrix.services.health_service import provider_label, provider_ready
from inqtrix.settings import Settings
from inqtrix.strategies import StrategyContext, create_default_strategies, resolve_claim_extract_model

if TYPE_CHECKING:
    from inqtrix.auth.principal import AuthProvider
    from inqtrix.storage.object_store import ObjectStore

log = logging.getLogger("inqtrix")


_PLACEHOLDER_SECRET_MARKER = "CHANGE_ME"


def _placeholder_secret_fields(settings: Settings) -> list[str]:
    """Env names of secret settings still holding a ``CHANGE_ME`` placeholder.

    Returns the variable NAMES only — never the value. Lets ``create_app``
    warn loudly at startup so a placeholder secret (the values shipped in
    ``deploy/.env.stack.example``) cannot be deployed silently
    (Designprinzip 1: No Silent Fallbacks).
    """
    candidates = {
        "INQTRIX_SESSION_SECRET": settings.auth.session_secret,
        "INQTRIX_PAT_PEPPER": settings.auth.pat_pepper,
        "INQTRIX_SERVER_API_KEY": settings.server.api_key,
        "INQTRIX_DATABASE_URL": settings.storage.database_url,
        "INQTRIX_OIDC_CLIENT_SECRET": settings.auth.oidc_client_secret,
        "INQTRIX_LDAP_BIND_PASSWORD": settings.auth.ldap_bind_password,
    }
    return [
        name
        for name, value in candidates.items()
        if isinstance(value, str)
        and _PLACEHOLDER_SECRET_MARKER in value.upper()
    ]


def create_app(
    *,
    settings: Settings | None = None,
    providers: ProviderContext | None = None,
    strategies: StrategyContext | None = None,
    auth_provider: "AuthProvider | None" = None,
    object_store_impl: "ObjectStore | None" = None,
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

    # Run-store selection (memory default, durable opt-in) happens in
    # the container's build_run_store bridge so the Postgres backends
    # share one engine; nothing is built here anymore.

    # Resolve opt-in security layers (all default to disabled). The auth
    # provider honours INQTRIX_AUTH_MODE with explicit-wins semantics and
    # raises at startup on contradictory configuration (Designprinzip 1).
    # An injected auth provider wins over env-driven mode resolution — the
    # Enterprise-Austausch seam for a custom AuthProvider (no need to edit the
    # build_auth_provider dispatch). See how-to/writing-a-custom-auth-provider.
    auth_provider = auth_provider or build_auth_provider(settings)
    # Loudly flag any secret still left at its CHANGE_ME placeholder — a
    # placeholder secret is a silent insecurity, so it must be visible at
    # startup (Designprinzip 1). Names only, never the value.
    placeholder_secrets = _placeholder_secret_fields(settings)
    if placeholder_secrets:
        log.warning(
            "INSECURE: %d secret(s) still hold a CHANGE_ME placeholder value "
            "(%s) — set real secrets before production.",
            len(placeholder_secrets),
            ", ".join(placeholder_secrets),
        )
    cors_kwargs = make_cors_middleware_kwargs(settings.server)
    api_key_active = auth_provider.mode == "apikey"
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

        llm_label = provider_label(providers.llm)
        search_label = provider_label(providers.search)
        llm_ready = provider_ready(providers.llm, label=llm_label)
        search_ready = provider_ready(providers.search, label=search_label)
        log.info(
            "Inqtrix server starting | llm=%s ready=%s | search=%s ready=%s "
            "| report_profile=%s | max_concurrent=%d | run_max_concurrent=%d "
            "| run_queue_max_size=%d | run_completed_ttl_seconds=%d "
            "| api_key_gate=%s | auth_mode=%s | cors=%s",
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
            auth_provider.mode,
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
            # Durable run stores own an engine and a background loop
            # thread; the memory store has no close() and is skipped.
            container = getattr(_app.state, "container", None)
            run_store = getattr(container, "run_store", None)
            if run_store is not None and hasattr(run_store, "close"):
                run_store.close()
            # The Postgres quota store owns a NullPool engine; dispose it
            # on the live loop here (record_blocking's throwaway loops
            # cannot). Memory store / disabled quota -> no-op.
            quota_service = getattr(container, "quota_service", None)
            if quota_service is not None:
                await quota_service.aclose()
            # The Postgres-canonical knowledge store owns its own NullPool
            # engine (loop-agnostic); dispose it on the live loop here.
            # Memory/Qdrant stores have no aclose -> guarded no-op.
            knowledge_service = getattr(container, "knowledge_service", None)
            knowledge = getattr(knowledge_service, "knowledge", None)
            store = getattr(knowledge, "store", None)
            if store is not None and hasattr(store, "aclose"):
                await store.aclose()
            # The Postgres chat-history store owns its own NullPool engine;
            # dispose it on the live loop here. Memory store -> no-op.
            chat_history_service = getattr(
                container, "chat_history_service", None
            )
            chat_store = getattr(chat_history_service, "store", None)
            if chat_store is not None and hasattr(chat_store, "aclose"):
                await chat_store.aclose()
            # The Postgres editor store owns its own NullPool engine too.
            editor_service = getattr(
                container, "editor_persistence_service", None
            )
            editor_store = getattr(editor_service, "store", None)
            if editor_store is not None and hasattr(editor_store, "aclose"):
                await editor_store.aclose()
            # The Postgres asset-record store owns its own NullPool engine too.
            asset_service = getattr(container, "asset_records_service", None)
            asset_store = getattr(asset_service, "store", None)
            if asset_store is not None and hasattr(asset_store, "aclose"):
                await asset_store.aclose()
            # The Postgres vector-index store owns its own NullPool engine too.
            vector_index_service = getattr(container, "vector_index_service", None)
            vector_index_store = getattr(vector_index_service, "store", None)
            if vector_index_store is not None and hasattr(vector_index_store, "aclose"):
                await vector_index_store.aclose()
            # The Postgres account-preferences store owns its own NullPool engine.
            account_prefs_service = getattr(container, "account_preferences_service", None)
            account_prefs_store = getattr(account_prefs_service, "store", None)
            if account_prefs_store is not None and hasattr(account_prefs_store, "aclose"):
                await account_prefs_store.aclose()
            # The Postgres knowledge-session store owns its own NullPool engine.
            knowledge_sessions_service = getattr(container, "knowledge_sessions_service", None)
            knowledge_sessions_store = getattr(knowledge_sessions_service, "store", None)
            if knowledge_sessions_store is not None and hasattr(knowledge_sessions_store, "aclose"):
                await knowledge_sessions_store.aclose()

    # Fresh router per create_app() call to avoid duplicate route handlers
    app_router = create_router()

    container = register_routes(
        app_router,
        providers=providers,
        strategies=strategies,
        settings=settings,
        semaphore_factory=semaphore_factory,
        auth_provider=auth_provider,
        object_store_impl=object_store_impl,
    )

    enable_openapi = settings.server.enable_openapi
    app = FastAPI(
        title="Inqtrix Research Agent",
        docs_url="/docs" if enable_openapi else None,
        redoc_url="/redoc" if enable_openapi else None,
        openapi_url="/openapi.json" if enable_openapi else None,
        lifespan=_lifespan,
    )
    if cors_kwargs is not None:
        app.add_middleware(CORSMiddleware, **cors_kwargs)
    app.include_router(app_router)
    app.state.container = container

    return app
