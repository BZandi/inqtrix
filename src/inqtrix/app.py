"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from inqtrix.config import InqtrixConfig, load_config
from inqtrix.config_bridge import config_to_settings, create_providers_from_config
from inqtrix.providers import create_providers, ProviderContext
from inqtrix.routes import create_router, register_routes
from inqtrix.session import SessionStore
from inqtrix.settings import Settings
from inqtrix.strategies import create_default_strategies

log = logging.getLogger("inqtrix")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Resolution order:

    1. If *settings* is provided, use it directly (env-var mode).
       No YAML file is loaded; providers come from env-var settings.
    2. Otherwise, attempt to load ``inqtrix.yaml``.  If a YAML config
       with providers is found, derive settings and providers from it.
    3. If no YAML file is found, fall back to pure env-var configuration
       (full backwards compatibility).

    Parameters
    ----------
    settings:
        Optional pre-built :class:`Settings` instance.  When *None*,
        YAML config is checked first, then env-vars.

    Returns
    -------
    FastAPI
        Fully configured application with all routes registered.
    """
    # Determine mode: explicit settings → env-var; no settings → try YAML
    use_yaml = False
    config = InqtrixConfig()  # empty default

    if settings is not None:
        # Explicit settings: always use env-var mode
        pass
    else:
        config = load_config()
        if config.providers:
            # YAML mode
            settings = config_to_settings(config)
            use_yaml = True
            log.info(
                "Loaded YAML config with %d provider(s), %d model(s)",
                len(config.providers),
                len(config.models),
            )
        else:
            # Env-var mode
            settings = Settings()

    # Configure logging based on testing mode
    log_level = logging.DEBUG if settings.agent.testing_mode else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("inqtrix").setLevel(log_level)

    # Wire up providers
    providers: ProviderContext
    if use_yaml:
        providers = create_providers_from_config(config, settings)
    else:
        providers = create_providers(settings)

    # Wire up strategies via the public provider interface so custom
    # providers remain compatible with the default Baukasten pipeline.
    strategies = create_default_strategies(
        settings.agent,
        llm=providers.llm,
        summarize_model=settings.models.effective_summarize_model,
        summarize_timeout=settings.agent.summarize_timeout,
    )

    # Session store
    session_store = SessionStore(
        ttl_seconds=settings.server.session_ttl_seconds,
        max_count=settings.server.session_max_count,
        max_context_blocks=settings.server.session_max_context_blocks,
        max_claim_ledger=settings.server.session_max_claim_ledger,
    )

    # Lazy semaphore (event loop may not exist yet)
    _semaphore: asyncio.Semaphore | None = None

    def semaphore_factory() -> asyncio.Semaphore:
        nonlocal _semaphore
        if _semaphore is None:
            _semaphore = asyncio.Semaphore(settings.server.max_concurrent)
        return _semaphore

    # Fresh router per create_app() call to avoid duplicate route handlers
    app_router = create_router()

    register_routes(
        app_router,
        providers=providers,
        strategies=strategies,
        settings=settings,
        session_store=session_store,
        semaphore_factory=semaphore_factory,
    )

    app = FastAPI(
        title="Inqtrix Research Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(app_router)

    return app
