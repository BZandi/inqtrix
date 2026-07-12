"""Enterprise-Austausch seams on the composition root (Phase 5).

`create_app` exposes additive injection points so an integrator can swap a
component WITHOUT editing the env-driven dispatch in `build_auth_provider` /
`build_object_store`:

* ``auth_provider=`` — a custom :class:`AuthProvider` wins over
  ``INQTRIX_AUTH_MODE`` resolution;
* ``object_store_impl=`` — a custom blob backend wins over the
  ``INQTRIX_OBJECT_STORE_BACKEND`` enum dispatch.

(The other stores already ride existing build_container params: a custom
run/queue store via ``run_store=``, a custom vector store via the knowledge
context, custom permissions via ``permissions=``.)
"""

from __future__ import annotations

import asyncio

from sqlalchemy.pool import NullPool

from inqtrix.auth.api_key import build_auth_provider
from inqtrix.providers.base import ProviderContext
from inqtrix.server import create_app
from inqtrix.server.app import _placeholder_secret_fields
from inqtrix.server.container import (
    build_container,
    build_platform_persistence_bundle,
)
from inqtrix.settings import (
    AuthSettings,
    ServerSettings,
    Settings,
    StorageSettings,
)

from tests.contract._app import StubSearch
from tests.test_knowledge_routes import KnowledgeStubLLM


def _memory_settings() -> Settings:
    return Settings(
        server=ServerSettings(public_base_url=""),
        storage=StorageSettings(backend="memory", database_url=""),
        auth=AuthSettings(mode="none"),
    )


def _providers() -> ProviderContext:
    return ProviderContext(llm=KnowledgeStubLLM(), search=StubSearch())


def test_create_app_uses_injected_auth_provider():
    # A specific provider instance passed in must be the one the container
    # resolves — env mode resolution is bypassed entirely.
    injected = build_auth_provider(Settings(auth=AuthSettings(mode="none")))
    app = create_app(
        settings=_memory_settings(),
        providers=_providers(),
        auth_provider=injected,
    )
    assert app.state.container.auth_provider is injected


def test_create_app_object_store_impl_is_wired_into_file_service():
    # The FileService is built in every mode that has a file registry (the
    # memory/local default included — the storage backend only changes the
    # metadata registry), so an injected object store is used verbatim there,
    # winning over the INQTRIX_OBJECT_STORE_BACKEND enum dispatch.
    sentinel = object()
    app = create_app(
        settings=_memory_settings(),
        providers=_providers(),
        object_store_impl=sentinel,  # type: ignore[arg-type]
    )
    file_service = app.state.container.file_service
    assert file_service is not None
    assert file_service._object_store is sentinel


def _pg_settings() -> Settings:
    # A dummy asyncpg URL: create_async_engine is lazy, so no connection is
    # ever opened — only the pool CLASS is exercised.
    return Settings(
        server=ServerSettings(public_base_url=""),
        storage=StorageSettings(
            backend="postgres",
            database_url="postgresql+asyncpg://u:p@localhost:5432/db",
        ),
        auth=AuthSettings(mode="none"),
    )


def _bundle_engine(bundle):
    return bundle.session_factory.kw["bind"]


def _permission_engine(container):
    # The permission service's identity backend is the store that crashed on
    # agent resume; assert the pool class on ITS engine, not just the bundle.
    return container.permission_service._members._session_factory.kw["bind"]


def test_platform_persistence_bundle_null_pool_is_loop_agnostic():
    # The workspace agent drives these repositories from a sync worker thread
    # via per-call asyncio.run; a pooled asyncpg connection reused across those
    # short-lived loops crashes ("Future attached to a different loop"). The
    # worker asks for a NullPool engine so the shared platform engine is
    # loop-agnostic.
    bundle = build_platform_persistence_bundle(_pg_settings(), null_pool=True)
    assert isinstance(_bundle_engine(bundle).pool, NullPool)


def test_platform_persistence_bundle_defaults_to_pooled_engine():
    # The API keeps the pooled engine (one persistent request loop): the flag
    # is OFF by default so existing deployments stay byte-identical.
    bundle = build_platform_persistence_bundle(_pg_settings())
    assert not isinstance(_bundle_engine(bundle).pool, NullPool)


def test_build_container_forwards_platform_persistence_null_pool():
    # The worker passes platform_persistence_null_pool=True; it must reach the
    # permission service's identity engine end-to-end, not just the bundle.
    container = build_container(
        providers=_providers(),
        strategies=None,
        settings=_pg_settings(),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        platform_persistence_null_pool=True,
    )
    assert isinstance(_permission_engine(container).pool, NullPool)


def test_build_container_defaults_platform_persistence_to_pooled():
    # Without the flag (the API path) the permission engine stays pooled — the
    # distinct-value assertion that keeps the flag load-bearing.
    container = build_container(
        providers=_providers(),
        strategies=None,
        settings=_pg_settings(),
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    assert not isinstance(_permission_engine(container).pool, NullPool)


def test_placeholder_secret_scan_flags_change_me_values():
    # A secret left at its deploy/.env.stack.example CHANGE_ME placeholder is
    # reported by name so create_app can warn loudly at startup (the guarantee
    # documented in how-to/deploy-to-production.md). The value is never echoed.
    settings = Settings(
        auth=AuthSettings(
            session_secret="CHANGE_ME_SESSION_SECRET",
            pat_pepper="real-pepper-value-not-a-placeholder",
        ),
        server=ServerSettings(public_base_url=""),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    flagged = _placeholder_secret_fields(settings)
    assert "INQTRIX_SESSION_SECRET" in flagged
    assert "INQTRIX_PAT_PEPPER" not in flagged
