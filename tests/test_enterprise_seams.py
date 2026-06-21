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

from inqtrix.auth.api_key import build_auth_provider
from inqtrix.providers.base import ProviderContext
from inqtrix.server import create_app
from inqtrix.server.app import _placeholder_secret_fields
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
