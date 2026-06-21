"""Contract: ``/v1/capabilities`` publishes the effective HTTP wait deadlines.

The browser derives its own abort timeouts from this block instead of
hardcoding them, so the published seconds MUST equal what the editor/text/chat
routes actually enforce. These tests pin that equality (the invariant the
silent-client-cap bug violated) and that the block carries only non-secret
integers.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from inqtrix.providers.base import ProviderContext
from inqtrix.search_result import GroundedSearchResult
from inqtrix.server.app import create_app
from inqtrix.services.request_parsing import (
    editor_wait_seconds,
    request_timeout_seconds,
    text_wait_seconds,
)
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings


class _StubLLM:
    def is_available(self) -> bool:
        return True


class _StubSearch:
    def search(self, *args: object, **kwargs: object) -> GroundedSearchResult:
        return GroundedSearchResult()

    def is_available(self) -> bool:
        return True


def _timeouts(agent: AgentSettings) -> dict[str, int]:
    app = create_app(
        settings=Settings(
            models=ModelSettings(), agent=agent, server=ServerSettings()
        ),
        providers=ProviderContext(llm=_StubLLM(), search=_StubSearch()),
    )
    return TestClient(app).get("/v1/capabilities").json()["timeouts"]


def test_capabilities_timeouts_match_enforced_waits() -> None:
    agent = AgentSettings(
        max_total_seconds=300,
        editor_assistant_timeout=200,
        claim_extract_timeout=60,
    )
    timeouts = _timeouts(agent)

    assert timeouts == {
        "editor_wait_seconds": editor_wait_seconds(agent),
        "chat_wait_seconds": request_timeout_seconds(agent),
        "text_wait_seconds": text_wait_seconds(agent),
    }
    # Distinct base budgets surface distinctly, not a single shared number.
    assert timeouts["editor_wait_seconds"] != timeouts["text_wait_seconds"]
    # Non-secret integers only.
    assert all(isinstance(value, int) for value in timeouts.values())


def test_capabilities_editor_wait_tracks_a_raised_editor_budget() -> None:
    # Raising EDITOR_ASSISTANT_TIMEOUT widens the published editor wait, so a
    # client that derives its abort from this block is not silently capped.
    low = _timeouts(AgentSettings(editor_assistant_timeout=120))[
        "editor_wait_seconds"
    ]
    high = _timeouts(
        AgentSettings(editor_assistant_timeout=600, max_total_seconds=1800)
    )["editor_wait_seconds"]
    assert high > low
