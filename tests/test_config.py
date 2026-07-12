"""Tests for environment-backed settings validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inqtrix.settings import AgentSettings, ServerSettings


def test_agent_depth_is_normalized_once_at_settings_boundary() -> None:
    """Engines receive the canonical token instead of repeating fallbacks."""
    assert AgentSettings(DEPTH=" Deep ").depth == "deep"


def test_agent_depth_rejects_unknown_values() -> None:
    """A typo must fail visibly instead of silently becoming normal depth."""
    with pytest.raises(ValidationError):
        AgentSettings(DEPTH="thorough")


class TestServerSettings:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("MAX_CONCURRENT", 0),
            ("RUN_MAX_CONCURRENT", 0),
            ("RUN_QUEUE_MAX_SIZE", -1),
            ("RUN_COMPLETED_TTL_SECONDS", -1),
            ("RUN_EVENT_BUFFER_SIZE", 0),
        ],
    )
    def test_rejects_invalid_run_limits(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            ServerSettings(**{field: value})
