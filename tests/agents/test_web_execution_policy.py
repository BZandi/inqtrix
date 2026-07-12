"""One policy derives web-research permission and profile in every agent."""

import pytest

from inqtrix.agents.web_execution_policy import derive_web_research_policy


def test_normal_requires_explicit_consent_and_always_uses_compact() -> None:
    automatic = derive_web_research_policy(depth="normal")
    directed = derive_web_research_policy(
        depth="normal", admitted_directive=True
    )
    edited = derive_web_research_policy(depth="normal", edited_plan=True)
    assert (automatic.allowed, automatic.profile) == (False, None)
    assert (directed.allowed, directed.profile) == (True, "compact")
    assert edited == directed


def test_deep_always_uses_deep_and_invalid_depth_fails_loudly() -> None:
    deep = derive_web_research_policy(depth="deep")
    assert (deep.allowed, deep.profile) == (True, "deep")
    with pytest.raises(ValueError):
        derive_web_research_policy(depth="invalid")  # type: ignore[arg-type]
