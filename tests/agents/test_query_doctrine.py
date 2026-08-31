"""P6A: ONE query doctrine across every Agent-Desk site.

The P0 A/B eval (3 clean pairs, DB-verified queries) settled it: the
naturally phrased, self-contained evidence question won twice, tied
once, never lost against the keyword chain — so the keyword doctrine
("SUCHQUERY, keine Gespraechsfrage") is replaced everywhere the agent
platform instructs query shape. These pins are SITE-precise: each site
must carry the doctrine wording itself, so a later edit cannot silently
revive keyword instructions at one surface while the others moved on.
The research-desk pipeline (graph.py, src/inqtrix/prompts.py) is
deliberately NOT covered — P6 does not touch it.
"""

from __future__ import annotations

import inspect

from inqtrix.agents.kernel.algorithm import KernelAgentAlgorithm
from inqtrix.agents.kernel.tools import build_kernel_tools
from inqtrix.agents.phase_models import DiscoveryGap, QuickWebQuery
from inqtrix.agents.plan_models import PlanTaskModel
from inqtrix.agents.prompts import (
    build_agent_kernel_system_prompt,
    build_agent_planner_prompt,
)

DOCTRINE = "eigenstaendige, natuerlich formulierte Evidenzfrage"
ANTI_KEYWORD = "keine Keyword-Kette"


def test_kernel_system_prompt_carries_the_doctrine() -> None:
    prompt = build_agent_kernel_system_prompt()
    assert DOCTRINE in prompt
    assert ANTI_KEYWORD in prompt
    assert "SUCHQUERY" not in prompt
    assert "Schluesselwoerter" not in prompt
    # B3: the recency rule folds the meant time range INTO the evidence
    # question instead of a provider-side recency filter alone.
    assert "nimm den gemeinten Zeitraum" in prompt
    assert "ausdruecklich in die Evidenzfrage" in prompt


def test_web_instant_tool_description_carries_the_doctrine() -> None:
    tools = build_kernel_tools()
    web_instant = next(tool for tool in tools if tool.name == "web_instant")
    description = web_instant.description
    assert "eigenstaendige, natuerlich formulierte Evidenzfrage" in description
    assert "Keyword-Kette" in description
    assert "SUCHQUERY" not in description
    assert "Schluesselwoerter" not in description


def test_planner_prompt_carries_the_doctrine_in_both_web_branches() -> None:
    for research_allowed in (False, True):
        prompt = build_agent_planner_prompt(
            "Wie entwickelt sich der Markt?",
            "(keine Erkundung)",
            ["Kriterium"],
            max_tasks=6,
            web_allowed=True,
            research_allowed=research_allowed,
            research_profile="deep" if research_allowed else None,
        )
        assert "natuerlich formulierte Evidenzfrage" in prompt
        assert "keine Keyword-Kette" in prompt
        assert "SUCHQUERY" not in prompt


def test_quick_web_derivation_prompt_carries_the_doctrine() -> None:
    # The derivation prompt is an inline string; the source pin keeps the
    # stub-visible prefix ("Formuliere genau EINE") AND the doctrine.
    source = inspect.getsource(KernelAgentAlgorithm._derive_quick_web_query)
    assert "Formuliere genau EINE" in source
    assert "natuerlich formulierte" in source
    assert "Evidenzfrage" in source
    assert "keine Keyword-Kette" in source


def test_structured_schemas_carry_the_doctrine() -> None:
    quick = QuickWebQuery.model_json_schema()
    assert "evidence question" in quick["properties"]["query"]["description"]
    assert "keyword chain" in quick["properties"]["query"]["description"]
    gap = DiscoveryGap.model_json_schema()
    assert (
        "evidence questions"
        in gap["properties"]["suggested_queries"]["description"]
    )
    task = PlanTaskModel.model_json_schema()
    assert "questions" in task["properties"]["queries"]["description"]
    assert "keyword chains" in task["properties"]["queries"]["description"]
