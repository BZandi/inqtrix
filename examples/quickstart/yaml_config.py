"""Use YAML provider/model routing in library mode.

Use this example when:
- you want the same provider routing as the HTTP server
- you prefer a Python script over `python -m inqtrix`
- secrets should stay in `.env` and YAML should reference them via `${ENV_VAR}`

Important detail:
- this example does not call `load_dotenv()` manually
- `load_config()` auto-loads the local `.env` before resolving `${ENV_VAR}` placeholders

Required files:
- `inqtrix.yaml` (or another path passed to `load_config()`)
- optional `.env` with the secrets referenced by `${...}` in the YAML file

Run with:
    uv run python examples/quickstart/yaml_config.py
"""

from __future__ import annotations

from inqtrix import AgentConfig, ResearchAgent
from inqtrix.config import load_config
from inqtrix.config_bridge import config_to_settings, create_providers_from_config


CONFIG_PATH = "inqtrix.yaml"
QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"


def main() -> None:
    config = load_config(CONFIG_PATH)
    settings = config_to_settings(config)
    providers = create_providers_from_config(config, settings)

    agent = ResearchAgent(AgentConfig(
        llm=providers.llm,
        search=providers.search,
        **settings.agent.model_dump(),
    ))

    result = agent.research(QUESTION)
    print(result.answer)
    print()
    print(f"Confidence: {result.metrics.confidence}/10")
    print(f"Sources: {result.metrics.total_citations}")
    print(f"Rounds: {result.metrics.rounds}")


if __name__ == "__main__":
    main()
