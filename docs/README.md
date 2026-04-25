# Documentation

## Scope

Start here when you know what you want to do but not which page owns it. The root [`README.md`](../README.md) stays the short project front page; this file is the task-oriented map for the full `docs/` tree.

## Start by task

| I want to... | Go to |
|--------------|-------|
| Install the project and run the offline suite | [Installation](getting-started/installation.md) |
| Run the first live research question | [First research run](getting-started/first-research-run.md) |
| Embed Inqtrix in a Python script | [Library mode](deployment/library-mode.md) |
| Run the OpenAI-compatible HTTP server | [Web server mode](deployment/webserver-mode.md) |
| Use the bundled Streamlit chat UI | [Streamlit UI](deployment/streamlit-ui.md) |
| Pick a provider stack | [Providers overview](providers/overview.md) |
| Find runnable example scripts | [Examples index](../examples/README.md) |
| Configure env vars, YAML, or report profiles | [Settings and env](configuration/settings-and-env.md), [inqtrix.yaml](configuration/inqtrix-yaml.md), [Report profiles](configuration/report-profiles.md) |
| Understand the LangGraph loop | [Architecture overview](architecture/overview.md), [Graph topology](architecture/graph-topology.md), [Nodes](architecture/nodes.md) |
| Debug a bad or expensive run | [Debugging runs](observability/debugging-runs.md), [Logging](observability/logging.md), [Troubleshooting](reference/troubleshooting.md) |
| Learn why a run stopped | [Stop criteria](scoring-and-stopping/stop-criteria.md), [Confidence](scoring-and-stopping/confidence.md), [Aspect coverage](scoring-and-stopping/aspect-coverage.md) |
| Add a custom provider or strategy | [Writing a custom provider](providers/writing-a-custom-provider.md), [Strategies](architecture/strategies.md) |
| Work on tests, replay cassettes, or releases | [Running tests](development/running-tests.md), [Testing strategy](development/testing-strategy.md), [Release process](development/release-process.md) |

## Paths by audience

**First-time user:** [Installation](getting-started/installation.md) -> [First research run](getting-started/first-research-run.md) -> [Examples index](../examples/README.md).

**Application integrator:** [Library mode](deployment/library-mode.md) -> [Providers overview](providers/overview.md) -> [Agent config](configuration/agent-config.md) -> [Result schema](architecture/result-schema.md).

**HTTP operator:** [Web server mode](deployment/webserver-mode.md) -> [Security hardening](deployment/security-hardening.md) -> [Logging](observability/logging.md) -> [Troubleshooting](reference/troubleshooting.md).

**Contributor:** [Architecture overview](architecture/overview.md) -> [State and iteration](architecture/state-and-iteration.md) -> [Coding standards](development/coding-standards.md) -> [Running tests](development/running-tests.md).

## Folder map

| Folder | Use it for |
|--------|------------|
| `getting-started/` | Install, first run, high-level product overview. |
| `deployment/` | Library mode, HTTP mode, Streamlit UI, Azure deployment, security layers. |
| `providers/` | Built-in provider pages and custom-provider guidance. |
| `configuration/` | `AgentConfig`, env vars, YAML config, report profiles. |
| `architecture/` | Graph topology, nodes, state, public API, result schema, strategies. |
| `scoring-and-stopping/` | Claim quality, aspect coverage, confidence, falsification, source tiering, stop criteria. |
| `observability/` | Logs, progress events, iteration logs, timeouts, debugging workflows. |
| `development/` | Contribution rules, docs maintenance, tests, parity tooling, releases. |
| `reference/` | FAQ, troubleshooting, glossary, worked example, research foundations, changelog. |

## Related docs

- [Examples index](../examples/README.md)
- [Docs maintenance](development/docs-maintenance.md)
- [Architecture overview](architecture/overview.md)
- [Troubleshooting](reference/troubleshooting.md)
