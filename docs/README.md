# Documentation

## Scope

Start here when you know what you want to do but not which page owns it. The root [`README.md`](../README.md) stays the short project front page; this file is the task-oriented map for the full `docs/` tree.

## Start by task

| I want to... | Go to |
|--------------|-------|
| Install the project and run the offline suite | [Installation](getting-started/installation.md) |
| Run the first live research question | [First research run](getting-started/first-research-run.md) |
| Run the full stack with one command (Docker) | [Stack quickstart](getting-started/stack-quickstart.md) |
| Deploy on Kubernetes or OpenShift (Helm chart) | [Kubernetes and OpenShift](deployment/kubernetes.md) |
| Upgrade a bundled or managed PostgreSQL database safely | [Database migrations](deployment/database-migrations.md) |
| Configure AWS S3, workload identity, or an S3-compatible store | [Object storage](deployment/object-storage.md) |
| Decide which way to run it | [Deployment modes](deployment/deployment-modes.md) |
| Decide which infrastructure I need (Postgres, S3, Qdrant, workers, OIDC) | [Platform components](getting-started/platform-components.md) |
| Operate the stack (start/stop/update/backup) | [Runbooks](deployment/runbooks.md) |
| Enable and operate live editor collaboration | [Deploy editor collaboration](deployment/editor-collaboration.md) |
| Share and review a live editor document | [Collaborate on editor documents](how-to/collaborate-on-editor-documents.md) |
| Get a cited answer from my own documents | [First knowledge answer](getting-started/first-knowledge-answer.md), [Knowledge engine](knowledge/overview.md), [Retrieval profiles](configuration/knowledge-profiles.md) |
| Choose or connect an authentication mode | [Auth modes](deployment/auth-modes.md) |
| Create or manage users (owner setup, roles, disabling, PATs) | [Create and manage users](how-to/create-and-manage-users.md) |
| Connect to my existing LDAP/AD directory | [Connect to an existing LDAP](how-to/connect-to-existing-ldap.md) |
| Deploy to production (TLS, secrets, backups, hardening) | [Deploy to production](how-to/deploy-to-production.md) |
| Embed Inqtrix in a Python script | [Library mode](deployment/library-mode.md) |
| Run the OpenAI-compatible HTTP server | [Web server mode](deployment/webserver-mode.md) |
| Build or deploy the React/browser UI | [React UI](deployment/react-ui.md), [Run events](observability/run-events.md), [Web server mode](deployment/webserver-mode.md) |
| Build my own UI or client on the native API | [Build a UI on Inqtrix](how-to/build-a-ui-on-inqtrix.md) |
| Pick a provider stack | [Providers overview](providers/overview.md) |
| Configure a provider combo via `.env` (Azure, Anthropic, Bedrock, ...) | [Provider recipes](getting-started/provider-recipes.md) |
| Find runnable example scripts | [Examples index](../examples/README.md) |
| Configure env vars or report profiles | [Configuration cookbook](configuration/configuration-cookbook.md), [Settings and env](configuration/settings-and-env.md), [Report profiles](configuration/report-profiles.md) |
| Understand the LangGraph loop and evidence flow | [Architecture overview](architecture/overview.md), [Graph topology](architecture/graph-topology.md), [Nodes](architecture/nodes.md), [Evidence pipeline](architecture/evidence-pipeline.md) |
| Understand the knowledge/RAG retrieval pipeline (hybrid, RRF, gate loop) | [Knowledge retrieval](architecture/knowledge-retrieval.md) |
| Understand the workspace agent (plans, approvals, child runs, memo canvas) | [Agent platform](architecture/agent-platform.md) |
| Understand the live editor data and trust model | [Editor collaboration](architecture/editor-collaboration.md) |
| Debug a bad or expensive run | [Debugging runs](observability/debugging-runs.md), [Logging](observability/logging.md), [Forensic cookbook](observability/forensic-cookbook.md), [Troubleshooting](reference/troubleshooting.md) |
| Add or change instrumentation (spans, events, attributes) | [Tracing legend](observability/tracing-legend.md) |
| Scrape metrics / wire readiness probes | [Metrics](observability/metrics.md) |
| Learn why a run stopped | [Stop criteria](scoring-and-stopping/stop-criteria.md), [Score ledger](scoring-and-stopping/score-ledger.md), [Confidence](scoring-and-stopping/confidence.md), [Aspect coverage](scoring-and-stopping/aspect-coverage.md) |
| Add a custom provider or strategy | [Writing a custom provider](providers/writing-a-custom-provider.md), [Strategies](architecture/strategies.md) |
| Plug in a custom auth provider or storage backend | [Custom auth provider](how-to/writing-a-custom-auth-provider.md), [Custom storage backend](how-to/writing-a-custom-storage.md) |
| Work on tests, replay cassettes, or releases | [Running tests](development/running-tests.md), [Testing strategy](development/testing-strategy.md), [Release process](development/release-process.md) |
| See how AI-generated content is marked, and what that means for operators | [AI transparency](reference/ai-transparency.md) |

## Paths by audience

**First-time user:** [Installation](getting-started/installation.md) -> [First research run](getting-started/first-research-run.md) -> [First knowledge answer](getting-started/first-knowledge-answer.md).

**Operator (self-hosting the product):** [Stack quickstart](getting-started/stack-quickstart.md) -> [Provider recipes](getting-started/provider-recipes.md) -> [Platform components](getting-started/platform-components.md) -> [Database migrations](deployment/database-migrations.md) / [Object storage](deployment/object-storage.md) -> [Create and manage users](how-to/create-and-manage-users.md) -> [Runbooks](deployment/runbooks.md) -> optional [Editor collaboration](deployment/editor-collaboration.md) -> [Deploy to production](how-to/deploy-to-production.md).

**Application integrator:** [Library mode](deployment/library-mode.md) -> [Build a UI on Inqtrix](how-to/build-a-ui-on-inqtrix.md) -> [Custom auth provider](how-to/writing-a-custom-auth-provider.md) / [storage backend](how-to/writing-a-custom-storage.md) -> [Result schema](architecture/result-schema.md).

**HTTP operator:** [Web server mode](deployment/webserver-mode.md) -> [Auth modes](deployment/auth-modes.md) -> [Security hardening](deployment/security-hardening.md) -> [Logging](observability/logging.md) -> [Troubleshooting](reference/troubleshooting.md).

**Contributor:** [Architecture overview](architecture/overview.md) -> [State and iteration](architecture/state-and-iteration.md) -> [Coding standards](development/coding-standards.md) -> [Running tests](development/running-tests.md).

## Folder map

| Folder | Use it for |
|--------|------------|
| `getting-started/` | Install, first run, full stack, first knowledge answer, high-level overview. |
| `how-to/` | End-to-end task guides: create/manage users, collaborate on editor documents, connect to existing LDAP, custom auth/storage, deploy to production, build a UI. |
| `deployment/` | Library mode, HTTP mode, auth modes, React UI, optional editor collaboration, Azure deployment, security layers. |
| `knowledge/` | Knowledge engine: collections, ingestion, answer path, Wissen workspace, evaluation tiers. |
| `providers/` | Built-in provider pages and custom-provider guidance. |
| `configuration/` | `AgentConfig`, env vars, report profiles, retrieval profiles, [configuration cookbook](configuration/configuration-cookbook.md). |
| `architecture/` | Graph topology, nodes, state, evidence pipeline, knowledge retrieval, editor collaboration, public API, result schema, strategies. |
| `scoring-and-stopping/` | Claim quality, aspect coverage, confidence, falsification, source tiering, stop criteria. |
| `observability/` | Logs, progress events, iteration logs, timeouts, debugging workflows. |
| `development/` | Contribution rules, docs maintenance, tests, parity tooling, releases. |
| `reference/` | FAQ, troubleshooting, glossary, worked example, research foundations, AI transparency, changelog. |

## Related docs

- [Examples index](../examples/README.md)
- [Docs maintenance](development/docs-maintenance.md)
- [Architecture overview](architecture/overview.md)
- [Evidence pipeline](architecture/evidence-pipeline.md)
- [Knowledge retrieval](architecture/knowledge-retrieval.md)
- [Agent platform](architecture/agent-platform.md)
- [AI transparency](reference/ai-transparency.md)
- [Troubleshooting](reference/troubleshooting.md)
