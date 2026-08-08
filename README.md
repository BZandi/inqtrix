<div align="center">
  <a name="readme-top"></a>
  <img src="assets/inqtrix-png-logo-kit-v5/inqtrix-github-hero-compact-fullwidth.png" alt="Inqtrix: iterative AI research agent" width="100%">

  <p></p>

  <p>
    <b>Self-hosted, iterative AI research agent with a unified research workspace around it.</b><br>
    <sub>Built for analysts · academics · journalists · consultants · knowledge teams · technical researchers</sub><br><br>
    <b>From question → sources → claims → evidence → report → editor → reusable knowledge</b><br>
    It searches the web <i>and</i> your own documents, verifies every claim, and returns cited reports;<br>
    then chat, write, and refine the results in one place. Your keys, one command to deploy.<br>
  </p>

  <p>
    <a href="#what-is-inqtrix"><b>What is it</b></a> &nbsp;·&nbsp;
    <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
    <a href="#ways-to-run-inqtrix"><b>Ways to run</b></a> &nbsp;·&nbsp;
    <a href="#how-the-research-loop-works"><b>How it works</b></a> &nbsp;·&nbsp;
    <a href="docs/README.md"><b>Documentation</b></a><br>
  </p>

  <p>
    <a href="docs/reference/changelog.md"><img alt="Version 0.2.0" src="https://img.shields.io/badge/Version-0.2.0-0B1B33?style=flat-square"></a>
    <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/License-AGPL--3.0-0F6E56?style=flat-square"></a>
    <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-185FA5?style=flat-square&logo=python&logoColor=white">
    <a href="docs/getting-started/stack-quickstart.md"><img alt="Deploy with Docker Compose" src="https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?style=flat-square&logo=docker&logoColor=white"></a>
    <img alt="OpenAI-compatible API" src="https://img.shields.io/badge/API-OpenAI--compatible-412991?style=flat-square&logo=openai&logoColor=white">
    <img alt="Built with LangGraph" src="https://img.shields.io/badge/Built%20with-LangGraph-993C1D?style=flat-square">
    <img alt="UI: React + Vite" src="https://img.shields.io/badge/UI-React%20%2B%20Vite-185FA5?style=flat-square&logo=react&logoColor=white">
    <img alt="Self-hosted" src="https://img.shields.io/badge/Self--hosted-bring%20your%20own%20keys-0B1B33?style=flat-square">
  </p>
</div>

<details>
<summary><b>Status &amp; disclaimer: experimental software. Read before use</b></summary>
<p></p>

> [!WARNING]
>This >repository is an experimental codebase and integration foundation for self-hosted or locally operated deployments. It does **not** provide a complete production-ready security configuration, hardened deployment profile, or any assurance that it is suitable for direct use in internet-facing, multi-user, regulated, or otherwise high-risk environments.
>
>Configurations, defaults, example values, example scripts, and helper paths included here may be useful for development, testing, or integration work, but must not be assumed to be secure, complete, or appropriate for production use without independent review and adaptation.
>
>Secure configuration, hardening, deployment architecture, access control, secret handling, logging, monitoring, compliance, and day-to-day operation remain the sole responsibility of the operator. The current test suite covers substantial internal logic, interface behavior, and regression scenarios, but is **not** evidence of production readiness or fully validated live >integrations. Perform your own technical and security review before using this >project in integration, staging, test, or production environments. See [Security hardening](docs/deployment/security-hardening.md).

</details>

<details>
<summary><kbd>Table of contents</kbd></summary>

- [What is Inqtrix](#what-is-inqtrix)
- [Demo](#demo)
- [Features](#features)
- [Quick start](#quick-start)
- [Ways to run Inqtrix](#ways-to-run-inqtrix)
  - [A. Full stack (Docker Compose)](#a-full-stack-docker-compose)
  - [B. Standalone API server](#b-standalone-api-server)
  - [C. Python library](#c-python-library)
  - [D. Developer components](#d-developer-components)
- [Capability tiers (opt-in)](#capability-tiers-opt-in)
- [The Research Desk (web app)](#the-research-desk-web-app)
- [How the research loop works](#how-the-research-loop-works)
- [Configuration essentials](#configuration-essentials)
- [Architecture &amp; components](#architecture--components)
- [Documentation map](#documentation-map)
- [Project &amp; license](#project--license)

</details>

## What is Inqtrix

Inqtrix began as an iterative **web-research agent**: ask a question and it plans queries, searches in parallel, extracts and verifies claims, tiers its sources, and returns a cited, audit-ready report. That engine is still the core, but Inqtrix has grown into an **open-source, self-hosted AI research workspace**: a single place to run that research, query your *own* documents, and turn the findings into finished writing. No tool-switching, no media breaks: everything from the first query to the final document stays in one workspace, on your own infrastructure and your own keys.

It comes in two halves you can adopt independently.

**The backend is one engine you can embed or deploy as a web server.** The core is two engines in one: the iterative web-research loop *and* a knowledge/RAG engine that answers from your own document collections with verified citations. `import ResearchAgent` to run it in-process as a **Python library**, or run `python -m inqtrix` to expose it as a **self-hostable HTTP service** with an OpenAI-compatible `/v1/chat/completions` endpoint, a native run API with live progress streaming, and the knowledge API, all from one process. And it is a real multi-user platform, not a demo wrapper: five pluggable auth modes (`none` / `apikey` / `local` / `oidc` / `ldap`), workspaces, per-user quotas, sharing, durable Postgres storage, S3 object storage, a Qdrant vector store for RAG, and optional worker processes for scale. RAG, auth, and multi-user all live in the backend; the frontend is optional.

**The Research Desk is the unified workspace around it.** On top of that API sits a React workspace that brings search, processing, and authoring into one place.

- **Research chat:** launch research runs and watch them stream live, then refine the results in a chat that can cite the same reports and files.
- **Knowledge:** ingest documents into collections and ask cited questions over them (the RAG engine, with a UI).
- **Editor:** write Markdown, select a passage or give a document-level instruction, and have the AI return revisions as **accept/reject tracked changes**; optionally co-edit shared documents live with carets and `edit`/`suggest` review modes; export to Word in one click.
- **Prompt Library:** reusable prompt templates (instructions, functions, context packs) that you chain together as chips in both chat and editor.
- **Files &amp; collaboration:** one private attachment pool referenced as numbered `[N]` chips everywhere, plus consent-based sharing for runs, knowledge collections, prompts, skills, and editor documents; editor shares add a constrained `suggest` role alongside `view`/`edit`.

What ties it together is a single mention/chip composer: research reports, files, and prompts all become reusable building blocks you drop into a message or a document. The result is one loop from **search &rarr; query your documents &rarr; chat &rarr; write &rarr; refine &rarr; export** in one workspace instead of five disconnected tools.

Two principles run through all of it. **Provider-neutral:** bring your own LLM and search keys (any OpenAI-compatible gateway, Anthropic, Azure OpenAI, Bedrock; Perplexity or Azure web search), nothing locked to one vendor. **Visibility over cleverness:** every fallback, confidence cap, and stop decision is logged and streamed, so you can always see *why* the agent did what it did.

## Demo

<div align="center">
  <img src="assets/Demo-1.gif" alt="An Inqtrix research run streaming live" width="100%">
  <br><br>
  <img src="assets/inqtrix-react-ui-preview.png" alt="The Research Desk workspace" width="100%">
</div>

## Features

<table>
<tr>
<td valign="top" width="50%">

**Research &amp; evidence engine**

- **Iterative research loop.** A bounded, multi-round LangGraph cycle (classify &rarr; plan &rarr; search &rarr; evaluate &rarr; answer) that repeats until it is confident or hits its budget.
- **Parallel search with claim extraction.** Queries run in parallel; each result is LLM-summarised and broken into structured atomic claims, with non-fatal per-source fallback.
- **Nine stopping heuristics.** Confidence, stagnation, utility and confidence plateaus, contradictions, competing events and more, instead of one gameable confidence number.
- **Verified / contested / unverified claims.** Claims are consolidated across rounds and labelled by evidence, with a depth-weighted quality score.
- **Five-tier source scoring.** Every URL is ranked primary / mainstream / stakeholder / unknown / low over a curated domain map.
- **Aspect coverage.** The question is decomposed into required aspects, and gap-filling queries are forced before high confidence is allowed.
- **Falsification mode.** After repeated low-confidence rounds the agent actively searches for *disproof* (anti-sycophancy).
- **Report profiles.** `compact` (default) or `deep` switches round budget and depth in one preset.

</td>
<td valign="top" width="50%">

**Knowledge, workspace &amp; platform**

- **Knowledge engine (RAG, opt-in).** Cited answers over your own documents with hybrid dense + BM25 retrieval, quote-verified grounding, and five retrieval profiles.
- **Two API surfaces.** A drop-in OpenAI-compatible `/v1/chat/completions`, plus a native `/v1/runs` API with queueing, live SSE progress, and cancellation.
- **Research Desk.** Research job cards, live progress streams, parallel runs, and an audit-ready report viewer.
- **Cited collaborative editor.** Write Markdown, have the AI return private revisions as accept/reject tracked changes, and optionally co-edit shared documents live with carets plus direct-edit and suggestion workflows; inline comments; one-click Export to Word.
- **Prompt Library.** Reusable templates (instructions, functions, context packs) that chain together as chips in both chat and the editor.
- **Private files &amp; portable projects.** One owner-bound attachment pool referenced as `[N]` chips everywhere; the whole workspace exports and re-imports as plain Markdown.
- **Multi-user platform.** Five auth modes (`none` / `apikey` / `local` / `oidc` / `ldap`), canonical UUID users, workspaces, per-user quotas, direct consent-based sharing, and an admin panel.
- **Baukasten &amp; durable backends.** Swap LLM/search providers and six strategy seams; zero-infra by default, opt-in Postgres / S3 / Qdrant / Valkey workers; every fallback is logged (no silent fallbacks).

</td>
</tr>
</table>


Inqtrix isn't trying to be another chat UI. The goal is to become the open-source home for **research work**: a place where every AI claim is tied to its sources, its counter-sources, and its evidence, and where research turns directly into reliable reports and documents.

If that direction excites you, contributions are very welcome. See [Contributing](docs/development/contributing.md).


## Quick start

Docker Compose runs the complete self-hosted stack in three steps: web UI, API, and database behind one URL, with no Python toolchain on your machine.

### Step 1: What you need

Inqtrix is provider-neutral: you bring your own accounts and keys, and each
entry in the paired stack configuration has a specific job. You cannot start
it empty.

| You need (one per row) | What it's for |
|---|---|
| **An LLM provider key:** Anthropic, Azure OpenAI, AWS Bedrock, or any OpenAI-compatible gateway via LiteLLM (this reaches OpenAI, OpenRouter, vLLM, Ollama today; a native OpenAI provider is coming) | the reasoning: query planning, summarisation, claim extraction, and answer writing |
| **A web-search provider key:** Perplexity, or Azure AI Foundry Web Search | the actual web search each research round runs |
| **Docker or Podman** | runs the stack (no account needed) |
| **Three secrets:** a Postgres password, `INQTRIX_SESSION_SECRET`, `INQTRIX_PAT_PEPPER` | local random strings you generate yourself (no account needed) |

Want a knowledge base (RAG) over your own documents too? Then you also need an **embedding provider** for indexing. It usually runs through your LLM provider when that is OpenAI-compatible (the LiteLLM gateway); with Anthropic or Bedrock you point it at a separate OpenAI-compatible endpoint or Azure. A **reranker** (Cohere, or served via Azure or Bedrock; tested with Cohere) is optional but recommended for the best retrieval quality. You configure both in Step 2's optional block.

### Step 2: Clone and configure

```bash
git clone https://github.com/BZandi/inqtrix.git && cd inqtrix
cp deploy/.env.stack.example deploy/.env.stack
cp deploy/.env.stack.secrets.example deploy/.env.stack.secrets
chmod 0600 deploy/.env.stack.secrets
```

`deploy/.env.stack` is the transparent, non-secret configuration: the complete
database URL, hosts, ports, profiles, providers, and model choices remain
visible there. `deploy/.env.stack.secrets` contains the values that must not be
committed. The visible DSN in the config references
`${INQTRIX_PG_PASSWORD}`; no helper script assembles or materializes it.

Set these independent values in `deploy/.env.stack.secrets` first:

```dotenv
INQTRIX_PG_PASSWORD=replace-with-url-safe-random-value
INQTRIX_SESSION_SECRET=replace-with-independent-random-value
INQTRIX_PAT_PEPPER=replace-with-another-independent-random-value
```

Then choose one LLM and one search provider below. Put each **Configuration**
block in `deploy/.env.stack` and each **Credentials** block in
`deploy/.env.stack.secrets`; replace the existing provider assignments in the
copied config instead of appending duplicate keys. Uncomment only the selected
credential lines and leave every unused provider credential commented. The
preflight rejects duplicate keys in either file. Complete provider combinations
and workload identity alternatives remain in
[Provider recipes](docs/getting-started/provider-recipes.md).

<details>
<summary><b>Anthropic + Perplexity</b> <sub>(simplest cloud setup)</sub></summary>

Configuration:

```dotenv
INQTRIX_LLM_PROVIDER=anthropic
REASONING_MODEL=claude-opus-4-8
TIER_HIGH_MODEL=claude-opus-4-8
TIER_MID_MODEL=claude-sonnet-4-6
TIER_FAST_MODEL=claude-haiku-4-5
TIER_HIGH_EFFORT=medium
INQTRIX_SEARCH_PROVIDER=perplexity
INQTRIX_SELECTABLE_CHAT_MODELS=claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5
```

Credentials:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
PERPLEXITY_API_KEY=pplx-...
```

</details>

<details>
<summary><b>OpenAI / OpenAI-compatible gateway (LiteLLM) + Perplexity</b></summary>

Configuration:

```dotenv
INQTRIX_LLM_PROVIDER=litellm
LITELLM_BASE_URL=https://api.openai.com/v1
REASONING_MODEL=gpt-4o
INQTRIX_SEARCH_PROVIDER=perplexity
SEARCH_MODEL=perplexity-sonar-pro-agent
INQTRIX_SELECTABLE_CHAT_MODELS=gpt-4o,gpt-4o-mini
```

Credentials:

```dotenv
LITELLM_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...
```

</details>

<details>
<summary><b>Azure OpenAI + Perplexity</b></summary>

Configuration:

```dotenv
INQTRIX_LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
REASONING_MODEL=gpt-5.4
TIER_HIGH_MODEL=gpt-5.4
TIER_FAST_MODEL=gpt-5.4-mini
INQTRIX_SEARCH_PROVIDER=perplexity
INQTRIX_SELECTABLE_CHAT_MODELS=gpt-5.4,gpt-5.4-mini
```

Credentials:

```dotenv
AZURE_OPENAI_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
```

</details>

<details>
<summary><b>All-Azure: Azure OpenAI + Azure AI Foundry Web Search</b></summary>

Configuration:

```dotenv
INQTRIX_LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
REASONING_MODEL=gpt-5.4
INQTRIX_SEARCH_PROVIDER=azure_foundry
AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
WEB_SEARCH_AGENT_NAME=web-search-agent
INQTRIX_SELECTABLE_CHAT_MODELS=gpt-5.4,gpt-5.4-mini
```

Credentials:

```dotenv
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
```

</details>

<details>
<summary><b>AWS Bedrock + Perplexity</b></summary>

Configuration:

```dotenv
INQTRIX_LLM_PROVIDER=bedrock
AWS_REGION=eu-central-1
REASONING_MODEL=eu.anthropic.claude-opus-4-8-v1
TIER_HIGH_MODEL=eu.anthropic.claude-opus-4-8-v1
TIER_MID_MODEL=eu.anthropic.claude-sonnet-4-6
TIER_FAST_MODEL=eu.anthropic.claude-haiku-4-5
INQTRIX_SEARCH_PROVIDER=perplexity
INQTRIX_SELECTABLE_CHAT_MODELS=eu.anthropic.claude-opus-4-8-v1,eu.anthropic.claude-sonnet-4-6
```

Credentials (prefer workload identity where available):

```dotenv
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
PERPLEXITY_API_KEY=pplx-...
```

</details>

<details>
<summary><b>Optional: add a knowledge base (RAG over your own documents)</b></summary>

Edit the matching canonical fields in `deploy/.env.stack` rather than appending
duplicate assignments, put any corresponding API keys in
`deploy/.env.stack.secrets`, then start with `--profile knowledge`. RAG is off
by default.

```dotenv
# --- Knowledge engine (RAG): persistent vector store via Qdrant ---
INQTRIX_KNOWLEDGE_ENABLED=true
INQTRIX_VECTOR_BACKEND=qdrant
# the `qdrant` compose service, started by --profile knowledge
INQTRIX_QDRANT_URL=http://qdrant:6333
# required for bundled Qdrant; belongs in deploy/.env.stack.secrets
# INQTRIX_QDRANT_API_KEY=generate-a-long-random-value

# --- Embedding model for indexing ---
# Reuses your OpenAI-compatible (LiteLLM) gateway by default. With Anthropic/Bedrock LLMs,
# set a dedicated embedding endpoint or use Azure (uncomment below).
INQTRIX_EMBEDDING_MODEL=text-embedding-3-small
# INQTRIX_EMBEDDING_PROVIDER=azure
# INQTRIX_EMBEDDING_BASE_URL=...

# --- Optional reranker for the best retrieval quality (tested with Cohere) ---
# INQTRIX_RERANKER_PROVIDER=cohere
# INQTRIX_RERANKER_BASE_URL=...    INQTRIX_RERANKER_MODEL=...
```

</details>

### Step 3: Start

```bash
# Research + chat (the default stack):
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets \
  --env-file deploy/.env.stack \
  up -d --build

# ...or with the knowledge base (RAG): add the knowledge profile
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets \
  --env-file deploy/.env.stack \
  --profile knowledge \
  up -d --build
```

> [!NOTE]
> **Using Podman?** The commands work unchanged with the Compose v2 provider:
> run `podman compose ...` in place of `docker compose ...`. The same ordered
> environment pair is used by Docker and Podman; no engine-specific secret
> materialization step exists.

The first build pulls images and installs dependencies (a few minutes); later starts are fast. Then open <http://localhost:8080>; the first visit walks you through creating the owner account. Verify with:

```bash
curl http://localhost:8080/health          # active providers + models
curl http://localhost:8080/v1/capabilities # which features are on (knowledge, files, sharing, ...)
```

The default stack is **Postgres + API + web** (research, chat, durable runs,
native login). Scaled workers (Valkey), bundled S3-compatible storage
(SeaweedFS), and the bundled SSO/LDAP references (Dex/LLDAP) are further
opt-in [capability tiers](#capability-tiers-opt-in) added with Compose profiles.
Managed or otherwise external S3 is selected only through configuration and
does not use the bundled `s3` profile.

<details>
<summary><b>All compose profiles (turn on more)</b></summary>

Each profile starts its bundled container(s); the matching variables in
`deploy/.env.stack` turn the feature on, so **you need both**. Profiles combine,
each with its own `--profile` flag, for example
`--profile knowledge --profile workers --profile s3` when the object store is
the bundled SeaweedFS service.

| `--profile` | Starts | Unlocks | Variables to set |
|---|---|---|---|
| *(none)* | postgres + migrate + api + web | research, chat, durable runs, login | (the default) |
| **`knowledge`** | qdrant | RAG / knowledge base over your documents | `INQTRIX_KNOWLEDGE_ENABLED=true`, `INQTRIX_VECTOR_BACKEND=qdrant`, `INQTRIX_QDRANT_URL`, required `INQTRIX_QDRANT_API_KEY` in the secrets file |
| **`workers`** | valkey + worker | scaled, restart-surviving runs | `INQTRIX_QUEUE_BACKEND=valkey`, `INQTRIX_VALKEY_*` |
| **`s3`** | seaweedfs | S3 object store instead of the local volume | `INQTRIX_OBJECT_STORE_BACKEND=s3`, `INQTRIX_S3_*` |
| **`collaboration`** | collaboration sidecar | real-time multi-user editing | `INQTRIX_COLLABORATION_ENABLED=true`, independent collaboration secret |
| **`pgbouncer`** | PgBouncer | optional transaction pooling for high database fan-in | runtime DSN explicitly changed to `pgbouncer:6432`; migrations remain direct |
| **`oidc`** | dex | enterprise SSO (Dex is the dev reference; any OIDC IdP works) | `INQTRIX_AUTH_MODE=oidc`, `INQTRIX_OIDC_*` |
| **`ldap`** | lldap | login against an LDAP/AD directory (LLDAP is the dev reference) | `INQTRIX_AUTH_MODE=ldap`, `INQTRIX_LDAP_*` |

`workers` deliberately starts two containers (`valkey` and `worker`) under one profile. Every variable and its meaning is in `deploy/.env.stack.example` and [Settings and env](docs/configuration/settings-and-env.md).
PgBouncer is never inferred from workload size or enabled by the deployment
CLI: both the `pgbouncer` profile and the visible pooler DSN are required.

</details>

<details>
<summary><b>Stop, start, update, and other lifecycle commands</b></summary>

The Compose file and both ordered environment files are required on every raw
command: secrets first, visible configuration second. With Podman it is
identical; only replace `docker compose` with `podman compose`.

Status and health of every service:

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack ps
```

Follow the API log (startup and config errors land here):

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack logs -f api
```

Stop and remove the containers, but keep your data (the volumes):

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack down
```

Start it again:

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack up -d
```

Restart without rebuilding:

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack restart api web
```

Update after a `git pull` (rebuild and re-run migrations):

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack up -d --build
```

Managed PostgreSQL in migration `owner` mode uses the guarded
`inqtrix-deploy ... owner-upgrade` command after the pull, not the generic
command above. Its raw-Compose maintenance sequence remains documented in
[Database migrations](docs/deployment/database-migrations.md).

Destroy everything, including the database and uploaded files (irreversible):

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack down -v
```

Backup, restore, and reset playbooks: [Runbooks](docs/deployment/runbooks.md). To type less, run `export COMPOSE_FILE=deploy/compose/compose.stack.yaml` once per shell and drop the `-f` flag from every command above.

</details><br>

**Go deeper:**

- [Provider recipes](docs/getting-started/provider-recipes.md) gives copy-paste `.env` blocks for each LLM and search combination.
- [Settings and env](docs/configuration/settings-and-env.md) documents every variable (secrets, models, storage, auth).
- [Platform components](docs/getting-started/platform-components.md) maps each
  feature to its required service and distinguishes bundled Compose profiles
  from managed dependencies (Qdrant / Valkey / S3 / OIDC / LDAP).
- [Auth modes](docs/deployment/auth-modes.md) covers login setup: native accounts, SSO, or LDAP.
- [Deployment quickstart](docs/getting-started/stack-quickstart.md) is the full step-by-step walkthrough.
- [Kubernetes and OpenShift](docs/deployment/kubernetes.md) deploys the same stack on a cluster via the bundled Helm chart.

> [!TIP]
> Prefer the engine on its own, or embedded in Python? Jump to [Ways to run Inqtrix](#ways-to-run-inqtrix). There are four, and this is just the first.

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>

## Ways to run Inqtrix

Same engine, four entry points: from a one-command deployment to a single Python import. Start with the row that matches you:

| You want to… | Run mode | Jump to |
| :--- | :--- | :--- |
| Self-host the complete stack, fast | **A.** Docker Compose | [→](#a-full-stack-docker-compose) |
| Add research to your own backend / UI over HTTP | **B.** API server | [→](#b-standalone-api-server) |
| Embed research inside Python code | **C.** Library | [→](#c-python-library) |
| Extend it with custom providers, strategies, or the UI in dev | **D.** Developer components | [→](#d-developer-components) |

---

### A. Full stack (Docker Compose)

> **What it is:** the complete stack (React web app + API + Postgres) in one `docker compose up`.
> **Who it's for:** teams and self-hosters who want the whole stack from one command, with no Python toolchain and no host-side build.

A single command builds the API and web images, starts Postgres, runs the schema
migration once (the one-shot `migrate` service), then starts the FastAPI
backend and the Python web gateway. The browser talks to **one origin** at
`http://localhost:8080`; the gateway reverse-proxies `/api`, `/v1`, `/health`,
and `/collaboration` to the API internally, so there is no CORS and no
build-time API-URL coupling.

**Prerequisites.** Docker (or Podman with the docker-compose v2 provider), one LLM API key, and one search API key.

**1. Configure.**

```bash
cp deploy/.env.stack.example deploy/.env.stack
cp deploy/.env.stack.secrets.example deploy/.env.stack.secrets
chmod 0600 deploy/.env.stack.secrets
# edit both files
```

Set `INQTRIX_PG_PASSWORD`, `INQTRIX_SESSION_SECRET`,
`INQTRIX_PAT_PEPPER`, and provider credentials only in the secrets file. The
config file keeps the complete
`postgresql+asyncpg://...@postgres:5432/inqtrix` form and references the
password variable. Then select one LLM and one search provider. Switching to
Azure, Anthropic, or Bedrock changes only this pair; no generated file or
provider-specific script is involved.

**2. Start.** Pass both files in the documented order (Compose does not
auto-load either filename):

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets \
  --env-file deploy/.env.stack \
  up -d --build
```

**3. Open** <http://localhost:8080>. The default stack uses **native accounts** (`INQTRIX_AUTH_MODE=local`): the first visit walks you through creating the instance owner; from then on the in-app admin area manages users, invitations, and tokens.

**4. Verify.**

```bash
curl http://localhost:8080/health          # status, active llm/search provider + models
curl http://localhost:8080/v1/capabilities # which features are on (knowledge, files, sharing, ...)
```

A missing or wrong credential surfaces as a loud startup error in `docker compose logs api`, naming the variable.

**You get:** the web UI, durable runs, native login, and the OpenAI-compatible API.
**You don't get (until enabled):** RAG over your documents, scaled/restart-surviving runs, S3 storage, or SSO. Those are opt-in [capability tiers](#capability-tiers-opt-in), turned on with compose profiles and no rebuild:

```bash
# Knowledge/RAG (Qdrant) and scaled runs (Valkey workers):
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack \
  --profile knowledge --profile workers up -d
```

> [!NOTE]
> Inside this Compose stack the API port `5100` is internal only; reach everything through the web origin `:8080`. The published web port is configurable (`INQTRIX_WEB_PORT`) and bound to loopback by default. For an explicit trusted-LAN test set `INQTRIX_WEB_BIND_ADDRESS=0.0.0.0`; for broader exposure terminate TLS in front. See the [Stack quickstart](docs/getting-started/stack-quickstart.md#3-open).

**Deeper:** [Deployment quickstart](docs/getting-started/stack-quickstart.md) (5-minute walkthrough) · [Platform components](docs/getting-started/platform-components.md) (which infra you need) · [Runbooks](docs/deployment/runbooks.md) (start/stop/update/backup/restore) · [Security hardening](docs/deployment/security-hardening.md).

---

### B. Standalone API server

> **What it is:** just the engine over HTTP, exposing an OpenAI-compatible endpoint plus a richer native run API.
> **Who it's for:** integrators building a frontend, an SDK client, or any HTTP integration without embedding Python.

```bash
# uv
uv sync --extra dev
uv run python -m inqtrix

# or standard Python/pip
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m inqtrix
```

Copy `.env.example` to `.env` and configure one LLM plus one search provider
before either start command. The API listens on <http://localhost:5100>.

Override the bind with `INQTRIX_SERVER_HOST` (default `0.0.0.0`) and `INQTRIX_SERVER_PORT` (default `5100`). Each request runs a fresh state; the server is stateless for research data unless you add durable backends.

```bash
curl -N http://localhost:5100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"research-agent","stream":true,
       "messages":[{"role":"user","content":"Was ist der Stand der GKV-Reform?"}]}'
```

Every endpoint accepts a top-level `mode`: `research` (the full loop), `direct_llm` (straight to the LLM), or `knowledge` (RAG over your collections, when enabled).

<details>
<summary><b>All endpoints &amp; the native run lifecycle</b></summary>

<p></p>

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness, active providers/models, `auth_mode`. *(open)* |
| `/v1/models` | GET | OpenAI-style model discovery. *(open)* |
| `/v1/capabilities` | GET | Feature manifest, which features are configured. *(open)* |
| `/v1/stacks` | GET | Multi-stack deployments: per-stack provider/model discovery. *(open)* |
| `/v1/chat/completions` | POST | OpenAI-compatible, streaming and non-streaming. *(gated)* |
| `/v1/runs` | POST | Submit a native run; returns immediately into a FIFO queue. *(gated)* |
| `/v1/runs/{id}/events` | GET | Buffered + live SSE event stream, the UI contract. |
| `/v1/runs/{id}/result` | GET | Final result export after completion. |
| `/v1/runs/{id}/cancel` | POST | Cancel a queued run immediately; a running run stops at the next node boundary. |
| `/v1/runs/{id}` | DELETE | Permanently delete a terminal run (owner-only); 409 while the run is still active. *(gated)* |

The SSE stream emits named lifecycle events (`inqtrix.run.queued/started/snapshot/completed/failed/cancelled`) plus progress (`inqtrix.node.started`, `inqtrix.progress.message`, `inqtrix.output_text.delta`).

**Auth gate.** `/health`, `/v1/models`, `/v1/capabilities`, `/v1/stacks` stay open; everything else is gated. With `INQTRIX_AUTH_MODE` unset (`infer`), a non-empty `INQTRIX_SERVER_API_KEY` enables a static Bearer gate; empty means open. A missing/wrong Bearer returns `401`.

**Multi-stack.** `examples/webserver_stacks/multi_stack.py` mounts every provider combination in one process; clients discover the active list via `GET /v1/stacks` and pick one per request with a top-level `"stack"` field.

</details>

**Deeper:** [Web server mode](docs/deployment/webserver-mode.md) (full endpoint surface, SSE schemas, concurrency, cancel) · [Auth modes](docs/deployment/auth-modes.md) · [Build a UI on Inqtrix](docs/how-to/build-a-ui-on-inqtrix.md) (consume the native run API from your own frontend).

---

### C. Python library

> **What it is:** the agent in-process; import `ResearchAgent`, call `.research(...)` or `.stream(...)`, get a typed result back.
> **Who it's for:** scripts, CLIs, notebooks, or embedding research inside a larger Python application.

```bash
# uv
uv sync --extra dev

# or standard Python/pip
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

cp .env.example .env
```

**Option A: auto-create providers from env** (both models reachable through one OpenAI-compatible endpoint):

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")

print(result.answer)
print(f"Confidence: {result.metrics.confidence}/10  "
      f"Sources: {result.metrics.total_citations}  "
      f"Rounds: {result.metrics.rounds}")
```

**Option B: explicit Baukasten constructors.** Providers are constructor-first (they never read env themselves); model names live on the *provider*, so swapping the whole stack means changing only the imports:

```python
from inqtrix import AgentConfig, AnthropicLLM, PerplexitySearch, ResearchAgent

llm = AnthropicLLM(api_key=..., default_model="claude-sonnet-4-6",
                   claim_extract_model="claude-haiku-4-5")
search = PerplexitySearch(api_key=...)

agent = ResearchAgent(AgentConfig(llm=llm, search=search))
result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")
```

LLM providers: `LiteLLM`, `AnthropicLLM`, `AzureOpenAILLM`, `BedrockLLM`. Search providers: `PerplexitySearch`, `AzureFoundryWebSearch`. `AgentConfig` carries *behaviour* (`max_rounds`, `confidence_stop`, `report_profile`, ...); `.stream(...)` yields live progress then answer chunks. The returned `ResearchResult` exposes `answer`, `metrics` (`confidence`, `total_citations`, `rounds`, ...), `top_sources`, `references`, and `top_claims`.

**Deeper:** [Library mode](docs/deployment/library-mode.md) (Option A vs B, model tiers, streaming) · [Agent config](docs/configuration/agent-config.md) (every field) · [Public API](docs/architecture/public-api.md).

---

### D. Developer components

> **What it is:** the building blocks, runnable example stacks, custom providers/strategies, and the React UI in dev mode.
> **Who it's for:** developers extending Inqtrix or hacking on one part in isolation.

- **Runnable examples:** `examples/provider_stacks/` (library) and `examples/webserver_stacks/` (HTTP) are 1:1 pairs; provider construction is byte-for-byte identical, only the run block differs. Standouts: `multi_stack.py` (all stacks, auto-discovered via `/v1/stacks`), `anthropic_perplexity_chat.py` (an interactive terminal REPL), `azure_knowledge_quickstart.py` (the RAG engine end-to-end). See [examples/](examples/README.md).
- **Custom providers &amp; strategies:** implement the `LLMProvider` / `SearchProvider` ABCs, or swap one of six strategy seams (source tiering, claim extraction/consolidation, context pruning, risk scoring, stop criteria). See [Writing a custom provider](docs/providers/writing-a-custom-provider.md).
- **Run the Research Desk in dev:** start a backend (`uv run python examples/webserver_stacks/multi_stack.py`, or plain `python` after a pip install), then `npm run ui:dev` for the Vite dev server on `http://localhost:5173`, proxying `/v1` + `/health` to `:5100`. Node ≥ 22.12 and npm ≥ 10.9. See [React UI](docs/deployment/react-ui.md).

**Deeper:** [Examples index](examples/README.md) · [Providers overview](docs/providers/overview.md) · [Strategies](docs/architecture/strategies.md) · [Contributing](docs/development/contributing.md).

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>

## Capability tiers (opt-in)

Inqtrix runs with **zero infrastructure** by default: in-memory storage,
in-memory queue, knowledge engine off. Each component below is opt-in, adds a
specific capability, and is reported live at `GET /v1/capabilities` so the UI
degrades *visibly* rather than failing silently. In Compose, bundled optional
components use a profile plus matching configuration; managed/external
services use configuration only and do not start the bundled profile.

| Add this | You unlock | What you lose without it | Turn it on |
|---|---|---|---|
| **Nothing** (default) | Research, chat, cited reports, OpenAI-compatible API | persistence, login, sharing | in-memory, zero infra |
| **Postgres** | Durable runs, login, multi-user, sharing, file uploads, prompt templates | everything is lost on restart | the Compose default |
| **Collaboration service** | Live editor co-authoring, carets, suggestions, and durable acknowledgements | shared editor documents remain read-only projections | `--profile collaboration` plus Postgres and cookie auth |
| **Qdrant** | Knowledge / RAG over your documents; hybrid dense + BM25 retrieval | in-memory, dense-only, lost on restart | `--profile knowledge` |
| **Valkey + worker** | Scaled, queued, restart-surviving runs | in-process queue only | `--profile workers` |
| **Object store (S3)** | Shared file storage across replicas | local volume only | managed/native S3: configuration only; bundled SeaweedFS: additionally `--profile s3` |
| **OIDC / LDAP** | Enterprise SSO / directory login | single-operator auth only | external IdP/directory: configuration only; bundled Dex/LLDAP development references: additionally `--profile oidc` / `--profile ldap` |

Each component, and the role it plays:

- **Postgres:** durable run rows, identity, knowledge metadata, prompt templates. The Compose deployment defaults to it; the `migrate` service applies the schema once before the API starts.
- **Collaboration service:** a private, single-replica Node/Hocuspocus coordinator for Yjs synchronization. FastAPI remains the identity and persistence authority, and the sidecar has neither a public port nor database credentials.
- **Qdrant:** the persistent vector + document store; the only backend with hybrid dense+BM25 retrieval. Without it the knowledge engine uses an in-memory, dense-only store.
- **Valkey + worker:** dispatches native runs to separate worker processes for horizontal scaling and restart survival. The worker refuses to start without both Postgres and Valkey.
- **Object store:** storage for uploaded file blobs; a local volume by default, or any S3 endpoint (SeaweedFS is the bundled reference).
- **OIDC / LDAP IdP:** browser SSO or directory bind login. External enterprise
  providers are configured without a Compose profile; `oidc` and `ldap`
  profiles start only the bundled Dex/LLDAP development references.

> [!NOTE]
> The simplest setups (`none`/`apikey` auth, in-memory storage) are perfectly usable for a single operator, but have no durable history, login, or sharing. Multi-user, invitations, and sharing require a cookie-session mode (`local`, `oidc`, or `ldap`) plus Postgres. That trade-off is by design.

**Deeper:** [Platform components](docs/getting-started/platform-components.md) (the full feature → component → auth-mode decision tree) · [Settings and env](docs/configuration/settings-and-env.md) (every variable).

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>

## The Research Desk (web app)

The Research Desk is Inqtrix's React workspace, a purpose-built surface for iterative research, not just a chat screen. It is a pure HTTP consumer of the backend (it never imports the Python package and never reads provider credentials).

- **Research job cards:** each question becomes its own queued `/v1/runs` card. Active jobs stream their current iteration (plan / search / evaluate / answer) live inside the card; new research can start while earlier runs keep going.
- **Audit-ready report viewer:** completed runs open in a right-side panel with tabs for **Preview** (the report), **Evidence** (the exact reference list), **Agent steps** (the archived protocol), and **Export** (a clean `.md` download).
- **Chat workspace:** a direct-LLM surface (OpenAI-compatible) with a model picker, KaTeX math, GFM, inline editing, and branch-from-response.
- **Editor workspace:** Markdown documents with live WYSIWYG editing, private inline comments, import from research reports, an AI assistant that publishes approved work as suggestions, optional multi-user co-authoring with `view`/`suggest`/`edit` roles, and one-click **Export to Word** (`.docx`).
- **Knowledge / Wissen workspace:** document collections with cited answers, quote verification, and retrieval profiles; an *ask / find / read* surface with passage highlighting.
- **File attachments:** an owner-bound library; files attached anywhere are parsed client-side and become auto-numbered `[N]` mention pills reused by chat and editor. Direct user-to-user file sharing is not supported.
- **Prompt Library:** project-scoped templates in three categories (Instructions, Functions, Context Packs), referenced in the composer via `@rules:` / `@research:`.
- **Project import / export:** the whole workspace is one portable project written as plain Markdown.

**You never build the React app yourself for a Compose deployment.** In the
packaged web image the bundle is built into the container and served by the
dependency-light Python web gateway on a single origin (`:8080`, no CORS). The
three ways the UI reaches a backend:

| Mode | Command | Serves | Origin |
|---|---|---|---|
| Full stack (Python web image) | container image | built `dist/` | single origin `:8080` → api `:5100` |
| Dev (HMR) | `npm run ui:dev` | live Vite build | `127.0.0.1:5173` → `:5100` |
| Standalone Python gateway | `python -m inqtrix_web_gateway --dist-dir apps/research-desk/dist` | pre-built `dist/` | `127.0.0.1:8080` → `INQTRIX_BACKEND_URL` |
| Optional nginx adapter | `compose.web-nginx.yaml` override | pre-built `dist/` | externally terminated HTTP/TLS edge |

> [!WARNING]
> Never put a Bearer token or API key in a `VITE_*` variable: Vite embeds those into the browser bundle. Enter Bearer tokens at runtime under Settings → Security.

**Deeper:** [React UI](docs/deployment/react-ui.md) (dev/build/gateway matrix,
Python web topology, API boundary) · [Editor collaboration](docs/architecture/editor-collaboration.md) · [Build a UI on Inqtrix](docs/how-to/build-a-ui-on-inqtrix.md).

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>

## How the research loop works

Inqtrix doesn't do single-pass "search → summarise." It runs a **bounded, multi-round loop** on a LangGraph state machine. A mutable state object threads through five nodes, and the loop keeps refining until independent stopping heuristics agree the answer is solid (or a hard round cap is hit).

```text
classify → plan → search → evaluate ──(not done)──► plan   (loop)
                                   └────(done)──────► answer → END
```

- **classify:** before any searching, detects answer/search language, decides whether web search is even needed, classifies the query type, decomposes it into sub-questions, and derives the *required aspects* the answer must cover.
- **plan:** generates the round's search queries. Round 0 is broad (default 6 queries, `compact` / 10, `deep`); later rounds are tight and *gap-targeted* (cross-check, primary-source, counter-evidence, perspective-diversity slots).
- **search:** runs all queries in parallel, then LLM-summarises each result and extracts structured atomic claims. Per-source failures are non-fatal and visibly marked.
- **evaluate:** the heart of the loop, where an LLM verdict (status, confidence, gaps, contradictions) is run through a cascade of deterministic guardrails that can only *lower* confidence, then through the stop heuristics. It either stops or hands the diagnosed gaps to the next `plan` round.
- **answer:** synthesises the final Markdown report with inline citations and returns the typed `ResearchResult`.

What makes each round different is that `evaluate` *diagnoses gaps* and feeds them forward: uncovered aspects become targeted queries; weak evidence becomes cross-checks; a low-confidence trajectory flips planning into **falsification mode**, where the agent actively searches for disproof instead of more confirmation.

<details>
<summary><b>The nine stopping heuristics, source tiering &amp; claim classification</b></summary>

<p></p>

**Why nine heuristics, not one threshold:** a single confidence number is gameable, so Inqtrix combines it with independent structural signals: (1) confidence threshold (default 8), (2) max rounds (4 `compact` / 5 `deep`), (3) contradictions, (4) competing events, (5) falsification mode, (6) stagnation, (7) utility plateau, (8) confidence plateau, (9) negative-evidence hinting. Global floors (`min_rounds`, a report-eligible-evidence floor) can suppress an early stop; `max_rounds` is the hard cap.

**Confidence can only go down.** The evaluator's raw 0-10 confidence passes through seven deterministic guardrails (e.g. no citations → cap 6, low-tier majority → cap 7, an uncovered required aspect → cap 8, two-plus contested claims → cap 7) before it is stored. None of them can *raise* it.

**Source tiering (5 tiers):** every cited URL is scored `primary` (1.0) / `mainstream` (0.8) / `stakeholder` (0.45) / `unknown` (0.35) / `low` (0.1) against a curated domain map; aggregate source quality is the weighted mean.

**Claim classification:** claims are deduplicated by a normalised signature, consolidated across rounds, and labelled `verified` (primary or cross-checked), `contested` (supporting *and* contradicting sources), or `unverified` (missing primary / weak evidence). A depth-weighted claim-quality score means a report built only on single-source claims can never reach a perfect score, by design.

</details>

**Knowledge engine (RAG).** The second engine answers from *your own documents* (`mode=knowledge`, off by default). Files are parsed to Markdown, chunked, optionally given situating context, embedded, and stored. Retrieval is hybrid (dense + German BM25), tunable through five profiles (`schnell` / `standard` / `gruendlich` / `tief` / `auto`), and answers are **grounded**: verbatim `[K#]` quotes are deterministically verified against the source text, and a three-way sufficiency gate yields an honest "no evidence" rather than a fabrication.

**Deeper:** [Architecture overview](docs/architecture/overview.md) · [Graph topology](docs/architecture/graph-topology.md) · [Stop criteria](docs/scoring-and-stopping/stop-criteria.md) · [Source tiering](docs/scoring-and-stopping/source-tiering.md) · [Knowledge engine](docs/knowledge/overview.md) · [Knowledge retrieval](docs/architecture/knowledge-retrieval.md).

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>

## Configuration essentials

Going from a quick try to the full self-hosted stack is an environment change,
not a code change. The pair has one clear responsibility per file:

- **Visible configuration in `deploy/.env.stack`:** the complete
  `INQTRIX_DATABASE_URL` expression, provider selection, model names, topology,
  and optional feature switches. The DSN references
  `${INQTRIX_PG_PASSWORD}` instead of repeating its value.
- **Credentials in `deploy/.env.stack.secrets`:** the PostgreSQL password,
  independent session and PAT secrets, and only the credentials for the chosen
  LLM/search and optional components.
- **Optional profiles:** `knowledge`, `workers`, `collaboration`, bundled
  `s3`, `oidc`, `ldap`, and `pgbouncer` each start only their corresponding
  local component. External S3 does not use the bundled `s3` profile.

Copy-paste blocks for every provider combination: [Provider recipes](docs/getting-started/provider-recipes.md). Every variable, grouped by purpose: [Settings and env](docs/configuration/settings-and-env.md).

## Architecture &amp; components

```mermaid
flowchart LR
    B["Browser"] --> W["Research Desk<br/>React web app"]
    W --> A["FastAPI server<br/>/v1 + services"]
    A <-->|"private collaboration"| C["Hocuspocus sidecar<br/>single replica"]
    A --> E["Research engine<br/>LangGraph loop"]
    E --> P["Providers<br/>LLM + web search"]
    A -. opt .-> PG[("Postgres")]
    A -. opt .-> Q[("Qdrant")]
    A -. opt .-> V[("Valkey + worker")]
    A -. opt .-> ID["OIDC / LDAP"]
```

The layering, top to bottom: the **Research Desk** (React) talks only to the **FastAPI server** (`server/routers` → `services` → `core`), which drives the **research engine** (`research/web_research.py` → the LangGraph loop) over swappable **providers**. The optional Hocuspocus sidecar receives only private WebSocket traffic and calls FastAPI for policy and persistence. Postgres, Qdrant, Valkey+worker, and an OIDC/LDAP IdP attach as optional components. The internal classify→plan→search→evaluate→answer algorithm is documented separately in [Graph topology](docs/architecture/graph-topology.md).

## Documentation map

The full navigation lives in the [documentation hub](docs/README.md). The pages you'll reach for most:

| I want to… | Start here |
|---|---|
| Run the full self-hosted stack in 5 minutes | [Deployment quickstart](docs/getting-started/stack-quickstart.md) |
| Decide which components I need | [Platform components](docs/getting-started/platform-components.md) |
| Do my first run with zero infra | [First research run](docs/getting-started/first-research-run.md) |
| Embed the engine in Python | [Library mode](docs/deployment/library-mode.md), [Agent config](docs/configuration/agent-config.md) |
| Serve the HTTP API / web UI | [Web server mode](docs/deployment/webserver-mode.md), [React UI](docs/deployment/react-ui.md) |
| Set up users &amp; login | [Auth modes](docs/deployment/auth-modes.md), [Create &amp; manage users](docs/how-to/create-and-manage-users.md) |
| Use RAG over my documents | [Knowledge engine](docs/knowledge/overview.md), [Knowledge retrieval](docs/architecture/knowledge-retrieval.md), [Knowledge profiles](docs/configuration/knowledge-profiles.md) |
| Choose / write providers | [Providers overview](docs/providers/overview.md), [Custom provider](docs/providers/writing-a-custom-provider.md) |
| Operate &amp; harden in production | [Runbooks](docs/deployment/runbooks.md), [Security hardening](docs/deployment/security-hardening.md) |
| Understand the scoring &amp; stopping | [Stop criteria](docs/scoring-and-stopping/stop-criteria.md), [Confidence](docs/scoring-and-stopping/confidence.md) |
| Debug a run | [Debugging runs](docs/observability/debugging-runs.md), [Troubleshooting](docs/reference/troubleshooting.md) |
| Contribute | [Contributing](docs/development/contributing.md), [Coding standards](docs/development/coding-standards.md) |

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>

## Project &amp; license

Copyright (c) 2026 Babak Zandi. Licensed under the [GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`); see [LICENSE](LICENSE) for the full text, warranty disclaimer, and limitation of liability. See [NOTICE](NOTICE) for the attribution and source notice.

```text
Inqtrix - Copyright (c) 2026 Babak Zandi - https://github.com/BZandi/inqtrix
```

Contributions are welcome. Start with [Contributing](docs/development/contributing.md) and [Coding standards](docs/development/coding-standards.md).

<details>
<summary><b>Acknowledgments &amp; third-party notices</b></summary>

<p></p>

Built on open-source Python and React libraries. The complete generated dependency inventory is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), with a machine-readable view in [`THIRD_PARTY_NOTICES.json`](THIRD_PARTY_NOTICES.json). Major direct runtime libraries include FastAPI (MIT), Uvicorn (BSD-3-Clause), the OpenAI Python SDK (Apache-2.0), LangGraph (MIT), Pydantic (MIT), and cachetools (MIT).

**Third-party services &amp; output notice.** When configured to use external model, search, or API providers, this project may transmit prompts, context, and search queries to those services, governed by their respective terms and privacy policies. Operators are solely responsible for legal, security, and data-protection compliance. Generated outputs are informational only and do not constitute professional advice; independent verification remains the user's responsibility.

**AI disclosure.** Developed with assistance from [Claude Code](https://www.anthropic.com/) (Anthropic), [GitHub Copilot](https://github.com/features/copilot) (GitHub / Microsoft), and [ChatGPT](https://openai.com/chatgpt) (OpenAI). Provided for transparency; use remains subject to the AGPL-3.0-only terms.

</details>

<div align="right"><a href="#readme-top"><sub>back to top ↑</sub></a></div>
