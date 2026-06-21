# Deployment modes

> Files: `deploy/compose/compose.stack.yaml`, `deploy/compose/compose.dev.yaml`, `src/inqtrix/settings.py`

## Scope

Inqtrix has one codebase and several ways to run it. This page is the decision matrix: pick the mode that matches who you are and what you need, then follow the linked page. Components and their feature unlocks are detailed in [Platform components](../getting-started/platform-components.md).

## The matrix

| Mode | Who it's for | Auth | Infrastructure | Start here |
|---|---|---|---|---|
| **Local demo** | Trying it out, presentations | none | none (in-memory) | [First research run](../getting-started/first-research-run.md) (UI demo mode) |
| **Env server** | A quick HTTP service, single user | none / apikey | none (in-memory) | [First research run](../getting-started/first-research-run.md) |
| **Stack mode** | Self-hosted stack, one command | local / oidc / ldap | Postgres (+ optional profiles) | [Stack quickstart](../getting-started/stack-quickstart.md) |
| **Library** | Embedding Inqtrix in Python | n/a | your choice | [Library mode](library-mode.md) |
| **Custom provider** | Your own LLM/search/storage backend | n/a | your choice | [Writing a custom provider](../providers/writing-a-custom-provider.md) |
| **Enterprise OIDC** | SSO via your IdP | oidc | Postgres + IdP | [Auth modes](auth-modes.md) |
| **Scaled workers** | Many concurrent / restart-surviving runs | any | Postgres + Valkey + workers | [Platform components](../getting-started/platform-components.md) |
| **Kubernetes / OpenShift** | Enterprise clusters | local / oidc / ldap / apikey / none | Postgres (+ optional Qdrant/Valkey, or bundled for a demo) | [Kubernetes and OpenShift](kubernetes.md) |

## How to choose

- **Just looking?** Local demo or the env server — no database, no containers.
- **Running it for a team, want it to "just work"?** Stack mode — one command, Postgres for durable data, a built-in UI. Add the `knowledge`/`workers`/`s3` profiles as needs grow.
- **Building software on top of Inqtrix?** Library mode (in-process) or the HTTP server with the native `/v1/runs` API.
- **Bringing your own backend or identity provider?** Inqtrix is a Baukasten — every provider, storage backend, and auth mode is swappable; see the custom-provider guide and [Auth modes](auth-modes.md).

## Picking a provider stack

Independently of the deployment mode, the LLM and search providers are chosen via two env axes (`INQTRIX_LLM_PROVIDER` × `INQTRIX_SEARCH_PROVIDER`). Copy-paste `.env` recipes for every combination are in [Provider recipes](../getting-started/provider-recipes.md).

## Related docs

- [Stack quickstart](../getting-started/stack-quickstart.md)
- [Kubernetes and OpenShift](kubernetes.md)
- [Platform components](../getting-started/platform-components.md)
- [Provider recipes](../getting-started/provider-recipes.md)
- [Auth modes](auth-modes.md)
- [Runbooks](runbooks.md)
