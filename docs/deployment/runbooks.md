# Runbooks — operate the Stack-mode deployment

> Files: `deploy/compose/compose.stack.yaml`

## Scope

Day-to-day lifecycle commands for the Stack mode compose stack: start, stop, restart, update, backup, restore, reset. Set up the stack first with [Stack quickstart](../getting-started/stack-quickstart.md).

All commands assume the compose file path in a shell variable:

```bash
# Both flags every time: -f selects the file, --env-file supplies the values
# Compose needs while reading it (Postgres password, ports, the required-var
# guards). Compose does not auto-load a file named .env.stack.
CF="-f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack"
```

Volumes are project-prefixed because the compose project is named `inqtrix`: `inqtrix_pgdata`, `inqtrix_objectstore`, `inqtrix_qdrant_storage`, `inqtrix_valkey_data`, `inqtrix_seaweedfs_data` (and `inqtrix_dex_data` with the `oidc` profile).

## Start

```bash
# Default stack: postgres + migrate + api + web
docker compose $CF up -d --build

# With optional profiles (mix as needed)
docker compose $CF --profile knowledge --profile workers up -d --build

# Live editor collaboration (requires matching env settings and cookie auth)
docker compose $CF --profile collaboration up -d --build

# A specific setup: point --env-file at its own file instead
docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/env/azure.env up -d
```

## Status & logs

```bash
docker compose $CF ps                 # wait for healthy
docker compose $CF logs -f api        # follow the API (startup config errors land here)
docker compose $CF logs migrate       # one-shot migration output
docker compose $CF logs -f collaboration # fencing/readiness/persistence status
```

## Stop / restart

```bash
docker compose $CF down               # stop, KEEP volumes/data
docker compose $CF restart api web    # restart without rebuilding
docker compose $CF restart api collaboration web # collaboration-enabled stack
```

## Update

Pull the new code, then choose the update path that matches the configured
migration RLS mode. Bundled/`auto` and dedicated `bypass` installations use
normal Compose orchestration:

```bash
git pull
docker compose $CF up -d --build

# Preserve active profiles in the same orchestrated update
docker compose $CF --profile collaboration up -d --build
```

Managed PostgreSQL `owner` mode uses the maintenance wrapper instead; do not
run `compose up` first:

```bash
git pull
deploy/scripts/compose-owner-upgrade.sh
```

The wrapper builds the new images before draining active database clients,
runs the one-shot migration, then rolls the previously active API, worker,
Collaboration and web services to those images. A failure before the migration
attempt restores old containers; an interrupted or failed attempt leaves all
database clients stopped because its commit state is not safe to infer from a
CLI exit code. Verify the job and database revision before recovery. Set
`INQTRIX_COMPOSE_FILE` and `INQTRIX_STACK_ENV_FILE` when the Compose file or
stack env lives outside the repository defaults.

Do not run `inqtrix-migrate` manually during a normal update. Managed
PostgreSQL uses a separate migration env file and explicit RLS mode. See
[Database migrations](database-migrations.md). A manual command is break-glass
only after a verified backup and full workload drain.

## Local development (changing the code)

The Stack-mode images serve a **built** bundle — editing source does **not** hot-reload. After a code change, rebuild the affected image:

```bash
docker compose $CF up -d --build web    # after a frontend change
docker compose $CF up -d --build api    # after a backend change
```

For active development with **hot reload**, do not use the Stack-mode images. Run the frontend dev server (and optionally the backend) on the host:

```bash
# 1. Backing services + API only (Postgres + migrate + API, no web container):
docker compose $CF up -d postgres migrate api

# 2. Frontend with hot reload (Vite, http://127.0.0.1:5173, proxies /api /v1 /health to the API):
pnpm run ui:dev
```

Edit files under `apps/research-desk/` and the browser updates instantly. To also iterate on the **backend** with a fast restart, run it on the host instead of the `api` container (`uv run python -m inqtrix`) against the same Postgres — see the manual/host setup in [Platform components](../getting-started/platform-components.md#manual--host-platform-framework-mode) and the UI paths in [First research run](../getting-started/first-research-run.md). Point the Vite proxy at a non-default backend with `VITE_INQTRIX_API_BASE_URL`.

## Backup

Capture **both** the Postgres database and the object-store volume - uploaded blobs live outside Postgres. PostgreSQL already contains every collaboration update, snapshot, projection, lease, and decision row; the Node service has no separate volume to back up.

```bash
# Postgres dump
docker compose $CF exec -T postgres pg_dump -U inqtrix inqtrix > backup-$(date +%F).sql

# Object-store volume (local backend)
docker run --rm -v inqtrix_objectstore:/data -v "$PWD:/out" \
  docker.io/library/busybox tar czf /out/objectstore-$(date +%F).tgz -C /data .
```

If you run `--profile knowledge`, also snapshot the Qdrant volume (`inqtrix_qdrant_storage`) the same way; re-ingestion can rebuild it, so it is optional.

## Restore

For a collaboration-enabled deployment, stop the API and Node writer before
restore. Restore a matching application/schema version, then start FastAPI
before exactly one collaboration instance. Node reconstructs each room from
the verified snapshot plus update tail; never reconstruct it from Markdown
alone. See [Deploy editor collaboration](editor-collaboration.md#backup-and-restore).

```bash
# Postgres (into a fresh, migrated database)
cat backup-YYYY-MM-DD.sql | docker compose $CF exec -T postgres psql -U inqtrix inqtrix

# Object-store volume
docker run --rm -v inqtrix_objectstore:/data -v "$PWD:/in" \
  docker.io/library/busybox sh -c "cd /data && tar xzf /in/objectstore-YYYY-MM-DD.tgz"
```

## Reset (destroy all data)

```bash
docker compose $CF down -v            # removes containers AND named volumes
```

Re-running `up -d --build` then re-migrates a clean database.

## Collaboration kill switch

Set `INQTRIX_COLLABORATION_ENABLED=false`, restart FastAPI, and then stop the
collaboration service. Legacy Markdown editing continues. Existing
collaboration documents retain their binary PostgreSQL state and expose only
the last saved Markdown projection read-only; the switch does not convert or
delete them.

## Related docs

- [Stack quickstart](../getting-started/stack-quickstart.md)
- [Platform components](../getting-started/platform-components.md)
- [Security hardening](security-hardening.md)
- [Editor collaboration](editor-collaboration.md)
