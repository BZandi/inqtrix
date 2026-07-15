# Database Migrations

## Scope

This page defines the production contract for schema upgrades on bundled and
managed PostgreSQL. Inqtrix uses tenant row-level security (RLS), including
`FORCE ROW LEVEL SECURITY`, so a generic application login is not sufficient
for data-moving migrations. The migration credential and the runtime
credential are deliberately separate.

Inqtrix schema revision 0049 requires PostgreSQL 15 or newer. The migration
preflight checks the server version before taking locks or changing schema
state and fails with an operator-facing diagnostic on older releases.

Inqtrix uses PostgreSQL relationally only (RLS-protected tables); no
PostgreSQL extension is required — in particular not pgvector. Vector search
lives in Qdrant (`INQTRIX_VECTOR_BACKEND=qdrant`) or the in-memory store. A
pgvector-flavoured image such as pgvector 0.5.1 on Postgres 15 is PostgreSQL
15 and therefore satisfies the requirement; the extension simply remains
unused.

## The two-role rule

Use two database identities in production:

| Identity | Where it is available | Required properties |
|---|---|---|
| Runtime login | API and worker only | Can `SET ROLE inqtrix_app`; the effective `inqtrix_app` role is `NOLOGIN NOSUPERUSER NOBYPASSRLS`, has schema `USAGE` without database/schema `CREATE`, cannot own, inherit or assume an object-owner/BYPASS role, and is always tenant-scoped. |
| Migration login | One-shot migration job only | A direct PostgreSQL connection and either the `bypass` or `owner` authority described below. Never expose it through API/worker env, Collaboration, PgBouncer transaction pooling, or the browser. |

`INQTRIX_MIGRATION_DATABASE_URL` supplies the migration-only connection. It
falls back to `INQTRIX_DATABASE_URL` for backwards compatibility with bundled
Compose/Helm installations, but a managed production database should always
use a separate Secret.

`INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY=restricted` is the managed-production
default. A representative role split, executed by the database administrator,
is:

```sql
CREATE ROLE inqtrix_app NOLOGIN NOSUPERUSER NOBYPASSRLS
  NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE inqtrix_runtime LOGIN NOINHERIT NOSUPERUSER NOBYPASSRLS
  NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD '<managed-secret>';
GRANT inqtrix_app TO inqtrix_runtime;
REVOKE CREATE ON DATABASE inqtrix FROM inqtrix_runtime;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO inqtrix_app;
```

The migration identity owns every Inqtrix schema object that migrations alter
(`alembic_version`, application tables, tenant-policy functions and identity
sequences), or has `BYPASSRLS` plus equivalent object authority. It is also
allowed to grant/use `inqtrix_app`; the runtime login is never a member of that
migration identity. Use a dedicated database before revoking `public` schema
creation if other applications currently share it. The migrations grant the
exact table DML contract, explicit tenant-policy function execution,
`SELECT`-only revision visibility and `USAGE`-only identity-sequence access to
`inqtrix_app`. They never grant table `TRUNCATE`, `REFERENCES`, `TRIGGER`,
`MAINTAIN`, grant options or schema `CREATE`; `audit_log` remains append-only.
PUBLIC table/column grants and column ACLs for the canonical app role are not
accepted. Explicit named reporting/backup roles remain database-operator policy;
their column-limited read grants do not make application readiness fail. A
configured custom/empty `INQTRIX_DATABASE_APP_ROLE` remains supported, but its
effective role must expose the same table-level contract. Do not make
`inqtrix_app` a login, table owner, database owner, database `CREATE` grantee, or
member of a privileged role.
`bundled_legacy` exists only for the historical owner login created by the
official bundled Postgres image; the runtime preflight verifies that exact
shape and does not treat it as a general compatibility switch.

PostgreSQL documents that superusers and roles with `BYPASSRLS` always bypass
RLS, while a table owner is subject to policies when `FORCE ROW LEVEL SECURITY`
is active. It also documents that `row_security=off` raises an error rather
than bypassing a policy. Inqtrix therefore never treats `row_security=off` or a
special tenant value as migration authority. See [PostgreSQL row security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
and [`ALTER TABLE ... NO FORCE ROW LEVEL SECURITY`](https://www.postgresql.org/docs/current/sql-altertable.html).

## RLS modes

Set `INQTRIX_MIGRATION_RLS_MODE` to one of:

| Mode | Intended deployment | Contract |
|---|---|---|
| `auto` | Existing bundled/dev installations | Accepts only a superuser or `BYPASSRLS` login and emits a visible compatibility warning. It never silently enters owner maintenance. |
| `bypass` | Preferred when the managed service permits it | Requires a dedicated `NOSUPERUSER BYPASSRLS` migration login. Runtime workloads still switch to the restricted `inqtrix_app` role. |
| `owner` | Managed services that do not grant `BYPASSRLS` | Requires ownership and DDL/grant authority over every managed Inqtrix table, function and sequence. For an existing schema, API, worker and Collaboration must be stopped and `INQTRIX_MIGRATION_SERVICES_QUIESCED=true` must explicitly confirm that maintenance boundary. |

If the provider grants neither `BYPASSRLS` nor ownership/DDL authority over all
managed schema objects, schema-changing upgrades are not technically possible.
The migration preflight rejects that state before changing the schema.

## What the migration command guarantees

`inqtrix-migrate` is the internal command run by the orchestrator's one-shot
job. Its preflight verifies a direct PostgreSQL connection, current/session
roles, superuser/BYPASS properties, ownership of the revision table, tenant
tables, policy function and identity sequences, required DDL/grant rights and
the selected RLS mode.

Owner mode then performs one atomic PostgreSQL transaction:

1. Acquire the global Inqtrix transaction-level advisory lock.
2. Acquire `ACCESS EXCLUSIVE NOWAIT` locks on `alembic_version` and every
   tenant table present at the source revision. A live workload or competing
   migrator makes the job fail immediately instead of racing an upgrade.
3. Change only those owner-controlled tables to `NO FORCE ROW LEVEL SECURITY`.
   After each Alembic revision, newly created or recreated tenant tables are
   locked and changed the same way before the next revision runs. RLS remains
   enabled; `DISABLE ROW LEVEL SECURITY` is never used.
4. Run all Alembic revisions on that same injected connection and transaction.
5. Verify the target revision, managed-object owners, enabled/forced RLS state,
   tenant policies and exact runtime ACLs before commit.

Any failure rolls back schema changes, data backfills, Alembic revision and RLS
state together. `SET LOCAL row_security=off` is retained only as a fail-closed
detector for an unexpected tenant table. PostgreSQL holds explicit locks to the
end of the transaction; `NOWAIT` avoids an unbounded maintenance wait. See
[explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html)
and [transaction advisory locks](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS).

## Normal orchestration

### Compose

Compose starts `migrate` automatically and starts API/worker only after that
one-shot service exits successfully. The job does not load the shared runtime
env file. To use a managed database:

1. Copy `deploy/.env.migrate.example` to an uncommitted migration env file.
2. Put only `INQTRIX_MIGRATION_DATABASE_URL` in it.
3. Set `INQTRIX_MIGRATION_ENV_FILE` and the non-secret RLS mode in the shared
   stack env.
4. Choose exactly one update path:
   - For bundled/`auto` or `bypass`, start or update the stack normally with
     `docker compose ... up -d --build`.
   - For `owner`, run `deploy/scripts/compose-owner-upgrade.sh` instead of a
     preceding `compose up`. The wrapper records the active workloads, builds
     the new migration and workload images while the old release is still
     serving, stops the active database clients, runs the one-shot migration,
     and recreates the previously active API, worker, Collaboration and web
     services from the new images. A failure before the migration attempt
     restarts stopped old containers. Once the attempt begins, any non-zero or
     interrupted result leaves database clients stopped: a disconnected CLI
     cannot prove whether PostgreSQL committed, rolled back, or still has a
     live migration session. Verify the migration job and database revision
     before deliberately resuming or rolling back.

The owner wrapper defaults to the repository Compose file and
`deploy/.env.stack`. Non-default locations are explicit operator inputs:

```bash
INQTRIX_COMPOSE_FILE=/srv/inqtrix/compose.stack.yaml \
INQTRIX_STACK_ENV_FILE=/srv/inqtrix/inqtrix.stack.env \
  deploy/scripts/compose-owner-upgrade.sh
```

The wrapper updates only workloads that were running when it started. It does
not update optional infrastructure images such as PostgreSQL, Valkey or
SeaweedFS.

Do not run a separate manual migration during a normal Compose update.

When no migration env file is configured, Compose passes the runtime
`INQTRIX_DATABASE_URL` into the migration job as the compatibility fallback;
it never substitutes a different bundled database. If the runtime URL points
to PgBouncer, a migration env file with a direct PostgreSQL URL is mandatory.

To run the Compose stack against an external PostgreSQL, add the
`deploy/compose/compose.external-db.yaml` override (it deactivates the
bundled database container) and follow the "External PostgreSQL" block in
`deploy/.env.stack.example`: external runtime DSN with a dedicated
unprivileged login, `INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY=restricted`, a
migration env file, and an explicit `INQTRIX_MIGRATION_RLS_MODE`. Swapping
the bundled service's image in place (Postgres 18 to a Postgres 15 based
image) works only with a FRESH data volume: an existing Postgres 18 data
directory is not downgrade-compatible and the images mount different data
paths — attaching an external database is the supported route.

### Helm and OpenShift

Configure:

```yaml
migrations:
  databaseSecret:
    name: inqtrix-migration-database
    key: INQTRIX_MIGRATION_DATABASE_URL
  rlsMode: bypass
  ownerMaintenanceConfirmed: false
```

The Secret is injected only into the hook Job. Helm blocks the release when the
job fails. On an owner-mode upgrade, `ownerMaintenanceConfirmed: true`
authorizes a chart-owned pre-upgrade hook to remove the API HPA, scale API,
worker and Collaboration to zero, and wait until their pods have terminated.
The hook revokes its own temporary RoleBinding before exit. The migration hook
runs only after that bounded quiesce job succeeds; after migration, the normal
release apply writes a positive API `minReplicas` once to reactivate an HPA from
Kubernetes' zero-replica maintenance state.

`migrations.databaseSecret.name` must not be the runtime application Secret.
The chart rejects that collision. It also rejects
`secret.data.INQTRIX_MIGRATION_DATABASE_URL`: the chart-managed runtime Secret
is not an alternative location for a privileged migration credential. The
workload templates defensively blank
`INQTRIX_MIGRATION_DATABASE_URL` in API/worker even when an externally managed
runtime Secret contains the key by mistake.

For owner mode, the production sequence is:

1. Back up PostgreSQL and verify the restore procedure.
2. Set `ownerMaintenanceConfirmed: true` to authorize the narrow maintenance
   ServiceAccount and quiesce hook.
3. Run the Helm upgrade; the hook scales database clients to zero and drains
   their pools before the migration job starts.
4. Require the hook postcondition and Alembic head check to pass.
5. Start the new workloads and wait for `/readyz` before admitting traffic.

If the migration hook fails, the chart deliberately leaves database clients
quiesced instead of starting an image whose schema contract is unknown. Correct
the migration authority or roll back, rerun the release, and only then restore
traffic. The successful Helm apply recreates the desired replicas and HPA.

### Custom charts

A custom Kubernetes/OpenShift deployment must preserve the same dependency
graph: stop workloads when owner mode requires it, run exactly one direct
migration Job, wait for successful completion, then start the new API and
workers. Do not use a post-start hook or allow new images to serve against an
old schema. Copying only the Deployment objects from the supplied chart omits
this safety boundary.

Concretely, a bring-your-own-manifests deployment needs three mechanics:

1. **A one-shot migration Job** running before API/worker. Skeleton (mirror
   of the chart's `job-migrate.yaml`):

   ```yaml
   apiVersion: batch/v1
   kind: Job
   spec:
     backoffLimit: 2
     template:
       spec:
         restartPolicy: Never
         containers:
           - name: migrate
             image: <inqtrix-api-image>
             command: ["inqtrix-migrate"]
             env:
               - name: INQTRIX_MIGRATION_RLS_MODE
                 value: "bypass"        # or "owner", see the mode table
               - name: INQTRIX_MIGRATION_SERVICES_QUIESCED
                 value: "false"         # "true" only for owner-mode upgrades
               - name: INQTRIX_MIGRATION_DATABASE_URL
                 valueFrom:
                   secretKeyRef:
                     name: inqtrix-migration-database
                     key: INQTRIX_MIGRATION_DATABASE_URL
   ```

   Ordering is the operator's tooling choice — a Helm hook
   (`pre-install,pre-upgrade`, how the supplied chart orders it for external
   databases), an Argo CD sync wave, or a pipeline step — but the invariant
   is fixed: the Job must complete successfully before the new API/worker
   pods start. The supplied Job also waits for database reachability before
   invoking `inqtrix-migrate`; replicate that or ensure reachability
   externally.
2. **A separate migration Secret** that only the Job mounts. Never add the
   privileged DSN to the runtime Secret; defensively set
   `INQTRIX_MIGRATION_DATABASE_URL: ""` on API/worker containers so a
   mistakenly shared Secret cannot leak migration authority into runtime
   pods (the supplied chart does both).
3. **Owner-mode quiesce**, when the database offers no BYPASSRLS login:
   before the Job runs, scale API, worker and Collaboration to zero and wait
   for pod termination (the supplied chart automates this in
   `job-owner-maintenance.yaml`), set
   `INQTRIX_MIGRATION_SERVICES_QUIESCED=true`, and restore replicas only
   after the Job succeeded.

The PostgreSQL requirements are unchanged from the top of this page:
version 15+, no extensions (a pgvector-enabled image is fine, the extension
stays unused), and the two-role split.

## Readiness and incident diagnosis

Runtime readiness validates more than `SELECT 1`: the database must be at the
packaged Alembic head, every expected tenant table must have enabled and forced
RLS with the canonical fail-closed policy and exact least-privilege grants, and
a transaction must be able to switch to the restricted application role with a
tenant GUC. `alembic_version` must resolve in that active schema with explicit
`SELECT` only, the policy function must be explicitly executable only by the
application role, and both runtime identity sequences must have exactly
`USAGE`. The probe audits rights held directly and through every role the
effective role or restricted session login can actually assume (`MEMBER` on
PostgreSQL 15, the separate `SET` option on PostgreSQL 16+). Owned, inherited or
assumable ownership-role access, database/schema `CREATE`, missing dependencies,
PUBLIC grants and
excessive table/column rights such as `TRUNCATE`, `REFERENCES`, grant options or
non-append-only `audit_log` mutation fail readiness. Named reporting roles that
the runtime cannot use remain the database operator's boundary. The probe also
requires `row_security=on`, active RLS, a writable transaction, a successful
policy-function call and a harmless protected-table query. The catalog checks
use the active schema rather than assuming `public`. A
mismatch is `not_ready`; workers do not take new jobs. This turns an incomplete
rollout into one deployment error instead of unrelated 500 responses from runs,
preferences, skills and files.

The characteristic managed-PostgreSQL failure is SQLSTATE `28000`, often shown
as `InvalidAuthorizationSpecificationError` with:

```text
inqtrix.tenant_id is not set; refusing the tenant scoped query
```

That means a data migration touched a forced-RLS table without valid migration
authority. It is not fixed by setting a tenant environment variable. Check the
migration job's selected RLS mode and role properties, then rerun the normal
orchestrated rollout after correcting the Secret or maintenance boundary.

Manual `inqtrix-migrate` execution is a break-glass action only. Stop all
workloads, use the same migration-only environment as the job, capture the
preflight output and take a backup first.

## Related docs

- [Kubernetes and OpenShift](kubernetes.md) — Helm values, Secrets and
  custom-chart parity.
- [Security hardening](security-hardening.md) — Runtime role and
  tenant-isolation requirements.
- [Runbooks](runbooks.md) — Deployment and incident-response procedures.
- [Settings and environment](../configuration/settings-and-env.md) — Complete
  variable reference.
