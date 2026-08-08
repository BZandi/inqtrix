# Database Migrations

## Scope

This page defines the production contract for schema upgrades on bundled and
managed PostgreSQL. Inqtrix uses tenant row-level security (RLS), including
`FORCE ROW LEVEL SECURITY`, so a generic application login is not sufficient
for data-moving migrations. Managed/external production databases deliberately
separate the privileged migration credential from the runtime credential. The
bundled PostgreSQL compatibility path instead derives a direct migration DSN
from the same config/secrets pair and does not require a third file.

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

This two-login split is mandatory for managed/external production databases.
The official bundled PostgreSQL service retains its documented
`bundled_legacy` compatibility identity, but the one-shot migration still uses
a direct `postgres:5432` DSN and never the optional PgBouncer runtime target.

When set, `INQTRIX_MIGRATION_DATABASE_URL` supplies the migration-only
connection. Otherwise the migrator uses `INQTRIX_DATABASE_URL`; bundled
Compose/Helm installations deliberately provide a direct database value on
that path. A managed production database must always supply the separate
migration Secret.

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
| `owner` | Managed services that do not grant `BYPASSRLS` | Requires ownership and DDL/grant authority over every managed Inqtrix table, function and sequence. |

RLS mode selects migration authority; it never makes a live schema change
safe. Every actual revision change on an installed schema requires API,
worker, Collaboration and every connection pooler to be stopped, all database
client sessions to be drained, and
`INQTRIX_MIGRATION_SERVICES_QUIESCED=true`. A no-op invocation at the already
installed target does not require a maintenance window.

If the provider grants neither `BYPASSRLS` nor ownership/DDL authority over all
managed schema objects, schema-changing upgrades are not technically possible.
The migration preflight rejects that state before changing the schema.

## What the migration command guarantees

`inqtrix-migrate` is the internal command run by the orchestrator's one-shot
job. Its preflight verifies a direct PostgreSQL connection, current/session
roles, superuser/BYPASS properties, ownership of the revision table, tenant
tables, policy function and identity sequences, required DDL/grant rights and
the selected RLS mode.

Every installed-schema transition performs one atomic PostgreSQL transaction:

1. Verify the explicit quiescence assertion and require the migration
   connection to be the database's only client backend.
2. Set a transaction-local ten-second `lock_timeout`, then acquire the global
   Inqtrix transaction-level advisory lock and `ACCESS EXCLUSIVE` locks on
   `alembic_version` and every tenant table present at the source revision.
   The timeout bounds only lock acquisition; it is not a data-processing or
   migration deadline. A timeout rolls back the transaction and reports that
   no schema transition was published.
3. In owner mode only, change those owner-controlled tables to
   `NO FORCE ROW LEVEL SECURITY`.
   After each Alembic revision, newly created or recreated tenant tables are
   locked and changed the same way before the next revision runs. RLS remains
   enabled; `DISABLE ROW LEVEL SECURITY` is never used.
4. Run all Alembic revisions on that same injected connection and transaction.
5. Verify the target revision, managed-object owners, enabled/forced RLS state,
   tenant policies and exact runtime ACLs before commit.

Any failure rolls back schema changes, data backfills, Alembic revision and RLS
state together. `SET LOCAL row_security=off` is retained only as a fail-closed
detector for an unexpected tenant table. PostgreSQL holds explicit locks to the
end of the transaction. See
[explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html)
and [transaction advisory locks](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS).

## Normal orchestration

### Compose

Compose starts `migrate` automatically and starts API/worker only after that
one-shot service exits successfully. With bundled PostgreSQL, the job receives
a transparent direct `postgres:5432` DSN assembled from the same
config/secrets pair—even when API and worker use optional PgBouncer. There is
no third file in bundled mode.

For a managed/external database:

1. Copy `deploy/.env.migrate.secrets.example` to a named, uncommitted file such
   as `deploy/.env.migrate.secrets.azure` and set mode `0600`.
2. Put only the direct privileged `INQTRIX_MIGRATION_DATABASE_URL` in it.
3. Set the root-relative pointer and non-secret RLS mode in the selected stack
   config:

   ```dotenv
   INQTRIX_MIGRATION_ENV_FILE=deploy/.env.migrate.secrets.azure
   INQTRIX_MIGRATION_RLS_MODE=bypass
   ```

4. Choose exactly one update path:
   - For every RLS mode, use the raw-Compose maintenance sequence below:
     build first, stop ingress and every database client, run the one-shot
     migration with the quiescence assertion, verify head, then recreate the
     recorded workloads.
   - For external `owner`, the optional CLI can run
     `inqtrix-deploy ... owner-upgrade` instead of a
     preceding `compose up`. The command records the active workloads, builds
     the new migration and workload images while the old release is still
     serving, stops the active web ingress and database clients, runs the
     one-shot migration,
     and recreates the previously active API, worker, Collaboration and web
     services from the new images. A failure before the migration attempt
     restarts stopped old containers. Once the attempt begins, any non-zero or
     interrupted result leaves database clients stopped: a disconnected CLI
     cannot prove whether PostgreSQL committed, rolled back, or still has a
     live migration session. Verify the migration job and database revision
     before deliberately resuming or rolling back.

The following command is suitable for a fresh external bypass installation or
an already-at-head restart. It is not an installed-schema upgrade procedure:

```bash
inqtrix-deploy \
  --external-db \
  --migration-env-file deploy/.env.migrate.secrets.azure \
  --secrets-file deploy/.env.stack.secrets.azure \
  --env-file deploy/.env.stack.azure \
  up --detach --build
```

The exact raw Compose counterpart consumes the same pair and the config-side
migration pointer. On an installed schema it is used only after the manual
maintenance migration has reached head:

```bash
docker compose \
  -f deploy/compose/compose.stack.yaml \
  -f deploy/compose/compose.external-db.yaml \
  --env-file deploy/.env.stack.secrets.azure \
  --env-file deploy/.env.stack.azure \
  up -d --build
```

For owner mode, the guarded command additionally requires explicit project
confirmation:

```bash
inqtrix-deploy \
  --external-db \
  --migration-env-file deploy/.env.migrate.secrets.azure \
  --secrets-file deploy/.env.stack.secrets.azure \
  --env-file deploy/.env.stack.azure \
  owner-upgrade --confirm-project inqtrix
```

The canonical raw-Compose upgrade sequence is: record
`compose ps --services --filter status=running`; build the new `migrate` plus
those workload images; stop `web`, `api`, and every running worker,
Collaboration, and pooler; run exactly one
`compose run --rm --no-deps -e INQTRIX_MIGRATION_SERVICES_QUIESCED=true
migrate`; verify Alembic head; then force-recreate only the workloads recorded
at the start. The migrator additionally refuses to start while another
database client session remains. External/operator-managed poolers cannot be
scaled by Inqtrix and must be drained by the operator. If the migration command
is interrupted or its commit outcome is
unknown, keep all clients stopped. The CLI automates these safety conditions;
the raw sequence is intentionally not hidden in another shell script.

Do not start new workload containers before this sequence completes. Fresh
installs and already-at-head restarts retain the normal Compose dependency on
the one-shot `migrate` service.

The base Compose stack always gives the migration job a direct
`postgres:5432` DSN derived from the bundled config/secrets pair. This remains
true when the API and worker runtime DSN points to optional PgBouncer, so
bundled mode never needs a migration env file. Only the external-database
override requires `INQTRIX_MIGRATION_ENV_FILE`; that file supplies the
privileged direct external DSN and is mounted only into the one-shot job.

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
  maintenanceConfirmed: false
```

The Secret is injected only into the hook Job. Helm blocks the release when the
job fails. On every upgrade, `maintenanceConfirmed: true` authorizes a
chart-owned pre-upgrade hook to remove the API HPA, scale API, worker,
Collaboration and the chart-owned PgBouncer to zero, and wait until their pods
have terminated.
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

For every RLS mode, the production sequence is:

1. Back up PostgreSQL and verify the restore procedure.
2. Set `maintenanceConfirmed: true` to authorize the narrow maintenance
   ServiceAccount and quiesce hook.
3. Run the Helm upgrade; the hook scales database clients to zero and drains
   their pools before the migration job starts.
4. Require the hook postcondition and Alembic head check to pass.
5. Start the new workloads and wait for `/readyz` before admitting traffic.

If the migration hook fails, the chart deliberately leaves database clients
quiesced instead of starting an image whose schema contract is unknown. Correct
the migration authority or roll back, rerun the release, and only then restore
traffic. The successful Helm apply recreates the desired replicas and HPA.

An external/operator-managed pooler is outside the chart's RBAC scope. Drain it
before starting the Helm upgrade; the migration job verifies that no residual
database client session remains.

### Custom charts

A custom Kubernetes/OpenShift deployment must preserve the same dependency
graph: stop workloads for every installed-schema change, run exactly one direct
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
                 value: "true"          # required for installed-schema changes
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
3. **Schema-maintenance quiesce**, independent of RLS authority: before the
   Job runs, scale API, worker, Collaboration and owned poolers to zero and
   wait for pod termination (the supplied chart automates this in its
   schema-maintenance hook), set
   `INQTRIX_MIGRATION_SERVICES_QUIESCED=true`, and restore replicas only
   after the Job succeeded.

The PostgreSQL requirements are unchanged from the top of this page:
version 15+, no extensions (a pgvector-enabled image is fine, the extension
stays unused), and the two-role split.

## Data-moving integrity contracts

Legacy Knowledge metadata is never trusted to mint an asset identity. The
migration resolves `source_id`, `fileId` and `file_id` only against canonical
same-tenant asset/file rows. Every supplied hint must resolve to one asset and
the collection must contain only one document claiming it; dangling,
contradictory or duplicate claims remain stored but are quarantined and cannot
become active evidence.

Every reconciled Knowledge document also receives a server-owned
tenant/owner/workspace source binding. Canonical asset rows are authoritative;
one unambiguous retained source-lifecycle tombstone is the recovery source
after asset metadata has already been removed. A row whose scope cannot be
proved stays quarantined. This binding is what keeps aggregate cleanup local
when equal `source_id` values exist in different owner or workspace scopes.

Ledger relationships include `tenant_id` in their database foreign keys, not
only in row-level-security predicates. Upload, deletion, document-revision and
index-generation children therefore cannot reference a known parent identifier
from another tenant. Stored-byte migration validates both the aggregate quota
counter against the file registry and each non-tombstoned file stock row
against its canonical bound file before the transaction commits.

Lifecycle fencing, deletion receipts, immutable Knowledge history and release
integrity reconciliation cannot be represented truthfully by older schemas.
Their migrations reject schema downgrade instead of deleting or relabelling
state. Production rollback follows the established contract: stop new
binaries, restore the verified pre-upgrade database backup, then start the
matching old binaries.

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

Alembic `--sql`/offline rendering is intentionally unsupported. The migration
chain performs data-dependent reconciliation and postcondition checks that
cannot be represented truthfully without the target database. Use only the
managed online runner so locks, data changes, revision stamping and validation
share one transaction.

## Related docs

- [Kubernetes and OpenShift](kubernetes.md) — Helm values, Secrets and
  custom-chart parity.
- [Security hardening](security-hardening.md) — Runtime role and
  tenant-isolation requirements.
- [Runbooks](runbooks.md) — Deployment and incident-response procedures.
- [Settings and environment](../configuration/settings-and-env.md) — Complete
  variable reference.
