#!/bin/sh
# Run a portable owner-mode migration without allowing old application
# processes to overlap the schema/RLS transaction.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
STACK_FILE=${INQTRIX_COMPOSE_FILE:-$ROOT_DIR/deploy/compose/compose.stack.yaml}
STACK_ENV_FILE=${INQTRIX_STACK_ENV_FILE:-$ROOT_DIR/deploy/.env.stack}

compose() {
    docker compose -f "$STACK_FILE" --env-file "$STACK_ENV_FILE" "$@"
}

running_services=$(compose ps --services --filter status=running)

is_running() {
    printf '%s\n' "$running_services" | grep -qx "$1"
}

restore_database_clients() {
    for service in api worker collaboration; do
        if is_running "$service"; then
            # Restart the stopped container itself. A failed migration must
            # never recreate an old-schema workload from the newly built image.
            compose start "$service"
        fi
    done
}

abort_before_migration() {
    # POSIX shells resume after a trapped signal unless the handler exits.
    # Clear every trap first so restoration runs exactly once and no signal can
    # fall through into the owner migration with workloads back online.
    trap - EXIT HUP INT TERM
    restore_database_clients
    echo "owner upgrade interrupted before migration; old database clients restored" >&2
    exit 130
}

set -- migrate
for service in api worker collaboration web; do
    if is_running "$service"; then
        set -- "$@" "$service"
    fi
done

# Building is deliberately completed before the maintenance boundary. It is
# safe while the old stack is live and guarantees that both the one-shot job
# and every previously active workload use the checked-out release afterward.
compose build "$@"

trap restore_database_clients EXIT
trap abort_before_migration HUP INT TERM
for service in api worker collaboration; do
    if is_running "$service"; then
        compose stop "$service"
    fi
done

# From this point onward a non-zero CLI result cannot prove that PostgreSQL
# rolled back: the client may have disconnected after commit or while the job
# is still running. Keep every database client quiesced until an operator has
# verified the revision instead of guessing and restarting old workloads.
trap - EXIT HUP INT TERM
if ! compose run --rm --no-deps \
    -e INQTRIX_MIGRATION_RLS_MODE=owner \
    -e INQTRIX_MIGRATION_SERVICES_QUIESCED=true \
    migrate; then
    echo "owner migration outcome is not verified; database clients remain stopped" >&2
    exit 1
fi

# Never route this through the migrate dependency: that would run a second
# migration and weaken the single, audited owner-maintenance boundary.
for service in api worker collaboration web; do
    if is_running "$service"; then
        compose up -d --no-deps --force-recreate "$service"
    fi
done
