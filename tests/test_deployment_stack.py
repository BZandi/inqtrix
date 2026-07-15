"""Deployment contracts shared by the production Compose stack."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parent.parent


def test_compose_scopes_migration_environment_to_one_shot_job() -> None:
    """A privileged migration DSN must never enter API or worker env files."""
    stack = yaml.safe_load(
        (_ROOT / "deploy" / "compose" / "compose.stack.yaml").read_text(
            encoding="utf-8"
        )
    )
    services = stack["services"]
    migrate = services["migrate"]

    assert migrate["command"] == ["inqtrix-migrate"]
    assert migrate["env_file"] == [
        {
            "path": "${INQTRIX_MIGRATION_ENV_FILE:-/dev/null}",
            "required": True,
        }
    ]
    assert migrate["environment"]["INQTRIX_MIGRATION_RLS_MODE"] == (
        "${INQTRIX_MIGRATION_RLS_MODE:-auto}"
    )
    assert migrate["environment"]["INQTRIX_DATABASE_URL"] == (
        "${INQTRIX_DATABASE_URL:?set INQTRIX_DATABASE_URL in the Compose "
        "--env-file}"
    )
    assert "@postgres:5432" not in migrate["environment"][
        "INQTRIX_DATABASE_URL"
    ]
    assert services["api"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["worker"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )

    for service_name in ("api", "worker"):
        runtime_env = str(services[service_name].get("env_file", []))
        assert "INQTRIX_ENV_FILE" in runtime_env
        assert "INQTRIX_MIGRATION_ENV_FILE" not in runtime_env
        assert services[service_name]["environment"][
            "INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY"
        ] == "${INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY:-bundled_legacy}"
        assert services[service_name]["environment"][
            "INQTRIX_MIGRATION_DATABASE_URL"
        ] == ""
        mounts = services[service_name]["volumes"]
        assert any(
            mount.get("source") == "${INQTRIX_S3_CA_BUNDLE_HOST:-/dev/null}"
            and mount.get("target") == "/etc/inqtrix/object-store/ca.pem"
            for mount in mounts
            if isinstance(mount, dict)
        )

    healthcheck = " ".join(services["api"]["healthcheck"]["test"])
    assert "/readyz" in healthcheck
    assert "/health" not in healthcheck


def test_compose_external_db_override_detaches_bundled_postgres() -> None:
    """The external-database switch stays consistent across both files.

    Contract: (1) the default stack keeps the bundled postgres profile-free
    (it always starts), (2) every ``depends_on: postgres`` entry is marked
    ``required: false`` so the override can deactivate the service without
    failing dependents, and (3) the override moves ONLY the bundled
    database behind an inactive profile — migrate/api/worker keep running
    against the external DSN.
    """
    stack = yaml.safe_load(
        (_ROOT / "deploy" / "compose" / "compose.stack.yaml").read_text(
            encoding="utf-8"
        )
    )
    services = stack["services"]
    assert "profiles" not in services["postgres"]

    dependents = [
        name
        for name, service in services.items()
        if "postgres" in (service.get("depends_on") or {})
    ]
    assert set(dependents) == {"pgbouncer", "migrate", "api", "worker"}
    for name in dependents:
        entry = services[name]["depends_on"]["postgres"]
        assert entry["condition"] == "service_healthy"
        assert entry["required"] is False, (
            f"depends_on.postgres of {name!r} must be required: false so the "
            "external-db override can deactivate the bundled database"
        )

    override = yaml.safe_load(
        (_ROOT / "deploy" / "compose" / "compose.external-db.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(override["services"]) == {"postgres"}
    assert override["services"]["postgres"]["profiles"] == [
        "bundled-db-disabled"
    ]


def test_compose_web_ingress_requires_explicit_non_loopback_bind() -> None:
    """LAN exposure is configurable without weakening the safe default."""
    stack = yaml.safe_load(
        (_ROOT / "deploy" / "compose" / "compose.stack.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert stack["services"]["web"]["ports"] == [
        "${INQTRIX_WEB_BIND_ADDRESS:-127.0.0.1}:${INQTRIX_WEB_PORT:-8080}:8080"
    ]


def test_compose_owner_upgrade_builds_before_drain_and_rolls_forward() -> None:
    script = (
        _ROOT / "deploy" / "scripts" / "compose-owner-upgrade.sh"
    ).read_text(encoding="utf-8")

    build = script.index('compose build "$@"')
    stop = script.index('compose stop "$service"')
    safety_boundary = script.index("trap - EXIT HUP INT TERM", stop)
    migrate = script.index("compose run --rm --no-deps")
    roll_forward = script.index(
        'compose up -d --no-deps --force-recreate "$service"'
    )

    assert build < stop < safety_boundary < migrate < roll_forward
    assert "INQTRIX_MIGRATION_RLS_MODE=owner" in script
    assert "INQTRIX_MIGRATION_SERVICES_QUIESCED=true" in script
    assert "for service in api worker collaboration web" in script
    # Failures before the migration begins may restart the old containers.
    # Once an attempt starts, an ambiguous CLI error must leave them stopped.
    assert 'compose start "$service"' in script
    assert "database clients remain stopped" in script


def test_compose_owner_upgrade_signal_before_migration_cannot_fall_through(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "docker.trace"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "$TRACE_FILE"
case "$*" in
  *"ps --services --filter status=running"*)
    printf 'api\\nworker\\ncollaboration\\nweb\\n'
    ;;
  *"stop api")
    kill -TERM "$PPID"
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    stack_file = tmp_path / "compose.yaml"
    stack_file.write_text("services: {}\n", encoding="utf-8")
    stack_env = tmp_path / "stack.env"
    stack_env.write_text("# test\n", encoding="utf-8")
    script = _ROOT / "deploy" / "scripts" / "compose-owner-upgrade.sh"
    env = {
        **os.environ,
        "INQTRIX_COMPOSE_FILE": str(stack_file),
        "INQTRIX_STACK_ENV_FILE": str(stack_env),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TRACE_FILE": str(trace),
    }

    completed = subprocess.run(
        ["/bin/sh", str(script)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    calls = trace.read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert "interrupted before migration" in completed.stderr
    assert "run --rm --no-deps" not in calls
    assert "start api" in calls
    assert "start worker" in calls
    assert "start collaboration" in calls
