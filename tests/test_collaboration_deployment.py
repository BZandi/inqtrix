"""Static contracts for the optional collaboration deployment surfaces."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent


def _node_collaboration_env_names() -> set[str]:
    """Return environment names consumed by the Node settings parser."""
    source = (
        _ROOT / "apps" / "collaboration-server" / "src" / "config.ts"
    ).read_text(encoding="utf-8")
    return set(
        re.findall(
            r"INQTRIX_(?:API_INTERNAL_URL|COLLABORATION_[A-Z0-9_]+)",
            source,
        )
    )


def test_collaboration_image_uses_frozen_nonroot_node_runtime() -> None:
    """The service builds reproducibly and the final image runs without root."""
    dockerfile = (
        _ROOT / "deploy" / "docker" / "Dockerfile.collaboration"
    ).read_text(encoding="utf-8")
    assert dockerfile.count("FROM node:22-bookworm-slim") == 2
    assert "corepack pnpm install --frozen-lockfile" in dockerfile
    assert "corepack pnpm --filter @inqtrix/collaboration-server build" in dockerfile

    runtime = dockerfile.split("FROM node:22-bookworm-slim AS runtime", 1)[1]
    assert "RUN chgrp -R 0 /app && chmod -R g=u /app" in runtime
    assert "USER 1001" in runtime
    assert 'CMD ["node", "--enable-source-maps", "dist/main.cjs"]' in runtime
    assert "pnpm install" not in runtime


def test_collaboration_production_artifact_is_consistently_commonjs() -> None:
    """Build, package, Docker, and context rules agree on dist/main.cjs."""
    service = _ROOT / "apps" / "collaboration-server"
    package = json.loads((service / "package.json").read_text(encoding="utf-8"))
    build = (service / "build.mjs").read_text(encoding="utf-8")
    dockerfile = (
        _ROOT / "deploy" / "docker" / "Dockerfile.collaboration"
    ).read_text(encoding="utf-8")
    dockerignore = (_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert package["scripts"]["build"] == "node build.mjs"
    assert package["scripts"]["pretest"] == "node build.mjs"
    assert package["scripts"]["start"] == "node dist/main.cjs"
    assert "format: 'cjs'" in build
    assert "outfile: 'dist/main.cjs'" in build
    assert 'CMD ["node", "--enable-source-maps", "dist/main.cjs"]' in dockerfile
    assert "apps/collaboration-server/dist" in dockerignore
    assert "dist/main.js" not in "\n".join((build, dockerfile, json.dumps(package)))


def test_compose_collaboration_profile_is_private_and_credential_scoped() -> None:
    """The optional Node service has no public or persistence/DB surface."""
    compose = yaml.safe_load(
        (_ROOT / "deploy" / "compose" / "compose.stack.yaml").read_text(
            encoding="utf-8"
        )
    )
    services = compose["services"]
    collaboration = services["collaboration"]

    assert collaboration["profiles"] == ["collaboration"]
    assert collaboration["read_only"] is True
    assert collaboration["user"] == "1001:0"
    assert collaboration["cap_drop"] == ["ALL"]
    assert "ports" not in collaboration
    assert "volumes" not in collaboration
    assert "env_file" not in collaboration

    env = collaboration["environment"]
    assert env["INQTRIX_API_INTERNAL_URL"] == "http://api:5100"
    assert env["INQTRIX_COLLABORATION_SECRET"] == (
        "${INQTRIX_COLLABORATION_SECRET:-}"
    )
    assert env["INQTRIX_COLLABORATION_TENANT_ID"] == (
        "${INQTRIX_COLLABORATION_TENANT_ID:-default}"
    )
    assert env["INQTRIX_COLLABORATION_MAINTENANCE_INTERVAL_SECONDS"] == (
        "${INQTRIX_COLLABORATION_MAINTENANCE_INTERVAL_SECONDS:-60}"
    )
    assert env["INQTRIX_COLLABORATION_RECONCILE_MAX_HASHES"] == (
        "${INQTRIX_COLLABORATION_RECONCILE_MAX_HASHES:-256}"
    )
    assert env["INQTRIX_COLLABORATION_RECONCILE_RATE_COUNT"] == (
        "${INQTRIX_COLLABORATION_RECONCILE_RATE_COUNT:-10}"
    )
    assert env["INQTRIX_COLLABORATION_RECONCILE_RATE_WINDOW_SECONDS"] == (
        "${INQTRIX_COLLABORATION_RECONCILE_RATE_WINDOW_SECONDS:-10}"
    )
    assert env["INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES"] == (
        "${INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES:-32}"
    )
    assert env["INQTRIX_COLLABORATION_MAX_QUEUED_BYTES"] == (
        "${INQTRIX_COLLABORATION_MAX_QUEUED_BYTES:-8388608}"
    )
    assert env["INQTRIX_COLLABORATION_SOCKET_BACKPRESSURE_BYTES"] == (
        "${INQTRIX_COLLABORATION_SOCKET_BACKPRESSURE_BYTES:-4194304}"
    )
    assert env["INQTRIX_COLLABORATION_SNAPSHOT_RETRY_BASE_MS"] == (
        "${INQTRIX_COLLABORATION_SNAPSHOT_RETRY_BASE_MS:-1000}"
    )
    assert env["INQTRIX_COLLABORATION_SNAPSHOT_RETRY_MAX_MS"] == (
        "${INQTRIX_COLLABORATION_SNAPSHOT_RETRY_MAX_MS:-30000}"
    )
    assert set(env) == _node_collaboration_env_names()
    assert not any(
        marker in name
        for name in env
        for marker in ("DATABASE", "POSTGRES", "PG_", "S3", "VALKEY")
    )

    # API and worker load the SAME shared env file. When the operator turns
    # INQTRIX_COLLABORATION_ENABLED on there, EVERY process constructing
    # Settings needs coherent collaboration URLs — a worker without them
    # crash-loops on the settings validator and queued runs are never
    # claimed (live incident 2026-07-15).
    for service_name in ("api", "worker"):
        service_env = services[service_name]["environment"]
        assert service_env["INQTRIX_COLLABORATION_HTTP_URL"] == (
            "http://collaboration:1234"
        ), f"{service_name} must carry the private collaboration HTTP URL"
        assert service_env["INQTRIX_COLLABORATION_WS_URL"] == (
            "ws://collaboration:1234/collaboration"
        ), f"{service_name} must carry the private collaboration WS URL"
    assert "collaboration" not in services["api"]["depends_on"]
    assert "collaboration" not in services["web"]["depends_on"]
    assert services["web"]["environment"]["INQTRIX_EXTERNAL_SCHEME"] == (
        "${INQTRIX_EXTERNAL_SCHEME:-http}"
    )


def test_nginx_collaboration_websocket_targets_fastapi_only() -> None:
    """The browser upgrade uses sanitized external forwarding only at FastAPI."""
    config = (
        _ROOT / "deploy" / "nginx" / "inqtrix.conf.template"
    ).read_text(encoding="utf-8")
    dockerfile = (
        _ROOT / "deploy" / "docker" / "Dockerfile.web"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"location = /collaboration \{(?P<body>.*?)\n    \}",
        config,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "proxy_pass http://inqtrix_api;" in body
    assert "proxy_set_header Upgrade $http_upgrade;" in body
    assert 'proxy_set_header Connection "upgrade";' in body
    assert "proxy_set_header Host $http_host;" in body
    assert "proxy_set_header Origin $http_origin;" in body
    assert (
        "proxy_set_header X-Forwarded-Proto ${INQTRIX_EXTERNAL_SCHEME};"
        in body
    )
    assert "proxy_set_header X-Forwarded-Host $http_host;" in body
    assert "$http_x_forwarded_proto" not in config
    assert "ENV INQTRIX_EXTERNAL_SCHEME=http" in dockerfile
    assert "collaboration:1234" not in body


def test_instance_probe_uses_the_same_public_fastapi_proxy_path() -> None:
    """nginx and Vite route the release probe to FastAPI, never to Node/SPA."""
    config = (
        _ROOT / "deploy" / "nginx" / "inqtrix.conf.template"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"location = /collaboration/instance \{(?P<body>.*?)\n    \}",
        config,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "proxy_pass http://inqtrix_api;" in body
    assert "proxy_set_header Host $http_host;" in body
    assert (
        "proxy_set_header X-Forwarded-Proto ${INQTRIX_EXTERNAL_SCHEME};"
        in body
    )
    assert "proxy_set_header X-Forwarded-Host $http_host;" in body
    assert "collaboration:1234" not in body

    vite = (
        _ROOT / "apps" / "research-desk" / "vite.config.ts"
    ).read_text(encoding="utf-8")
    vite_match = re.search(
        r"'/collaboration': \{(?P<body>.*?)\n\s*\}",
        vite,
        flags=re.DOTALL,
    )
    assert vite_match is not None
    assert "target: apiProxyTarget" in vite_match.group("body")
    assert "ws: true" in vite_match.group("body")
