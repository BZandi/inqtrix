"""Contract tests for the Kubernetes/OpenShift Helm chart (deploy/helm/inqtrix).

These render the chart with ``helm template`` and assert the invariants the
deployment relies on: OpenShift arbitrary-UID compatibility (no pinned UID, a
Route instead of an Ingress), bundled-service auto-wiring, the queue-needs-worker
gate, the fail-fast guard for a misconfigured database, and secret-change
rollout. They are skipped when the ``helm`` binary is not on PATH (e.g. the
offline CI image), exactly as the replay tests skip without cassettes -- the
chart logic is only verifiable against the real templating engine.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_CHART = Path(__file__).resolve().parent.parent / "deploy" / "helm" / "inqtrix"
_ROOT = _CHART.parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm binary not available"
)


def _template(*set_args: str, extra: list[str] | None = None) -> str:
    """Render the chart, raising on a helm error (so fail-guards are testable)."""
    cmd = ["helm", "template", "rel", str(_CHART)]
    for item in set_args:
        cmd += ["--set", item]
    if extra:
        cmd += extra
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


def _docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if isinstance(d, dict)]


def _by_kind(docs: list[dict], kind: str) -> list[dict]:
    return [d for d in docs if d.get("kind") == kind]


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


# A database source is mandatory for the default (postgres) storage backend, so
# every positive-path render supplies one.
_EXTERNAL = "secret.existingSecret=app-secret"


def test_bare_install_fails_without_database():
    """postgres backend (the default) with no DB source must fail at render time,
    not silently produce an empty INQTRIX_DATABASE_URL."""
    with pytest.raises(RuntimeError, match="no database is configured"):
        _template()


def test_memory_backend_needs_no_database():
    rendered = _template(
        "config.INQTRIX_STORAGE_BACKEND=memory", "migrations.enabled=false"
    )
    assert _by_kind(_docs(rendered), "Deployment")


def test_collaboration_is_absent_by_default() -> None:
    """The default release keeps every legacy resource and config path off."""
    docs = _docs(_template(_EXTERNAL))
    names = {d.get("metadata", {}).get("name", "") for d in docs}
    assert not any(name.endswith("-collaboration") for name in names)

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert "INQTRIX_COLLABORATION_ENABLED" not in config
    assert "INQTRIX_COLLABORATION_HTTP_URL" not in config
    assert "INQTRIX_COLLABORATION_WS_URL" not in config

    api = next(
        d
        for d in _by_kind(docs, "Deployment")
        if d["metadata"]["name"].endswith("-api")
    )
    api_env = api["spec"]["template"]["spec"]["containers"][0].get(
        "env", []
    )
    assert api["spec"]["template"]["spec"]["automountServiceAccountToken"] is False
    assert not any(
        item["name"] == "INQTRIX_COLLABORATION_SECRET" for item in api_env
    )


def test_collaboration_renders_one_private_recreate_replica() -> None:
    """Enabled collaboration is private, secret-scoped, and never scalable."""
    docs = _docs(
        _template(
            _EXTERNAL,
            "collaboration.enabled=true",
            "collaboration.replicaCount=4",
            "collaboration.secret.existingSecret=collaboration-secret",
            "config.INQTRIX_COLLABORATION_SNAPSHOT_IDLE_SECONDS=7",
            "config.INQTRIX_COLLABORATION_UPDATE_RATE_WINDOW_SECONDS=9",
            "config.INQTRIX_COLLABORATION_MAX_FRAME_BYTES=3145728",
            "config.INQTRIX_COLLABORATION_MAX_QUEUED_BYTES=524288",
            "config.INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES=11",
            "config.INQTRIX_COLLABORATION_MAINTENANCE_INTERVAL_SECONDS=17",
            "config.INQTRIX_COLLABORATION_RECONCILE_MAX_HASHES=77",
            "config.INQTRIX_COLLABORATION_RECONCILE_RATE_COUNT=19",
            "config.INQTRIX_COLLABORATION_RECONCILE_RATE_WINDOW_SECONDS=23",
            "config.INQTRIX_COLLABORATION_SNAPSHOT_RETRY_BASE_MS=300",
            "config.INQTRIX_COLLABORATION_SNAPSHOT_RETRY_MAX_MS=9000",
            "config.INQTRIX_COLLABORATION_SOCKET_BACKPRESSURE_BYTES=262144",
            "config.INQTRIX_COLLABORATION_TENANT_ID=tenant-primary",
        )
    )
    deployment = next(
        d
        for d in _by_kind(docs, "Deployment")
        if d["metadata"]["name"].endswith("-collaboration")
    )
    service = next(
        d
        for d in _by_kind(docs, "Service")
        if d["metadata"]["name"].endswith("-collaboration")
    )

    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [
        {
            "name": "http",
            "port": 1234,
            "targetPort": "http",
            "protocol": "TCP",
        }
    ]

    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == "inqtrix-collaboration:0.2.0"
    assert "volumes" not in pod
    assert "volumeMounts" not in container
    assert "envFrom" not in container
    assert container["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["readOnlyRootFilesystem"] is True

    env = {item["name"]: item for item in container["env"]}
    assert env["INQTRIX_API_INTERNAL_URL"]["value"] == (
        "http://rel-inqtrix-api:5100"
    )
    assert env["INQTRIX_COLLABORATION_SECRET"]["valueFrom"][
        "secretKeyRef"
    ] == {
        "name": "collaboration-secret",
        "key": "INQTRIX_COLLABORATION_SECRET",
    }
    assert env["INQTRIX_COLLABORATION_SNAPSHOT_IDLE_SECONDS"]["value"] == "7"
    assert env["INQTRIX_COLLABORATION_UPDATE_RATE_WINDOW_SECONDS"]["value"] == (
        "9"
    )
    assert env["INQTRIX_COLLABORATION_MAX_FRAME_BYTES"]["value"] == (
        "3145728"
    )
    assert env["INQTRIX_COLLABORATION_TENANT_ID"]["value"] == "tenant-primary"
    assert env["INQTRIX_COLLABORATION_MAINTENANCE_INTERVAL_SECONDS"][
        "value"
    ] == "17"
    assert env["INQTRIX_COLLABORATION_RECONCILE_MAX_HASHES"]["value"] == "77"
    assert env["INQTRIX_COLLABORATION_RECONCILE_RATE_COUNT"]["value"] == "19"
    assert env["INQTRIX_COLLABORATION_RECONCILE_RATE_WINDOW_SECONDS"][
        "value"
    ] == "23"
    assert env["INQTRIX_COLLABORATION_MAX_QUEUED_BYTES"]["value"] == "524288"
    assert env["INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES"]["value"] == "11"
    assert env["INQTRIX_COLLABORATION_SNAPSHOT_RETRY_BASE_MS"]["value"] == "300"
    assert env["INQTRIX_COLLABORATION_SNAPSHOT_RETRY_MAX_MS"]["value"] == "9000"
    assert env["INQTRIX_COLLABORATION_SOCKET_BACKPRESSURE_BYTES"]["value"] == "262144"
    assert set(env) == _node_collaboration_env_names()
    assert not any("DATABASE" in name for name in env)

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_COLLABORATION_ENABLED"] == "true"
    assert config["INQTRIX_COLLABORATION_HTTP_URL"] == (
        "http://rel-inqtrix-collaboration:1234"
    )
    assert config["INQTRIX_COLLABORATION_WS_URL"] == (
        "ws://rel-inqtrix-collaboration:1234/collaboration"
    )

    forbidden_kinds = {"HorizontalPodAutoscaler", "PodDisruptionBudget"}
    assert not any(
        d.get("kind") in forbidden_kinds
        and d.get("metadata", {}).get("name", "").endswith("-collaboration")
        for d in docs
    )
    assert not any(
        d.get("kind") == "PersistentVolumeClaim"
        and d.get("metadata", {}).get("name", "").endswith("-collaboration")
        for d in docs
    )

    api = next(
        d
        for d in _by_kind(docs, "Deployment")
        if d["metadata"]["name"].endswith("-api")
    )
    api_env = {
        item["name"]: item
        for item in api["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert api_env["INQTRIX_COLLABORATION_SECRET"]["valueFrom"][
        "secretKeyRef"
    ]["name"] == "collaboration-secret"


@pytest.mark.parametrize(
    "secret, message",
    [
        (None, "requires secret.data.INQTRIX_COLLABORATION_SECRET"),
        ("too-short", "must contain at least 32 characters"),
    ],
)
def test_collaboration_managed_secret_fails_loudly(
    secret: str | None, message: str
) -> None:
    args = ["postgres.enabled=true", "collaboration.enabled=true"]
    if secret is not None:
        args.append(f"secret.data.INQTRIX_COLLABORATION_SECRET={secret}")
    with pytest.raises(RuntimeError, match=message):
        _template(*args)


def test_vanilla_sets_nonroot_uid_and_no_route():
    rendered = _template(_EXTERNAL)
    docs = _docs(rendered)
    assert not _by_kind(docs, "Route"), "Route must not render without openshift.enabled"
    api = next(d for d in _by_kind(docs, "Deployment") if d["metadata"]["name"].endswith("-api"))
    pod_sc = api["spec"]["template"]["spec"]["securityContext"]
    assert pod_sc["runAsUser"] == 1001
    assert pod_sc["fsGroup"] == 1001


def test_tls_ingress_wires_the_public_origin_through_nginx() -> None:
    """TLS termination produces one trusted HTTPS origin at web and API."""
    docs = _docs(
        _template(
            _EXTERNAL,
            "ingress.enabled=true",
            "ingress.host=desk.example",
            "ingress.tls.enabled=true",
            "ingress.tls.secretName=desk-tls",
        )
    )
    ingress = _by_kind(docs, "Ingress")[0]
    assert ingress["spec"]["tls"] == [
        {"hosts": ["desk.example"], "secretName": "desk-tls"}
    ]
    ingress_path = ingress["spec"]["rules"][0]["http"]["paths"][0]
    assert ingress_path["path"] == "/"
    assert ingress_path["pathType"] == "Prefix"
    assert ingress_path["backend"]["service"]["name"] == "rel-inqtrix-web"

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_PUBLIC_BASE_URL"] == "https://desk.example"

    web = next(
        document
        for document in _by_kind(docs, "Deployment")
        if document["metadata"]["name"].endswith("-web")
    )
    env = {
        item["name"]: item["value"]
        for item in web["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["INQTRIX_EXTERNAL_SCHEME"] == "https"


def test_explicit_public_origin_overrides_ingress_derivation() -> None:
    """Operators retain the documented explicit public URL precedence."""
    docs = _docs(
        _template(
            _EXTERNAL,
            "ingress.enabled=true",
            "ingress.host=internal.example",
            "ingress.tls.enabled=true",
            "config.INQTRIX_PUBLIC_BASE_URL=https://public.example",
        )
    )
    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_PUBLIC_BASE_URL"] == "https://public.example"


def test_openshift_omits_uid_and_renders_route_not_ingress():
    rendered = _template(
        _EXTERNAL,
        "openshift.enabled=true",
        "ingress.enabled=true",  # must be ignored under openshift
    )
    docs = _docs(rendered)
    assert _by_kind(docs, "Route"), "Route must render under openshift.enabled"
    assert not _by_kind(docs, "Ingress"), "Ingress must not render under openshift.enabled"
    for dep in _by_kind(docs, "Deployment"):
        sc = dep["spec"]["template"]["spec"].get("securityContext", {})
        assert "runAsUser" not in sc, "OpenShift SCC assigns the UID; chart must not pin it"
        assert "fsGroup" not in sc


def test_openshift_auto_host_collaboration_requires_public_origin() -> None:
    """A post-render Route hostname cannot silently break TLS WebSockets."""
    with pytest.raises(
        RuntimeError,
        match="automatically assigned OpenShift Route host requires",
    ):
        _template(
            _EXTERNAL,
            "openshift.enabled=true",
            "collaboration.enabled=true",
            "collaboration.secret.existingSecret=collaboration-secret",
        )


def test_openshift_auto_host_collaboration_accepts_explicit_public_origin() -> None:
    """The final platform-assigned hostname may be supplied as a trust anchor."""
    docs = _docs(
        _template(
            _EXTERNAL,
            "openshift.enabled=true",
            "collaboration.enabled=true",
            "collaboration.secret.existingSecret=collaboration-secret",
            "config.INQTRIX_PUBLIC_BASE_URL=https://desk.apps.example",
        )
    )

    route = _by_kind(docs, "Route")[0]
    assert "host" not in route["spec"]
    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_PUBLIC_BASE_URL"] == "https://desk.apps.example"
    web = next(
        document
        for document in _by_kind(docs, "Deployment")
        if document["metadata"]["name"].endswith("-web")
    )
    env = {
        item["name"]: item["value"]
        for item in web["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["INQTRIX_EXTERNAL_SCHEME"] == "https"


def test_bundled_services_autowire_connections():
    # s3.enabled is part of the fixture: a worker pod shares the object
    # store with the API, and the chart refuses per-pod-disk "local"
    # for more than one sharer (see the I.2 guard test below).
    rendered = _template(
        "postgres.enabled=true",
        "qdrant.enabled=true",
        "valkey.enabled=true",
        "worker.enabled=true",
        "s3.enabled=true",
        "secret.data.INQTRIX_S3_ACCESS_KEY=k",
        "secret.data.INQTRIX_S3_SECRET_KEY=s",
    )
    docs = _docs(rendered)
    names = {f"{d['kind']}/{d['metadata']['name']}" for d in docs}
    assert "StatefulSet/rel-inqtrix-postgres" in names
    assert "StatefulSet/rel-inqtrix-qdrant" in names
    assert "StatefulSet/rel-inqtrix-valkey" in names

    secret = _by_kind(docs, "Secret")[0]["stringData"]
    assert "rel-inqtrix-postgres:5432" in secret["INQTRIX_DATABASE_URL"]
    assert "rel-inqtrix-valkey:6379" in secret["INQTRIX_VALKEY_URL"]

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_QDRANT_URL"] == "http://rel-inqtrix-qdrant:6333"
    assert config["INQTRIX_VECTOR_BACKEND"] == "qdrant"
    assert config["INQTRIX_QUEUE_BACKEND"] == "valkey"


def test_qdrant_mutable_paths_are_on_storage_volume():
    """Bundled Qdrant runs as non-root, so mutable paths must not target the
    image-owned /qdrant directory."""
    rendered = _template("postgres.enabled=true", "qdrant.enabled=true")
    qdrant = next(
        d
        for d in _by_kind(_docs(rendered), "StatefulSet")
        if d["metadata"]["name"].endswith("-qdrant")
    )
    container = qdrant["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    mounts = {item["name"]: item["mountPath"] for item in container["volumeMounts"]}

    assert mounts["storage"] == "/qdrant/storage"
    assert env["QDRANT_INIT_FILE_PATH"] == "/qdrant/storage/.qdrant-initialized"
    assert env["QDRANT__STORAGE__STORAGE_PATH"] == "/qdrant/storage"
    assert env["QDRANT__STORAGE__SNAPSHOTS_PATH"] == "/qdrant/storage/snapshots"
    assert env["QDRANT__STORAGE__TEMP_PATH"] == "/qdrant/storage/tmp"


def test_azure_s3_stack_renders_freeform_env_values():
    rendered = _template(
        "postgres.enabled=true",
        "qdrant.enabled=true",
        "valkey.enabled=true",
        "worker.enabled=true",
        "config.INQTRIX_AUTH_MODE=local",
        "config.INQTRIX_LOCAL_REGISTRATION=closed",
        "config.INQTRIX_LLM_PROVIDER=azure",
        "config.AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/",
        "config.INQTRIX_SEARCH_PROVIDER=azure_foundry",
        "config.AZURE_AI_PROJECT_ENDPOINT=https://example.services.ai.azure.com/api/projects/proj-default",
        "config.WEB_SEARCH_AGENT_NAME=web-search-agent",
        "config.WEB_SEARCH_AGENT_VERSION=2",
        "config.INQTRIX_EMBEDDING_PROVIDER=azure",
        "config.INQTRIX_EMBEDDING_AZURE_ENDPOINT=https://example.openai.azure.com/",
        "config.INQTRIX_EMBEDDING_MODEL=text-embedding-3-large",
        "config.INQTRIX_OBJECT_STORE_BACKEND=s3",
        "config.INQTRIX_S3_ENDPOINT_URL=https://s3.example.test",
        "config.INQTRIX_S3_BUCKET=inqtrix-files",
        "config.INQTRIX_QUOTA_ENABLED=true",
        "secret.data.AZURE_OPENAI_API_KEY=dummy-openai-key",
        "secret.data.AZURE_AI_PROJECT_API_KEY=dummy-foundry-key",
        "secret.data.INQTRIX_EMBEDDING_AZURE_API_KEY=dummy-embedding-key",
        "secret.data.INQTRIX_S3_ACCESS_KEY=dummy-s3-access",
        "secret.data.INQTRIX_S3_SECRET_KEY=dummy-s3-secret",
        extra=[
            "--set-string",
            "config.REASONING_MODEL=gpt-5.4",
            "--set-string",
            "config.TIER_HIGH_MODEL=gpt-5.4",
            "--set-string",
            "config.TIER_MID_MODEL=gpt-5.4",
            "--set-string",
            "config.TIER_FAST_MODEL=gpt-5.4-mini",
            "--set-string",
            r"config.INQTRIX_SELECTABLE_CHAT_MODELS=gpt-5.4\,gpt-5.4-mini\,gpt-5.4-nano",
        ],
    )
    docs = _docs(rendered)

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_AUTH_MODE"] == "local"
    assert config["INQTRIX_LLM_PROVIDER"] == "azure"
    assert config["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"
    assert config["REASONING_MODEL"] == "gpt-5.4"
    assert config["TIER_FAST_MODEL"] == "gpt-5.4-mini"
    assert config["INQTRIX_SELECTABLE_CHAT_MODELS"] == "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano"
    assert config["INQTRIX_SEARCH_PROVIDER"] == "azure_foundry"
    assert config["AZURE_AI_PROJECT_ENDPOINT"] == "https://example.services.ai.azure.com/api/projects/proj-default"
    assert config["WEB_SEARCH_AGENT_NAME"] == "web-search-agent"
    assert config["WEB_SEARCH_AGENT_VERSION"] == "2"
    assert config["INQTRIX_EMBEDDING_PROVIDER"] == "azure"
    assert config["INQTRIX_EMBEDDING_AZURE_ENDPOINT"] == "https://example.openai.azure.com/"
    assert config["INQTRIX_EMBEDDING_MODEL"] == "text-embedding-3-large"
    assert config["INQTRIX_KNOWLEDGE_ENABLED"] == "true"
    assert config["INQTRIX_VECTOR_BACKEND"] == "qdrant"
    assert config["INQTRIX_QUEUE_BACKEND"] == "valkey"
    assert config["INQTRIX_OBJECT_STORE_BACKEND"] == "s3"
    assert config["INQTRIX_S3_ENDPOINT_URL"] == "https://s3.example.test"
    assert config["INQTRIX_S3_BUCKET"] == "inqtrix-files"
    assert config["INQTRIX_QUOTA_ENABLED"] == "true"

    secret = _by_kind(docs, "Secret")[0]["stringData"]
    assert secret["AZURE_OPENAI_API_KEY"] == "dummy-openai-key"
    assert secret["AZURE_AI_PROJECT_API_KEY"] == "dummy-foundry-key"
    assert secret["INQTRIX_EMBEDDING_AZURE_API_KEY"] == "dummy-embedding-key"
    assert secret["INQTRIX_S3_ACCESS_KEY"] == "dummy-s3-access"
    assert secret["INQTRIX_S3_SECRET_KEY"] == "dummy-s3-secret"

    rendered_names = {item.get("metadata", {}).get("name", "").lower() for item in docs}
    assert not any("seaweed" in name for name in rendered_names)


def test_valkey_without_worker_does_not_switch_queue_backend():
    """Bundling Valkey without a worker must keep the in-process queue, not strand
    runs in a queue with no consumer."""
    rendered = _template("postgres.enabled=true", "valkey.enabled=true")
    config = _by_kind(_docs(rendered), "ConfigMap")[0]["data"]
    assert config.get("INQTRIX_QUEUE_BACKEND") != "valkey"


def test_bundled_password_with_url_reserved_chars_is_rejected():
    with pytest.raises(RuntimeError, match="URL-unreserved characters"):
        _template("postgres.enabled=true", "postgres.auth.password=p@ss:w/rd")


def test_migrate_hook_phase_tracks_database_origin():
    external = _template(_EXTERNAL)
    migrate_ext = next(
        d for d in _by_kind(_docs(external), "Job") if d["metadata"]["name"].endswith("-migrate")
    )
    assert migrate_ext["metadata"]["annotations"]["helm.sh/hook"] == "pre-install,pre-upgrade"

    bundled = _template("postgres.enabled=true")
    migrate_bun = next(
        d for d in _by_kind(_docs(bundled), "Job") if d["metadata"]["name"].endswith("-migrate")
    )
    assert migrate_bun["metadata"]["annotations"]["helm.sh/hook"] == "post-install,pre-upgrade"


def test_migration_secret_is_scoped_to_job_and_owner_upgrade_is_confirmed():
    rendered = _template(
        _EXTERNAL,
        "migrations.databaseSecret.name=migration-database",
        "migrations.databaseSecret.key=direct-url",
        "migrations.rlsMode=bypass",
    )
    docs = _docs(rendered)
    migrate = next(
        document
        for document in _by_kind(docs, "Job")
        if document["metadata"]["name"].endswith("-migrate")
    )
    pod_spec = migrate["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == "rel-inqtrix-migrate"
    assert pod_spec["automountServiceAccountToken"] is False
    env = {
        item["name"]: item
        for item in pod_spec["containers"][0]["env"]
    }
    assert env["INQTRIX_MIGRATION_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "migration-database",
        "key": "direct-url",
    }
    assert env["INQTRIX_MIGRATION_RLS_MODE"]["value"] == "bypass"
    assert "INQTRIX_DATABASE_URL" not in env

    for deployment in _by_kind(docs, "Deployment"):
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "migration-database" not in str(container.get("envFrom", []))
        component = deployment["metadata"]["name"].removeprefix("rel-inqtrix-")
        env = {item["name"]: item for item in container.get("env", [])}
        if component in {"api", "worker"}:
            assert env["INQTRIX_MIGRATION_DATABASE_URL"] == {
                "name": "INQTRIX_MIGRATION_DATABASE_URL",
                "value": "",
            }
        else:
            assert "INQTRIX_MIGRATION_DATABASE_URL" not in env

    with pytest.raises(RuntimeError, match="must differ from the runtime"):
        _template(
            _EXTERNAL,
            "migrations.databaseSecret.name=app-secret",
            "migrations.rlsMode=bypass",
        )
    with pytest.raises(RuntimeError, match="forbidden in the runtime"):
        _template(
            _EXTERNAL,
            "secret.data.INQTRIX_MIGRATION_DATABASE_URL=privileged",
        )

    with pytest.raises(RuntimeError, match="ownerMaintenanceConfirmed"):
        _template(
            _EXTERNAL,
            "migrations.rlsMode=owner",
            extra=["--is-upgrade"],
        )
    confirmed = _template(
        _EXTERNAL,
        "migrations.rlsMode=owner",
        "migrations.ownerMaintenanceConfirmed=true",
        extra=["--is-upgrade"],
    )
    confirmed_docs = _docs(confirmed)
    assert "INQTRIX_MIGRATION_SERVICES_QUIESCED" in confirmed
    maintenance = next(
        document
        for document in _by_kind(confirmed_docs, "Job")
        if document["metadata"]["name"].endswith("-owner-maintenance")
    )
    assert maintenance["metadata"]["annotations"]["helm.sh/hook-weight"] == "-20"
    assert maintenance["spec"]["template"]["spec"]["serviceAccountName"] == (
        "rel-inqtrix-owner-maintenance"
    )
    assert maintenance["spec"]["template"]["spec"][
        "automountServiceAccountToken"
    ] is True
    command = maintenance["spec"]["template"]["spec"]["containers"][0][
        "command"
    ]
    assert command == [
        "python",
        "-m",
        "inqtrix.deployment.kubernetes_maintenance",
    ]
    role = next(
        document
        for document in _by_kind(confirmed_docs, "Role")
        if document["metadata"]["name"].endswith("-owner-maintenance")
    )
    assert {rule["resources"][0] for rule in role["rules"]} == {
        "deployments/scale",
        "horizontalpodautoscalers",
        "pods",
        "rolebindings",
    }
    self_revoke = next(
        rule for rule in role["rules"] if rule["resources"] == ["rolebindings"]
    )
    assert self_revoke["verbs"] == ["delete"]
    maintenance_env = {
        item["name"]: item["value"]
        for item in maintenance["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }
    assert maintenance_env["INQTRIX_K8S_MAINTENANCE_ROLE_BINDING"] == (
        "rel-inqtrix-owner-maintenance"
    )


def test_owner_upgrade_reactivates_api_hpa_only_after_migration() -> None:
    owner = _docs(_template(
        _EXTERNAL,
        "api.autoscaling.enabled=true",
        "api.autoscaling.minReplicas=2",
        "config.INQTRIX_OBJECT_STORE_BACKEND=s3",
        "config.INQTRIX_S3_AUTH_MODE=default",
        "config.INQTRIX_S3_BUCKET=managed-bucket",
        "migrations.rlsMode=owner",
        "migrations.ownerMaintenanceConfirmed=true",
        extra=["--is-upgrade"],
    ))
    owner_api = next(
        document for document in _by_kind(owner, "Deployment")
        if document["metadata"]["name"].endswith("-api")
    )
    assert owner_api["spec"]["replicas"] == 2

    bypass = _docs(_template(
        _EXTERNAL,
        "api.autoscaling.enabled=true",
        "config.INQTRIX_OBJECT_STORE_BACKEND=s3",
        "config.INQTRIX_S3_AUTH_MODE=default",
        "config.INQTRIX_S3_BUCKET=managed-bucket",
        "migrations.rlsMode=bypass",
        extra=["--is-upgrade"],
    ))
    bypass_api = next(
        document for document in _by_kind(bypass, "Deployment")
        if document["metadata"]["name"].endswith("-api")
    )
    assert "replicas" not in bypass_api["spec"]


def test_byo_service_account_creates_none_and_requires_explicit_name() -> None:
    with pytest.raises(RuntimeError, match="requires an explicit"):
        _template(_EXTERNAL, "serviceAccount.create=false")

    rendered = _template(
        _EXTERNAL,
        "serviceAccount.create=false",
        "serviceAccount.name=inqtrix-runtime",
        "worker.enabled=true",
        "collaboration.enabled=true",
        "config.INQTRIX_OBJECT_STORE_BACKEND=s3",
        "config.INQTRIX_S3_AUTH_MODE=default",
        "config.INQTRIX_S3_BUCKET=managed-bucket",
    )
    docs = _docs(rendered)
    assert _by_kind(docs, "ServiceAccount") == []
    for document in docs:
        if document.get("kind") not in {"Deployment", "Job", "Pod"}:
            continue
        pod_spec = (
            document["spec"]["template"]["spec"]
            if document["kind"] in {"Deployment", "Job"}
            else document["spec"]
        )
        assert pod_spec["serviceAccountName"] == "inqtrix-runtime"

    with pytest.raises(RuntimeError, match="annotations requires"):
        _template(
            _EXTERNAL,
            extra=[
                "--set-string",
                r"serviceAccount.api.annotations.eks\.amazonaws\.com/role-arn=ignored",
            ],
        )


def test_chart_version_tracks_chart_contract_changes() -> None:
    chart = yaml.safe_load((_CHART / "Chart.yaml").read_text(encoding="utf-8"))
    assert chart["version"] == "0.1.6"


def test_workload_identity_and_ca_are_scoped_to_api_and_worker():
    rendered = _template(
        _EXTERNAL,
        "worker.enabled=true",
        "config.INQTRIX_OBJECT_STORE_BACKEND=s3",
        "config.INQTRIX_S3_AUTH_MODE=default",
        "config.INQTRIX_S3_BUCKET=managed-bucket",
        "serviceAccount.api.create=true",
        "serviceAccount.api.automountServiceAccountToken=true",
        "serviceAccount.worker.create=true",
        "serviceAccount.worker.automountServiceAccountToken=true",
        "s3.caBundle.existingConfigMap=managed-s3-ca",
        extra=[
            "--set-string",
            r"serviceAccount.api.annotations.eks\.amazonaws\.com/role-arn=api-role",
            "--set-string",
            r"serviceAccount.worker.annotations.eks\.amazonaws\.com/role-arn=worker-role",
        ],
    )
    docs = _docs(rendered)
    accounts = {
        account["metadata"]["name"]: account
        for account in _by_kind(docs, "ServiceAccount")
    }
    assert accounts["rel-inqtrix-api"]["metadata"]["annotations"] == {
        "eks.amazonaws.com/role-arn": "api-role"
    }
    assert accounts["rel-inqtrix-worker"]["metadata"]["annotations"] == {
        "eks.amazonaws.com/role-arn": "worker-role"
    }

    deployments = {
        deployment["metadata"]["name"].removeprefix("rel-inqtrix-"): deployment
        for deployment in _by_kind(docs, "Deployment")
    }
    for component in ("api", "worker"):
        pod_spec = deployments[component]["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == f"rel-inqtrix-{component}"
        assert pod_spec["automountServiceAccountToken"] is True
        mounts = pod_spec["containers"][0]["volumeMounts"]
        assert any(mount["name"] == "object-store-ca" for mount in mounts)
        ca_volume = next(
            volume for volume in pod_spec["volumes"]
            if volume["name"] == "object-store-ca"
        )
        assert ca_volume["configMap"]["name"] == "managed-s3-ca"

    web_spec = deployments["web"]["spec"]["template"]["spec"]
    assert web_spec["serviceAccountName"] == "rel-inqtrix-web"
    assert web_spec["automountServiceAccountToken"] is False
    assert "object-store-ca" not in str(web_spec)
    migrate = next(
        document for document in _by_kind(docs, "Job")
        if document["metadata"]["name"].endswith("-migrate")
    )
    assert "object-store-ca" not in str(migrate)
    assert "rel-inqtrix-api" not in str(migrate["spec"]["template"]["spec"])
    assert migrate["spec"]["template"]["spec"]["serviceAccountName"] == (
        "rel-inqtrix-migrate"
    )

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_S3_CA_BUNDLE"] == "/etc/inqtrix/object-store/ca.crt"


def test_helm_smoke_uses_readiness_and_requires_configured_s3():
    rendered = _template(
        _EXTERNAL,
        "config.INQTRIX_OBJECT_STORE_BACKEND=s3",
        "config.INQTRIX_S3_AUTH_MODE=default",
        "config.INQTRIX_S3_BUCKET=managed-bucket",
    )
    test_pod = next(
        document
        for document in _by_kind(_docs(rendered), "Pod")
        if document["metadata"]["name"].endswith("-test-health")
    )
    script = test_pod["spec"]["containers"][0]["command"][-1]
    assert "/readyz" in script
    assert 'get("object_store") != "ok"' in script
    assert test_pod["spec"]["automountServiceAccountToken"] is False
    assert test_pod["spec"]["serviceAccountName"] == "rel-inqtrix-test"


def test_database_runtime_login_policy_tracks_database_origin() -> None:
    external = _by_kind(_docs(_template(_EXTERNAL)), "ConfigMap")[0]["data"]
    bundled = _by_kind(
        _docs(_template("postgres.enabled=true")), "ConfigMap"
    )[0]["data"]

    assert external["INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY"] == "restricted"
    assert bundled["INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY"] == "bundled_legacy"

    explicit = _by_kind(
        _docs(
            _template(
                _EXTERNAL,
                "config.INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY=bundled_legacy",
            )
        ),
        "ConfigMap",
    )[0]["data"]
    assert explicit["INQTRIX_DATABASE_RUNTIME_LOGIN_POLICY"] == "bundled_legacy"


def test_secret_change_triggers_pod_rollout():
    """The api/worker pod template must carry a secret checksum so a changed
    secret value rolls the pods (envFrom alone would keep stale env)."""

    def api_checksum(*extra: str) -> str:
        docs = _docs(_template("postgres.enabled=true", "qdrant.enabled=true", *extra))
        api = next(d for d in _by_kind(docs, "Deployment") if d["metadata"]["name"].endswith("-api"))
        return api["spec"]["template"]["metadata"]["annotations"]["checksum/secret"]

    assert api_checksum() != api_checksum("secret.data.LITELLM_API_KEY=rotated-key")


def test_bundled_minio_renders_and_autowires_s3():
    """s3.enabled bundles a MinIO StatefulSet+Service and auto-wires the object-store
    backend/endpoint/bucket (ConfigMap) plus the access/secret keys (Secret)."""
    rendered = _template("postgres.enabled=true", "s3.enabled=true")
    docs = _docs(rendered)
    names = {f"{d['kind']}/{d['metadata']['name']}" for d in docs}
    assert "StatefulSet/rel-inqtrix-minio" in names
    assert "Service/rel-inqtrix-minio" in names

    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_OBJECT_STORE_BACKEND"] == "s3"
    assert config["INQTRIX_S3_ENDPOINT_URL"] == "http://rel-inqtrix-minio:9000"
    assert config["INQTRIX_S3_BUCKET"] == "inqtrix-files"
    assert config["INQTRIX_S3_AUTH_MODE"] == "static"
    assert config["INQTRIX_S3_ADDRESSING_STYLE"] == "path"
    assert config["INQTRIX_S3_BUCKET_PROVISIONING"] == "create_if_missing"

    secret = _by_kind(docs, "Secret")[0]["stringData"]
    assert secret["INQTRIX_S3_ACCESS_KEY"] == "inqtrix"
    assert secret["INQTRIX_S3_SECRET_KEY"] == "change-me-minio"


def test_openshift_omits_uid_on_bundled_qdrant_valkey_minio():
    """Under openshift.enabled the bundled qdrant/valkey/minio pods must NOT pin a
    UID/fsGroup (the restricted-v2 SCC assigns an arbitrary UID + GID 0), while still
    setting the seccomp profile. Postgres is intentionally excluded (vanilla only)."""
    rendered = _template(
        _EXTERNAL,
        "openshift.enabled=true",
        "qdrant.enabled=true",
        "valkey.enabled=true",
        "s3.enabled=true",
    )
    docs = _docs(rendered)
    for name in ("qdrant", "valkey", "minio"):
        sts = next(
            d for d in _by_kind(docs, "StatefulSet")
            if d["metadata"]["name"].endswith(f"-{name}")
        )
        sc = sts["spec"]["template"]["spec"]["securityContext"]
        assert sc["seccompProfile"]["type"] == "RuntimeDefault"
        assert "runAsUser" not in sc, f"{name}: SCC assigns the UID under openshift"
        assert "fsGroup" not in sc, f"{name}: SCC assigns the fsGroup under openshift"


def test_vanilla_pins_uid_on_bundled_qdrant_valkey_minio():
    """Without openshift.enabled the bundled services keep their fixed non-root UID so
    the data PVC is writable (vanilla behaviour must stay unchanged)."""
    rendered = _template(
        _EXTERNAL,
        "qdrant.enabled=true",
        "valkey.enabled=true",
        "s3.enabled=true",
    )
    docs = _docs(rendered)
    for name, uid in {"qdrant": 1000, "valkey": 999, "minio": 1000}.items():
        sts = next(
            d for d in _by_kind(docs, "StatefulSet")
            if d["metadata"]["name"].endswith(f"-{name}")
        )
        sc = sts["spec"]["template"]["spec"]["securityContext"]
        assert sc["runAsUser"] == uid
        assert sc["fsGroup"] == uid


def test_explicit_s3_config_overrides_bundled_minio():
    """A value set explicitly in config/secret must win over the auto-wired MinIO
    default (config > derived; secret.data > derived)."""
    rendered = _template(
        "postgres.enabled=true",
        "s3.enabled=true",
        "config.INQTRIX_S3_ENDPOINT_URL=https://external.example.test",
        "config.INQTRIX_S3_BUCKET=custom-bucket",
        "secret.data.INQTRIX_S3_ACCESS_KEY=explicit-access",
    )
    docs = _docs(rendered)
    config = _by_kind(docs, "ConfigMap")[0]["data"]
    assert config["INQTRIX_S3_ENDPOINT_URL"] == "https://external.example.test"
    assert config["INQTRIX_S3_BUCKET"] == "custom-bucket"

    secret = _by_kind(docs, "Secret")[0]["stringData"]
    assert secret["INQTRIX_S3_ACCESS_KEY"] == "explicit-access"
    # the key left unset still falls back to the bundled MinIO default
    assert secret["INQTRIX_S3_SECRET_KEY"] == "change-me-minio"


def test_local_object_store_refuses_multi_replica_render():
    """I.2: per-pod-disk blobs must not be shared by >1 pod."""
    # 2 API replicas on the default local store: refused.
    with pytest.raises(RuntimeError, match="per-pod disk"):
        _template(_EXTERNAL, "api.replicaCount=2")
    # Autoscaling ceiling counts even with replicaCount=1.
    with pytest.raises(RuntimeError, match="per-pod disk"):
        _template(
            _EXTERNAL,
            "api.autoscaling.enabled=true",
            "api.autoscaling.maxReplicas=3",
        )
    # A worker pod shares the store too.
    with pytest.raises(RuntimeError, match="per-pod disk"):
        _template(
            _EXTERNAL,
            "worker.enabled=true",
            "valkey.enabled=true",
        )
    # Bundled MinIO (s3.enabled) makes multi-replica renderable again, and
    # the replica hint reaches the app config for its own startup guard.
    rendered = _template(
        _EXTERNAL,
        "api.replicaCount=2",
        "s3.enabled=true",
        "secret.data.INQTRIX_S3_ACCESS_KEY=k",
        "secret.data.INQTRIX_S3_SECRET_KEY=s",
    )
    config = _by_kind(_docs(rendered), "ConfigMap")[0]["data"]
    assert config["INQTRIX_OBJECT_STORE_BACKEND"] == "s3"
    assert config["INQTRIX_REPLICA_COUNT"] == "2"
    # Single replica on local keeps working (zero-infra default).
    single = _template(_EXTERNAL)
    config = _by_kind(_docs(single), "ConfigMap")[0]["data"]
    assert config["INQTRIX_OBJECT_STORE_BACKEND"] == "local"
    assert config["INQTRIX_REPLICA_COUNT"] == "1"


def test_bundled_pgbouncer_pools_app_url_but_not_migrate():
    """I.1: the app routes through the pooler; Alembic stays direct."""
    # Requires the bundled Postgres.
    with pytest.raises(RuntimeError, match="requires postgres.enabled"):
        _template(_EXTERNAL, "pgbouncer.enabled=true")
    rendered = _template(
        "postgres.enabled=true",
        "pgbouncer.enabled=true",
    )
    docs = _docs(rendered)
    names = {f"{d['kind']}/{d['metadata']['name']}" for d in docs}
    assert "Deployment/rel-inqtrix-pgbouncer" in names
    assert "Service/rel-inqtrix-pgbouncer" in names

    secret = _by_kind(docs, "Secret")[0]["stringData"]
    app_url = secret["INQTRIX_DATABASE_URL"]
    assert "rel-inqtrix-pgbouncer:6432" in app_url
    assert "prepared_statement_cache_size=0" in app_url

    migrate = [
        d for d in _by_kind(docs, "Job")
        if "migrate" in d["metadata"]["name"]
    ][0]
    env = migrate["spec"]["template"]["spec"]["containers"][0]["env"]
    direct = [e for e in env if e["name"] == "INQTRIX_DATABASE_URL"][0]
    assert "rel-inqtrix-postgres:5432" in direct["value"]
    assert "pgbouncer" not in direct["value"]

    pooler = [
        d for d in _by_kind(docs, "Deployment")
        if "pgbouncer" in d["metadata"]["name"]
    ][0]
    container = pooler["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["POOL_MODE"] == "transaction"
    assert env["MAX_PREPARED_STATEMENTS"] == "200"
    assert env["DB_HOST"] == "rel-inqtrix-postgres"


def test_pgbouncer_disabled_keeps_direct_app_url():
    rendered = _template("postgres.enabled=true")
    secret = _by_kind(_docs(rendered), "Secret")[0]["stringData"]
    assert "rel-inqtrix-postgres:5432" in secret["INQTRIX_DATABASE_URL"]


def _api_deployment(docs: list[dict]) -> dict:
    return [
        d for d in _by_kind(docs, "Deployment")
        if d["metadata"]["name"] == "rel-inqtrix-api"
    ][0]


def test_metrics_disabled_by_default():
    """2.5: no INQTRIX_METRICS_ENABLED and no scrape annotations by default."""
    docs = _docs(_template(_EXTERNAL))
    config = [
        d for d in _by_kind(docs, "ConfigMap")
        if d["metadata"]["name"] == "rel-inqtrix"
    ][0]["data"]
    assert "INQTRIX_METRICS_ENABLED" not in config
    annotations = (
        _api_deployment(docs)["spec"]["template"]["metadata"].get(
            "annotations", {}
        )
    )
    assert "prometheus.io/scrape" not in annotations


def test_metrics_enabled_sets_flag_and_scrape_annotations():
    """2.5: the toggle wires the env flag and the classic scrape annotations."""
    docs = _docs(_template(_EXTERNAL, "metrics.enabled=true"))
    config = [
        d for d in _by_kind(docs, "ConfigMap")
        if d["metadata"]["name"] == "rel-inqtrix"
    ][0]["data"]
    assert config["INQTRIX_METRICS_ENABLED"] == "true"
    annotations = _api_deployment(docs)["spec"]["template"]["metadata"][
        "annotations"
    ]
    assert annotations["prometheus.io/scrape"] == "true"
    assert annotations["prometheus.io/path"] == "/metrics"

    # podAnnotations=false keeps the flag but drops the scrape annotations
    # (for a PodMonitor/ServiceMonitor-based Prometheus instead).
    docs2 = _docs(
        _template(
            _EXTERNAL, "metrics.enabled=true", "metrics.podAnnotations=false"
        )
    )
    annotations2 = (
        _api_deployment(docs2)["spec"]["template"]["metadata"].get(
            "annotations", {}
        )
    )
    assert "prometheus.io/scrape" not in annotations2
