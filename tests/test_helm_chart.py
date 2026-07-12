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

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_CHART = Path(__file__).resolve().parent.parent / "deploy" / "helm" / "inqtrix"

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


def test_vanilla_sets_nonroot_uid_and_no_route():
    rendered = _template(_EXTERNAL)
    docs = _docs(rendered)
    assert not _by_kind(docs, "Route"), "Route must not render without openshift.enabled"
    api = next(d for d in _by_kind(docs, "Deployment") if d["metadata"]["name"].endswith("-api"))
    pod_sc = api["spec"]["template"]["spec"]["securityContext"]
    assert pod_sc["runAsUser"] == 1001
    assert pod_sc["fsGroup"] == 1001


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
