{{/*
Common naming helpers.
*/}}
{{- define "inqtrix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "inqtrix.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "inqtrix.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Selector and full label sets. Both take a dict: { "root": $, "component": "api" }.
The component label keeps each Deployment's selector distinct.
*/}}
{{- define "inqtrix.selectorLabels" -}}
{{- $root := .root -}}
app.kubernetes.io/name: {{ include "inqtrix.name" $root }}
app.kubernetes.io/instance: {{ $root.Release.Name }}
{{- with .component }}
app.kubernetes.io/component: {{ . }}
{{- end }}
{{- end -}}

{{- define "inqtrix.labels" -}}
{{- $root := .root -}}
helm.sh/chart: {{ include "inqtrix.chart" $root }}
{{ include "inqtrix.selectorLabels" . }}
{{- with $root.Chart.AppVersion }}
app.kubernetes.io/version: {{ . | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ $root.Release.Service }}
app.kubernetes.io/part-of: inqtrix
{{- end -}}

{{- define "inqtrix.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "inqtrix.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Image references: [registry/]repository:tag, tag defaulting to the appVersion.
*/}}
{{- define "inqtrix.image.api" -}}
{{- $tag := default .Chart.AppVersion .Values.image.api.tag -}}
{{- if .Values.image.registry -}}
{{- printf "%s/%s:%s" .Values.image.registry .Values.image.api.repository $tag -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.api.repository $tag -}}
{{- end -}}
{{- end -}}

{{- define "inqtrix.image.web" -}}
{{- $tag := default .Chart.AppVersion .Values.image.web.tag -}}
{{- if .Values.image.registry -}}
{{- printf "%s/%s:%s" .Values.image.registry .Values.image.web.repository $tag -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.web.repository $tag -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret carrying the INQTRIX_* secret env: the user-supplied
existingSecret, otherwise the chart-managed Secret (named after the release).
*/}}
{{- define "inqtrix.secretName" -}}
{{- if .Values.secret.existingSecret -}}
{{- .Values.secret.existingSecret -}}
{{- else -}}
{{- include "inqtrix.fullname" . -}}
{{- end -}}
{{- end -}}

{{/*
Shared envFrom for the api/worker/migrate pods: the non-secret ConfigMap plus
the Secret (chart-managed or user-supplied existingSecret) when one is present.
*/}}
{{- define "inqtrix.envFrom" -}}
envFrom:
  - configMapRef:
      name: {{ include "inqtrix.fullname" . }}
{{- if or .Values.secret.existingSecret .Values.secret.create }}
  - secretRef:
      name: {{ include "inqtrix.secretName" . }}
{{- end }}
{{- end -}}

{{/*
Effective database URL: an explicit secret.data.INQTRIX_DATABASE_URL wins;
otherwise, when the bundled Postgres is enabled, the in-cluster connection is
derived. Used by both the chart Secret and the migrate hook so they agree.
*/}}
{{- define "inqtrix.databaseUrl" -}}
{{- if hasKey .Values.secret.data "INQTRIX_DATABASE_URL" -}}
{{- .Values.secret.data.INQTRIX_DATABASE_URL -}}
{{- else if .Values.postgres.enabled -}}
{{- if not (regexMatch "^[A-Za-z0-9._~-]+$" .Values.postgres.auth.password) -}}
{{- fail "postgres.auth.password must use only URL-unreserved characters [A-Za-z0-9._~-] when the bundled Postgres is enabled (it is embedded verbatim into INQTRIX_DATABASE_URL). For an arbitrary password, use an external database and set secret.data.INQTRIX_DATABASE_URL with your own URL-encoding." -}}
{{- end -}}
{{- printf "postgresql+asyncpg://%s:%s@%s-postgres:5432/%s" .Values.postgres.auth.username .Values.postgres.auth.password (include "inqtrix.fullname" .) .Values.postgres.auth.database -}}
{{- end -}}
{{- end -}}

{{/*
Effective database URL for the APP (api/worker): an explicit
secret.data.INQTRIX_DATABASE_URL wins; with the bundled PgBouncer enabled the
derived URL points at the pooler (transaction mode) and disables asyncpg's
prepared-statement cache; otherwise it is the direct bundled-Postgres URL.
The migrate hook deliberately keeps "inqtrix.databaseUrl" (DIRECT) -- Alembic
DDL must never run through transaction pooling.
*/}}
{{- define "inqtrix.appDatabaseUrl" -}}
{{- if hasKey .Values.secret.data "INQTRIX_DATABASE_URL" -}}
{{- .Values.secret.data.INQTRIX_DATABASE_URL -}}
{{- else if and .Values.pgbouncer.enabled .Values.postgres.enabled -}}
{{- /* Trigger the shared password-charset guard (single source). */ -}}
{{- $_ := include "inqtrix.databaseUrl" . -}}
{{- printf "postgresql+asyncpg://%s:%s@%s-pgbouncer:6432/%s?prepared_statement_cache_size=0" .Values.postgres.auth.username .Values.postgres.auth.password (include "inqtrix.fullname" .) .Values.postgres.auth.database -}}
{{- else -}}
{{- include "inqtrix.databaseUrl" . -}}
{{- end -}}
{{- end -}}

{{/*
Effective Valkey URL: explicit secret.data.INQTRIX_VALKEY_URL wins; otherwise
derived from the bundled Valkey (password embedded in the URL).
*/}}
{{- define "inqtrix.valkeyUrl" -}}
{{- if hasKey .Values.secret.data "INQTRIX_VALKEY_URL" -}}
{{- .Values.secret.data.INQTRIX_VALKEY_URL -}}
{{- else if .Values.valkey.enabled -}}
{{- if not (regexMatch "^[A-Za-z0-9._~-]+$" .Values.valkey.password) -}}
{{- fail "valkey.password must use only URL-unreserved characters [A-Za-z0-9._~-] when the bundled Valkey is enabled (it is embedded verbatim into INQTRIX_VALKEY_URL). For an arbitrary password, use an external Valkey and set secret.data.INQTRIX_VALKEY_URL with your own URL-encoding." -}}
{{- end -}}
{{- printf "redis://:%s@%s-valkey:6379/0" .Values.valkey.password (include "inqtrix.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Pod security context. seccompProfile is always RuntimeDefault. On vanilla
Kubernetes a fixed non-root UID and fsGroup are set (so a mounted PVC is
group-owned and writable); on OpenShift they are omitted so the restricted-v2
SCC assigns an arbitrary UID with supplemental GID 0.
*/}}
{{- define "inqtrix.podSecurityContext" -}}
seccompProfile:
  type: RuntimeDefault
{{- if not .Values.openshift.enabled }}
runAsNonRoot: true
runAsUser: {{ .Values.nonRootUid }}
fsGroup: {{ .Values.fsGroup }}
{{- end }}
{{- with .Values.podSecurityContext }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
Pod security context for a BUNDLED backing service (qdrant/valkey/minio).
seccompProfile is always RuntimeDefault. On vanilla Kubernetes the service's own
podSecurityContext (a fixed non-root UID + fsGroup, so the data volume is owned and
writable) is applied; under OpenShift (openshift.enabled) it is OMITTED so the
restricted-v2 SCC assigns an arbitrary UID with a supplemental GID 0 and the matching
fsGroup -- the data PVC/emptyDir is group-0-writable, which qdrant/valkey/minio (all
write only to their data dir and need no /etc/passwd entry) tolerate. Takes a dict:
{ "root": $, "svc": .Values.<svc>.podSecurityContext }. NOTE: the bundled Postgres
does NOT use this -- the official image's initdb needs the UID in /etc/passwd, so it
stays pinned (vanilla-k8s only); on OpenShift use an external/managed database.
*/}}
{{- define "inqtrix.bundledPodSecurityContext" -}}
seccompProfile:
  type: RuntimeDefault
{{- if not .root.Values.openshift.enabled }}
{{- with .svc }}
{{ toYaml . }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
Writable volumes/mounts for the api and worker pods under a read-only root
filesystem: scratch /tmp, the XDG cache (fastembed/qdrant-client models),
/app/logs (so INQTRIX_LOG_ENABLED=true does not crash on the read-only root --
note these logs are ephemeral; container stdout is the durable sink), and the
local object store. The object store mounts at the SAME path the app writes to
(config.INQTRIX_OBJECT_STORE_PATH) so the two can never drift; it is a PVC when
persistence is enabled, else an emptyDir. An emptyDir or ReadWriteOnce PVC is
not shared across pods, so a worker plus the local object store needs
co-location or the S3 object-store backend.
*/}}
{{- define "inqtrix.appWritableVolumes" -}}
- name: tmp
  emptyDir: {}
- name: cache
  emptyDir: {}
- name: logs
  emptyDir: {}
- name: objectstore
{{- if .Values.persistence.enabled }}
  persistentVolumeClaim:
    claimName: {{ include "inqtrix.fullname" . }}-objectstore
{{- else }}
  emptyDir: {}
{{- end }}
{{- end -}}

{{- define "inqtrix.appWritableVolumeMounts" -}}
- name: tmp
  mountPath: /tmp
- name: cache
  mountPath: /app/.cache
- name: logs
  mountPath: /app/logs
- name: objectstore
  mountPath: {{ .Values.config.INQTRIX_OBJECT_STORE_PATH | default "/var/lib/inqtrix/objects" }}
{{- end -}}

{{/*
Writable volumes/mounts for the nginx web pod under a read-only root filesystem:
the PID file and temp paths live under /tmp; the nginx cache dir; and
/etc/nginx/conf.d, where the entrypoint renders the config template at startup
(the read-only template itself stays under /etc/nginx/templates).
*/}}
{{- define "inqtrix.webWritableVolumes" -}}
- name: tmp
  emptyDir: {}
- name: nginx-cache
  emptyDir: {}
- name: nginx-conf
  emptyDir: {}
{{- end -}}

{{- define "inqtrix.webWritableVolumeMounts" -}}
- name: tmp
  mountPath: /tmp
- name: nginx-cache
  mountPath: /var/cache/nginx
- name: nginx-conf
  mountPath: /etc/nginx/conf.d
{{- end -}}
