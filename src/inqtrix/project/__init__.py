"""Project-persistence tier (M6): server-persistent chat/editor/assets.

When the platform runs with a Postgres backend, the entities that
otherwise live only in the local markdown project become server-
persistent, scoped per ``(tenant_id, created_by_sub, workspace_id)`` like
the other owned resources. Without Postgres the app stays local-first —
the tier is capability-gated, never forced.

This package holds the store ports and their memory/Postgres
implementations per sub-etappe (chat first; editor and assets follow).
The relational schemas live in ``inqtrix.storage.*_orm``; the HTTP
surface in ``inqtrix.server.routers``; orchestration in
``inqtrix.services``.
"""
