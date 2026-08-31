# Editor Collaboration

## Scope

This page describes the optional live-collaboration architecture for editor
documents: ownership boundaries, persistence, synchronization, suggestions,
and failure semantics. It does not describe the end-user controls in detail;
see [Collaborate on editor documents](../how-to/collaborate-on-editor-documents.md).
The transport remains online-only and single-replica. It does not provide guest
links, offline document writes, mouse pointers, or full version restore.

## Guarantees

- The Yjs binary state is the body source of truth after conversion.
- PostgreSQL stores the update journal, verified snapshots, access leases,
  attribution, decisions, and the latest Markdown projection.
- A Hocuspocus `synced` event means CRDT synchronization only. A browser shows
  an edit as saved only after Inqtrix returns a durable acknowledgement for the
  committed update hash and sequence.
- FastAPI remains the authority for identity, direct shares, permissions,
  sequencing, retention, and row-level tenant isolation. The Node service has
  no database credentials.
- Legacy Markdown documents keep their existing revision-CAS save path. The
  collaboration path never falls back to a full-body Markdown `PUT`.

## Components and trust boundaries

The diagram answers: "Which process is allowed to decide and persist what?"

```mermaid
flowchart LR
    browser["Browser: React, Tiptap, Y.Doc"] -->|"REST + session cookie"| api["FastAPI"]
    browser -->|"same-origin /collaboration"| proxy["Vite dev proxy or selected web adapter"]
    proxy -->|"binary WebSocket relay"| gateway["FastAPI collaboration gateway"]
    gateway -->|"private WebSocket + service bearer"| node["Hocuspocus Node service"]
    node -->|"private authenticated HTTP"| api
    api --> postgres[("PostgreSQL")]
    schema["packages/editor-schema"] --> browser
    schema --> node
```

The public browser transport is always `/collaboration`. The Vite development
proxy and the selected production web adapter (Python by default, nginx only
when explicitly selected) forward it to FastAPI, not to Node. FastAPI validates
the browser Origin and relays binary frames to the private Node URL.
The same proxies forward the content-free `GET /collaboration/instance`
release probe to FastAPI. That probe reports an identity only when two
RLS-scoped reads of the canonical tenant's unexpired fencing lease agree around
a live Node readiness request. The Node service remains a single-writer
coordinator with no public host port in bundled deployments.

`packages/editor-schema` is the shared schema boundary. It contains the
headless Tiptap extensions, canonical Markdown parser and serializer,
suggestion marks, Yjs fragment name, relative-position helpers, room/parser
contracts, and deterministic schema fingerprint. React NodeViews and visual
decorations stay in the Research Desk.

## Stored representations

| Representation | Source of truth | Purpose |
|---|---|---|
| Yjs update journal | Yes, with a verified snapshot and remaining tail | Live body reconstruction, merge history, attribution, and idempotent replay. |
| Yjs snapshot | Checkpoint of the same truth | Bounds replay time; it never replaces uncommitted tail updates. |
| `content_markdown` | No, derived projection | Read-only outage view, search/context, source view, and export. |
| Patch and activity rows | Metadata, not body truth | Suggestion ownership, decisions, audit-oriented history, and inspector lists. |
| Shared comment threads and messages | Durable collaborative metadata | Anchored discussions, replies, mentions, resolution, tombstones, and per-user read cursors. |
| Awareness | No, ephemeral | Current participants and carets. It is not restored or backed up separately. |

Migration `0048_editor_collaboration` adds collaboration mode and sequence
metadata to `editor_documents`, then creates:

- `editor_collaboration_updates` for idempotent update hashes and ordered
  per-document sequences;
- `editor_collaboration_snapshots` for binary state, state vectors, hashes, and
  covered sequence;
- `editor_collaboration_leases` for short-lived document access;
- `editor_collaboration_instances` for the single active Node writer and its
  fencing epoch.

The tables use the existing PostgreSQL database and tenant RLS. There is no
SQLite database, second PostgreSQL server, or sidecar volume.

Migration `0051_editor_comments` adds separate thread, message, and read-cursor
tables for team discussions. Private AI notes remain in `editor_comments` and
are never migrated into shared threads. Migration `0052_editor_review` adds
bounded activity summaries and decision outcomes and advances compatible
collaboration documents to editor schema version 2.

## Document lifecycle

A document starts in `content_mode=markdown`. Only its owner can convert it,
and conversion checks the body revision, metadata revision, schema version,
document size, service readiness, and absence of active shares. Node parses
Markdown into a Y.Doc and returns an initial verified snapshot without writing
the database. FastAPI rechecks the preconditions and commits mode, generation,
snapshot, projection, and sequence metadata atomically.

Conversion is irreversible. To return to ordinary Markdown editing, export a
detached snapshot and import it as a new document. Deleting a collaboration
document creates a tombstone, increments its generation, revokes access, and
eventually permits physical cleanup. Version 1 has no restore UI.

The room name is
`inqtrix-editor-v1:{document_id}:g{collaboration_generation}`. A room name is
an address, never an authorization credential. Every connection and durable
write is checked against tenant, document, generation, lease, current user,
permission, and active Node fencing epoch.

## Authentication and authorization

Collaboration sessions require a real cookie-backed `local`, `ldap`, or `oidc`
login. Anonymous principals, the legacy static API key, and personal access
tokens cannot obtain a collaboration lease. A lease expires after 60 seconds
by default and is rotated while the document stays open. The browser and Node
treat its `cl1...` value as opaque; it never belongs in a URL, log, or metric.

Editor documents extend the existing direct-share system with one additional
permission:

| Permission | Live read | Team comment | Direct edit | Suggest | Accept/reject | Metadata/share/delete |
|---|---:|---:|---:|---:|---:|---:|
| Owner | Yes | Yes | Yes | Yes | Yes | Yes |
| `edit` | Yes | Yes | Yes | Yes | Yes | No |
| `suggest` | Yes | Yes | No | Yes, enforced | No | No |
| `view` | Yes | No | No | No | No | No |

Other resource types keep their existing `view|edit` matrix. Share, session,
and account invalidations reuse `user_events`; Node polls that feed, rechecks
affected leases, and closes access that is no longer valid.

## Public API surface

The editor API remains additive and keeps Markdown compatibility:

| Route | Collaboration behavior |
|---|---|
| `GET /v1/editor/documents?scope=owned|shared|all` | Returns metadata plus `content_mode`, `metadata_revision`, `access`, and collaboration sequence/projection metadata. The default remains `owned`. |
| `GET /v1/editor/documents/{id}` | Returns Markdown body for legacy documents or the latest confirmed projection for collaboration documents. |
| `PUT /v1/editor/documents/{id}` | Keeps the legacy full-body contract and returns `409 content_mode_changed` for a collaboration document. |
| `PATCH /v1/editor/documents/{id}` | Owner-only title/folder/metadata update using `expected_metadata_revision`. |
| `POST /v1/editor/documents/{id}/collaboration:enable` | Owner-only, atomic Markdown-to-Yjs conversion. |
| `POST /v1/editor/documents/{id}/collaboration/session` | Issues or rotates the opaque lease and returns room, access, user, and protocol/schema metadata. |
| `GET /v1/editor/documents/{id}/activity` | Keyset page of attributed direct, suggestion, decision, comment, and system updates. Bounded summaries expose at most three edits, each a verbatim excerpt of at most 160 characters. The excerpt is display text, never markup: it is never sanitised and never rejected on its characters, because it is the user's own typing and the only consumer renders it as escaped text. |
| `GET/POST /v1/editor/documents/{id}/collaboration/comments` | Incrementally lists or creates durable shared threads with participant metadata. |
| `POST .../comments/{thread_id}/replies` | Adds an idempotent reply with optional participant mentions. |
| `PATCH .../comments/{thread_id}` | Resolves or reopens a thread under live permission and revision checks. |
| `PATCH/DELETE .../comments/{thread_id}/messages/{message_id}` | Lets an author edit or tombstone only their own contribution. |
| `POST .../collaboration/comments/read` | Advances the caller's personal comment read cursor. |
| `POST /v1/editor/documents/{id}/patches:decide` | Idempotent accept/reject for explicit patch IDs using `expected_sequence` and a decision UUID. |
| `POST /v1/editor/documents/{id}/collaboration/projection:flush` | Waits for the room queue and returns a current confirmed Markdown projection. |
| `GET /collaboration/instance` | Unauthenticated, no-store release probe returning only contract/service/status and the stable DB-fenced instance ID/epoch; unavailable or changing state returns 503. |
| `GET /v1/capabilities` | Reports configured/available state, `/collaboration`, protocol/schema versions, and `single_replica`. |

Missing or hidden documents return 404; a visible but unauthorized action
returns 403; revision, sequence, mode, generation, or schema conflicts return
409; size limits return 413; and an unavailable optional service returns 503.
Errors use the normal Inqtrix envelope with a machine-readable `reason`.

## Durable update flow

This sequence explains why a visible remote edit and a saved edit are separate
states.

```mermaid
sequenceDiagram
    participant B as Browser
    participant N as Node coordinator
    participant A as FastAPI
    participant P as PostgreSQL
    B->>N: Yjs binary update + opaque lease
    N->>N: clone, schema/size/role/suggestion validation
    N->>A: persist update + actor + fencing epoch
    A->>P: lock document, recheck access, allocate sequence, commit
    P-->>A: original update sequence + current persisted sequence
    A-->>N: both coordinates and idempotency result
    N->>N: apply committed update to active Y.Doc
    N-->>B: durable_ack(hash, sequence)
    N-->>B: broadcast committed update to peers
```

Each room has a serial queue. The coordinator validates on a clone before
persistence. FastAPI locks the document row, rejects stale generation or
fencing epochs, and writes the update plus suggestion/decision metadata in one
transaction. Only then does Node apply and broadcast the update. Repeating an
update hash returns its original sequence instead of inserting a duplicate,
alongside the locked document's current persisted sequence so a replay after
later edits cannot regress the room watermark.

The initial Yjs sync response is allowed to be a no-op when applying its
canonical V1 payload introduces no novel state. This semantic check is
deliberately not limited to the two-byte empty update: after a sidecar restart,
a fully synchronized browser can legitimately send a non-empty redundant
state update. Ordinary update frames still require either novel validated
state or a matching durable journal hash, and suggestion topology validation
still runs for any update that changes Yjs structure.

If applying a committed update to the active in-memory Y.Doc fails, the room is
closed with `1011` and rebuilt from the latest verified snapshot plus update
tail. The committed database state remains authoritative.

If a client transport disappears after persistence starts but before the
committed update can be applied to the active Y.Doc, the room enters a
reconstruction quarantine with a new room epoch. Frames already queued by
other clients against the previous epoch are neither applied nor acknowledged.
Those clients receive the retryable `1012/restarting` close and reconnect only
after the room has been rebuilt from durable state. The transport whose
in-flight operation was interrupted may retain the more specific
`4401/invalid_lease` close. This separates one initiating transport failure
from collateral recovery and prevents stale queued frames from becoming
internal-consistency errors or duplicate mutations.

Awareness remains ephemeral and is never persisted. Hocuspocus 4.3 and 4.4
decode an inbound awareness frame through a scratch `Awareness` object whose
constructor inserts one leading empty local state. The sidecar removes only
that exact leading sentinel before ownership and single-sender identity
validation. Any different multi-state shape is rejected rather than guessed
at, and the normalization counter makes the compatibility adapter observable.

## Suggestions and AI

Direct edits are immediately part of the final document and appear in activity
history; Inqtrix does not reconstruct them later as accept/reject patches.
Suggest mode records five semantic kinds: insertion, deletion, replacement,
format, and structure. Text changes use suggestion marks; reversible heading,
paragraph, list, quote, and code-block transformations use canonical structure
metadata. Every suggestion carries a secure UUID, patch UUID, author, and
creation time. Server validation requires the original projection to remain
unchanged for a `suggest` user and prevents a user from rewriting or deciding
another user's suggestion.

Accept and reject are idempotent server mutations with an expected sequence
and command UUID. Table topology, merge/split operations, and other structure
changes without a safe inverse are rejected with a specific editor-action
error; they do not close the collaboration transport.

Shared comments and private AI work are separate domains. Team threads are
persisted with a Yjs-relative anchor and bounded quote fallback, delivered
through a transactional `user_events` invalidation, and incrementally reloaded
after reconnect. Replies, mentions, resolve/reopen, author-only editing, and
tombstones never mutate the Y.Doc. A thread that can no longer resolve its
anchor remains visible as orphaned. Private AI notes remain creator-only and
are sent to the assistant only through an explicit user action.

AI first waits for all durable acknowledgements and a current Markdown
projection. Its result is then published through the same Node coordinator as
a shared suggestion, never as a direct collaboration edit and never through
the legacy Markdown patch path.

## Projection, source, and export

The final Markdown projection is generated with the shared serializer after a
durable sequence. Source mode is read-only for collaboration documents. An
explicit projection barrier is required before AI context or a current export;
the system does not silently consume a stale projection. If durable writes
advance while projection is being published, the barrier reprojects at the new
watermark. It returns Markdown only when projection and persisted sequences
are equal; repeated races end in a visible `409 projection_not_current`.

Owned collaboration documents export as detached Markdown snapshots. Import
creates a new document ID in Markdown mode, so an exported project cannot
reconnect to the source room. Shared incoming documents are not copied into the
recipient's local project directories or project archive. When only an older
projection is available, the UI must label its timestamp and require an
explicit choice to export that saved state.

## Failure and compatibility model

The editor becomes read-only while connecting, reconnecting, after access
revocation, after a durability error, or when protocol/schema versions differ.
Transport and lease failures retry with jittered 1, 2, 4, 8, 15, then
30-second delays. A manual retry is single-flight: it disconnects the stale
transport, rotates the lease, reconnects, synchronizes authoritative Yjs state,
and idempotently replays retained local updates. Authentication, compatibility,
revocation, and durable rejection use distinct recovery actions instead of one
generic retry loop.

An expired collaboration lease is a controller-owned recovery condition, not a
lost browser login. The session endpoint may return one `401` for the stale
lease after a sidecar epoch change; the controller then requests a fresh lease
without reloading the SPA or leaving the open editor. A second unauthenticated
response becomes the explicit sign-in recovery state.

There is no offline document-write queue and no fallback to legacy autosave.
Unsaved shared-comment drafts are retained in browser storage, but their
messages are not considered persisted until the API confirms them. Disabling
the feature removes the collaboration routes but leaves legacy documents
unchanged; existing collaboration documents remain available through their
last persisted Markdown projection in read-only form.

Protocol close codes are stable operator signals:

| Code | Meaning |
|---|---|
| `4401` | Lease invalid or expired. |
| `4403` | Origin or document access rejected. |
| `4409` | Binary protocol, generation, or schema incompatible. |
| `4429` | Connection/update/awareness rate limit. |
| `4503` | Collaboration service not ready. |
| `1009` | Frame exceeds the configured limit. |
| `1011` | Persistence or internal consistency failure. |
| `1012` | Retryable room reconstruction or planned service restart. |

## Related docs

- [Collaborate on editor documents](../how-to/collaborate-on-editor-documents.md)
- [Deploy editor collaboration](../deployment/editor-collaboration.md)
- [Data architecture](data-architecture.md)
- [Authentication modes](../deployment/auth-modes.md)
