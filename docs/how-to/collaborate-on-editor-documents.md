# Collaborate on Editor Documents

## Scope

This guide explains the user workflow for shared live editor documents: access
roles, Edit, Suggest, and Comment modes, shared discussions, the Changes
inspector, visible statuses, private AI work, source view, and export. It
assumes an operator has enabled the
optional service and an owner has converted the document. It does not cover
deployment or API integration.

## Before you start

Live collaboration requires:

- a `local`, `ldap`, or `oidc` browser login;
- an operator-enabled collaboration service;
- an editor document converted from Markdown to collaboration mode; and
- an accepted direct share for anyone other than the owner.

Conversion is permanent for that document. Export and import a detached
Markdown copy if you need an independently editable non-collaborative version.
Guest links, anonymous editing, PAT access, and offline writing are not
supported.

## Choose a role when sharing

The owner grants one role through the existing share workflow:

| Role | What the recipient can do |
|---|---|
| `View` | Read the live document and export it. |
| `Suggest` | Propose tracked changes and participate in team discussions. The recipient cannot accept or reject changes. |
| `Edit` | Edit directly, switch between Suggest and Comment, participate in discussions, and accept or reject suggestions. |

Only the owner can rename, move, share, or delete the document. A recipient
must accept the incoming share before it appears under **Shared with me**.
Revoking or downgrading a share takes effect on an open document; the editor
becomes read-only or closes access rather than waiting for a reload.

Shared rows are visually separated from personal documents under **Shared with
me** and retain a share icon, owner, and permission in search and pinned views.

## Edit, suggest, or comment

The write-mode control above the document is independent from how tracked
changes are displayed:

- **Edit** changes the shared final document immediately. The activity history
  records who changed it and when, but these edits have no later accept/reject
  action.
- **Suggest** records insertions, deletions, and modifications as open changes.
  Owners and `edit` recipients may choose either mode. A `suggest` share is
  locked to Suggest; a `view` share is locked to read-only.
- **Comment** blocks document mutations while keeping text selection and team
  discussion actions available. A `suggest` recipient may switch between
  Suggest and Comment.

Suggest mode supports insertions, deletions, replacements, formatting, and
reversible paragraph, heading, list, quote, and code-block transformations.
Table topology, merge/split, and atomic mathematics changes require Edit mode.
The editor shows a specific action error instead of converting an unsupported
operation into a direct edit or reporting it as a connection failure.

## Review changes

The right-side inspector has three top-level tabs:

- **Comments** contains shared threads, replies, mentions, and open/resolved
  filters.
- **Changes** contains current participants, open human/AI suggestions, and the
  document activity history.
- **AI** contains **Private AI notes**, AI review work, and private proposal
  preparation. The lock label confirms that these notes are visible only to
  you.

Under **Changes**, use **Open** for pending suggestions and **History** for
direct edits and completed decisions. Filter by person or change type, move
with Previous/Next, and select a row to focus its exact location. Rows stay
compact until selected. Accept and Reject are available to the owner and
`edit` recipients; batch actions show the affected count and require
confirmation.

The display menu changes presentation without changing the document:

| Display | Result |
|---|---|
| **Simple** | Shows the proposed final text with compact margin indicators; the selected change expands inline. |
| **All** | Shows all insertions, deletions, and modifications inline. |
| **Final** | Shows the projection as if every open suggestion were accepted. |
| **Original** | Shows the projection as if every open suggestion were rejected. |

The AI and Changes overlays are mutually exclusive. Switching to AI uses the
final presentation and shows only your private note anchor. Switching back
restores your previous Changes display. Team comments use gutter markers and a
quiet anchor line rather than a second full-text highlight, so comments and
tracked changes remain distinguishable when they overlap. Remote updates never
switch your tab or steal focus.

## Presence and save status

Other participants appear as text carets with server-verified names and stable
colors. Inqtrix does not show mouse pointers or remote selection fills. The
participant preview shows up to three people and then a `+N` count.

Connection and durability are separate:

| Status | Meaning and action |
|---|---|
| Connecting | The session and Yjs state are loading. Editing remains disabled. |
| Connected / Saved | The socket is live and every local update has a durable database acknowledgement. |
| Connected / Saving | One or more local update hashes are waiting for a durable acknowledgement. Do not treat `synced` alone as saved. |
| Reconnecting | The socket or lease was interrupted. The editor is read-only while it reloads the room. The status popover shows the next attempt; after repeated failures, use **Reconnect now** without reloading the page. |
| Read only | Your role, the feature kill switch, or a service outage permits only the stored projection. |
| Access revoked | The share, session, or account is no longer authorized. |
| Update required | The client protocol or editor schema does not match the document. Reload or deploy matching versions. |
| Error | An update was rejected or durability could not be proved. Editing stops instead of falling back to Markdown autosave. |

There is no offline document editing. Text entered before a connection failure
is safe only when the status reached Saved. On reconnect, the client reloads
the authoritative room and reconciles retained update hashes. Draft team
comments and replies remain in the browser until the server confirms them.

## AI and comments

Select text and choose **Team comment** to start a shared thread. The adjacent
menu contains **Private AI note**. Team threads support live replies, mentions,
author-only editing/deletion, resolve/reopen, and previous/next navigation.
Deleted contributions retain a tombstone. When the original passage disappears,
the thread remains under **No longer anchored** with its quote fallback.

Private AI notes, AI instructions, and draft AI results remain visible only to
their creator. The owner does not automatically see another user's private
work, and team threads are never sent to AI automatically. Choose **Use with
assistant** on one thread to attach it deliberately.

AI waits until current edits are durably acknowledged and the Markdown
projection has caught up. An AI result becomes visible to collaborators only
when it is published as a tracked suggestion. AI never writes directly into a
collaboration document. The initiating user is taken to Changes once; other
participants see the updated count without losing focus.

## Source and export

Source mode is read-only because raw Markdown and structured collaborative
editing cannot both be authoritative. It uses the same canonical serializer as
the server.

All readers may explicitly export the document. A current export waits for the
projection barrier. If the service cannot produce a current projection, the UI
may offer **Export last saved version** with its timestamp; it never uses that
older state silently.

Project export treats collaboration documents as detached Markdown snapshots.
Import creates new Markdown-mode document IDs. Shared incoming documents are
not copied into the recipient's project folders or project archive.

## Related docs

- [Editor collaboration architecture](../architecture/editor-collaboration.md)
- [Deploy editor collaboration](../deployment/editor-collaboration.md)
- [Authentication modes](../deployment/auth-modes.md)
- [React UI](../deployment/react-ui.md)
