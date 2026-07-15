# Collaborate on Editor Documents

## Scope

This guide explains the user workflow for shared live editor documents: access
roles, Edit and Suggest modes, the Changes inspector, visible statuses, private
AI work, source view, and export. It assumes an operator has enabled the
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
| `Suggest` | Read and propose tracked insertions, deletions, and text changes. The recipient cannot accept or reject changes. |
| `Edit` | Edit directly, switch to Suggest, and accept or reject suggestions. |

Only the owner can rename, move, share, or delete the document. A recipient
must accept the incoming share before it appears under **Shared with me**.
Revoking or downgrading a share takes effect on an open document; the editor
becomes read-only or closes access rather than waiting for a reload.

## Edit or suggest

The write-mode control above the document is independent from how tracked
changes are displayed:

- **Edit** changes the shared final document immediately. The activity history
  records who changed it and when, but these edits have no later accept/reject
  action.
- **Suggest** records insertions, deletions, and modifications as open changes.
  Owners and `edit` recipients may choose either mode. A `suggest` share is
  locked to Suggest; a `view` share is locked to read-only.

Suggest mode is intended for text-level review. Table cell text is supported,
but row/column operations, merge/split, and atomic mathematics changes require
Edit mode. The editor shows a visible rejection instead of silently converting
an unsupported structural action into a direct edit.

## Review changes

The right-side inspector has two top-level tabs:

- **Assistant** contains your comments, AI review work, and private proposal
  preparation.
- **Changes** contains current participants, open human/AI suggestions, and the
  document activity history.

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

The Assistant and Changes overlays are mutually exclusive. Switching to
Assistant uses the final presentation and shows only your private assistant or
comment anchor. Switching back restores your previous Changes display. Normal
highlight formatting in the document is content, not an overlay, and remains
visible. Remote updates never switch your tab or steal focus.

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
| Reconnecting | The socket or lease was interrupted. The editor is read-only while it reloads the room. |
| Read only | Your role, the feature kill switch, or a service outage permits only the stored projection. |
| Access revoked | The share, session, or account is no longer authorized. |
| Update required | The client protocol or editor schema does not match the document. Reload or deploy matching versions. |
| Error | An update was rejected or durability could not be proved. Editing stops instead of falling back to Markdown autosave. |

There is no offline editing. Text entered before a connection failure is safe
only when the status reached Saved. On reconnect, the client reloads the
authoritative room and reconciles acknowledged update hashes.

## AI and comments

Comments, AI instructions, and draft AI results are private to the user who
created them. The owner does not automatically see another user's private
work. Comment anchors move with the Yjs document and retain a text-context
fallback when their target is deleted.

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
