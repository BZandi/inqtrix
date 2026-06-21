/**
 * The autosave diff loop shared by every project-entity sync (M6).
 *
 * Each entity hook (chat threads/groups, editor documents/folders/comments)
 * keeps an in-memory ``synced`` map of "what the server holds" per id; this
 * helper diffs the current local collection against it and issues the
 * minimal writes: push entities whose fingerprint changed (or are new),
 * delete entities the server has but the local collection no longer does.
 * Defined once (Designprinzip 4) so the diff cannot drift between entities.
 *
 * Generic over the fingerprint type ``F`` and its ``changed`` comparator, so
 * an entity may use a simple string fingerprint OR a structured one (chat
 * threads compare ``{updatedAt, groupId}``) without this helper caring.
 * ``synced`` is mutated in place: an id's fingerprint advances only after
 * its push succeeds, so a failed/aborted push is retried on the next pass
 * (idempotent server upserts make that safe).
 */

export type SyncCollectionArgs<T, F> = {
  /** The current local entities, keyed by id. */
  current: Record<string, T>
  /** Last successfully-synced fingerprint per id (mutated in place). */
  synced: Map<string, F>
  /** The fingerprint of an entity (the fields whose change needs a push). */
  fingerprintOf: (item: T) => F
  /** Whether ``current`` differs from the last-synced ``previous``. */
  changed: (previous: F | undefined, current: F) => boolean
  /** Push one new/changed entity to the server. */
  pushOne: (item: T) => Promise<void>
  /** Delete one entity (id) the local collection no longer has. */
  deleteOne: (id: string) => Promise<void>
}

export async function syncCollection<T, F>({
  current,
  synced,
  fingerprintOf,
  changed,
  pushOne,
  deleteOne,
}: SyncCollectionArgs<T, F>): Promise<void> {
  for (const item of Object.values(current)) {
    const fingerprint = fingerprintOf(item)
    const id = idOf(item)
    if (changed(synced.get(id), fingerprint)) {
      await pushOne(item)
      synced.set(id, fingerprint)
    }
  }
  for (const id of [...synced.keys()]) {
    if (!(id in current)) {
      await deleteOne(id)
      synced.delete(id)
    }
  }
}

/** Every project entity record carries a string ``id`` — the map key. */
function idOf(item: unknown): string {
  return (item as { id: string }).id
}

/**
 * Run a delete, swallowing a 404. The desired post-state of a delete is
 * "gone", so a target the server no longer knows is already that state. This
 * also makes a parent-cascade safe: deleting a file-library section cascades
 * its groups + assets server-side, so the autosave's subsequent explicit
 * child deletes would otherwise 404 and wedge in a retry loop. Shared by the
 * asset + vector-index sync hooks (Designprinzip 4).
 */
export async function deleteTolerant404(
  run: () => Promise<void>,
  isNotFound: (error: unknown) => boolean,
): Promise<void> {
  try {
    await run()
  } catch (error) {
    if (!isNotFound(error)) throw error
  }
}
