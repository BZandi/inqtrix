import type { FileLibrarySectionRecord } from '@/features/project/types'
import { createProjectEntityId } from '@/features/project/entityId'

export const LEGACY_FILE_SECTION_IDS = [
  'file-section-temp',
  'file-section-library',
  'file-section-sources',
] as const

export function createFileSectionId(): string {
  return createProjectEntityId('file-section')
}

/**
 * Build the three prepared library sections. Titles are plain, renameable
 * strings (German defaults, matching the app's default locale) — once renamed
 * they become user-owned data, so they are not resolved through i18n.
 */
export function createDefaultFileLibrarySections(now: string): FileLibrarySectionRecord[] {
  return [
    { createdAt: now, id: createFileSectionId(), kind: 'temporary', title: 'Temporäre Dateien', updatedAt: now },
    { createdAt: now, id: createFileSectionId(), kind: 'custom', title: 'Bibliothek', updatedAt: now },
    { createdAt: now, id: createFileSectionId(), kind: 'custom', title: 'Projekt-Quellen', updatedAt: now },
  ]
}

export function temporaryFileSectionId(
  sections: Iterable<FileLibrarySectionRecord>,
): string {
  for (const section of sections) {
    if (section.kind === 'temporary') return section.id
  }
  throw new Error('File library has no temporary section.')
}

export function legacyFileSectionIdReplacements(
  sections: Record<string, FileLibrarySectionRecord>,
  serverIds: ReadonlySet<string>,
): Record<string, string> {
  const occupied = new Set([...Object.keys(sections), ...serverIds])
  const replacements: Record<string, string> = {}
  for (const legacyId of LEGACY_FILE_SECTION_IDS) {
    if (!sections[legacyId] || serverIds.has(legacyId)) continue
    let replacement = createFileSectionId()
    while (occupied.has(replacement)) replacement = createFileSectionId()
    occupied.add(replacement)
    replacements[legacyId] = replacement
  }
  return replacements
}
