import type { FileLibrarySectionRecord } from '@/features/project/types'

/**
 * Stable ids for the three prepared file-library sections. The temporary
 * section is the implicit target for chat/editor uploads and is recognised by
 * its `kind: 'temporary'` flag, not by id comparisons scattered across the app.
 */
export const FILE_SECTION_TEMP_ID = 'file-section-temp'
export const FILE_SECTION_LIBRARY_ID = 'file-section-library'
export const FILE_SECTION_SOURCES_ID = 'file-section-sources'

/**
 * Build the three prepared library sections. Titles are plain, renameable
 * strings (German defaults, matching the app's default locale) — once renamed
 * they become user-owned data, so they are not resolved through i18n.
 */
export function createDefaultFileLibrarySections(now: string): FileLibrarySectionRecord[] {
  return [
    { createdAt: now, id: FILE_SECTION_TEMP_ID, kind: 'temporary', title: 'Temporäre Dateien', updatedAt: now },
    { createdAt: now, id: FILE_SECTION_LIBRARY_ID, kind: 'custom', title: 'Bibliothek', updatedAt: now },
    { createdAt: now, id: FILE_SECTION_SOURCES_ID, kind: 'custom', title: 'Projekt-Quellen', updatedAt: now },
  ]
}
