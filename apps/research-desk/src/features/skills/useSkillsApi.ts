import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createSkill,
  deleteSkill,
  exportSkillMarkdown,
  hasHttpStatus,
  importSkillMarkdown,
  listSkills,
  updateSkill,
  type ClientOptions,
} from '@/api/inqtrixClient'
import type { SkillInfo, SkillPayload } from './skillLibrary'

/**
 * Server-first skill state (plan M3 `3.6`): skills are a NEW vertical
 * with no browser-local legacy, so — unlike chat rules — there is no
 * local/synced dual state to reconcile. The hook loads the visible
 * list once (and after every write), surfaces errors loudly, and in
 * demo mode serves an in-memory list seeded with the demo skills so
 * the whole editor is demo-visible without a server.
 */

export type SkillsApiHandle = {
  skills: SkillInfo[]
  loading: boolean
  /** Loud, user-visible failure of the LAST operation ('' = none). */
  error: string
  /** True when writes are possible (server present or demo). */
  writable: boolean
  /** True when SKILL.md export/import works — server only; the demo
   * list is in-memory and the canonical serializer lives server-side,
   * so demo shows the controls visibly disabled instead of faking a
   * second serializer. */
  transferEnabled: boolean
  refresh: () => Promise<void>
  create: (payload: SkillPayload) => Promise<SkillInfo | null>
  update: (
    skillId: string,
    payload: SkillPayload,
    expectedUpdatedAt: number,
  ) => Promise<SkillInfo | null>
  remove: (skillId: string) => Promise<boolean>
  /** SKILL.md text of one skill, or null on failure (error is set). */
  exportMarkdown: (skillId: string) => Promise<string | null>
  /** Create a skill from SKILL.md text (server-validated). */
  importMarkdown: (markdown: string) => Promise<SkillInfo | null>
}

const DEMO_SKILLS: SkillInfo[] = [
  {
    id: 'sk_demo_sprechzettel',
    label: 'sprechzettel',
    title: 'Sprechzettel',
    description: 'Kompakter Sprechzettel fuer Termine und Gremien.',
    when_to_use:
      'Wenn Stichpunkte fuer einen Termin oder Auftritt gebraucht werden.',
    instructions_markdown:
      'Erstelle einen Sprechzettel fuer {{anlass}} mit Blick auf '
      + '{{publikum}}.\n\n- Kernbotschaften zuerst\n- Maximal eine Seite',
    clarification_points: [
      {
        id: 'p1',
        name: 'anlass',
        question: 'Fuer welchen Anlass ist der Sprechzettel?',
        options: [
          { id: 'p1_o1', label: 'Vorstandssitzung' },
          { id: 'p1_o2', label: 'Kundentermin' },
        ],
        required: true,
        default_assumption: 'Interner Termin',
      },
      {
        id: 'p2',
        name: 'publikum',
        question: 'Wer ist das Publikum?',
        options: [],
        required: false,
        default_assumption: 'Fachpublikum',
      },
    ],
    deliverable: 'talking_points',
    allowed_tools: [],
    requires_plan: 'never',
    invocation: 'model_allowed',
    argument_hint: 'Anlass und Kernbotschaft',
    model_tier: '',
    effort: '',
    include_in_autocomplete: true,
    created_at: 1751328000,
    updated_at: 1751328000,
  },
  {
    id: 'sk_demo_email_stil',
    label: 'email-stil',
    title: 'E-Mail-Stil',
    description: 'Formuliert E-Mail-Entwuerfe im Hausstil.',
    when_to_use: 'Wenn eine E-Mail entworfen oder umformuliert werden soll.',
    instructions_markdown:
      'Schreibe E-Mails knapp, freundlich und ohne Floskeln. '
      + 'Adressiere {{empfaenger}} passend zur Beziehung.',
    clarification_points: [
      {
        id: 'p1',
        name: 'empfaenger',
        question: 'An wen geht die E-Mail?',
        options: [],
        required: true,
        default_assumption: '',
      },
    ],
    deliverable: 'email',
    allowed_tools: ['search_project_knowledge', 'write_canvas'],
    requires_plan: 'auto',
    invocation: 'model_allowed',
    argument_hint: 'Empfaenger und Anliegen',
    model_tier: '',
    effort: '',
    include_in_autocomplete: true,
    created_at: 1751328000,
    updated_at: 1751328000,
  },
]

export function useSkillsApi({
  clientOptions,
  demo,
  enabled,
}: {
  /** Auth/workspace options for the server calls; null while locked. */
  clientOptions: ClientOptions | null
  /** Demo mode: serve the in-memory demo list, no server calls. */
  demo: boolean
  /** features.skills — false hides the tab, the hook stays inert. */
  enabled: boolean
}): SkillsApiHandle {
  const [skills, setSkills] = useState<SkillInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const demoCounter = useRef(0)

  const refresh = useCallback(async () => {
    if (!enabled) return
    if (demo) {
      setSkills((current) => (current.length ? current : [...DEMO_SKILLS]))
      return
    }
    if (!clientOptions) return
    setLoading(true)
    try {
      setSkills(await listSkills(clientOptions))
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }, [clientOptions, demo, enabled])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const create = useCallback(
    async (payload: SkillPayload) => {
      if (demo) {
        demoCounter.current += 1
        const now = Date.now() / 1000
        const record: SkillInfo = {
          ...payload,
          id: `sk_demo_neu_${demoCounter.current}`,
          created_at: now,
          updated_at: now,
        }
        setSkills((current) => [record, ...current])
        return record
      }
      if (!clientOptions) return null
      try {
        const record = await createSkill(payload, clientOptions)
        setSkills((current) => [record, ...current])
        setError('')
        return record
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
        return null
      }
    },
    [clientOptions, demo],
  )

  const update = useCallback(
    async (
      skillId: string,
      payload: SkillPayload,
      expectedUpdatedAt: number,
    ) => {
      if (demo) {
        // Computed from the CURRENT list, not inside the state updater:
        // an updater side effect only looks synchronous while React
        // runs it eagerly — deferred, the return value would be null.
        const existing = skills.find((skill) => skill.id === skillId)
        if (!existing) return null
        const updated: SkillInfo = {
          ...existing,
          ...payload,
          updated_at: Date.now() / 1000,
        }
        setSkills((current) =>
          current.map((skill) => (skill.id === skillId ? updated : skill)))
        return updated
      }
      if (!clientOptions) return null
      try {
        const record = await updateSkill(
          skillId,
          { ...payload, expected_updated_at: expectedUpdatedAt },
          clientOptions,
        )
        setSkills((current) =>
          current.map((skill) => (skill.id === skillId ? record : skill)))
        setError('')
        return record
      } catch (cause) {
        if (hasHttpStatus(cause, 409)) {
          // Conflict recovery: pull the winning server version so the
          // editor can rebase its precondition — without this every
          // retry hits the same 409 until a full reload.
          try {
            setSkills(await listSkills(clientOptions))
          } catch {
            // The stale list stays; the conflict error below is shown.
          }
        }
        setError(cause instanceof Error ? cause.message : String(cause))
        return null
      }
    },
    [clientOptions, demo, skills],
  )

  const remove = useCallback(
    async (skillId: string) => {
      if (demo) {
        setSkills((current) =>
          current.filter((skill) => skill.id !== skillId))
        return true
      }
      if (!clientOptions) return false
      try {
        await deleteSkill(skillId, clientOptions)
        setSkills((current) =>
          current.filter((skill) => skill.id !== skillId))
        setError('')
        return true
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
        return false
      }
    },
    [clientOptions, demo],
  )

  const exportMarkdown = useCallback(
    async (skillId: string) => {
      if (demo || !clientOptions) return null
      try {
        const text = await exportSkillMarkdown(skillId, clientOptions)
        setError('')
        return text
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
        return null
      }
    },
    [clientOptions, demo],
  )

  const importMarkdown = useCallback(
    async (markdown: string) => {
      if (demo || !clientOptions) return null
      try {
        const record = await importSkillMarkdown(markdown, clientOptions)
        setSkills((current) => [record, ...current])
        setError('')
        return record
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
        return null
      }
    },
    [clientOptions, demo],
  )

  return {
    skills,
    loading,
    error,
    writable: demo || clientOptions != null,
    transferEnabled: !demo && clientOptions != null,
    refresh,
    create,
    update,
    remove,
    exportMarkdown,
    importMarkdown,
  }
}
