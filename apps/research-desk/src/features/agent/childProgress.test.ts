import { describe, expect, it } from 'vitest'

import { translations } from '../../i18n/translations'
import { childProgressLine } from './childProgress'

const t = translations.de

describe('childProgressLine', () => {
  it('names the phase a delegated mission is in', () => {
    // The regression: for fifty minutes this returned nothing to render.
    expect(
      childProgressLine(
        { runStatus: 'running', snapshot: { phase: 'execution' } },
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus')
  })

  it('counts the current task from one, not from zero', () => {
    expect(
      childProgressLine(
        {
          runStatus: 'running',
          snapshot: { phase: 'execution' },
          openTasks: [1],
        },
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus · Aufgabe 2')
  })

  it('drops the task once it settled, keeping the phase', () => {
    expect(
      childProgressLine(
        {
          runStatus: 'running',
          snapshot: { phase: 'evidence' },
          openTasks: [],
        },
        t,
      ),
    ).toBe('Unterauftrag · Konsolidiert Belege')
  })

  it('says why nothing moves when the child parked on a decision', () => {
    expect(
      childProgressLine(
        { runStatus: 'waiting_for_approval', snapshot: { phase: 'planning' } },
        t,
      ),
    ).toBe('Unterauftrag · wartet auf eine Entscheidung')
  })

  it('falls back to the child message when no phase arrived', () => {
    expect(
      childProgressLine({ runStatus: 'running', message: 'Sucht Quellen' }, t),
    ).toBe('Unterauftrag · Sucht Quellen')
  })

  it('shows nothing rather than inventing progress', () => {
    expect(childProgressLine({ runStatus: 'running' }, t)).toBeNull()
  })

  it('shows nothing once the child has finished', () => {
    // The delegation row's own status already says it completed.
    expect(
      childProgressLine(
        { runStatus: 'completed', snapshot: { phase: 'done' } },
        t,
      ),
    ).toBeNull()
  })

  it('keeps saying so when the child failed', () => {
    // The tool call returns normally either way, so the row shows a
    // check. Without this line a failed subtask reads as a done one.
    expect(
      childProgressLine(
        { runStatus: 'failed', snapshot: { phase: 'execution' } },
        t,
      ),
    ).toBe('Unterauftrag · fehlgeschlagen')
  })

  it('names the reason a child failed when one arrived', () => {
    expect(
      childProgressLine(
        { runStatus: 'failed', error: 'Kein Projektwissen erreichbar' },
        t,
      ),
    ).toBe('Unterauftrag · fehlgeschlagen: Kein Projektwissen erreichbar')
  })

  it('says a cancelled child was cancelled', () => {
    expect(childProgressLine({ runStatus: 'cancelled' }, t)).toBe(
      'Unterauftrag · abgebrochen',
    )
  })
})

describe('childProgressLine with parallel tasks', () => {
  it('names the parallel wave instead of one task of many', () => {
    // The regression: a mission starts five tasks in one burst, and the
    // line said "Aufgabe 4" as if the other four did not exist.
    expect(
      childProgressLine(
        {
          runStatus: 'running',
          snapshot: { phase: 'execution' },
          openTasks: [0, 1, 2, 3, 4],
        },
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus · 5 Aufgaben parallel')
  })

  it('names the straggler once its siblings settled', () => {
    // The case a reader most needs: four done, one still running.
    expect(
      childProgressLine(
        {
          runStatus: 'running',
          snapshot: { phase: 'execution' },
          openTasks: [3],
        },
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus · Aufgabe 4')
  })
})

describe('childProgressLine liveness', () => {
  it('carries the count that actually moves', () => {
    // Everything else can stand still for ten minutes while the child
    // works; this cannot. It is what answers "is it still alive".
    expect(
      childProgressLine(
        {
          runStatus: 'running',
          snapshot: { phase: 'execution' },
          openTasks: [5],
          checkedAnswers: 12,
        },
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus · Aufgabe 6 · 12 Belege geprüft')
  })

  it('says nothing about evidence before the first one is verified', () => {
    expect(
      childProgressLine(
        { runStatus: 'running', snapshot: { phase: 'discovery' } },
        t,
      ),
    ).toBe('Unterauftrag · Erkundet den Bestand')
  })

  it('still reports progress when the phase is unknown', () => {
    expect(
      childProgressLine({ runStatus: 'running', checkedAnswers: 3 }, t),
    ).toBe('Unterauftrag · 3 Belege geprüft')
  })
})
