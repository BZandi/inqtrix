import { describe, expect, it } from 'vitest'

import { ApiRequestError, collaborationError } from '../src/errors.js'

describe('collaborationError: der Grund gehoert dem, der ihn erzeugt hat', () => {
  // Der Sidecar bildete GENAU DREI der rund vierzig Gruende ab, die das
  // interne API senden kann, und machte aus allem uebrigen
  // "update_required" -- eine Behauptung ueber einen veralteten Client.
  // Damit war der echte Grund vernichtet: er stand danach weder im
  // Sidecar-Log noch in der Antwort an den Nutzer, und der Betreiber
  // konnte nicht mehr feststellen, warum abgelehnt wurde.
  it('traegt einen unbekannten 409-Grund weiter, statt Veraltung zu behaupten', () => {
    const fehler = collaborationError(new ApiRequestError(409, 'instance_fenced'))

    expect(fehler.upstreamReason).toBe('instance_fenced')
    expect(fehler.reason).not.toBe('update_required')
  })

  it('erfindet auch fuer einen kuenftigen, hier unbekannten Grund nichts', () => {
    const fehler = collaborationError(new ApiRequestError(409, 'patch_decision_incomplete'))

    expect(fehler.upstreamReason).toBe('patch_decision_incomplete')
    expect(fehler.reason).not.toBe('update_required')
  })

  it('behaelt die bekannten Zuordnungen bei', () => {
    // Gegenprobe: die drei bisher abgebildeten Gruende duerfen sich nicht
    // aendern -- sie steuern den WebSocket-Schliesscode.
    expect(collaborationError(new ApiRequestError(409, 'sequence_conflict')).reason)
      .toBe('sequence_conflict')
    expect(collaborationError(new ApiRequestError(409, 'command_conflict')).reason)
      .toBe('sequence_conflict')
    expect(collaborationError(new ApiRequestError(409, 'generation_mismatch')).reason)
      .toBe('generation_mismatch')
  })

  it('nennt update_required nur, wenn das interne API es selbst sagt', () => {
    const fehler = collaborationError(new ApiRequestError(409, 'update_required'))
    expect(fehler.reason).toBe('update_required')
    expect(fehler.upstreamReason).toBe('update_required')
  })

  // Fuer 400 gab es ueberhaupt keinen Zweig. Eine abgelehnte Anfrage fiel
  // damit in den Auffangzweig und hiess ab da "internal_consistency" --
  // ein Raum-Zustand, den es nie gab. Der Sidecar schloss die Verbindung
  // mit 1011, der Browser verband neu, und das wiederholte sich endlos,
  // ohne dass irgendwo der wirkliche Ablehnungsgrund stand.
  it('behaelt den Grund einer 400-Ablehnung, statt ihn zu verschlucken', () => {
    const fehler = collaborationError(new ApiRequestError(400, 'update_hash_mismatch'))

    expect(fehler.upstreamReason).toBe('update_hash_mismatch')
  })

  it('behaelt den Grund auch bei einem sonst unbehandelten Status', () => {
    const fehler = collaborationError(new ApiRequestError(418, 'teapot'))

    expect(fehler.upstreamReason).toBe('teapot')
  })
})
