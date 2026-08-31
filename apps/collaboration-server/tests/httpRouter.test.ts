import type { IncomingMessage } from 'node:http'

import { correlationField } from '../src/httpRouter'

function anfrage(headers: IncomingMessage['headers']): IncomingMessage {
  return { headers } as IncomingMessage
}

describe('correlationField: eine Ablehnung muss sich einem Klick zuordnen lassen', () => {
  // Ein Klick erzeugt Logzeilen an drei Stellen -- Gateway, Sidecar, interne
  // API. Ohne mitgereichte Id lassen sie sich nur ueber Zeitstempel raten,
  // und bei zwei gleichzeitigen Nutzern gar nicht mehr. Genau daran hat die
  // Suche nach dem 400 gestern eine Stunde gekostet.
  it('reicht die Id des Aufrufers als Logfeld weiter', () => {
    expect(correlationField(anfrage({ 'x-request-id': 'abc123' })))
      .toEqual({ request_id: 'abc123' })
  })

  // Eine LEERE Id ist schlimmer als keine: sie sammelt alle unkorrelierten
  // Zeilen unter demselben Wert und behauptet damit einen Zusammenhang, den
  // es nicht gibt.
  it('liefert lieber gar nichts als eine leere Id', () => {
    expect(correlationField(anfrage({}))).toEqual({})
    expect(correlationField(anfrage({ 'x-request-id': '' }))).toEqual({})
    expect(correlationField(anfrage({ 'x-request-id': '   ' }))).toEqual({})
  })

  it('deckelt die Laenge, weil der Kopf aus dem Netz kommt', () => {
    const feld = correlationField(anfrage({ 'x-request-id': 'a'.repeat(500) }))

    expect('request_id' in feld).toBe(true)
    expect((feld as { request_id: string }).request_id).toHaveLength(128)
  })

  // Node verwirft doppelte Koepfe dieses Namens nicht, sondern fuegt sie mit
  // ", " zu EINEM String zusammen -- an einem echten node:http-Server
  // gemessen. Ein von Hand gebautes Array haette eine Form geprueft, die HTTP
  // nie erzeugt, und der Test waere gruen geblieben, waehrend die Wirklichkeit
  // das Gegenteil tat.
  it('nimmt bei zusammengefuegten Koepfen den ersten Wert', () => {
    expect(correlationField(anfrage({ 'x-request-id': '7f3ac1, mesh-9942' })))
      .toEqual({ request_id: '7f3ac1' })
  })

  it('vertraegt weiterhin die Array-Form, falls ein Server sie doch liefert', () => {
    expect(correlationField(anfrage({ 'x-request-id': ['erste', 'zweite'] })))
      .toEqual({ request_id: 'erste' })
  })
})
