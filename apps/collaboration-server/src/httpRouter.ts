import { Buffer } from 'node:buffer'
import { timingSafeEqual } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'

import type { CollaborationSettings, SidecarLogger } from './contracts'
import {
  CloseCodes,
  CollaborationError,
  collaborationError,
} from './errors'
import { InstanceLeaseManager } from './instanceLease'
import { SidecarMetrics } from './metrics'
import { CollaborationOperations } from './operations'

const DOCUMENT_ROUTE = /^\/internal\/documents\/([A-Za-z0-9][A-Za-z0-9_-]{0,127})\/(decisions|project|suggestions)$/

export class InternalHttpRouter {
  constructor(
    private readonly settings: CollaborationSettings,
    private readonly leaseManager: InstanceLeaseManager,
    private readonly operations: CollaborationOperations,
    private readonly metrics: SidecarMetrics,
    private readonly logger: SidecarLogger,
    private readonly isReady: () => boolean = () => this.leaseManager.isReady(),
  ) {}

  async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const path = new URL(request.url ?? '/', 'http://collaboration.internal').pathname
    if (request.method === 'GET' && path === '/health/live') {
      this.json(response, 200, { status: 'alive' })
      return
    }
    if (request.method === 'GET' && path === '/health/ready') {
      const ready = this.isReady()
      this.json(response, ready ? 200 : 503, {
        mode: 'single_replica',
        protocol_version: this.settings.protocolVersion,
        schema_version: this.settings.schemaVersion,
        status: ready ? 'ready' : 'not_ready',
      })
      return
    }
    if (request.method === 'GET' && path === '/metrics') {
      response.writeHead(200, {
        'Cache-Control': 'no-store',
        'Content-Type': 'text/plain; version=0.0.4; charset=utf-8',
      })
      response.end(this.metrics.render())
      return
    }

    if (!hasBearerSecret(request.headers.authorization, this.settings.secret)) {
      this.metrics.increment('inqtrix_collaboration_http_rejections_total', { reason: 'unauthorized' })
      this.json(response, 401, { error: { reason: 'unauthorized' } })
      return
    }
    if (!this.isReady()) {
      this.metrics.increment('inqtrix_collaboration_http_rejections_total', {
        reason: 'service_unavailable',
      })
      this.json(response, 503, { error: { reason: 'service_unavailable' } })
      return
    }
    if (request.method !== 'POST') {
      this.json(response, 405, { error: { reason: 'method_not_allowed' } })
      return
    }

    const operation = path === '/internal/convert'
      ? 'convert'
      : DOCUMENT_ROUTE.test(path)
        ? path.endsWith('/decisions')
          ? 'decisions'
          : path.endsWith('/suggestions')
            ? 'suggestions'
            : 'project'
        : 'unknown'
    if (operation === 'unknown') {
      this.json(response, 404, { error: { reason: 'not_found' } })
      return
    }

    const startedAt = performance.now()
    try {
      const payload = await this.readJson(request)
      const route = DOCUMENT_ROUTE.exec(path)
      const result = operation === 'convert'
        ? await this.operations.convert(payload)
        : operation === 'decisions'
          ? await this.operations.decide(route![1]!, payload)
          : operation === 'suggestions'
            ? await this.operations.publishSuggestion(route![1]!, payload)
            : await this.operations.project(route![1]!, payload)
      this.metrics.increment('inqtrix_collaboration_http_requests_total', {
        operation,
        status: 'success',
      })
      this.metrics.observeMilliseconds(
        'inqtrix_collaboration_http_request_seconds',
        performance.now() - startedAt,
      )
      this.json(response, 200, result)
    } catch (error) {
      const mapped = collaborationError(error)
      this.metrics.increment('inqtrix_collaboration_http_requests_total', {
        operation,
        status: mapped.reason,
      })
      // Der urspruengliche Grund muss Log UND Antwort erreichen, sonst
      // kann weder Betreiber noch Nutzer feststellen, warum abgelehnt wurde.
      const upstream = mapped.upstreamReason && mapped.upstreamReason !== mapped.reason
        ? { upstream_reason: mapped.upstreamReason }
        : {}
      // Die Korrelations-Id des Aufrufers, damit eine Ablehnung hier und die
      // Zeile im API-Log nachweislich DASSELBE Ereignis sind statt nur
      // ungefaehr dieselbe Sekunde. Fehlt sie, bleibt das Feld weg -- eine
      // leere Id waere schlimmer als keine, weil sie zusammenfuehrt, was
      // nicht zusammengehoert.
      const correlation = correlationField(request)
      if (mapped.httpStatus >= 500) {
        this.logger.error('internal_http_operation_failed', {
          operation,
          reason: mapped.reason,
          ...correlation,
          ...upstream,
        })
      } else {
        this.logger.warn('internal_http_operation_rejected', {
          operation,
          reason: mapped.reason,
          ...correlation,
          ...upstream,
        })
      }
      this.json(response, mapped.httpStatus, {
        error: { reason: mapped.reason, ...upstream },
      })
    }
  }

  private async readJson(request: IncomingMessage): Promise<unknown> {
    const contentType = request.headers['content-type']?.split(';', 1)[0]?.trim().toLowerCase()
    if (contentType !== 'application/json') {
      throw invalidRequest()
    }
    const maximum = this.settings.documentLimitBytes * 2 + 64 * 1024
    const declared = Number(request.headers['content-length'] ?? 0)
    if (Number.isFinite(declared) && declared > maximum) {
      throw requestTooLarge()
    }
    const chunks: Buffer[] = []
    let length = 0
    for await (const chunk of request) {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      length += buffer.length
      if (length > maximum) {
        throw requestTooLarge()
      }
      chunks.push(buffer)
    }
    try {
      return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown
    } catch {
      throw invalidRequest()
    }
  }

  private json(response: ServerResponse, status: number, payload: unknown): void {
    response.writeHead(status, {
      'Cache-Control': 'no-store',
      'Content-Type': 'application/json; charset=utf-8',
    })
    response.end(JSON.stringify(payload))
  }
}

export function hasBearerSecret(
  authorization: string | string[] | undefined,
  secret: string,
): boolean {
  if (typeof authorization !== 'string') return false
  const actualBuffer = Buffer.from(authorization, 'utf8')
  const expectedBuffer = Buffer.from(`Bearer ${secret}`, 'utf8')
  return actualBuffer.length === expectedBuffer.length
    && timingSafeEqual(actualBuffer, expectedBuffer)
}

function invalidRequest(): CollaborationError {
  return new CollaborationError('invalid_request', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 400,
  })
}

function requestTooLarge(): CollaborationError {
  return new CollaborationError('document_too_large', {
    closeCode: CloseCodes.messageTooLarge,
    httpStatus: 413,
  })
}

/** Die Korrelations-Id des Aufrufers als Logfeld -- oder gar nichts.
 *
 * Ohne mitgereichte Id lassen sich die Logzeilen eines Klicks nur ueber
 * Zeitstempel raten, und bei zwei gleichzeitigen Nutzern gar nicht mehr.
 * Die API setzt den Kopf seit `_correlation_headers`; hier wird er gelesen.
 *
 * Verbunden werden damit ZWEI Stationen, Gateway und Sidecar. Der Rueckweg
 * dieses Dienstes zur internen API traegt die Id NICHT weiter -- siehe den
 * Docstring auf der Python-Seite. Die Luecke ist benannt, nicht uebersehen.
 *
 * Ein fehlender oder leerer Kopf liefert KEIN Feld. Eine leere Id waere
 * schlimmer als keine: sie sammelte alle unkorrelierten Zeilen unter
 * demselben Wert und behauptete damit einen Zusammenhang. */
export function correlationField(
  request: IncomingMessage,
): { request_id: string } | Record<string, never> {
  const raw = request.headers['x-request-id']
  // Node verwirft doppelte Koepfe dieses Namens NICHT, sondern fuegt sie mit
  // ", " zu EINEM String zusammen -- ein Array kommt hier nie an. Haengt ein
  // Ingress seine eigene Id an die der API, entstuende sonst ein Wert wie
  // "7f3a…c1, mesh-9942", der gegen KEIN Log greppt: die Korrelation waere
  // still falsch statt schlicht abwesend. Massgeblich ist der erste Wert,
  // denn der stammt vom urspruenglichen Aufrufer.
  const joined = Array.isArray(raw) ? raw[0] : raw
  const first = typeof joined === 'string' ? joined.split(',', 1)[0] : undefined
  const trimmed = first?.trim() ?? ''
  // Gedeckelt, weil ein Kopf aus dem Netz kommt und ungebremst ins Log
  // liefe; die Id des Gateways ist ein 32-stelliger Hex-String.
  return trimmed ? { request_id: trimmed.slice(0, 128) } : {}
}
