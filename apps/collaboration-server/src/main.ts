import { randomUUID } from 'node:crypto'

import { settingsFromEnv } from './config'
import { createJsonLogger } from './logger'
import { CollaborationSidecar } from './sidecar'

async function main(): Promise<void> {
  const logger = createJsonLogger()
  const settings = settingsFromEnv(process.env, randomUUID())
  const sidecar = new CollaborationSidecar(settings, { logger })
  let closing: Promise<void> | null = null

  const stop = (signal: NodeJS.Signals): void => {
    if (closing) return
    logger.info('shutdown_requested', { signal })
    closing = sidecar.stop().catch(() => {
      logger.error('shutdown_failed')
      process.exitCode = 1
    })
  }

  process.once('SIGINT', stop)
  process.once('SIGTERM', stop)
  await sidecar.start()
  if (closing) await closing
}

main().catch(() => {
  createJsonLogger().error('collaboration_sidecar_start_failed')
  process.exitCode = 1
})
