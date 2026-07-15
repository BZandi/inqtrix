import process from 'node:process'
import { pathToFileURL } from 'node:url'

import {
  API_DEGRADATION_LIMIT_PERCENT,
  FatalSocketState,
  RELEASE_MIN_ACK_ROUNDS_PER_WRITER,
  RELEASE_MIN_DURATION_MS,
  RELEASE_OBSERVER_COHORT,
  RawCollaborationClient,
  SessionRotationSupervisor,
  allLoadGatesPassed,
  assertReleasePreflight,
  connectInBatches,
  delay,
  evaluateGates,
  loadFixture,
  measureApiProbe,
  parseArguments,
  performUngracefulRestart,
  prepareSessions,
  reissueSessions,
  resolveApiProbe,
  resolveInstanceProbe,
  resolveRestartControl,
  resolveSessionReissueControl,
  runSustainedWriterLoad,
  summarize,
  summarizeApiProbe,
  verifyObserverCohort,
} from './collaboration-load-lib.mjs'

export async function main(args = process.argv.slice(2), environment = process.env) {
  const options = parseArguments(args, environment)
  if (options.help) {
    printHelp()
    return
  }
  if (!options.fixturePath) {
    throw new Error(
      'No lease/session fixture supplied. Pass --fixture PATH or set INQTRIX_LOAD_SESSION_FIXTURE.',
    )
  }

  const fixture = loadFixture(options.fixturePath)
  const sessions = prepareSessions(fixture, options)
  const apiProbeConfiguration = resolveApiProbe(fixture, options)
  const instanceProbe = resolveInstanceProbe(fixture, options)
  const restartControl = resolveRestartControl(fixture, options, environment)
  const sessionReissueControl = resolveSessionReissueControl(fixture, options, environment)
  if ((restartControl === null) !== (instanceProbe === null)) {
    throw new Error(
      'fixture.restart_control and fixture.instance_probe must be supplied together.',
    )
  }
  assertReleasePreflight(
    options,
    sessions,
    apiProbeConfiguration,
    restartControl,
    instanceProbe,
    sessionReissueControl,
  )

  let baselineApiMeasurement = null
  if (apiProbeConfiguration) {
    progress('Measuring baseline API latency (20 samples)...', options.json)
    baselineApiMeasurement = await measureApiProbe(apiProbeConfiguration)
  }

  const fatal = new FatalSocketState()
  const clients = sessions.map((session, index) => new RawCollaborationClient({
    allowInsecureTls: options.allowInsecureTls,
    index,
    onFatal: (error) => fatal.record(error),
    session,
  }))
  let freshObservers = []
  let clientsClosed = false
  let rotationStopped = false
  let rotationSupervisor = null

  try {
    progress(
      `Connecting ${clients.length} ${options.mode} clients in batches of ${options.connectConcurrency}...`,
      options.json,
    )
    await connectInBatches(
      clients,
      options.connectConcurrency,
      options.connectTimeoutMs,
      fatal,
      (connected, total) => progress(`Connected ${connected}/${total}`, options.json),
    )

    const writers = clients.filter((client) => client.session.access === 'edit')
      .slice(0, options.writers)
    if (writers.length !== options.writers) {
      throw new Error(
        `Fixture provides ${writers.length} edit-capable sessions; ${options.writers} writers are required.`,
      )
    }
    const writerSet = new Set(writers)
    const observers = clients.filter((client) => !writerSet.has(client))
      .slice(0, options.observers)
    if (observers.length !== options.observers) {
      throw new Error(
        `Fixture provides ${observers.length} non-writer observers; ${options.observers} are required.`,
      )
    }

    let connectedRotations = 0
    let scheduledRotations = 0
    if (sessionReissueControl) {
      progress(
        `Rotating leases for all ${clients.length} connected clients through authenticated session reissue...`,
        options.json,
      )
      rotationSupervisor = new SessionRotationSupervisor({
        clients,
        concurrency: options.connectConcurrency,
        control: sessionReissueControl,
        fatal,
        timeoutMs: options.connectTimeoutMs,
      })
      connectedRotations = await rotationSupervisor.rotateNow('connected_rotation')
      rotationSupervisor.start()
    }

    progress(
      apiProbeConfiguration
        ? `Sustaining ${writers.length} writers across ${observers.length} observers while the loaded API probe runs...`
        : `Running ${writers.length} writers with the API gate explicitly skipped...`,
      options.json,
    )
    const load = await runSustainedWriterLoad({
      apiProbe: apiProbeConfiguration,
      fatal,
      minAckRoundsPerWriter: options.minAckRoundsPerWriter,
      minDurationMs: options.minDurationMs,
      observers,
      sampleTimeoutMs: options.sampleTimeoutMs,
      writers,
    })
    const apiProbe = summarizeApiProbe(
      baselineApiMeasurement?.latencies ?? null,
      load.loadedApiLatencies,
    )
    const gates = evaluateGates(
      load.visibleLatencies,
      load.durableLatencies,
      apiProbe,
      options,
      load,
    )

    fatal.throwIfSet()
    await delay(options.postSampleQuietMs)
    fatal.throwIfSet()
    if (rotationSupervisor) {
      const rotations = await rotationSupervisor.stop()
      rotationStopped = true
      connectedRotations = rotations.connected
      scheduledRotations = rotations.scheduled
    }
    let restart = null
    if (restartControl) {
      progress(
        'Ungracefully restarting the collaboration sidecar while original sockets remain open...',
        options.json,
      )
      const exercised = await performUngracefulRestart(
        restartControl,
        observers[0].session.room,
        clients,
        options.connectTimeoutMs,
        instanceProbe,
      )
      restart = {
        afterEpoch: exercised.transition.after.epoch,
        beforeEpoch: exercised.transition.before.epoch,
        closedSockets: exercised.closedSockets,
        epochAdvanced: exercised.transition.after.epoch > exercised.transition.before.epoch,
        instanceIdentityChanged: (
          exercised.transition.after.instanceId !== exercised.transition.before.instanceId
        ),
        kind: exercised.transition.restartKind,
      }
    }
    await Promise.all(clients.map((client) => client.close()))
    clientsClosed = true

    let freshObserverSessions
    try {
      freshObserverSessions = sessionReissueControl
        ? await reissueSessions(
            sessionReissueControl,
            observers.map((observer) => observer.session),
            restart ? 'post_restart_observer' : 'fresh_observer',
          )
        : observers.map((observer) => observer.session)
    } catch (error) {
      fatal.record(error)
      throw error
    }
    freshObservers = freshObserverSessions.map((session, index) => new RawCollaborationClient({
      allowInsecureTls: options.allowInsecureTls,
      index: clients.length + index,
      onFatal: (error) => fatal.record(error),
      session,
    }))
    await connectInBatches(
      freshObservers,
      Math.min(options.connectConcurrency, freshObservers.length),
      options.connectTimeoutMs,
      fatal,
    )
    const reconstruction = verifyObserverCohort(
      freshObservers,
      load.markers,
      load.runId,
    )
    await delay(Math.min(options.postSampleQuietMs, 500))
    fatal.throwIfSet()
    await Promise.all(freshObservers.map((observer) => observer.close()))
    freshObservers = []

    const result = {
      apiProbe,
      connections: clients.length,
      durableAckMs: summarize(load.durableLatencies),
      gates,
      mode: options.mode,
      reconstruction: {
        ...reconstruction,
        restart,
      },
      sessionRotation: {
        connectedClients: connectedRotations,
        freshObservers: sessionReissueControl ? freshObserverSessions.length : 0,
        passed: (
          options.mode !== 'release'
          || (
            connectedRotations === clients.length
            && freshObserverSessions.length === observers.length
          )
        ),
        scheduledClients: scheduledRotations,
      },
      sustainedDurationMs: load.durationMs,
      loadedApiSampleSpanMs: load.loadedApiSampleSpanMs,
      visibleUpdateMs: summarize(load.visibleLatencies),
      writeSamples: load.markers.length,
      writerRounds: {
        maximum: Math.max(...load.roundsPerWriter),
        minimum: Math.min(...load.roundsPerWriter),
      },
      writers: writers.length,
    }
    printResult(result, options.json)
    if (!allLoadGatesPassed(gates, reconstruction, result.sessionRotation)) {
      process.exitCode = 1
    }
  } finally {
    let rotationStopError = null
    if (rotationSupervisor && !rotationStopped) {
      try {
        await rotationSupervisor.stop()
      } catch (error) {
        rotationStopError = error
      }
    }
    if (freshObservers.length > 0) {
      await Promise.allSettled(freshObservers.map((observer) => observer.close()))
    }
    if (!clientsClosed) await Promise.allSettled(clients.map((client) => client.close()))
    if (rotationStopError) throw rotationStopError
  }
}

function printHelp() {
  process.stdout.write('Usage: pnpm load:collaboration:dev -- --fixture PATH [options]\n')
  process.stdout.write('       pnpm load:collaboration:release -- --fixture PATH [--json]\n\n')
  process.stdout.write(`Release: exactly 1000 connections, 100 writers, ${RELEASE_OBSERVER_COHORT} observers, at least ${RELEASE_MIN_DURATION_MS}ms and ${RELEASE_MIN_ACK_ROUNDS_PER_WRITER} acknowledged rounds/writer.\n`)
  process.stdout.write('Release latency: visible p95 <250ms, durable p95 <500ms, FastAPI /health degradation <=20%.\n')
  process.stdout.write('Developer options: --connections N --writers N --observers N --connect-concurrency N\n')
  process.stdout.write('  --connect-timeout-ms N --sample-timeout-ms N --post-sample-quiet-ms N\n')
  process.stdout.write('  --min-duration-ms N --min-ack-rounds N\n')
  process.stdout.write('  --visible-p95-ms N --durable-p95-ms N --allow-insecure-tls\n')
  process.stdout.write('  --skip-api-probe --json\n')
}

function printResult(result, json) {
  if (json) {
    process.stdout.write(`${JSON.stringify(result)}\n`)
    return
  }
  process.stdout.write(`Collaboration load result (${result.mode})\n`)
  process.stdout.write(`  connections: ${result.connections}\n`)
  process.stdout.write(`  writers: ${result.writers}\n`)
  process.stdout.write(`  observers: ${result.reconstruction.observerCount}\n`)
  process.stdout.write(`  write samples: ${result.writeSamples}\n`)
  process.stdout.write(`  sustained: ${formatMs(result.sustainedDurationMs)} gate>=${result.gates.minimumDurationMs}ms ${result.gates.minimumDurationPassed ? 'PASS' : 'FAIL'}\n`)
  process.stdout.write(`  acknowledged rounds/writer: min=${result.writerRounds.minimum} max=${result.writerRounds.maximum} gate>=${result.gates.minimumAckRounds} ${result.gates.minimumAckRoundsPassed ? 'PASS' : 'FAIL'}\n`)
  process.stdout.write(`  visible-update: p50=${formatMs(result.visibleUpdateMs.p50)} p95=${formatMs(result.visibleUpdateMs.p95)} gate<${result.gates.visibleUpdateP95Ms}ms ${result.gates.visibleUpdatePassed ? 'PASS' : 'FAIL'}\n`)
  process.stdout.write(`  durable-ack: p50=${formatMs(result.durableAckMs.p50)} p95=${formatMs(result.durableAckMs.p95)} gate<${result.gates.durableAckP95Ms}ms ${result.gates.durableAckPassed ? 'PASS' : 'FAIL'}\n`)
  if (result.apiProbe.status === 'skipped') {
    process.stdout.write('  api-latency: SKIPPED (explicit developer protocol-smoke opt-out)\n')
  } else {
    process.stdout.write(`  api-latency: baseline-p95=${formatMs(result.apiProbe.baselineP95Ms)} loaded-p95=${formatMs(result.apiProbe.loadedP95Ms)} degradation=${formatPercent(result.apiProbe.degradationPercent)} gate<=${API_DEGRADATION_LIMIT_PERCENT}% ${result.apiProbe.status === 'passed' ? 'PASS' : 'FAIL'}\n`)
    process.stdout.write(`  api-sample-span: ${formatMs(result.loadedApiSampleSpanMs)} gate>=${result.gates.minimumDurationMs}ms ${result.gates.apiSampleSpanPassed ? 'PASS' : 'FAIL'}\n`)
  }
  const restart = result.reconstruction.restart
  process.stdout.write(`  reconstruction: observers=${result.reconstruction.observerCount} expected-per-observer=${result.reconstruction.expectedPerObserver} failed-observers=${result.reconstruction.failedObservers} missing=${result.reconstruction.missing} duplicates=${result.reconstruction.duplicates} unexpected=${result.reconstruction.unexpected} ${result.reconstruction.passed ? 'PASS' : 'FAIL'}\n`)
  process.stdout.write(`  restart: ${restart ? `${restart.kind} sockets=${restart.closedSockets} instance-changed=${restart.instanceIdentityChanged} epoch=${restart.beforeEpoch}->${restart.afterEpoch}` : 'not exercised'} ${restart?.instanceIdentityChanged && restart?.epochAdvanced ? 'PASS' : result.mode === 'release' ? 'FAIL' : 'SKIPPED'}\n`)
  process.stdout.write(`  lease-rotation: connected=${result.sessionRotation.connectedClients} scheduled=${result.sessionRotation.scheduledClients} fresh-observers=${result.sessionRotation.freshObservers} ${result.sessionRotation.passed ? 'PASS' : 'FAIL'}\n`)
}

function progress(message, json) {
  const stream = json ? process.stderr : process.stdout
  stream.write(`${message}\n`)
}

function formatMs(value) {
  return `${value.toFixed(1)}ms`
}

function formatPercent(value) {
  return `${value.toFixed(1)}%`
}

const entryPoint = process.argv[1] ? pathToFileURL(process.argv[1]).href : null
if (entryPoint === import.meta.url) {
  main().catch((error) => {
    const message = error instanceof Error
      ? error.message
      : 'Unknown collaboration load-test failure.'
    process.stderr.write(`ERROR: ${message}\n`)
    process.exitCode = 1
  })
}
