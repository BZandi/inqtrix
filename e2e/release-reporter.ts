import type {
  FullConfig,
  FullResult,
  Reporter,
  Suite,
  TestCase,
  TestResult,
} from '@playwright/test/reporter'

import { COLLABORATION_TRANSPORTS } from './config.ts'
import {
  RELEASE_DESKTOP_SCENARIOS,
  RELEASE_MOBILE_SCENARIOS,
} from './release-contract.ts'

export default class CollaborationReleaseReporter implements Reporter {
  private preflightFailures: string[] = []
  private skippedTests: string[] = []

  onBegin(_config: FullConfig, suite: Suite): void {
    const testsByProject = new Map<string, TestCase[]>()
    for (const test of suite.allTests()) {
      const projectName = test.parent.project()?.name
      if (!projectName) continue
      const tests = testsByProject.get(projectName) ?? []
      tests.push(test)
      testsByProject.set(projectName, tests)
    }

    for (const transport of COLLABORATION_TRANSPORTS) {
      for (const formFactor of ['desktop', 'mobile'] as const) {
        const projectName = `${transport}-${formFactor}`
        const tests = testsByProject.get(projectName) ?? []
        if (tests.length === 0) {
          this.preflightFailures.push(`${projectName} selected no tests`)
          continue
        }
        const required = formFactor === 'mobile'
          ? RELEASE_MOBILE_SCENARIOS
          : RELEASE_DESKTOP_SCENARIOS
        for (const tag of required) {
          if (!tests.some((test) => test.tags.includes(tag))) {
            this.preflightFailures.push(`${projectName} is missing required scenario ${tag}`)
          }
        }
      }
    }
  }

  onTestEnd(test: TestCase, result: TestResult): void {
    if (result.status !== 'skipped') return
    const projectName = test.parent.project()?.name ?? 'unknown-project'
    this.skippedTests.push(`${projectName}: ${test.title}`)
  }

  async onEnd(
    result: FullResult,
  ): Promise<{ status?: FullResult['status'] } | undefined> {
    const failures = [
      ...this.preflightFailures,
      ...this.skippedTests.map((test) => `required test skipped: ${test}`),
    ]
    if (failures.length === 0) return undefined
    process.stderr.write(
      `Collaboration release E2E gate failed:\n- ${failures.join('\n- ')}\n`,
    )
    if (result.status === 'passed') return { status: 'failed' }
    return undefined
  }
}
