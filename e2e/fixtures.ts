import {
  devices,
  test as base,
  type BrowserContext,
  type Page,
} from '@playwright/test'

import {
  collaborationE2EConfiguration,
  type CollaborationE2EStack,
  type CollaborationTransport,
} from './config'

export type CollaborationHarness = {
  collaboratorContext: BrowserContext
  collaboratorPage: Page
  ownerContext: BrowserContext
  ownerPage: Page
  stack: CollaborationE2EStack
  transport: CollaborationTransport
}

type Fixtures = {
  collaboration: CollaborationHarness | null
  configuredStack: {
    stack: CollaborationE2EStack
    transport: CollaborationTransport
  } | null
}

const reportedConfigurationSkips = new Set<string>()

export const test = base.extend<Fixtures>({
  configuredStack: async ({}, use, testInfo) => {
    const transport = testInfo.project.metadata.transport as CollaborationTransport
    const stack = collaborationE2EConfiguration.stack
    const reasons = [
      ...collaborationE2EConfiguration.reasons,
      ...(stack?.transports[transport]?.reasons ?? []),
    ]
    if (!stack || reasons.length > 0) {
      const reason = `External collaboration stack unavailable for ${transport}: ${reasons.join('; ')}`
      if (!reportedConfigurationSkips.has(testInfo.project.name)) {
        reportedConfigurationSkips.add(testInfo.project.name)
        process.stdout.write(`[SKIP ${testInfo.project.name}] ${reason}\n`)
      }
      testInfo.skip(
        true,
        reason,
      )
      await use(null)
      return
    }
    await use({ stack, transport })
  },
  collaboration: async ({ configuredStack, browser }, use, testInfo) => {
    if (!configuredStack) {
      await use(null)
      return
    }
    const { stack, transport } = configuredStack
    const formFactor = testInfo.project.metadata.formFactor as 'desktop' | 'mobile'

    const endpoint = stack.transports[transport]
    const descriptor = formFactor === 'desktop' ? devices['Desktop Chrome'] : devices['Pixel 7']
    const { defaultBrowserType: _defaultBrowserType, ...deviceOptions } = descriptor
    const ownerContext = await browser.newContext({
      ...deviceOptions,
      baseURL: endpoint.baseURL!,
      storageState: endpoint.ownerStorageState,
    })
    const collaboratorContext = await browser.newContext({
      ...deviceOptions,
      baseURL: endpoint.baseURL!,
      storageState: endpoint.collaboratorStorageState,
    })

    try {
      await use({
        collaboratorContext,
        collaboratorPage: await collaboratorContext.newPage(),
        ownerContext,
        ownerPage: await ownerContext.newPage(),
        stack,
        transport,
      })
    } finally {
      await Promise.allSettled([ownerContext.close(), collaboratorContext.close()])
    }
  },
})

export { expect } from '@playwright/test'
