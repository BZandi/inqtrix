import { describe, expect, it } from 'vitest'

import { appMotion } from './transitions'

describe('appMotion vocabulary', () => {
  it('runs the view entry at desktop tempo on the shared curve', () => {
    // 200ms is the cross-system desktop standard for a view switch
    // (Material desktop 150–200ms, Fluent normal 200ms, Carbon 150–240ms);
    // panels stay slower because a size change reads differently.
    expect(appMotion.view.duration).toBe(0.2)
    expect(appMotion.view.duration).toBeLessThan(appMotion.panel.duration)
    expect(appMotion.view.ease).toEqual(appMotion.panel.ease)
  })
})
