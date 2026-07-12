import { describe, expect, it } from 'vitest'
import { agentOverridesFromSelection } from './modelSelection'

describe('agentOverridesFromSelection', () => {
  it('serializes an explicit normal override even when no model is selected', () => {
    expect(agentOverridesFromSelection('normal', null, null, null)).toEqual({
      depth: 'normal',
    })
  })

  it('combines deep with the existing explicit-model precedence', () => {
    expect(
      agentOverridesFromSelection('deep', 'high', 'model-x', 'high'),
    ).toEqual({ depth: 'deep', model: 'model-x', effort: 'high' })
  })
})
