import { describe, expect, it } from 'vitest'
import { createSeedProjectState } from './seedProject'

describe('createSeedProjectState database demo seed', () => {
  it('models EU law as a large grouped collection and vector index', () => {
    const state = createSeedProjectState()
    const legalSection = Object.values(state.fileLibrarySections).find(
      (section) => section.title === 'Rechtliche Grundlagen',
    )
    if (!legalSection) throw new Error('Expected the legal demo collection to exist.')

    const legalAssets = Object.values(state.fileAssets).filter(
      (asset) => asset.sectionId === legalSection.id,
    )
    expect(legalAssets).toHaveLength(50)

    const legalGroups = Object.values(state.fileGroups).filter(
      (group) => group.sectionId === legalSection.id,
    )
    const legalGroupIds = new Set(legalGroups.map((group) => group.id))
    const groupIdsWithFiles = new Set(
      legalAssets
        .filter((asset) => asset.groupId && legalGroupIds.has(asset.groupId))
        .map((asset) => asset.groupId),
    )
    expect(groupIdsWithFiles.size).toBeGreaterThanOrEqual(4)
    expect(legalAssets.some((asset) => asset.groupId === null)).toBe(true)

    const euLawIndex = state.vectorIndexes['vector-index-eu-recht']
    if (!euLawIndex) throw new Error('Expected the EU law vector index to exist.')
    expect(euLawIndex.members.length).toBeGreaterThanOrEqual(50)
    expect(euLawIndex.members.length).toBeLessThanOrEqual(55)

    const assetIds = new Set(Object.keys(state.fileAssets))
    expect(euLawIndex.members.every((member) => assetIds.has(member.fileId))).toBe(true)

    const pendingMembers = euLawIndex.members.filter((member) => member.state === 'pending')
    expect(pendingMembers).toEqual([{ fileId: 'file-asset-rechtsgutachten', state: 'pending' }])
    expect(state.indexingJobs[euLawIndex.id]?.runningFileIds).toEqual(['file-asset-rechtsgutachten'])
  })
})
