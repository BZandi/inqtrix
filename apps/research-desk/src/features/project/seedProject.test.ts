import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { describe, expect, it } from 'vitest'
import { createSeedProjectState } from './seedProject'

type MarkdownTreeNode = {
  children?: MarkdownTreeNode[]
  type?: string
}

function parsedTableWidths(markdown: string): number[][] {
  const tree = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .parse(markdown) as MarkdownTreeNode
  const pending = [tree]
  const tableWidths: number[][] = []

  while (pending.length > 0) {
    const node = pending.pop()
    if (!node?.children) continue

    if (node.type === 'table') {
      tableWidths.push(node.children.map((row) => row.children?.length ?? 0))
    }
    pending.push(...node.children)
  }

  return tableWidths
}

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

describe('createSeedProjectState chat demo seed', () => {
  it('loads the long battery digital-twin renderer reference as the newest chat', () => {
    const state = createSeedProjectState()
    const thread = Object.values(state.chatThreads).find(
      (candidate) => candidate.title === 'Digitaler Zwilling eines Batteriespeichers',
    )
    if (!thread) throw new Error('Expected the battery digital-twin demo chat to exist.')

    expect(thread.messages).toHaveLength(6)
    expect(state.chatThreadGroupMemberships[thread.id]).toBe('chat-group-demos')
    expect(state.ui.selectedChatThreadId).toBe(thread.id)
    expect(state.ui.pinnedExplorer.chatThreadIds).toEqual([thread.id])

    const assistantMarkdown = thread.messages
      .filter((message) => message.role === 'assistant')
      .map((message) => message.contentMarkdown)
      .join('\n')
    const wordCount = assistantMarkdown.trim().split(/\s+/).length
    const tableSeparators = assistantMarkdown.match(/^\|(?:\s*:?-+:?\s*\|){2,}$/gm) ?? []
    const tableWidths = parsedTableWidths(assistantMarkdown)

    expect(wordCount).toBeGreaterThanOrEqual(2400)
    expect(assistantMarkdown.match(/```python\b/g) ?? []).toHaveLength(2)
    expect(assistantMarkdown.match(/```mermaid\b/g) ?? []).toHaveLength(3)
    expect((assistantMarkdown.match(/\$\$/g) ?? []).length).toBeGreaterThanOrEqual(24)
    expect(tableSeparators.length).toBeGreaterThanOrEqual(5)
    expect(tableWidths.length).toBeGreaterThanOrEqual(5)
    expect(tableWidths.every((rows) => rows.every((width) => width === rows[0]))).toBe(true)
  })
})
