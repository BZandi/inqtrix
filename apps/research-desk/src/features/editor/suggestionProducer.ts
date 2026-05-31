import { instructEditorDocument, suggestEditorBlock, type EditorInstructEdit } from '@/api/inqtrixClient'
import type { ReferenceDoc } from '@/features/files/referenceBlocks'
import type { ChatModelTier } from '@/features/researchRuns/types'
import type {
  EditorCommentAnchorRecord,
  EditorEvidencePreset,
  EditorSuggestionEvidence,
  EditorSuggestionOrigin,
  EditorSuggestionRecord,
} from '@/features/project/types'

export type SuggestionInput = {
  anchor: EditorCommentAnchorRecord
  attachments?: ReferenceDoc[]
  documentId: string
  documentMarkdown: string
  globalInstruction?: string
  instruction: string
  modelTier: ChatModelTier | null
  origin: EditorSuggestionOrigin
  originalMarkdown?: string
  originalText: string
  signal?: AbortSignal
  snippet?: string
}

export type SuggestionProposal = {
  changeSummary?: string[]
  evidence?: EditorSuggestionEvidence
  proposedText: string
  warnings?: string[]
}

export type InstructionInput = {
  attachments?: ReferenceDoc[]
  documentMarkdown: string
  instruction: string
  modelTier: ChatModelTier | null
  signal?: AbortSignal
  snippet?: string
}

export type InstructionProposal = {
  assistantMessage: string
  edits: EditorInstructEdit[]
  warnings?: string[]
}

export type RefinementInput = {
  attachments?: ReferenceDoc[]
  documentMarkdown: string
  instruction: string
  modelTier: ChatModelTier | null
  originalInstruction?: string
  signal?: AbortSignal
  suggestion: EditorSuggestionRecord
}

/**
 * Boundary for turning an anchored passage plus an instruction into a proposed
 * rewrite. Implementations either mock the result or call the backend; the
 * editor UI only depends on this interface so the whole
 * suggestion/diff/accept-reject flow stays testable without a backend.
 */
export interface SuggestionProducer {
  refine(input: RefinementInput): Promise<SuggestionProposal>
  produceInstruction(input: InstructionInput): Promise<InstructionProposal>
  produce(input: SuggestionInput): Promise<SuggestionProposal>
}

export type LlmSuggestionProducerConfig = {
  apiKey?: string
  locale: 'de' | 'en'
  stack: string
  workspaceId?: string
}

/**
 * Calls the backend `/v1/editor/suggest` endpoint to rewrite one paragraph,
 * using the document as read-only context and the composer-selected model tier.
 * Evidence-review (Beleg) needs the search backend, which is not wired yet, so
 * that kind is delegated to the deterministic mock until it lands.
 */
export class LlmSuggestionProducer implements SuggestionProducer {
  private readonly config: LlmSuggestionProducerConfig
  private readonly fallback = new MockSuggestionProducer()

  constructor(config: LlmSuggestionProducerConfig) {
    this.config = config
  }

  async produce(input: SuggestionInput): Promise<SuggestionProposal> {
    if (input.origin.kind === 'evidence_review') {
      return this.fallback.produce(input)
    }
    const response = await suggestEditorBlock(
      {
        attachments: input.attachments,
        background: input.documentMarkdown,
        blockMarkdown: input.originalMarkdown,
        blockText: input.originalText,
        globalInstruction: input.globalInstruction,
        instruction: input.instruction,
        locale: this.config.locale,
        modelTier: input.modelTier,
        snippet: input.snippet,
        stack: this.config.stack,
      },
      {
        apiKey: this.config.apiKey,
        signal: input.signal,
        workspaceId: this.config.workspaceId,
      },
    )
    return {
      changeSummary: response.change_summary,
      proposedText: response.improved_text,
      warnings: response.warnings,
    }
  }

  async refine(input: RefinementInput): Promise<SuggestionProposal> {
    const response = await suggestEditorBlock(
      {
        attachments: input.attachments,
        background: input.documentMarkdown,
        blockMarkdown: input.suggestion.originalMarkdown,
        blockText: input.suggestion.originalText,
        currentSuggestionMarkdown: input.suggestion.proposedText,
        instruction: input.originalInstruction,
        locale: this.config.locale,
        modelTier: input.modelTier,
        refinementInstruction: input.instruction,
        stack: this.config.stack,
      },
      {
        apiKey: this.config.apiKey,
        signal: input.signal,
        workspaceId: this.config.workspaceId,
      },
    )
    return {
      changeSummary: response.change_summary,
      proposedText: response.improved_text,
      warnings: response.warnings,
    }
  }

  async produceInstruction(input: InstructionInput): Promise<InstructionProposal> {
    const response = await instructEditorDocument(
      {
        attachments: input.attachments,
        documentMarkdown: input.documentMarkdown,
        instruction: input.snippet
          ? `${input.instruction}\n\n${input.snippet}`
          : input.instruction,
        locale: this.config.locale,
        modelTier: input.modelTier,
        stack: this.config.stack,
      },
      {
        apiKey: this.config.apiKey,
        signal: input.signal,
        workspaceId: this.config.workspaceId,
      },
    )
    return {
      assistantMessage: response.assistant_message,
      edits: response.edits,
      warnings: response.warnings,
    }
  }
}

/**
 * Deterministic stand-in used until the model is wired up. It applies a small,
 * visible transformation so the diff renderer shows real insertions and
 * deletions, and attaches mock sources for evidence-review runs.
 */
export class MockSuggestionProducer implements SuggestionProducer {
  async refine(input: RefinementInput): Promise<SuggestionProposal> {
    const proposed = input.suggestion.proposedText.trim()
    return {
      changeSummary: ['Lokale Beispiel-Revision.'],
      proposedText: `${proposed}\n\n${input.instruction}`,
    }
  }

  async produceInstruction(input: InstructionInput): Promise<InstructionProposal> {
    const trimmed = input.documentMarkdown.trim()
    if (!trimmed) {
      return {
        assistantMessage: 'Ein neuer Dokumententwurf wurde vorbereitet.',
        edits: [{
          find: '',
          note: 'Neuen Inhalt eingefügt.',
          position: 'append',
          quote_after: '',
          quote_before: '',
          text: `# Neuer Entwurf\n\n${input.instruction}`,
        }],
      }
    }
    const firstParagraph = trimmed.split(/\n{2,}/)[0] ?? trimmed
    return {
      assistantMessage: 'Eine lokale Beispieländerung wurde vorbereitet.',
      edits: [{
        find: firstParagraph,
        note: 'Beispielhaft gekürzt.',
        position: 'replace',
        quote_after: '',
        quote_before: '',
        text: `${firstParagraph} (überarbeitet)`,
      }],
    }
  }

  async produce(input: SuggestionInput): Promise<SuggestionProposal> {
    const original = input.originalText.trim()
    if (input.origin.kind === 'evidence_review') {
      return {
        evidence: { mode: input.origin.preset, sources: mockSourcesForPreset(input.origin.preset) },
        proposedText: `${original} [Beleg: example.org]`,
      }
    }
    return { proposedText: mockRewrite(original) }
  }
}

function mockRewrite(text: string): string {
  if (!text) return text
  const replaced = text.replace(/\b(\p{L}{4,})\b/u, (word) => `präzise ${word}`)
  return `${replaced} (überarbeitet)`
}

function mockSourcesForPreset(preset: EditorEvidencePreset): EditorSuggestionEvidence['sources'] {
  if (preset === 'fact_check') {
    return [{ title: 'Faktencheck-Quelle (Mock)', url: 'https://example.org/factcheck' }]
  }
  if (preset === 'verify_citations') {
    return [{ title: 'Zitationsabgleich (Mock)', url: 'https://example.org/citation' }]
  }
  return [
    { title: 'Primaerquelle (Mock)', url: 'https://example.org/primary' },
    { title: 'Zweitquelle (Mock)', url: 'https://example.org/secondary' },
  ]
}
