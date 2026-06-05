import { phaseOrder } from '@/features/researchDesk/types'
import { createDefaultFileLibrarySections, FILE_SECTION_LIBRARY_ID } from '@/features/files/sections'
import { reportReferencesFromMarkdown } from './reportReferences'
import type {
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorFolderRecord,
  EditorUiState,
  ChatRuleRecord,
  ChatThreadGroupRecord,
  ChatThreadRecord,
  FileAssetRecord,
  FileGroupRecord,
  ProjectState,
  ResearchRunRecord,
} from './types'
import { PROJECT_SCHEMA_VERSION } from './types'
import { getOrCreateBrowserWorkspaceId } from './workspaceId'

const seedCreatedAt = '2026-05-15T06:00:00.000Z'

const emptyEditorUi: EditorUiState = {
  activeDocumentId: null,
  assistantDraft: '',
  isAssistantVisible: true,
  isCommentPanelVisible: true,
  isDiffVisible: false,
  isTreeVisible: true,
  openDocumentIds: [],
  panelTab: 'comments',
  selectedCommentId: null,
  viewMode: 'live',
}

export function createEmptyProjectState(): ProjectState {
  const now = new Date().toISOString()
  const defaultSections = createDefaultFileLibrarySections(now)

  return {
    chatRuleOrder: [],
    chatRules: {},
    chatThreadGroupMemberships: {},
    chatThreadGroupOrder: [],
    chatThreadGroups: {},
    chatThreadOrder: [],
    chatThreads: {},
    connection: {
      kind: 'local',
      writable: false,
    },
    dirty: false,
    editorComments: {},
    editorDocumentOrder: [],
    editorDocuments: {},
    editorFolderOrder: [],
    editorFolders: {},
    editorSuggestionGroups: {},
    editorSuggestions: {},
    editorUi: emptyEditorUi,
    fileAssetOrder: [],
    fileAssets: {},
    fileGroupOrder: [],
    fileGroups: {},
    fileLibrarySectionOrder: defaultSections.map((section) => section.id),
    fileLibrarySections: Object.fromEntries(defaultSections.map((section) => [section.id, section])),
    localRunCounter: 1,
    preferences: {
      contrastMode: 'standard',
      locale: 'de',
      theme: 'system',
      themePreset: 'standard',
    },
    project: {
      createdAt: now,
      name: 'Untitled Inqtrix Project',
      schemaVersion: PROJECT_SCHEMA_VERSION,
      updatedAt: now,
    },
    researchRunOrder: [],
    researchRuns: {},
    workspaceId: getOrCreateBrowserWorkspaceId(),
    ui: {
      activeFilter: 'all',
      activeView: 'research',
      chatChainingEnabled: false,
      expandedJobId: null,
      isChatHistoryVisible: true,
      isComposerVisible: true,
      isReportExpanded: false,
      isReportVisible: true,
      pendingChatAttachmentRefs: [],
      pendingChatReportRunId: null,
      selectedChatModelTier: null,
      selectedChatThreadId: null,
      selectedJobId: null,
      selectedStack: 'anthropic_perplexity',
    },
  }
}

export function createSeedProjectState(): ProjectState {
  const researchRuns = seedResearchRuns()
  const researchRunOrder = researchRuns.map((run) => run.runId)
  const chatThreads = seedChatThreads()
  const chatThreadOrder = chatThreads.map((thread) => thread.id)
  const chatThreadGroups = seedChatThreadGroups()
  const chatThreadGroupOrder = chatThreadGroups.map((group) => group.id)
  const chatRules = seedChatRules()
  const chatRuleOrder = chatRules.map((rule) => rule.id)
  const editorFolders = seedEditorFolders()
  const editorFolderOrder = editorFolders.map((folder) => folder.id)
  const editorDocuments = seedEditorDocuments()
  const editorDocumentOrder = editorDocuments.map((document) => document.id)
  const editorComments = seedEditorComments()
  const openDocumentIds = editorDocumentOrder.slice(0, 2)
  const fileLibrarySections = createDefaultFileLibrarySections(seedCreatedAt)
  const fileGroups = seedFileGroups()
  const fileAssets = seedFileAssets()

  return {
    chatRuleOrder,
    chatRules: Object.fromEntries(chatRules.map((rule) => [rule.id, rule])),
    chatThreadGroupMemberships: {
      'chat-briefing': 'chat-group-policy',
      'chat-method': 'chat-group-policy',
      'chat-vendors': 'chat-group-vendors',
    },
    chatThreadGroupOrder,
    chatThreadGroups: Object.fromEntries(chatThreadGroups.map((group) => [group.id, group])),
    chatThreadOrder,
    chatThreads: Object.fromEntries(chatThreads.map((thread) => [thread.id, thread])),
    connection: {
      kind: 'demo',
      writable: false,
    },
    dirty: false,
    editorComments: Object.fromEntries(editorComments.map((comment) => [comment.id, comment])),
    editorDocumentOrder,
    editorDocuments: Object.fromEntries(editorDocuments.map((document) => [document.id, document])),
    editorFolderOrder,
    editorFolders: Object.fromEntries(editorFolders.map((folder) => [folder.id, folder])),
    editorSuggestionGroups: {},
    editorSuggestions: {},
    editorUi: {
      ...emptyEditorUi,
      activeDocumentId: openDocumentIds[0] ?? null,
      openDocumentIds,
    },
    fileAssetOrder: fileAssets.map((asset) => asset.id),
    fileAssets: Object.fromEntries(fileAssets.map((asset) => [asset.id, asset])),
    fileGroupOrder: fileGroups.map((group) => group.id),
    fileGroups: Object.fromEntries(fileGroups.map((group) => [group.id, group])),
    fileLibrarySectionOrder: fileLibrarySections.map((section) => section.id),
    fileLibrarySections: Object.fromEntries(fileLibrarySections.map((section) => [section.id, section])),
    localRunCounter: 248,
    preferences: {
      contrastMode: 'standard',
      locale: 'de',
      theme: 'dark',
      themePreset: 'standard',
    },
    project: {
      createdAt: seedCreatedAt,
      name: 'Inqtrix Research Desk Demo',
      schemaVersion: PROJECT_SCHEMA_VERSION,
      updatedAt: seedCreatedAt,
    },
    researchRunOrder,
    researchRuns: Object.fromEntries(researchRuns.map((run) => [run.runId, run])),
    workspaceId: 'ws_demo_research_desk',
    ui: {
      activeFilter: 'all',
      activeView: 'research',
      chatChainingEnabled: false,
      expandedJobId: researchRunOrder[0] ?? null,
      isChatHistoryVisible: true,
      isComposerVisible: true,
      isReportExpanded: false,
      isReportVisible: true,
      pendingChatAttachmentRefs: [],
      pendingChatReportRunId: null,
      selectedChatModelTier: null,
      selectedChatThreadId: chatThreadOrder[0] ?? null,
      selectedJobId: researchRunOrder[0] ?? null,
      selectedStack: 'anthropic_perplexity',
    },
  }
}

function seedChatThreadGroups(): ChatThreadGroupRecord[] {
  return [
    {
      createdAt: seedCreatedAt,
      id: 'chat-group-policy',
      title: 'Policy briefing',
      updatedAt: seedCreatedAt,
    },
    {
      createdAt: seedCreatedAt,
      id: 'chat-group-vendors',
      title: 'Provider comparisons',
      updatedAt: seedCreatedAt,
    },
  ]
}

function seedChatRules(): ChatRuleRecord[] {
  return [
    {
      contentMarkdown: 'Answer in a concise executive briefing style. Start with the bottom line, then list the strongest evidence and the main uncertainty.',
      createdAt: seedCreatedAt,
      id: 'rule-executive-brief',
      label: 'executive-brief',
      title: 'Executive briefing',
      updatedAt: seedCreatedAt,
    },
    {
      contentMarkdown: 'Focus on evidence quality. Distinguish verified claims, weak single-source claims, and points that need another source before they should be trusted.',
      createdAt: seedCreatedAt,
      id: 'rule-evidence-review',
      label: 'evidence-review',
      title: 'Evidence review',
      updatedAt: seedCreatedAt,
    },
    {
      category: 'instruction',
      contentMarkdown: 'SYSTEM PROMPT:\nYour role is to assist the user by providing helpful, clear, and contextually relevant information. Respond in an informative, friendly, and neutral tone, adapting to the user\'s style and preferences based on the conversation history. Your purpose is to help solve problems, answer questions, generate ideas, write content, and support the user in a wide range of tasks.\n\nBEHAVIORAL GUIDELINES:\n1. Maintain a helpful, friendly, and professional demeanor.\n2. Avoid using jargon unless specifically requested by the user. Strive to communicate clearly, breaking down complex concepts into simple explanations.\n3. Respond accurately based on your training data, with knowledge defined training cutoff.\n4. Acknowledge uncertainties and suggest further ways to explore the topic if the answer is outside your knowledge.',
      createdAt: seedCreatedAt,
      id: 'rule-basis-prompt',
      includeInAutocomplete: true,
      label: 'base',
      title: 'Basis Prompt',
      updatedAt: seedCreatedAt,
      visibility: { chat: true, editor: true },
    },
    {
      category: 'function',
      contentMarkdown: 'ROLLE: Du bist ein präziser zweisprachiger Lektor (Deutsch/Englisch).\n\nAUFGABE:\n- Korrigiere Grammatik, Rechtschreibung, Zeichensetzung und Stil.\n- Erhalte Tonalität und Fachbegriffe des Originals.\n- Gib nur den überarbeiteten Text zurück, ohne Kommentare.\n\nAUSGABE: ausschließlich der lektorierte Text.',
      createdAt: seedCreatedAt,
      id: 'rule-lektor-de-eng',
      includeInAutocomplete: true,
      label: 'lektor',
      title: 'Lektor DE ENG',
      updatedAt: seedCreatedAt,
      visibility: { chat: true, editor: true },
    },
    {
      category: 'context',
      contentMarkdown: 'ROLLE: Professioneller Fachübersetzer Deutsch nach Englisch.\n\nKONTEXT:\n{{context}}\n\nVORGEHEN:\n- Übersetze sinngemäß, nicht wörtlich.\n- Übernimm Terminologie aus dem angehängten Glossar.\n- Behalte Formatierung und Absätze bei.',
      createdAt: seedCreatedAt,
      id: 'rule-uebersetzer-de-eng',
      includeInAutocomplete: true,
      label: 'translator',
      linkedContextRefs: [{ fileId: 'file-asset-demo-1', kind: 'file-asset' }],
      title: 'Übersetzer DE zu ENG',
      updatedAt: seedCreatedAt,
      visibility: { chat: true, editor: true },
    },
  ]
}

function seedFileGroups(): FileGroupRecord[] {
  return [
    {
      createdAt: seedCreatedAt,
      id: 'file-group-vendor-specs',
      sectionId: FILE_SECTION_LIBRARY_ID,
      title: 'Vendor specs',
      updatedAt: seedCreatedAt,
    },
    {
      createdAt: seedCreatedAt,
      id: 'file-group-demo-docs',
      sectionId: FILE_SECTION_LIBRARY_ID,
      title: 'Dokumentengruppe 1',
      updatedAt: seedCreatedAt,
    },
  ]
}

function seedFileAssets(): FileAssetRecord[] {
  return [
    {
      createdAt: seedCreatedAt,
      extractedText: 'Example vendor specification excerpt. EU data residency available in Frankfurt and Dublin. Audit logs retained for 365 days. Citation chains can be exported as Markdown.',
      fileName: 'vendor-spec-eu.txt',
      groupId: 'file-group-vendor-specs',
      id: 'file-asset-vendor-spec',
      label: 'vendor-spec-eu',
      mimeType: 'text/plain',
      origin: 'library',
      pageCount: null,
      parseStatus: 'parsed',
      parseWarning: null,
      sectionId: FILE_SECTION_LIBRARY_ID,
      sizeBytes: 184,
      textTruncated: false,
      title: 'vendor-spec-eu.txt',
      updatedAt: seedCreatedAt,
    },
    {
      createdAt: seedCreatedAt,
      extractedText: 'Demo-1.docx Beispieltext für die Prompt-Library-Vorführung. Enthält Überschriften, Absätze und Aufzählungen, an denen sich die Fachübersetzung demonstrieren lässt.',
      fileName: 'Demo-1.docx',
      groupId: 'file-group-demo-docs',
      id: 'file-asset-demo-1',
      label: 'demo-1',
      mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      origin: 'library',
      pageCount: 3,
      parseStatus: 'parsed',
      parseWarning: null,
      sectionId: FILE_SECTION_LIBRARY_ID,
      sizeBytes: 18243,
      textTruncated: false,
      title: 'Demo-1.docx',
      updatedAt: seedCreatedAt,
    },
    {
      createdAt: seedCreatedAt,
      extractedText: 'Demo-2.docx Beispieltext mit weiteren Absätzen und einer Tabelle, um Formatierung und Terminologie in der Übersetzung zu prüfen.',
      fileName: 'Demo-2.docx',
      groupId: 'file-group-demo-docs',
      id: 'file-asset-demo-2',
      label: 'demo-2',
      mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      origin: 'library',
      pageCount: 2,
      parseStatus: 'parsed',
      parseWarning: null,
      sectionId: FILE_SECTION_LIBRARY_ID,
      sizeBytes: 15120,
      textTruncated: false,
      title: 'Demo-2.docx',
      updatedAt: seedCreatedAt,
    },
    {
      createdAt: seedCreatedAt,
      extractedText: 'Glossar-DE-EN: zweispaltige Terminologieliste (Deutsch/Englisch) mit bevorzugten Fachbegriffen für die Übersetzung.',
      fileName: 'Glossar-DE-EN.xlsx',
      groupId: null,
      id: 'file-asset-glossar',
      label: 'glossar',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      origin: 'library',
      pageCount: null,
      parseStatus: 'parsed',
      parseWarning: null,
      sectionId: FILE_SECTION_LIBRARY_ID,
      sizeBytes: 9472,
      textTruncated: false,
      title: 'Glossar-DE-EN.xlsx',
      updatedAt: seedCreatedAt,
    },
  ]
}

function seedEditorFolders(): EditorFolderRecord[] {
  return [
    {
      createdAt: seedCreatedAt,
      id: 'editor-folder-briefs',
      title: 'Briefs',
      updatedAt: seedCreatedAt,
    },
    {
      createdAt: seedCreatedAt,
      id: 'editor-folder-drafts',
      title: 'Drafts',
      updatedAt: seedCreatedAt,
    },
  ]
}

function seedEditorDocuments(): EditorDocumentRecord[] {
  return [
    {
      contentMarkdown: editorSovereignAiBrief(),
      createdAt: '2026-05-15T10:42:00.000Z',
      folderId: 'editor-folder-briefs',
      id: 'editor-doc-sovereign-ai-2026',
      revision: 7,
      source: 'imported-research-report',
      sourceRunId: 'RO-0245',
      title: 'Sovereign AI research 2026.md',
      updatedAt: '2026-05-15T11:12:00.000Z',
    },
    {
      contentMarkdown: editorMailDraft(),
      createdAt: '2026-05-15T11:20:00.000Z',
      folderId: 'editor-folder-drafts',
      id: 'editor-doc-vendor-mail',
      revision: 2,
      source: 'blank',
      title: 'Vendor follow-up mail.md',
      updatedAt: '2026-05-15T11:37:00.000Z',
    },
  ]
}

function seedEditorComments(): EditorCommentThreadRecord[] {
  return [
    {
      anchor: {
        from: 97,
        quoteAfter: 'for German public agencies',
        quoteBefore: 'providers against',
        selectedText: 'sovereignty requirements',
        to: 121,
      },
      commentMarkdown: 'Make this criterion more concrete before sending the brief.',
      createdAt: '2026-05-15T11:02:00.000Z',
      documentId: 'editor-doc-sovereign-ai-2026',
      id: 'editor-comment-sovereignty',
      kind: 'collect',
      status: 'open',
      updatedAt: '2026-05-15T11:02:00.000Z',
    },
  ]
}

function seedResearchRuns(): ResearchRunRecord[] {
  return [
    {
      agentOverrides: { report_profile: 'deep', max_rounds: 5, first_round_queries: 10 },
      createdAt: '2026-05-15T08:12:00.000Z',
      events: [
        runEvent('RO-0247-1', '2026-05-15T08:12:00.000Z', 'Analyzing question...'),
        runEvent('RO-0247-2', '2026-05-15T08:12:20.000Z', 'Web search required (English search, recency: this week, news)'),
        runEvent('RO-0247-3', '2026-05-15T08:13:00.000Z', 'Detected analysis goals: 2 sub-questions and 8 required aspects'),
        runEvent('RO-0247-4', '2026-05-15T08:13:40.000Z', 'Planning search queries (round 1/5)...'),
        runEvent('RO-0247-5', '2026-05-15T08:14:00.000Z', 'Generated 10 new search queries (DEEP required perspectives, STORM perspectives)'),
        runEvent('RO-0247-6', '2026-05-15T08:14:30.000Z', 'Searching 10 queries (round 1/5)...'),
        runEvent('RO-0247-7', '2026-05-15T08:15:00.000Z', 'Semantically grouping 60 evidence claims (LLM call, timeout 900s)...'),
        runEvent('RO-0247-8', '2026-05-15T08:16:00.000Z', 'Processed 10 search responses, collected 59 references, created 61 evidence units'),
        runEvent('RO-0247-9', '2026-05-15T08:16:30.000Z', 'Detected 50 related questions from search results'),
        runEvent('RO-0247-10', '2026-05-15T08:17:00.000Z', 'Report evidence: 4 verified / 16 unverified candidates, context coverage 100%'),
        runEvent('RO-0247-11', '2026-05-15T08:18:00.000Z', 'Evaluating information quality (after round 1/5)...'),
        runEvent('RO-0247-12', '2026-05-15T08:18:30.000Z', 'Minor contradictions detected; confidence 7/10, more research required'),
        runEvent('RO-0247-13', '2026-05-15T08:19:00.000Z', 'Planning search queries (round 2/5)...'),
        runEvent('RO-0247-14', '2026-05-15T08:20:00.000Z', 'Searching 3 queries (round 2/5)...'),
        runEvent('RO-0247-15', '2026-05-15T08:21:00.000Z', 'Evaluating information quality (after round 2/5)...', true),
      ],
      metrics: {
        claims: 41,
        queries: 13,
        rounds: '2 / 5',
        sources: 72,
      },
      phaseState: {
        activePhase: 'evaluation',
        completedPhases: ['analysis', 'planning', 'search'],
      },
      runId: 'RO-0247',
      source: 'mock',
      stack: 'anthropic_perplexity',
      startedAt: '2026-05-15T08:12:00.000Z',
      status: 'running',
      submittedAt: '2026-05-15T08:12:00.000Z',
      summary: {
        title: 'Which providers meet the 2026 requirements for sovereign AI research in German public agencies?',
      },
    },
    {
      agentOverrides: { report_profile: 'compact', max_rounds: 10, first_round_queries: 6 },
      createdAt: '2026-05-15T08:18:00.000Z',
      events: [],
      metrics: {
        claims: 0,
        queries: 0,
        rounds: '0 / 10',
        sources: 0,
      },
      phaseState: {
        activePhase: 'analysis',
        completedPhases: [],
      },
      runId: 'RO-0246',
      source: 'mock',
      stack: 'azure_web_search',
      status: 'queued',
      submittedAt: '2026-05-15T08:18:00.000Z',
      summary: {
        queueNote: 'Starts in ~2 min',
        title: 'Compare Perplexity, Azure OpenAI Web Search, and Azure Foundry Web Search as search providers for Inqtrix.',
      },
    },
    completedRun(
      'RO-0245',
      'Market analysis: sovereign LLMs in Europe - vendors, maturity, and use cases',
      '2026-05-15T07:54:00.000Z',
      751,
      9,
      '4 / 4',
      31,
      '8.7 / 10',
      marketReport(),
    ),
    completedRun(
      'RO-0244',
      'Briefing: Which EU cloud vendors fit confidential research workflows?',
      '2026-05-15T07:31:00.000Z',
      584,
      7,
      '3 / 3',
      18,
      '8.1 / 10',
      cloudReport(),
    ),
    completedRun(
      'RO-0243',
      'Evidence review: open-source search indexes for policy monitoring in Germany',
      '2026-05-15T07:08:00.000Z',
      968,
      12,
      '5 / 5',
      42,
      '9.0 / 10',
      indexReport(),
    ),
    completedRun(
      'RO-0242',
      'Vendor comparison: web search with citations and traceable source chains',
      '2026-05-15T06:42:00.000Z',
      472,
      6,
      '2 / 2',
      14,
      '7.8 / 10',
      citationsReport(),
    ),
    completedRun(
      'RO-0241',
      'Risk analysis: low-hallucination research for regulatory decisions',
      '2026-05-15T06:15:00.000Z',
      799,
      10,
      '4 / 4',
      29,
      '8.4 / 10',
      riskReport(),
    ),
    completedRun(
      'RO-0240',
      'Briefing: audit-log requirements for AI-assisted research agents',
      '2026-05-15T05:56:00.000Z',
      627,
      8,
      '3 / 3',
      23,
      '8.6 / 10',
      auditReport(),
    ),
  ]
}

function completedRun(
  runId: string,
  title: string,
  submittedAt: string,
  durationSeconds: number,
  queries: number,
  rounds: string,
  sources: number,
  score: string,
  markdown: string,
): ResearchRunRecord {
  return {
    agentOverrides: { report_profile: 'compact' },
    createdAt: submittedAt,
    durationSeconds,
    events: completedRunEvents(runId, submittedAt, durationSeconds, queries, rounds, sources),
    finishedAt: addSeconds(submittedAt, durationSeconds),
    metrics: {
      claims: 0,
      queries,
      rounds,
      sources,
    },
    phaseState: {
      activePhase: 'answer',
      completedPhases: [...phaseOrder],
    },
    result: {
      markdown,
      references: reportReferencesFromMarkdown(markdown, []),
      topClaims: [],
      topSources: [],
    },
    runId,
    source: 'mock',
    stack: 'anthropic_perplexity',
    status: 'completed',
    submittedAt,
    summary: {
      score,
      title,
    },
  }
}

function completedRunEvents(
  runId: string,
  submittedAt: string,
  durationSeconds: number,
  queries: number,
  rounds: string,
  sources: number,
) {
  const completedAt = addSeconds(submittedAt, durationSeconds)

  return [
    runEvent(`${runId}-1`, submittedAt, 'Analyzing question...'),
    runEvent(`${runId}-2`, addSeconds(submittedAt, 45), 'Web search required (English search, recency: this month)'),
    runEvent(`${runId}-3`, addSeconds(submittedAt, 95), `Planning search queries (round ${rounds})...`),
    runEvent(`${runId}-4`, addSeconds(submittedAt, 150), `Searching ${queries} queries...`),
    runEvent(`${runId}-5`, addSeconds(submittedAt, Math.max(210, durationSeconds - 180)), `Processed ${queries} search responses, collected ${sources} references`),
    runEvent(`${runId}-6`, addSeconds(submittedAt, Math.max(260, durationSeconds - 120)), 'Evaluating information quality...'),
    runEvent(`${runId}-7`, addSeconds(submittedAt, Math.max(320, durationSeconds - 70)), 'Confidence target reached; preparing final report'),
    runEvent(`${runId}-8`, completedAt, 'Run completed'),
  ]
}

function seedChatThreads(): ChatThreadRecord[] {
  return [
    {
      createdAt: '2026-05-15T08:24:00.000Z',
      id: 'chat-briefing',
      messages: [
        chatMessage('msg-briefing-1', 'assistant', '2026-05-15T08:24:00.000Z', 'I can sharpen the question, collect source hypotheses, or prepare a research job. What should happen first?'),
        chatMessage('msg-briefing-2', 'user', '2026-05-15T08:25:00.000Z', 'Help me formulate a short search strategy for German public-sector requirements around sovereign AI research.'),
        chatMessage('msg-briefing-3', 'assistant', '2026-05-15T08:26:00.000Z', 'Three search lines would work well: legal requirements, operating model, and traceability. That can later become a research job with 5-6 starter queries.'),
      ],
      preview: 'Search strategy for public-sector requirements',
      source: 'mock',
      title: 'Prepare policy brief',
      updatedAt: '2026-05-15T08:26:00.000Z',
    },
    {
      createdAt: '2026-05-15T07:48:00.000Z',
      id: 'chat-vendors',
      messages: [
        chatMessage('msg-vendors-1', 'assistant', '2026-05-15T07:48:00.000Z', 'Which vendors should be roughly classified?'),
        chatMessage('msg-vendors-2', 'user', '2026-05-15T07:49:00.000Z', 'Perplexity, Azure OpenAI Web Search, and Azure Foundry Web Search. I care about source quality, controllability, and EU operations.'),
      ],
      preview: 'Sort vendors by source quality',
      source: 'mock',
      title: 'Classify LLM vendors',
      updatedAt: '2026-05-15T07:49:00.000Z',
    },
    {
      createdAt: '2026-05-14T15:20:00.000Z',
      id: 'chat-method',
      messages: [
        chatMessage('msg-method-1', 'assistant', '2026-05-14T15:20:00.000Z', 'I would separate the terms first: search, retrieval, answer synthesis, and auditability. After that, the metrics can be defined more tightly.'),
      ],
      preview: 'Separate search and retrieval metrics',
      source: 'mock',
      title: 'Refine search strategy',
      updatedAt: '2026-05-14T15:20:00.000Z',
    },
  ]
}

function runEvent(
  id: string,
  createdAt: string,
  title: string,
  active = false,
) {
  return {
    active: active || undefined,
    createdAt,
    id,
    kind: 'progress' as const,
    severity: title.toLowerCase().includes('contradiction') ? 'warning' as const : 'info' as const,
    title,
  }
}

function chatMessage(
  id: string,
  role: 'assistant' | 'user',
  createdAt: string,
  contentMarkdown: string,
) {
  return { contentMarkdown, createdAt, id, role }
}

function addSeconds(iso: string, seconds: number) {
  return new Date(new Date(iso).getTime() + seconds * 1000).toISOString()
}

function marketReport() {
  return `## Executive Summary

The European sovereign LLM market is maturing from model comparison into procurement architecture. Buyers increasingly care about deployability, auditability, data residency, and support contracts, not just benchmark claims.

## Findings

- Sovereign offerings are strongest when the provider can combine model access, regional hosting, operational support, and clear contractual controls.
- The most credible deployment paths use hybrid stacks: a high-quality model layer, an auditable search layer, and explicit retention controls.
- The remaining risk is fragmentation. Many vendors can satisfy one requirement, but fewer can satisfy governance, performance, and operational continuity together.

## Recommendation

Treat sovereign LLM selection as a stack decision. Keep the research record attached to metrics, sources, and report text so the decision can be reloaded and audited later.

## Referenzen

- [E1](https://digital-strategy.ec.europa.eu/en/policies/ai-act)
- [E2](https://www.bsi.bund.de/EN/Themen/Unternehmen-und-Organisationen/Informationen-und-Empfehlungen/Kuenstliche-Intelligenz/kuenstliche-intelligenz_node.html)
- [E3](https://www.enisa.europa.eu/topics/artificial-intelligence)`
}

function cloudReport() {
  return `## Briefing

Confidential research workflows fit best on EU-operated cloud platforms that expose strong identity controls, regional data processing, and clear logging boundaries.

## Assessment

- Cloud locality alone is not sufficient; the platform also needs reliable audit logs and policy controls.
- Search and LLM providers should be separable so sensitive workflows can swap providers without changing the whole app.
- The operational winner is the platform that makes retention, source tracing, and incident review boringly explicit.

## Next Step

Validate provider claims with a small project export/import test before connecting live workloads.`
}

function indexReport() {
  return `## Evidence Review

Open-source search indexes are useful for policy monitoring when source coverage, freshness, and ranking behavior are visible enough to audit.

## Key Points

- Index transparency matters more than raw result volume for regulated research.
- Public-sector monitoring should capture query history, result summaries, and report outputs together.
- A reloadable project folder gives reviewers the minimum durable artifact: what was asked, what was found, and how it was summarized.

## Conclusion

Use open-source indexes as part of a provider-neutral research stack, but preserve the full project context for later review.`
}

function citationsReport() {
  return `## Vendor Comparison

Citation quality depends on whether the search provider returns stable URLs, extractable source metadata, and enough context for claim-level review.

## Findings

- Providers with good snippets but weak metadata create attractive reports that are hard to audit.
- Providers with fewer but clearer sources can be better for policy and compliance workflows.
- Exported Markdown should therefore carry both the report body and the run metrics that explain how the card was produced.

## Recommendation

Prefer vendors that make citation chains explicit and test them through project reloads, not only through live UI inspection.`
}

function riskReport() {
  return `## Risk Analysis

Low-hallucination research depends on more than model quality. The workflow needs visible fallback states, source diversity, and durable evidence records.

## Risks

- A polished report can hide weak source coverage if metrics are not stored with it.
- A deleted card that survives in a project folder creates review confusion.
- Running jobs should not be saved as completed artifacts because their state is incomplete.

## Mitigation

Persist only completed research reports, preserve their metrics in frontmatter, and physically remove deleted reports on save.`
}

function auditReport() {
  return `## Briefing

Audit logs for AI-assisted research agents need to preserve enough context to replay the user's working state without depending on a live backend.

## Required Artifact

- A project manifest for global UI and project metadata.
- One Markdown file per completed research run.
- One Markdown file per chat thread with explicit user and assistant message roles.

## Outcome

This structure keeps the export human-readable while still giving the React app enough metadata to rebuild cards, reports, and conversations.`
}

function editorSovereignAiBrief() {
  return `# Sovereign AI research 2026

This working note compares search and LLM providers against sovereignty requirements for German public agencies.

## Decision Criteria

- **Operational control:** data locality, provider transparency, and visible audit events.
- **Source quality:** stable URLs, source metadata, and evidence that can be checked later.
- **Workflow fit:** importable reports, editable drafts, and review comments before publication.

## Draft Recommendation

Start with the providers that expose the clearest source chain. Use the editor to turn completed research reports into policy-ready briefs, then review sensitive passages through inline comments before sending the final text.`
}

function editorMailDraft() {
  return `# Vendor follow-up mail

Hello,

thank you for the initial material. We are currently comparing provider options for traceable AI research workflows and would appreciate clarification on three points:

1. Which parts of the workflow are operated in the EU?
2. How are search citations preserved for later audit?
3. Can customers export working notes as Markdown?

Best regards`
}
