import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type ReactNode,
} from 'react'
import { EditorContent, useEditor, type Editor } from '@tiptap/react'
import { BubbleMenu } from '@tiptap/react/menus'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'
import {
  AlertTriangle,
  Anchor,
  Bold,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Eye,
  FileDown,
  FileText,
  Folder,
  FolderPlus,
  GripVertical,
  Highlighter,
  Italic,
  Link,
  List,
  ListFilter,
  ListOrdered,
  LoaderCircle,
  MessageSquarePlus,
  MessageSquareText,
  MessagesSquare,
  PanelBottomClose,
  PanelBottomOpen,
  PanelLeftClose,
  PanelRightClose,
  Paperclip,
  PencilLine,
  Redo2,
  SearchCheck,
  SendHorizontal,
  Scale,
  Sparkles,
  SquarePen,
  Strikethrough,
  Trash2,
  Underline,
  Undo2,
  X,
} from '@/components/icons'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { PanelRail } from '@/components/ui/panel-rail'
import { ComposerIconButton } from '@/features/composer/ComposerIconButton'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  chatAttachmentChipsFromRefs,
  chatContextRefKey,
  chatRuleOptions,
  dedupeChatContextRefs,
  editorCommentsForDocument,
  fileGroupMentionOptions,
  fileMentionOptions,
  projectEditorDocuments,
  projectEditorFolders,
  projectFileAssets,
  selectedEditorDocument,
  type ChatAttachmentChipModel,
  type ChatRuleOption,
  type CompletedReportOption,
  type FileGroupMentionOption,
  type FileMentionOption,
} from '@/features/project/selectors'
import type {
  ChatModelOption,
  ChatModelTier,
  NodeModelResolution,
} from '@/features/researchRuns/types'
import type {
  ChatContextReferenceRecord,
  EditorCommentKind,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorEvidencePreset,
  EditorFolderRecord,
  EditorSuggestionRecord,
  ProjectState,
} from '@/features/project/types'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { commentDecorationPluginKey, createEditorExtensions, normalizeEditorMarkdownForTiptap, suggestionDecorationPluginKey } from './tiptap'
import { MarkdownSourceEditor } from './MarkdownSourceEditor'
import { documentDiffPlan, suggestionDiffPlan, type DocumentDiffBlock, type SuggestionDiffSegment } from './suggestionDiff'
import {
  blockInsertionPositionForRange,
  blockWidgetPositionForRange,
  clampAnchor,
  createCommentFromSelection,
  resolveMaterializedAnchor,
  resolveAnchorRange,
  shouldParsePastedMarkdown,
} from './anchoring'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import {
  TextImproveButton,
  TextImproveFloatingLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import { EditorDocumentChangesSection } from './EditorDocumentChangesSection'
import { EditorInstructionFeedbackCard } from './EditorInstructionFeedbackCard'
import { useEditorSuggestions, type EditorInstructionFeedback } from './useEditorSuggestions'
import {
  type MentionCategoryLabels,
  type MentionSources,
} from '@/features/composer/mention'
import { ContextChipLegend } from '@/features/composer/ContextChipLegend'
import { MentionComposer, type MentionComposerHandle } from '@/features/composer/MentionComposer'
import { type LabelResolver } from '@/features/composer/mentionDoc'
import { moveItem } from '@/features/composer/reorder'
import { Dropzone } from '@/features/files/Dropzone'
import { ingestFiles } from '@/features/files/ingest'
import { FILE_SECTION_TEMP_ID } from '@/features/files/sections'

type EditorWorkspaceProps = {
  apiKey?: string
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  defaultChatModel: NodeModelResolution | null
  dispatch: Dispatch<ResearchDeskAction>
  reportOptions: CompletedReportOption[]
  selectedModelTier: ChatModelTier | null
  state: ProjectState
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}

type EditorDocumentDropTarget = {
  folderId: string | null
  targetIndex: number
}

type EditorCopy = { [Key in keyof typeof editorCopy.de]: string }

const editorCopy = {
  de: {
    exportWord: 'Als Word exportieren',
    exportWordFailed: 'Word-Export fehlgeschlagen',
    assistant: 'Editor-Assistent',
    assistantPlaceholder: 'Beschreiben Sie, was am Dokument geändert werden soll...',
    attachComments: 'Kommentare anhängen',
    attachedComments: 'Kommentare angehängt',
    assistantDone: 'Antwort',
    assistantThinking: 'Denke nach',
    documentChanges: 'Dokument-Änderungen',
    setDiffAnchor: 'Vergleichsanker setzen',
    showDiff: 'Diff-Ansicht einschalten',
    hideDiff: 'Diff-Ansicht ausschalten',
    diffView: 'Diff-Ansicht',
    noDiffAnchor: 'Setzen Sie zuerst einen Vergleichsanker.',
    acceptAll: 'Alle übernehmen',
    rejectAll: 'Alle verwerfen',
    noDocumentChanges: 'Noch keine Dokument-Änderungen.',
    refineSuggestion: 'Vorschlag verfeinern',
    editSuggestion: 'Vorschlag bearbeiten',
    saveSuggestion: 'Speichern',
    cancelEdit: 'Abbrechen',
    sendRefinement: 'Senden',
    refinementPlaceholder: 'Was soll am Vorschlag geändert werden?',
    refiningSuggestion: 'Überarbeite Vorschlag …',
    revision: 'Revision',
    removeFromQueue: 'Aus Warteschlange entfernen',
    templates: 'Vorlagen',
    runSuggestion: 'Vorschlag erzeugen',
    runningSuggestion: 'Wird erzeugt …',
    accept: 'Übernehmen',
    reject: 'Verwerfen',
    proposedChange: 'Vorgeschlagene Änderung',
    proposedText: 'Vorschlag',
    warnings: 'Hinweise',
    changeSummary: 'Änderungen',
    reviewInEditor: 'Im Dokument übernehmen oder verwerfen',
    reviewInPanel: 'In der Karte übernehmen oder verwerfen',
    sources: 'Quellen',
    suggestionStale: 'Textstelle hat sich geändert. Bitte Vorschlag neu erzeugen.',
    regenerate: 'Neu erzeugen',
    changeKind: 'Typ ändern',
    comments: 'Kommentare',
    filterAll: 'Alle',
    kindCollect: 'Sammeln',
    kindEvidence: 'Beleg',
    kindInline: 'Direkt',
    preset: 'Preset',
    presetAddSources: 'Quellen ergänzen',
    presetFactCheck: 'Fakten prüfen',
    presetVerifyCitations: 'Zitate prüfen',
    reopen: 'Wieder öffnen',
    tabOpen: 'Offen',
    tabResolved: 'Erledigt',
    createDocument: 'Neue Datei',
    cancel: 'Abbrechen',
    createFolder: 'Neuer Ordner',
    deleteComment: 'Kommentar löschen',
    deleteDocument: 'Dokument löschen',
    editComment: 'Kommentar bearbeiten',
    save: 'Speichern',
    deleteFolder: 'Ordner löschen',
    documents: 'Dokumente',
    dropIntoFolder: 'In Ordner verschieben',
    emptyBody: 'Legen Sie eine neue Markdown-Datei an oder importieren Sie einen abgeschlossenen Research Report.',
    emptyTitle: 'Noch kein Dokument geöffnet',
    focus: 'Fokus',
    hideAssistant: 'Assistant ausblenden',
    hideComments: 'Kommentare ausblenden',
    hideTree: 'Dateibaum ausblenden',
    importReport: 'Research Report importieren',
    importedFrom: 'Importiert aus Research-Run',
    inlineComment: 'Kommentar hinzufügen...',
    inlineCommentSubmit: 'Kommentar',
    addColumn: 'Spalte hinzufügen',
    addRow: 'Zeile hinzufügen',
    closeTableEditor: 'Tabelleneditor schließen',
    columnLabel: 'Spalte',
    deleteColumn: 'Spalte löschen',
    deleteRow: 'Zeile löschen',
    formatTables: 'Markdown-Tabellen bereinigen',
    insertOrEditTable: 'Tabelle einfügen oder bearbeiten',
    sourceEditor: 'Markdown Source',
    sourceLineWrap: 'Zeilenumbruch umschalten',
    tableAlignmentCenter: 'Spalte zentrieren',
    tableAlignmentLeft: 'Spalte linksbündig ausrichten',
    tableAlignmentRight: 'Spalte rechtsbündig ausrichten',
    tableColumn: 'Spalten',
    tableEditor: 'Tabelleneditor',
    tableLines: 'Zeilen',
    tableRows: 'Datenzeilen',
    live: 'Live',
    markdown: 'Markdown',
    noComments: 'Noch keine Kommentare in diesem Dokument.',
    noDocuments: 'Noch keine Dokumente.',
    noReports: 'Keine abgeschlossenen Reports verfügbar.',
    moveDocument: 'Dokument verschieben',
    moveFolder: 'Ordner verschieben',
    renameDocument: 'Dokument umbenennen',
    renameFolder: 'Ordner umbenennen',
    resolve: 'Erledigen',
    send: 'Senden',
    showAssistant: 'Assistant einblenden',
    showComments: 'Kommentare einblenden',
    showTree: 'Dateibaum einblenden',
    source: 'Source',
    stopRun: 'Lauf abbrechen',
    updated: 'zuletzt bearbeitet',
  },
  en: {
    exportWord: 'Export to Word',
    exportWordFailed: 'Word export failed',
    assistant: 'Editor assistant',
    assistantPlaceholder: 'Describe what should change in this document...',
    attachComments: 'Attach comments',
    attachedComments: 'comments attached',
    assistantDone: 'Response',
    assistantThinking: 'Thinking',
    documentChanges: 'Document changes',
    setDiffAnchor: 'Set comparison anchor',
    showDiff: 'Show diff view',
    hideDiff: 'Hide diff view',
    diffView: 'Diff view',
    noDiffAnchor: 'Set a comparison anchor first.',
    acceptAll: 'Accept all',
    rejectAll: 'Reject all',
    noDocumentChanges: 'No document changes yet.',
    refineSuggestion: 'Refine suggestion',
    editSuggestion: 'Edit suggestion',
    saveSuggestion: 'Save',
    cancelEdit: 'Cancel',
    sendRefinement: 'Send',
    refinementPlaceholder: 'What should change in this suggestion?',
    refiningSuggestion: 'Refining suggestion …',
    revision: 'Revision',
    removeFromQueue: 'Remove from queue',
    templates: 'Templates',
    runSuggestion: 'Generate suggestion',
    runningSuggestion: 'Generating …',
    accept: 'Accept',
    reject: 'Reject',
    proposedChange: 'Proposed change',
    proposedText: 'Suggestion',
    warnings: 'Warnings',
    changeSummary: 'Changes',
    reviewInEditor: 'Accept or reject it in the document',
    reviewInPanel: 'Accept or reject it in this card',
    sources: 'Sources',
    suggestionStale: 'The passage changed. Please regenerate the suggestion.',
    regenerate: 'Regenerate',
    changeKind: 'Change type',
    comments: 'Comments',
    filterAll: 'All',
    kindCollect: 'Collect',
    kindEvidence: 'Evidence',
    kindInline: 'Direct',
    preset: 'Preset',
    presetAddSources: 'Add sources',
    presetFactCheck: 'Fact-check',
    presetVerifyCitations: 'Verify citations',
    reopen: 'Reopen',
    tabOpen: 'Open',
    tabResolved: 'Resolved',
    createDocument: 'New file',
    cancel: 'Cancel',
    createFolder: 'New folder',
    deleteComment: 'Delete comment',
    deleteDocument: 'Delete document',
    editComment: 'Edit comment',
    save: 'Save',
    deleteFolder: 'Delete folder',
    documents: 'Documents',
    dropIntoFolder: 'Move into folder',
    emptyBody: 'Create a Markdown file or import a completed research report.',
    emptyTitle: 'No document open',
    focus: 'Focus',
    hideAssistant: 'Hide assistant',
    hideComments: 'Hide comments',
    hideTree: 'Hide file tree',
    importReport: 'Import research report',
    importedFrom: 'Imported from research run',
    inlineComment: 'Add comment...',
    inlineCommentSubmit: 'Comment',
    addColumn: 'Add column',
    addRow: 'Add row',
    closeTableEditor: 'Close table editor',
    columnLabel: 'Column',
    deleteColumn: 'Delete column',
    deleteRow: 'Delete row',
    formatTables: 'Clean up Markdown tables',
    insertOrEditTable: 'Insert or edit table',
    sourceEditor: 'Markdown source',
    sourceLineWrap: 'Toggle line wrap',
    tableAlignmentCenter: 'Center column',
    tableAlignmentLeft: 'Align column left',
    tableAlignmentRight: 'Align column right',
    tableColumn: 'Columns',
    tableEditor: 'Table editor',
    tableLines: 'Lines',
    tableRows: 'Data rows',
    live: 'Live',
    markdown: 'Markdown',
    noComments: 'No comments in this document yet.',
    noDocuments: 'No documents yet.',
    noReports: 'No completed reports available.',
    moveDocument: 'Move document',
    moveFolder: 'Move folder',
    renameDocument: 'Rename document',
    renameFolder: 'Rename folder',
    resolve: 'Resolve',
    send: 'Send',
    showAssistant: 'Show assistant',
    showComments: 'Show comments',
    showTree: 'Show file tree',
    source: 'Source',
    stopRun: 'Cancel run',
    updated: 'last edited',
  },
} as const

export default function EditorWorkspace({
  apiKey,
  chatModelOptions,
  chatModelOptionsStatus,
  defaultChatModel,
  dispatch,
  reportOptions,
  selectedModelTier,
  state,
  textImprovement,
}: EditorWorkspaceProps) {
  const { locale } = useLocale()
  const copy = editorCopy[locale]
  const folders = useMemo(() => projectEditorFolders(state), [state.editorFolderOrder, state.editorFolders])
  const documents = useMemo(() => projectEditorDocuments(state), [state.editorDocumentOrder, state.editorDocuments])
  const activeDocument = selectedEditorDocument(state)
  const [activeEditor, setActiveEditor] = useState<Editor | null>(null)
  const comments = useMemo(
    () => editorCommentsForDocument(state, activeDocument?.id ?? null),
    [activeDocument?.id, state.editorComments],
  )
  const ruleOptions = useMemo(() => chatRuleOptions(state, 'editor'), [state.chatRuleOrder, state.chatRules])
  const fileOptions = useMemo(() => fileMentionOptions(state), [state.fileAssetOrder, state.fileAssets])
  const fileGroupOptions = useMemo(() => fileGroupMentionOptions(state), [state.fileGroupOrder, state.fileGroups])
  const [attachedCommentIds, setAttachedCommentIds] = useState<string[]>([])
  // Editor attachments come from two sources: positional `[N]` pills owned by the
  // composer (files/groups) and rules/dropped files attached out-of-band. They
  // merge -- pills first -- into one deduplicated ref list, mirroring the chat.
  const [pillRefs, setPillRefs] = useState<ChatContextReferenceRecord[]>([])
  const [extraRefs, setExtraRefs] = useState<ChatContextReferenceRecord[]>([])
  const attachedRefs = useMemo(
    () => dedupeChatContextRefs([...pillRefs, ...extraRefs]),
    [pillRefs, extraRefs],
  )
  const attachmentChips = useMemo(
    () => chatAttachmentChipsFromRefs(state, attachedRefs),
    [state, attachedRefs],
  )
  const composerRef = useRef<MentionComposerHandle | null>(null)
  const [isAttachActive, setIsAttachActive] = useState(false)

  const handleCreateComment = useCallback((comment: EditorCommentThreadRecord) => {
    dispatch({ comment, type: 'createEditorComment' })
  }, [dispatch])

  const addExtraRef = useCallback((ref: ChatContextReferenceRecord) => {
    setExtraRefs((prev) => dedupeChatContextRefs([...prev, ref]))
  }, [])

  const clearEditorAttachments = useCallback(() => {
    setPillRefs([])
    setExtraRefs([])
    composerRef.current?.clear()
  }, [])

  const handleRemoveEditorChip = useCallback((ref: ChatContextReferenceRecord) => {
    composerRef.current?.removeRef(ref)
    setExtraRefs((prev) => prev.filter((item) => chatContextRefKey(item) !== chatContextRefKey(ref)))
  }, [])

  const editorPillKeys = useMemo(() => pillRefs.map(chatContextRefKey), [pillRefs])
  const editorPendingKeys = useMemo(() => extraRefs.map(chatContextRefKey), [extraRefs])

  const handleReorderEditorPill = useCallback((fromIndex: number, toIndex: number) => {
    composerRef.current?.reorderPill(fromIndex, toIndex)
  }, [])

  const handleReorderEditorPending = useCallback((fromIndex: number, toIndex: number) => {
    setExtraRefs((prev) => moveItem(prev, fromIndex, toIndex))
  }, [])

  const handleAttachEditorFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return
    const existingLabels = projectFileAssets(state).map((asset) => asset.label)
    const assets = await ingestFiles(files, { kind: 'editor', sectionId: FILE_SECTION_TEMP_ID }, undefined, existingLabels)
    if (assets.length === 0) return
    dispatch({ assets, type: 'ingestFileAssets' })
    for (const asset of assets) addExtraRef({ fileId: asset.id, kind: 'file-asset' })
  }, [addExtraRef, dispatch, state])

  const {
    clearInstructionFeedback,
    documentSuggestions,
    handleAcceptSuggestionGroup,
    handleAcceptSuggestion,
    handleEditSuggestionProposal,
    handleGlobalRun,
    handleInstructionRun,
    handleMarkSuggestionStale,
    handleRefineSuggestion,
    handleRejectSuggestion,
    handleRejectSuggestionGroup,
    handleRunComment,
    handleStopRun,
    handleStopSuggestionRun,
    instructionFeedback,
    isGlobalRunning,
    runErrors,
    runningCommentIds,
    runningSuggestionIds,
    suggestionErrors,
  } = useEditorSuggestions({
    activeDocument,
    activeEditor,
    apiKey,
    attachedCommentIds,
    attachedRefs,
    comments,
    dispatch,
    locale,
    onGlobalSuccess: () => {
      setAttachedCommentIds([])
      clearEditorAttachments()
      setIsAttachActive(false)
    },
    selectedModelTier,
    state,
  })

  const handleSelectSuggestion = useCallback((suggestionId: string) => {
    if (state.editorUi.isDiffVisible) {
      dispatch({ isVisible: false, type: 'setEditorDiffVisible' })
    }
    globalThis.setTimeout(() => {
      const escapedId = escapeCssIdentifier(suggestionId)
      const target = globalThis.document?.querySelector<HTMLElement>(
        `[data-suggestion-block-card="${escapedId}"], [data-suggestion-id="${escapedId}"]`,
      )
      if (!target) return
      target.scrollIntoView({ block: 'center', behavior: 'smooth' })
      target.classList.add('suggestion-scroll-pulse')
      globalThis.setTimeout(() => target.classList.remove('suggestion-scroll-pulse'), 900)
    }, 0)
  }, [dispatch, state.editorUi.isDiffVisible])

  return (
    <div className="flex h-[calc(100svh-var(--header-h))] min-w-0 bg-canvas text-foreground">
      {state.editorUi.isTreeVisible ? (
        <EditorFileTree
          activeDocumentId={activeDocument?.id ?? null}
          copy={copy}
          dispatch={dispatch}
          documents={documents}
          folders={folders}
          reportOptions={reportOptions}
        />
      ) : (
        <PanelRail
          label={copy.showTree}
          onExpand={() => dispatch({ isVisible: true, type: 'setEditorTreeVisible' })}
          side="left"
        />
      )}
      <main className="flex min-w-0 flex-1 flex-col border-r border-border bg-background">
        {activeDocument ? (
          <>
            <EditorTopBar
              commentCount={comments.length}
              copy={copy}
              dispatch={dispatch}
              document={activeDocument}
              editor={activeEditor}
              isDiffVisible={state.editorUi.isDiffVisible}
              isDirty={state.dirty}
              viewMode={state.editorUi.viewMode}
            />
            <div className="flex min-h-0 flex-1 flex-col">
              <MarkdownLiveEditor
                comments={comments}
                copy={copy}
                document={activeDocument}
                diffAnchorMarkdown={activeDocument.diffAnchorMarkdown ?? null}
                isDiffVisible={state.editorUi.isDiffVisible}
                mode={state.editorUi.viewMode}
                onChange={(contentMarkdown) => {
                  dispatch({
                    contentMarkdown,
                    documentId: activeDocument.id,
                    type: 'updateEditorDocumentMarkdown',
                  })
                }}
                onCreateComment={handleCreateComment}
                onEditorReady={setActiveEditor}
                onAcceptSuggestion={handleAcceptSuggestion}
                onEditSuggestion={handleEditSuggestionProposal}
                onMarkSuggestionStale={handleMarkSuggestionStale}
                onRefineSuggestion={handleRefineSuggestion}
                onRejectSuggestion={handleRejectSuggestion}
                onSelectComment={(commentId) => dispatch({ commentId, type: 'selectEditorComment' })}
                onStopSuggestion={handleStopSuggestionRun}
                runningSuggestionIds={runningSuggestionIds}
                selectedCommentId={state.editorUi.selectedCommentId}
                suggestionErrors={suggestionErrors}
                suggestions={documentSuggestions.filter((suggestion) => suggestion.status === 'pending')}
                textImprovement={textImprovement}
              />
              <EditorAssistantComposer
                attachedCommentIds={attachedCommentIds}
                attachmentChips={attachmentChips}
                chatModelOptions={chatModelOptions}
                chatModelOptionsStatus={chatModelOptionsStatus}
                comments={comments}
                composerRef={composerRef}
                copy={copy}
                defaultChatModel={defaultChatModel}
                dispatch={dispatch}
                draft={state.editorUi.assistantDraft}
                fileGroupOptions={fileGroupOptions}
                fileOptions={fileOptions}
                reportOptions={reportOptions}
                isAttachActive={isAttachActive}
                instructionFeedback={instructionFeedback}
                isRunning={isGlobalRunning}
                isVisible={state.editorUi.isAssistantVisible}
                isWideCanvas={!state.editorUi.isTreeVisible && !state.editorUi.isCommentPanelVisible}
                onAttachFiles={(files) => void handleAttachEditorFiles(files)}
                onAttachRule={(ruleId) => addExtraRef({ kind: 'chat-rule', ruleId })}
                onRefsChange={setPillRefs}
                onReorderPending={handleReorderEditorPending}
                onReorderPill={handleReorderEditorPill}
                onRemoveAttachedComment={(commentId) =>
                  setAttachedCommentIds((prev) => prev.filter((id) => id !== commentId))}
                onRemoveChip={handleRemoveEditorChip}
                pendingKeys={editorPendingKeys}
                pillKeys={editorPillKeys}
                onToggleAttach={() => {
                  const openCollectIds = comments
                    .filter((comment) => comment.status === 'open' && comment.kind === 'collect')
                    .map((comment) => comment.id)
                  setAttachedCommentIds(isAttachActive ? [] : openCollectIds)
                  setIsAttachActive(!isAttachActive)
                }}
                onDismissInstructionFeedback={clearInstructionFeedback}
                onSend={() => {
                  const instruction = composerRef.current?.getInstructionText().trim() ?? ''
                  const hasAttachedComments = comments.some((comment) =>
                    comment.status === 'open'
                    && comment.kind === 'collect'
                    && attachedCommentIds.includes(comment.id))
                  if (hasAttachedComments) {
                    void handleGlobalRun(instruction)
                  } else {
                    void handleInstructionRun(instruction)
                  }
                }}
                onStop={handleStopRun}
                ruleOptions={ruleOptions}
                selectedModelTier={selectedModelTier}
                textImprovement={textImprovement}
              />
            </div>
          </>
        ) : (
          <EditorEmptyState copy={copy} dispatch={dispatch} reportOptions={reportOptions} />
        )}
      </main>
      {state.editorUi.isCommentPanelVisible ? (
        <EditorCommentsPanel
          comments={comments}
          copy={copy}
          dispatch={dispatch}
          onAcceptSuggestionGroup={handleAcceptSuggestionGroup}
          onAcceptSuggestion={handleAcceptSuggestion}
          onRejectSuggestionGroup={handleRejectSuggestionGroup}
          onRejectSuggestion={handleRejectSuggestion}
          onRunComment={handleRunComment}
          onSelectSuggestion={handleSelectSuggestion}
          runErrors={runErrors}
          runningCommentIds={runningCommentIds}
          selectedCommentId={state.editorUi.selectedCommentId}
          suggestions={documentSuggestions}
        />
      ) : (
        <PanelRail
          label={copy.showComments}
          onExpand={() => dispatch({ isVisible: true, type: 'setEditorCommentPanelVisible' })}
          side="right"
        />
      )}
    </div>
  )
}

function EditorFileTree({
  activeDocumentId,
  copy,
  dispatch,
  documents,
  folders,
  reportOptions,
}: {
  activeDocumentId: string | null
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  documents: EditorDocumentRecord[]
  folders: EditorFolderRecord[]
  reportOptions: CompletedReportOption[]
}) {
  const [expandedFolderIds, setExpandedFolderIds] = useState<ReadonlySet<string>>(() => new Set(folders.map((folder) => folder.id)))
  const [draggedDocumentId, setDraggedDocumentId] = useState<string | null>(null)
  const [draggedFolderId, setDraggedFolderId] = useState<string | null>(null)
  const [documentDropTarget, setDocumentDropTarget] = useState<EditorDocumentDropTarget | null>(null)
  const [editingDocumentId, setEditingDocumentId] = useState<string | null>(null)
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null)
  const [folderDropTargetIndex, setFolderDropTargetIndex] = useState<number | null>(null)
  const [documentTitleDraft, setDocumentTitleDraft] = useState('')
  const [folderTitleDraft, setFolderTitleDraft] = useState('')
  const documentTitleInputRef = useRef<HTMLInputElement | null>(null)
  const folderTitleInputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)
  const ungroupedDocuments = documents.filter((document) => !document.folderId || !folders.some((folder) => folder.id === document.folderId))
  const hasFolders = folders.length > 0

  useEffect(() => {
    setExpandedFolderIds((current) => new Set([...current, ...folders.map((folder) => folder.id)]))
  }, [folders])

  useLayoutEffect(() => {
    if (!editingDocumentId) return
    documentTitleInputRef.current?.focus()
    documentTitleInputRef.current?.select()
  }, [editingDocumentId])

  useLayoutEffect(() => {
    if (!editingFolderId) return
    folderTitleInputRef.current?.focus()
    folderTitleInputRef.current?.select()
  }, [editingFolderId])

  function toggleFolder(folderId: string) {
    setExpandedFolderIds((current) => {
      const next = new Set(current)
      if (next.has(folderId)) {
        next.delete(folderId)
      } else {
        next.add(folderId)
      }
      return next
    })
  }

  function startDocumentTitleEdit(document: EditorDocumentRecord) {
    setEditingDocumentId(document.id)
    setDocumentTitleDraft(document.title)
  }

  function commitDocumentTitleEdit() {
    if (!editingDocumentId) return
    const title = documentTitleDraft.trim()
    if (title) {
      dispatch({ documentId: editingDocumentId, title, type: 'renameEditorDocument' })
    }
    setEditingDocumentId(null)
    setDocumentTitleDraft('')
  }

  function cancelDocumentTitleEdit() {
    setEditingDocumentId(null)
    setDocumentTitleDraft('')
  }

  function startFolderTitleEdit(folder: EditorFolderRecord) {
    setEditingFolderId(folder.id)
    setFolderTitleDraft(folder.title)
  }

  function commitFolderTitleEdit() {
    if (!editingFolderId) return
    const title = folderTitleDraft.trim()
    if (title) {
      dispatch({ folderId: editingFolderId, title, type: 'renameEditorFolder' })
    }
    setEditingFolderId(null)
    setFolderTitleDraft('')
  }

  function cancelFolderTitleEdit() {
    setEditingFolderId(null)
    setFolderTitleDraft('')
  }

  function readFolderDropTarget(clientY: number, excludedFolderId = draggedFolderId) {
    const container = listRef.current
    if (!container) return null
    const folderElements = Array.from(container.querySelectorAll<HTMLElement>('[data-editor-draggable-folder-id]'))
      .filter((folderElement) => (
        folderElement.dataset.editorDraggableFolderId !== excludedFolderId
        && folderElement.getBoundingClientRect().height > 0
      ))
    for (const [index, folderElement] of folderElements.entries()) {
      const rect = folderElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) return index
    }
    return folderElements.length
  }

  function beginFolderDrag(event: ReactPointerEvent<HTMLButtonElement>, folderId: string) {
    if (event.button !== 0) return
    event.preventDefault()
    setDraggedFolderId(folderId)
    setFolderDropTargetIndex(readFolderDropTarget(event.clientY, folderId))

    function handlePointerMove(moveEvent: PointerEvent) {
      setFolderDropTargetIndex(readFolderDropTarget(moveEvent.clientY, folderId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const nextDropTarget = readFolderDropTarget(upEvent.clientY, folderId)
      cleanupPointerDrag()
      if (nextDropTarget === null) return
      dispatch({ folderId, targetIndex: nextDropTarget, type: 'moveEditorFolder' })
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cleanupPointerDrag)
      setDraggedFolderId(null)
      setFolderDropTargetIndex(null)
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cleanupPointerDrag)
  }

  function readDocumentDropTarget(clientY: number, excludedDocumentId = draggedDocumentId): EditorDocumentDropTarget | null {
    const container = listRef.current
    if (!container) return null
    const sectionElements = Array.from(container.querySelectorAll<HTMLElement>('[data-editor-document-section]'))
      .filter((sectionElement) => sectionElement.getBoundingClientRect().height > 0)
    if (sectionElements.length === 0) return null
    const sectionElement = sectionElements.find((candidate) => {
      const rect = candidate.getBoundingClientRect()
      return clientY >= rect.top - 8 && clientY <= rect.bottom + 8
    }) ?? sectionElements.reduce((nearest, candidate) => {
      const nearestRect = nearest.getBoundingClientRect()
      const candidateRect = candidate.getBoundingClientRect()
      const nearestDistance = Math.min(Math.abs(clientY - nearestRect.top), Math.abs(clientY - nearestRect.bottom))
      const candidateDistance = Math.min(Math.abs(clientY - candidateRect.top), Math.abs(clientY - candidateRect.bottom))
      return candidateDistance < nearestDistance ? candidate : nearest
    })
    const folderKey = sectionElement.dataset.editorFolderId ?? '__ungrouped__'
    const folderId = folderKey === '__ungrouped__' ? null : folderKey
    const documentElements = Array.from(sectionElement.querySelectorAll<HTMLElement>('[data-editor-document-id]'))
      .filter((documentElement) => (
        documentElement.dataset.editorDocumentId !== excludedDocumentId
        && documentElement.getBoundingClientRect().height > 0
      ))
    for (const [index, documentElement] of documentElements.entries()) {
      const rect = documentElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) return { folderId, targetIndex: index }
    }
    return { folderId, targetIndex: documentElements.length }
  }

  function beginDocumentDrag(event: ReactPointerEvent<HTMLButtonElement>, documentId: string) {
    if (event.button !== 0) return
    event.preventDefault()
    setDraggedDocumentId(documentId)
    setDocumentDropTarget(readDocumentDropTarget(event.clientY, documentId))

    function handlePointerMove(moveEvent: PointerEvent) {
      setDocumentDropTarget(readDocumentDropTarget(moveEvent.clientY, documentId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const nextDropTarget = readDocumentDropTarget(upEvent.clientY, documentId)
      cleanupPointerDrag()
      if (!nextDropTarget) return
      dispatch({
        documentId,
        folderId: nextDropTarget.folderId,
        targetIndex: nextDropTarget.targetIndex,
        type: 'moveEditorDocumentToFolder',
      })
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cleanupPointerDrag)
      setDraggedDocumentId(null)
      setDocumentDropTarget(null)
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cleanupPointerDrag)
  }

  return (
    <aside className="flex w-[17.5rem] shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex h-12 items-center justify-between border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <Folder className="size-4 text-muted-foreground" />
          <h2 className="truncate text-sm font-semibold">{copy.documents}</h2>
        </div>
        <div className="flex items-center gap-1">
          <ImportReportMenu copy={copy} dispatch={dispatch} reportOptions={reportOptions} />
          <TooltipButton
            label={copy.createFolder}
            onClick={() => dispatch({ title: copy.createFolder, type: 'createEditorFolder' })}
          >
            <FolderPlus className="size-4" />
          </TooltipButton>
          <TooltipButton
            label={copy.createDocument}
            onClick={() => dispatch({ type: 'createEditorDocument' })}
          >
            <SquarePen className="size-4" />
          </TooltipButton>
          <TooltipButton
            label={copy.hideTree}
            onClick={() => dispatch({ isVisible: false, type: 'setEditorTreeVisible' })}
          >
            <PanelLeftClose className="size-4" />
          </TooltipButton>
        </div>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-1 p-2" ref={listRef}>
          {folders.map((folder, folderIndex) => {
            const isExpanded = expandedFolderIds.has(folder.id)
            const isDraggingFolder = draggedFolderId === folder.id
            const showFolderBeforeIndicator = folderDropTargetIndex === folderIndex
            const showFolderAfterIndicator = folderDropTargetIndex === folders.length && folderIndex === folders.length - 1
            const showDropFrame = documentDropTarget?.folderId === folder.id
            const folderDocuments = documents.filter((document) => document.folderId === folder.id)
            return (
              <section
                className={cn(
                  'relative transition-colors',
                  showDropFrame && 'bg-brand-subtle/45',
                  isDraggingFolder && 'scale-[0.995] opacity-80 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/40',
                )}
                data-editor-document-section
                data-editor-draggable-folder-id={folder.id}
                data-editor-folder-id={folder.id}
                key={folder.id}
              >
                {showFolderBeforeIndicator ? <DropIndicator className="-top-1" /> : null}
                {showFolderAfterIndicator ? <DropIndicator className="-bottom-1" /> : null}
                <div className="group/folder grid min-h-8 grid-cols-[1.35rem_1rem_minmax(0,1fr)_auto_auto_auto_auto] items-center gap-1 px-1.5 text-foreground/75 transition-colors hover:text-foreground">
                  <button
                    aria-expanded={isExpanded}
                    aria-label={`${isExpanded ? copy.hideTree : copy.showTree}: ${folder.title}`}
                    className="grid size-6 place-items-center rounded-sm hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    onClick={() => toggleFolder(folder.id)}
                    type="button"
                  >
                    {isExpanded ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                  </button>
                  <Folder className="size-3.5 shrink-0" />
                  {editingFolderId === folder.id ? (
                    <input
                      aria-label={copy.renameFolder}
                      className="min-w-0 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 text-xs font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      onBlur={commitFolderTitleEdit}
                      onChange={(event) => setFolderTitleDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault()
                          commitFolderTitleEdit()
                        }
                        if (event.key === 'Escape') {
                          event.preventDefault()
                          cancelFolderTitleEdit()
                        }
                      }}
                      ref={folderTitleInputRef}
                      value={folderTitleDraft}
                    />
                  ) : (
                    <button
                      className="min-w-0 truncate rounded-sm px-1 py-0.5 text-left text-xs font-semibold text-foreground/75 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      onDoubleClick={() => startFolderTitleEdit(folder)}
                      onClick={() => toggleFolder(folder.id)}
                      title={copy.renameFolder}
                      type="button"
                    >
                      {folder.title}
                    </button>
                  )}
                  <span className="shrink-0 rounded-sm px-1 text-[10px] font-semibold tabular-nums text-muted-foreground">
                    {folderDocuments.length}
                  </span>
                  <button
                    aria-label={copy.createDocument}
                    className="grid size-6 shrink-0 place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/folder:opacity-100"
                    onClick={() => dispatch({ folderId: folder.id, type: 'createEditorDocument' })}
                    type="button"
                  >
                    <SquarePen className="size-3.5" />
                  </button>
                  <button
                    aria-label={copy.moveFolder}
                    className="grid size-6 shrink-0 cursor-grab place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/folder:opacity-100 active:cursor-grabbing"
                    onPointerDown={(event) => beginFolderDrag(event, folder.id)}
                    type="button"
                  >
                    <GripVertical className="size-3.5" />
                  </button>
                  <button
                    aria-label={copy.deleteFolder}
                    className="grid size-6 shrink-0 place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-destructive focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/folder:opacity-100"
                    onClick={() => dispatch({ folderId: folder.id, type: 'deleteEditorFolder' })}
                    type="button"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
                {isExpanded ? (
                  <div className="ml-4 space-y-0.5 border-l border-border/70 pl-2">
                    {folderDocuments.map((document, index) => (
                      <EditorDocumentTreeItem
                        beginDocumentDrag={beginDocumentDrag}
                        cancelTitleEdit={cancelDocumentTitleEdit}
                        commitTitleEdit={commitDocumentTitleEdit}
                        copy={copy}
                        document={document}
                        isActive={activeDocumentId === document.id}
                        isDragging={draggedDocumentId === document.id}
                        isEditing={editingDocumentId === document.id}
                        isNested
                        key={document.id}
                        onDelete={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
                        onDraftChange={setDocumentTitleDraft}
                        onOpen={() => dispatch({ documentId: document.id, type: 'openEditorDocument' })}
                        showAfterIndicator={documentDropTarget?.folderId === folder.id && documentDropTarget.targetIndex === folderDocuments.length && index === folderDocuments.length - 1}
                        showBeforeIndicator={documentDropTarget?.folderId === folder.id && documentDropTarget.targetIndex === index}
                        startTitleEdit={startDocumentTitleEdit}
                        titleDraft={documentTitleDraft}
                        titleInputRef={documentTitleInputRef}
                      />
                    ))}
                    {folderDocuments.length === 0 ? (
                      <p className="rounded-md px-2 py-1.5 text-[11px] font-medium text-muted-foreground">{copy.dropIntoFolder}</p>
                    ) : null}
                  </div>
                ) : null}
              </section>
            )
          })}
          {ungroupedDocuments.length > 0 || documents.length === 0 || hasFolders ? (
            <section
              className={cn('relative space-y-0.5 rounded-md', documentDropTarget?.folderId === null && 'bg-brand-subtle/30')}
              data-editor-document-section
              data-editor-folder-id="__ungrouped__"
            >
              {hasFolders && <p className="px-1.5 py-1 text-[11px] font-semibold uppercase text-muted-foreground">{copy.documents}</p>}
              {ungroupedDocuments.map((document, index) => (
                <EditorDocumentTreeItem
                  beginDocumentDrag={beginDocumentDrag}
                  cancelTitleEdit={cancelDocumentTitleEdit}
                  commitTitleEdit={commitDocumentTitleEdit}
                  copy={copy}
                  document={document}
                  isActive={activeDocumentId === document.id}
                  isDragging={draggedDocumentId === document.id}
                  isEditing={editingDocumentId === document.id}
                  isNested={false}
                  key={document.id}
                  onDelete={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
                  onDraftChange={setDocumentTitleDraft}
                  onOpen={() => dispatch({ documentId: document.id, type: 'openEditorDocument' })}
                  showAfterIndicator={documentDropTarget?.folderId === null && documentDropTarget.targetIndex === ungroupedDocuments.length && index === ungroupedDocuments.length - 1}
                  showBeforeIndicator={documentDropTarget?.folderId === null && documentDropTarget.targetIndex === index}
                  startTitleEdit={startDocumentTitleEdit}
                  titleDraft={documentTitleDraft}
                  titleInputRef={documentTitleInputRef}
                />
              ))}
              {documents.length === 0 ? (
                <p className="px-2 py-6 text-center text-xs text-muted-foreground">{copy.noDocuments}</p>
              ) : null}
            </section>
          ) : null}
        </div>
      </ScrollArea>
    </aside>
  )
}

function EditorDocumentTreeItem({
  beginDocumentDrag,
  cancelTitleEdit,
  commitTitleEdit,
  copy,
  document,
  isActive,
  isDragging,
  isEditing,
  isNested,
  onDelete,
  onDraftChange,
  onOpen,
  showAfterIndicator,
  showBeforeIndicator,
  startTitleEdit,
  titleDraft,
  titleInputRef,
}: {
  beginDocumentDrag: (event: ReactPointerEvent<HTMLButtonElement>, documentId: string) => void
  cancelTitleEdit: () => void
  commitTitleEdit: () => void
  copy: EditorCopy
  document: EditorDocumentRecord
  isActive: boolean
  isDragging: boolean
  isEditing: boolean
  isNested: boolean
  onDelete: () => void
  onDraftChange: (value: string) => void
  onOpen: () => void
  showAfterIndicator: boolean
  showBeforeIndicator: boolean
  startTitleEdit: (document: EditorDocumentRecord) => void
  titleDraft: string
  titleInputRef: RefObject<HTMLInputElement | null>
}) {
  return (
    <div
      className={cn(
        'group/document relative transition-colors',
        isNested
          ? 'bg-transparent hover:text-foreground'
          : 'border-border/60 bg-card/60 shadow-[0_1px_1px_var(--shadow-hairline)] hover:border-border hover:bg-background',
        !isNested && 'rounded-md border',
        isNested && isActive && 'text-foreground before:absolute before:-left-[9px] before:bottom-1.5 before:top-1.5 before:w-0.5 before:rounded-full before:bg-brand',
        !isNested && isActive && 'bg-brand-subtle text-foreground ring-1 ring-brand/25',
        isDragging && 'scale-[0.99] opacity-75 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/50',
      )}
      data-editor-document-id={document.id}
    >
      {showBeforeIndicator ? <DropIndicator className="-top-1" /> : null}
      {showAfterIndicator ? <DropIndicator className="-bottom-1" /> : null}
      <button
        aria-label={copy.moveDocument}
        className={cn(
          'absolute top-1/2 z-10 grid -translate-y-1/2 cursor-grab place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/document:opacity-100 active:cursor-grabbing',
          isNested ? 'right-7 size-6' : 'right-8 size-7',
        )}
        onPointerDown={(event) => beginDocumentDrag(event, document.id)}
        type="button"
      >
        <GripVertical className="size-3.5" />
      </button>
      {isEditing ? (
        <div className={cn(
          'grid w-full min-w-0 grid-cols-[1rem_minmax(0,1fr)] items-center gap-2 text-left',
          isNested ? 'min-h-8 px-2 py-1 pr-14' : 'min-h-9 px-3 py-1.5 pr-16',
        )}>
          <FileText
            className={cn(
              'size-3.5 shrink-0',
              isNested && isActive ? 'text-brand' : 'text-muted-foreground',
            )}
          />
          <input
            aria-label={copy.renameDocument}
            className="min-w-0 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 text-sm font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onBlur={commitTitleEdit}
            onChange={(event) => onDraftChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                commitTitleEdit()
              }
              if (event.key === 'Escape') {
                event.preventDefault()
                cancelTitleEdit()
              }
            }}
            ref={titleInputRef}
            value={titleDraft}
          />
        </div>
      ) : (
        <button
          aria-pressed={isActive}
          className={cn(
            'grid w-full min-w-0 grid-cols-[1rem_minmax(0,1fr)] items-center gap-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isNested ? 'min-h-8 px-2 py-1 pr-14' : 'min-h-9 px-3 py-1.5 pr-16',
          )}
          onClick={onOpen}
          onDoubleClick={() => startTitleEdit(document)}
          title={copy.renameDocument}
          type="button"
        >
          <FileText
            className={cn(
              'size-3.5 shrink-0',
              isNested && isActive ? 'text-brand' : 'text-muted-foreground',
            )}
          />
          <span className={cn(
            'min-w-0 truncate text-sm font-semibold',
            isNested ? 'text-foreground/85' : 'text-foreground',
            isActive && 'text-foreground',
          )}>
            {document.title}
          </span>
        </button>
      )}
      <Button
        aria-label={copy.deleteDocument}
        className={cn(
          'absolute top-1/2 -translate-y-1/2 text-foreground/55 opacity-0 transition hover:text-destructive focus-visible:opacity-100 group-hover/document:opacity-100',
          isNested ? 'right-1 size-6' : 'right-1.5 size-7',
        )}
        onClick={onDelete}
        size="icon"
        type="button"
        variant="ghost"
      >
        <Trash2 className="size-3.5" />
      </Button>
    </div>
  )
}

function DropIndicator({ className }: { className?: string }) {
  return (
    <span className={cn('pointer-events-none absolute left-1 right-1 z-20 h-0.5 rounded-full bg-brand shadow-[0_0_0_1px_var(--background)]', className)} />
  )
}

function ImportReportMenu({
  copy,
  dispatch,
  reportOptions,
  triggerClassName,
  variant = 'icon',
}: {
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  reportOptions: CompletedReportOption[]
  triggerClassName?: string
  variant?: 'button' | 'icon'
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={copy.importReport}
          className={cn(
            variant === 'icon' && 'size-8 rounded-md',
            variant === 'button' && 'justify-center',
            triggerClassName,
          )}
          size={variant === 'button' ? 'default' : 'icon'}
          type="button"
          variant={variant === 'button' ? 'outline' : 'ghost'}
        >
          <BookOpen className="size-4" />
          {variant === 'button' ? <span className="min-w-0 truncate">{copy.importReport}</span> : null}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        <DropdownMenuLabel>{copy.importReport}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {reportOptions.length === 0 ? (
          <DropdownMenuItem disabled>{copy.noReports}</DropdownMenuItem>
        ) : reportOptions.map((report) => (
          <DropdownMenuItem
            key={report.runId}
            onClick={() => dispatch({ runId: report.runId, type: 'importResearchReportToEditor' })}
          >
            <FileText className="size-4" />
            <span className="min-w-0 truncate">{report.title}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function EditorTopBar({
  commentCount,
  copy,
  dispatch,
  document,
  editor,
  isDiffVisible,
  isDirty,
  viewMode,
}: {
  commentCount: number
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  document: EditorDocumentRecord
  editor: Editor | null
  isDiffVisible: boolean
  isDirty: boolean
  viewMode: ProjectState['editorUi']['viewMode']
}) {
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [titleDraft, setTitleDraft] = useState(document.title)
  const titleInputRef = useRef<HTMLInputElement | null>(null)
  const [isExporting, setIsExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)

  useLayoutEffect(() => {
    if (!isEditingTitle) return
    titleInputRef.current?.focus()
    titleInputRef.current?.select()
  }, [isEditingTitle])

  useEffect(() => {
    if (isEditingTitle) return
    setTitleDraft(document.title)
  }, [document.title, isEditingTitle])

  function commitTitleEdit() {
    const title = titleDraft.trim()
    if (title) {
      dispatch({ documentId: document.id, title, type: 'renameEditorDocument' })
    } else {
      setTitleDraft(document.title)
    }
    setIsEditingTitle(false)
  }

  async function handleExportWord() {
    if (isExporting) return
    setExportError(null)
    setIsExporting(true)
    try {
      const { exportMarkdownToDocx } = await import('./export/docxExport')
      await exportMarkdownToDocx(document.contentMarkdown, document.title)
    } catch (error) {
      setExportError(copy.exportWordFailed)
      console.error('Inqtrix Word export failed.', error)
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <header className="grid h-12 shrink-0 grid-cols-[minmax(12rem,1fr)_auto_minmax(12rem,1fr)] items-center gap-2 border-b border-border bg-background px-3">
      <div className="flex min-w-0 items-center gap-2">
        <FileText className="size-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-1.5">
            {isEditingTitle ? (
              <input
                aria-label={copy.renameDocument}
                className="min-w-0 flex-1 rounded-sm border-0 bg-transparent px-0 text-sm font-semibold text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onBlur={commitTitleEdit}
                onChange={(event) => setTitleDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    commitTitleEdit()
                  }
                  if (event.key === 'Escape') {
                    event.preventDefault()
                    setTitleDraft(document.title)
                    setIsEditingTitle(false)
                  }
                }}
                ref={titleInputRef}
                value={titleDraft}
              />
            ) : (
              <button
                className="min-w-0 flex-1 truncate rounded-sm text-left text-sm font-semibold hover:text-brand focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onDoubleClick={() => setIsEditingTitle(true)}
                title={copy.renameDocument}
                type="button"
              >
                {document.title}
              </button>
            )}
            {isDirty ? <span className="size-1.5 shrink-0 rounded-full bg-brand" aria-label="Unsaved changes" /> : null}
          </div>
          <p
            className="truncate text-[11px] leading-4 text-muted-foreground"
            title={
              document.source === 'imported-research-report' && document.sourceRunId
                ? `${copy.importedFrom} ${document.sourceRunId} · ${copy.updated} ${formatEditorTime(document.updatedAt)}`
                : undefined
            }
          >
            {document.source === 'imported-research-report' && document.sourceRunId
              ? `${copy.importedFrom} ${shortenRunId(document.sourceRunId)} · `
              : ''}
            {copy.updated} {formatEditorTime(document.updatedAt)}
          </p>
        </div>
      </div>
      <EditorCommandToolbar editor={editor} isSource={viewMode === 'source'} />
      <div className="flex min-w-0 justify-end gap-0.5">
        <Badge className="h-5 rounded-full px-1.5 text-[10px]" variant="outline">{commentCount}</Badge>
        <Badge className="h-5 rounded-full px-1.5 text-[10px]" variant="outline">R{document.revision}</Badge>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <TooltipButton
          label={viewMode === 'source' ? copy.live : copy.source}
          onClick={() => dispatch({ mode: viewMode === 'source' ? 'live' : 'source', type: 'setEditorViewMode' })}
        >
          {viewMode === 'source' ? <Eye className="size-4" /> : <Code2 className="size-4" />}
        </TooltipButton>
        <TooltipButton
          className={document.diffAnchorMarkdown ? 'bg-brand-subtle text-brand hover:bg-brand-subtle/80 hover:text-brand' : undefined}
          label={copy.setDiffAnchor}
          onClick={() => dispatch({ documentId: document.id, type: 'setEditorDiffAnchor' })}
        >
          <Anchor className="size-4" />
        </TooltipButton>
        <TooltipButton
          className={isDiffVisible ? 'bg-brand text-white hover:bg-brand/90 hover:text-white' : undefined}
          label={isDiffVisible ? copy.hideDiff : copy.showDiff}
          onClick={() => dispatch({ isVisible: !isDiffVisible, type: 'setEditorDiffVisible' })}
        >
          <Scale className="size-4" />
        </TooltipButton>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <TooltipButton
          disabled={isExporting}
          label={exportError ?? copy.exportWord}
          onClick={() => { void handleExportWord() }}
        >
          {isExporting ? <LoaderCircle className="size-4 animate-spin" /> : <FileDown className="size-4" />}
        </TooltipButton>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <TooltipButton
          label={copy.deleteDocument}
          onClick={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
        >
          <Trash2 className="size-4" />
        </TooltipButton>
      </div>
    </header>
  )
}

function MarkdownLiveEditor({
  comments,
  copy,
  diffAnchorMarkdown,
  document,
  isDiffVisible,
  mode,
  onChange,
  onCreateComment,
  onEditorReady,
  onAcceptSuggestion,
  onEditSuggestion,
  onRejectSuggestion,
  onMarkSuggestionStale,
  onRefineSuggestion,
  onSelectComment,
  onStopSuggestion,
  runningSuggestionIds,
  selectedCommentId,
  suggestionErrors,
  suggestions,
  textImprovement,
}: {
  comments: EditorCommentThreadRecord[]
  copy: EditorCopy
  diffAnchorMarkdown: string | null
  document: EditorDocumentRecord
  isDiffVisible: boolean
  mode: ProjectState['editorUi']['viewMode']
  onChange: (contentMarkdown: string) => void
  onCreateComment: (comment: EditorCommentThreadRecord) => void
  onEditorReady: (editor: Editor | null) => void
  onAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  onEditSuggestion: (suggestionId: string, proposedText: string) => void
  onMarkSuggestionStale: (suggestionId: string) => void
  onRefineSuggestion: (suggestionId: string, instruction: string) => Promise<void>
  onRejectSuggestion: (suggestionId: string) => void
  onSelectComment: (commentId: string) => void
  onStopSuggestion: (suggestionId: string) => void
  runningSuggestionIds: readonly string[]
  selectedCommentId: string | null
  suggestionErrors: Record<string, string>
  suggestions: EditorSuggestionRecord[]
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const documentIdRef = useRef(document.id)
  const editorInstanceRef = useRef<Editor | null>(null)
  const isApplyingExternalContentRef = useRef(false)
  const onAcceptSuggestionRef = useRef(onAcceptSuggestion)
  const onEditSuggestionRef = useRef(onEditSuggestion)
  const onMarkSuggestionStaleRef = useRef(onMarkSuggestionStale)
  const onRefineSuggestionRef = useRef(onRefineSuggestion)
  const onRejectSuggestionRef = useRef(onRejectSuggestion)
  const onSelectCommentRef = useRef(onSelectComment)
  const onStopSuggestionRef = useRef(onStopSuggestion)
  const suggestionsRef = useRef(suggestions)
  const previousModeRef = useRef(mode)
  const commentsSignature = comments.map((comment) => `${comment.id}:${comment.status}:${comment.kind}:${comment.anchor.from}:${comment.anchor.to}`).join('|')
  const suggestionsSignature = suggestions.map((suggestion) =>
    `${suggestion.id}:${suggestion.revision ?? 1}:${suggestion.editPosition ?? 'replace'}:${suggestion.anchorText ?? ''}:${suggestion.originalText.length}:${suggestion.proposedText}`).join('|')
  const suggestionUiSignature = suggestions.map((suggestion) =>
    `${suggestion.id}:${runningSuggestionIds.includes(suggestion.id) ? 'running' : 'idle'}:${suggestionErrors[suggestion.id] ?? ''}`).join('|')
  const tiptapContentMarkdown = normalizeEditorMarkdownForTiptap(document.contentMarkdown)

  useEffect(() => {
    suggestionsRef.current = suggestions
  }, [suggestions])

  useEffect(() => {
    onAcceptSuggestionRef.current = onAcceptSuggestion
    onEditSuggestionRef.current = onEditSuggestion
    onMarkSuggestionStaleRef.current = onMarkSuggestionStale
    onRefineSuggestionRef.current = onRefineSuggestion
    onRejectSuggestionRef.current = onRejectSuggestion
    onSelectCommentRef.current = onSelectComment
    onStopSuggestionRef.current = onStopSuggestion
  }, [onAcceptSuggestion, onEditSuggestion, onMarkSuggestionStale, onRefineSuggestion, onRejectSuggestion, onSelectComment, onStopSuggestion])

  const editor = useEditor({
    content: tiptapContentMarkdown,
    contentType: 'markdown',
    editable: mode === 'live',
    editorProps: {
      attributes: {
        class: 'editor-prose min-h-full focus:outline-none',
      },
      handlePaste: (_view, event) => {
        const pastedMarkdown = event.clipboardData?.getData('text/plain') ?? ''
        const currentEditor = editorInstanceRef.current
        if (!currentEditor?.isEditable || !shouldParsePastedMarkdown(pastedMarkdown)) return false
        event.preventDefault()
        currentEditor.commands.insertContent(normalizeEditorMarkdownForTiptap(pastedMarkdown), {
          contentType: 'markdown',
        })
        return true
      },
    },
    extensions: createEditorExtensions({
      onClick: (commentId) => onSelectCommentRef.current(commentId),
      onSuggestionAccept: (suggestionId) => {
        const suggestion = suggestionsRef.current.find((item) => item.id === suggestionId)
        if (suggestion) onAcceptSuggestionRef.current(suggestion)
      },
      onSuggestionReject: (suggestionId) => onRejectSuggestionRef.current(suggestionId),
      onSuggestionEdit: (suggestionId, proposedText) => onEditSuggestionRef.current(suggestionId, proposedText),
      onSuggestionRefine: (suggestionId, instruction) => {
        void onRefineSuggestionRef.current(suggestionId, instruction)
      },
      onSuggestionCancel: (suggestionId) => onStopSuggestionRef.current(suggestionId),
      onSuggestionSelect: (suggestionId) => {
        const suggestion = suggestionsRef.current.find((item) => item.id === suggestionId)
        if (suggestion?.origin.commentId) onSelectCommentRef.current(suggestion.origin.commentId)
      },
    }),
    immediatelyRender: false,
    onCreate: ({ editor: createdEditor }) => {
      editorInstanceRef.current = createdEditor
    },
    onDestroy: () => {
      editorInstanceRef.current = null
    },
    onUpdate: ({ editor: currentEditor }) => {
      if (isApplyingExternalContentRef.current || !currentEditor.isEditable) return
      onChange(currentEditor.getMarkdown())
    },
  })

  useEffect(() => {
    onEditorReady(editor)
    return () => onEditorReady(null)
  }, [editor, onEditorReady])

  useEffect(() => {
    if (!editor || documentIdRef.current === document.id) return
    documentIdRef.current = document.id
    isApplyingExternalContentRef.current = true
    editor.commands.setContent(tiptapContentMarkdown, {
      contentType: 'markdown',
      emitUpdate: false,
    })
    resetExternalContentFlag(isApplyingExternalContentRef)
  }, [document.id, editor, tiptapContentMarkdown])

  useEffect(() => {
    if (!editor) return
    const previousMode = previousModeRef.current
    previousModeRef.current = mode
    if (mode !== 'live') return
    const shouldReparseMarkdown = previousMode === 'source'
    if (!shouldReparseMarkdown && editor.getMarkdown() === tiptapContentMarkdown) return
    isApplyingExternalContentRef.current = true
    editor.commands.setContent(tiptapContentMarkdown, {
      contentType: 'markdown',
      emitUpdate: false,
    })
    resetExternalContentFlag(isApplyingExternalContentRef)
  }, [editor, mode, tiptapContentMarkdown])

  useEffect(() => {
    editor?.setEditable(mode === 'live')
  }, [editor, mode])

  useEffect(() => {
    if (!editor || mode !== 'live') return
    const items = comments
      .filter((comment) => comment.status !== 'resolved')
      .map((comment) => {
        const resolved = resolveMaterializedAnchor(editor, comment.anchor)
        if (!resolved) return null
        return {
          from: resolved.range.from,
          id: comment.id,
          kind: comment.kind,
          selected: selectedCommentId === comment.id,
          status: comment.status,
          to: resolved.range.to,
        }
      })
      .filter((item): item is NonNullable<typeof item> => Boolean(item))
      .filter((item) => item.from < item.to)
    isApplyingExternalContentRef.current = true
    editor.view.dispatch(editor.state.tr.setMeta(commentDecorationPluginKey, { items }))
    resetExternalContentFlag(isApplyingExternalContentRef)
  }, [commentsSignature, document.revision, editor, mode, selectedCommentId])

  useEffect(() => {
    if (!editor || mode !== 'live') return
    const staleSuggestionIds: string[] = []
    const items = suggestions.flatMap((suggestion) => {
      const target = resolveSuggestionDecorationTarget(editor, suggestion)
      if (!target) {
        staleSuggestionIds.push(suggestion.id)
        return []
      }
      const plan = suggestionDiffPlan(suggestion.originalText, suggestion.proposedText)
      return [{
        acceptLabel: copy.accept,
        active: selectedCommentId === suggestion.origin.commentId,
        display: plan.display,
        editLabel: copy.editSuggestion,
        error: suggestionErrors[suggestion.id],
        from: target.from,
        id: suggestion.id,
        isRunning: runningSuggestionIds.includes(suggestion.id),
        proposedLabel: copy.proposedText,
        proposedText: suggestion.proposedText,
        refineLabel: copy.refineSuggestion,
        refinementPlaceholder: copy.refinementPlaceholder,
        rejectLabel: copy.reject,
        revision: suggestion.revision ?? 1,
        revisionLabel: copy.revision,
        reviewSurface: plan.reviewSurface,
        saveLabel: copy.saveSuggestion,
        segments: plan.segments,
        cancelLabel: copy.cancelEdit,
        sendLabel: copy.sendRefinement,
        runningLabel: copy.refiningSuggestion,
        stopLabel: copy.stopRun,
        to: target.to,
        widgetAt: plan.display === 'block' ? target.widgetAt : undefined,
      }]
    })
    isApplyingExternalContentRef.current = true
    editor.view.dispatch(editor.state.tr.setMeta(suggestionDecorationPluginKey, { items }))
    resetExternalContentFlag(isApplyingExternalContentRef)
    for (const suggestionId of staleSuggestionIds) onMarkSuggestionStaleRef.current(suggestionId)
  }, [copy.accept, copy.cancelEdit, copy.editSuggestion, copy.proposedText, copy.refineSuggestion, copy.refinementPlaceholder, copy.reject, copy.revision, copy.saveSuggestion, copy.sendRefinement, copy.refiningSuggestion, copy.stopRun, document.revision, editor, mode, selectedCommentId, suggestionUiSignature, suggestionsSignature])

  useEffect(() => {
    if (!selectedCommentId) return
    const target = globalThis.document?.querySelector<HTMLElement>(
      `[data-editor-comment-anchor="${escapeCssIdentifier(selectedCommentId)}"]`,
    )
    target?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [selectedCommentId])

  if (mode === 'source') {
    return (
      <MarkdownSourceEditor
        labels={{
          addColumn: copy.addColumn,
          addRow: copy.addRow,
          closeTableEditor: copy.closeTableEditor,
          columnLabel: copy.columnLabel,
          deleteColumn: copy.deleteColumn,
          deleteRow: copy.deleteRow,
          editor: copy.sourceEditor,
          formatTables: copy.formatTables,
          insertOrEditTable: copy.insertOrEditTable,
          lineWrap: copy.sourceLineWrap,
          tableAlignmentCenter: copy.tableAlignmentCenter,
          tableAlignmentLeft: copy.tableAlignmentLeft,
          tableAlignmentRight: copy.tableAlignmentRight,
          tableColumn: copy.tableColumn,
          tableEditor: copy.tableEditor,
          tableLines: copy.tableLines,
          tableRows: copy.tableRows,
        }}
        onChange={onChange}
        value={document.contentMarkdown}
      />
    )
  }

  if (isDiffVisible) {
    return (
      <EditorDocumentDiffView
        anchorMarkdown={diffAnchorMarkdown}
        copy={copy}
        currentMarkdown={document.contentMarkdown}
      />
    )
  }

  return (
    <ScrollArea className="min-h-0 flex-1 bg-background">
      <div className="min-h-[calc(100svh-var(--header-h)-10rem)] w-full px-10 py-8">
        {editor ? (
          <EditorBubbleMenu
            copy={copy}
            editor={editor}
            onCreateComment={(commentMarkdown, kind) => {
              const comment = createCommentFromSelection(editor, document.id, commentMarkdown, kind)
              if (!comment) return
              onCreateComment(comment)
            }}
            textImprovement={textImprovement}
          />
        ) : null}
        <EditorContent className="min-h-full" editor={editor} />
      </div>
    </ScrollArea>
  )
}

function resolveSuggestionDecorationTarget(
  editor: Editor,
  suggestion: EditorSuggestionRecord,
): { from: number; to: number; widgetAt: number } | null {
  const position = suggestion.editPosition ?? 'replace'
  if (position === 'append') {
    const end = editor.state.doc.content.size
    return { from: end, to: end, widgetAt: end }
  }
  const anchorText = (suggestion.anchorText ?? suggestion.originalText).trim()
  if (!anchorText) return null
  const range = resolveAnchorRange(editor, {
    hint: clampAnchor(suggestion.anchor, editor).from,
    quoteAfter: suggestion.anchor.quoteAfter,
    quoteBefore: suggestion.anchor.quoteBefore,
    text: anchorText,
  })
  if (!range) return null
  if (position === 'replace') {
    return { ...range, widgetAt: blockWidgetPositionForRange(editor, range) }
  }
  const at = blockInsertionPositionForRange(editor, range, position)
  return { from: at, to: at, widgetAt: at }
}

function EditorDocumentDiffView({
  anchorMarkdown,
  copy,
  currentMarkdown,
}: {
  anchorMarkdown: string | null
  copy: EditorCopy
  currentMarkdown: string
}) {
  if (!anchorMarkdown) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center bg-background p-8">
        <div className="rounded-md border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
          {copy.noDiffAnchor}
        </div>
      </div>
    )
  }
  const blocks = documentDiffPlan(anchorMarkdown, currentMarkdown)
  return (
    <ScrollArea className="min-h-0 flex-1 bg-background">
      <div className="editor-document-diff mx-auto min-h-full max-w-[72rem] px-4 py-6 sm:px-10 sm:py-8">
        <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-brand">
          <Scale className="size-3.5" />
          {copy.diffView}
        </div>
        <div className="editor-document-diff-body editor-prose">
          {blocks.map((block, index) => (
            <EditorDocumentDiffBlock block={block} index={index} key={documentDiffBlockKey(block, index)} />
          ))}
        </div>
      </div>
    </ScrollArea>
  )
}

function EditorDocumentDiffBlock({ block, index }: { block: DocumentDiffBlock; index: number }) {
  if (block.kind === 'replace') {
    if (block.inlineSegments) {
      return (
        <div className="editor-document-diff-replace editor-document-diff-replace-inline">
          <p className="editor-document-diff-inline-row">
            {block.inlineSegments.map((segment, segmentIndex) => (
              <EditorDocumentDiffInlineSegment
                key={`${index}-${segmentIndex}-${segment.type}-${segment.text.length}`}
                segment={segment}
              />
            ))}
          </p>
        </div>
      )
    }

    return (
      <div className="editor-document-diff-replace editor-document-diff-replace-structured">
        <div className="editor-document-diff-layer editor-document-diff-delete">
          <MarkdownRenderer markdown={block.beforeMarkdown} variant="report" />
        </div>
        <div className="editor-document-diff-layer editor-document-diff-insert">
          <MarkdownRenderer markdown={block.afterMarkdown} variant="report" />
        </div>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'editor-document-diff-chunk',
        block.kind === 'equal' && 'editor-document-diff-equal',
        block.kind === 'insert' && 'editor-document-diff-layer editor-document-diff-insert',
        block.kind === 'delete' && 'editor-document-diff-layer editor-document-diff-delete',
      )}
    >
      <MarkdownRenderer markdown={block.markdown} variant="report" />
    </div>
  )
}

function EditorDocumentDiffInlineSegment({ segment }: { segment: SuggestionDiffSegment }) {
  if (segment.type === 'insert') {
    return (
      <ins className="editor-document-diff-token editor-document-diff-token-insert">
        {renderInlineMarkdownText(segment.text)}
      </ins>
    )
  }
  if (segment.type === 'delete') {
    return (
      <del className="editor-document-diff-token editor-document-diff-token-delete">
        {renderInlineMarkdownText(segment.text)}
      </del>
    )
  }
  return (
    <span className="editor-document-diff-token">
      {renderInlineMarkdownText(segment.text)}
    </span>
  )
}

function documentDiffBlockKey(block: DocumentDiffBlock, index: number): string {
  if (block.kind === 'replace') {
    return `${block.kind}-${index}-${block.beforeMarkdown.length}-${block.afterMarkdown.length}`
  }
  return `${block.kind}-${index}-${block.markdown.length}`
}

function renderInlineMarkdownText(text: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const tokenPattern = /(\[[^\]\n]+\]\([^) \n]+(?:\s+"[^"\n]*")?\)|`[^`\n]+`|\*\*[^*\n][^*\n]*\*\*|\*[^*\n][^*\n]*\*)/g
  let cursor = 0
  let match: RegExpExecArray | null
  while ((match = tokenPattern.exec(text))) {
    if (match.index > cursor) {
      nodes.push(<Fragment key={`text-${cursor}`}>{text.slice(cursor, match.index)}</Fragment>)
    }
    nodes.push(renderInlineMarkdownToken(match[0], match.index))
    cursor = match.index + match[0].length
  }
  if (cursor < text.length) {
    nodes.push(<Fragment key={`text-${cursor}`}>{text.slice(cursor)}</Fragment>)
  }
  return nodes
}

function renderInlineMarkdownToken(token: string, index: number): ReactNode {
  const link = token.match(/^\[([^\]\n]+)\]\(([^) \n]+)(?:\s+"[^"\n]*")?\)$/)
  if (link) {
    return (
      <a
        className="editor-document-diff-inline-link"
        href={link[2]}
        key={`link-${index}`}
        rel="noreferrer"
        target="_blank"
      >
        {link[1]}
      </a>
    )
  }
  if (token.startsWith('`') && token.endsWith('`')) {
    return <code className="editor-document-diff-inline-code" key={`code-${index}`}>{token.slice(1, -1)}</code>
  }
  if (token.startsWith('**') && token.endsWith('**')) {
    return <strong key={`strong-${index}`}>{token.slice(2, -2)}</strong>
  }
  if (token.startsWith('*') && token.endsWith('*')) {
    return <em key={`em-${index}`}>{token.slice(1, -1)}</em>
  }
  return <Fragment key={`token-${index}`}>{token}</Fragment>
}

function EditorCommandToolbar({
  editor,
  isSource,
}: {
  editor: Editor | null
  isSource: boolean
}) {
  const disabled = !editor || isSource
  const setLink = () => {
    if (!editor) return
    const previousUrl = editor.getAttributes('link').href as string | undefined
    const url = window.prompt('URL', previousUrl ?? 'https://')
    if (url === null) return
    if (!url.trim()) {
      editor.chain().focus().unsetLink().run()
      return
    }
    editor.chain().focus().extendMarkRange('link').setLink({ href: url.trim() }).run()
  }

  return (
    <div className="flex min-w-0 items-center justify-center gap-0.5 overflow-x-auto px-1 [scrollbar-width:none]">
      <ToolbarButton disabled={disabled} icon={Undo2} label="Undo" onClick={() => editor?.chain().focus().undo().run()} />
      <ToolbarButton disabled={disabled} icon={Redo2} label="Redo" onClick={() => editor?.chain().focus().redo().run()} />
      <Separator className="mx-0.5 h-5" orientation="vertical" />
      <ToolbarButton active={editor?.isActive('bulletList')} disabled={disabled} icon={List} label="Bullet list" onClick={() => editor?.chain().focus().toggleBulletList().run()} />
      <ToolbarButton active={editor?.isActive('orderedList')} disabled={disabled} icon={ListOrdered} label="Ordered list" onClick={() => editor?.chain().focus().toggleOrderedList().run()} />
      <Separator className="mx-0.5 h-5" orientation="vertical" />
      <ToolbarButton active={editor?.isActive('link')} disabled={disabled} icon={Link} label="Link" onClick={setLink} />
    </div>
  )
}

function EditorBubbleMenu({
  copy,
  editor,
  onCreateComment,
  textImprovement,
}: {
  copy: EditorCopy
  editor: Editor
  onCreateComment: (commentMarkdown: string, kind: EditorCommentKind) => void
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const { locale, t } = useLocale()
  const reduceMotion = useReducedMotion()
  const [isCommenting, setIsCommenting] = useState(false)
  const [commentDraft, setCommentDraft] = useState('')
  const [commentKind, setCommentKind] = useState<EditorCommentKind>('collect')
  const [commentImproveError, setCommentImproveError] = useState<string | null>(null)
  const commentTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const commentTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })

  useLayoutEffect(() => {
    if (!isCommenting) return
    resizeTextareaToRows(commentTextareaRef.current, 6)
  }, [commentDraft, isCommenting])

  function closeCommentComposer() {
    const collapseAt = editor.state.selection.to
    editor.commands.setTextSelection(collapseAt)
    editor.commands.blur()
    setCommentDraft('')
    setCommentKind('collect')
    setCommentImproveError(null)
    commentTextImprove.clearProposal()
    setIsCommenting(false)
  }

  function cancelComment() {
    closeCommentComposer()
  }

  function submitComment() {
    const value = commentDraft.trim()
    if (!value) return
    onCreateComment(value, commentKind)
    closeCommentComposer()
  }

  function handleCommentDraftChange(value: string) {
    setCommentDraft(value)
    setCommentImproveError(null)
    commentTextImprove.clearProposal()
  }

  async function improveCommentDraft() {
    setCommentImproveError(null)
    try {
      await commentTextImprove.improve('chat_input', commentDraft)
    } catch (error) {
      setCommentImproveError(messageFromUnknown(error))
    }
  }

  function acceptCommentImprovement(text: string) {
    handleCommentDraftChange(text)
    window.requestAnimationFrame(() => {
      commentTextareaRef.current?.focus()
      resizeTextareaToRows(commentTextareaRef.current, 6)
    })
  }

  return (
    <BubbleMenu
      editor={editor}
      appendTo={() => globalThis.document.body}
      options={{
        flip: { padding: { bottom: 132, left: 12, right: 12, top: 12 } },
        inline: true,
        offset: 8,
        placement: 'top-start',
        shift: { padding: { bottom: 132, left: 12, right: 12, top: 12 } },
        strategy: 'fixed',
      }}
      shouldShow={({ editor: currentEditor, state }) => {
        const { empty } = state.selection
        return currentEditor.isEditable && !empty
      }}
    >
      <div className="z-50 flex min-w-0 items-center gap-1 rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-lg">
        {isCommenting ? (
          <form
            className="relative flex w-[26rem] max-w-[calc(100vw-5rem)] flex-col gap-2 p-1.5"
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault()
                cancelComment()
              }
            }}
            onSubmit={(event) => {
              event.preventDefault()
              submitComment()
            }}
          >
            <button
              aria-label={copy.cancel}
              className="absolute right-2 top-2 z-10 inline-grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              onClick={cancelComment}
              type="button"
            >
              <X className="size-3.5" />
            </button>
            <TextImproveFloatingLayer
              labels={{
                accept: t.textImprove.accept,
                changes: t.textImprove.changes,
                noChanges: t.textImprove.noChanges,
                reject: t.textImprove.reject,
                title: t.textImprove.title,
                warnings: t.textImprove.warnings,
              }}
              onAccept={acceptCommentImprovement}
              onReject={commentTextImprove.clearProposal}
              proposal={commentTextImprove.proposal}
              reduceMotion={reduceMotion}
            />
            <Textarea
              autoFocus
              className="min-h-16 resize-none border-border/70 bg-background/60 pr-16 text-sm focus-visible:ring-1 [scrollbar-width:thin]"
              onChange={(event) => handleCommentDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  submitComment()
                }
              }}
              placeholder={copy.inlineComment}
              ref={commentTextareaRef}
              value={commentDraft}
            />
            <TextImproveButton
              className="absolute right-9 top-2 z-10"
              disabled={!commentDraft.trim()}
              isLoading={commentTextImprove.isImproving}
              label={t.textImprove.improve}
              loadingLabel={t.textImprove.improving}
              onClick={() => void improveCommentDraft()}
              reduceMotion={reduceMotion}
            />
            {commentImproveError ? (
              <p className="rounded-md border border-destructive/20 bg-destructive/5 px-2 py-1 text-[11px] leading-4 text-destructive">
                {commentImproveError}
              </p>
            ) : null}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1">
                {COMMENT_KIND_ORDER.map((kind) => {
                  const kindMeta = commentKindMeta(kind, copy)
                  const KindIcon = kindMeta.Icon
                  const active = commentKind === kind
                  return (
                    <button
                      aria-pressed={active}
                      className={cn(
                        'inline-flex h-6 shrink-0 items-center gap-1 rounded-full border px-2 text-[11px] font-medium transition-colors',
                        active
                          ? cn(kindMeta.selectedBorderClass, kindMeta.selectedBgClass, kindMeta.accentText)
                          : 'border-border text-muted-foreground hover:text-foreground',
                      )}
                      key={kind}
                      onClick={() => setCommentKind(kind)}
                      type="button"
                    >
                      <KindIcon className="size-3" />
                      {kindMeta.label}
                    </button>
                  )
                })}
              </div>
              <Button disabled={!commentDraft.trim()} size="sm" type="submit">
                {copy.inlineCommentSubmit}
              </Button>
            </div>
          </form>
        ) : (
          <>
            <MiniToolbarButton active={editor.isActive('bold')} icon={Bold} label="Bold" onClick={() => editor.chain().focus().toggleBold().run()} />
            <MiniToolbarButton active={editor.isActive('italic')} icon={Italic} label="Italic" onClick={() => editor.chain().focus().toggleItalic().run()} />
            <MiniToolbarButton active={editor.isActive('strike')} icon={Strikethrough} label="Strike" onClick={() => editor.chain().focus().toggleStrike().run()} />
            <MiniToolbarButton active={editor.isActive('underline')} icon={Underline} label="Underline" onClick={() => editor.chain().focus().toggleUnderline().run()} />
            <MiniToolbarButton active={editor.isActive('highlight')} icon={Highlighter} label="Highlight" onClick={() => editor.chain().focus().toggleHighlight().run()} />
            <Separator className="mx-1 h-6" orientation="vertical" />
            <MiniToolbarButton icon={MessageSquarePlus} label="Comment" onClick={() => setIsCommenting(true)} />
          </>
        )}
      </div>
    </BubbleMenu>
  )
}

function EditorAssistantComposer({
  attachedCommentIds,
  attachmentChips,
  chatModelOptions,
  chatModelOptionsStatus,
  comments,
  composerRef,
  copy,
  defaultChatModel,
  dispatch,
  draft,
  fileGroupOptions,
  fileOptions,
  reportOptions,
  isAttachActive,
  instructionFeedback,
  isRunning,
  isVisible,
  isWideCanvas,
  onAttachFiles,
  onAttachRule,
  onRefsChange,
  onReorderPending,
  onReorderPill,
  onRemoveAttachedComment,
  onRemoveChip,
  pendingKeys,
  pillKeys,
  onSend,
  onDismissInstructionFeedback,
  onStop,
  onToggleAttach,
  ruleOptions,
  selectedModelTier,
  textImprovement,
}: {
  attachedCommentIds: string[]
  attachmentChips: ChatAttachmentChipModel[]
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  comments: EditorCommentThreadRecord[]
  composerRef: RefObject<MentionComposerHandle | null>
  copy: EditorCopy
  defaultChatModel: NodeModelResolution | null
  dispatch: Dispatch<ResearchDeskAction>
  draft: string
  fileGroupOptions: FileGroupMentionOption[]
  fileOptions: FileMentionOption[]
  reportOptions: CompletedReportOption[]
  isAttachActive: boolean
  instructionFeedback: EditorInstructionFeedback | null
  isRunning: boolean
  isVisible: boolean
  isWideCanvas: boolean
  onAttachFiles: (files: File[]) => void
  onAttachRule: (ruleId: string) => void
  onRefsChange: (refs: ChatContextReferenceRecord[]) => void
  onReorderPending: (fromIndex: number, toIndex: number) => void
  onReorderPill: (fromIndex: number, toIndex: number) => void
  onRemoveAttachedComment: (commentId: string) => void
  onRemoveChip: (ref: ChatContextReferenceRecord) => void
  pendingKeys: string[]
  pillKeys: string[]
  onSend: () => void
  onDismissInstructionFeedback: () => void
  onStop: () => void
  onToggleAttach: () => void
  ruleOptions: ChatRuleOption[]
  selectedModelTier: ChatModelTier | null
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const { locale, t } = useLocale()
  const reduceMotion = useReducedMotion()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [improveError, setImproveError] = useState<string | null>(null)
  const assistantTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })

  const attachedComments = comments.filter(
    (comment) => comment.status === 'open'
      && comment.kind === 'collect'
      && attachedCommentIds.includes(comment.id),
  )
  const commentNumbers = buildCommentNumbers(comments)
  const mentionCategoryLabels: MentionCategoryLabels = {
    files: t.chat.mentionFilesCategory,
    filegroups: t.chat.mentionFilegroupsCategory,
    research: t.chat.mentionResearchCategory,
    rules: t.chat.mentionRulesCategory,
  }
  const mentionSources: MentionSources = { fileGroupOptions, fileOptions, reportOptions, ruleOptions }
  const resolveMentionLabel: LabelResolver = (kind, label) => {
    if (kind === 'file-asset') {
      const option = fileOptions.find((file) => file.label === label)
      return option ? { id: option.fileId, label: option.label } : null
    }
    if (kind === 'file-group') {
      const option = fileGroupOptions.find((group) => group.label === label)
      return option ? { id: option.groupId, label: option.label } : null
    }
    const option = reportOptions.find((report) => report.label === label)
    return option ? { id: option.runId, label: option.label } : null
  }

  function handleComposerChange() {
    dispatch({ draft: composerRef.current?.getMentionText() ?? '', type: 'setEditorAssistantDraft' })
    setImproveError(null)
    assistantTextImprove.clearProposal()
  }

  async function improveDraft() {
    setImproveError(null)
    try {
      await assistantTextImprove.improve('chat_input', composerRef.current?.getMentionText() ?? draft)
    } catch (error) {
      setImproveError(messageFromUnknown(error))
    }
  }

  function acceptDraftImprovement(text: string) {
    composerRef.current?.setMentionText(text)
    dispatch({ draft: composerRef.current?.getMentionText() ?? text, type: 'setEditorAssistantDraft' })
    assistantTextImprove.clearProposal()
    setImproveError(null)
    window.requestAnimationFrame(() => composerRef.current?.focus())
  }

  // Restore the persisted draft (with its `[N]` pills) once when the composer
  // mounts: the assistant draft survives hiding/showing the panel and switching
  // documents, mirroring the previous textarea behaviour.
  const didRestoreRef = useRef(false)
  useEffect(() => {
    if (didRestoreRef.current) return
    didRestoreRef.current = true
    if (draft) composerRef.current?.setMentionText(draft)
  }, [draft, composerRef])

  if (!isVisible) {
    return (
      <div className="shrink-0 px-4 pb-4 pt-2">
        <Button
          className="h-8 rounded-md"
          onClick={() => dispatch({ isVisible: true, type: 'setEditorAssistantVisible' })}
          size="sm"
          type="button"
          variant="outline"
        >
          <PanelBottomOpen className="size-4" />
          {copy.showAssistant}
        </Button>
      </div>
    )
  }

  return (
    <div className="relative z-10 shrink-0 px-4 pb-4 pt-2">
      {isWideCanvas ? (
        <>
          <div
            aria-hidden="true"
            className="pointer-events-none absolute bottom-0 left-4 top-2 hidden rounded-r-xl bg-background/30 backdrop-blur-xl xl:block xl:right-[calc(50%+28rem)]"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute bottom-0 right-4 top-2 hidden rounded-l-xl bg-background/30 backdrop-blur-xl xl:left-[calc(50%+28rem)] xl:block"
          />
        </>
      ) : null}
      <div className="relative mx-auto max-w-4xl">
        <AnimatePresence initial={false}>
          {isAttachActive ? (
            <motion.div
              animate={{ height: 'auto', opacity: 1 }}
              className="mb-2 overflow-hidden rounded-md border border-brand/30 bg-brand-subtle/30"
              exit={{ height: 0, opacity: 0 }}
              initial={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <div className="flex items-center gap-1.5 px-3 pb-1 pt-2 text-[11px] font-semibold text-brand">
                <MessagesSquare className="size-3.5" />
                {attachedComments.length} {copy.attachedComments}
              </div>
              <div className="flex flex-wrap gap-1.5 px-3 pb-2.5">
                <AnimatePresence initial={false}>
                  {attachedComments.map((comment) => (
                    <motion.span
                      animate={{ opacity: 1, scale: 1 }}
                      className="inline-flex max-w-full items-center gap-1 rounded-full border border-brand/30 bg-background px-2 py-0.5 text-[11px] text-foreground"
                      exit={{ opacity: 0, scale: 0.85 }}
                      initial={{ opacity: 0, scale: 0.85 }}
                      key={comment.id}
                      layout
                      transition={{ duration: 0.15 }}
                    >
                      <span className="grid size-4 shrink-0 place-items-center rounded-[4px] bg-brand-subtle text-[9px] font-semibold tabular-nums text-brand">
                        {commentNumbers.get(comment.id) ?? 0}
                      </span>
                      <span className="max-w-40 truncate text-muted-foreground">“{comment.anchor.selectedText}”</span>
                      <button
                        aria-label={copy.removeFromQueue}
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => onRemoveAttachedComment(comment.id)}
                        type="button"
                      >
                        <X className="size-3" />
                      </button>
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
        <EditorInstructionFeedbackCard
          feedback={instructionFeedback}
          labels={{
            assistantDone: copy.assistantDone,
            assistantThinking: copy.assistantThinking,
            dismiss: copy.reject,
            hide: copy.hideAssistant,
            show: copy.showAssistant,
          }}
          onDismiss={onDismissInstructionFeedback}
        />
        <input
          className="hidden"
          multiple
          onChange={(event) => {
            const files = Array.from(event.target.files ?? [])
            if (files.length > 0) onAttachFiles(files)
            event.target.value = ''
          }}
          ref={fileInputRef}
          type="file"
        />
        <Dropzone disabled={isRunning} label={t.chat.dropFiles} onFiles={onAttachFiles}>
        <div className="relative rounded-xl border border-border bg-card px-3 py-2 shadow-[0_8px_28px_-12px_var(--shadow-soft)] transition-[border-color,box-shadow] duration-150 focus-within:border-brand/60 focus-within:ring-2 focus-within:ring-brand/15">
          <TextImproveFloatingLayer
            labels={{
              accept: t.textImprove.accept,
              changes: t.textImprove.changes,
              noChanges: t.textImprove.noChanges,
              reject: t.textImprove.reject,
              title: t.textImprove.title,
              warnings: t.textImprove.warnings,
            }}
            onAccept={acceptDraftImprovement}
            onReject={assistantTextImprove.clearProposal}
            proposal={assistantTextImprove.proposal}
            reduceMotion={reduceMotion}
          />
          <ContextChipLegend
            chips={attachmentChips}
            labels={{
              removeContext: copy.removeFromQueue,
              reorderHint: t.chat.reorderContextHint,
            }}
            onRemove={onRemoveChip}
            onReorderPending={onReorderPending}
            onReorderPill={onReorderPill}
            pendingKeys={pendingKeys}
            pillKeys={pillKeys}
          />
          <MentionComposer
            ariaLabel={copy.assistantPlaceholder}
            categoryLabels={mentionCategoryLabels}
            contentClassName="min-h-16 pb-2 pl-2 pr-9 pt-2 text-sm leading-6"
            enabledKinds={['research', 'rules', 'files', 'filegroups']}
            maxRows={6}
            mentionSources={mentionSources}
            onAttachRule={onAttachRule}
            onChange={handleComposerChange}
            onRefsChange={onRefsChange}
            onSubmit={onSend}
            placeholder={copy.assistantPlaceholder}
            ref={composerRef}
            resolveLabel={resolveMentionLabel}
          />
          <TextImproveButton
            className="absolute right-3 top-2"
            disabled={!draft.trim() || isRunning}
            isLoading={assistantTextImprove.isImproving}
            label={t.textImprove.improve}
            loadingLabel={t.textImprove.improving}
            onClick={() => void improveDraft()}
            reduceMotion={reduceMotion}
          />
          {improveError ? (
            <p className="mb-1 rounded-md border border-destructive/20 bg-destructive/5 px-2 py-1 text-[11px] leading-4 text-destructive">
              {improveError}
            </p>
          ) : null}
          <div className="flex items-center justify-between gap-2 border-t border-border/70 pt-1.5">
            <div className="flex min-w-0 items-center gap-1">
              <ComposerIconButton
                icon={PanelBottomClose}
                label={copy.hideAssistant}
                onClick={() => dispatch({ isVisible: false, type: 'setEditorAssistantVisible' })}
              />
              <ComposerIconButton
                active={isAttachActive}
                icon={MessageSquareText}
                label={copy.attachComments}
                onClick={onToggleAttach}
              />
              <ComposerIconButton
                icon={Paperclip}
                label={t.chat.attachFiles}
                onClick={() => fileInputRef.current?.click()}
              />
              <EditorModelPicker
                defaultModel={defaultChatModel}
                disabled={false}
                onChange={(tier) => dispatch({ tier, type: 'setSelectedChatModelTier' })}
                options={chatModelOptions}
                optionsStatus={chatModelOptionsStatus}
                selectedTier={selectedModelTier}
              />
            </div>
            <Button
              aria-label={isRunning ? copy.stopRun : copy.send}
              className={cn(
                'size-7 rounded-md',
                !isRunning && attachedComments.length === 0 && draft.trim().length === 0
                  ? 'text-muted-foreground/45'
                  : 'bg-brand text-white hover:bg-brand/90 hover:text-white',
              )}
              disabled={!isRunning && attachedComments.length === 0 && draft.trim().length === 0}
              onClick={isRunning ? onStop : onSend}
              size="icon"
              type="button"
              variant={!isRunning && attachedComments.length === 0 && draft.trim().length === 0 ? 'ghost' : 'default'}
            >
              {isRunning ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : (
                <SendHorizontal className="size-4" />
              )}
            </Button>
          </div>
        </div>
        </Dropzone>
      </div>
    </div>
  )
}

const editorModelTierOrder: ChatModelTier[] = ['high', 'mid', 'fast']

function EditorModelPicker({
  defaultModel,
  disabled,
  onChange,
  options,
  optionsStatus,
  selectedTier,
}: {
  defaultModel: NodeModelResolution | null
  disabled: boolean
  onChange: (tier: ChatModelTier | null) => void
  options: ChatModelOption[]
  optionsStatus: 'available' | 'missing' | 'unresolved'
  selectedTier: ChatModelTier | null
}) {
  const { t } = useLocale()
  const selectedOption = selectedTier ? editorModelOptionForTier(options, selectedTier) : null
  const activeModel = selectedOption ?? defaultModel ?? editorModelOptionForTier(options, 'mid') ?? null
  const unavailableLabel = optionsStatus === 'unresolved'
    ? t.chat.modelMetadataMissing
    : t.chat.modelDiscoveryMissing
  const activeLabel = selectedTier && optionsStatus !== 'available'
    ? `${editorTierLabel(selectedTier, t)} · ${unavailableLabel}`
    : `${editorModelNameLabel(activeModel, t.chat.modelUnknown)} · ${editorEffortLabel(activeModel, t)}`
  const pickerValue = selectedTier ?? 'default'

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={t.chat.modelPicker}
          className="h-7 min-w-0 max-w-[min(48vw,17rem)] shrink rounded-md px-1.5 text-[11px] font-semibold text-muted-foreground hover:bg-accent/70 hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0 data-[state=open]:bg-accent data-[state=open]:text-foreground"
          disabled={disabled}
          type="button"
          variant="ghost"
        >
          <span className="min-w-0 truncate">{activeLabel}</span>
          <ChevronDown className="size-3 shrink-0 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-80 max-w-[calc(100vw-2rem)]" side="top" sideOffset={8}>
        <DropdownMenuLabel className="text-xs text-muted-foreground">
          {t.chat.modelPicker}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={(value) => onChange(isEditorModelTier(value) ? value : null)}
          value={pickerValue}
        >
          <DropdownMenuRadioItem className="items-start py-2 pr-3" value="default">
            <span className="grid min-w-0 flex-1 text-left leading-tight">
              <span className="truncate text-sm font-medium">{t.chat.modelServerDefault}</span>
              <span className="truncate text-xs text-muted-foreground">
                {editorModelDetailLabel(defaultModel, t)}
              </span>
            </span>
          </DropdownMenuRadioItem>
          <DropdownMenuSeparator />
          {optionsStatus === 'available' ? editorModelTierOrder.map((tier) => {
            const option = editorModelOptionForTier(options, tier)
            return (
              <DropdownMenuRadioItem className="items-start py-2 pr-3" key={tier} value={tier}>
                <span className="grid min-w-0 flex-1 text-left leading-tight">
                  <span className="flex min-w-0 items-baseline gap-2">
                    <span className="shrink-0 text-sm font-medium">{editorTierLabel(tier, t)}</span>
                    <span className="min-w-0 truncate text-xs font-medium text-muted-foreground">
                      {editorModelNameLabel(option, t.chat.modelUnknown)}
                    </span>
                  </span>
                  <span className="truncate text-xs text-muted-foreground">
                    {editorEffortLabel(option, t)}
                  </span>
                </span>
              </DropdownMenuRadioItem>
            )
          }) : (
            <DropdownMenuItem disabled className="items-start py-2">
              <span className="grid min-w-0 flex-1 text-left leading-tight">
                <span className="truncate text-sm font-medium">{unavailableLabel}</span>
                <span className="truncate text-xs text-muted-foreground">{t.chat.modelServerDefault}</span>
              </span>
            </DropdownMenuItem>
          )}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function editorModelOptionForTier(
  options: readonly ChatModelOption[],
  tier: ChatModelTier,
): ChatModelOption | null {
  return options.find((option) => option.tier === tier) ?? null
}

function isEditorModelTier(value: string): value is ChatModelTier {
  return value === 'high' || value === 'mid' || value === 'fast'
}

function editorModelNameLabel(
  option: Pick<NodeModelResolution, 'model'> | null | undefined,
  fallback: string,
) {
  const model = option?.model?.trim()
  if (!model) return fallback
  return model.replace(/^.+\//, '')
}

function editorModelDetailLabel(
  option: NodeModelResolution | null,
  t: ReturnType<typeof useLocale>['t'],
) {
  return `${editorModelNameLabel(option, t.chat.modelUnknown)} · ${editorEffortLabel(option, t)}`
}

function editorEffortLabel(
  option: Pick<NodeModelResolution, 'effort'> | null | undefined,
  t: ReturnType<typeof useLocale>['t'],
) {
  const effort = option?.effort?.trim().toLowerCase()
  if (!effort) return t.chat.modelEffortDefault
  if (effort === 'none') return t.chat.modelThinkingOff
  return `${t.chat.modelThinkingOn} ${editorShortEffort(effort)}`
}

function editorShortEffort(effort: string) {
  if (effort === 'medium') return 'med'
  if (effort === 'minimal') return 'min'
  return effort
}

function editorTierLabel(tier: ChatModelTier, t: ReturnType<typeof useLocale>['t']) {
  if (tier === 'high') return t.chat.modelTierHigh
  if (tier === 'fast') return t.chat.modelTierFast
  return t.chat.modelTierMid
}

const COMMENT_KIND_ORDER: EditorCommentKind[] = ['collect', 'inline_edit', 'evidence_review']
const EVIDENCE_PRESET_ORDER: EditorEvidencePreset[] = ['add_sources', 'fact_check', 'verify_citations']

type CommentKindMeta = {
  Icon: typeof MessagesSquare
  accentText: string
  bgClass: string
  borderClass: string
  dotClass: string
  label: string
  selectedBgClass: string
  selectedBorderClass: string
}

function commentKindMeta(kind: EditorCommentKind, copy: EditorCopy): CommentKindMeta {
  if (kind === 'inline_edit') {
    return {
      Icon: Sparkles,
      accentText: 'text-warning',
      bgClass: 'bg-warning-subtle/20',
      borderClass: 'border-l-warning',
      dotClass: 'bg-warning',
      label: copy.kindInline,
      selectedBgClass: 'bg-warning-subtle/45',
      selectedBorderClass: 'border-warning',
    }
  }
  if (kind === 'evidence_review') {
    return {
      Icon: SearchCheck,
      accentText: 'text-success',
      bgClass: 'bg-success-subtle/20',
      borderClass: 'border-l-success',
      dotClass: 'bg-success',
      label: copy.kindEvidence,
      selectedBgClass: 'bg-success-subtle/45',
      selectedBorderClass: 'border-success',
    }
  }
  return {
    Icon: MessagesSquare,
    accentText: 'text-brand',
    bgClass: 'bg-brand-subtle/25',
    borderClass: 'border-l-brand',
    dotClass: 'bg-brand',
    label: copy.kindCollect,
    selectedBgClass: 'bg-brand-subtle/45',
    selectedBorderClass: 'border-brand',
  }
}

function evidencePresetLabel(preset: EditorEvidencePreset, copy: EditorCopy) {
  if (preset === 'fact_check') return copy.presetFactCheck
  if (preset === 'verify_citations') return copy.presetVerifyCitations
  return copy.presetAddSources
}

function EditorCommentsPanel({
  comments,
  copy,
  dispatch,
  onAcceptSuggestionGroup,
  onAcceptSuggestion,
  onRejectSuggestionGroup,
  onRejectSuggestion,
  onRunComment,
  onSelectSuggestion,
  runErrors,
  runningCommentIds,
  selectedCommentId,
  suggestions,
}: {
  comments: EditorCommentThreadRecord[]
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  onAcceptSuggestionGroup: (groupId: string) => void
  onAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  onRejectSuggestionGroup: (groupId: string) => void
  onRejectSuggestion: (suggestionId: string) => void
  onRunComment: (comment: EditorCommentThreadRecord) => void
  onSelectSuggestion: (suggestionId: string) => void
  runErrors: Record<string, string>
  runningCommentIds: readonly string[]
  selectedCommentId: string | null
  suggestions: EditorSuggestionRecord[]
}) {
  const [statusTab, setStatusTab] = useState<'open' | 'resolved'>('open')
  const [kindFilter, setKindFilter] = useState<'all' | EditorCommentKind>('all')

  const openComments = comments.filter((comment) => comment.status !== 'resolved')
  const resolvedComments = comments.filter((comment) => comment.status === 'resolved')
  const tabComments = statusTab === 'open' ? openComments : resolvedComments
  const visibleComments = kindFilter === 'all'
    ? tabComments
    : tabComments.filter((comment) => comment.kind === kindFilter)
  const commentNumbers = buildCommentNumbers(comments)
  const documentChangeSuggestions = suggestions
    .filter((suggestion) =>
      (suggestion.status === 'pending' || suggestion.status === 'stale')
      && suggestion.origin.kind === 'global_run'
      && !suggestion.origin.commentId)
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt))

  return (
    <aside className="flex w-[22rem] shrink-0 flex-col bg-background">
      <div className="flex h-12 items-center justify-between border-b border-border px-3">
        <div className="flex items-center gap-2">
          <MessageSquarePlus className="size-4 text-brand" />
          <h2 className="text-sm font-semibold">{copy.assistant}</h2>
        </div>
        <TooltipButton
          label={copy.hideComments}
          onClick={() => dispatch({ isVisible: false, type: 'setEditorCommentPanelVisible' })}
        >
          <PanelRightClose className="size-4" />
        </TooltipButton>
      </div>
      <div className="flex flex-col gap-2 border-b border-border px-3 py-2">
        <div className="grid grid-cols-2 gap-0.5 rounded-md bg-muted/60 p-0.5">
          <CommentStatusTab active={statusTab === 'open'} count={openComments.length} label={copy.tabOpen} onClick={() => setStatusTab('open')} />
          <CommentStatusTab active={statusTab === 'resolved'} count={resolvedComments.length} label={copy.tabResolved} onClick={() => setStatusTab('resolved')} />
        </div>
        <div className="flex items-center gap-1 overflow-x-auto pb-0.5 [scrollbar-width:none]">
          <ListFilter className="size-3.5 shrink-0 text-muted-foreground" />
          <CommentKindChip active={kindFilter === 'all'} label={copy.filterAll} onClick={() => setKindFilter('all')} />
          {COMMENT_KIND_ORDER.map((kind) => {
            const meta = commentKindMeta(kind, copy)
            return (
              <CommentKindChip
                active={kindFilter === kind}
                dotClass={meta.dotClass}
                key={kind}
                label={meta.label}
                onClick={() => setKindFilter(kind)}
              />
            )
          })}
        </div>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-2 p-3">
          <EditorDocumentChangesSection
            labels={{
              accept: copy.accept,
              acceptAll: copy.acceptAll,
              documentChanges: copy.documentChanges,
              proposedChange: copy.proposedChange,
              reject: copy.reject,
              rejectAll: copy.rejectAll,
            }}
            onAcceptGroup={onAcceptSuggestionGroup}
            onAcceptSuggestion={onAcceptSuggestion}
            onRejectGroup={onRejectSuggestionGroup}
            onRejectSuggestion={onRejectSuggestion}
            onSelectSuggestion={onSelectSuggestion}
            suggestions={documentChangeSuggestions}
          />
          {visibleComments.length === 0 ? (
            <p className="rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground">{copy.noComments}</p>
          ) : visibleComments.map((comment) => (
            <EditorCommentCard
              comment={comment}
              commentNumber={commentNumbers.get(comment.id) ?? 0}
              copy={copy}
              dispatch={dispatch}
              isRunning={runningCommentIds.includes(comment.id)}
              isSelected={selectedCommentId === comment.id}
              key={comment.id}
              onAcceptSuggestion={onAcceptSuggestion}
              onRejectSuggestion={onRejectSuggestion}
              onRunComment={onRunComment}
              runError={runErrors[comment.id]}
              suggestion={activeSuggestionFor(suggestions, comment.id)}
            />
          ))}
        </div>
      </ScrollArea>
    </aside>
  )
}

function CommentStatusTab({ active, count, label, onClick }: {
  active: boolean
  count: number
  label: string
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'flex h-7 items-center justify-center gap-1.5 rounded-[5px] text-xs font-medium transition-colors',
        active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
      )}
      onClick={onClick}
      type="button"
    >
      {label}
      <span className="text-[10px] tabular-nums text-muted-foreground/80">{count}</span>
    </button>
  )
}

function CommentKindChip({ active, dotClass, label, onClick }: {
  active: boolean
  dotClass?: string
  label: string
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'inline-flex h-6 shrink-0 items-center gap-1 rounded-full border px-2 text-[11px] font-medium transition-colors',
        active ? 'border-brand/40 bg-brand-subtle text-brand' : 'border-border bg-background text-muted-foreground hover:text-foreground',
      )}
      onClick={onClick}
      type="button"
    >
      {dotClass ? <span className={cn('size-1.5 rounded-full', dotClass)} /> : null}
      {label}
    </button>
  )
}

function EditorCommentCard({
  comment,
  commentNumber,
  copy,
  dispatch,
  isRunning,
  isSelected,
  onAcceptSuggestion,
  onRejectSuggestion,
  onRunComment,
  runError,
  suggestion,
}: {
  comment: EditorCommentThreadRecord
  commentNumber: number
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  isRunning: boolean
  isSelected: boolean
  onAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  onRejectSuggestion: (suggestionId: string) => void
  onRunComment: (comment: EditorCommentThreadRecord) => void
  runError?: string
  suggestion?: EditorSuggestionRecord
}) {
  const meta = commentKindMeta(comment.kind, copy)
  const isResolved = comment.status === 'resolved'
  const isRunnable = comment.kind === 'inline_edit' || comment.kind === 'evidence_review'
  const [isEditing, setIsEditing] = useState(false)
  const [editDraft, setEditDraft] = useState(comment.commentMarkdown)
  const editTextareaRef = useRef<HTMLTextAreaElement | null>(null)

  useLayoutEffect(() => {
    if (!isEditing) return
    resizeTextareaToRows(editTextareaRef.current, 6)
  }, [editDraft, isEditing])

  function startEditing() {
    setEditDraft(comment.commentMarkdown)
    setIsEditing(true)
    dispatch({ commentId: comment.id, type: 'selectEditorComment' })
  }

  function saveEdit() {
    dispatch({ commentId: comment.id, contentMarkdown: editDraft, type: 'updateEditorCommentText' })
    setIsEditing(false)
  }

  return (
    <div
      className={cn(
        'rounded-md border border-l-[3px] text-sm shadow-sm transition-colors',
        meta.borderClass,
        isSelected
          ? cn(meta.selectedBorderClass, meta.selectedBgClass)
          : cn('border-border', meta.bgClass, 'hover:brightness-110'),
        isResolved && 'opacity-65',
      )}
      onClick={() => dispatch({ commentId: comment.id, type: 'selectEditorComment' })}
    >
      <div className="flex items-center gap-1.5 px-3 pt-2.5">
        <span className={cn('grid size-5 shrink-0 place-items-center rounded-[5px] bg-background/70 text-[10px] font-semibold tabular-nums', meta.accentText)}>
          {commentNumber}
        </span>
        <CommentKindMenu comment={comment} copy={copy} dispatch={dispatch} meta={meta} />
        <Button
          aria-label={copy.deleteComment}
          className="ml-auto size-6 shrink-0 text-muted-foreground hover:text-destructive"
          onClick={(event) => {
            event.stopPropagation()
            dispatch({ commentId: comment.id, type: 'deleteEditorComment' })
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Trash2 className="size-3.5" />
        </Button>
        <Button
          aria-label={copy.editComment}
          className="size-6 shrink-0 text-muted-foreground hover:text-foreground"
          onClick={(event) => {
            event.stopPropagation()
            startEditing()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <PencilLine className="size-3.5" />
        </Button>
        <Button
          aria-label={isResolved ? copy.reopen : copy.resolve}
          className="size-6 shrink-0 text-muted-foreground hover:text-foreground"
          onClick={(event) => {
            event.stopPropagation()
            dispatch({ commentId: comment.id, status: isResolved ? 'open' : 'resolved', type: 'setEditorCommentStatus' })
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          {isResolved ? <Undo2 className="size-3.5" /> : <Check className="size-3.5" />}
        </Button>
        <span className="shrink-0 text-[11px] text-muted-foreground">{formatEditorTime(comment.updatedAt)}</span>
      </div>
      <div className="px-3 pb-2.5 pt-1">
        <p className={cn('text-muted-foreground', isSelected ? 'line-clamp-2' : 'line-clamp-1')}>
          <span className="text-muted-foreground/60">“</span>
          {compactCommentQuote(comment.anchor.selectedText, isSelected ? 180 : 96)}
          <span className="text-muted-foreground/60">”</span>
        </p>
        {isEditing ? (
          <div className="mt-1.5" onClick={(event) => event.stopPropagation()}>
            <Textarea
              autoFocus
              className="min-h-16 resize-none text-sm [scrollbar-width:thin]"
              onChange={(event) => setEditDraft(event.target.value)}
              ref={editTextareaRef}
              value={editDraft}
            />
            <div className="mt-1.5 flex items-center justify-end gap-1.5">
              <Button className="h-7" onClick={() => setIsEditing(false)} size="sm" type="button" variant="ghost">
                {copy.cancel}
              </Button>
              <Button className="h-7" disabled={!editDraft.trim()} onClick={saveEdit} size="sm" type="button">
                {copy.save}
              </Button>
            </div>
          </div>
        ) : (
          <p className={cn('mt-1 text-foreground', isSelected ? 'whitespace-pre-wrap' : 'line-clamp-1')}>
            {comment.commentMarkdown}
          </p>
        )}
        {isSelected && !isEditing && comment.kind === 'evidence_review' ? (
          <EvidencePresetPicker comment={comment} copy={copy} dispatch={dispatch} />
        ) : null}
        {isSelected && !isEditing ? (
          isRunning ? (
            <div className="mt-2.5 flex items-center justify-center gap-2 rounded-md border border-brand/25 bg-brand-subtle/30 py-2 text-[11px] font-medium text-brand">
              <LoaderCircle className="size-3.5 animate-spin" />
              {copy.runningSuggestion}
            </div>
          ) : suggestion && suggestion.status === 'pending' ? (
            <SuggestionReview
              copy={copy}
              onAccept={onAcceptSuggestion}
              onReject={onRejectSuggestion}
              suggestion={suggestion}
            />
          ) : suggestion && suggestion.status === 'stale' ? (
            <div
              className="mt-2.5 rounded-md border border-warning/30 bg-warning-subtle/40 p-2.5"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-center gap-1.5 text-[11px] text-warning">
                <AlertTriangle className="size-3.5 shrink-0" />
                {copy.suggestionStale}
              </div>
              <div className="mt-2 flex items-center justify-end gap-1.5">
                <Button className="h-7" onClick={() => onRejectSuggestion(suggestion.id)} size="sm" type="button" variant="ghost">
                  <X className="size-3.5" />
                  {copy.reject}
                </Button>
                {isRunnable ? (
                  <Button className="h-7" onClick={() => onRunComment(comment)} size="sm" type="button" variant="outline">
                    <Sparkles className="size-3.5" />
                    {copy.regenerate}
                  </Button>
                ) : null}
              </div>
            </div>
          ) : (
            <>
              {runError ? (
                <div
                  className="mt-2.5 rounded-md border border-destructive/30 bg-destructive-subtle/40 p-2 text-[11px] text-destructive"
                  onClick={(event) => event.stopPropagation()}
                >
                  {runError}
                </div>
              ) : null}
              {isRunnable ? (
                <Button
                  className="mt-2.5 h-7 w-full"
                  onClick={(event) => {
                    event.stopPropagation()
                    onRunComment(comment)
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <Sparkles className="size-3.5" />
                  {copy.runSuggestion}
                </Button>
              ) : null}
            </>
          )
        ) : null}
      </div>
    </div>
  )
}

function buildCommentNumbers(comments: EditorCommentThreadRecord[]): Map<string, number> {
  const ordered = [...comments].sort((a, b) =>
    a.anchor.from - b.anchor.from || a.createdAt.localeCompare(b.createdAt))
  return new Map(ordered.map((comment, index) => [comment.id, index + 1]))
}

function compactCommentQuote(value: string, maxLength: number): string {
  const text = value.replace(/\s+/g, ' ').trim()
  if (text.length <= maxLength) return text
  return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
}

function shortenRunId(runId: string): string {
  return runId.length > 12 ? `${runId.slice(0, 12)}…` : runId
}

function activeSuggestionFor(
  suggestions: EditorSuggestionRecord[],
  commentId: string,
): EditorSuggestionRecord | undefined {
  return suggestions
    .filter((suggestion) =>
      (suggestion.status === 'pending' || suggestion.status === 'stale')
      && suggestion.origin.commentId === commentId)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0]
}

function SuggestionReview({ copy, onAccept, onReject, suggestion }: {
  copy: EditorCopy
  onAccept: (suggestion: EditorSuggestionRecord) => void
  onReject: (suggestionId: string) => void
  suggestion: EditorSuggestionRecord
}) {
  const plan = suggestionDiffPlan(suggestion.originalText, suggestion.proposedText)
  const reviewInEditor = plan.reviewSurface === 'editor'
  return (
    <div
      className="mt-2.5 rounded-md border border-brand/25 bg-brand-subtle/30 p-2.5"
      onClick={(event) => event.stopPropagation()}
    >
      <div className="flex items-center gap-1.5 text-[11px] font-medium text-brand">
        <Sparkles className="size-3.5 shrink-0" />
        {reviewInEditor ? copy.reviewInEditor : copy.reviewInPanel}
      </div>
      <div className="mt-2 rounded-md border border-success/20 bg-success-subtle/25 p-2">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-success">{copy.proposedText}</div>
        <p className="mt-1 line-clamp-4 whitespace-pre-wrap text-xs leading-5 text-foreground">{suggestion.proposedText}</p>
      </div>
      {suggestion.changeSummary?.length ? (
        <div className="mt-2 border-t border-border/60 pt-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{copy.changeSummary}</div>
          <ul className="mt-1 space-y-0.5">
            {suggestion.changeSummary.map((item, index) => (
              <li className="text-xs leading-4 text-muted-foreground" key={`${index}-${item}`}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {suggestion.warnings?.length ? (
        <div className="mt-2 rounded-md border border-warning/25 bg-warning-subtle/35 p-2">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-warning">
            <AlertTriangle className="size-3" />
            {copy.warnings}
          </div>
          <ul className="mt-1 space-y-0.5">
            {suggestion.warnings.map((item, index) => (
              <li className="text-xs leading-4 text-warning" key={`${index}-${item}`}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {suggestion.evidence ? (
        <div className="mt-2 border-t border-border/60 pt-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-success">{copy.sources}</div>
          <ul className="mt-1 space-y-0.5">
            {suggestion.evidence.sources.map((source) => (
              <li className="truncate text-xs" key={source.url}>
                <a className="text-brand hover:underline" href={source.url} rel="noopener noreferrer" target="_blank">
                  {source.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {reviewInEditor ? null : (
        <div className="mt-2 flex items-center justify-end gap-1.5">
          <Button className="h-7" onClick={() => onReject(suggestion.id)} size="sm" type="button" variant="ghost">
            <X className="size-3.5" />
            {copy.reject}
          </Button>
          <Button
            className="h-7 bg-brand text-white hover:bg-brand/90 hover:text-white"
            onClick={() => onAccept(suggestion)}
            size="sm"
            type="button"
          >
            <Check className="size-3.5" />
            {copy.accept}
          </Button>
        </div>
      )}
    </div>
  )
}

function CommentKindMenu({ comment, copy, dispatch, meta }: {
  comment: EditorCommentThreadRecord
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  meta: CommentKindMeta
}) {
  const KindIcon = meta.Icon
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          aria-label={copy.changeKind}
          className={cn('inline-flex h-6 shrink-0 items-center gap-1 rounded-full bg-background px-1.5 text-[11px] font-semibold', meta.accentText)}
          onClick={(event) => event.stopPropagation()}
          type="button"
        >
          <KindIcon className="size-3.5" />
          {meta.label}
          <ChevronDown className="size-3 opacity-60" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-52" onClick={(event) => event.stopPropagation()}>
        <DropdownMenuLabel className="text-xs text-muted-foreground">{copy.changeKind}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={(value) => dispatch({ commentId: comment.id, kind: value as EditorCommentKind, type: 'setEditorCommentKind' })}
          value={comment.kind}
        >
          {COMMENT_KIND_ORDER.map((kind) => {
            const kindMeta = commentKindMeta(kind, copy)
            const ItemIcon = kindMeta.Icon
            return (
              <DropdownMenuRadioItem className="gap-2 text-sm" key={kind} value={kind}>
                <ItemIcon className={cn('size-3.5', kindMeta.accentText)} />
                {kindMeta.label}
              </DropdownMenuRadioItem>
            )
          })}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function EvidencePresetPicker({ comment, copy, dispatch }: {
  comment: EditorCommentThreadRecord
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
}) {
  const activePreset = comment.evidencePreset ?? 'add_sources'
  return (
    <div className="mt-2.5 flex flex-col gap-1 rounded-md border border-success/25 bg-success-subtle/40 p-2">
      <span className="text-[10px] font-semibold uppercase tracking-wide text-success">{copy.preset}</span>
      <div className="flex flex-wrap gap-1">
        {EVIDENCE_PRESET_ORDER.map((preset) => (
          <button
            className={cn(
              'inline-flex h-6 items-center rounded-full border px-2 text-[11px] font-medium transition-colors',
              activePreset === preset ? 'border-success/50 bg-success-subtle text-success' : 'border-border bg-background text-muted-foreground hover:text-foreground',
            )}
            key={preset}
            onClick={(event) => {
              event.stopPropagation()
              dispatch({ commentId: comment.id, preset, type: 'setEditorCommentEvidencePreset' })
            }}
            type="button"
          >
            {evidencePresetLabel(preset, copy)}
          </button>
        ))}
      </div>
    </div>
  )
}

function EditorEmptyState({
  copy,
  dispatch,
  reportOptions,
}: {
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  reportOptions: CompletedReportOption[]
}) {
  return (
    <div className="grid min-h-0 flex-1 place-items-center bg-canvas p-8">
      <div className="w-full max-w-2xl rounded-md border border-border bg-background p-8 text-center shadow-sm">
        <FileText className="mx-auto mb-4 size-8 text-muted-foreground" />
        <h2 className="text-lg font-semibold">{copy.emptyTitle}</h2>
        <p className="mt-2 text-sm text-muted-foreground">{copy.emptyBody}</p>
        <div className="mx-auto mt-5 grid w-full max-w-[31rem] grid-cols-1 gap-2 sm:grid-cols-2">
          <Button
            className="h-10 w-full justify-center gap-1.5 px-2 text-[13px]"
            onClick={() => dispatch({ type: 'createEditorDocument' })}
            type="button"
          >
            <SquarePen className="size-4" />
            {copy.createDocument}
          </Button>
          <ImportReportMenu
            copy={copy}
            dispatch={dispatch}
            reportOptions={reportOptions}
            triggerClassName="h-10 w-full gap-1.5 px-2 text-[13px]"
            variant="button"
          />
        </div>
      </div>
    </div>
  )
}

function MiniToolbarButton({
  active,
  icon: Icon,
  label,
  onClick,
}: {
  active?: boolean
  icon: typeof Bold
  label: string
  onClick: () => void
}) {
  return (
    <Button
      aria-label={label}
      aria-pressed={active}
      className={cn('size-8 rounded-md', active && 'bg-brand-subtle text-brand')}
      onClick={onClick}
      size="icon"
      type="button"
      variant="ghost"
    >
      <Icon className="size-4" />
    </Button>
  )
}

function ToolbarButton({
  active,
  disabled,
  icon: Icon,
  label,
  onClick,
}: {
  active?: boolean
  disabled?: boolean
  icon: typeof Bold
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          aria-pressed={active}
          className={cn('size-6 rounded-md', active && 'bg-brand-subtle text-brand')}
          disabled={disabled}
          onClick={onClick}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Icon className="size-3.5" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function TooltipButton({
  children,
  className,
  disabled = false,
  label,
  onClick,
}: {
  children: ReactNode
  className?: string
  disabled?: boolean
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className={cn('size-7 rounded-md', className)}
          disabled={disabled}
          onClick={onClick}
          size="icon"
          type="button"
          variant="ghost"
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function resetExternalContentFlag(ref: { current: boolean }) {
  if (globalThis.queueMicrotask) {
    globalThis.queueMicrotask(() => {
      ref.current = false
    })
    return
  }
  globalThis.setTimeout(() => {
    ref.current = false
  }, 0)
}

function formatEditorTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function messageFromUnknown(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}

function escapeCssIdentifier(value: string) {
  if (globalThis.CSS?.escape) return globalThis.CSS.escape(value)
  return value.replace(/["\\]/g, '\\$&')
}
