import {
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import {
  AnimatedPanelBody,
  AnimatedResizableHandle,
} from '@/components/ui/animated-panel'
import { useAnimatedResizablePanelCollapse } from '@/components/ui/animated-panel-motion'
import type { CreateResearchRunRequest } from '@/features/researchRuns/types'
import { ReportPanel } from '@/features/report/ReportPanel'
import type { ResearchRunRecord } from '@/features/project/types'
import type { JobFilter, ResearchJob } from '../types'
import { ResearchRunColumn } from './ResearchRunColumn'

const RESEARCH_REPORT_PANEL_ID = 'research-report-panel'
const RESEARCH_RUN_PANEL_ID = 'research-run-panel'

type ResearchWorkspaceProps = {
  activeFilter: JobFilter
  allJobs: ResearchJob[]
  cancelErrorByRunId: Record<string, string>
  cancelSubmittingRunIds: ReadonlySet<string>
  expandedJobId: string | null
  isComposerVisible: boolean
  isDesktop: boolean
  isReportExpanded: boolean
  isReportVisible: boolean
  jobs: ResearchJob[]
  onActiveFilterChange: (filter: JobFilter) => void
  onCancelJob: (jobId: string) => void
  onComposerSubmit: (request: CreateResearchRunRequest) => void
  onComposerVisibleChange: (isComposerVisible: boolean) => void
  onResearchQuestionChange: (question: string) => void
  researchQuestion: string
  onDeleteJob: (jobId: string) => void
  onReportExpandedChange: (isExpanded: boolean) => void
  onReportPanelSizeChange: (size: number) => void
  onReportVisibleChange: (isVisible: boolean) => void
  onSelectJob: (jobId: string) => void
  onShareJob?: (jobId: string) => void
  onToggleJob: (jobId: string) => void
  onUseReportInChat: (runId: string) => void
  reduceMotion: boolean | null
  reportPanelSize: number
  selectedJobId: string | null
  selectedRun: ResearchRunRecord | null
  selectedStack: string
  shareCountByRunId?: Record<string, number>
  sharedByLabelByRunId?: ReadonlyMap<string, string>
}

export function ResearchWorkspace({
  activeFilter,
  allJobs,
  cancelErrorByRunId,
  cancelSubmittingRunIds,
  expandedJobId,
  isComposerVisible,
  isDesktop,
  isReportExpanded,
  isReportVisible,
  jobs,
  onActiveFilterChange,
  onCancelJob,
  onComposerSubmit,
  onComposerVisibleChange,
  onResearchQuestionChange,
  researchQuestion,
  onDeleteJob,
  onReportExpandedChange,
  onReportPanelSizeChange,
  onReportVisibleChange,
  onSelectJob,
  onShareJob,
  onToggleJob,
  onUseReportInChat,
  reduceMotion,
  reportPanelSize,
  selectedJobId,
  selectedRun,
  selectedStack,
  shareCountByRunId,
  sharedByLabelByRunId,
}: ResearchWorkspaceProps) {
  const reportPanelMotion = useAnimatedResizablePanelCollapse({
    expandedSize: reportPanelSize,
    expanded: isReportVisible,
    reduceMotion,
  })
  const reportPanelLayout = {
    [RESEARCH_REPORT_PANEL_ID]: isReportVisible ? reportPanelSize : 0,
    [RESEARCH_RUN_PANEL_ID]: isReportVisible ? 100 - reportPanelSize : 100,
  }
  const runColumn = (
    <ResearchRunColumn
      activeFilter={activeFilter}
      allJobs={allJobs}
      cancelErrorByRunId={cancelErrorByRunId}
      cancelSubmittingRunIds={cancelSubmittingRunIds}
      expandedJobId={expandedJobId}
      isComposerVisible={isComposerVisible}
      isReportVisible={isReportVisible}
      jobs={jobs}
      onActiveFilterChange={onActiveFilterChange}
      onCancelJob={onCancelJob}
      onComposerSubmit={onComposerSubmit}
      onComposerVisibleChange={onComposerVisibleChange}
      onReportVisibleChange={onReportVisibleChange}
      reportPanelId={RESEARCH_REPORT_PANEL_ID}
      onResearchQuestionChange={onResearchQuestionChange}
      researchQuestion={researchQuestion}
      onDeleteJob={onDeleteJob}
      onSelectJob={onSelectJob}
      onShareJob={onShareJob}
      onToggleJob={onToggleJob}
      reduceMotion={reduceMotion}
      selectedJobId={selectedJobId}
      selectedStack={selectedStack}
      shareCountByRunId={shareCountByRunId}
      sharedByLabelByRunId={sharedByLabelByRunId}
    />
  )

  if (isDesktop) {
    return (
      <ResizablePanelGroup
        className="h-full w-full overflow-hidden lg:min-h-0"
        defaultLayout={reportPanelLayout}
        elementRef={reportPanelMotion.groupRef}
        onLayoutChanged={(layout) => {
          const size = layout[RESEARCH_REPORT_PANEL_ID]
          if (
            isReportVisible
            && !reportPanelMotion.isProgrammaticLayoutChange()
            && Number.isFinite(size)
            && size > 0
          ) {
            onReportPanelSizeChange(size)
          }
        }}
        orientation="horizontal"
      >
        <ResizablePanel
          className="min-h-0 min-w-0 overflow-hidden"
          defaultSize={reportPanelLayout[RESEARCH_RUN_PANEL_ID]}
          id={RESEARCH_RUN_PANEL_ID}
          maxSize={isReportVisible ? '74%' : '100%'}
          minSize="42%"
        >
          {runColumn}
        </ResizablePanel>
        <AnimatedResizableHandle
          aria-label="Resize report panel"
          expanded={isReportVisible}
        />
        <ResizablePanel
          className="min-h-0 min-w-0 overflow-hidden"
          collapsedSize="0%"
          collapsible
          defaultSize={reportPanelLayout[RESEARCH_REPORT_PANEL_ID]}
          elementRef={reportPanelMotion.panelElementRef}
          id={RESEARCH_REPORT_PANEL_ID}
          maxSize="58%"
          minSize={isReportVisible ? '26%' : '0%'}
          panelRef={reportPanelMotion.panelRef}
        >
          <AnimatedPanelBody expanded={isReportVisible} side="right">
            <ReportPanel
              isExpanded={isReportVisible && isReportExpanded}
              onExpandedChange={onReportExpandedChange}
              onHide={() => { onReportExpandedChange(false); onReportVisibleChange(false) }}
              onUseReportInChat={onUseReportInChat}
              selectedRun={selectedRun}
            />
          </AnimatedPanelBody>
        </ResizablePanel>
      </ResizablePanelGroup>
    )
  }

  return (
    <div className="grid w-full grid-cols-1 gap-4 py-4">
      {runColumn}
      {isReportVisible ? (
        <ReportPanel
          isExpanded={isReportExpanded}
          onExpandedChange={onReportExpandedChange}
          onHide={() => { onReportExpandedChange(false); onReportVisibleChange(false) }}
          onUseReportInChat={onUseReportInChat}
          selectedRun={selectedRun}
        />
      ) : null}
    </div>
  )
}
