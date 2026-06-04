import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import type { CreateResearchRunRequest } from '@/features/researchRuns/types'
import { ReportPanel, ReportRestoreRail } from '@/features/report/ReportPanel'
import type { ResearchRunRecord } from '@/features/project/types'
import type { JobFilter, ResearchJob } from '../types'
import { ResearchRunColumn } from './ResearchRunColumn'

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
  onDeleteJob: (jobId: string) => void
  onReportExpandedChange: (isExpanded: boolean) => void
  onReportVisibleChange: (isVisible: boolean) => void
  onSelectJob: (jobId: string) => void
  onToggleJob: (jobId: string) => void
  onUseReportInChat: (runId: string) => void
  reduceMotion: boolean | null
  selectedJobId: string | null
  selectedRun: ResearchRunRecord | null
  selectedStack: string
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
  onDeleteJob,
  onReportExpandedChange,
  onReportVisibleChange,
  onSelectJob,
  onToggleJob,
  onUseReportInChat,
  reduceMotion,
  selectedJobId,
  selectedRun,
  selectedStack,
}: ResearchWorkspaceProps) {
  const runColumn = (
    <ResearchRunColumn
      activeFilter={activeFilter}
      allJobs={allJobs}
      cancelErrorByRunId={cancelErrorByRunId}
      cancelSubmittingRunIds={cancelSubmittingRunIds}
      expandedJobId={expandedJobId}
      isComposerVisible={isComposerVisible}
      jobs={jobs}
      onActiveFilterChange={onActiveFilterChange}
      onCancelJob={onCancelJob}
      onComposerSubmit={onComposerSubmit}
      onComposerVisibleChange={onComposerVisibleChange}
      onDeleteJob={onDeleteJob}
      onSelectJob={onSelectJob}
      onToggleJob={onToggleJob}
      reduceMotion={reduceMotion}
      selectedJobId={selectedJobId}
      selectedStack={selectedStack}
    />
  )

  if (isDesktop && isReportVisible) {
    return (
      <ResizablePanelGroup
        className="h-full w-full overflow-hidden lg:min-h-0"
        orientation="horizontal"
      >
        <ResizablePanel
          className="min-h-0 min-w-0 overflow-hidden"
          defaultSize="58%"
          maxSize="74%"
          minSize="42%"
        >
          {runColumn}
        </ResizablePanel>
        <ResizableHandle aria-label="Resize report panel" />
        <ResizablePanel
          className="min-h-0 min-w-0 overflow-hidden"
          defaultSize="42%"
          maxSize="58%"
          minSize="26%"
        >
          <ReportPanel
            isExpanded={isReportExpanded}
            onExpandedChange={onReportExpandedChange}
            onHide={() => onReportVisibleChange(false)}
            onUseReportInChat={onUseReportInChat}
            selectedRun={selectedRun}
          />
        </ResizablePanel>
      </ResizablePanelGroup>
    )
  }

  if (isDesktop) {
    return (
      <div className="grid h-full grid-cols-[minmax(0,1fr)_44px] gap-2 overflow-hidden lg:min-h-0">
        <div className="min-h-0 overflow-hidden">{runColumn}</div>
        <ReportRestoreRail onShow={() => onReportVisibleChange(true)} />
      </div>
    )
  }

  return (
    <div className="grid w-full grid-cols-1 gap-4 py-4">
      {runColumn}
      {isReportVisible ? (
        <ReportPanel
          isExpanded={isReportExpanded}
          onExpandedChange={onReportExpandedChange}
          onHide={() => onReportVisibleChange(false)}
          onUseReportInChat={onUseReportInChat}
          selectedRun={selectedRun}
        />
      ) : (
        <ReportRestoreRail onShow={() => onReportVisibleChange(true)} />
      )}
    </div>
  )
}
