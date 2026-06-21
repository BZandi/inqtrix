import { PanelBottomOpen, Users } from '@/components/icons'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { WelcomeState } from '@/components/ui/welcome-state'
import type { CreateResearchRunRequest } from '@/features/researchRuns/types'
import { partitionJobsByAccess } from '@/features/sharing/shareModel'
import { useLocale } from '@/i18n/LocaleProvider'
import { appMotion } from '@/motion/transitions'
import type { JobFilter, ResearchJob } from '../types'
import {
  Composer,
  buildComposerRequest,
  defaultComposerFormState,
} from './Composer'
import { JobFilterMenu } from './JobFilterMenu'
import { ResearchJobCard } from './ResearchJobCard'

type ResearchRunColumnProps = {
  activeFilter: JobFilter
  allJobs: ResearchJob[]
  cancelErrorByRunId: Record<string, string>
  cancelSubmittingRunIds: ReadonlySet<string>
  expandedJobId: string | null
  isComposerVisible: boolean
  jobs: ResearchJob[]
  onActiveFilterChange: (filter: JobFilter) => void
  onCancelJob: (jobId: string) => void
  onComposerSubmit: (request: CreateResearchRunRequest) => void
  onComposerVisibleChange: (isComposerVisible: boolean) => void
  onResearchQuestionChange: (question: string) => void
  researchQuestion: string
  onDeleteJob: (jobId: string) => void
  onSelectJob: (jobId: string) => void
  onShareJob?: (jobId: string) => void
  onToggleJob: (jobId: string) => void
  reduceMotion: boolean | null
  selectedJobId: string | null
  selectedStack: string
  shareCountByRunId?: Record<string, number>
  sharedByLabelByRunId?: ReadonlyMap<string, string>
}

export function ResearchRunColumn({
  activeFilter,
  allJobs,
  cancelErrorByRunId,
  cancelSubmittingRunIds,
  expandedJobId,
  isComposerVisible,
  jobs,
  onActiveFilterChange,
  onCancelJob,
  onComposerSubmit,
  onComposerVisibleChange,
  onResearchQuestionChange,
  researchQuestion,
  onDeleteJob,
  onSelectJob,
  onShareJob,
  onToggleJob,
  reduceMotion,
  selectedJobId,
  selectedStack,
  shareCountByRunId,
  sharedByLabelByRunId,
}: ResearchRunColumnProps) {
  const { t } = useLocale()
  // Seed the question from the session-scoped shell draft so it survives a view
  // switch (this column unmounts on switch); the lazy initializer restores it on
  // remount. Form settings stay ephemeral by design -- only the typed question is
  // worth preserving.
  const [composerForm, setComposerForm] = useState(() => ({
    ...defaultComposerFormState,
    question: researchQuestion,
  }))
  // Mirror question edits (and the submit-time clear) back up to the shell.
  useEffect(() => {
    onResearchQuestionChange(composerForm.question)
  }, [composerForm.question, onResearchQuestionChange])
  const { own: ownJobs, shared: sharedJobs } = partitionJobsByAccess(jobs)

  return (
    <section className="relative flex min-h-[calc(100svh-var(--header-h))] min-w-0 flex-col overflow-hidden bg-background lg:h-full lg:min-h-0">
      <JobFilterMenu
        activeFilter={activeFilter}
        jobs={allJobs}
        onActiveFilterChange={onActiveFilterChange}
      />
      {allJobs.length === 0 ? (
        <ResearchEmptyState
          onSuggestionSelect={(question) => onComposerSubmit(
            buildComposerRequest(composerForm, question, selectedStack),
          )}
        />
      ) : null}
      <div className="relative flex min-h-0 flex-1 flex-col gap-3 px-4 pt-3">
        {allJobs.length === 0 ? (
          <div className="min-h-0 flex-1" />
        ) : (
          <ScrollArea className="min-h-0 flex-1 pr-2">
            <motion.div
              layout
              className="flex flex-col gap-2.5 pb-3"
              transition={appMotion.list}
            >
              <AnimatePresence initial={false} mode={reduceMotion ? 'sync' : 'popLayout'}>
                {ownJobs.map((job) => (
                  <ResearchJobCard
                    isExpanded={expandedJobId === job.id}
                    isSelected={selectedJobId === job.id}
                    job={job}
                    key={job.id}
                    cancelError={cancelErrorByRunId[job.id]}
                    isCancelSubmitting={cancelSubmittingRunIds.has(job.id)}
                    onCancel={() => onCancelJob(job.id)}
                    onDelete={() => onDeleteJob(job.id)}
                    onSelect={() => onSelectJob(job.id)}
                    onShare={onShareJob ? () => onShareJob(job.id) : undefined}
                    onToggleExpanded={() => onToggleJob(job.id)}
                    shareCount={shareCountByRunId?.[job.id]}
                  />
                ))}
                {sharedJobs.length > 0 && (
                  <motion.div
                    animate={{ opacity: 1 }}
                    className="flex items-center gap-1.5 pt-1.5 text-muted-foreground"
                    exit={{ opacity: 0 }}
                    initial={reduceMotion ? false : { opacity: 0 }}
                    key="shared-with-me-divider"
                    transition={appMotion.list}
                  >
                    <Users className="size-3.5" />
                    <span className="t-caption">{t.sharing.sharedWithMe}</span>
                  </motion.div>
                )}
                {sharedJobs.map((job) => (
                  <ResearchJobCard
                    isExpanded={expandedJobId === job.id}
                    isSelected={selectedJobId === job.id}
                    job={job}
                    key={job.id}
                    cancelError={cancelErrorByRunId[job.id]}
                    isCancelSubmitting={cancelSubmittingRunIds.has(job.id)}
                    onCancel={() => onCancelJob(job.id)}
                    onDelete={() => onDeleteJob(job.id)}
                    onSelect={() => onSelectJob(job.id)}
                    onToggleExpanded={() => onToggleJob(job.id)}
                    sharedByLabel={sharedByLabelByRunId?.get(job.id) ?? ''}
                  />
                ))}
              </AnimatePresence>
              {jobs.length === 0 && (
                <div className="flex min-h-40 items-center justify-center rounded-lg border border-dashed border-border px-4 text-sm text-muted-foreground">
                  {t.home.emptyFilter}
                </div>
              )}
            </motion.div>
          </ScrollArea>
        )}
      </div>

      <AnimatePresence initial={false} mode={reduceMotion ? 'sync' : 'popLayout'}>
        {isComposerVisible ? (
          <Composer
            form={composerForm}
            key="composer"
            onHide={() => onComposerVisibleChange(false)}
            onSubmit={onComposerSubmit}
            reduceMotion={reduceMotion}
            selectedStack={selectedStack}
            setForm={setComposerForm}
          />
        ) : (
          <motion.div
            key="composer-collapsed"
            initial={reduceMotion ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={appMotion.composer}
            className="shrink-0 px-4 pb-4 pt-2"
          >
            <Button
              aria-label={t.composer.show}
              className="h-8 gap-1.5 rounded-md"
              onClick={() => onComposerVisibleChange(true)}
              size="sm"
              type="button"
              variant="outline"
            >
              <PanelBottomOpen className="size-4" />
              {t.composer.show}
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}

function ResearchEmptyState({
  onSuggestionSelect,
}: {
  onSuggestionSelect: (question: string) => void
}) {
  const { t } = useLocale()
  const suggestions = [
    {
      label: t.home.suggestionAiNews,
      question: t.home.suggestionAiNewsPrompt,
    },
    {
      label: t.home.suggestionExplainRag,
      question: t.home.suggestionExplainRagPrompt,
    },
    {
      label: t.home.suggestionCompareLlms,
      question: t.home.suggestionCompareLlmsPrompt,
    },
  ]

  return (
    <div className="pointer-events-none absolute inset-x-4 bottom-40 top-12 z-10 flex items-center justify-center px-4 py-8">
      <WelcomeState
        actions={(
          <div className="flex flex-wrap justify-center gap-2">
          {suggestions.map((suggestion) => (
            <Button
              className="h-8 rounded-md px-2.5 text-xs"
              key={suggestion.label}
              onClick={() => onSuggestionSelect(suggestion.question)}
              type="button"
              variant="outline"
            >
              <span>{suggestion.label}</span>
            </Button>
          ))}
          </div>
        )}
        className="pointer-events-auto"
        kicker={t.home.emptyKicker}
        subtitle={t.home.emptyDescription}
        title={t.home.emptyTitle}
      />
    </div>
  )
}
