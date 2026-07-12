import { ListChecks } from '@/components/icons'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { OptionMenuHeader, optionMenuContentClassName } from '@/components/ui/option-menu'
import { StatusRow, SummaryGroup, type StatusRowTone } from '@/components/ui/status-summary'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { composerIconButtonClassName } from '@/features/composer/ComposerIconButton'
import { useLocale } from '@/i18n/LocaleProvider'
import { effortLevelLabel } from '@/lib/modelCard'
import type {
  AgentOverview,
  AgentToolUseCounts,
  ApprovalRow,
} from './agentStatusOverview'
import type {
  AgentExecutionDirective,
  AgentExecutionSnapshot,
  AgentSourcePolicy,
} from './executionPolicy'
import { resolveAgentExecutionDisplay } from './executionPolicy'

/**
 * Read-only overview of what THIS run would actually do — the Agent Desk
 * counterpart of the Knowledge Desk's run overview (same icon, same
 * primitives, same tri-state honesty). Every row renders server-published
 * facts (`capabilities.agent`, published == enforced) or live composer
 * state; nothing here is a hardcoded claim.
 */

function approvalValue(
  row: ApprovalRow,
  o: ReturnType<typeof useLocale>['t']['agent']['overview'],
): { tone: StatusRowTone; value: string } {
  if (row.state === 'always') return { tone: 'success', value: o.stateAlways }
  if (row.state === 'asks') {
    return {
      tone: 'success',
      value: row.conditional ? o.stateAsksConditional : o.stateAsks,
    }
  }
  return { tone: 'muted', value: o.stateFree }
}

export function AgentStatusMenu({
  autonomyHint,
  autonomyLabel,
  depthDeep = false,
  disabled,
  memoryEnabled,
  modelValue = '',
  overview,
  responseFormValue,
  responseForm,
  autonomyMode,
  execution = null,
  executionDirective = null,
  sourcePolicy,
  toolUseCounts = { web: 0, knowledge: 0 },
}: {
  /** The selected mode's meaning, reused from the toggle tooltip. */
  autonomyHint: string
  /** Display label of the selected permission mode (header value). */
  autonomyLabel: string
  /** Live composer depth toggle (plan M4): true = Deep selected. */
  depthDeep?: boolean
  disabled: boolean
  /** Account preference `enable_agent_memory` (live). */
  memoryEnabled: boolean
  /** Live model-override readout from the composer picker (R3);
   * '' hides the row (no picker wired). */
  modelValue?: string
  overview: AgentOverview
  /** Display label of the selected response form (live). */
  responseFormValue: string
  /** Raw selected values are needed to preview directive overrides. */
  responseForm: 'auto' | 'chat' | 'canvas'
  autonomyMode: string
  execution?: AgentExecutionSnapshot | null
  executionDirective?: AgentExecutionDirective | null
  sourcePolicy: AgentSourcePolicy
  /** Actual completed tool calls when task-level data is available. */
  toolUseCounts?: AgentToolUseCounts
}) {
  const { t } = useLocale()
  const o = t.agent.overview
  const display = resolveAgentExecutionDisplay({
    execution,
    pendingDirective: executionDirective,
    selectedDepth: depthDeep ? 'deep' : 'normal',
    selectedMode: overview.brain,
    selectedResponseForm: responseForm,
  })
  const acceptedExecution = executionDirective ? null : execution
  const effectiveDirective = display.executionDirective
  const effectiveSourcePolicy = acceptedExecution?.sourcePolicy ?? sourcePolicy
  const effectiveMode = display.effectiveMode
  const effectiveResponseForm = display.responseForm === 'chat'
    ? t.agent.composer.responseFormChat
    : display.responseForm === 'canvas'
      ? t.agent.composer.responseFormCanvas
      : display.responseForm === 'auto'
        ? t.agent.composer.responseFormAuto
        : responseFormValue
  const effectiveModel = acceptedExecution
    ? acceptedExecution.model
      ? `${acceptedExecution.model}${acceptedExecution.reasoningEffort
        ? ` · ${effortLevelLabel(acceptedExecution.reasoningEffort)}`
        : ''}`
      : o.modelAutoValue
    : modelValue
  const effectiveDepthDeep = display.depth === 'deep'
  const effectiveToolUseCounts = executionDirective
    ? { web: 0, knowledge: 0 }
    : acceptedExecution?.toolUseCounts ?? toolUseCounts
  const quickWebConsented = effectiveDirective === 'quick_web'
    && (
      acceptedExecution?.consentReason === 'explicit_directive'
      || acceptedExecution?.consentReason === 'explicit_quick_web'
      || acceptedExecution?.consentReason === 'strict_approval'
      || (!acceptedExecution && autonomyMode !== 'strict')
    )
  const routeValue = effectiveDirective === 'quick_web'
    ? o.routeQuickWeb
    : effectiveDirective === 'knowledge_only'
      ? o.routeKnowledgeOnly
      : effectiveMode === 'agent_kernel'
        ? o.routeAutomatic
        : o.brainMission

  const sourceValue = (
    id: 'web_search' | 'knowledge_search',
    available: boolean,
  ): { tone: StatusRowTone; value: string } => {
    if (!available) return { tone: 'warning', value: o.stateUnavailable }
    const source = id === 'web_search' ? 'web' : 'knowledge'
    const forced =
      (source === 'web' && effectiveDirective === 'quick_web')
      || (source === 'knowledge' && effectiveDirective === 'knowledge_only')
    if (forced) return { tone: 'success', value: o.stateForcedOnce }
    if (effectiveDirective) {
      return { tone: 'muted', value: o.stateExcludedMessage }
    }
    return effectiveSourcePolicy[source] === 'disabled'
      ? { tone: 'muted', value: o.stateDisabledSession }
      : { tone: 'success', value: o.stateAvailable }
  }

  const approvalLabel = (row: ApprovalRow): string => {
    switch (row.id) {
      case 'plan':
        return o.rowPlan
      case 'web_search':
        return o.rowWebSearch
      case 'knowledge_search':
        return o.rowKnowledgeSearch
      case 'research':
        return o.rowResearch
      case 'skill_activation':
        return o.rowSkillActivation
      case 'editor_patch':
        return o.rowEditorPatch
      default:
        // An unknown gated tool surfaces under its raw name — a policy
        // addition must never silently vanish from the overview.
        return row.id
    }
  }

  return (
    <DropdownMenu modal={false}>
      <Tooltip>
        <DropdownMenuTrigger asChild>
          <TooltipTrigger asChild>
            <Button
              aria-label={o.title}
              className={composerIconButtonClassName}
              disabled={disabled}
              type="button"
              variant="ghost"
            >
              <ListChecks />
            </Button>
          </TooltipTrigger>
        </DropdownMenuTrigger>
        <TooltipContent>{o.title}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className={optionMenuContentClassName} side="top" sideOffset={8}>
        <OptionMenuHeader count={0} title={o.title} value={autonomyLabel} />
        <div className="py-1">
          <SummaryGroup label={o.groupExecution}>
            <StatusRow
              label={o.rowRoute}
              value={routeValue}
            />
            <StatusRow
              label={o.rowBrain}
              value={effectiveMode === 'agent_kernel' ? o.brainKernel : o.brainMission}
            />
            {/* Feature transparency: the second engine is listed even when
               the deployment does not offer it — an absent picker alone
               would hide that it exists. Neutral tone on purpose: an
               unactivated beta is an operator choice, not a defect. */}
            <StatusRow
              label={o.brainKernel}
              {...(overview.kernel === 'active'
                ? { tone: 'success' as const, value: o.stateActive }
                : overview.kernel === 'selectable'
                  ? { tone: 'muted' as const, value: o.stateSelectable }
                  : { tone: 'muted' as const, value: o.stateUnavailable })}
            />
            <StatusRow label={t.agent.composer.responseForm} value={effectiveResponseForm} />
            {effectiveModel && (
              <StatusRow label={o.rowModel} value={effectiveModel} />
            )}
            <StatusRow
              label={o.rowDepth}
              {...(effectiveDepthDeep
                ? { tone: 'success' as const, value: t.agent.composer.deep }
                : { tone: 'muted' as const, value: o.depthNormal })}
            />
            <StatusRow
              label={o.rowDurable}
              {...(overview.durable
                ? { tone: 'success' as const, value: o.stateOn }
                : { tone: 'warning' as const, value: o.stateVolatile })}
            />
            <StatusRow
              label={o.rowMemory}
              {...(memoryEnabled
                ? { tone: 'success' as const, value: o.stateOn }
                : { tone: 'muted' as const, value: o.stateOff })}
            />
          </SummaryGroup>
          {overview.approvals ? (
            <>
              <DropdownMenuSeparator className="mx-0 my-1" />
              <SummaryGroup label={o.groupApprovals}>
                {effectiveDirective === 'quick_web' ? (
                  <StatusRow
                    label={o.rowOneTimeWebConsent}
                    tone={quickWebConsented ? 'success' : 'warning'}
                    value={quickWebConsented ? o.stateThisMessage : o.stateAsks}
                  />
                ) : null}
                {execution?.consentReason && effectiveDirective !== 'quick_web' ? (
                  <StatusRow
                    label={o.rowConsentReason}
                    value={execution.consentReason.startsWith('explicit_')
                      ? o.stateThisMessage
                      : execution.consentReason}
                  />
                ) : null}
                {overview.approvals.map((row) => (
                  <StatusRow
                    key={row.id}
                    label={approvalLabel(row)}
                    {...(row.id === 'web_search' && quickWebConsented
                      ? { tone: 'success' as const, value: o.stateThisMessage }
                      : approvalValue(row, o))}
                  />
                ))}
              </SummaryGroup>
              <div className="px-2.5 pb-1 pt-1.5 t-meta-sm text-muted-foreground">
                {autonomyHint}
              </div>
            </>
          ) : null}
          <DropdownMenuSeparator className="mx-0 my-1" />
          <SummaryGroup label={o.groupTools}>
            {overview.tools.map((row) => (
              <StatusRow
                key={row.id}
                label={
                  row.id === 'web_search'
                    ? o.toolWebSearch
                    : row.id === 'knowledge_search'
                      ? o.toolKnowledge
                      : o.toolEditor
                }
                {...(row.id === 'editor_access'
                  ? row.available
                    ? { tone: 'success' as const, value: o.stateReady }
                    : { tone: 'warning' as const, value: o.stateUnavailable }
                  : sourceValue(row.id, row.available))}
              />
            ))}
            {effectiveToolUseCounts.web > 0 ? (
              <StatusRow
                label={o.toolWebUsed}
                tone="success"
                value={o.toolUseCount.replace('{count}', String(effectiveToolUseCounts.web))}
              />
            ) : null}
            {effectiveToolUseCounts.knowledge > 0 ? (
              <StatusRow
                label={o.toolKnowledgeUsed}
                tone="success"
                value={o.toolUseCount.replace('{count}', String(effectiveToolUseCounts.knowledge))}
              />
            ) : null}
          </SummaryGroup>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
