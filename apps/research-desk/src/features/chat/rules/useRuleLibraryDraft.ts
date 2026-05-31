import { useCallback, useReducer } from 'react'
import type { ChatRuleRecord } from '@/features/project/types'

export type RuleLibraryDraftState = {
  contentDraft: string
  error: string | null
  isDirty: boolean
  labelDraft: string
  selectedRuleId: string | null
  titleDraft: string
}

type RuleLibraryDraftAction =
  | { rule: ChatRuleRecord | null; type: 'loadRule' }
  | { type: 'startNewRule' }
  | { label: string; type: 'setLabelDraft' }
  | { title: string; type: 'setTitleDraft' }
  | { content: string; type: 'setContentDraft' }
  | { error: string | null; type: 'setError' }
  | { rule: ChatRuleRecord; type: 'markSaved' }

const emptyDraftState: RuleLibraryDraftState = {
  contentDraft: '',
  error: null,
  isDirty: false,
  labelDraft: '',
  selectedRuleId: null,
  titleDraft: '',
}

export function useRuleLibraryDraft() {
  const [draft, dispatch] = useReducer(ruleLibraryDraftReducer, emptyDraftState)

  const loadRule = useCallback((rule: ChatRuleRecord | null) => {
    dispatch({ rule, type: 'loadRule' })
  }, [])

  const startNewRule = useCallback(() => {
    dispatch({ type: 'startNewRule' })
  }, [])

  const setLabelDraft = useCallback((label: string) => {
    dispatch({ label, type: 'setLabelDraft' })
  }, [])

  const setTitleDraft = useCallback((title: string) => {
    dispatch({ title, type: 'setTitleDraft' })
  }, [])

  const setContentDraft = useCallback((content: string) => {
    dispatch({ content, type: 'setContentDraft' })
  }, [])

  const setError = useCallback((error: string | null) => {
    dispatch({ error, type: 'setError' })
  }, [])

  const markSaved = useCallback((rule: ChatRuleRecord) => {
    dispatch({ rule, type: 'markSaved' })
  }, [])

  return {
    draft,
    loadRule,
    markSaved,
    setContentDraft,
    setError,
    setLabelDraft,
    setTitleDraft,
    startNewRule,
  }
}

function ruleLibraryDraftReducer(
  state: RuleLibraryDraftState,
  action: RuleLibraryDraftAction,
): RuleLibraryDraftState {
  if (action.type === 'loadRule') {
    return action.rule
      ? {
        contentDraft: action.rule.contentMarkdown,
        error: null,
        isDirty: false,
        labelDraft: action.rule.label,
        selectedRuleId: action.rule.id,
        titleDraft: action.rule.title,
      }
      : emptyDraftState
  }

  if (action.type === 'startNewRule') {
    return emptyDraftState
  }

  if (action.type === 'setLabelDraft') {
    return {
      ...state,
      error: null,
      isDirty: true,
      labelDraft: action.label,
    }
  }

  if (action.type === 'setTitleDraft') {
    return {
      ...state,
      isDirty: true,
      titleDraft: action.title,
    }
  }

  if (action.type === 'setContentDraft') {
    return {
      ...state,
      error: null,
      isDirty: true,
      contentDraft: action.content,
    }
  }

  if (action.type === 'setError') {
    return {
      ...state,
      error: action.error,
    }
  }

  if (action.type === 'markSaved') {
    return {
      contentDraft: action.rule.contentMarkdown,
      error: null,
      isDirty: false,
      labelDraft: action.rule.label,
      selectedRuleId: action.rule.id,
      titleDraft: action.rule.title,
    }
  }

  return state
}
