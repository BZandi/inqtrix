import { createSeedProjectState } from '@/features/project/seedProject'
import { projectChatThreads } from '@/features/project/selectors'

export const initialChatThreads = projectChatThreads(createSeedProjectState())
