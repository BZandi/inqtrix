import { createSeedProjectState } from '@/features/project/seedProject'
import { projectResearchJobs } from '@/features/project/selectors'

export const researchJobs = projectResearchJobs(createSeedProjectState())
