import { useAuthConfig } from '@/features/auth/useAuthConfig'
import { OwnerSetupScreen } from '@/features/onboarding/OwnerSetupScreen'
import { ResearchDesk } from '@/features/researchDesk/ResearchDesk'

export function App() {
  // One pre-login discovery probe (GET /api/auth/config) at the root. It
  // both gates the first-run owner setup (registration.needs_owner, local
  // mode only) and is threaded into the desk for the SSO provider name and
  // the PAT availability flag — so there is no second /api/setup/status
  // probe and no duplicate fetch inside the desk.
  const auth = useAuthConfig(true)
  // Hold rendering until the first probe answers so neither the setup screen
  // nor the desk flashes; a probe failure degrades open (config is null, the
  // desk renders and runs its own auth gate).
  if (auth.status === 'pending') return null
  if (auth.config?.registration.needs_owner) {
    return <OwnerSetupScreen onCreated={() => void auth.refresh()} />
  }
  return <ResearchDesk authConfig={auth.config} />
}
