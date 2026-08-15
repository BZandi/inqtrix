import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { LocaleProvider } from '@/i18n/LocaleProvider'
import { AuthLockScreen } from './AuthLockScreen'

function lockMarkup(): string {
  return renderToStaticMarkup(
    <LocaleProvider>
      <AuthLockScreen
        error={null}
        identifier=""
        isSubmitting={false}
        mode="local"
        onCredentialSubmit={() => undefined}
        onIdentifierChange={() => undefined}
        onPasswordChange={() => undefined}
        onSubmit={() => undefined}
        onTokenChange={() => undefined}
        password=""
        reduceMotion
        token=""
      />
    </LocaleProvider>,
  )
}

describe('AuthLockScreen', () => {
  it('announces itself as a modal dialog', () => {
    // role alone is not enough: without aria-modal, assistive technology
    // keeps offering the covered application behind the lock screen.
    const markup = lockMarkup()

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('aria-labelledby="auth-lock-title"')
  })

  it('labels the dialog with a rendered element', () => {
    // aria-labelledby pointing at nothing announces an unnamed dialog.
    expect(lockMarkup()).toContain('id="auth-lock-title"')
  })
})

describe('AuthLockScreen AI disclosure', () => {
  it('states that Inqtrix is an AI system before the quality caveat', () => {
    const markup = lockMarkup()
    const disclosure = markup.indexOf('Inqtrix ist ein KI-System.')
    const caveat = markup.indexOf('KI-Ergebnisse können falsch')

    expect(disclosure).toBeGreaterThan(-1)
    expect(caveat).toBeGreaterThan(-1)
    // The disclosure precedes the caveat: what the system IS comes before
    // how reliable it is.
    expect(disclosure).toBeLessThan(caveat)
  })
})
