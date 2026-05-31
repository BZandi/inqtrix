import {
  AlertTriangle,
  BadgeCheck,
  BookOpen,
  ExternalLink,
  Github,
  KeyRound,
  Languages,
  LockKeyhole,
  MessageSquareText,
  Scale,
  SearchCheck,
  ShieldCheck,
  type LucideIcon,
} from '@/components/icons'
import { motion } from 'motion/react'
import { type FormEvent } from 'react'
import { Button } from '@/components/ui/button'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'

type AuthLockScreenProps = {
  error: string | null
  isSubmitting: boolean
  onSubmit: (token: string) => void
  onTokenChange: (token: string) => void
  reduceMotion: boolean | null
  token: string
}

export function AuthLockScreen({
  error,
  isSubmitting,
  onSubmit,
  onTokenChange,
  reduceMotion,
  token,
}: AuthLockScreenProps) {
  const { locale, setLocale, t } = useLocale()

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onSubmit(token)
  }

  return (
    <div className="fixed inset-0 z-[100] flex min-h-svh items-start justify-center overflow-y-auto bg-background/72 px-4 py-6 text-foreground backdrop-blur-xl sm:py-8">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,color-mix(in_oklch,var(--brand)_22%,transparent),transparent_42%)]" />
      <motion.section
        animate={{ opacity: 1, scale: 1, y: 0 }}
        aria-labelledby="auth-lock-title"
        className="relative w-full max-w-4xl overflow-hidden rounded-lg border border-border bg-card/95 shadow-[0_24px_80px_var(--shadow-soft)] backdrop-blur-2xl"
        initial={reduceMotion ? false : { opacity: 0, scale: 0.98, y: 10 }}
        role="dialog"
        transition={appMotion.panel}
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-surface/70 px-5 py-4">
          <div className="inline-flex h-7 items-center gap-1.5 rounded-md border border-brand/20 bg-brand-subtle px-2 text-xs font-semibold text-brand">
            <SearchCheck className="size-3.5" />
            {t.authLock.eyebrow}
          </div>
          <div
            aria-label={t.common.language}
            className="inline-flex rounded-md border border-border bg-background/80 p-0.5 text-xs font-semibold shadow-[0_1px_2px_var(--shadow-hairline)] backdrop-blur"
            role="group"
          >
            <Languages className="mx-1.5 my-1 size-3.5 text-muted-foreground" />
            {(['de', 'en'] as const).map((nextLocale) => (
              <button
                aria-pressed={locale === nextLocale}
                className={cn(
                  'rounded px-2 py-1 text-muted-foreground transition hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  locale === nextLocale && 'bg-brand text-brand-foreground shadow-sm hover:text-brand-foreground',
                )}
                key={nextLocale}
                onClick={() => setLocale(nextLocale)}
                type="button"
              >
                {nextLocale.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_minmax(320px,0.82fr)]">
          <div className="border-b border-border bg-surface/45 p-5 sm:p-7 lg:border-b-0 lg:border-r">
            <h1
              className="max-w-lg text-3xl font-semibold tracking-normal text-foreground"
              id="auth-lock-title"
            >
              {t.authLock.title}
            </h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
              {t.authLock.description}
            </p>

            <div className="mt-6 grid gap-1 border-y border-border py-2 text-sm">
              <LockBenefit
                icon={SearchCheck}
                label={t.authLock.researchBenefit}
              />
              <LockBenefit
                icon={BadgeCheck}
                label={t.authLock.sourceBenefit}
              />
              <LockBenefit
                icon={MessageSquareText}
                label={t.authLock.chatBenefit}
              />
              <LockBenefit
                icon={Github}
                label={t.authLock.openSourceBenefit}
              />
            </div>
            <div className="mt-5 rounded-md border border-warning/25 bg-warning-subtle/65 p-3 text-sm">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
                <div>
                  <p className="font-semibold text-foreground">{t.authLock.prototypeTitle}</p>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">
                    {t.authLock.prototypeDescription}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center p-5 sm:p-7">
            <div className="w-full rounded-md border border-border bg-background/80 p-4 shadow-[0_1px_2px_var(--shadow-hairline)]">
              <div className="flex items-start gap-2">
                <ShieldCheck className="mt-0.5 size-4 shrink-0 text-brand" />
                <div>
                  <p className="text-sm font-semibold text-foreground">
                    {t.authLock.accessTitle}
                  </p>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">
                    {t.authLock.accessDescription}
                  </p>
                </div>
              </div>

              <form className="mt-5 space-y-3" onSubmit={handleSubmit}>
                <label className="block text-sm font-semibold text-foreground" htmlFor="inqtrix-auth-token">
                  {t.authLock.tokenLabel}
                </label>
                <div className="relative">
                  <KeyRound className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    aria-describedby="inqtrix-auth-help"
                    aria-invalid={Boolean(error)}
                    autoComplete="off"
                    autoFocus
                    className={cn(
                      'h-11 w-full rounded-md border border-border bg-card px-9 text-sm text-foreground shadow-[0_1px_2px_var(--shadow-hairline)] outline-none transition placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring',
                      error && 'border-destructive/50 focus-visible:ring-destructive/35',
                    )}
                    id="inqtrix-auth-token"
                    inputMode="text"
                    onChange={(event) => onTokenChange(event.target.value)}
                    placeholder={t.authLock.tokenPlaceholder}
                    spellCheck={false}
                    type="password"
                    value={token}
                  />
                </div>
                <p className="text-xs leading-5 text-muted-foreground" id="inqtrix-auth-help">
                  {t.authLock.memoryOnly}
                </p>
                {error && (
                  <p className="flex gap-1.5 text-xs font-medium leading-5 text-destructive">
                    <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                    <span>{error}</span>
                  </p>
                )}
                <Button className="h-10 w-full gap-2" disabled={isSubmitting} type="submit">
                  <LockKeyhole className="size-4" />
                  {isSubmitting ? t.authLock.submitting : t.authLock.submit}
                </Button>
              </form>
            </div>
          </div>
        </div>

        <div className="border-t border-border bg-background/70 px-5 py-4 sm:px-7">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-normal text-muted-foreground">
            <Scale className="size-3.5" />
            {t.authLock.noticeTitle}
          </div>
          <ul className="mt-3 grid gap-x-5 gap-y-2 text-xs leading-5 text-muted-foreground sm:grid-cols-2">
            {t.authLock.notices.map((notice) => (
              <li className="flex gap-2" key={notice}>
                <span className="mt-2 size-1 shrink-0 rounded-full bg-muted-foreground/60" />
                <span>{notice}</span>
              </li>
            ))}
          </ul>
          <div className="mt-4 flex flex-col gap-3 border-t border-border pt-3 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <p>{t.authLock.copyright}</p>
            <div className="flex flex-wrap gap-2 font-medium">
              <a
                className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2 py-1 text-muted-foreground transition hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                href={t.authLock.repositoryUrl}
                rel="noreferrer"
                target="_blank"
              >
                <Github className="size-3.5" />
                {t.authLock.repositoryLabel}
                <ExternalLink className="size-3" />
              </a>
              <a
                className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2 py-1 text-muted-foreground transition hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                href={t.authLock.documentationUrl}
                rel="noreferrer"
                target="_blank"
              >
                <BookOpen className="size-3.5" />
                {t.authLock.documentationLabel}
                <ExternalLink className="size-3" />
              </a>
              <a
                className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2 py-1 text-muted-foreground transition hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                href={t.authLock.licenseUrl}
                rel="noreferrer"
                target="_blank"
              >
                <Scale className="size-3.5" />
                {t.authLock.licenseLabel}
                <ExternalLink className="size-3" />
              </a>
            </div>
          </div>
        </div>
      </motion.section>
    </div>
  )
}

function LockBenefit({
  icon: Icon,
  label,
}: {
  icon: LucideIcon
  label: string
}) {
  return (
    <div className="flex items-start gap-2 rounded-md px-1 py-2">
      <Icon className="mt-0.5 size-4 shrink-0 text-brand" />
      <span className="text-sm leading-5 text-foreground">{label}</span>
    </div>
  )
}
