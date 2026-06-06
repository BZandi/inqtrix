import { type ReactNode } from 'react'
import { BrandMark } from '@/components/BrandMark'
import { cn } from '@/lib/utils'

type WelcomeStateProps = {
  actions?: ReactNode
  className?: string
  kicker?: string
  subtitle: ReactNode
  title: string
}

export function WelcomeState({
  actions,
  className,
  kicker,
  subtitle,
  title,
}: WelcomeStateProps) {
  return (
    <div className={cn('mx-auto flex w-full max-w-md flex-col items-center text-center', className)}>
      <div className="flex size-10 items-center justify-center rounded-xl border border-border bg-surface text-brand shadow-[0_1px_2px_var(--shadow-hairline)]">
        <BrandMark className="size-5" />
      </div>
      {kicker ? (
        <p className="mt-4 t-caption text-muted-foreground/65">{kicker}</p>
      ) : null}
      <h2 className={cn('t-display text-foreground', kicker ? 'mt-1.5' : 'mt-4')}>
        {title}
      </h2>
      <p className="mt-1.5 max-w-sm t-meta text-muted-foreground">
        {subtitle}
      </p>
      {actions ? <div className="mt-5">{actions}</div> : null}
    </div>
  )
}
