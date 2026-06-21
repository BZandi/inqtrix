import { cn } from '@/lib/utils'

/** Derive up to two uppercase initials from a display name or email.
 * Exported for the unit tests; the fallback chain is name words ->
 * email local part -> "?" so the avatar never renders empty. */
export function initialsFor(displayName: string | null, email: string | null): string {
  const name = (displayName ?? '').trim()
  if (name) {
    const words = name.split(/\s+/).filter(Boolean)
    const first = words[0]?.[0] ?? ''
    const second = words.length > 1 ? words[words.length - 1][0] : ''
    return `${first}${second}`.toUpperCase() || '?'
  }
  const local = (email ?? '').split('@')[0]?.trim()
  if (local) return local[0].toUpperCase()
  return '?'
}

type InitialsAvatarProps = {
  displayName: string | null
  email: string | null
  size?: 'sm' | 'md'
  className?: string
}

/** Round initials badge for the signed-in identity.
 * Control primitive: owns its raw text sizes per DESIGN.md section 4
 * (feature code must not use ad-hoc pixel sizes). Brand tones are the
 * sanctioned identity colour (P2). */
export function InitialsAvatar({ displayName, email, size = 'md', className }: InitialsAvatarProps) {
  return (
    <span
      aria-hidden
      className={cn(
        'grid shrink-0 select-none place-items-center rounded-full bg-brand-subtle font-semibold uppercase text-brand',
        size === 'md' ? 'size-7 text-[11px]' : 'size-5 text-[10px]',
        className,
      )}
    >
      {initialsFor(displayName, email)}
    </span>
  )
}
