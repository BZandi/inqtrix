import { useId } from 'react'

type BrandMarkProps = {
  className?: string
}

/**
 * Small-size Inqtrix mark for compact UI chrome.
 *
 * The full asset-kit mark has several rings, connector lines, shadows, and many
 * nodes. Those details are useful in hero placements, but become pixel-noisy in
 * the 24-32px header range. This version turns the brand idea into a compact
 * "verified query" mark: a Q-like inquiry orbit, one evidence node, and a check.
 * It uses theme tokens for the gradient so it stays balanced across the app's
 * light, dark, preset, and high contrast palettes.
 *
 * Decorative: the adjacent "Inqtrix" wordmark carries the accessible name, so
 * this is `aria-hidden`.
 */
export function BrandMark({ className }: BrandMarkProps) {
  const gradientId = `inqtrix-mark-${useId().replaceAll(':', '')}`
  const gradient = `url(#${gradientId})`

  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 32 32"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient
          gradientUnits="userSpaceOnUse"
          id={gradientId}
          x1="6"
          x2="27"
          y1="5"
          y2="27"
        >
          <stop stopColor="var(--brand)" />
          <stop offset="0.55" stopColor="var(--file)" />
          <stop offset="1" stopColor="var(--success)" />
        </linearGradient>
      </defs>
      <g strokeLinecap="round" strokeLinejoin="round">
        <circle cx="15.25" cy="15.25" r="10.1" stroke={gradient} strokeWidth="4.25" />
        <path
          d="m21.75 22.05 4.85 4.75"
          stroke={gradient}
          strokeWidth="4.25"
        />
        <circle cx="22.05" cy="8.45" fill={gradient} r="3.05" />
        <circle cx="22.05" cy="8.45" fill="var(--background)" r="1.3" />
        <path
          d="m10.55 15.75 3.55 3.6 6.65-7.3"
          stroke="var(--success)"
          strokeWidth="2.65"
        />
      </g>
    </svg>
  )
}
