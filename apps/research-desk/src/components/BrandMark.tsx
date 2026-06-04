type BrandMarkProps = {
  className?: string
}

/**
 * Inqtrix brand mark, simplified for small UI sizes (header, ~24–32px): a broken
 * research-orbit ring with a few nodes around a central verified-check core, in
 * the brand teal→purple gradient. Hand-derived from the detailed logo-kit mark,
 * which turns to mush below ~40px (too many thin rings + ~20 tiny nodes). Drawn
 * as a vector so it stays crisp at any size, and self-contained (filled core +
 * mid-tone gradient) so it reads on both light and dark headers without a swap.
 *
 * Decorative: the adjacent "Inqtrix" wordmark carries the accessible name, so
 * this is `aria-hidden`.
 */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 32 32"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="inqtrix-mark" x1="4" y1="5" x2="27" y2="27" gradientUnits="userSpaceOnUse">
          <stop stopColor="#7F77DD" />
          <stop offset="1" stopColor="#34BD98" />
        </linearGradient>
      </defs>
      <g strokeLinecap="round" strokeLinejoin="round">
        <circle cx="16" cy="16" r="12.3" stroke="url(#inqtrix-mark)" strokeWidth="2.1" strokeDasharray="58 19" transform="rotate(108 16 16)" />
        <path d="M16 16 5.4 9.9M16 16l11-1.6" stroke="url(#inqtrix-mark)" strokeWidth="1.3" opacity="0.5" />
        <circle cx="5.4" cy="9.9" r="2.3" fill="#7F77DD" />
        <circle cx="27" cy="14.4" r="2" fill="#34BD98" />
        <circle cx="13.6" cy="27.9" r="1.7" fill="#46C0A6" />
        <circle cx="16" cy="16" r="6.2" fill="url(#inqtrix-mark)" />
        <path d="m13 16.3 2.1 2.2 4.1-4.6" stroke="#fff" strokeWidth="2" />
      </g>
    </svg>
  )
}
