import type { ReactNode } from 'react'
import * as React from 'react'

import { X } from '@/components/icons'
import { cn } from '@/lib/utils'

type DialogProps = {
  children: ReactNode
  className?: string
  /** Localised label for the close affordance (defaults to "Close"). */
  closeLabel?: string
  description?: ReactNode
  /** When false, overlay-click and Esc do NOT close (e.g. mid-submit). */
  dismissable?: boolean
  footer?: ReactNode
  onClose: () => void
  open: boolean
  title: ReactNode
}

const FOCUSABLE =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])'

/**
 * Hand-built modal dialog (overlay tier §6: `rounded-xl` + `shadow-lg`),
 * modelled on the share dialog but extracted as the shared primitive.
 * Accessibility (P7): focus moves into the panel on open and is trapped
 * with Tab/Shift+Tab, Esc closes (when dismissable), and focus returns to
 * the previously-focused element on unmount. Wrap the children in a
 * `<form>` to get Enter-to-submit. Renders nothing when `open` is false.
 */
export function Dialog({
  children,
  className,
  closeLabel = 'Close',
  description,
  dismissable = true,
  footer,
  onClose,
  open,
  title,
}: DialogProps) {
  const panelRef = React.useRef<HTMLDivElement>(null)
  const titleId = React.useId()
  // onClose/dismissable live in refs so the focus effect can depend on `open`
  // ALONE — otherwise an inline onClose (new identity each render) re-runs the
  // effect on every parent re-render and yanks focus back to the panel
  // mid-interaction (e.g. clicking "Copy" on the one-time-reveal dialog).
  const onCloseRef = React.useRef(onClose)
  const dismissableRef = React.useRef(dismissable)
  onCloseRef.current = onClose
  dismissableRef.current = dismissable

  React.useEffect(() => {
    if (!open) return undefined
    const previouslyFocused = document.activeElement as HTMLElement | null
    const visibleFocusables = () => {
      const panel = panelRef.current
      if (!panel) return [] as HTMLElement[]
      return Array.from(
        panel.querySelectorAll<HTMLElement>(FOCUSABLE),
      ).filter((element) => element.offsetParent !== null)
    }
    ;(visibleFocusables()[0] ?? panelRef.current)?.focus()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && dismissableRef.current) {
        event.preventDefault()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab') return
      const items = visibleFocusables()
      if (items.length === 0) return
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      previouslyFocused?.focus?.()
    }
  }, [open])

  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-background/75 px-4 py-8 backdrop-blur"
      onMouseDown={(event) => {
        if (dismissable && event.target === event.currentTarget) onClose()
      }}
    >
      <section
        aria-labelledby={titleId}
        aria-modal="true"
        className={cn(
          'w-full max-w-lg overflow-hidden rounded-xl border border-border bg-background shadow-lg',
          className,
        )}
        ref={panelRef}
        role="dialog"
        tabIndex={-1}
      >
        <div className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h2 className="t-section truncate text-foreground" id={titleId}>
              {title}
            </h2>
            {description ? (
              <p className="t-meta text-muted-foreground">{description}</p>
            ) : null}
          </div>
          <button
            aria-label={closeLabel}
            className="flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="icon-sm" />
          </button>
        </div>
        <div className="px-4 py-4">{children}</div>
        {footer ? (
          <div className="flex items-center justify-end gap-2 border-t border-border px-4 py-3">
            {footer}
          </div>
        ) : null}
      </section>
    </div>
  )
}
