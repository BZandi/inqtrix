import { useEffect, useRef, type RefObject } from 'react'

const FOCUSABLE =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])'

type ModalFocusTrapOptions = {
  dismissable?: boolean
  onClose: () => void
  open: boolean
  panelRef: RefObject<HTMLElement | null>
}

export function useModalFocusTrap({
  dismissable = true,
  onClose,
  open,
  panelRef,
}: ModalFocusTrapOptions) {
  const onCloseRef = useRef(onClose)
  const dismissableRef = useRef(dismissable)
  onCloseRef.current = onClose
  dismissableRef.current = dismissable

  useEffect(() => {
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
  }, [open, panelRef])
}
