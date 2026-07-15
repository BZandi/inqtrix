import type { ReactNode } from 'react'

export function EditorTopBarLayout({
  actions,
  leading,
  toolbar,
}: {
  actions: ReactNode
  leading: ReactNode
  toolbar: ReactNode
}) {
  return (
    <header
      className="grid min-w-0 inqtrix-panel-header grid-cols-[minmax(0,1fr)_minmax(0,auto)_minmax(0,1fr)] items-center gap-2 overflow-hidden border-b border-border bg-background px-3 lg:grid-cols-[minmax(12rem,1fr)_auto_minmax(12rem,1fr)]"
      data-editor-topbar
    >
      <div className="flex min-w-0 items-center gap-2" data-editor-topbar-leading>
        {leading}
      </div>
      <div className="min-w-0 overflow-hidden" data-editor-topbar-toolbar>
        {toolbar}
      </div>
      <div
        className="flex min-w-0 justify-end gap-0.5 overflow-x-auto [scrollbar-width:none]"
        data-editor-topbar-actions
      >
        {actions}
      </div>
    </header>
  )
}
