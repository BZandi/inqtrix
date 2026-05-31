import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = {
  children: ReactNode
  retryLabel: string
  title: string
}

type State = {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    if (typeof console !== 'undefined') {
      console.error('ErrorBoundary caught:', error, info.componentStack)
    }
  }

  reset = (): void => {
    this.setState({ error: null })
  }

  render(): ReactNode {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div className="my-2 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
        <p className="font-medium">{this.props.title}</p>
        <p className="mt-1 break-words text-xs text-muted-foreground">
          {error.message || error.name}
        </p>
        <button
          type="button"
          onClick={this.reset}
          className="mt-2 text-xs underline underline-offset-4"
        >
          {this.props.retryLabel}
        </button>
      </div>
    )
  }
}
