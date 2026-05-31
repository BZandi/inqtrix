import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'

type MarkdownReportProps = {
  markdown: string
}

export function MarkdownReport({ markdown }: MarkdownReportProps) {
  return (
    <div className="report-markdown w-full min-w-0 max-w-full [overflow-wrap:anywhere]">
      <MarkdownRenderer markdown={markdown} variant="report" />
    </div>
  )
}
