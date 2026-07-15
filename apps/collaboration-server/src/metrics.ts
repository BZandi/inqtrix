const HISTOGRAM_BUCKETS_MS = [5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000]

type Histogram = {
  bucketCounts: number[]
  count: number
  sumMs: number
}

export class SidecarMetrics {
  private readonly counters = new Map<string, number>()
  private readonly gauges = new Map<string, number>()
  private readonly histograms = new Map<string, Histogram>()

  increment(name: string, labels: Readonly<Record<string, string>> = {}): void {
    this.add(name, 1, labels)
  }

  add(
    name: string,
    value: number,
    labels: Readonly<Record<string, string>> = {},
  ): void {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error('Metric counter increments must be non-negative safe integers')
    }
    const key = metricKey(name, labels)
    this.counters.set(key, (this.counters.get(key) ?? 0) + value)
  }

  observeMilliseconds(name: string, valueMs: number): void {
    const histogram = this.histograms.get(name) ?? {
      bucketCounts: HISTOGRAM_BUCKETS_MS.map(() => 0),
      count: 0,
      sumMs: 0,
    }
    histogram.count += 1
    histogram.sumMs += valueMs
    HISTOGRAM_BUCKETS_MS.forEach((bucket, index) => {
      if (valueMs <= bucket) histogram.bucketCounts[index] = (histogram.bucketCounts[index] ?? 0) + 1
    })
    this.histograms.set(name, histogram)
  }

  set(name: string, value: number, labels: Readonly<Record<string, string>> = {}): void {
    this.gauges.set(metricKey(name, labels), value)
  }

  render(): string {
    const lines: string[] = []
    for (const [key, value] of [...this.counters].sort(([left], [right]) => left.localeCompare(right))) {
      lines.push(`${key} ${value}`)
    }
    for (const [key, value] of [...this.gauges].sort(([left], [right]) => left.localeCompare(right))) {
      lines.push(`${key} ${value}`)
    }
    for (const [name, histogram] of [...this.histograms].sort(([left], [right]) => left.localeCompare(right))) {
      HISTOGRAM_BUCKETS_MS.forEach((bucket, index) => {
        lines.push(`${name}_bucket{le="${bucket / 1_000}"} ${histogram.bucketCounts[index] ?? 0}`)
      })
      lines.push(`${name}_bucket{le="+Inf"} ${histogram.count}`)
      lines.push(`${name}_sum ${histogram.sumMs / 1_000}`)
      lines.push(`${name}_count ${histogram.count}`)
    }
    return `${lines.join('\n')}\n`
  }
}

function metricKey(name: string, labels: Readonly<Record<string, string>>): string {
  if (!/^[a-z_:][a-z0-9_:]*$/.test(name)) throw new Error('Invalid metric name')
  const entries = Object.entries(labels).sort(([left], [right]) => left.localeCompare(right))
  if (entries.length === 0) return name
  const encoded = entries.map(([key, value]) => {
    if (!/^[a-z_][a-z0-9_]*$/.test(key)) throw new Error('Invalid metric label')
    const safeValue = value.replace(/[^a-zA-Z0-9_.:-]/g, '_')
    return `${key}="${safeValue}"`
  })
  return `${name}{${encoded.join(',')}}`
}
