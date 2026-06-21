/** Time-formatting helpers shared across features. */

/** Unix-seconds (the server wire format) -> ISO-8601 string (the format
 * every ProjectState record uses). Shared so the wire<->record timestamp
 * boundary is defined once (Designprinzip 4). */
export function isoFromUnixSeconds(seconds: number): string {
  return new Date(seconds * 1000).toISOString()
}

/** ISO-8601 string -> unix-seconds (float, preserving sub-second precision)
 * for sending a record timestamp back to the server. */
export function unixSecondsFromIso(iso: string): number {
  return new Date(iso).getTime() / 1000
}

/** Zero-padded ``HH:MM:SS`` elapsed time from a seconds count. Used by the
 * research-run timer and anywhere a stopwatch-style duration is shown. */
export function formatDuration(seconds: number): string {
  const wholeSeconds = Math.max(0, Math.round(seconds))
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const remainingSeconds = wholeSeconds % 60

  return [hours, minutes, remainingSeconds]
    .map((part) => part.toString().padStart(2, '0'))
    .join(':')
}

/** Compact elapsed form for inline history rows: ``m:ss`` under an hour,
 * ``h:mm:ss`` above. Input is milliseconds. */
export function formatDurationMsShort(ms: number): string {
  const wholeSeconds = Math.max(0, Math.round(ms / 1000))
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const seconds = wholeSeconds % 60
  const ss = seconds.toString().padStart(2, '0')
  if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${ss}`
  return `${minutes}:${ss}`
}
