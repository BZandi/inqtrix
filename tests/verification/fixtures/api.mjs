export function assertFixture(condition, message) {
  if (!condition) throw new Error(message)
}

export async function fetchActorJson(actor, method, path, options = {}) {
  const expected = options.expected ?? [200]
  const unsafe = !['GET', 'HEAD'].includes(method)
  const headers = {
    'X-Inqtrix-Expected-User-Id': actor.user.id,
    'X-Inqtrix-Workspace-Id': actor.workspaceId,
  }
  if (unsafe) headers['X-CSRF-Token'] = actor.csrf
  const response = await actor.context.request.fetch(path, {
    data: options.data,
    headers,
    method,
  })
  const text = await response.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { text: text.slice(0, 500) }
    }
  }
  if (!expected.includes(response.status())) {
    throw new Error(
      `${actor.label} ${method} ${redactPath(path)} returned HTTP `
      + `${response.status()}: ${JSON.stringify(body)}`,
    )
  }
  return body
}

export function redactPath(path) {
  return String(path)
    .replace(/\/s\/[^/?#\s]+/gi, '/s/[REDACTED]')
    .replace(
      /\/v1\/editor\/share-links\/[^/?#\s:]+/gi,
      '/v1/editor/share-links/[REDACTED]',
    )
}
