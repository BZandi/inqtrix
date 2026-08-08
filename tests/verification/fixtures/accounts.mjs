import { assertFixture, fetchActorJson } from './api.mjs'
import { temporaryUserDescriptors } from './run-scope.mjs'

export {
  temporaryUserBelongsToRun,
  temporaryUserDescriptors,
} from './run-scope.mjs'

export async function ensureTemporaryUsers({
  adminActor,
  lifecycle,
  password,
  runId,
  descriptors = temporaryUserDescriptors(runId),
}) {
  const listed = await fetchActorJson(adminActor, 'GET', '/v1/admin/users')
  const rows = []
  for (const descriptor of descriptors) {
    const cleanupHandle = await lifecycle.register({
      email: descriptor.email,
      id: `${runId}:${descriptor.email}`,
      kind: 'temporary_user',
    })
    let user = listed.users.find(
      (candidate) => candidate.email === descriptor.email,
    )
    if (!user) {
      user = await fetchActorJson(adminActor, 'POST', '/v1/admin/users', {
        data: {
          display_name: descriptor.displayName,
          email: descriptor.email,
          instance_role: 'user',
          password,
        },
        expected: [201],
      })
    } else {
      await fetchActorJson(
        adminActor,
        'POST',
        `/v1/admin/users/${user.id}:enable`,
      )
      await fetchActorJson(
        adminActor,
        'POST',
        `/v1/admin/users/${user.id}:reset-password`,
        { data: { password } },
      )
    }
    assertFixture(user?.id, 'Temporary user creation returned no user ID.')
    rows.push({ ...descriptor, cleanupHandle, id: user.id })
  }
  return rows
}

export async function disableTemporaryUser(adminActor, user, lifecycle) {
  await fetchActorJson(
    adminActor,
    'POST',
    `/v1/admin/users/${user.id}:disable`,
    { expected: [200, 404] },
  )
  await lifecycle.complete(user.cleanupHandle)
}
