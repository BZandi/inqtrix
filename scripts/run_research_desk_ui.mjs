import { spawnSync } from 'node:child_process'

const scriptName = process.argv[2]
const allowedScripts = new Set(['dev', 'build', 'lint', 'test', 'typecheck', 'preview'])

if (!allowedScripts.has(scriptName)) {
  console.error(`Unknown Research Desk script: ${scriptName ?? '<missing>'}`)
  process.exit(1)
}

const userAgent = process.env.npm_config_user_agent ?? ''
const isNpm = userAgent.startsWith('npm/')

const command = isNpm ? 'npm' : 'corepack'
const args = isNpm
  ? ['--workspace', '@inqtrix/research-desk', 'run', scriptName]
  : ['pnpm', '-C', 'apps/research-desk', 'run', scriptName]

// On Windows, npm/corepack ship only as .cmd/.ps1 shims (no .exe). With the default
// shell:false, libuv's CreateProcess does not consult PATHEXT and cannot resolve or run
// a bare 'npm'/'corepack' name, raising "spawnSync npm ENOENT". Routing through the
// platform shell on win32 applies PATHEXT so 'npm' -> npm.cmd and 'corepack' ->
// corepack.cmd resolve. POSIX (macOS/Linux) deliberately keeps shell:false: it resolves
// the on-PATH shim directly and thereby avoids Node's DEP0190 (args-not-escaped) warning
// that shell:true + an args array triggers on Node >= 24. Shelling out is safe here
// because every arg is a constant or scriptName, validated against the fixed allow-set
// above, so no untrusted input reaches the shell.
const result = spawnSync(command, args, {
  stdio: 'inherit',
  shell: process.platform === 'win32',
})

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

process.exit(result.status ?? 1)
