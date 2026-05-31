import { spawnSync } from 'node:child_process'

const scriptName = process.argv[2]
const allowedScripts = new Set(['dev', 'build', 'lint', 'typecheck', 'preview'])

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

const result = spawnSync(command, args, { stdio: 'inherit' })

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

process.exit(result.status ?? 1)
