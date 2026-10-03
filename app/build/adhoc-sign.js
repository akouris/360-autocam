// electron-builder afterPack hook: ad-hoc sign the whole app (free, no Apple
// Developer ID). Apple Silicon refuses to run unsigned code, and a bundle whose
// signature doesn't match its contents shows as "damaged". So: sign every
// Mach-O binary we ship (bundled Python libs, ffmpeg) first, then the app.
const { execFileSync } = require('child_process')
const fs = require('fs')
const path = require('path')

const MACHO = new Set(['feedfacf', 'cffaedfe', 'cafebabe', 'bebafeca'])

function isMachO(file) {
  try {
    const fd = fs.openSync(file, 'r')
    const buf = Buffer.alloc(4)
    fs.readSync(fd, buf, 0, 4, 0)
    fs.closeSync(fd)
    return MACHO.has(buf.toString('hex'))
  } catch (_) { return false }
}

function walk(dir, out) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isSymbolicLink()) continue
    if (e.isDirectory()) walk(p, out)
    else if (e.isFile() && isMachO(p)) out.push(p)
  }
  return out
}

exports.default = async function (context) {
  if (context.electronPlatformName !== 'darwin') return
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`)
  const res = path.join(app, 'Contents', 'Resources')

  // Python would otherwise write __pycache__ into the app on first run, which
  // breaks the signature ("damaged"). Ship the bytecode pre-compiled and sealed;
  // "unchecked-hash" pycs are never re-validated/rewritten. The app also runs
  // Python with PYTHONDONTWRITEBYTECODE=1.
  const py = path.join(res, 'python', 'bin', 'python3')
  if (fs.existsSync(py)) {
    execFileSync('find', [res, '-name', '__pycache__', '-type', 'd', '-prune', '-exec', 'rm', '-rf', '{}', '+'])
    console.log('  • pre-compiling Python bytecode')
    try {
      execFileSync(py, ['-m', 'compileall', '-q', '-j', '0', '--invalidation-mode', 'unchecked-hash',
                        path.join(res, 'python', 'lib'), path.join(res, 'scripts')],
                   { stdio: 'pipe', env: { PATH: '/usr/bin:/bin', HOME: process.env.HOME } })
    } catch (e) { /* a few vendored files don't compile; harmless */ }
  }

  const bins = walk(res, [])
  console.log(`  • ad-hoc signing ${bins.length} bundled binaries`)
  for (let i = 0; i < bins.length; i += 200) {
    execFileSync('codesign', ['--force', '--sign', '-', ...bins.slice(i, i + 200)], { stdio: 'pipe' })
  }
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', app], { stdio: 'inherit' })
  execFileSync('codesign', ['--verify', '--deep', '--strict', app], { stdio: 'inherit' })
  console.log('  • ad-hoc signature verified')
}
