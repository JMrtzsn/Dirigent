import { existsSync, mkdirSync, copyFileSync } from "fs"
import { resolve, dirname } from "path"
import { fileURLToPath } from "url"
import { execSync } from "child_process"
import { homedir } from "os"

const __dirname = dirname(fileURLToPath(import.meta.url))
const pluginRoot = resolve(__dirname, "..")

// --- Build Go binary if source is available ---
const goDir = resolve(pluginRoot, "..", "go")
const binDir = resolve(pluginRoot, "bin")
const binPath = resolve(binDir, "dirigent")

if (!existsSync(binPath)) {
  if (existsSync(resolve(goDir, "go.mod"))) {
    console.log("[opencode-dirigent] Building Go binary...")
    if (!existsSync(binDir)) mkdirSync(binDir, { recursive: true })
    try {
      execSync(`cd "${goDir}" && go build -o "${binPath}" ./cmd/dirigent`, {
        stdio: "inherit",
      })
      console.log("[opencode-dirigent] Binary built successfully.")
    } catch {
      console.error(
        "[opencode-dirigent] Failed to build binary. Ensure Go is installed.",
      )
    }
  } else {
    console.error(
      "[opencode-dirigent] No binary found and no Go source available.",
    )
  }
}

// --- Install /dirigent command globally ---
const commandSrc = resolve(pluginRoot, "commands", "dirigent.md")
const commandDest = resolve(
  homedir(),
  ".config",
  "opencode",
  "commands",
  "dirigent.md",
)

if (existsSync(commandSrc)) {
  const destDir = dirname(commandDest)
  if (!existsSync(destDir)) mkdirSync(destDir, { recursive: true })
  copyFileSync(commandSrc, commandDest)
  console.log(`[opencode-dirigent] Installed /dirigent command to ${commandDest}`)
}
