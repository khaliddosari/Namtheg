// Builds and serves an isolated copy of the frontend for the e2e scripts.
//
//   node app.js build   copy Frontend/ to a temp folder, link node_modules, build
//                       it against the mock backend
//   node app.js serve   serve that build on :3001
//   node app.js clean   unlink node_modules and delete the copy
//
// Never builds inside Frontend/: `next build` there would overwrite the .next
// folder a running `npm run dev` serves from and break it until restarted.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");
const { MOCK } = require("./lib");

const FRONTEND = path.resolve(__dirname, "..");
const COPY = path.resolve(process.env.E2E_APP_DIR ?? path.join(os.tmpdir(), "namtheg-e2e-app"));
const PORT = new URL(process.env.E2E_APP_URL ?? "http://localhost:3001").port || "3001";
const LINK = path.join(COPY, "node_modules");
// Not copied: dependencies (linked instead), build output, this folder, local secrets.
const SKIP = new Set(["node_modules", ".next", "e2e", ".env.local"]);

const same = (p) => (process.platform === "win32" ? p.toLowerCase() : p);
if (same(COPY) === same(FRONTEND) || same(COPY).startsWith(same(FRONTEND) + path.sep)) {
  console.error(`Refusing to use ${COPY}: the copy must live outside Frontend/.`);
  process.exit(1);
}

// Removes the node_modules link itself, never what it points at.
function unlinkModules() {
  let stat;
  try {
    stat = fs.lstatSync(LINK);
  } catch {
    return;
  }
  if (!stat.isSymbolicLink()) throw new Error(`${LINK} is a real folder, not a link; leaving it alone.`);
  fs.unlinkSync(LINK);
}

function build() {
  fs.mkdirSync(COPY, { recursive: true });
  // Start from a clean mirror, keeping .next/cache so rebuilds stay quick.
  for (const entry of fs.readdirSync(COPY)) {
    if (entry === ".next" || entry === "node_modules") continue;
    fs.rmSync(path.join(COPY, entry), { recursive: true, force: true });
  }
  fs.cpSync(FRONTEND, COPY, {
    recursive: true,
    filter: (src) => path.dirname(src) !== FRONTEND || !SKIP.has(path.basename(src)),
  });
  if (!fs.existsSync(LINK)) fs.symlinkSync(path.join(FRONTEND, "node_modules"), LINK, "junction");

  console.log(`Building ${COPY} against ${MOCK}`);
  const res = spawnSync(process.execPath, [path.join(LINK, "next", "dist", "bin", "next"), "build"], {
    cwd: COPY,
    env: { ...process.env, BACKEND_URL: MOCK },
    stdio: "inherit",
  });
  process.exit(res.status ?? 1);
}

function serve() {
  if (!fs.existsSync(path.join(COPY, ".next", "BUILD_ID"))) {
    console.error("No build yet. Run `npm run build` first.");
    process.exit(1);
  }
  console.log(`Serving ${COPY} on http://localhost:${PORT}`);
  const res = spawnSync(process.execPath, [path.join(LINK, "next", "dist", "bin", "next"), "start", "-p", PORT], {
    cwd: COPY,
    stdio: "inherit",
  });
  process.exit(res.status ?? 0);
}

function clean() {
  unlinkModules();
  fs.rmSync(COPY, { recursive: true, force: true });
  console.log(`Removed ${COPY}`);
}

const commands = { build, serve, clean };
const command = commands[process.argv[2]];
if (!command) {
  console.error("Usage: node app.js build | serve | clean");
  process.exit(1);
}
command();
