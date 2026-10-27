// Shared setup for the e2e scripts: where the app and the mock backend live,
// and how to get a browser.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { chromium } = require("playwright-core");

const APP = process.env.E2E_APP_URL ?? "http://localhost:3001";
const MOCK = process.env.E2E_MOCK_URL ?? "http://localhost:8001";
const SAMPLE_CSV = path.join(__dirname, "..", "public", "insurance.csv");

// playwright-core ships no browser. Use CHROME_PATH if set, else the newest
// Chromium that `npx playwright-core install chromium` left in its cache.
function findChromium() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const home = os.homedir();
  const caches = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "ms-playwright"),
    path.join(home, "Library", "Caches", "ms-playwright"),
    path.join(home, ".cache", "ms-playwright"),
  ].filter(Boolean);
  const binaries = [
    "chrome-win64/chrome.exe",
    "chrome-win/chrome.exe",
    "chrome-linux64/chrome",
    "chrome-linux/chrome",
    "chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium",
    "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
  ];
  for (const cache of caches) {
    if (!fs.existsSync(cache)) continue;
    const builds = fs
      .readdirSync(cache)
      .filter((d) => /^chromium-\d+$/.test(d))
      .sort((a, b) => Number(b.split("-")[1]) - Number(a.split("-")[1]));
    for (const build of builds) {
      for (const bin of binaries) {
        const exe = path.join(cache, build, bin);
        if (fs.existsSync(exe)) return exe;
      }
    }
  }
  return undefined; // let playwright-core report what to install
}

const launch = () => {
  const executablePath = findChromium();
  return chromium.launch(executablePath ? { executablePath } : {});
};

const resetMock = () => fetch(`${MOCK}/__mock/reset`, { method: "POST" });

function shotsDir(name) {
  const dir = path.join(__dirname, "shots", name);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

module.exports = { APP, MOCK, SAMPLE_CSV, launch, resetMock, shotsDir };
