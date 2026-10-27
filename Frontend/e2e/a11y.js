// Accessibility audit: axe-core WCAG 2.1 A/AA on the visible parts of each
// page, plus a keyboard walk that checks every focus stop shows a ring.
// Run: npm run a11y
const fs = require("fs");
const { APP, launch, resetMock } = require("./lib");

const AXE = fs.readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
const PAGES = [
  ["empty", "/"],
  ["data", "/?run=reg-done&stage=data"],
  ["uploaded", "/?run=fresh&stage=data"],
  ["reading", "/?run=reading&stage=data"],
  ["training", "/?run=running&stage=training"],
  ["failed", "/?run=failed&stage=training"],
  ["results", "/?run=reg-done&stage=results"],
  ["deploy", "/?run=live-done&stage=deploy"],
  ["share", "/share/live-done"],
];

async function axe(page, include) {
  await page.addScriptTag({ content: AXE });
  return page.evaluate(async (include) => {
    const r = await window.axe.run(
      { include },
      { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] }, resultTypes: ["violations", "incomplete"] }
    );
    return {
      violations: r.violations.map((v) => ({
        id: v.id,
        impact: v.impact,
        nodes: v.nodes.slice(0, 4).map((n) => `${n.target.join(" ")} :: ${n.failureSummary?.split("\n")[1]?.trim() ?? ""}`),
        count: v.nodes.length,
      })),
      incomplete: r.incomplete.map((v) => `${v.id} (${v.nodes.length})`),
    };
  }, include);
}

async function keyboardWalk(page) {
  const missing = [];
  const stops = [];
  await page.locator("body").click({ position: { x: 2, y: 2 } });
  for (let i = 0; i < 70; i++) {
    await page.keyboard.press("Tab");
    const info = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      const s = getComputedStyle(el);
      const ring = (s.outlineStyle !== "none" && parseFloat(s.outlineWidth) > 0) || (s.boxShadow && s.boxShadow !== "none");
      const name = (el.getAttribute("aria-label") || el.textContent || el.getAttribute("placeholder") || el.id || el.tagName).trim().slice(0, 40);
      const inInert = !!el.closest("[inert]");
      return { key: el.tagName + ":" + name, ring, inInert };
    });
    if (!info) continue;
    if (stops.includes(info.key)) break;
    stops.push(info.key);
    if (!info.ring) missing.push(info.key);
    if (info.inInert) missing.push(`INERT ${info.key}`);
  }
  return { stops: stops.length, missing };
}

(async () => {
  const browser = await launch();
  let problems = 0;
  for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    const ctx = await browser.newContext({ viewport: vp });
    const page = await ctx.newPage();
    for (const [name, url] of PAGES) {
      await resetMock();
      await page.goto(APP + url);
      await page.waitForTimeout(1500);
      const include = [["header"], ['nav[aria-label="Stages"]'], ["main > div > section:not([inert])"], ["main:not(:has(section))"]];
      const r = await axe(page, include);
      const walk = vp.width > 1000 ? await keyboardWalk(page) : null;
      const bad = r.violations.length || (walk && walk.missing.length);
      if (bad) problems++;
      console.log(`${bad ? "ISSUES" : "clean "} ${vp.width} ${name}: violations ${r.violations.length}${walk ? `, focus stops ${walk.stops}, without ring ${walk.missing.length}` : ""}${r.incomplete.length ? `, needs review: ${r.incomplete.join(", ")}` : ""}`);
      for (const v of r.violations) console.log(`    ${v.impact} ${v.id} x${v.count}\n      ${v.nodes.join("\n      ")}`);
      if (walk?.missing.length) console.log(`    no focus ring: ${walk.missing.join(" | ")}`);
    }
    await ctx.close();
  }
  // Dialogs, open, over content.
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await page.waitForTimeout(1500);
  await page.getByRole("button", { name: "Read the full rationale" }).click();
  await page.waitForTimeout(400);
  const r = await axe(page, [['[data-slot="dialog-content"]']]);
  console.log(`dialog: violations ${r.violations.length}`, r.violations.map((v) => v.id).join(", "));
  if (r.violations.length) problems++;
  await browser.close();
  if (problems) process.exitCode = 1;
})();
