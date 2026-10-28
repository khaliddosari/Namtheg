// Layout audit: every stage with data at seven viewports. Automated checks plus
// screenshots in shots/layout (viewport-sized, since the wash is fixed) for a
// look by eye. Run: npm run layout
const path = require("path");
const { APP, launch, resetMock, shotsDir } = require("./lib");

const OUT = shotsDir("layout");

const VIEWPORTS = [
  [1920, 1080], [1440, 900], [1366, 657], [1024, 768], [768, 1024], [390, 844], [360, 740],
];
const PAGES = [
  ["intro", "/"],
  ["empty", "/?stage=data"],
  ["data", "/?run=reg-done&stage=data"],
  ["uploaded", "/?run=fresh&stage=data"],
  ["reading", "/?run=reading&stage=data"],
  ["training", "/?run=running&stage=training"],
  ["results", "/?run=reg-done&stage=results"],
  ["deploy", "/?run=live-done&stage=deploy"],
  ["share", "/share/live-done"],
];

async function audit(page) {
  return page.evaluate(() => {
    const issues = [];
    const vw = window.innerWidth;
    if (document.documentElement.scrollWidth > vw + 1) issues.push(`page scrolls sideways (${document.documentElement.scrollWidth} > ${vw})`);
    const root = document.querySelector("main > div > section:not([inert])") ?? document.querySelector("main");
    const box = root.getBoundingClientRect();
    const scrollsX = (el) => {
      for (let a = el.parentElement; a && a !== root; a = a.parentElement) {
        const s = getComputedStyle(a);
        if ((s.overflowX === "auto" || s.overflowX === "scroll") && a.scrollWidth > a.clientWidth) return true;
        if (s.overflowX === "hidden" || s.overflowX === "clip") return true;
      }
      return false;
    };
    for (const el of root.querySelectorAll("*")) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      const s = getComputedStyle(el);
      if (s.visibility === "hidden") continue;
      if ((r.right > box.right + 1 || r.left < box.left - 1) && !scrollsX(el)) {
        issues.push(`escapes its stage: <${el.tagName.toLowerCase()} class="${String(el.className).slice(0, 60)}"> ${Math.round(r.left)}..${Math.round(r.right)} vs ${Math.round(box.left)}..${Math.round(box.right)}`);
      }
      // Text cut off with no ellipsis to say so.
      const hasText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
      if (hasText && el.scrollWidth > el.clientWidth + 1 && s.overflowX !== "visible" && s.textOverflow !== "ellipsis" && !/auto|scroll/.test(s.overflowX)) {
        issues.push(`text clipped without ellipsis: "${el.textContent.trim().slice(0, 40)}"`);
      }
    }
    // Controls whose own content spills past their edge, clipped or not.
    for (const el of root.querySelectorAll('button, a, [role="button"], label')) {
      const r = el.getBoundingClientRect();
      if (!r.width) continue;
      if (el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).textOverflow !== "ellipsis") {
        issues.push(`content overflows its control: "${(el.getAttribute("aria-label") || el.textContent).trim().slice(0, 40)}" (${el.scrollWidth} > ${el.clientWidth})`);
      }
    }
    // Locked pages: panels should reach the bottom of the stage.
    const panels = [...root.querySelectorAll('[data-slot="panel"]')];
    if (panels.length && getComputedStyle(root).overflowY !== "visible" && root.scrollHeight <= root.clientHeight + 1) {
      const bottom = Math.max(...panels.map((p) => p.getBoundingClientRect().bottom));
      if (box.bottom - bottom > 48) issues.push(`empty gap of ${Math.round(box.bottom - bottom)}px under the cards`);
    }
    return [...new Set(issues)].slice(0, 12);
  });
}

(async () => {
  const browser = await launch();
  let total = 0;
  for (const [w, h] of VIEWPORTS) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, hasTouch: w < 800, isMobile: w < 800 });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    for (const [name, url] of PAGES) {
      await resetMock();
      await page.goto(APP + url);
      await page.waitForTimeout(1600);
      const issues = await audit(page);
      await page.screenshot({ path: path.join(OUT, `${w}x${h}-${name}.png`) });
      total += issues.length;
      if (issues.length) console.log(`${w}x${h} ${name}:\n  ${issues.join("\n  ")}`);
    }
    if (errors.length) console.log(`${w}x${h} page errors: ${errors.join("; ")}`);
    await ctx.close();
  }
  await browser.close();
  console.log(`layout issues: ${total}`);
  if (total) process.exitCode = 1;
})();
