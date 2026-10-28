// Draws the site's Open Graph image, app/opengraph-image.png (1200×630), from
// the page below, in the browser the checks use. A browser draws it with the
// site's own fonts, the wordmark's swash forms (which the ImageResponse
// renderer can't apply) and the navy wash, the way the intro shows them.
// Next serves the file as og:image. Run: npm run og-image
const path = require("path");
const { launch } = require("./lib");

const OUT = path.join(__dirname, "..", "app", "opengraph-image.png");
const FONTS = "https://cdn.jsdelivr.net/gh/khaliddosari/thmanyah-fonts@v1/thmanyah%20typeface/";

// Tokens from app/globals.css (the .dark block), and the intro's wordmark lockup.
const PAGE = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  @font-face { font-family: "Thmanyah Sans"; font-weight: 400; src: url("${FONTS}thmanyahsans/woff2/thmanyahsans-Regular.woff2") format("woff2"); }
  @font-face { font-family: "Thmanyah Sans"; font-weight: 700; src: url("${FONTS}thmanyahsans/woff2/thmanyahsans-Bold.woff2") format("woff2"); }
  @font-face { font-family: "Thmanyah Serif Display"; font-weight: 900; src: url("${FONTS}thmanyahserifdisplay/woff2/thmanyahserifdisplay-Black.woff2") format("woff2"); }
  * { margin: 0; box-sizing: border-box; }
  html, body { width: 1200px; height: 630px; overflow: hidden; }
  body {
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 26px;
    font-family: "Thmanyah Sans", sans-serif; color: oklch(0.95 0.012 250); text-align: center;
    background-color: oklch(0.145 0.032 263);
    background-image:
      radial-gradient(44rem 30rem at 6% -10%, oklch(0.55 0.14 250 / 0.34), transparent 60%),
      radial-gradient(38rem 28rem at 100% 4%, oklch(0.62 0.12 232 / 0.2), transparent 62%),
      radial-gradient(40rem 34rem at 88% 104%, oklch(0.45 0.16 276 / 0.3), transparent 62%),
      radial-gradient(34rem 30rem at 18% 100%, oklch(0.5 0.13 264 / 0.2), transparent 64%);
  }
  .card {
    width: 1080px; padding: 50px 64px 54px; border-radius: 40px;
    display: flex; flex-direction: column; align-items: center; gap: 24px;
    background: oklch(0.88 0.035 252 / 0.07);
    box-shadow: 0 0 0 1px oklch(1 0 0 / 0.1), inset 0 1px 0 0 oklch(1 0 0 / 0.08),
      0 18px 44px -18px oklch(0.06 0.04 264 / 0.8);
  }
  .lockup { display: flex; flex-direction: column; align-items: center; font-size: 84px; line-height: 1; }
  .brand {
    font-family: "Thmanyah Serif Display", serif; font-weight: 900; font-size: 1.22em; color: oklch(0.8 0.11 240);
    font-feature-settings: "ss01", "ss03", "ss04", "ss05", "ss06", "ss07", "ss08", "salt", "swsh", "dlig";
  }
  .latin { margin-top: 0.62em; transform: translateX(-0.037em); font-size: 0.565em; font-weight: 700; color: oklch(0.74 0.035 255); }
  h1 { font-size: 62px; font-weight: 700; line-height: 1.1; letter-spacing: -0.01em; }
  p.tagline { max-width: 880px; font-size: 27px; line-height: 1.4; color: oklch(0.74 0.035 255); }
  .site { font-size: 21px; letter-spacing: 0.04em; color: oklch(0.8 0.11 240 / 0.9); }
</style>
</head>
<body>
  <main class="card">
    <p class="lockup"><span class="brand" lang="ar" dir="rtl">نَمذِج</span><span class="latin">Namtheg</span></p>
    <h1>From a raw table to a live model</h1>
    <p class="tagline">An AI analyst studies your data, models train on an NVIDIA H200, and the winner is explained and served for live predictions.</p>
  </main>
  <p class="site">namtheg.khalid-ai.dev</p>
</body>
</html>`;

(async () => {
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
  await page.setContent(PAGE, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  const loaded = await page.evaluate(() => [...document.fonts].filter((f) => f.status === "loaded").length);
  if (loaded < 3) throw new Error(`Only ${loaded} of 3 fonts loaded; check the connection to cdn.jsdelivr.net.`);
  await page.screenshot({ path: OUT });
  await browser.close();
  console.log(`Wrote ${OUT}`);
})();
