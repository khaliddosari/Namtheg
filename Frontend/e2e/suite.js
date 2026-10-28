// End-to-end flows for the Namtheg dashboard, against the mock backend and the
// isolated build (see README.md). Run: npm test [-- "name filter"]
const path = require("path");
const fs = require("fs");
const { APP, MOCK, SAMPLE_CSV: CSV, launch, shotsDir } = require("./lib");

const OUT = shotsDir("suite");
// The bare address opens the intro; the Data stage has its own link.
const DATA = `${APP}/?stage=data`;

const mock = (p, body) =>
  fetch(MOCK + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

const tests = [];
const test = (name, fn, opts = {}) => tests.push({ name, fn, ...opts });

class Expect {
  constructor(t) {
    this.t = t;
  }
  ok(cond, msg) {
    if (!cond) throw new Error(msg);
  }
}

async function run(filter) {
  const browser = await launch();
  const results = [];
  for (const t of tests.filter((t) => !filter || t.name.includes(filter))) {
    await mock("/__mock/reset");
    const ctx = await browser.newContext({
      viewport: t.viewport ?? { width: 1440, height: 900 },
      acceptDownloads: true,
      hasTouch: !!t.touch,
      isMobile: !!t.touch,
      permissions: ["clipboard-read", "clipboard-write"],
      reducedMotion: t.reducedMotion ?? "no-preference",
    });
    const page = await ctx.newPage();
    const problems = [];
    const allow = t.allow ?? [];
    page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
    page.on("console", (m) => {
      if (m.type() !== "error") return;
      const url = m.location()?.url ?? "";
      const text = m.text();
      const net = text.match(/status of (\d{3})/);
      if (net && allow.some((a) => (a.status ? String(a.status) === net[1] : true) && (a.url ? url.includes(a.url) : true))) return;
      if (net && url.endsWith("/favicon.ico")) return problems.push(`favicon 404`);
      problems.push(`console.error: ${text} ${url}`);
    });
    const started = Date.now();
    let error = null;
    try {
      await t.fn(page, new Expect(t), ctx);
    } catch (e) {
      error = e.message.split("\n")[0];
      await page.screenshot({ path: path.join(OUT, `FAIL-${t.name.replace(/\W+/g, "_")}.png`) }).catch(() => {});
    }
    const pass = !error && problems.length === 0;
    results.push({ name: t.name, pass, error, problems, secs: ((Date.now() - started) / 1000).toFixed(1) });
    console.log(`${pass ? "PASS" : "FAIL"}  ${t.name}  (${((Date.now() - started) / 1000).toFixed(1)}s)${error ? "\n      " + error : ""}${problems.length ? "\n      " + problems.join("\n      ") : ""}`);
    await ctx.close();
  }
  await browser.close();
  const failed = results.filter((r) => !r.pass);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  if (failed.length || !results.length) process.exitCode = 1;
}

// ---------- helpers
const stage = (page) => page.evaluate(() => new URL(location.href).searchParams.get("stage"));
const activeStage = (page) => page.evaluate(() => document.querySelector("main > div > section:not([inert])")?.id);
const settle = (page, ms = 700) => page.waitForTimeout(ms);
const inStage = (page, key) => page.locator(`#stage-${key}`);
async function uploadCsv(page) {
  await page.setInputFiles("#dataset-input", CSV);
  await page.getByRole("button", { name: "Upload and profile" }).click();
  await inStage(page, "data").getByText("insurance.csv").first().waitFor();
  await page.getByRole("button", { name: "Start training" }).waitFor();
}
const pager = (page, label) => page.getByRole("navigation", { name: "Stages" }).getByRole("button", { name: new RegExp(`^${label}`) });
const shot = (page, name) => page.screenshot({ path: path.join(OUT, `${name}.png`) });
const toastText = (page) => page.locator('[data-slot="toast"]').allInnerTexts();

// ---------- intro
test("intro: a bare link opens the poster, and its button opens Data", async (page, x) => {
  await page.goto(APP);
  await settle(page);
  x.ok((await activeStage(page)) === "stage-intro", `lands on ${await activeStage(page)}`);
  const intro = inStage(page, "intro");
  await intro.getByRole("heading", { name: "From a raw table to a live model" }).waitFor();
  for (const step of ["Upload a table", "The analyst investigates it", "Models train on an NVIDIA H200", "Results with a grounded explanation", "Deploy for live predictions"]) {
    x.ok(await intro.getByText(step).count(), `explains "${step}"`);
  }
  const nav = page.getByRole("navigation", { name: "Stages" });
  x.ok((await nav.getByRole("button", { name: "Intro" }).getAttribute("aria-current")) === "step", "the pager marks the intro");
  const numbers = (await nav.getByRole("listitem").allInnerTexts()).slice(1).map((t) => t.trim().match(/^\d+/)?.[0]);
  x.ok(JSON.stringify(numbers) === JSON.stringify(["1", "2", "3", "4"]), `stages numbered 1 to 4: ${numbers}`);
  await shot(page, "intro");
  await intro.getByRole("button", { name: "Start with your data" }).click();
  await page.waitForFunction(() => new URL(location.href).searchParams.get("stage") === "data");
  x.ok((await page.evaluate(() => document.activeElement?.id)) === "stage-data", "focus lands on the Data stage");
  await inStage(page, "data").getByRole("button", { name: "Browse files" }).waitFor();
  await page.reload();
  await settle(page);
  x.ok((await activeStage(page)) === "stage-data", "a reload stays on Data");
  await page.locator("header").getByRole("button", { name: "Namtheg, back to the intro" }).click();
  await settle(page);
  x.ok((await activeStage(page)) === "stage-intro" && !new URL(page.url()).searchParams.has("stage"), `the brand goes back to the intro: ${page.url()}`);
});

test("intro: deep links still open their stage, and a run stays loaded on the intro", async (page, x) => {
  for (const [url, id] of [
    ["/?run=reg-done&stage=deploy", "stage-deploy"],
    ["/?run=fresh&stage=data", "stage-data"],
    ["/?stage=training", "stage-training"],
    ["/?run=reg-done&stage=intro", "stage-intro"],
  ]) {
    await page.goto(APP + url);
    await settle(page);
    x.ok((await activeStage(page)) === id, `${url} -> ${await activeStage(page)}`);
  }
  await inStage(page, "intro").getByRole("button", { name: "Start with your data" }).click();
  await inStage(page, "data").getByText("insurance.csv").first().waitFor();
  const params = new URL(page.url()).searchParams;
  x.ok(params.get("run") === "reg-done" && params.get("stage") === "data", `the run is kept: ${page.url()}`);
  // A run link without a stage waits on Data, not the intro, and says there when the run is missing.
  await page.goto(`${APP}/?run=nope`);
  await inStage(page, "data").getByText("Run not found").waitFor();
  x.ok((await activeStage(page)) === "stage-data", `an unknown run lands on ${await activeStage(page)}`);
}, { allow: [{ status: 404 }] });

test("intro: phones get the poster and a pager that fits", async (page, x) => {
  await page.goto(APP);
  await inStage(page, "intro").getByRole("button", { name: "Start with your data" }).waitFor();
  const fits = await page.evaluate(() =>
    [...document.querySelectorAll('nav[aria-label="Stages"] button')]
      .filter((b) => b.offsetWidth)
      .every((b) => {
        const r = b.getBoundingClientRect();
        return r.left >= 0 && r.right <= window.innerWidth;
      })
  );
  x.ok(fits, "every pager button is on screen");
  await pager(page, "Results").click();
  await settle(page);
  x.ok((await pager(page, "Results").innerText()).includes("Results"), "the current stage shows its name");
  x.ok(!(await pager(page, "Training").innerText()).includes("Training"), "the others show their number");
  await shot(page, "intro-phone");
}, { viewport: { width: 360, height: 740 }, touch: true });

// ---------- upload
test("upload: browse, R2 409 falls back to the proxy", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  x.ok((await stage(page)) === "data", "should stay on data");
  x.ok(/run=/.test(page.url()), "url should carry the run");
  await shot(page, "upload-browse");
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

test("upload: R2 direct PUT when configured", async (page, x) => {
  await mock("/__mock/config", { r2: true });
  await page.goto(DATA);
  await uploadCsv(page);
  x.ok(/run=/.test(page.url()), "url should carry the run");
});

test("upload: drag and drop onto the dropzone", async (page, x) => {
  await page.goto(DATA);
  const csv = fs.readFileSync(CSV, "utf8");
  await page.evaluate((text) => {
    const dt = new DataTransfer();
    dt.items.add(new File([text], "dropped.csv", { type: "text/csv" }));
    const zone = document.querySelector("#stage-data .border-dashed");
    zone.dispatchEvent(new DragEvent("dragover", { bubbles: true, cancelable: true, dataTransfer: dt }));
    zone.dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: dt }));
  }, csv);
  await page.getByText("dropped.csv").waitFor();
  await page.getByRole("button", { name: "Upload and profile" }).click();
  await page.getByRole("button", { name: "Start training" }).waitFor();
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

test("upload: wrong file types, image zips included, are refused", async (page, x) => {
  for (const name of ["photo.png", "pets.zip"]) {
    await page.goto(DATA);
    await page.setInputFiles("#dataset-input", { name, mimeType: "application/octet-stream", buffer: Buffer.from("x") });
    await page.getByRole("alert").filter({ hasText: "Unsupported file type" }).waitFor();
    x.ok(!(await page.getByRole("button", { name: "Upload and profile" }).count()), `${name} should not be offered for upload`);
  }
});

test("upload: over 2 GB is refused", async (page, x) => {
  await page.addInitScript(() => {
    const size = Object.getOwnPropertyDescriptor(Blob.prototype, "size");
    Object.defineProperty(File.prototype, "size", { get() { return this.name === "huge.csv" ? 3 * 1024 ** 3 : size.get.call(this); } });
  });
  await page.goto(DATA);
  await page.setInputFiles("#dataset-input", { name: "huge.csv", mimeType: "text/csv", buffer: Buffer.from("a,b\n1,2\n") });
  await page.getByRole("alert").filter({ hasText: "2 GB" }).waitFor();
});

test("upload: empty file is refused", async (page) => {
  await page.goto(DATA);
  await page.setInputFiles("#dataset-input", { name: "empty.csv", mimeType: "text/csv", buffer: Buffer.alloc(0) });
  await page.getByRole("alert").filter({ hasText: "empty" }).waitFor();
});

async function uploadSampleDirect(page) {
  await page.goto(DATA);
  await page.setInputFiles("#dataset-input", CSV);
  await page.getByRole("button", { name: "Upload and profile" }).click();
}

test("upload: the progress bar moves while a large file is sent", async (page, x, ctx) => {
  await mock("/__mock/config", { r2: true });
  await page.goto(DATA);
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: 400 * 1024 });
  const rows = "age,bmi,charges\n" + "34,27.5,1234.5\n".repeat(200000);
  await page.setInputFiles("#dataset-input", { name: "big.csv", mimeType: "text/csv", buffer: Buffer.from(rows) });
  await page.getByRole("button", { name: "Upload and profile" }).click();
  const bar = page.getByRole("progressbar", { name: "Upload progress" });
  await bar.waitFor();
  let midway = 0;
  for (let i = 0; i < 60 && !midway; i++) {
    const v = Number((await bar.getAttribute("aria-valuenow").catch(() => "0")) ?? "0");
    if (v > 0 && v < 100) midway = v;
    else await page.waitForTimeout(100);
  }
  x.ok(midway > 0, "progress should be seen between 0 and 100%");
  await shot(page, "upload-progress");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  await page.getByRole("button", { name: "Start training" }).waitFor({ timeout: 60000 });
});

test("upload: a proxy giving up on the read is not an error", async (page, x) => {
  await mock("/__mock/config", { r2: true, complete: "dropped" });
  await uploadSampleDirect(page);
  const data = inStage(page, "data");
  await data.getByText("Reading it on the server").waitFor();
  x.ok(/run=/.test(page.url()), "the run is in the address bar while the file is read");
  await data.getByText("Almost ready").waitFor();
  await shot(page, "upload-reading");
  await page.getByRole("button", { name: "Start training" }).waitFor({ timeout: 60000 });
  x.ok(!(await data.getByRole("alert").count()), "no error shown");
}, { allow: [{ status: 500, url: "/complete" }] });

test("upload: reloading while the file is read picks up where it left off", async (page) => {
  await mock("/__mock/config", { r2: true, read_seconds: 8 });
  await uploadSampleDirect(page);
  const data = inStage(page, "data");
  await data.getByText("Reading it on the server").waitFor();
  await page.reload();
  await data.getByText("Reading it on the server").waitFor();
  await page.getByRole("button", { name: "Start training" }).waitFor({ timeout: 30000 });
});

// "fail": the backend reads in the background and the read fails; "reject":
// an older backend that read inside the request answers 400.
for (const mode of ["fail", "reject"]) {
  test(`upload: a file the server can't read says why (${mode})`, async (page) => {
    await mock("/__mock/config", { r2: true, complete: mode });
    await uploadSampleDirect(page);
    const data = inStage(page, "data");
    await data.getByText("The file couldn't be read").waitFor({ timeout: 15000 });
    await data.getByText("line 12 has 9 fields").first().waitFor();
    await data.getByRole("button", { name: "Start a new run" }).waitFor();
    await shot(page, `upload-rejected-${mode}`);
  }, { allow: [{ status: 400, url: "/complete" }] });
}

test("upload: a read that fails at once can be tried again", async (page) => {
  await mock("/__mock/config", { r2: true, complete: "crash" });
  await uploadSampleDirect(page);
  const data = inStage(page, "data");
  await data.getByText("The server stopped reading it").waitFor();
  await mock("/__mock/config", { complete: "ok" });
  await data.getByRole("button", { name: "Try again" }).click();
  await page.getByRole("button", { name: "Start training" }).waitFor({ timeout: 15000 });
}, { allow: [{ status: 500, url: "/complete" }] });

// ---------- configure
test("configure: combobox search picks a target and highlights it", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  await page.locator("#target-column").click();
  await page.locator("#target-column").fill("bm");
  await page.getByRole("option", { name: "bmi" }).click();
  x.ok((await page.locator("#target-column").inputValue()) === "bmi", "target should be bmi");
  const head = await page.locator('#stage-data th span[title="Target column"]').innerText();
  x.ok(head === "bmi", `preview should mark bmi, got ${head}`);
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

test("configure: predicting a column is the only option", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  for (const name of ["Find groups", "Forecast over time"]) {
    x.ok(!(await page.getByRole("button", { name }).count()), `no "${name}" mode`);
  }
  x.ok((await page.locator("#target-column").inputValue()) === "charges", "the last column is the default target");
  x.ok(await page.getByRole("button", { name: "Start training" }).isEnabled(), "start enabled with a target");
  await shot(page, "configure");
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

test("configure: a stated prediction time reaches the run and is shown once started", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  await page.locator("#prediction-time").fill("  when the quote   is issued ");
  await page.getByRole("button", { name: "Start training" }).click();
  await page.waitForFunction(() => new URL(location.href).searchParams.get("stage") === "training");
  const run = new URL(page.url()).searchParams.get("run");
  await page.goto(`${APP}/?run=${run}&stage=data`);
  const predicting = inStage(page, "data").locator("dt", { hasText: "Predicting" }).locator("xpath=following-sibling::dd");
  await predicting.waitFor();
  const shown = await predicting.innerText();
  x.ok(shown === "when the quote is issued", `the run should show the stated time, got ${shown}`);
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

// ---------- training
test("training: start slides to training, tracks live stages, lands on results", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  await page.getByRole("button", { name: "Start training" }).click();
  await page.waitForFunction(() => new URL(location.href).searchParams.get("stage") === "training");
  x.ok((await page.evaluate(() => document.activeElement?.id)) === "stage-training", "focus should land on the training stage");
  const seen = new Set();
  for (let i = 0; i < 40 && !(await page.getByRole("button", { name: "View results" }).count()); i++) {
    const active = await inStage(page, "training").locator("li.bg-primary\\/10").allInnerTexts();
    active.forEach((a) => seen.add(a.trim()));
    await page.waitForTimeout(500);
  }
  x.ok(seen.size >= 3, `should see several live stages, saw ${[...seen].join(", ")}`);
  await shot(page, "training-done");
  await page.getByRole("button", { name: "View results" }).click();
  await page.waitForFunction(() => new URL(location.href).searchParams.get("stage") === "results");
  x.ok((await page.evaluate(() => document.activeElement?.id)) === "stage-results", "focus should land on results");
  await inStage(page, "results").getByText("GradientBoostingRegressor").first().waitFor();
}, { allow: [{ status: 409, url: "/uploads/direct" }, { status: 409, url: "/result" }, { status: 409, url: "/model_schema" }] });

test("training: cancel through the dialog", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  await page.getByRole("button", { name: "Start training" }).click();
  await inStage(page, "training").getByRole("button", { name: "Cancel run" }).click();
  await page.getByRole("dialog").getByText("Cancel this run?").waitFor();
  await page.keyboard.press("ArrowRight");
  x.ok((await stage(page)) === "training", "arrow keys must not page while the dialog is open");
  await page.getByRole("dialog").getByRole("button", { name: "Cancel run" }).click();
  await inStage(page, "training").getByText("Run cancelled").waitFor();
  x.ok((await toastText(page)).join(" ").includes("Run cancelled"), "toast should confirm");
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

test("training: failed run shows the failing stage and error", async (page, x) => {
  await page.goto(`${APP}/?run=failed`);
  await inStage(page, "training").getByText("Run failed").waitFor();
  x.ok((await stage(page)) === "training", "should land on training");
  await inStage(page, "training").getByText("CUDA out of memory").first().waitFor();
  await inStage(page, "training").getByText("[ERROR]").waitFor();
  await shot(page, "training-failed");
});

test("training: cancelled run", async (page) => {
  await page.goto(`${APP}/?run=cancelled`);
  await inStage(page, "training").getByText("Run cancelled").waitFor();
});

test("training: queued run", async (page) => {
  await page.goto(`${APP}/?run=queued`);
  await inStage(page, "training").getByText("Waiting for a GPU container").waitFor();
});

test("training: the elapsed clock counts up a second at a time, never back", async (page, x) => {
  await page.goto(`${APP}/?run=running&stage=training`);
  const t = inStage(page, "training");
  await t.getByText("NVIDIA H200", { exact: true }).waitFor();
  x.ok(!(await t.getByText("requested").count()), "no 'requested' next to the GPU");
  const clock = t.locator("dt", { hasText: "Elapsed" }).locator("xpath=following-sibling::dd");
  const seconds = (text) => {
    const m = text.trim().match(/^(?:(\d+)h )?(?:(\d+)m )?(\d+)s$/);
    return m ? Number(m[1] ?? 0) * 3600 + Number(m[2] ?? 0) * 60 + Number(m[3]) : NaN;
  };
  for (let i = 0; i < 40 && Number.isNaN(seconds(await clock.innerText().catch(() => ""))); i++) await page.waitForTimeout(100);
  // Ten seconds spans three diagnostics polls, where the old clock jumped.
  const seen = [];
  for (const end = Date.now() + 10000; Date.now() < end; ) {
    seen.push(seconds(await clock.innerText()));
    await page.waitForTimeout(150);
  }
  const steps = seen.slice(1).map((v, i) => v - seen[i]);
  x.ok(seen.every(Number.isFinite), `the clock reads as a time: ${seen}`);
  x.ok(steps.every((d) => d === 0 || d === 1), `it moves a second at a time: ${seen.join(" ")}`);
  const gained = seen.at(-1) - seen[0];
  x.ok(gained >= 8 && gained <= 11, `it counted ${gained}s in 10s`);
});

// ---------- results
test("results: full regression report, tuning, dialogs, downloads", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  const r = inStage(page, "results");
  await r.getByText("GradientBoostingRegressor").first().waitFor();
  await r.getByText("87.35%").first().waitFor();
  await r.getByText("smoker_yes").waitFor();
  await r.getByRole("button", { name: "Tuning" }).click();
  await r.getByText("New champion").waitFor();
  x.ok(await r.getByRole("link", { name: "Cleaned CSV" }).count(), "cleaned csv link");
  x.ok(await r.getByRole("link", { name: "Model + weights" }).count(), "model link");
  await r.getByRole("button", { name: "Read the full rationale" }).click();
  await page.getByRole("dialog").getByText("Why GradientBoostingRegressor").waitFor();
  await page.keyboard.press("Escape");
  await r.getByRole("button", { name: "Zoom the plot" }).click();
  await page.getByRole("dialog").getByRole("img").waitFor();
  await page.keyboard.press("Escape");
  await shot(page, "results-reg");
});

test("results: bare result with no extras and no plot", async (page, x) => {
  await page.goto(`${APP}/?run=bare-done&stage=results`);
  const r = inStage(page, "results");
  await r.getByText("Ridge").first().waitFor();
  await r.getByText("Not reported").waitFor();
  await r.getByText("Single model").waitFor();
  await r.getByText("No plot for this run").waitFor();
  x.ok(!(await r.getByRole("button", { name: "Tuning" }).count()), "no tuning tab");
  await shot(page, "results-bare");
}, { allow: [{ status: 404, url: "/plot" }] });

test("results: what the analyst found opens in a dialog", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await inStage(page, "results").getByRole("button", { name: "What the analyst found" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByText("One insured person.").waitFor();
  await dialog.getByText("Trained on a random sample of 1,000 of 1,338 rows").waitFor();
  x.ok(await dialog.getByRole("cell", { name: "bmi", exact: true }).count(), "the data dictionary lists bmi");
  x.ok(await dialog.getByText("From the name").count(), "a meaning that rests on the name says so");
  await dialog.getByText("Treated children as categories").waitFor();
  await dialog.getByText("Are the charges yearly or lifetime totals?").waitFor();
  await shot(page, "analyst-dialog");
  await page.keyboard.press("Escape");
  await dialog.waitFor({ state: "hidden" });

  await page.goto(`${APP}/?run=bare-done&stage=results`);
  await inStage(page, "results").getByText("Ridge").first().waitFor();
  x.ok(!(await page.getByRole("button", { name: "What the analyst found" }).count()), "no analysis, no link");
}, { allow: [{ status: 404, url: "/plot" }] });

test("results: classification overfit warning", async (page) => {
  await page.goto(`${APP}/?run=clf-done&stage=results`);
  await inStage(page, "results").getByText("Overfit gap 18%").waitFor();
});

test("results: export PNG and PDF", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  const r = inStage(page, "results");
  await r.getByText("GradientBoostingRegressor").first().waitFor();
  // Measure the off-screen report as the export draws it: card order, text
  // cut off, and space left empty at the bottom of each card.
  await page.evaluate(() => {
    window.__exportLayouts = [];
    new MutationObserver((records) => {
      for (const node of records.flatMap((rec) => [...rec.addedNodes])) {
        const clone = node.querySelector?.(".__export-clone");
        if (!clone) continue;
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            const box = (el) => el.getBoundingClientRect();
            const cards = [...clone.querySelectorAll("[data-export-grid] > *")].map((card) => {
              const body = card.lastElementChild;
              const content = Math.max(...[...body.children].map((c) => box(c).bottom));
              return { title: card.querySelector("h2")?.textContent, spare: Math.round(box(body).bottom - content) };
            });
            const clipped = [...clone.querySelectorAll("*")]
              .filter((el) => [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()))
              .filter((el) => el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflowX !== "visible")
              .map((el) => el.textContent.trim().slice(0, 40));
            window.__exportLayouts.push({ header: clone.querySelector(":scope > header")?.innerText ?? "", cards, clipped });
          })
        );
      }
    }).observe(document.body, { childList: true });
  });

  const [png] = await Promise.all([page.waitForEvent("download", { timeout: 60000 }), r.getByRole("button", { name: "Export PNG" }).click()]);
  const pngPath = path.join(OUT, "export.png");
  await png.saveAs(pngPath);
  x.ok((await toastText(page)).join(" ").includes("Report saved as PNG"), "png toast");
  const head = fs.readFileSync(pngPath);
  const [width, height] = [head.readUInt32BE(16), head.readUInt32BE(20)];
  x.ok(width === 1920 && height > 2000 && height < 8000, `png is ${width}x${height}`);
  const [layout] = await page.evaluate(() => window.__exportLayouts);
  x.ok(layout, "the export drew a report");
  x.ok(["Namtheg", "Model report", "insurance.csv", "predicting charges", "reg-done", "Exported"].every((t) => layout.header.includes(t)), `header: ${layout.header}`);
  const titles = layout.cards.map((c) => c.title);
  x.ok(JSON.stringify(titles) === JSON.stringify(["Champion model", "R²", "Predicted vs actual", "Feature importance", "Model selection"]), `cards: ${titles}`);
  x.ok(!layout.clipped.length, `text cut off: ${JSON.stringify(layout.clipped)}`);
  const blank = layout.cards.filter((c) => c.spare > 120);
  x.ok(!blank.length, `empty space at the bottom of ${JSON.stringify(blank)}`);

  const [pdf] = await Promise.all([page.waitForEvent("download", { timeout: 60000 }), r.getByRole("button", { name: "Export PDF" }).click()]);
  const pdfPath = path.join(OUT, "export.pdf");
  await pdf.saveAs(pdfPath);
  const doc = fs.readFileSync(pdfPath, "latin1");
  const pages = doc.match(/\/Type \/Page\b(?!s)/g)?.length ?? 0;
  // A4 in points; jsPDF writes them as 595.2799999999999727 and 841.8899999999999864.
  const a4 = doc.match(/\/MediaBox \[0 0 595\.2(?:8|79+\d*) 841\.8(?:9|89+\d*)\]/g)?.length ?? 0;
  x.ok(pages >= 2 && pages <= 4 && a4 === pages, `pdf: ${pages} pages, ${a4} of them A4`);
  x.ok(doc.includes("/Title (Namtheg model report: GradientBoostingRegressor for charges)"), "pdf title");
  x.ok(fs.statSync(pdfPath).size > 50000, "pdf has content");
});

test("results: results that never arrive offer a retry", async (page, x) => {
  await page.goto(`${APP}/?run=noresult-done&stage=results`);
  const r = inStage(page, "results");
  await r.getByRole("status").first().waitFor();
  await r.getByText("The results didn't load").waitFor({ timeout: 75000 });
  x.ok(await r.getByRole("button", { name: "Try again" }).count(), "retry offered");
  await shot(page, "results-error");
}, { allow: [{ status: 409 }] , slow: true });

// ---------- deploy and predict
test("deploy: success, toast, share, try", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=deploy`);
  const d = inStage(page, "deploy");
  await d.getByRole("button", { name: "Deploy to the cloud" }).click();
  await d.getByRole("button", { name: "Deploying..." }).waitFor();
  await d.getByRole("button", { name: "Predict", exact: true }).waitFor({ timeout: 15000 });
  x.ok((await toastText(page)).join(" ").includes("The model is live"), "live toast");
  await d.getByRole("button", { name: "Copy share link" }).click();
  await page.locator('[data-slot="toast"]').filter({ hasText: "Share link copied" }).waitFor();
  await shot(page, "deploy-live");
});

test("deploy: failure then retry", async (page, x) => {
  await page.goto(`${APP}/?run=deployfail-done&stage=deploy`);
  const d = inStage(page, "deploy");
  await d.getByRole("button", { name: "Deploy to the cloud" }).click();
  await d.getByText("Deployment failed").first().waitFor({ timeout: 15000 });
  x.ok((await toastText(page)).join(" ").includes("Deployment failed"), "failure toast");
  await d.getByRole("button", { name: "Retry deployment" }).click();
  await d.getByRole("button", { name: "Predict", exact: true }).waitFor({ timeout: 15000 });
});

test("deploy: model details that fail offer a retry", async (page) => {
  await page.goto(`${APP}/?run=schemaerr-done&stage=deploy`);
  await inStage(page, "deploy").getByText("The model details didn't load").waitFor();
  await inStage(page, "deploy").getByRole("button", { name: "Try again" }).waitFor();
}, { allow: [{ status: 409, url: "/model_schema" }] });

test("predict: classification with probabilities", async (page, x) => {
  await page.goto(`${APP}/?run=liveclf-done&stage=deploy`);
  const d = inStage(page, "deploy");
  await d.getByRole("button", { name: "Predict", exact: true }).click();
  await d.getByText("88.0% confidence").waitFor();
  x.ok((await d.getByText("12.00%").count()) > 0, "class probabilities listed");
});

test("predict: regression, JSON batch, invalid JSON", async (page, x) => {
  await page.goto(`${APP}/?run=live-done&stage=deploy`);
  const d = inStage(page, "deploy");
  await d.locator("#feature-age").fill("45");
  await d.getByRole("button", { name: "Predict", exact: true }).click();
  await d.getByText("16884.9241").waitFor();
  await d.getByRole("button", { name: "JSON" }).click();
  await d.getByLabel("Request body").fill('{"rows": [[19, "female", 27.9, 0, "yes", "southwest"], [18, "male", 33.8, 1, "no", "southeast"]]}');
  await d.getByRole("button", { name: "Predict", exact: true }).click();
  await d.getByText("17616.1241").waitFor();
  await d.getByLabel("Request body").fill("{not json");
  await d.getByRole("button", { name: "Predict", exact: true }).click();
  await d.getByRole("alert").filter({ hasText: "Invalid JSON" }).waitFor();
});

test("predict: API error is readable", async (page, x) => {
  await page.goto(`${APP}/?run=predicterr-done&stage=deploy`);
  const d = inStage(page, "deploy");
  await d.getByRole("button", { name: "Predict", exact: true }).click();
  const msg = await d.getByRole("alert").innerText();
  x.ok(msg.includes("Could not reach Modal endpoint") && !msg.includes("{"), `readable error, got ${msg}`);
}, { allow: [{ status: 502, url: "/predict" }] });

// ---------- share page
test("share: live model predicts", async (page) => {
  await page.goto(`${APP}/share/live-done`);
  await page.getByRole("button", { name: "Predict", exact: true }).click();
  await page.getByText("16884.9241").waitFor();
  await shot(page, "share-live");
});

test("share: not live", async (page) => {
  await page.goto(`${APP}/share/reg-done`);
  await page.getByText("Not live right now").waitFor();
});

test("share: load error offers a retry", async (page) => {
  await page.goto(`${APP}/share/nope`);
  await page.getByText("This model isn't available").waitFor();
  await page.getByRole("button", { name: "Try again" }).waitFor();
}, { allow: [{ status: 404 }] });

// ---------- routing
test("routing: legacy redirects and unknown stage 404", async (page, x) => {
  await page.goto(`${APP}/reg-done/result`);
  await page.waitForURL(/stage=results/);
  x.ok((await activeStage(page)) === "stage-results", "results active");
  const res = await page.goto(`${APP}/reg-done/bogus`);
  x.ok(res.status() === 404, `bogus stage should 404, got ${res.status()}`);
}, { allow: [{ status: 404, url: "/bogus" }] });

test("routing: deep links land on the right stage", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=deploy`);
  await settle(page);
  x.ok((await activeStage(page)) === "stage-deploy", "deploy");
  await page.goto(`${APP}/?run=reg-done`);
  await page.waitForURL(/stage=results/);
  x.ok((await activeStage(page)) === "stage-results", "no stage lands on results for a finished run");
  await page.goto(`${APP}/?run=running`);
  await page.waitForURL(/stage=training/);
});

test("routing: reload keeps run and stage", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await settle(page);
  await pager(page, "Training").click();
  await settle(page);
  await page.reload();
  await settle(page, 1200);
  x.ok((await activeStage(page)) === "stage-training" && page.url().includes("run=reg-done"), `after reload: ${page.url()}`);
});

test("routing: back and forward step through stages without reloading", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await settle(page);
  await page.evaluate(() => (window.__marker = 1));
  await pager(page, "Training").click();
  await settle(page);
  await pager(page, "Data").click();
  await settle(page);
  await page.goBack();
  await settle(page);
  x.ok((await activeStage(page)) === "stage-training", `back 1: ${await activeStage(page)}`);
  await page.goBack();
  await settle(page);
  x.ok((await activeStage(page)) === "stage-results", `back 2: ${await activeStage(page)}`);
  await page.goForward();
  await settle(page);
  x.ok((await activeStage(page)) === "stage-training", `forward: ${await activeStage(page)}`);
  x.ok((await page.evaluate(() => window.__marker)) === 1, "must not reload");
  await page.getByRole("button", { name: "New run" }).click();
  await settle(page);
  x.ok(!page.url().includes("run="), "new run clears the url");
  await page.goBack();
  await settle(page, 1200);
  x.ok(page.url().includes("run=reg-done"), "back after new run returns to the run");
  await inStage(page, "training").getByText("Complete").first().waitFor();
  x.ok((await page.evaluate(() => window.__marker)) === 1, "still no reload");
});

test("routing: unknown run", async (page) => {
  await page.goto(`${APP}/?run=nope&stage=data`);
  await inStage(page, "data").getByText("Run not found").waitFor();
  await inStage(page, "data").getByRole("button", { name: "Start a new run" }).click();
  await page.waitForURL((u) => !u.href.includes("run="));
}, { allow: [{ status: 404 }] });

// ---------- navigation
test("nav: pager, arrow keys, and where arrows are ignored", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=data`);
  await settle(page);
  for (const [label, id] of [["Training", "stage-training"], ["Results", "stage-results"], ["Deploy", "stage-deploy"], ["Data", "stage-data"]]) {
    await pager(page, label).click();
    await settle(page);
    x.ok((await activeStage(page)) === id, `pager ${label}`);
  }
  await page.locator("body").click({ position: { x: 700, y: 30 } });
  await page.keyboard.press("ArrowRight");
  await settle(page);
  x.ok((await activeStage(page)) === "stage-training", "arrow right pages");
  await page.keyboard.press("ArrowLeft");
  await settle(page);
  x.ok((await activeStage(page)) === "stage-data", "arrow left pages");
  await inStage(page, "data").getByLabel("Filter rows").click();
  await page.keyboard.press("ArrowRight");
  await settle(page);
  x.ok((await activeStage(page)) === "stage-data", "ignored while typing");
  await inStage(page, "data").getByRole("button", { name: "Raw rows" }).focus();
  await page.keyboard.press("ArrowRight");
  await settle(page);
  x.ok((await activeStage(page)) === "stage-data", "ignored inside a toggle group");
  const inert = await page.evaluate(() => [...document.querySelectorAll("main > div > section")].map((s) => s.inert));
  x.ok(inert.filter(Boolean).length === 4 && !inert[1], `inert pattern ${inert}`);
});

test("nav: arrow keys are ignored inside the target combobox", async (page, x) => {
  await page.goto(`${APP}/?run=fresh&stage=data`);
  await settle(page);
  await page.locator("#target-column").click();
  await page.locator("#target-column").fill("s");
  await page.getByRole("listbox").waitFor();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowLeft");
  await settle(page);
  x.ok((await activeStage(page)) === "stage-data", "combobox keeps its arrow keys");
  await page.keyboard.press("Escape");
});

test("nav: trackpad scroll settles on a stage", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=training`);
  await settle(page);
  await page.mouse.move(700, 450);
  await page.mouse.wheel(900, 0);
  await settle(page, 1500);
  x.ok((await activeStage(page)) === "stage-results", `wheel -> ${await activeStage(page)}`);
  const aligned = await page.evaluate(() => {
    const t = document.querySelector("main > div");
    return t.scrollLeft % t.clientWidth === 0;
  });
  x.ok(aligned, "track snapped");
});

test("nav: touch swipe settles on a stage", async (page, x, ctx) => {
  await page.goto(`${APP}/?run=reg-done&stage=training`);
  await settle(page);
  const cdp = await ctx.newCDPSession(page);
  const drag = async (from, to) => {
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [from] });
    for (let i = 1; i <= 18; i++) {
      await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: from.x + ((to.x - from.x) * i) / 18, y: from.y + ((to.y - from.y) * i) / 18 }] });
      await page.waitForTimeout(16);
    }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  };
  await drag({ x: 330, y: 300 }, { x: 60, y: 300 });
  await settle(page, 1500);
  x.ok((await activeStage(page)) === "stage-results", `swipe left -> ${await activeStage(page)}`);
  await drag({ x: 200, y: 600 }, { x: 200, y: 150 });
  await settle(page, 800);
  const top = await page.evaluate(() => document.querySelector("#stage-results").scrollTop);
  x.ok(top > 100, `vertical swipe scrolls the stage, scrollTop ${top}`);
  x.ok((await activeStage(page)) === "stage-results", "vertical swipe keeps the stage");
  await drag({ x: 60, y: 300 }, { x: 330, y: 300 });
  await settle(page, 1500);
  x.ok((await activeStage(page)) === "stage-training", `swipe right -> ${await activeStage(page)}`);
}, { viewport: { width: 390, height: 844 }, touch: true });

test("nav: resize keeps the current stage", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await settle(page);
  for (const v of [{ width: 1024, height: 768 }, { width: 390, height: 844 }, { width: 1920, height: 1080 }]) {
    await page.setViewportSize(v);
    await settle(page, 600);
    const ok = await page.evaluate(() => {
      const t = document.querySelector("main > div");
      return Math.round(t.scrollLeft / t.clientWidth) === 3 && t.scrollLeft % t.clientWidth === 0;
    });
    x.ok(ok && (await activeStage(page)) === "stage-results", `after resize to ${v.width}x${v.height}`);
  }
});

// ---------- accounts
const codeFor = async (email) => (await (await fetch(`${MOCK}/__mock/last_code?email=${encodeURIComponent(email)}`)).json()).code;
const toastWith = (page, text) => page.locator('[data-slot="toast"]').filter({ hasText: text }).first();
// The modal dialog only: toasts are dialogs too (non-modal ones), so getByRole("dialog") can match both.
const modal = (page) => page.locator('[data-slot="dialog-content"]');

test("account: create one with an emailed code, the open run is saved, sign out", async (page, x) => {
  await page.goto(DATA);
  await uploadCsv(page);
  const header = page.locator("header");
  await header.getByRole("button", { name: "Sign in" }).click();
  const dialog = modal(page);
  await dialog.getByRole("button", { name: "Create account" }).click();
  await dialog.getByLabel("Name", { exact: true }).fill("Ada Lovelace");
  await dialog.getByLabel("Email", { exact: true }).fill(" Ada@Example.com");
  await settle(page, 400);
  await shot(page, "account-create");
  await dialog.getByRole("button", { name: "Email me a code" }).click();
  await dialog.getByText("We emailed a 6-digit code to").waitFor();
  const resend = dialog.getByRole("button", { name: /^Resend in \d+ s$/ });
  x.ok((await resend.getAttribute("aria-disabled")) === "true" || (await resend.isDisabled()), "resend waits a minute");
  await shot(page, "account-code");
  await dialog.getByLabel("Code", { exact: true }).fill(await codeFor("ada@example.com"));
  await dialog.waitFor({ state: "hidden" });
  const account = header.getByRole("button", { name: "Account: Ada Lovelace" });
  await account.waitFor();
  await toastWith(page, "Signed in as Ada Lovelace").waitFor();

  await account.click();
  const list = modal(page).getByRole("list", { name: "My runs" });
  await list.getByText("Ready to train").waitFor();
  const focused = await page.evaluate(() => document.activeElement?.getAttribute("data-slot"));
  x.ok(focused === "dialog-content", `focus starts on the dialog, not on Sign out: ${focused}`);
  const items = await list.getByRole("listitem").allInnerTexts();
  x.ok(items.length === 1 && items[0].includes("insurance.csv"), `the run open at sign-in is saved: ${JSON.stringify(items)}`);
  await settle(page, 400);
  await shot(page, "account-runs");
  await modal(page).getByRole("button", { name: "Sign out" }).click();
  await header.getByRole("button", { name: "Sign in" }).waitFor();
  await toastWith(page, "Signed out").waitFor();
}, { allow: [{ status: 409, url: "/uploads/direct" }] });

test("account: sign in, a wrong code says so, a run opens where it got to", async (page, x) => {
  await mock("/__mock/account", { email: "grace@example.com", name: "Grace Hopper", runs: ["fresh", "failed", "reg-done"] });
  await page.goto(APP);
  const header = page.locator("header");
  await header.getByRole("button", { name: "Sign in" }).click();
  const dialog = modal(page);
  await dialog.getByLabel("Email", { exact: true }).fill("grace@example.com");
  await dialog.getByLabel("Email", { exact: true }).press("Enter");
  await dialog.getByText("If grace@example.com has an account").waitFor();
  const code = await codeFor("grace@example.com");
  const wrong = String((Number(code) + 1) % 1e6).padStart(6, "0");
  await dialog.getByLabel("Code", { exact: true }).fill(wrong);
  await dialog.getByRole("alert").filter({ hasText: "That code isn't right. 4 tries left." }).waitFor();
  x.ok((await dialog.getByLabel("Code", { exact: true }).getAttribute("aria-invalid")) === "true", "the code field is marked invalid");
  await shot(page, "account-wrong-code");
  await dialog.getByLabel("Code", { exact: true }).fill(code);
  await dialog.waitFor({ state: "hidden" });

  await header.getByRole("button", { name: "Account: Grace Hopper" }).click();
  const list = modal(page).getByRole("list", { name: "My runs" });
  await list.getByText("GradientBoostingRegressor").waitFor();
  const items = await list.getByRole("listitem").allInnerTexts();
  x.ok(
    items.length === 3 && /Trained/.test(items[0]) && /R² 87\.35%/.test(items[0]) && /Failed/.test(items[1]) && /Ready to train/.test(items[2]),
    `runs, newest first: ${JSON.stringify(items)}`
  );
  await settle(page, 400);
  await shot(page, "account-list");
  await list.getByRole("button", { name: /GradientBoostingRegressor/ }).click();
  await modal(page).waitFor({ state: "hidden" });
  await inStage(page, "results").getByText("GradientBoostingRegressor").first().waitFor();
  x.ok(/run=reg-done/.test(page.url()) && (await stage(page)) === "results", `opens on results: ${page.url()}`);
}, { allow: [{ status: 400, url: "/auth/verify" }] });

test("account: an address without an account is offered one, and nothing is sent", async (page, x) => {
  await page.goto(APP);
  await page.locator("header").getByRole("button", { name: "Sign in" }).click();
  const dialog = modal(page);
  await dialog.getByLabel("Email", { exact: true }).fill("new@example.com");
  await dialog.getByRole("button", { name: "Email me a code" }).click();
  await dialog.getByText("If new@example.com has an account").waitFor();
  x.ok((await codeFor("new@example.com")) === null, "no code for an address without an account");
  await dialog.getByRole("button", { name: "create an account" }).click();
  await dialog.getByRole("heading", { name: "Create an account" }).waitFor();
  x.ok((await dialog.getByLabel("Email", { exact: true }).inputValue()) === "new@example.com", "the address is kept");
  x.ok((await page.evaluate(() => document.activeElement?.getAttribute("autocomplete"))) === "name", "focus moves to the name");
  await dialog.getByRole("button", { name: "Email me a code" }).click();
  await dialog.getByRole("alert").filter({ hasText: "Enter your name." }).waitFor();
});

test("account: no sign-in when the server has no accounts", async (page, x) => {
  await mock("/__mock/config", { auth: false });
  await page.goto(APP);
  await page.locator("header").getByRole("button", { name: "New run" }).waitFor();
  await settle(page);
  x.ok(!(await page.locator("header").getByRole("button", { name: "Sign in" }).count()), "no sign-in button");
});

// ---------- UX states
test("ux: skeletons while a linked run loads", async (page, x) => {
  await mock("/__mock/config", { delay: 2 });
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await page.waitForTimeout(700);
  const skeletons = await inStage(page, "results").locator('[data-slot="skeleton"]').count();
  x.ok(skeletons > 5, `skeletons on first load: ${skeletons}`);
  await shot(page, "ux-skeleton");
  await inStage(page, "results").getByText("GradientBoostingRegressor").first().waitFor({ timeout: 20000 });
});

test("ux: reduced motion pages instantly", async (page, x) => {
  await page.goto(`${APP}/?run=reg-done&stage=data`);
  await settle(page);
  await pager(page, "Deploy").click();
  await page.waitForTimeout(60);
  const left = await page.evaluate(() => {
    const t = document.querySelector("main > div");
    return t.scrollLeft / t.clientWidth;
  });
  x.ok(left === 4, `instant jump expected, at ${left}`);
}, { reducedMotion: "reduce" });

test("ux: reduced transparency makes surfaces nearly opaque", async (page, x, ctx) => {
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-transparency", value: "reduce" }] });
  await page.goto(`${APP}/?run=reg-done&stage=results`);
  await settle(page);
  const [card, wash] = await page.evaluate(() => [
    getComputedStyle(document.documentElement).getPropertyValue("--card").trim(),
    getComputedStyle(document.body, "::before").opacity,
  ]);
  x.ok(/\.97\)$/.test(card), `card token ${card}`);
  x.ok(wash === "0.35", `wash opacity ${wash}`);
});

run(process.argv[2]);
