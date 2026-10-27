/**
 * PNG and PDF export of the results report.
 *
 * The report is cloned into an off-screen sandbox 1280px wide and rasterised
 * with html-to-image. Its grid is laid out with container queries on the
 * report itself (@container/report), so the clone gets the wide layout at any
 * viewport width, and the page-locked heights (the `lock` variant, a container
 * query on the stage page) do not apply outside the page, so every card comes
 * out at full height.
 */

const CAPTURE_WIDTH = 1280;
// The page background, as a colour a canvas can fill: oklch(0.145 0.032 263).
const EXPORT_BACKGROUND = "#040a17";
// The dark wash from globals.css, painted onto the clone since the fixed
// body::before layer is not part of it.
const EXPORT_WASH = [
  "radial-gradient(50rem 34rem at 6% -4%, oklch(0.55 0.14 250 / 0.34), transparent 60%)",
  "radial-gradient(44rem 32rem at 100% 4%, oklch(0.62 0.12 232 / 0.2), transparent 62%)",
  "radial-gradient(46rem 40rem at 88% 100%, oklch(0.45 0.16 276 / 0.3), transparent 62%)",
].join(", ");

const FONT_BASE = "https://cdn.jsdelivr.net/gh/khaliddosari/thmanyah-fonts@v1/thmanyah%20typeface/";
const EXPORT_FONTS: [family: string, weight: number, file: string][] = [
  ["Thmanyah Sans", 400, "thmanyahsans/woff2/thmanyahsans-Regular.woff2"],
  ["Thmanyah Sans", 500, "thmanyahsans/woff2/thmanyahsans-Medium.woff2"],
  ["Thmanyah Sans", 700, "thmanyahsans/woff2/thmanyahsans-Bold.woff2"],
  ["Thmanyah Serif Display", 900, "thmanyahserifdisplay/woff2/thmanyahserifdisplay-Black.woff2"],
];

// The inlined fonts are large and identical every time, so they are built once.
let fontEmbedCSS: string | null = null;

async function fetchAsDataUrl(url: string): Promise<string | null> {
  try {
    const res = await fetch(url, { mode: "cors" });
    if (!res.ok) return null;
    const blob = await res.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

async function getFontEmbedCSS(): Promise<string> {
  if (fontEmbedCSS !== null) return fontEmbedCSS;
  const faces = await Promise.all(
    EXPORT_FONTS.map(async ([family, weight, file]) => {
      const data = await fetchAsDataUrl(FONT_BASE + file);
      return data
        ? `@font-face{font-family:"${family}";src:url(${data}) format("woff2");font-weight:${weight};font-style:normal;}`
        : "";
    })
  );
  fontEmbedCSS = faces.join("\n");
  return fontEmbedCSS;
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve(null)));

async function captureReport(node: HTMLElement): Promise<{ dataUrl: string; width: number; height: number }> {
  const sandbox = document.createElement("div");
  Object.assign(sandbox.style, {
    position: "fixed",
    left: "-100000px",
    top: "0",
    width: `${CAPTURE_WIDTH}px`,
    zIndex: "-1",
    pointerEvents: "none",
  });

  const clone = node.cloneNode(true) as HTMLElement;
  clone.classList.add("__export-clone");
  Object.assign(clone.style, {
    width: `${CAPTURE_WIDTH}px`,
    maxWidth: `${CAPTURE_WIDTH}px`,
    height: "auto",
    padding: "24px",
    background: `${EXPORT_WASH}, ${EXPORT_BACKGROUND}`,
  });
  clone.querySelectorAll("[data-export-ignore]").forEach((el) => el.remove());
  // Views that sit behind a switch on screen (leaderboard and tuning) all print.
  clone.querySelectorAll("[data-export-show]").forEach((el) => el.removeAttribute("hidden"));
  // A card that fills the space an ignored card leaves behind.
  clone.querySelectorAll<HTMLElement>("[data-export-span]").forEach((el) => {
    const span = el.dataset.exportSpan;
    el.style.gridColumn = `span ${span} / span ${span}`;
  });

  // Scroll areas open up to their full content, clamps release, and no
  // scrollbar gutter is left behind in the image. Backdrop blur goes too: an
  // image has nothing behind it to blur, and inside html-to-image's SVG it
  // clips each card to a rectangle offset from where the card is drawn.
  const reset = document.createElement("style");
  reset.textContent = `
    .__export-clone, .__export-clone * { -webkit-backdrop-filter: none !important; backdrop-filter: none !important; }
    .__export-clone .overflow-auto, .__export-clone .overflow-y-auto, .__export-clone .overflow-x-auto,
    .__export-clone [data-slot="table-container"] {
      overflow: visible !important; max-height: none !important; scrollbar-width: none !important;
    }
    .__export-clone [class*="line-clamp"] { -webkit-line-clamp: unset !important; display: block !important; }
    .__export-clone .flex-col > .flex-1 { flex: 0 0 auto !important; }
    .__export-clone .flex-col > .min-h-0 { min-height: auto !important; }
    .__export-clone ::-webkit-scrollbar { display: none !important; }
  `;
  sandbox.appendChild(reset);
  sandbox.appendChild(clone);
  document.body.appendChild(sandbox);

  try {
    await nextFrame();
    await nextFrame();
    await Promise.all(
      Array.from(clone.querySelectorAll("img")).map((img) =>
        img.complete && img.naturalWidth > 0
          ? Promise.resolve()
          : new Promise<void>((resolve) => {
              img.addEventListener("load", () => resolve(), { once: true });
              img.addEventListener("error", () => resolve(), { once: true });
            })
      )
    );
    try {
      await document.fonts?.ready;
    } catch {
      /* font loading API unavailable; proceed */
    }
    const fontCSS = await getFontEmbedCSS();
    await nextFrame();

    const rect = clone.getBoundingClientRect();
    const height = Math.ceil(Math.max(clone.scrollHeight, clone.offsetHeight, rect.height));
    const { toPng } = await import("html-to-image");
    const dataUrl = await toPng(clone, {
      backgroundColor: EXPORT_BACKGROUND,
      pixelRatio: 2,
      cacheBust: true,
      width: CAPTURE_WIDTH,
      height,
      fontEmbedCSS: fontCSS,
    });
    return { dataUrl, width: CAPTURE_WIDTH, height };
  } finally {
    document.body.removeChild(sandbox);
  }
}

export async function exportReportPng(node: HTMLElement, filename: string) {
  const snap = await captureReport(node);
  const link = document.createElement("a");
  link.href = snap.dataUrl;
  link.download = filename;
  link.click();
}

export async function exportReportPdf(node: HTMLElement, filename: string) {
  const snap = await captureReport(node);

  // JPEG at 92%: PNG is lossless and would balloon a tall report's PDF past
  // 40 MB; this is visually the same and an order of magnitude smaller.
  const jpeg = await new Promise<string>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("2d context unavailable"));
      ctx.fillStyle = EXPORT_BACKGROUND;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL("image/jpeg", 0.92));
    };
    img.onerror = () => reject(new Error("snapshot failed to decode"));
    img.src = snap.dataUrl;
  });

  // One page exactly the size of the snapshot (1px = 0.75pt at 96dpi): no
  // pagination, no letterboxing.
  const { jsPDF } = await import("jspdf");
  const widthPt = snap.width * 0.75;
  const heightPt = snap.height * 0.75;
  const pdf = new jsPDF({
    orientation: widthPt > heightPt ? "landscape" : "portrait",
    unit: "pt",
    format: [widthPt, heightPt],
    compress: true,
  });
  pdf.addImage(jpeg, "JPEG", 0, 0, widthPt, heightPt);
  pdf.save(filename);
}
