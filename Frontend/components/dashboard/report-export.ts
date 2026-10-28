/**
 * PNG and PDF export of the results report.
 *
 * The report is cloned into an off-screen sandbox and laid out there as a
 * document rather than as the dashboard: a header with the run's details that
 * only the export shows, the champion beside its score, the plot beside the
 * feature importances, and model selection across the full width with both
 * the leaderboard and the tuning. Each card's place comes from its
 * data-export-span. Text the dashboard truncates to fit is shown in full. The
 * clone sits outside the stage page, so the page-locked heights (the `lock`
 * variant, a container query on the page) do not apply and every card comes
 * out at its full height.
 *
 * html-to-image rasterises the clone. The PNG is that image; the PDF cuts it
 * into A4 pages between rows of cards, or between the rows of a card taller
 * than a page.
 */

// 960 CSS px: a 1920 px wide PNG, and on an A4 page the smallest text still
// comes out at about 8 pt.
const CAPTURE_WIDTH = 960;
const CAPTURE_PADDING = 24;
// The space between rows of cards (gap-3 on the grid, mb-3 under the header).
const ROW_GAP = 12;
const PIXEL_RATIO = 2;
// Below the largest canvas iOS Safari will draw, so a long report still exports.
const MAX_CANVAS_AREA = 16_000_000;
// The page background, as a colour a canvas and a PDF can fill: oklch(0.145 0.032 263).
// Flat rather than the dashboard's wash, so PDF pages meet their margins without a seam.
const EXPORT_BACKGROUND = "#040a17";

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

// Safari and every iOS browser: WebKit paints the SVG html-to-image builds
// before that SVG's fonts and pictures have loaded the first time, which
// leaves text in fallback fonts and the plot blank. A second pass is whole.
const isAppleWebKit = () => /AppleWebKit/i.test(navigator.userAgent) && !/Chrome\//i.test(navigator.userAgent);

/** One band of the report: the header, or a row of cards side by side. */
interface Row {
  top: number;
  bottom: number;
  /** Where a page can end inside the row: between its list items, table rows or sections. Only a row of one card has any. */
  breaks: number[];
}

interface Snapshot {
  dataUrl: string;
  width: number;
  height: number;
  rows: Row[];
}

async function captureReport(node: HTMLElement): Promise<Snapshot> {
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
    padding: `${CAPTURE_PADDING}px`,
    background: EXPORT_BACKGROUND,
  });
  clone.querySelectorAll("[data-export-ignore]").forEach((el) => el.remove());
  // The header and the views that sit behind a switch on screen (leaderboard and tuning) all print.
  clone.querySelectorAll("[data-export-show]").forEach((el) => el.removeAttribute("hidden"));
  clone.querySelectorAll("[data-export-date]").forEach((el) => {
    el.textContent = new Date().toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
  });
  // The report's own grid: twelve columns, each card spanning what its data-export-span says.
  const grid = clone.querySelector<HTMLElement>("[data-export-grid]");
  if (grid) {
    Object.assign(grid.style, { gridTemplateColumns: "repeat(12, minmax(0, 1fr))", gridTemplateRows: "none", height: "auto" });
    grid.querySelectorAll<HTMLElement>("[data-export-span]").forEach((el) => {
      const span = el.dataset.exportSpan;
      el.style.gridColumn = `span ${span} / span ${span}`;
    });
  }

  // Scroll areas open up to their full content, clamps and truncation
  // release, and no scrollbar gutter is left behind in the image. Backdrop
  // blur goes too: an image has nothing behind it to blur, and inside
  // html-to-image's SVG it clips each card to a rectangle offset from where
  // the card is drawn.
  const reset = document.createElement("style");
  reset.textContent = `
    .__export-clone, .__export-clone * { -webkit-backdrop-filter: none !important; backdrop-filter: none !important; }
    .__export-clone .overflow-auto, .__export-clone .overflow-y-auto, .__export-clone .overflow-x-auto,
    .__export-clone [data-slot="table-container"] {
      overflow: visible !important; max-height: none !important; scrollbar-width: none !important;
    }
    .__export-clone [class*="line-clamp"] { -webkit-line-clamp: unset !important; display: block !important; }
    .__export-clone .truncate { overflow: visible !important; text-overflow: clip !important; white-space: normal !important; }
    .__export-clone td, .__export-clone th { max-width: none !important; }
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

    const origin = clone.getBoundingClientRect().top;
    const height = Math.ceil(Math.max(clone.scrollHeight, clone.offsetHeight, clone.getBoundingClientRect().height));
    const rows = measureRows(clone, origin);
    const ratio = Math.min(PIXEL_RATIO, Math.sqrt(MAX_CANVAS_AREA / (CAPTURE_WIDTH * height)));

    const { toPng } = await import("html-to-image");
    const options = {
      backgroundColor: EXPORT_BACKGROUND,
      pixelRatio: ratio,
      cacheBust: true,
      width: CAPTURE_WIDTH,
      height,
      fontEmbedCSS: fontCSS,
    };
    if (isAppleWebKit()) await toPng(clone, options);
    const dataUrl = await toPng(clone, options);
    return { dataUrl, width: CAPTURE_WIDTH, height, rows };
  } finally {
    document.body.removeChild(sandbox);
  }
}

/** The report's bands, top to bottom, in CSS px from the top of the clone. */
function measureRows(clone: HTMLElement, origin: number): Row[] {
  const y = (el: Element) => Math.round(el.getBoundingClientRect().top - origin);
  const blocks = [clone.querySelector(":scope > header"), ...(clone.querySelector("[data-export-grid]")?.children ?? [])].filter(
    (el): el is Element => !!el && el.getBoundingClientRect().height > 0
  );
  const bands = new Map<number, { bottom: number; cards: Element[] }>();
  for (const el of blocks) {
    const top = y(el);
    const band = bands.get(top) ?? { bottom: top, cards: [] };
    band.bottom = Math.max(band.bottom, Math.round(el.getBoundingClientRect().bottom - origin));
    band.cards.push(el);
    bands.set(top, band);
  }
  return [...bands.entries()]
    .sort(([a], [b]) => a - b)
    .map(([top, { bottom, cards }]) => ({
      top,
      bottom,
      breaks:
        cards.length === 1
          ? [...new Set(Array.from(cards[0].querySelectorAll("li, tr, section, h3"), y))].filter((b) => b > top && b < bottom).sort((a, b) => a - b)
          : [],
    }));
}

/**
 * Where each page starts and ends, in CSS px. A page ends in the gap above
 * the last row of cards that starts on it, so cards are not cut, unless the
 * row running past its end is taller than any page: that one is cut between
 * its own rows. Nothing is cut in the first quarter of a page, so no page is a
 * sliver.
 */
function paginate(height: number, pageHeight: number, rows: Row[]): [number, number][] {
  const pages: [number, number][] = [];
  let top = 0;
  while (height - top > pageHeight) {
    const limit = top + pageHeight;
    const floor = top + pageHeight / 4;
    // Mid-gap, so the edge a card draws just outside itself stays with it.
    const gaps = rows.map((r) => r.top - ROW_GAP / 2);
    let cut = Math.max(0, ...gaps.filter((g) => g > floor && g <= limit));
    const across = rows.find((r) => r.top < limit && r.bottom > limit);
    if (across && (!cut || across.bottom - across.top > pageHeight)) {
      const inside = across.breaks.filter((b) => b > floor && b <= limit);
      if (inside.length) cut = Math.max(...inside);
    }
    if (!cut) cut = limit;
    pages.push([top, cut]);
    top = cut;
  }
  pages.push([top, height]);
  return pages;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("snapshot failed to decode"));
    img.src = src;
  });
}

export async function exportReportPng(node: HTMLElement, filename: string) {
  const snap = await captureReport(node);
  const link = document.createElement("a");
  link.href = snap.dataUrl;
  link.download = filename;
  link.click();
}

// A4 portrait, in points.
const PAGE_WIDTH = 595.28;
const PAGE_HEIGHT = 841.89;
const FOOTER = 26;

export async function exportReportPdf(node: HTMLElement, filename: string, title: string) {
  const snap = await captureReport(node);
  const img = await loadImage(snap.dataUrl);
  // Image pixels per CSS pixel, as drawn: html-to-image scales a very tall
  // report down further than the pixel ratio it was given.
  const ratio = img.naturalWidth / snap.width;

  // The report spans the page width; its own padding is the side margin, and
  // pages after the first get the same margin on top.
  const scale = PAGE_WIDTH / snap.width;
  const margin = CAPTURE_PADDING * scale;
  const pages = paginate(snap.height, (PAGE_HEIGHT - margin - FOOTER) / scale, snap.rows);

  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4", compress: true });
  pdf.setProperties({ title, creator: "Namtheg" });
  pages.forEach(([top, bottom], i) => {
    // JPEG at 92%: PNG is lossless and would make a long report's PDF tens of
    // megabytes; this looks the same and is an order of magnitude smaller.
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = Math.round((bottom - top) * ratio);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    ctx.fillStyle = EXPORT_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, Math.round(top * ratio), canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);

    if (i > 0) pdf.addPage("a4", "portrait");
    pdf.setFillColor(EXPORT_BACKGROUND);
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, "F");
    pdf.addImage(canvas.toDataURL("image/jpeg", 0.92), "JPEG", 0, i === 0 ? 0 : margin, PAGE_WIDTH, (bottom - top) * scale);
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(7.5);
    pdf.setTextColor(139, 149, 173);
    pdf.text("Namtheg model report", margin, PAGE_HEIGHT - 12);
    pdf.text(`Page ${i + 1} of ${pages.length}`, PAGE_WIDTH - margin, PAGE_HEIGHT - 12, { align: "right" });
  });
  pdf.save(filename);
}
