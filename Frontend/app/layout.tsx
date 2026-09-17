import type { Metadata } from "next";
import "./globals.css";
import { LayoutShell } from "@/components/layout-shell";

// Canonical/OG base URL: og:image and og:url are resolved against it, so it must be
// a host that actually serves this app, or link previews lose their image.
const FALLBACK_SITE_URL = "https://namtheg.khalid-ai.dev";
const rawSiteUrl = process.env.NEXT_PUBLIC_SITE_URL || FALLBACK_SITE_URL;
const SITE_URL = (
  /^https?:\/\//i.test(rawSiteUrl) ? rawSiteUrl : `https://${rawSiteUrl}`
).replace(/\/+$/, "");

function getMetadataBase(): URL {
  try {
    return new URL(SITE_URL);
  } catch {
    return new URL(FALLBACK_SITE_URL);
  }
}

const TITLE = "Namtheg AutoML";
const DESCRIPTION =
  "Upload a CSV, pick a target column, and Namtheg trains and compares ML models, then uses an LLM to pick the best one and explain why in plain language.";

export const metadata: Metadata = {
  metadataBase: getMetadataBase(),
  title: TITLE,
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    url: SITE_URL,
    title: TITLE,
    description: DESCRIPTION,
    siteName: "Namtheg",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
          crossOrigin="anonymous"
        />
        {/* Font Awesome 6.5.1 — the only icon set (design-system hard rule) */}
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
          integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA=="
          crossOrigin="anonymous"
          referrerPolicy="no-referrer"
        />
      </head>
      <body className="bg-surface text-on-surface antialiased">
        <LayoutShell>{children}</LayoutShell>
      </body>
    </html>
  );
}
