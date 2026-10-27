import type { Metadata } from "next";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/toast";

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
    // Dark is the only theme: the class is permanent and nothing toggles it
    // (see the .dark block at the end of globals.css).
    <html lang="en" dir="ltr" className="dark">
      <body className="antialiased">
        <Toaster>
          <TooltipProvider>{children}</TooltipProvider>
        </Toaster>
      </body>
    </html>
  );
}
