import * as React from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { Brand } from "./brand";

const AUTHORS = [
  { first: "Khalid", full: "Khalid Al Dosari", href: "https://www.linkedin.com/in/khalid-al-dosari/" },
  { first: "Ahmed", full: "Ahmed Alasmari", href: "https://www.linkedin.com/in/ahmed-alasmari-sa/" },
];

function LinkedInGlyph({ className }: { className?: string }) {
  // Simple Icons "LinkedIn" path (CC0); lucide no longer ships brand icons.
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={cn("size-3.5 fill-current", className)}>
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 1 1 0-4.125 2.062 2.062 0 0 1 0 4.125zM7.119 20.452H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

function BrandLockup() {
  return (
    <span className="text-xl">
      <Brand />
    </span>
  );
}

/**
 * The one header: an optional action at the start, the authors centred, and the
 * logo at the end. Full names on desktop give way to first names on phones.
 */
export function SiteHeader({ start, onBrandClick }: { start?: React.ReactNode; onBrandClick?: () => void }) {
  const brandClass =
    "flex items-center gap-2.5 rounded-lg px-1 py-1 transition-colors hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 outline-none";

  return (
    <header className="relative z-20 grid h-14 shrink-0 grid-cols-[1fr_auto_1fr] items-center gap-2 border-b border-primary/10 bg-background/65 px-3 backdrop-blur-xl backdrop-saturate-150 sm:px-5">
      <div className="flex min-w-0 items-center justify-start">{start}</div>

      <nav aria-label="Authors" className="flex items-center gap-1.5 sm:gap-2">
        {AUTHORS.map((a, i) => (
          <React.Fragment key={a.href}>
            {i > 0 && (
              <span aria-hidden="true" className="text-muted-foreground">
                ·
              </span>
            )}
            <a
              href={a.href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-9 items-center gap-1.5 rounded-lg px-1.5 text-sm font-medium text-foreground/90 transition-colors outline-none hover:text-primary focus-visible:ring-3 focus-visible:ring-ring/50 lg:h-8"
            >
              <LinkedInGlyph className="text-primary" />
              <span className="sm:hidden">{a.first}</span>
              <span className="hidden sm:inline">{a.full}</span>
            </a>
          </React.Fragment>
        ))}
      </nav>

      <div className="flex min-w-0 items-center justify-end">
        {onBrandClick ? (
          <button type="button" onClick={onBrandClick} aria-label="Namtheg, back to the intro" className={brandClass}>
            <BrandLockup />
          </button>
        ) : (
          <Link href="/" aria-label="Namtheg home" className={brandClass}>
            <BrandLockup />
          </Link>
        )}
      </div>
    </header>
  );
}
