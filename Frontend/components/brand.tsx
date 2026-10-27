import React from "react";
import { cn } from "@/lib/utils";

/**
 * The product wordmark: نَمذِج (Arabic for "model"), set in Thmanyah Serif
 * Display Black via font-brand. The ornate OpenType features go in a style
 * rather than the font-ornate class, because cn() reads font-ornate as a font
 * family and would drop it next to font-brand.
 */
const BRAND_FEATURES: React.CSSProperties = {
  fontFeatureSettings: '"ss01", "ss03", "ss04", "ss05", "ss06", "ss07", "ss08", "salt", "swsh", "dlig"',
};

export function Brand({ className }: { className?: string }) {
  return (
    <span
      lang="ar"
      dir="rtl"
      style={BRAND_FEATURES}
      className={cn("inline-flex items-center font-brand text-[1.22em] leading-none font-black text-primary", className)}
    >
      نَمذِج
    </span>
  );
}

/** Three joined nodes: the brand mark, drawn in theme colours so it follows the hue. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={cn("size-5", className)} fill="none">
      <g className="stroke-primary" strokeWidth="2" strokeLinecap="round">
        <line x1="12" y1="6" x2="6" y2="18" />
        <line x1="12" y1="6" x2="18" y2="18" />
        <line x1="6" y1="18" x2="18" y2="18" />
      </g>
      <g className="fill-primary">
        <circle cx="12" cy="6" r="2.6" />
        <circle cx="6" cy="18" r="2.6" />
        <circle cx="18" cy="18" r="2.6" />
      </g>
    </svg>
  );
}

/**
 * Wraps any occurrence of "نَمذِج" in a string (the LLM writes it into
 * justifications) in the wordmark style so it reads as the product name.
 */
export function formatArabicBrand(text: string): React.ReactNode {
  if (!text) return "";
  const regex = /(ن[َْ]?م[َْ]?ذ[ِْ]?ج)/g;
  const parts = text.split(regex);
  return (
    <>
      {parts.map((part, index) =>
        index % 2 === 1 ? <Brand key={index} className="mx-0.5 text-[1.1em]" /> : part
      )}
    </>
  );
}
