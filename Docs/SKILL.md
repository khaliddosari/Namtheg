---
name: redesign-frontend
description: Workflow for redesigning a web frontend to be on-brand with the Liquid Glass design system (dark, frosted-glass surfaces over an ambient cyan/blue glow). Use when asked to redesign, restyle, or rebrand a site's frontend in this repo. Orchestrates the local liquid-glass-design skill: reads its tokens, then applies glass material, type, iconography, and motion to the target pages while preserving all existing markup IDs and JS behavior.
user-invocable: true
---

# Redesign a frontend with Liquid Glass

A repeatable workflow for taking any frontend in this repo and bringing it on-brand with
the **Liquid Glass** design system. Liquid Glass = a premium dark aesthetic: frosted,
translucent surfaces floating over a fixed ambient cyan/blue/violet glow, cyan→blue accent
used sparingly, IBM Plex Sans headings + JetBrains Mono tags + Inter body, Font Awesome icons,
calm `translateY` hover lifts. No light mode, no flat opaque blocks, no emoji.

## Inputs
- **Target**: which page(s)/stylesheet to redesign (default: everything under `static/`).
- **Design source**: the `liquid-glass-design` skill at
  `skills/Liquid Glass-Portfolio Design System/`, the single source of truth.

## Workflow

1. **Load the design system.** Read these from the design skill, in order:
   - `SKILL.md`: entry point and key-file map.
   - `README.md`: visual foundations + the **§6 Hard rules** (the non-negotiables).
   - `colors_and_type.css`: the full token set (`:root` variables + `@font-face`).
   - `ui_kits/portfolio/portfolio.css`: exact glass component patterns to copy
     (ambient field `body::before`, glass cards, buttons, tags, navbar, mobile drawer).

2. **Audit the target.** Read the target HTML/CSS/JS. List:
   - Every element id/class the JS reads or toggles (e.g. `navbar.scrolled`, `.open`,
     `.active`, `.placeholder-mode`, render-output classes). **These must survive the redesign.**
   - Every surface that should become glass (cards, panels, bars, controls).
   - Current type/icons/colors to replace.

3. **Port the tokens.** Bring the Liquid Glass `:root` token block into the target stylesheet
   (color, glass material, type families, spacing, radii, motion). Reuse the variable names
   verbatim so component CSS reads the same as the source kit.

4. **Add the ambient field.** Add the fixed, blurred `body::before` radial-glow layer. Without
   it the glass has nothing to refract and looks flat; never skip it.

5. **Convert surfaces to glass.** For each card/panel/bar: `background: var(--glass-bg)`,
   `backdrop-filter: var(--glass-blur)` (+ `-webkit-` prefix), `border: var(--glass-border)`,
   `box-shadow: var(--glass-highlight), var(--glass-shadow)`. Hover → `--glass-bg-hover`,
   `translateY(-2px/-4px)`, `--glass-shadow-hover`.

6. **Apply type & icons.** Headings/UI/buttons → `--font-heading` (IBM Plex Sans); tags/code →
   `--font-mono` (JetBrains Mono); body → `--font` (Inter). Load the web fonts in `<head>`.
   Keep Font Awesome 6.5.1; no emoji, no other icon library.

7. **Restyle accents.** Cyan→blue gradient only on CTAs, links, underlines, glows, tag text,
   never as large fills. Section titles centered with the 50px gradient underline.

8. **Preserve behavior.** Do not rename ids/classes the JS depends on; do not change the JS
   logic. Style only. Verify against the audit list from step 2.

9. **Verify.** Open the page (use the `run`/`verify` skills or a screenshot). Check: ambient
   glow visible, surfaces read as glass, hover lifts work, mobile drawer slides, all interactive
   controls still function. Compare against the **Hard rules** checklist below.

## Hard rules (copy from the design skill, keep on-brand)
1. Dark only. Near-black bg (`--bg-primary`), off-white text. Never light mode.
2. Surfaces are translucent **glass** (blur + hairline border + inset bevel), never flat blocks.
3. Accent = cyan→blue gradient, used sparingly.
4. Keep the ambient glow field behind everything.
5. Icons = Font Awesome 6.5.1 (+ the CV SVG if used). No emoji, no unicode icons.
6. Generous spacing, centered section titles w/ gradient underline, 16px radius, subtle hover lifts.
7. Calm motion only: `0.3s ease`, no bounce/spring/parallax.

## Notes
- The bilingual Arabic-name identity in the design skill is specific to the portfolio owner.
  For other products in this repo, keep the brand *material* (glass, glow, accent, type, icons)
  but only carry the Arabic identity if the product is Khalid's personal page.
- If the target is plain HTML/CSS/JS (no build), edit the files in place. If it's a framework,
  reuse the JSX components in `ui_kits/portfolio/` as references.
