# Prédicteur de dimensionnement matériel LLM — Design System

## 0. Research Log (greenfield)

- **Embedded refs**: brief is an operational single-page tool (dashboard-like) with no named brand; Layer A `taste-skill.md` + `layout-skill.md` loaded for execution discipline and app-shell mechanics. No Layer B brand picked because the product is a technical utility, not a marketed brand surface.
- **Lazyweb / imagen**: skipped — the task supplied a complete functional contract (plan §6 & §7) and no visual reference is required.

## 1. Atmosphere & Identity

A calm, technical command center: dense when needed, breathable by default. The signature is **clarity over decoration** — generous spacing, disciplined typography, and a single teal accent that guides attention without shouting. The interface feels like a precise instrument: every number has a label, every assumption is visible, every error is explicit.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/primary | `--surface-primary` | `#fafafa` | `#0a0a0a` | Page background |
| Surface/secondary | `--surface-secondary` | `#f4f4f5` | `#18181b` | Cards, panels |
| Surface/elevated | `--surface-elevated` | `#ffffff` | `#27272a` | Autocomplete dropdown, elevated cards |
| Text/primary | `--text-primary` | `#18181b` | `#fafafa` | Headings, body |
| Text/secondary | `--text-secondary` | `#71717a` | `#a1a1aa` | Captions, hints |
| Text/tertiary | `--text-tertiary` | `#a1a1aa` | `#71717a` | Disabled, muted |
| Border/default | `--border-default` | `#e4e4e7` | `#3f3f46` | Dividers, input borders |
| Border/subtle | `--border-subtle` | `#f4f4f5` | `#27272a` | Soft separations |
| Accent/primary | `--accent-primary` | `#0d9488` | `#14b8a6` | Primary actions, links, focus rings, weight segment |
| Accent/hover | `--accent-hover` | `#0f766e` | `#2dd4bf` | Button hover |
| Accent/subtle | `--accent-subtle` | `#ccfbf1` | `#115e59` | Tinted backgrounds, badges |
| Status/success | `--status-success` | `#16a34a` | `#22c55e` | Checkmarks, absorbed load |
| Status/warning | `--status-warning` | `#ca8a04` | `#eab308` | Warnings, inferred values |
| Status/error | `--status-error` | `#dc2626` | `#ef4444` | Errors, destructive |
| Status/info | `--status-info` | `#2563eb` | `#3b82f6` | Informational badges |

### Rules

- Surface hierarchy is created with tonal shifts and 1 px borders, not heavy shadows.
- Accent is reserved for interactive elements and the primary data series (weight segment in the donut).
- No purple / violet / neon gradients.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Usage |
|-------|------|--------|-------------|-------|
| Display | `2rem` / 32 px | 700 | 1.2 | Page title |
| H1 | `1.5rem` / 24 px | 700 | 1.25 | Section titles |
| H2 | `1.125rem` / 18 px | 600 | 1.3 | Card titles |
| H3 | `1rem` / 16 px | 600 | 1.4 | Subsection labels |
| Body | `1rem` / 16 px | 400 | 1.6 | Default text |
| Body/sm | `0.875rem` / 14 px | 400 | 1.5 | Secondary info |
| Caption | `0.75rem` / 12 px | 500 | 1.4 | Labels, metadata |

### Font Stack

- Primary: `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- Mono: `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace`

### Rules

- Max 2 font families.
- Body text never below 14 px.
- Numbers in metrics use the mono stack for tabular alignment.

## 4. Spacing & Layout

### Base Unit

4 px base unit.

| Token | Value | Usage |
|-------|-------|-------|
| `--space-1` | 4 px | Icon-to-label gaps |
| `--space-2` | 8 px | Tight groups |
| `--space-3` | 12 px | Inline form padding |
| `--space-4` | 16 px | Card padding, section gaps |
| `--space-5` | 20 px | Form row gaps |
| `--space-6` | 24 px | Column gaps |
| `--space-8` | 32 px | Section separation |
| `--space-10` | 40 px | Page vertical rhythm |
| `--space-12` | 48 px | Major breaks |

### Grid

- Page: max-width `1280px`, centered, padding `var(--space-4)` to `var(--space-6)`.
- Main layout: CSS grid `1fr 1fr` at `>= 900px`, single column below.
- Form column max-width `560px`.
- Results column is a vertical stack (`stack` primitive) with `var(--space-6)` gap.

### Layout Primitives

- **stack**: flex column + gap for vertical rhythm.
- **cluster**: flex row + wrap + gap for tags and chips.
- **sidebar**: two-column grid that collapses to single column on narrow viewports.

## 5. Components

### Button

- Structure: `<button>` with padding `0.625rem 1rem`, radius `8px`, font-weight 600.
- Variants: primary (accent background, white text), secondary (transparent + border), ghost (text only).
- States: hover (background darkens / accent-hover), active (`scale(0.98)`), focus (`2px accent outline, 2px offset`), disabled (`opacity 0.5`, `cursor: not-allowed`).
- Accessibility: `type="button"` when not submitting, visible focus ring.

### Input / Select

- Label above input, `gap: var(--space-1)`.
- Input: `padding: 0.625rem 0.75rem`, radius `8px`, border `1px solid var(--border-default)`, background `var(--surface-elevated)`.
- Focus: border accent + outline ring.
- Select: same treatment, native `<select>` for reliability.

### Card

- Background `var(--surface-secondary)`, border `1px solid var(--border-default)`, radius `12px`, padding `var(--space-5)`.
- No shadow by default; tonal shift provides depth.

### Badge / Chip

- Small rounded pill (`999px` radius), padding `0.25rem 0.625rem`, `font-size: 0.75rem`, font-weight 600.
- Variants: accent-subtle, warning, error, info, neutral.

### Donut Chart

- SVG with `viewBox="0 0 120 120"`, stroke-width `18`, radius `50`.
- Three segments: weight (accent), KV (info), overhead (neutral/tertiary).
- Center label shows total VRAM.
- No chart library.

### BOM Table

- `<table>` with column headers: libellé, qté, prix unitaire, total.
- Row dividers `1px solid var(--border-subtle)`.
- Total row bold, top border `2px solid var(--border-default)`.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Micro | 100 ms | ease-out | Button press |
| Standard | 200 ms | ease-in-out | Border/focus transitions |
| Emphasis | 300 ms | cubic-bezier(0.16, 1, 0.3, 1) | Card entrance |

### Rules

- Only animate `transform` and `opacity`.
- Every interactive element has hover + active + focus states.
- Respect `prefers-reduced-motion`: disable non-essential transitions.
- No decorative infinite animations.

## 7. Depth & Surface

### Strategy

**Borders + tonal-shift**: cards and panels sit on top of the page background through subtle background changes and 1 px borders. No drop shadows on light backgrounds; on dark mode a very subtle shadow may reinforce elevated surfaces.

| Level | Value | Usage |
|-------|-------|-------|
| Card | `1px solid var(--border-default)` | Default card |
| Elevated | `1px solid var(--border-default)` + `0 4px 12px rgba(0,0,0,0.08)` (dark only) | Dropdowns, modals |
| Divider | `1px solid var(--border-subtle)` | Section separators |

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- WCAG 2.2 AA target: 4.5:1 body text, 3:1 large text, visible focus on every interactive element.
- All form controls have associated `<label>`.
- Autocomplete implements `role="listbox"` / `role="option"` with `aria-activedescendant`.
- `prefers-reduced-motion` disables transitions.
- No emojis in visible UI — inline SVG icons only.

### Accepted Debt

| Item | Location | Why accepted | Owner / Exit |
|------|----------|--------------|--------------|
| No third-party icon library | inline SVG icons | Project forbids new runtime dependencies | Add Lucide/Phosphor if deps become allowed |
| Native `<select>` for context & margin | LoadForm.tsx | Accessibility and zero-dependency; custom listbox deferred | Replace only if keyboard UX demands it |
