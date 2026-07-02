# HTML paper template — specification

A single-page (or multi-page) research report styled as an academic paper, deployable as
a GitHub Pages site. Engineering-blueprint aesthetic: white paper, Fraunces display serif +
JetBrains Mono body, sharp rules, standardized plot colors. Self-contained except Google
Fonts + Plotly.

## Files

```
docs/                         ← GitHub Pages source (Settings → Pages → main /docs)
  index.html                  ← redirect to html/<page>.html (canonical entry)
  .nojekyll                   ← serve files as-is (no Jekyll processing of .md)
  html/
    <page>.html               ← the report (copy template.html, rename)
    css/site.css              ← design system (do not fork per page)
    js/nav.js                 ← global navbar, single source of truth
    js/plotly-basic.min.js    ← vendored; drop in, not in this template (≈1.1 MB)
```

Enable Pages once: `gh api -X POST repos/<owner>/<repo>/pages -f 'source[branch]=main' -f 'source[path]=/docs'`.
Live at `https://<owner>.github.io/<repo>/`.

## Site structure

- **One page per artifact/result-set.** Add a page = copy `template.html` → `html/<slug>.html`, then
  append `{href:"<slug>.html", label:"…"}` to `PAGES` in `js/nav.js`. Never hand-write a `<div class="topnav">`
  body — leave `<div class="topnav" id="site-nav"></div>` empty; `nav.js` fills it and marks the current page.
- Brand text = the site name (e.g. `HarnessCloak`). Links are black (`--ink`).
- `index.html` redirects `/` to the primary page so the bare URL lands on content.

## Page structure

Order, top to bottom (all inside `.sheet`):

1. `#site-nav` placeholder (navbar).
2. `header.masthead` — **centered**: `h1.title` with a block `.title-sub` (same black, non-italic, 34px),
   a `.rev` date line, then the **Abstract** block.
3. `§01 Introduction` — `.lede` (the problem + the single headline finding, self-contained) then `.prose`.
4. `§02 Preliminaries` — a `table.spec` **glossary** (required whenever the vocabulary is dense).
5. `§03 Method` — pipeline, mechanism, measurement loop.
6. `§04 Measures` — one row per metric: what it reports, which direction is better.
7. `§05 Results` — `<h4>` subsection per factor/lever; each holds prose + a `.plot-frame` + a **Conclusion**.
8. `§06 Findings` · `§07 Analysis` · `§08 Discussion` (see skills below).
9. `footer.colophon` — Code / Notes / References.

Every section uses `.section-head` = `.section-num` (§NN) + `.section-title` + `.section-meta`. Renumber
contiguously if you drop a section. `<h4>` results subsections: `font-family:Fraunces; font-size:20px`.

## Abstract

- **Write it last**, from the finished Results/Findings, with the **`/abstract` skill**, target **~100 words**.
- Results-first arc: context → gap → approach → the headline number(s) → significance; state the central
  caveat (N, single-seed, proxy metric).
- Renders in `.abstract` (centered label "Abstract", justified body column, **not italic**). The body reuses
  `.subtitle` styling.

## Prose

- Lead with the contribution; never "In this paper/report we…" or restate the title.
- Active voice; past tense for what was done, present for what holds; concrete nouns; name the number.
- **House style (enforced by term-audit):** **zero em-dashes** in prose (use commas/colons/parentheses),
  ≤2 semicolons per ~1000 words, cut hype ("novel/comprehensive/robust/crucial") and filler transitions.
- Honesty: never overclaim beyond results; put scope/caveats where the claim is made.

## Plots

- Plotly, one `<figure>`-equivalent per `.plot-frame`: a top `.plot-cap` (`<span class="l">FIG · NN — title</span>`
  `<span class="r">source: results/…</span>`), the plot `<div>`(s), and a bottom `.plot-cap` "Read" caption
  that tells the reader the takeaway in one sentence.
- Config: transparent bg, JetBrains Mono 11px, legends horizontal **below** the plot, `displayModeBar:false`,
  `responsive:true`, gridcolor `rgba(20,24,28,0.08)`. Use `Plotly.react` (not `newPlot`) so toggles re-render.
- Side-by-side panels: a flex row of `flex:1 1 320px; min-width:300px` divs.
- Keep the JS palette + helper functions (`linePanel`, `barPanel`) from `template.html` — they encode the
  standardized colors. Replace only the data in the example calls.
- Any color named in a caption ("violet = anisotropic") must match the actual series color.

## Tables

- `table.spec` (2-col: term/def) or `table.spec--wide` (multi-col results). Right-align numeric cells with
  `class="num"`; mark good/bad with `accent`/`warn` num classes.
- Wide tables: wrap in `<div style="overflow-x:auto;">` so they scroll instead of breaking the layout.
- Never uppercase or reflow data/file paths.

## Skills to use (workflow order)

1. **Results → claims: `/result-to-claim`.** After experiments finish, run the gate (Codex) to judge which
   claims the numbers support (yes/partial/no) and flag confounds. Write **§06 Findings** and **§07 Analysis**
   *from its verdicts* — tag each finding Supported/Partial/Refuted with grounds + scope. Do not headline a
   claim the gate rated Partial.
2. **Harden: `/auto-review-loop`** on §06–08 — iterate review → fix → re-review until it passes, for the
   sections that state claims.
3. **Abstract: `/abstract`** (~100 words), once Findings are final.
4. **Proofreading pass, in this order:**
   - **`/proofread`** — typos, doubled words, search-replace artifacts, structure/overflow, stale
     cross-references (also reconcile the intro/masthead against the final Findings).
   - **`/humanize`** — strip AI-voice tells (boilerplate transitions, tricolons, hedging stacks, formulaic openers).
   - **`/term-audit`** — word choice + register + **em-dash removal** (house rule: 0 in prose) + canonical terms.

Run 4 after every substantive content change. Keep the HTML valid throughout (balanced tags, prose-only edits).

## Footer

- `footer.colophon`, `grid-template-columns: repeat(3, 1fr)` (three **equal** columns), `align-items:start`.
  Each column = `<div>` with an `<h5>` and a `<ul>`, **one link per `<li>`** (never comma-separate links on one line).
- **Code** and **Notes**: every file links to the repo on GitHub — `https://<owner>/<repo>/blob/main/<path>`
  (so on the Pages site the link opens the source). Verify each path exists before publishing (no 404s).
- **References**: for a paper with a repo note (`research-wiki/papers/<slug>.md`), link the **paper name** to
  that note; then **always** append an arXiv (or equivalent DOI) link, e.g.
  `InferDPT · Tong et al. 2023 · arXiv 2310.12214`. Papers without a note: plain name + arXiv link.
  Verify arXiv IDs (search/`arxiv`) — do not guess.

## Standardized colors

Color encodes **metric role**, consistently across every figure. Warm = privacy-side, cool = utility-side,
gray = baseline, and a reserved pair for entity comparisons. Defined in the page JS (see `template.html`);
metric-family and ε colors also exist as CSS tokens in `site.css`.

| Group | Members → hex |
|---|---|
| **Privacy / leakage** (amber family) | overlap `#a3460a` · S_w `#cf7a2e` · PII-leak `#6e4a2b` · inv@10 `#8c3b3b` |
| **Utility / coherence** (navy family) | utility `#2a5092` · control `#5b86b5` · coherence `#2e6a5e` · reranker `#3a3f6b` |
| **ε (ordered)** | navy sequential ramp `#cfe0f0 → #8fb0d6 → #4f7bb0 → #2a5092` |
| **Entity/condition A vs B** | violet `#7a5aa8` · green `#57a94a` (both off the amber/navy families) |
| **Baseline series** | gray `#8a8a8a`, dashed |

Rules: (1) the same hue always means the same role across plots; (2) within a family, members are distinct
same-temperature hues; (3) comparison plots (color = entity, y-axis mixes metrics) must use the violet/green
pair or gray baseline — never amber/navy, which would falsely signal a metric role; (4) never re-introduce the
old terracotta accent — the site accent (`--accent`) is neutral gray.

Methodology for extending: assign a new privacy metric a distinct warm hue, a new utility metric a distinct
cool hue, a new ordered variable its own sequential ramp, a new compared entity a reserved off-family hue.
