---
type: plan
status: current
created: 2026-07-23
updated: 2026-07-23
tags: [html-report, diagrams, graphviz, viz-js, css-grid, architecture]
supersedes: docs/plans/2026-07-23-graphviz-architecture-diagrams.md
---

# Grid-Pinned Graphviz Diagrams

**Goal:** Make M1's within-section layout an explicit, consistently spaced grid while keeping
automatic, obstacle-aware edge routing.

**Architecture (implemented):** The page owns everything visual — nodes are styled HTML in a
CSS grid; Graphviz routes only the edges at grid-pinned positions.

- **Nodes and section frames are static HTML** in `docs/html/interactive-ranker-v2.html`.
  Every level of nesting is one of two primitives — `ranker-model-diagram__row` / `__col` — with
  column templates inline on the element, so composition is editable in place. All spacing derives
  from four custom properties (`--dg-gap-lg`, `--dg-gap`, `--dg-gap-sm`, `--dg-pad`) declared on
  `.ranker-model-diagram` in `docs/html/css/site.css`. Each node is a `[data-node-id]` div with
  its title/detail as selectable HTML, styled per kind in CSS; the browser sizes nodes from their
  text, so no width tuning against Graphviz font metrics is needed. Sloped shapes (trapezium,
  hexagon) get their outline from an inline SVG path with `vector-effect="non-scaling-stroke"` and
  `preserveAspectRatio="none"` — a uniform stroke at any box size, unlike the earlier clip-path
  inset (uneven) or Graphviz-drawn nodes (double definition).
- **The page DOT block (`data-ranker-model-dot`) contains only edge topology** and per-edge style
  attributes. No node declarations, positions, ranks, subgraphs, or invisible constraint edges.
- **The generic renderer** (`docs/html/js/ranker-model-diagram.js`) measures every
  `[data-node-id]` box, emits a DOT prologue pinning each as an invisible fixed-size obstacle
  (`pos="x,y!"`, `fixedsize=true`, circle or box per `data-port-shape`), appends the page's edge
  statements, and renders with **neato + `splines=ortho`** via the pinned Viz.js runtime — the
  obstacle-aware routing the jsPlumb attempt lacked. A per-axis least-squares fit from rendered
  node centers back to measured centers calibrates the overlay exactly; node groups are stripped
  and only edges are injected. A `ResizeObserver` re-routes on reflow.
  - Firefox note: the overlay SVG gets explicit `width`/`height`/`viewBox` attributes each render
    and the coordinate origin is measured from the layout **div** — Firefox returns the 300×150
    intrinsic default from `getBoundingClientRect()` on an auto-sized `<svg>`.

**Why the predecessors failed:** jsPlumb kept HTML nodes but routed edges through them (no
obstacle awareness); pure dot routed well but placed nodes arbitrarily, needing invisible edges,
`rank=same` hacks, and post-hoc cluster-rectangle rewriting to fake a grid.

**Contract** (enforced by `scripts/spikes/check_ranker_m1_dom_layout.mjs`):

- ≥ 40 `data-node-id` HTML nodes with in-page labels and the six section frames live in the
  report page; sloped shapes use non-scaling-stroke SVG outlines.
- The DOT block is edges-only.
- The renderer knows no diagram-specific node or edge names.
- Spacing variables and the row/col primitives exist in the stylesheet.
- The three downstream inline figures remain byte-for-byte unchanged.
