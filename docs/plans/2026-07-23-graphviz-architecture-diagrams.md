---
type: plan
status: current
created: 2026-07-23
updated: 2026-07-23
tags: [html-report, diagrams, graphviz, viz-js, architecture]
supersedes: docs/plans/2026-07-23-html-native-architecture-diagrams.md
---

# Graphviz Architecture Diagrams Implementation Plan

**Goal:** Replace M1's diagram-specific HTML/JavaScript renderer with editable DOT source and one reusable Viz.js renderer.

**Architecture:** The report embeds DOT as the maintained graph artifact. Graphviz owns node sizing, ranks, clusters, and edge routing; a generic browser module renders the source to SVG and exposes errors without knowing any M1 node or edge names.

**Constraints:**

- Preserve the accepted M1 semantic content and overall top-to-bottom hierarchy.
- Keep all graph topology, labels, shapes, clusters, and layout hints in page DOT.
- Do not encode node IDs, edges, coordinates, or source-target routing conditions in JavaScript.
- Use no absolute node positions.
- Preserve the three downstream figures byte-for-byte.
- Reject the approach if M1 requires more than fifteen explicit ordering or rank constraints.

## Implementation

1. Write a failing structural test requiring embedded DOT, nested clusters, representative nodes and edges, the pinned Viz.js runtime, and a generic renderer.
2. Replace M1's generated-node shell with a DOT source block and SVG output container.
3. Implement the generic renderer with `Viz.instance()` and `renderSVGElement()`.
4. Encode the architecture with semantic node classes, nested clusters, rank constraints, and Graphviz-managed routing.
5. Render M1 in Firefox and adjust only DOT attributes or shared renderer behavior.
6. Verify syntax, structure, overflow, downstream-figure preservation, and rendered Graphviz element counts.
