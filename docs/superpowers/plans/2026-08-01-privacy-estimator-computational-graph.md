# Privacy Estimator Computational Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the coarse privacy-estimator SVG with a component-level, rigid-grid computational graph whose orthogonal arrows are routed by Graphviz.

**Architecture:** Reuse the HTML-node, CSS-grid, and pinned-neato edge system already used by `interactive-ranker-v2.html` FIG M1. Two independent forward-pass modules expose relation and count operations, tensor shapes, trainability, objectives, and the relation-to-count checkpoint handoff without changing the report prose or experimental claims.

**Tech Stack:** Semantic HTML5, existing `docs/html/css/site.css`, `@viz-js/viz@3.28.0`, existing `docs/html/js/ranker-model-diagram.js`, Graphviz DOT, Chromium.

## Global Constraints

- Modify only `docs/html/privacy-estimator.html` plus this plan during implementation.
- Do not change the abstract, architecture prose, metrics, navigation, model code, or training artifacts.
- Do not add a dependency, stylesheet, or diagram-specific JavaScript file.
- Every forward operation named in the approved design must appear as a separate visual node or operator.
- CSS grid owns layout; Graphviz owns edges only.
- The figure must remain understandable if Graphviz fails to load.
- Preserve the report's zero-em-dash prose rule.

---

### Task 1: Replace the inline SVG with the computational graph

**Files:**
- Modify: `docs/html/privacy-estimator.html`

**Interfaces:**
- Consumes: `.ranker-model-diagram*` classes in `docs/html/css/site.css`, `globalThis.Viz.instance()`, and the generic `[data-ranker-model-dot]` router in `docs/html/js/ranker-model-diagram.js`.
- Produces: one `#privacy-estimator-diagram.ranker-model-diagram` with unique `data-node-id` nodes, one DOT source, one edge overlay, and one routing-status element.

- [x] **Step 1: Record a failing structural baseline for the requested graph**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
text = Path('docs/html/privacy-estimator.html').read_text()
assert 'data-privacy-estimator-dot' in text
assert 'ranker-model-diagram.js' in text
assert 'H_rel [B × L × 768]' in text
PY
```

Expected: failure because the page still contains an inline SVG.

- [x] **Step 2: Import the existing Graphviz runtime and router**

Add these scripts in the page head, after `css/site.css`:

```html
<script defer src="https://cdn.jsdelivr.net/npm/@viz-js/viz@3.28.0/dist/viz-global.js"></script>
<script defer src="js/ranker-model-diagram.js"></script>
```

- [x] **Step 3: Replace the inline SVG with two rigid graph modules**

Use this container contract:

```html
<div id="privacy-estimator-diagram" class="ranker-model-diagram" role="img" aria-label="Component-level computational graph for the ML k-anonymity privacy estimator">
  <div class="ranker-model-diagram__layout">...</div>
  <svg class="ranker-model-diagram__edges" aria-hidden="true" focusable="false"></svg>
  <script type="text/vnd.graphviz" data-ranker-model-dot data-privacy-estimator-dot>...</script>
  <div class="ranker-model-diagram__status">Computing architecture rendering…</div>
</div>
```

Build separate relation and count modules. Give every operation listed in the approved design its own `data-node-id`. Use existing `--linear`, `--activation`, `--operator`, `--normalization`, `--tensor`, `--encoder`, `--output`, and `--trainable` variants. Use inline outline SVGs from the reference figure for trapezoid linear nodes and hexagonal GELU nodes.

- [x] **Step 4: Declare the complete forward, target, loss, and gradient graph in DOT**

The DOT block must include solid forward edges, dashed supervision and checkpoint edges, and colored gradient-ownership edges. Its structure must cover these chains:

```dot
relation_fields -> relation_serialization -> relation_tokenizer -> relation_batch;
relation_batch -> frozen_encoder; frozen_encoder -> trainable_encoder;
trainable_encoder -> relation_hidden; relation_hidden -> relation_pool;
relation_mask -> relation_pool; relation_pool -> relation_norm;
relation_norm -> relation_linear_128 -> relation_gelu -> relation_dropout -> relation_linear_4 -> relation_logits;
relation_targets -> relation_ce; relation_logits -> relation_ce;

count_fields -> count_serialization -> count_tokenizer -> count_encoder;
trainable_encoder -> count_encoder [style=dashed];
count_encoder -> count_hidden -> count_pool -> count_cache -> count_norm;
count_norm -> count_linear_64 -> count_gelu -> count_linear_1 -> count_standardized;
count_standardized -> count_inverse -> count_output;
count_target -> target_standardize -> count_regression;
count_standardized -> count_regression;
formal_pairs -> shared_pair_head -> pair_difference -> pair_loss;
count_regression -> count_loss_sum; pair_loss -> count_loss_sum; count_loss_sum -> count_loss;
```

All DOT endpoint identifiers must match the HTML `data-node-id` values exactly.

- [x] **Step 5: Run the structural validator**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
import re

text = Path('docs/html/privacy-estimator.html').read_text()
HTMLParser().feed(text)
node_ids = re.findall(r'data-node-id="([^"]+)"', text)
dot = re.search(r'<script type="text/vnd.graphviz"[^>]*data-privacy-estimator-dot[^>]*>(.*?)</script>', text, re.S).group(1)
endpoints = set(re.findall(r'\b([a-z][a-z0-9_]*)\s*->\s*([a-z][a-z0-9_]*)', dot))
referenced = {value for edge in endpoints for value in edge}
assert len(node_ids) == len(set(node_ids))
assert referenced <= set(node_ids), sorted(referenced - set(node_ids))
assert text.count('data-privacy-estimator-dot') == 1
assert text.count('ranker-model-diagram.js') == 1
assert text.count('viz-global.js') == 1
assert '<svg viewBox="0 0 1420 1010"' not in text
for label in ('H_rel [B × L × 768]', 'z_rel [B × 768]', 'Linear 768 → 128', 'Dropout 0.1', 'Linear 768 → 64', 'ŷ_logK = ŷ_std × σ_train + μ_train', 'softplus(−Δ)', 'AdamW · head only'):
    assert label in text, label
print(f'graph structure: PASS; nodes={len(node_ids)}; routed_endpoints={len(referenced)}')
PY
```

Expected: `graph structure: PASS` with no unmatched endpoints.

---

### Task 2: Validate the Graphviz rendering and finish the report revision

**Files:**
- Modify: `docs/html/privacy-estimator.html` only if validation exposes a visual or terminology defect.
- Modify: `docs/superpowers/plans/2026-08-01-privacy-estimator-computational-graph.md` to mark completed steps.

**Interfaces:**
- Consumes: the completed graph from Task 1 and a Chromium browser with JavaScript enabled.
- Produces: a verified desktop and narrow rendering with `data-renderer="grid-neato"` and no prose regression.

- [x] **Step 1: Check JavaScript-enabled Graphviz completion**

Run Chromium against the local file and wait for `#privacy-estimator-diagram[data-renderer="grid-neato"]`. Evaluate that the status element is hidden and the edge overlay contains Graphviz edge groups.

Expected browser predicates:

```javascript
diagram.dataset.renderer === "grid-neato"
diagram.querySelector(".ranker-model-diagram__status").hidden === true
diagram.querySelectorAll(".ranker-model-diagram__edges g.edge").length >= 30
```

- [x] **Step 2: Capture and inspect desktop and narrow screenshots**

Render at `1440 × 5200` and `768 × 5200`. Confirm the relation and count grids are aligned, node shapes remain distinct, labels are readable, the complete graph is visible, and no edge crosses a non-endpoint node interior.

- [x] **Step 3: Run prose and terminology checks**

Apply `proofread`, `humanize`, and `term-audit` to all newly added labels and accessibility text. Preserve exact implementation terms such as `special_tokens_mask`, Smooth-L1, and ClinicalMosaic. Re-run the Task 1 structural validator after any edit.

- [x] **Step 4: Run final static checks**

Run:

```bash
node --check docs/html/js/nav.js
git diff --check
rg -n '\{\{' docs/html/privacy-estimator.html
```

Expected: JavaScript syntax and diff checks exit zero; the placeholder scan prints no matches.

- [x] **Step 5: Commit the scoped implementation**

```bash
git add docs/html/privacy-estimator.html docs/superpowers/plans/2026-08-01-privacy-estimator-computational-graph.md
git commit -m "docs: detail privacy estimator computational graph"
```
