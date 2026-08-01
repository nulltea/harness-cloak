# Privacy Estimator Diagram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `docs/html/privacy-estimator.html` as a focused architecture report for the trained ML k-anonymity privacy estimator.

**Architecture:** Use a compact masthead abstract and one model-architecture section followed by a self-contained inline SVG inside the established `diagram-frame` component. The overview and graph show the relation and count inputs, their distinct packing, shared ClinicalMosaic encoder, mean pooling, disjoint heads, trainability boundaries, objectives, and development outcomes without proposing a replacement architecture.

**Tech Stack:** Semantic HTML5, inline SVG, the existing `docs/html/css/site.css` design system, and the existing global navigation script.

## Global Constraints

- Create the page in `/home/timo/repos/agent-cloak/docs/html` as explicitly requested.
- The rendered content contains only a masthead abstract, one model-architecture overview, and the architecture diagram; no separate results, findings, discussion, or footer.
- Reuse the FIG · M1 blueprint language from `docs/html/interactive-ranker-v2.html`.
- Represent the implemented v3 model faithfully: context-bearing four-way relation input, candidate-only count input, shared ClinicalMosaic encoder, masked mean pooling, disjoint MLP heads, top-four relation fine-tuning, and frozen-encoder count training.
- Do not imply that the architecture is promoted or that relation performance improved.
- Do not add dependencies or page-specific stylesheets.

---

### Task 1: Build and publish the architecture diagram

**Files:**
- Create: `docs/html/privacy-estimator.html`
- Modify: `docs/html/js/nav.js`

**Interfaces:**
- Consumes: `docs/html/css/site.css`, `docs/html/js/nav.js`, and the diagram conventions in `docs/html/interactive-ranker-v2.html`.
- Produces: a standalone responsive page at `docs/html/privacy-estimator.html` and a `Privacy estimator` navbar entry.

- [x] **Step 1: Record the missing-page baseline**

Run:

```bash
test ! -e /home/timo/repos/agent-cloak/docs/html/privacy-estimator.html
```

Expected: exit status 0 before creation.

- [x] **Step 2: Create the focused architecture page and inline SVG**

Create a valid HTML document containing the shared nav placeholder, a masthead abstract, one model-architecture overview, and one `diagram-frame`. The SVG must include these connected groups:

```text
relation example -> relation packing -> ClinicalMosaic -> masked mean pool -> 128-wide GELU head -> four relation logits
count universe   -> count packing    -> ClinicalMosaic -> masked mean pool -> 64-wide GELU head  -> predicted log count
formal ordering pairs ---------------------------------------------------------> count ordering loss
```

The shared encoder group must label relation-time ownership (`layers 8-11 trainable`) and count-time ownership (`encoder frozen; cached features`). The bottom training strip must distinguish balanced four-class cross-entropy from smooth-L1 plus pairwise ordering loss.

- [x] **Step 3: Register the page in global navigation**

Append exactly one entry to `PAGES` in `docs/html/js/nav.js`:

```javascript
{ href: "privacy-estimator.html", label: "Privacy estimator" }
```

- [x] **Step 4: Verify structural and semantic requirements**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

path = Path('/home/timo/repos/agent-cloak/docs/html/privacy-estimator.html')
text = path.read_text()
HTMLParser().feed(text)
assert '{{' not in text
assert '<header class="masthead">' in text
assert text.count('<section') == 1
assert 'Model Architecture' in text
assert '<footer' not in text
assert text.count('<svg') == 1
assert 'valid generalization' in text
assert 'related but non-substitutable' in text
assert 'predicted log count' in text
assert 'layers 8-11 trainable' in text
assert 'encoder frozen' in text
print('privacy-estimator structure: PASS')
PY
```

Expected: `privacy-estimator structure: PASS`.

- [x] **Step 5: Render-check the page**

Open the local page in a browser at desktop and narrow viewport widths. Confirm that all nodes and arrows remain visible, SVG text is legible, the diagram scrolls or scales without clipping, and the nav marks `Privacy estimator` current.

- [x] **Step 6: Run the required prose and terminology checks**

Apply the `proofread`, `humanize`, and `term-audit` skills in that order to SVG labels, captions, and accessibility text. Re-run Step 4 after any label change.

- [x] **Step 7: Commit only the page, nav entry, and plan**

```bash
git -C /home/timo/repos/agent-cloak add docs/html/privacy-estimator.html docs/html/js/nav.js docs/superpowers/plans/2026-08-01-privacy-estimator-diagram.md
git -C /home/timo/repos/agent-cloak commit -m "docs: diagram privacy estimator architecture"
```
