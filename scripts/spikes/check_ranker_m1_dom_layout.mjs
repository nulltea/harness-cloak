import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const reportPath = "docs/html/interactive-ranker-v2.html";
const rendererPath = "docs/html/js/ranker-model-diagram.js";
const stylesheetPath = "docs/html/css/site.css";
const report = readFileSync(reportPath, "utf8");
const renderer = readFileSync(rendererPath, "utf8");
const stylesheet = readFileSync(stylesheetPath, "utf8");

// Contract: the page owns nodes as styled HTML (CSS grid placement, SVG
// outlines for sloped shapes); the page DOT block owns only edge topology;
// the renderer is generic (measure -> pin -> neato ortho routing).

assert.match(
  report,
  /id="ranker-model-diagram"[^>]*class="ranker-model-diagram"/,
  "M1 must keep its isolated figure container",
);
assert.ok(
  (report.match(/data-node-id="/g) ?? []).length >= 40,
  "M1 must declare its complete architecture as HTML nodes",
);
assert.ok(
  (report.match(/ranker-model-diagram__node-title/g) ?? []).length >= 40,
  "M1 nodes must carry their labels as page HTML",
);
assert.ok(
  report.includes('vector-effect="non-scaling-stroke"'),
  "sloped node shapes must use SVG outlines with non-scaling stroke",
);
for (const nodeId of [
  "clinical_encoder",
  "document_bank",
  "augmented_tokens",
  "utility_logit",
  "history_attention",
  "privacy_score",
  "utility_gap",
  "additive_controller",
  "policy_output",
]) {
  assert.ok(
    report.includes(`data-node-id="${nodeId}"`),
    `M1 must declare HTML node ${nodeId}`,
  );
}
for (const sectionLabel of [
  "Shared frozen encoder",
  "Utility scoring",
  "Token feature fusion",
  "Selected-action memory",
  "Semantic privacy estimation",
  "Additive lambda controller",
]) {
  assert.ok(
    report.includes(sectionLabel),
    `M1 must keep the ${sectionLabel} section frame`,
  );
}
assert.equal(
  (report.match(/<svg\b/g) ?? []).length
    - (report.match(/ranker-model-diagram__node-outline/g) ?? []).length,
  3,
  "each grid figure (M1, T2, I1) contributes shape outlines plus one empty edge overlay",
);
assert.ok(
  report.indexOf("@viz-js/viz@3.28.0/dist/viz-global.js")
    < report.indexOf("js/ranker-model-diagram.js"),
  "the pinned Viz.js runtime must load before the generic renderer",
);

const dotMatch = report.match(
  /<script type="text\/vnd\.graphviz" data-ranker-model-dot>([\s\S]*?)<\/script>/,
);
assert.ok(dotMatch, "M1 must keep editable DOT edge source beside the figure");
const dot = dotMatch[1];
for (const edge of [
  "document_chunks -> clinical_encoder",
  "document_bank -> document_projection",
  "relation_pair -> privacy_join",
  "token_sum -> augmented_tokens",
  "query_projection -> global_attention",
  "utility_relation -> context_interaction",
  "memory_query -> history_attention",
  "profile_normalization -> privacy_score",
  "lambda_transform -> alpha",
  "log_softmax -> policy_output",
]) {
  assert.ok(dot.includes(edge), `DOT source must contain edge ${edge}`);
}
assert.ok(
  (dot.match(/->/g) ?? []).length >= 45,
  "DOT source must carry the full edge topology",
);
// The DOT block is edges-only: no node declarations, placement, or clusters.
for (const forbidden of [/\bpos\s*=/, /\brank\s*=/, /subgraph/, /\bgroup\s*=/, /invis/, /\bconstraint\s*=/, /\bweight\s*=/, /\bshape\s*=/, /\bfillcolor\s*=/, /\blabel\s*=/]) {
  assert.doesNotMatch(dot, forbidden, `DOT edge source must not contain ${forbidden}`);
}

for (const requiredFragment of [
  "Viz.instance",
  "renderSVGElement",
  "data-ranker-model-dot",
  "data-node-id",
  "engine: \"neato\"",
  "fixedsize=true",
  "splines=ortho",
  "ResizeObserver",
]) {
  assert.ok(renderer.includes(requiredFragment), `renderer must contain ${requiredFragment}`);
}
for (const forbiddenFragment of [
  "nodeSpecs",
  "edgeSpecs",
  "busSpecs",
  "clinical_encoder",
  "utility_relation",
  "privacy_control",
]) {
  assert.ok(
    !renderer.includes(forbiddenFragment),
    `generic renderer must not contain diagram-specific ${forbiddenFragment}`,
  );
}

// Spacing consistency: one variable scale, reusable row/col primitives.
for (const cssFragment of [
  "--dg-gap-lg:",
  "--dg-gap:",
  "--dg-gap-sm:",
  "--dg-pad:",
  ".ranker-model-diagram__row {",
  ".ranker-model-diagram__col {",
  ".ranker-model-diagram__edges {",
]) {
  assert.ok(stylesheet.includes(cssFragment), `stylesheet must contain ${cssFragment}`);
}

assert.ok(!report.includes("FIG · T1"), "figure T1 is retired; RL training lives in T2");
for (const caption of ["FIG · T2: Training process and reward flow", "FIG · I1: Deployed document inference"]) {
  assert.ok(report.includes(caption), `page must keep ${caption}`);
}
assert.equal(
  (report.match(/data-ranker-model-dot/g) ?? []).length,
  3,
  "M1, T2, and I1 each carry an edge-only DOT block for the shared renderer",
);
for (const t2Node of ["env_docp", "warmstart_policy", "rloo_credit", "optimizer_step"]) {
  assert.ok(
    report.includes(`data-node-id="${t2Node}"`),
    `T2 must declare HTML node ${t2Node}`,
  );
}
for (const i1Node of ["i1_doc", "i1_menu", "i1_docp", "i1_remote", "i1_record", "i1_outfinal"]) {
  assert.ok(
    report.includes(`data-node-id="${i1Node}"`),
    `I1 must declare HTML node ${i1Node}`,
  );
}
assert.ok(
  !report.includes("absent at inference"),
  "I1 no longer carries the absent-at-inference strip",
);

console.log("M1 HTML nodes, edge-only DOT, generic neato renderer, and figure roster are valid.");
