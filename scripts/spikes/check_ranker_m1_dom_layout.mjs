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
  "lex_semantic_side_stack",
  "lex_feature_concat",
  "residual_logit",
  "lexicographic_policy_output",
  "preference_conditioning",
  "routed_utility_return",
  "previous_policy",
  "utility_reference",
  "utility_advantage",
  "policy_ratio",
  "utility_ppo",
  "actor_objective",
  "utility_critic",
  "critic_losses",
  "utility_violation",
  "document_dual",
  "dual_loss",
  "count_reward_source",
  "count_critic",
  "count_advantage",
  "count_policy_ratio",
  "count_ppo",
  "count_actor_gradient",
]) {
  assert.ok(
    report.includes(`data-node-id="${nodeId}"`),
    `M1 must declare HTML node ${nodeId}`,
  );
}
for (const sectionLabel of [
  "Tier 1 · Frozen clinical substrate",
  "Tier 2 · Immutable utility branch",
  "Token feature fusion",
  "Selected-action memory",
  "Tier 3 · Trainable lexicographic side path",
  "Utility-first/count-second · learned branch",
  "Training-only actor-critic",
  "Count reward trains πlex through PPO; no count value enters the deployed policy input",
  "Training-only actor-critic and document dual",
  "Frozen measurements",
  "Advantages and constraint",
  "Separate objective terms",
  "Owned updates",
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
const m1NodeIds = new Set(
  [...report.matchAll(/data-node-id="([^"]+)"/g)].map((match) => match[1]),
);
for (const match of dot.matchAll(/^\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)/gm)) {
  assert.ok(m1NodeIds.has(match[1]), `DOT edge source is missing HTML node ${match[1]}`);
  assert.ok(m1NodeIds.has(match[2]), `DOT edge target is missing HTML node ${match[2]}`);
}
for (const edge of [
  "document_chunks -> clinical_encoder",
  "document_bank -> document_projection",
  "relation_pair -> lex_semantic_side_stack",
  "token_sum -> augmented_tokens",
  "query_projection -> global_attention",
  "utility_relation -> context_interaction",
  "memory_query -> history_attention",
  "residual_output -> residual_logit",
  "lexicographic_log_softmax -> lexicographic_policy_output",
  "preference_conditioning -> lex_semantic_side_stack",
  "count_reward_source -> count_advantage",
  "count_critic -> count_advantage",
  "count_advantage -> count_ppo",
  "count_policy_ratio -> count_ppo",
  "count_ppo -> count_actor_gradient",
  "lexicographic_policy_output -> count_policy_ratio",
  "count_actor_gradient -> lexicographic_policy_output",
  "lexicographic_policy_output -> lexicographic_log_softmax",
  "residual_input_norm -> lex_feature_concat",
  "lex_feature_concat -> lex_semantic_side_stack",
  "critic_losses -> utility_critic",
  "dual_loss -> document_dual",
]) {
  assert.ok(dot.includes(edge), `DOT source must contain edge ${edge}`);
}
for (const countSourceCopy of [
  "Prototype reward source",
  "temporary exact level_count lookup",
  "Final reward source",
  "frozen k-anonymity estimator",
  "not a policy input",
  "User setting λk",
  "λ0 forces rψ = 0",
  "τ(λk) sets the utility floor",
  "πlex(a | s, λk)",
]) {
  assert.ok(report.includes(countSourceCopy), `count flow must explain ${countSourceCopy}`);
}
assert.match(
  report,
  /data-module-id="count-training-only"[^>]*style="justify-self:center;width:max-content"[\s\S]*?grid-template-columns:repeat\(4,max-content\)/,
  "the extracted count actor-critic must use four intrinsic-width columns",
);
for (const intrinsicLayout of [
  'ranker-model-diagram__layout" style="justify-items:center"',
  'data-module-id="encoder" style="justify-self:center;width:max-content"',
  'grid-template-columns:max-content max-content;gap:var(--dg-gap-lg);justify-self:center;width:max-content',
  'data-module-id="training-only" style="justify-self:center;width:max-content"',
]) {
  assert.ok(
    report.includes(intrinsicLayout),
    `M1 must size major sections from their content: ${intrinsicLayout}`,
  );
}
assert.ok(
  !report.includes("grid-template-columns:minmax(0,2fr) minmax(320px,1fr)"),
  "M1 must not inflate intrinsic content through a fractional top-level ratio",
);
assert.ok(
  report.indexOf("Training-only actor-critic")
    < report.indexOf("Training-only actor-critic and document dual"),
  "the count-only actor-critic section must precede the remaining training section",
);
for (const forbiddenCrossSectionEdge of [
  "feature_concat -> utility_critic",
  "lex_feature_concat -> count_critic",
  "lexicographic_policy_output -> policy_ratio",
  "actor_objective -> lexicographic_policy_output",
]) {
  assert.ok(
    !dot.includes(forbiddenCrossSectionEdge),
    `the remaining training section must be disconnected from the model graph: ${forbiddenCrossSectionEdge}`,
  );
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
