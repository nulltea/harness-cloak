import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const reportPath = "docs/html/interactive-ranker-v2.html";
const rendererPath = "docs/html/js/ranker-model-diagram.js";
const stylesheetPath = "docs/html/css/site.css";
const report = readFileSync(reportPath, "utf8");
const renderer = readFileSync(rendererPath, "utf8");
const stylesheet = readFileSync(stylesheetPath, "utf8");
const committedReport = execFileSync(
  "git",
  ["show", `HEAD:${reportPath}`],
  { encoding: "utf8" },
);

assert.match(
  report,
  /id="ranker-model-diagram"[^>]*class="ranker-model-diagram"/,
  "M1 must keep its isolated figure container",
);
assert.equal(
  (report.match(/<svg\b/g) ?? []).length,
  3,
  "M1 SVG must be generated at runtime; the three downstream figures remain inline",
);
assert.ok(
  report.indexOf("@viz-js/viz@3.28.0/dist/viz-global.js")
    < report.indexOf("js/ranker-model-diagram.js"),
  "the pinned Viz.js runtime must load before the generic renderer",
);

const dotMatch = report.match(
  /<script type="text\/vnd\.graphviz" data-ranker-model-dot>([\s\S]*?)<\/script>/,
);
assert.ok(dotMatch, "M1 must keep editable DOT source beside the figure");
const dot = dotMatch[1];
for (const fragment of [
  "digraph RankerV2",
  "rankdir=TB",
  "newrank=true",
  "subgraph cluster_encoder",
  "subgraph cluster_utility",
  "subgraph cluster_right_column",
  "subgraph cluster_privacy",
  "subgraph cluster_controller",
  "grid-top",
  "grid-left",
  "grid-right-top",
  "grid-right-bottom",
  "clinical_encoder",
  "document_context",
  "utility_logit",
  "privacy_score",
  "additive_controller",
  "policy_output",
  "utility_relation -> context_interaction",
  "document_context -> context_interaction",
  "lambda_transform -> privacy_control",
  "alpha -> lambda_transform",
]) {
  assert.ok(dot.includes(fragment), `DOT source must contain ${fragment}`);
}
assert.ok(
  (dot.match(/\bclass="node-/g) ?? []).length >= 40,
  "M1 must declare its complete architecture directly in DOT",
);
assert.ok(
  (dot.match(/subgraph cluster_/g) ?? []).length >= 6,
  "M1 must retain its semantic module hierarchy as Graphviz clusters",
);
assert.ok(
  (dot.match(/rank\s*=\s*same/g) ?? []).length <= 15,
  "M1 must not require excessive manual rank constraints",
);
assert.doesNotMatch(dot, /\bpos\s*=/, "M1 must not hardcode node coordinates");

for (const requiredFragment of [
  "Viz.instance",
  "renderSVGElement",
  "data-ranker-model-dot",
  "replaceChildren",
  "snapSectionGrid",
  "setRoundedRect",
]) {
  assert.ok(renderer.includes(requiredFragment), `renderer must contain ${requiredFragment}`);
}
for (const forbiddenFragment of [
  "nodeSpecs",
  "edgeSpecs",
  "busSpecs",
  "getBoundingClientRect",
  "clinical_encoder",
  "utility_relation",
  "privacy_control",
]) {
  assert.ok(
    !renderer.includes(forbiddenFragment),
    `generic renderer must not contain diagram-specific ${forbiddenFragment}`,
  );
}

assert.match(
  stylesheet,
  /\.ranker-model-diagram__graph\s+svg\s*\{/,
  "stylesheet must size the generated SVG through one generic graph container",
);

const unchangedBoundary = '<section id="preliminaries">';
assert.equal(
  report.slice(report.indexOf(unchangedBoundary)),
  committedReport.slice(committedReport.indexOf(unchangedBoundary)),
  "Figures T1, R1, and I1 and their surrounding report content must remain unchanged",
);

console.log("M1 Graphviz source, generic renderer, and downstream boundary are valid.");
