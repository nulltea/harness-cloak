import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const reportPath = "docs/html/interactive-ranker-v2.html";
const rendererPath = "docs/html/js/ranker-model-diagram.js";
const stylesheetPath = "docs/html/css/site.css";
const report = readFileSync(reportPath, "utf8");
const committedReport = execFileSync(
  "git",
  ["show", `HEAD:${reportPath}`],
  { encoding: "utf8" },
);

assert.match(
  report,
  /id="ranker-model-diagram"[^>]*class="ranker-model-diagram"/,
  "M1 must use the DOM-first ranker model diagram container",
);
assert.doesNotMatch(report, /elk\.bundled\.js/, "M1 must not depend on global ELK layout");
assert.match(
  report,
  /ranker-model-diagram\.js/,
  "M1 must load its isolated renderer",
);
assert.equal(
  (report.match(/<svg\b/g) ?? []).length,
  4,
  "M1 may add one SVG connector layer; the other three diagrams remain inline SVG",
);
assert.match(
  report,
  /<svg class="ranker-model-diagram__edges"[^>]*aria-hidden="true"/,
  "M1 SVG must be a non-semantic connector layer",
);
assert.match(
  report,
  /<div class="ranker-model-diagram__nodes"><\/div>/,
  "M1 blocks must render in a dedicated HTML node layer",
);
assert.doesNotMatch(
  report,
  /ranker-model-diagram__edge-labels/,
  "M1 must not retain detached arrow-label overlays",
);

const renderer = readFileSync(rendererPath, "utf8");
const stylesheet = readFileSync(stylesheetPath, "utf8");
for (const requiredFragment of [
  "ResizeObserver",
  "getBoundingClientRect",
  "connectionRect",
  "ranker-model-diagram__node",
  "ranker-model-diagram__module",
  "noIncomingArrowTargets",
  "isOperatorConnection",
]) {
  assert.ok(
    renderer.includes(requiredFragment),
    `renderer must contain ${requiredFragment}`,
  );
}
assert.match(
  renderer,
  /\["privacy-control", "lambda-transform", "privacy"\][\s\S]*\["lambda-transform", "alpha", "privacy"\][\s\S]*\["alpha", "additive-controller", "privacy"\]/,
  "controller circles must sit directly on one continuous privacy-bias path",
);
assert.doesNotMatch(
  renderer,
  /\["(?:privacy-control|lambda-transform)", "additive-controller"/,
  "controller inputs must not use detached parallel arrows into the additive sum",
);
assert.doesNotMatch(renderer, /privacy bias/, "privacy-bias edge labels must be removed");
assert.doesNotMatch(renderer, /"default", "u\(a\)"/, "utility edge labels must be removed");
assert.ok(
  renderer.includes("ranker-model-diagram__edge--mid-arrow"),
  "the privacy chain must place its sole direction marker immediately after alpha",
);
assert.match(
  renderer,
  /function isOperatorConnection\(edge, sourceRect, targetRect\)[\s\S]{0,500}<= 360/,
  "only nearby operator links may use direct segments",
);
for (const source of ["utility-relation", "document-context"]) {
  assert.ok(
    renderer.includes(`["${source}", "context-interaction"]`),
    `${source} must visibly connect to the interaction operator`,
  );
}
assert.ok(
  renderer.includes('edge.target === "context-interaction"'),
  "interaction operands must use an obstacle-free shared route",
);
for (const busId of [
  "embedding-sum-bus",
  "attention-input-bus",
  "attention-context-bus",
  "feature-input-bus",
]) {
  assert.ok(renderer.includes(busId), `renderer must define ${busId}`);
}
assert.doesNotMatch(
  renderer,
  /\["(?:role-embedding|relative-embedding|position-embedding)", "token-sum"/,
  "embedding inputs must use one shared sum bus",
);
assert.doesNotMatch(
  renderer,
  /\["(?:augmented-tokens|query-projection)", "(?:target|local|global)-attention"/,
  "attention Q/KV inputs must use one shared input bus",
);
assert.ok(
  renderer.includes('edge.source === "utility-relation" && edge.target === "memory-query-join"'),
  "the long memory-query operand must route through the utility gutter",
);
for (const route of [
  'edge.source === "document-bank" && edge.target === "document-projection"',
  'edge.source === "relation-pair" && edge.target === "utility-projection"',
  'edge.source === "relation-pair" && edge.target === "privacy-join"',
]) {
  assert.ok(renderer.includes(route), `renderer must obstacle-route ${route}`);
}
assert.match(
  renderer,
  /edge\.source === "relation-pair" && edge\.target === "privacy-join"[\s\S]{0,500}anchor\(sourceRect, "bottom:0\.75"\)/,
  "r_pair privacy and utility branches must leave through distinct visible source ports",
);
assert.doesNotMatch(
  renderer,
  /\["(?:mode-embedding|type-embedding)", "feature-concat"/,
  "same-row feature inputs must route beneath the interaction node through a shared bus",
);
for (const requiredFragment of [
  "ranker-model-diagram__encoder-grid",
  "ranker-model-diagram__main-grid",
  "ranker-model-diagram__utility-grid",
  "ranker-model-diagram__controller-grid",
]) {
  assert.ok(
    stylesheet.includes(requiredFragment),
    `stylesheet must contain ${requiredFragment}`,
  );
}
assert.match(
  stylesheet,
  /\.ranker-model-diagram__node--linear::before[\s\S]*clip-path:/,
  "linear trapezoids must use an inset fill over a continuous polygon outline",
);
assert.match(
  stylesheet,
  /\.ranker-model-diagram__node--activation::before[\s\S]*clip-path:/,
  "activation diamonds must use an inset fill over a continuous polygon outline",
);
assert.match(
  stylesheet,
  /\.ranker-model-diagram__module \{[\s\S]*background: rgb\([^;]+\/ 0\.[4-9]/,
  "outlined modules need a visibly tinted translucent background",
);
assert.match(
  stylesheet,
  /\.ranker-model-diagram__edge--representation \{[\s\S]*stroke-width: 1\.[4-9]/,
  "shared-encoder representation arrows must remain clearly visible",
);
assert.doesNotMatch(
  stylesheet.match(/\.ranker-model-diagram__edge--representation \{[\s\S]*?\}/)?.[0] ?? "",
  /stroke-dasharray/,
  "shared-encoder representation arrows must not use a faint dashed treatment",
);
const edgeLayerRules = [...stylesheet.matchAll(/\.ranker-model-diagram__edges \{[\s\S]*?\}/g)].map(([rule]) => rule);
const nodeLayerRules = [...stylesheet.matchAll(/\.ranker-model-diagram__nodes \{[\s\S]*?\}/g)].map(([rule]) => rule);
assert.ok(edgeLayerRules.some((rule) => /z-index: 3/.test(rule)), "connectors must render above colored nodes");
assert.ok(nodeLayerRules.some((rule) => /z-index: 2/.test(rule)), "HTML nodes must stay below the connector layer");
assert.ok(
  stylesheet.lastIndexOf(".ranker-model-diagram__edge--no-arrow")
    > stylesheet.lastIndexOf(".ranker-model-diagram__edge--privacy"),
  "markerless pass-through styling must override privacy-edge markers",
);
assert.match(
  stylesheet,
  /\.ranker-model-diagram__edge--mid-arrow \{[\s\S]*marker-mid: url\("#ranker-model-arrow-privacy"\)/,
  "alpha output must use a mid-edge privacy marker without adding an incoming marker to the sum",
);

const unchangedBoundary = '<section id="preliminaries">';
assert.equal(
  report.slice(report.indexOf(unchangedBoundary)),
  committedReport.slice(committedReport.indexOf(unchangedBoundary)),
  "Figures T1, R1, and I1 and their surrounding report content must remain unchanged",
);

console.log("M1 structured DOM layout and downstream diagram boundary are valid.");
