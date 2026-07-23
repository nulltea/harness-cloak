(() => {
  const diagram = document.getElementById("ranker-model-diagram");
  if (!diagram) return;

  const nodeSpecs = [
    { id: "document-chunks", kind: "input", title: "Document chunks", detail: "512 tokens · 64-token overlap" },
    { id: "relation-sequence", kind: "input", title: "TYPE · SOURCE · CANDIDATE", detail: "ordered relation sequence" },
    { id: "clinical-encoder", kind: "encoder", title: "BioClinical ModernBERT-base", detail: "frozen shared encoder · h = 768 · bidirectional token states" },
    { id: "document-bank", kind: "tensor", title: "H_doc", detail: "token bank [T × 768]" },
    { id: "field-means", kind: "tensor", title: "Field-aware means", detail: "type · source · candidate [768]" },
    { id: "pair-composition", kind: "process", title: "Pair composition", detail: "concat · difference · product" },
    { id: "relation-pair", kind: "tensor", title: "r_pair", detail: "[A × 3840]" },

    { id: "document-projection", kind: "linear", title: "Linear", detail: "H_doc 768 → 768" },
    { id: "role-embedding", kind: "embedding", title: "E_role", detail: "3 × 768" },
    { id: "relative-embedding", kind: "embedding", title: "E_relative", detail: "17 × 768" },
    { id: "position-embedding", kind: "embedding", title: "E_doc-position", detail: "16 × 768" },
    { id: "token-sum", kind: "operator", title: "+", detail: "feature sum" },
    { id: "augmented-tokens", kind: "tensor", title: "Augmented tokens", detail: "H_doc + M_j" },

    { id: "utility-projection", kind: "linear", title: "Utility projection", detail: "Linear 3840 → 768" },
    { id: "utility-relation", kind: "tensor", title: "r_U", detail: "[A × 768]" },
    { id: "query-projection", kind: "linear", title: "Query projection", detail: "Linear 768 → 768" },
    { id: "target-attention", kind: "attention", title: "Target attention", detail: "MHA · 8 heads · occurrence summaries + E_occ-pos" },
    { id: "local-attention", kind: "attention", title: "Local attention", detail: "MHA · 8 heads · target-containing chunks" },
    { id: "global-attention", kind: "attention", title: "Global attention", detail: "MHA · 8 heads · complete token bank" },
    { id: "context-merge", kind: "operator", title: "⊕", detail: "concatenate" },
    { id: "context-projection", kind: "linear", title: "Context projection", detail: "Linear 2304 → 768" },
    { id: "document-context", kind: "tensor", title: "c_context", detail: "[A × 768]" },

    { id: "earlier-selections", kind: "input", title: "Earlier selections", detail: "r_U + mode/type/position embeddings" },
    { id: "memory-kv", kind: "linear", title: "Memory KV", detail: "Linear 3072 → 768" },
    { id: "memory-query-join", kind: "operator", title: "⊕", detail: "[r_U ; c]" },
    { id: "memory-query", kind: "linear", title: "Memory query", detail: "Linear 1536 → 768" },
    { id: "history-attention", kind: "attention", title: "Selected-action cross-attention", detail: "MHA · 8 heads · output h_hist [A × 768]" },

    { id: "mode-embedding", kind: "embedding", title: "E_mode", detail: "3 × 768" },
    { id: "type-embedding", kind: "embedding", title: "E_runtime-type", detail: "N_type × 768" },
    { id: "context-interaction", kind: "operator", title: "⊙", detail: "r_U × Linear(c)" },
    { id: "feature-concat", kind: "process", title: "Feature concat", detail: "r_U · c_context · interaction · h_hist · mode · type" },
    { id: "utility-hidden", kind: "linear", title: "4608 → 768" },
    { id: "utility-gelu", kind: "activation", title: "GELU" },
    { id: "utility-output", kind: "linear", title: "768 → 1" },
    { id: "utility-logit", kind: "tensor", title: "u(a)", detail: "utility logit" },

    { id: "count-basis", kind: "embedding", title: "count-basis token", detail: "optional categorical B" },
    { id: "privacy-join", kind: "operator", title: "⊕", detail: "[r_pair ; B]" },
    { id: "privacy-projection", kind: "linear", title: "Privacy projection", detail: "Linear (3840 + B) → 256" },
    { id: "privacy-hidden", kind: "linear", title: "Hidden projection", detail: "Linear 256 → 256" },
    { id: "privacy-gelu", kind: "activation", title: "GELU" },
    { id: "distribution-projection", kind: "linear", title: "Distribution projection", detail: "Linear 256 → 2" },
    { id: "positive-parameters", kind: "activation", title: "Softplus", detail: "positive μ_logK and σ_logK" },
    { id: "log-count", kind: "tensor", title: "Log-count distribution", detail: "μ_logK · σ_logK · σ retained for audit" },
    { id: "profile-normalization", kind: "normalization", title: "Profile-menu normalization", detail: "μ / max(level μ) · KEEP = 0 · placeholder = 1" },
    { id: "privacy-score", kind: "tensor", title: "p̂(a)", detail: "profile-relative" },

    { id: "alpha", kind: "controller", title: "α", detail: "softplus(alpha_raw)" },
    { id: "lambda-transform", kind: "controller", title: "g(λ)", detail: "fixed lambda transform" },
    { id: "privacy-control", kind: "controller", title: "p̂", detail: "predicted privacy" },
    { id: "additive-controller", kind: "operator", title: "+", detail: "combined z" },
    { id: "legal-gather", kind: "process", title: "Legal-action gather" },
    { id: "log-softmax", kind: "normalization", title: "log_softmax" },
    { id: "policy-output", kind: "output", title: "π(a | s, λ)" },
  ];
  const nodeKinds = new Map(nodeSpecs.map(({ id, kind }) => [id, kind]));
  const noIncomingArrowTargets = new Set([
    ...nodeSpecs.filter(({ kind }) => kind === "operator" || kind === "controller").map(({ id }) => id),
    "feature-concat",
    "pair-composition",
  ]);

  const edgeSpecs = [
    ["document-chunks", "clinical-encoder"],
    ["relation-sequence", "clinical-encoder"],
    ["clinical-encoder", "document-bank"],
    ["clinical-encoder", "field-means"],
    ["field-means", "pair-composition"],
    ["pair-composition", "relation-pair"],
    ["document-bank", "document-projection", "representation"],
    ["document-projection", "token-sum"],
    ["token-sum", "augmented-tokens"],
    ["relation-pair", "utility-projection", "representation"],
    ["utility-projection", "utility-relation"],
    ["utility-relation", "query-projection"],
    ["context-merge", "context-projection"],
    ["context-projection", "document-context"],
    ["earlier-selections", "memory-kv"],
    ["utility-relation", "memory-query-join"],
    ["document-context", "memory-query-join"],
    ["memory-query-join", "memory-query"],
    ["memory-kv", "history-attention"],
    ["memory-query", "history-attention"],
    ["utility-relation", "context-interaction"],
    ["document-context", "context-interaction"],
    ["context-interaction", "feature-concat"],
    ["history-attention", "feature-concat"],
    ["feature-concat", "utility-hidden"],
    ["utility-hidden", "utility-gelu"],
    ["utility-gelu", "utility-output"],
    ["utility-output", "utility-logit"],
    ["relation-pair", "privacy-join", "representation"],
    ["count-basis", "privacy-join"],
    ["privacy-join", "privacy-projection"],
    ["privacy-projection", "privacy-hidden"],
    ["privacy-hidden", "privacy-gelu"],
    ["privacy-gelu", "distribution-projection"],
    ["distribution-projection", "positive-parameters"],
    ["positive-parameters", "log-count"],
    ["log-count", "profile-normalization"],
    ["profile-normalization", "privacy-score"],
    ["privacy-score", "privacy-control", "privacy"],
    ["privacy-control", "lambda-transform", "privacy"],
    ["lambda-transform", "alpha", "privacy"],
    ["alpha", "additive-controller", "privacy"],
    ["utility-logit", "additive-controller"],
    ["additive-controller", "legal-gather"],
    ["legal-gather", "log-softmax"],
    ["log-softmax", "policy-output"],
  ].map(([source, target, kind = "default"], index) => ({
    id: `model-edge-${index}`,
    source,
    target,
    kind,
  }));

  const busSpecs = [
    {
      id: "embedding-sum-bus",
      mode: "merge",
      sources: ["role-embedding", "relative-embedding", "position-embedding"],
      target: "token-sum",
      side: "above",
      gap: 8,
    },
    {
      id: "attention-input-bus",
      mode: "bridge",
      sources: ["augmented-tokens", "query-projection"],
      targets: ["target-attention", "local-attention", "global-attention"],
      side: "above",
      gap: 10,
      targetPort: "top",
    },
    {
      id: "attention-context-bus",
      mode: "merge",
      sources: ["target-attention", "local-attention", "global-attention"],
      target: "context-merge",
      side: "above",
      gap: 8,
    },
    {
      id: "feature-input-bus",
      mode: "merge",
      sources: ["mode-embedding", "type-embedding"],
      target: "feature-concat",
      side: "below",
      gap: 10,
      targetPort: "bottom:0.28",
    },
  ];

  const viewport = diagram.querySelector(".ranker-model-diagram__viewport");
  const groupsLayer = diagram.querySelector(".ranker-model-diagram__groups");
  const nodesLayer = diagram.querySelector(".ranker-model-diagram__nodes");
  const edgesLayer = diagram.querySelector(".ranker-model-diagram__edges");
  const edgePathsLayer = diagram.querySelector(".ranker-model-diagram__edge-paths");
  const status = diagram.querySelector(".ranker-model-diagram__status");
  const elements = new Map();
  let layoutElement;
  let scheduledFrame;

  function createNode(spec) {
    const element = document.createElement("div");
    element.className = `ranker-model-diagram__node ranker-model-diagram__node--${spec.kind}`;
    element.dataset.nodeId = spec.id;
    const title = document.createElement("div");
    title.className = "ranker-model-diagram__node-title";
    title.textContent = spec.title;
    element.append(title);
    if (spec.detail) {
      const detail = document.createElement("div");
      detail.className = "ranker-model-diagram__node-detail";
      detail.textContent = spec.detail;
      element.append(detail);
    }
    elements.set(spec.id, element);
    return element;
  }

  function node(id) {
    const element = elements.get(id);
    if (!element) throw new Error(`Unknown diagram node: ${id}`);
    return element;
  }

  function stack(className, ids) {
    const element = document.createElement("div");
    element.className = className;
    element.append(...ids.map(node));
    return element;
  }

  function module(id, label, kind = "primary") {
    const element = document.createElement("section");
    element.className = `ranker-model-diagram__module ranker-model-diagram__module--${kind} ranker-model-diagram__module--${id}`;
    element.dataset.moduleId = id;
    const heading = document.createElement("div");
    heading.className = "ranker-model-diagram__module-label";
    heading.textContent = label;
    element.append(heading);
    return element;
  }

  function buildLayout() {
    for (const spec of nodeSpecs) createNode(spec);
    groupsLayer.replaceChildren();
    const layout = document.createElement("div");
    layout.className = "ranker-model-diagram__layout";

    const encoder = module("encoder", "Shared frozen encoder");
    const encoderGrid = document.createElement("div");
    encoderGrid.className = "ranker-model-diagram__encoder-grid";
    encoderGrid.append(
      stack("ranker-model-diagram__encoder-inputs", ["document-chunks", "relation-sequence"]),
      node("clinical-encoder"),
      stack("ranker-model-diagram__encoder-outputs", ["document-bank", "field-means"]),
      stack("ranker-model-diagram__encoder-pair", ["pair-composition", "relation-pair"]),
    );
    encoder.append(encoderGrid);

    const utility = module("utility", "Utility scoring");
    const utilityGrid = document.createElement("div");
    utilityGrid.className = "ranker-model-diagram__utility-grid";
    const utilityTop = document.createElement("div");
    utilityTop.className = "ranker-model-diagram__utility-top";
    const tokenFusion = module("token-fusion", "Token feature fusion", "secondary");
    const tokenGrid = document.createElement("div");
    tokenGrid.className = "ranker-model-diagram__token-grid";
    tokenGrid.append(
      node("document-projection"),
      stack("ranker-model-diagram__embedding-row", ["role-embedding", "relative-embedding", "position-embedding"]),
      stack("ranker-model-diagram__token-output-row", ["token-sum", "augmented-tokens"]),
    );
    tokenFusion.append(tokenGrid);
    utilityTop.append(
      tokenFusion,
      stack("ranker-model-diagram__utility-relation", ["utility-projection", "utility-relation", "query-projection"]),
    );
    const attentions = stack("ranker-model-diagram__attention-row", ["target-attention", "local-attention", "global-attention"]);
    const context = stack("ranker-model-diagram__context-stack", ["context-merge", "context-projection", "document-context"]);
    const memory = module("selected-memory", "Selected-action memory", "secondary");
    const memoryTop = document.createElement("div");
    memoryTop.className = "ranker-model-diagram__memory-top";
    memoryTop.append(
      stack("ranker-model-diagram__memory-lane", ["earlier-selections", "memory-kv"]),
      stack("ranker-model-diagram__memory-lane", ["memory-query-join", "memory-query"]),
    );
    memory.append(memoryTop, node("history-attention"));
    const outputRow = document.createElement("div");
    outputRow.className = "ranker-model-diagram__output-row";
    const utilityMlp = module("utility-mlp", "Utility MLP", "secondary");
    utilityMlp.append(stack("ranker-model-diagram__mlp-stack", ["utility-hidden", "utility-gelu", "utility-output"]));
    const utilityMlpColumn = document.createElement("div");
    utilityMlpColumn.className = "ranker-model-diagram__utility-mlp-column";
    utilityMlpColumn.append(utilityMlp, node("utility-logit"));
    outputRow.append(
      node("mode-embedding"),
      node("type-embedding"),
      node("context-interaction"),
      node("feature-concat"),
      utilityMlpColumn,
    );
    utilityGrid.append(utilityTop, attentions, context, memory, outputRow);
    utility.append(utilityGrid);

    const privacy = module("privacy", "Semantic privacy estimation");
    privacy.append(stack("ranker-model-diagram__privacy-stack", [
      "count-basis", "privacy-join", "privacy-projection", "privacy-hidden",
      "privacy-gelu", "distribution-projection", "positive-parameters",
      "log-count", "profile-normalization", "privacy-score",
    ]));
    const mainGrid = document.createElement("div");
    mainGrid.className = "ranker-model-diagram__main-grid";
    mainGrid.append(utility, privacy);

    const controller = module("controller", "Additive lambda controller");
    const controllerGrid = document.createElement("div");
    controllerGrid.className = "ranker-model-diagram__controller-grid";
    const controllerInputs = stack("ranker-model-diagram__controller-inputs", ["alpha", "lambda-transform", "privacy-control"]);
    const controllerFormula = document.createElement("div");
    controllerFormula.className = "ranker-model-diagram__controller-formula";
    controllerFormula.textContent = "b(a, λ) = α · g(λ) · p̂(a)";
    controllerInputs.append(controllerFormula);
    controllerGrid.append(
      stack("ranker-model-diagram__controller-output", ["additive-controller", "legal-gather", "log-softmax", "policy-output"]),
      controllerInputs,
    );
    controller.append(controllerGrid);
    layout.append(encoder, mainGrid, controller);
    nodesLayer.replaceChildren(layout);
    diagram.dataset.layout = "structured";
    return layout;
  }

  function relativeRect(element) {
    const viewportRect = viewport.getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    return {
      left: rect.left - viewportRect.left,
      right: rect.right - viewportRect.left,
      top: rect.top - viewportRect.top,
      bottom: rect.bottom - viewportRect.top,
      width: rect.width,
      height: rect.height,
      centerX: rect.left - viewportRect.left + rect.width / 2,
      centerY: rect.top - viewportRect.top + rect.height / 2,
    };
  }

  function anchor(rect, port) {
    const [side, rawRatio] = port.split(":");
    const ratio = rawRatio === undefined ? 0.5 : Number(rawRatio);
    if (side === "left") return { x: rect.left, y: rect.top + rect.height * ratio };
    if (side === "right") return { x: rect.right, y: rect.top + rect.height * ratio };
    if (side === "top") return { x: rect.left + rect.width * ratio, y: rect.top };
    return { x: rect.left + rect.width * ratio, y: rect.bottom };
  }

  const portOverrides = new Map([
    ["document-chunks>clinical-encoder", ["right", "left"]],
    ["relation-sequence>clinical-encoder", ["right", "left"]],
    ["clinical-encoder>document-bank", ["right", "left"]],
    ["clinical-encoder>field-means", ["right", "left"]],
    ["field-means>pair-composition", ["right", "left"]],
    ["pair-composition>relation-pair", ["bottom", "top"]],
    ["privacy-control>lambda-transform", ["left", "right"]],
    ["lambda-transform>alpha", ["left", "right"]],
    ["alpha>additive-controller", ["left", "right"]],
    ["memory-kv>history-attention", ["bottom", "top:0.36"]],
    ["memory-query>history-attention", ["bottom", "top:0.64"]],
  ]);

  function choosePorts(edge, sourceRect, targetRect) {
    const override = portOverrides.get(`${edge.source}>${edge.target}`);
    if (override) return override;
    const dx = targetRect.centerX - sourceRect.centerX;
    const dy = targetRect.centerY - sourceRect.centerY;
    if (Math.abs(dx) > Math.abs(dy) * 1.15) {
      return dx >= 0 ? ["right", "left"] : ["left", "right"];
    }
    return dy >= 0 ? ["bottom", "top"] : ["top", "bottom"];
  }

  function route(edge) {
    const sourceRect = connectionRect(node(edge.source));
    const targetRect = connectionRect(node(edge.target));
    if (edge.source === "document-projection" && edge.target === "token-sum") {
      const moduleRect = relativeRect(layoutElement.querySelector('[data-module-id="token-fusion"]'));
      const start = anchor(sourceRect, "left");
      const end = anchor(targetRect, "left");
      const laneX = moduleRect.left + 9;
      return `M${start.x} ${start.y}L${laneX} ${start.y}L${laneX} ${end.y}L${end.x} ${end.y}`;
    }
    if (edge.source === "document-bank" && edge.target === "document-projection") {
      const encoderRect = relativeRect(layoutElement.querySelector('[data-module-id="encoder"]'));
      const pairRect = connectionRect(node("pair-composition"));
      const start = anchor(sourceRect, "right");
      const end = anchor(targetRect, "top:0.75");
      const laneX = sourceRect.right + (pairRect.left - sourceRect.right) / 2;
      const laneY = encoderRect.bottom + 8;
      return `M${start.x} ${start.y}L${laneX} ${start.y}L${laneX} ${laneY}L${end.x} ${laneY}L${end.x} ${end.y}`;
    }
    if (edge.source === "relation-pair" && edge.target === "utility-projection") {
      const encoderRect = relativeRect(layoutElement.querySelector('[data-module-id="encoder"]'));
      const start = anchor(sourceRect, "bottom");
      const end = anchor(targetRect, "top");
      const laneY = encoderRect.bottom + 16;
      return `M${start.x} ${start.y}L${start.x} ${laneY}L${end.x} ${laneY}L${end.x} ${end.y}`;
    }
    if (edge.source === "relation-pair" && edge.target === "privacy-join") {
      const privacyRect = relativeRect(layoutElement.querySelector('[data-module-id="privacy"]'));
      const start = anchor(sourceRect, "bottom:0.75");
      const end = anchor(targetRect, "right");
      const laneX = privacyRect.right - 10;
      return `M${start.x} ${start.y}L${laneX} ${start.y}L${laneX} ${end.y}L${end.x} ${end.y}`;
    }
    if (edge.target === "context-interaction") {
      const utilityRect = relativeRect(layoutElement.querySelector('[data-module-id="utility"]'));
      const memoryRect = relativeRect(layoutElement.querySelector('[data-module-id="selected-memory"]'));
      const fromUtilityRelation = edge.source === "utility-relation";
      const start = anchor(sourceRect, fromUtilityRelation ? "right" : "left");
      const end = anchor(targetRect, "top");
      const laneX = fromUtilityRelation ? utilityRect.right - 12 : utilityRect.left + 12;
      const laneY = memoryRect.bottom + 10;
      return `M${start.x} ${start.y}L${laneX} ${start.y}L${laneX} ${laneY}L${end.x} ${laneY}L${end.x} ${end.y}`;
    }
    if (edge.source === "utility-relation" && edge.target === "memory-query-join") {
      const utilityRect = relativeRect(layoutElement.querySelector('[data-module-id="utility"]'));
      const start = anchor(sourceRect, "right");
      const end = anchor(targetRect, "top");
      const laneX = utilityRect.right - 12;
      const laneY = targetRect.top - 10;
      return `M${start.x} ${start.y}L${laneX} ${start.y}L${laneX} ${laneY}L${end.x} ${laneY}L${end.x} ${end.y}`;
    }
    const [sourcePort, targetPort] = choosePorts(edge, sourceRect, targetRect);
    const start = anchor(sourceRect, sourcePort);
    const end = anchor(targetRect, targetPort);
    if (edge.source === "alpha" && edge.target === "additive-controller") {
      const markerPointX = start.x - 18;
      return `M${start.x} ${start.y}L${markerPointX} ${start.y}L${end.x} ${end.y}`;
    }
    if (isOperatorConnection(edge, sourceRect, targetRect)) return `M${start.x} ${start.y}L${end.x} ${end.y}`;
    const horizontal = sourcePort === "left" || sourcePort === "right";
    if (horizontal) {
      const middleX = start.x + (end.x - start.x) / 2;
      return `M${start.x} ${start.y}L${middleX} ${start.y}L${middleX} ${end.y}L${end.x} ${end.y}`;
    }
    const middleY = start.y + (end.y - start.y) / 2;
    return `M${start.x} ${start.y}L${start.x} ${middleY}L${end.x} ${middleY}L${end.x} ${end.y}`;
  }

  function connectionRect(element) {
    const visibleShape = element.classList.contains("ranker-model-diagram__node--operator")
      ? element.querySelector(".ranker-model-diagram__node-title")
      : element;
    return relativeRect(visibleShape);
  }

  function isOperatorConnection(edge, sourceRect, targetRect) {
    const involvesOperator = nodeKinds.get(edge.source) === "operator" || nodeKinds.get(edge.target) === "operator";
    const operatorDistance = Math.hypot(
      targetRect.centerX - sourceRect.centerX,
      targetRect.centerY - sourceRect.centerY,
    );
    return involvesOperator && operatorDistance <= 360;
  }

  function createPath(pathData, kind = "default", hasArrow = true) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathData);
    path.setAttribute(
      "class",
      `ranker-model-diagram__edge ranker-model-diagram__edge--${kind}${hasArrow ? "" : " ranker-model-diagram__edge--no-arrow"}`,
    );
    path.dataset.hasArrow = String(hasArrow);
    return path;
  }

  function renderMergeBus(spec) {
    const sourcePoints = spec.sources.map((source) => (
      anchor(connectionRect(node(source)), spec.side === "below" ? "bottom" : "bottom")
    ));
    const targetRect = connectionRect(node(spec.target));
    const targetPort = spec.targetPort ?? (spec.side === "below" ? "bottom" : "top");
    const targetPoint = anchor(targetRect, targetPort);
    const laneY = spec.side === "below"
      ? Math.max(targetRect.bottom, ...spec.sources.map((source) => connectionRect(node(source)).bottom)) + spec.gap
      : targetRect.top - spec.gap;
    const xs = [...sourcePoints.map(({ x }) => x), targetPoint.x];
    const branches = sourcePoints.map(({ x, y }) => `M${x} ${y}L${x} ${laneY}`).join("");
    const bus = `M${Math.min(...xs)} ${laneY}L${Math.max(...xs)} ${laneY}`;
    const trunk = `M${targetPoint.x} ${laneY}L${targetPoint.x} ${targetPoint.y}`;
    return [
      createPath(`${branches}${bus}`, spec.kind, false),
      createPath(trunk, spec.kind, !noIncomingArrowTargets.has(spec.target)),
    ];
  }

  function renderBridgeBus(spec) {
    const sourcePoints = spec.sources.map((source) => anchor(connectionRect(node(source)), "bottom"));
    const targetPoints = spec.targets.map((target) => (
      anchor(connectionRect(node(target)), spec.targetPort ?? "top")
    ));
    const laneY = Math.min(...targetPoints.map(({ y }) => y)) - spec.gap;
    const xs = [...sourcePoints.map(({ x }) => x), ...targetPoints.map(({ x }) => x)];
    const sourceBranches = sourcePoints.map(({ x, y }) => `M${x} ${y}L${x} ${laneY}`).join("");
    const bus = `M${Math.min(...xs)} ${laneY}L${Math.max(...xs)} ${laneY}`;
    const targetBranches = targetPoints.map(({ x, y }) => (
      createPath(`M${x} ${laneY}L${x} ${y}`, spec.kind, true)
    ));
    return [createPath(`${sourceBranches}${bus}`, spec.kind, false), ...targetBranches];
  }

  function renderBus(spec) {
    if (spec.mode === "bridge") return renderBridgeBus(spec);
    return renderMergeBus(spec);
  }

  function drawConnectors() {
    const width = Math.ceil(nodesLayer.getBoundingClientRect().width);
    const height = Math.ceil(layoutElement.getBoundingClientRect().height);
    viewport.style.height = `${height}px`;
    edgesLayer.setAttribute("viewBox", `0 0 ${width} ${height}`);
    edgesLayer.setAttribute("width", width);
    edgesLayer.setAttribute("height", height);
    const rendered = edgeSpecs.map((edge) => {
      const path = createPath(route(edge), edge.kind, !noIncomingArrowTargets.has(edge.target));
      path.dataset.edgeId = edge.id;
      path.dataset.source = edge.source;
      path.dataset.target = edge.target;
      if (edge.source === "alpha" && edge.target === "additive-controller") {
        path.classList.add("ranker-model-diagram__edge--mid-arrow");
      }
      return { edge, path };
    });
    const busPaths = busSpecs.flatMap((spec) => renderBus(spec).map((path) => {
      path.dataset.busId = spec.id;
      path.dataset.members = [...(spec.sources ?? [spec.source]), ...(spec.targets ?? [spec.target])].join(",");
      return path;
    }));
    edgePathsLayer.replaceChildren(...rendered.map(({ path }) => path), ...busPaths);
    status.hidden = true;
    diagram.dataset.ready = "true";
  }

  function scheduleConnectors() {
    cancelAnimationFrame(scheduledFrame);
    scheduledFrame = requestAnimationFrame(drawConnectors);
  }

  layoutElement = buildLayout();
  const observer = new ResizeObserver(scheduleConnectors);
  observer.observe(layoutElement);
  for (const element of elements.values()) observer.observe(element);
  window.addEventListener("resize", scheduleConnectors);
  Promise.resolve(document.fonts?.ready).then(scheduleConnectors);
})();
