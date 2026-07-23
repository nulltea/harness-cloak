(() => {
  const diagram = document.getElementById("ranker-model-diagram");
  if (!diagram) return;

  const source = diagram.querySelector("[data-ranker-model-dot]");
  const graph = diagram.querySelector(".ranker-model-diagram__graph");
  const status = diagram.querySelector(".ranker-model-diagram__status");

  function fail(message, error) {
    if (status) {
      status.hidden = false;
      status.textContent = message;
    }
    if (error) console.error(message, error);
  }

  if (!source || !graph) {
    fail("Diagram source or output container is missing.");
    return;
  }
  if (!globalThis.Viz?.instance) {
    fail("Graphviz runtime failed to load.");
    return;
  }

  function clusterPath(cluster) {
    return Array.from(cluster?.children ?? []).find(
      (child) => child.tagName?.toLowerCase() === "path",
    );
  }

  function setRoundedRect(path, { x, y, width, height }, radius = 12) {
    const right = x + width;
    const bottom = y + height;
    const corner = Math.min(radius, width / 2, height / 2);
    path.setAttribute(
      "d",
      [
        `M${x + corner},${y}`,
        `H${right - corner}`,
        `C${right - corner / 2},${y} ${right},${y + corner / 2} ${right},${y + corner}`,
        `V${bottom - corner}`,
        `C${right},${bottom - corner / 2} ${right - corner / 2},${bottom} ${right - corner},${bottom}`,
        `H${x + corner}`,
        `C${x + corner / 2},${bottom} ${x},${bottom - corner / 2} ${x},${bottom - corner}`,
        `V${y + corner}`,
        `C${x},${y + corner / 2} ${x + corner / 2},${y} ${x + corner},${y}`,
        "Z",
      ].join(" "),
    );
  }

  function shiftClusterLabel(cluster, deltaX) {
    Array.from(cluster.children)
      .filter((child) => child.tagName?.toLowerCase() === "text")
      .forEach((label) => {
        const currentX = Number(label.getAttribute("x"));
        if (Number.isFinite(currentX)) {
          label.setAttribute("x", String(currentX + deltaX));
        }
      });
  }

  function snapSectionGrid(svg) {
    const top = svg.querySelector(".grid-top");
    const left = svg.querySelector(".grid-left");
    const rightTop = svg.querySelector(".grid-right-top");
    const rightBottom = svg.querySelector(".grid-right-bottom");
    const paths = {
      top: clusterPath(top),
      left: clusterPath(left),
      rightTop: clusterPath(rightTop),
      rightBottom: clusterPath(rightBottom),
    };
    if (Object.values(paths).some((path) => !path)) return;

    const boxes = Object.fromEntries(
      Object.entries(paths).map(([name, path]) => [name, path.getBBox()]),
    );
    const gutter = boxes.left.y - (boxes.top.y + boxes.top.height);
    if (!(gutter > 0)) return;

    const rowLeft = boxes.left.x;
    const rowRight = boxes.top.x + boxes.top.width;
    const rowTop = boxes.left.y;
    const rowBottom = boxes.rightBottom.y + boxes.rightBottom.height;
    const rightLeft = boxes.left.x + boxes.left.width + gutter;
    const rightWidth = rowRight - rightLeft;
    const privacyBottom = boxes.rightTop.y + boxes.rightTop.height;

    const targets = {
      top: {
        x: rowLeft,
        y: boxes.top.y,
        width: rowRight - rowLeft,
        height: rowTop - gutter - boxes.top.y,
      },
      left: {
        x: rowLeft,
        y: rowTop,
        width: boxes.left.width,
        height: rowBottom - rowTop,
      },
      rightTop: {
        x: rightLeft,
        y: rowTop,
        width: rightWidth,
        height: privacyBottom - rowTop,
      },
      rightBottom: {
        x: rightLeft,
        y: privacyBottom + gutter,
        width: rightWidth,
        height: rowBottom - privacyBottom - gutter,
      },
    };

    for (const [name, path] of Object.entries(paths)) {
      const cluster = { top, left, rightTop, rightBottom }[name];
      shiftClusterLabel(cluster, targets[name].x - boxes[name].x);
      setRoundedRect(path, targets[name]);
    }
  }

  globalThis.Viz.instance()
    .then((viz) => {
      const svg = viz.renderSVGElement(source.textContent, {
        engine: "dot",
      });
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.setAttribute("role", "presentation");
      svg.setAttribute("focusable", "false");
      svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
      graph.replaceChildren(svg);
      snapSectionGrid(svg);
      if (status) status.hidden = true;
      diagram.dataset.renderer = "graphviz";
    })
    .catch((error) => fail("Graphviz could not render this diagram.", error));
})();
