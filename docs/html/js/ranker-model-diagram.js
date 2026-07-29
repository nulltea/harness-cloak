/* Generic edge router for HTML-grid diagrams: CSS grid owns node placement
   and styling; Graphviz (neato with pinned positions, splines=ortho) routes
   only the edges declared in the page's DOT block, treating every
   [data-node-id] box as an obstacle. Knows no diagram-specific node or edge
   names. */
(() => {
  const diagram = document.getElementById("ranker-model-diagram");
  if (!diagram) return;

  const source = diagram.querySelector("[data-ranker-model-dot]");
  const layout = diagram.querySelector(".ranker-model-diagram__layout");
  const overlay = diagram.querySelector(".ranker-model-diagram__edges");
  const status = diagram.querySelector(".ranker-model-diagram__status");

  function fail(message, error) {
    if (status) {
      status.hidden = false;
      status.textContent = message;
    }
    if (error) console.error(message, error);
  }

  if (!source || !layout || !overlay) {
    fail("Diagram source, layout, or overlay container is missing.");
    return;
  }
  if (!globalThis.Viz?.instance) {
    fail("Graphviz runtime failed to load.");
    return;
  }

  const dotText = source.textContent;
  const dotBody = dotText.slice(
    dotText.indexOf("{") + 1,
    dotText.lastIndexOf("}"),
  );

  function measureNodes() {
    // The layout div is the overlay's coordinate origin; a plain div rect is
    // reliable everywhere, unlike an auto-sized <svg> rect in Firefox.
    const origin = layout.getBoundingClientRect();
    return Array.from(layout.querySelectorAll("[data-node-id]"), (element) => {
      const rect = element.getBoundingClientRect();
      return {
        id: element.dataset.nodeId,
        shape: element.dataset.portShape === "circle" ? "circle" : "box",
        cx: rect.left - origin.left + rect.width / 2,
        cy: rect.top - origin.top + rect.height / 2,
        width: rect.width,
        height: rect.height,
      };
    });
  }

  // Pins measured px geometry into neato (72pt = 1in = 72px, y flipped).
  // Nodes are declared invisible but keep their geometry, so they act as
  // routing obstacles and calibration anchors.
  function pinnedDot(nodes) {
    const pins = nodes
      .map(
        (n) =>
          `"${n.id}" [pos="${(n.cx / 72).toFixed(4)},${(-n.cy / 72).toFixed(4)}!", ` +
          `width=${(n.width / 72).toFixed(4)}, height=${(n.height / 72).toFixed(4)}, ` +
          `shape=${n.shape}, fixedsize=true, label=""];`,
      )
      .join("\n");
    return `digraph Routing {
graph [splines=ortho, sep="+10"];
node [color=none, fillcolor=none, style=filled];
${pins}
${dotBody}
}`;
  }

  // Bounding box of a rendered node group, in raw Graphviz SVG coordinates.
  function shapeBox(group) {
    const ellipse = group.querySelector("ellipse");
    if (ellipse) {
      return {
        cx: Number(ellipse.getAttribute("cx")),
        cy: Number(ellipse.getAttribute("cy")),
      };
    }
    const polygon = group.querySelector("polygon");
    if (!polygon) return null;
    const coords = polygon
      .getAttribute("points")
      .trim()
      .split(/[\s,]+/)
      .map(Number);
    const xs = coords.filter((_, i) => i % 2 === 0);
    const ys = coords.filter((_, i) => i % 2 === 1);
    return {
      cx: (Math.min(...xs) + Math.max(...xs)) / 2,
      cy: (Math.min(...ys) + Math.max(...ys)) / 2,
    };
  }

  // Least-squares per-axis affine fit from rendered node centers to the
  // measured px centers, so the overlay is exact whatever pad/translate/scale
  // Graphviz applied.
  function fitAxis(pairs) {
    const n = pairs.length;
    const meanIn = pairs.reduce((sum, [a]) => sum + a, 0) / n;
    const meanOut = pairs.reduce((sum, [, b]) => sum + b, 0) / n;
    let cov = 0;
    let variance = 0;
    for (const [a, b] of pairs) {
      cov += (a - meanIn) * (b - meanOut);
      variance += (a - meanIn) ** 2;
    }
    const scale = variance > 0 ? cov / variance : 1;
    return { scale, offset: meanOut - scale * meanIn };
  }

  function renderOverlay(viz, nodes) {
    const svg = viz.renderSVGElement(pinnedDot(nodes), { engine: "neato" });
    const graph = svg.querySelector("g.graph");
    if (!graph) throw new Error("Graphviz output has no graph group.");

    const byId = new Map(nodes.map((n) => [n.id, n]));
    const xPairs = [];
    const yPairs = [];
    for (const group of graph.querySelectorAll("g.node")) {
      const node = byId.get(group.querySelector("title")?.textContent);
      const box = shapeBox(group);
      if (node && box) {
        xPairs.push([box.cx, node.cx]);
        yPairs.push([box.cy, node.cy]);
      }
      group.remove();
    }
    if (xPairs.length < 2) throw new Error("Too few nodes for calibration.");
    const fx = fitAxis(xPairs);
    const fy = fitAxis(yPairs);

    // drop the background polygon and graph title
    graph.querySelector(":scope > polygon")?.remove();
    graph.querySelector(":scope > title")?.remove();
    graph.setAttribute(
      "transform",
      `translate(${fx.offset} ${fy.offset}) scale(${fx.scale} ${fy.scale})`,
    );

    const box = layout.getBoundingClientRect();
    overlay.setAttribute("width", box.width);
    overlay.setAttribute("height", box.height);
    overlay.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    overlay.replaceChildren(graph);
  }

  Promise.all([
    globalThis.Viz.instance(),
    document.fonts?.ready ?? Promise.resolve(),
  ])
    .then(([viz]) => {
      let frame;
      const render = () => {
        try {
          renderOverlay(viz, measureNodes());
          if (status) status.hidden = true;
          diagram.dataset.renderer = "grid-neato";
        } catch (error) {
          fail("Graphviz could not route the diagram edges.", error);
        }
      };
      render();
      const observer = new ResizeObserver(() => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(render);
      });
      observer.observe(layout);
    })
    .catch((error) => fail("Graphviz could not route this diagram.", error));
})();
