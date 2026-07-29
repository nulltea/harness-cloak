/* Generic renderer for HTML-grid diagrams: CSS grid owns node placement via
   invisible placeholder divs; Graphviz draws the nodes and routes the edges.
   A sizing pass renders the page's DOT once to read each node's
   Graphviz-computed size and applies it to the matching [data-node-id]
   placeholder, so grid geometry and drawn geometry always agree. A routing
   pass then pins the measured placeholder centers (neato, splines=ortho) and
   overlays the resulting SVG. Knows no diagram-specific node or edge names. */
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

  // Bounding box of a rendered node group, in raw Graphviz SVG coordinates.
  function shapeBox(group) {
    const ellipse = group.querySelector("ellipse");
    if (ellipse) {
      const rx = Number(ellipse.getAttribute("rx"));
      const ry = Number(ellipse.getAttribute("ry"));
      return {
        cx: Number(ellipse.getAttribute("cx")),
        cy: Number(ellipse.getAttribute("cy")),
        width: rx * 2,
        height: ry * 2,
      };
    }
    const polygon = group.querySelector("polygon, path");
    if (!polygon) return null;
    const coords = (polygon.getAttribute("points") ?? polygon.getAttribute("d"))
      .replace(/[MCLZ]/gi, " ")
      .trim()
      .split(/[\s,]+/)
      .map(Number);
    const xs = coords.filter((_, i) => i % 2 === 0);
    const ys = coords.filter((_, i) => i % 2 === 1);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return {
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  function nodeGroups(svg) {
    const groups = new Map();
    for (const group of svg.querySelectorAll("g.node")) {
      const title = group.querySelector("title")?.textContent;
      if (title) groups.set(title, group);
    }
    return groups;
  }

  // Sizing pass: Graphviz decides every node's size from the DOT; the
  // placeholders adopt those sizes so the grid reserves the exact space.
  function sizePlaceholders(viz, placeholders) {
    const svg = viz.renderSVGElement(`digraph Sizing {\n${dotBody}\n}`, {
      engine: "dot",
    });
    const groups = nodeGroups(svg);
    for (const element of placeholders) {
      const group = groups.get(element.dataset.nodeId);
      const box = group && shapeBox(group);
      if (!box) throw new Error(`No DOT node sizes ${element.dataset.nodeId}`);
      element.style.width = `${box.width}px`;
      element.style.height = `${box.height}px`;
    }
  }

  function measureCenters(placeholders) {
    const origin = layout.getBoundingClientRect();
    return placeholders.map((element) => {
      const rect = element.getBoundingClientRect();
      return {
        id: element.dataset.nodeId,
        cx: rect.left - origin.left + rect.width / 2,
        cy: rect.top - origin.top + rect.height / 2,
      };
    });
  }

  // Pins measured px centers into neato (72pt = 1in = 72px, y flipped).
  function pinnedDot(centers) {
    const pins = centers
      .map(
        (n) =>
          `"${n.id}" [pos="${(n.cx / 72).toFixed(4)},${(-n.cy / 72).toFixed(4)}!"];`,
      )
      .join("\n");
    return `digraph Routing {
graph [splines=ortho, sep="+10"];
${dotBody}
${pins}
}`;
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

  function renderOverlay(viz, centers) {
    const svg = viz.renderSVGElement(pinnedDot(centers), { engine: "neato" });
    const graph = svg.querySelector("g.graph");
    if (!graph) throw new Error("Graphviz output has no graph group.");

    const byId = new Map(centers.map((n) => [n.id, n]));
    const xPairs = [];
    const yPairs = [];
    for (const [title, group] of nodeGroups(svg)) {
      const node = byId.get(title);
      const box = shapeBox(group);
      if (node && box) {
        xPairs.push([box.cx, node.cx]);
        yPairs.push([box.cy, node.cy]);
      }
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
      const placeholders = Array.from(layout.querySelectorAll("[data-node-id]"));
      sizePlaceholders(viz, placeholders);
      let frame;
      const render = () => {
        try {
          renderOverlay(viz, measureCenters(placeholders));
          if (status) status.hidden = true;
          diagram.dataset.renderer = "grid-neato";
        } catch (error) {
          fail("Graphviz could not render this diagram.", error);
        }
      };
      render();
      const observer = new ResizeObserver(() => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(render);
      });
      observer.observe(layout);
    })
    .catch((error) => fail("Graphviz could not render this diagram.", error));
})();
