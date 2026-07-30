/* Generic edge router for HTML-grid diagrams: CSS grid owns node placement
   and styling; Graphviz (neato with pinned positions, splines=ortho) routes
   only the edges declared in each diagram's DOT block, treating every
   [data-node-id] box as an obstacle. Supports any number of diagrams on the
   page and knows no diagram-specific node or edge names. */
(() => {
  const diagrams = Array.from(
    document.querySelectorAll(".ranker-model-diagram"),
  );
  if (!diagrams.length) return;

  function fail(diagram, message, error) {
    const status = diagram.querySelector(".ranker-model-diagram__status");
    if (status) {
      status.hidden = false;
      status.textContent = message;
    }
    if (error) console.error(message, error);
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

  function initDiagram(diagram, viz) {
    const source = diagram.querySelector("[data-ranker-model-dot]");
    const layout = diagram.querySelector(".ranker-model-diagram__layout");
    const overlay = diagram.querySelector(".ranker-model-diagram__edges");
    if (!source || !layout || !overlay) {
      fail(diagram, "Diagram source, layout, or overlay container is missing.");
      return;
    }

    const dotText = source.textContent;
    const dotBody = dotText.slice(
      dotText.indexOf("{") + 1,
      dotText.lastIndexOf("}"),
    );

    function measureNodes() {
      // The layout div is the overlay's coordinate origin; a plain div rect
      // is reliable everywhere, unlike an auto-sized <svg> rect in Firefox.
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
    // routing obstacles and calibration anchors. Attempts > 0 add a
    // deterministic sub-pixel jitter: exactly pixel-aligned coordinates can
    // hit degenerate maze configurations where the ortho router silently
    // gives up on an edge and draws it straight through obstacles.
    function pinnedDot(nodes, attempt) {
      const pins = nodes
        .map((n, i) => {
          const jx = attempt ? Math.sin(attempt * 37 + i * 13.7) * 0.01 : 0;
          const jy = attempt ? Math.cos(attempt * 61 + i * 7.3) * 0.01 : 0;
          return (
            `"${n.id}" [pos="${(n.cx / 72 + jx).toFixed(4)},${(-n.cy / 72 + jy).toFixed(4)}!", ` +
            `width=${(n.width / 72).toFixed(4)}, height=${(n.height / 72).toFixed(4)}, ` +
            `shape=${n.shape}, fixedsize=true, label=""];`
          );
        })
        .join("\n");
      return `digraph Routing {
graph [splines=ortho, sep="+6"];
node [color=none, fillcolor=none, style=filled];
${pins}
${dotBody}
}`;
    }

    // Counts edge samples that fall inside a non-endpoint node box — the
    // signature of a silent ortho routing give-up.
    function routingViolations(graph, boxes) {
      let violations = 0;
      for (const group of graph.querySelectorAll("g.edge")) {
        const title = group.querySelector("title")?.textContent ?? "";
        const [src, dst] = title.split("->");
        const d = group.querySelector("path")?.getAttribute("d");
        if (!d) continue;
        const nums = d.replace(/[MCL]/gi, " ").trim().split(/[\s,]+/).map(Number);
        for (let i = 0; i + 3 < nums.length && !violations; i += 2) {
          for (let t = 0; t <= 1; t += 0.1) {
            const x = nums[i] + (nums[i + 2] - nums[i]) * t;
            const y = nums[i + 1] + (nums[i + 3] - nums[i + 1]) * t;
            for (const [id, b] of boxes) {
              if (id === src || id === dst) continue;
              if (x > b.minX + 3 && x < b.maxX - 3 && y > b.minY + 3 && y < b.maxY - 3) {
                violations += 1;
                break;
              }
            }
            if (violations) break;
          }
        }
      }
      return violations;
    }

    function renderOverlay(nodes) {
      // Retry with jittered pins until no edge cuts through a node box.
      let svg;
      let graph;
      for (let attempt = 0; attempt < 5; attempt++) {
        svg = viz.renderSVGElement(pinnedDot(nodes, attempt), { engine: "neato" });
        graph = svg.querySelector("g.graph");
        if (!graph) throw new Error("Graphviz output has no graph group.");
        const boxes = [];
        for (const group of graph.querySelectorAll("g.node")) {
          const title = group.querySelector("title")?.textContent;
          const polygon = group.querySelector("polygon");
          if (!title || !polygon) continue;
          const coords = polygon.getAttribute("points").trim().split(/[\s,]+/).map(Number);
          const xs = coords.filter((_, i) => i % 2 === 0);
          const ys = coords.filter((_, i) => i % 2 === 1);
          boxes.push([title, {
            minX: Math.min(...xs), maxX: Math.max(...xs),
            minY: Math.min(...ys), maxY: Math.max(...ys),
          }]);
        }
        if (routingViolations(graph, boxes) === 0) break;
      }

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

    let frame;
    const render = () => {
      try {
        renderOverlay(measureNodes());
        const status = diagram.querySelector(".ranker-model-diagram__status");
        if (status) status.hidden = true;
        diagram.dataset.renderer = "grid-neato";
      } catch (error) {
        fail(diagram, "Graphviz could not route the diagram edges.", error);
      }
    };
    render();
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(render);
    });
    observer.observe(layout);
  }

  if (!globalThis.Viz?.instance) {
    for (const diagram of diagrams) {
      fail(diagram, "Graphviz runtime failed to load.");
    }
    return;
  }

  Promise.all([
    globalThis.Viz.instance(),
    document.fonts?.ready ?? Promise.resolve(),
  ])
    .then(([viz]) => {
      for (const diagram of diagrams) initDiagram(diagram, viz);
    })
    .catch((error) => {
      for (const diagram of diagrams) {
        fail(diagram, "Graphviz could not route this diagram.", error);
      }
    });
})();
