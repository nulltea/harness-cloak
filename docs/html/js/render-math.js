/* KaTeX pass over report prose. Only \( \) and \[ \] are treated as math, so
   ordinary text and currency-style dollars are never captured. Code spans,
   pre blocks, and the grid-diagram figures are left untouched: identifiers and
   file names stay in the monospace face, math renders in the math face. */
(() => {
  const run = () => {
    if (!globalThis.renderMathInElement) {
      console.error("KaTeX auto-render unavailable; math left as source.");
      return;
    }
    globalThis.renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      ignoredClasses: ["ranker-model-diagram"],
      throwOnError: false,
      errorColor: "#a3460a",
    });
    // KaTeX emits inline-block spans, which introduces a line-break
    // opportunity before any punctuation that follows. Glue short formulas to
    // their trailing punctuation so a comma or period never starts a line.
    for (const math of document.querySelectorAll(".katex")) {
      // auto-render wraps each formula in a classless span, so climb to the
      // node that actually sits beside the following text.
      let host = math;
      while (
        host.parentElement &&
        !host.parentElement.className &&
        host.parentElement.childNodes.length === 1 &&
        host.parentElement !== document.body
      ) {
        host = host.parentElement;
      }
      const next = host.nextSibling;
      if (!next || next.nodeType !== Node.TEXT_NODE) continue;
      const punct = next.nodeValue.match(/^[,.;:!?)\]]+/);
      if (!punct) continue;
      const parent = host.parentElement;
      if (!parent || host.offsetWidth > parent.clientWidth * 0.4) continue;
      const glue = document.createElement("span");
      glue.style.whiteSpace = "nowrap";
      host.before(glue);
      glue.append(host, document.createTextNode(punct[0]));
      next.nodeValue = next.nodeValue.slice(punct[0].length);
    }
    document.body.dataset.mathRendered = "katex";
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
