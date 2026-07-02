// Global site navbar — single source of truth for every page.
// To add a page: append to PAGES (or add a navgroup block, see .navgroup CSS).
// Each page needs: <div class="topnav" id="site-nav"></div> and <script src="js/nav.js"></script>.
(function () {
  var PAGES = [
    { href: "infer-dtp.html", label: "InferDPT &amp; RANTEXT" },
    { href: "LatticeCloak.html", label: "LatticeCloak" }
  ];
  var here = location.pathname.split("/").pop() || "infer-dtp.html";
  var links = PAGES.map(function (p) {
    var cur = p.href === here ? ' class="current"' : "";
    return '<a href="' + p.href + '"' + cur + ">" + p.label + "</a>";
  }).join("");
  var nav = document.getElementById("site-nav");
  if (nav) {
    nav.innerHTML =
      '<a class="brand" href="' + PAGES[0].href + '">HarnessCloak</a>' +
      '<span class="links">' + links + "</span>";
  }

  // Heading anchors: hover-reveal "#" links on section titles (h2) and
  // subsections (h3/h4) so any section is directly linkable. Styling lives in
  // css/site.css (.heading-anchor). Idempotent — safe if the DOM re-renders.
  function slug(t) {
    return t.replace(/[#¶]\s*$/, "").toLowerCase()
      .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
  }
  function anchor(id) {
    var a = document.createElement("a");
    a.className = "heading-anchor";
    a.href = "#" + id;
    a.textContent = "#";
    a.setAttribute("aria-label", "Link to this section");
    return a;
  }
  function addHeadingAnchors() {
    // h2 → link the enclosing section's existing id
    document.querySelectorAll("section[id] > .section-head > .section-title").forEach(function (h) {
      if (h.querySelector(".heading-anchor")) return;
      var sec = h.closest("section[id]");
      if (sec) h.appendChild(anchor(sec.id));
    });
    // h3/h4 subsections → give a slug id, then link it
    document.querySelectorAll("section h3, section h4").forEach(function (h) {
      if (h.querySelector(".heading-anchor")) return;
      if (!h.id) {
        var s = slug(h.textContent);
        if (s) h.id = s;
      }
      if (h.id) h.appendChild(anchor(h.id));
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addHeadingAnchors);
  } else {
    addHeadingAnchors();
  }
})();
