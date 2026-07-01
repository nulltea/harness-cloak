// Global site navbar — single source of truth for every page.
// To add a page: append to PAGES (or add a navgroup block, see .navgroup CSS).
// Each page needs: <div class="topnav" id="site-nav"></div> and <script src="js/nav.js"></script>.
(function () {
  var PAGES = [
    { href: "infer-dtp.html", label: "InferDPT &amp; RANTEXT" }
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
})();
