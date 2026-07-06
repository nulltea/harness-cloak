"""Design 3 reconstructor: recover the survived-span residue the rule cascade
(cloak.extract.invert) misses — paraphrase-reworded / lossy fill mentions — by editing
out_p with the original surfaces in R. Residue-targeted constrained edit; copy-bias guard
enforces do-no-harm (only R originals may enter). Plan: docs/plans/2026-07-06-reconstructor-design3-plan.md.
"""
import re


def linearize_restore_map(residue: list[dict]) -> str:
    """One line per residue entry: 'TYPE: "fill" => "original"'. Order preserved."""
    return "\n".join(
        f'{e.get("type", "MISC")}: "{e["replacement"]}" => "{e["surface"]}"'
        for e in residue)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def splice_at_quote(text: str, quote: str, replacement: str) -> tuple[str, bool]:
    """Locate `quote` in `text` ignoring case and inner whitespace; replace that slice
    with `replacement`. Returns (new_text, spliced?)."""
    if not quote:
        return text, False
    pat = re.compile(r"\s+".join(re.escape(w) for w in quote.split()), re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return text, False
    return text[:m.start()] + replacement + text[m.end():], True
