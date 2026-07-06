"""Design 3 reconstructor: recover the survived-span residue the rule cascade
(cloak.extract.invert) misses — paraphrase-reworded / lossy fill mentions — by editing
out_p with the original surfaces in R. Residue-targeted constrained edit; copy-bias guard
enforces do-no-harm (only R originals may enter). Plan: docs/plans/2026-07-06-reconstructor-design3-plan.md.
"""
import difflib
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


def edit_guard(prepass: str, candidate: str, residue: list[dict], max_edits: int) -> bool:
    """do-no-harm, fill-occurrence anchored (Round-2/3 fix). At inference there is no judge
    quote, so anchor to each residue entry's own mention: fuzzy-locate the entry's fill in
    `prepass`, and require EVERY diff op to be a REPLACE that (a) overlaps one entry's located
    fill span, (b) inserts that SAME entry's surface (tight match, not just any allowed
    surface, and not surrounded by extra content), (c) has a non-empty deleted span. Rejects
    pure insert (surface dropped elsewhere), pure delete (mention removed w/o restoring),
    wrong-location replace, multi-surface hunks, and >max_edits regions.
    ponytail: word-level diff + fuzzy-anchor guard — upgrade to token-constrained decoding
    (prefix_allowed_tokens_fn) if the measured reject/fallback rate is high."""
    from rapidfuzz import fuzz
    anchors = []   # (lo, hi, surface_norm) — fill mention span in prepass + its target surface
    for e in residue:
        al = fuzz.partial_ratio_alignment(e["replacement"].lower(), prepass.lower())
        if al and al.score >= 60.0 and al.dest_end > al.dest_start:
            anchors.append((al.dest_start, al.dest_end, _norm(e["surface"])))
    # Round-4 hardening: reject when two anchors of DIFFERENT surfaces overlap the same region
    # (ambiguous fuzzy match on a repeated/generic phrase — we cannot tell which original the
    # edit restores). Bail rather than risk a wrong-surface substitution.
    for a in range(len(anchors)):
        for b in range(a + 1, len(anchors)):
            (lo1, hi1, s1), (lo2, hi2, s2) = anchors[a], anchors[b]
            if s1 != s2 and not (hi1 <= lo2 or hi2 <= lo1):
                return False
    # Word-level diff (char-level difflib fragments a valid in-place replace whose old/new share
    # letters — "a disease"->"arthritis" splits into replace+equal('is')+delete AND absorbs the
    # shared leading 'a' into the prior equal block, so no char-region carries the full surface).
    # Diff on whitespace tokens, but map each token op back to EXACT char offsets in the original
    # strings (never rejoin tokens — that loses spacing/punctuation and desyncs the anchor spans).
    def _tokens(s):   # (char_start, char_end) per whitespace-delimited token
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", s)]
    p_tok, c_tok = _tokens(prepass), _tokens(candidate)

    def _span(tok, t1, t2, full):   # char span of a token-op; zero-width at boundary if empty
        if t2 > t1:
            return tok[t1][0], tok[t2 - 1][1]
        pos = tok[t1][0] if t1 < len(tok) else full
        return pos, pos
    # Coalesce consecutive non-equal token ops into ONE region; an `equal` op closes it. Region
    # char span on each side = min start .. max end over its ops (robust to a delete/insert op
    # that is empty on one side, which must not collapse the other side's span).
    regions = []   # (i1, i2, j1, j2) char offsets into prepass/candidate
    cur = None
    for op, ti1, ti2, tj1, tj2 in difflib.SequenceMatcher(
            a=[prepass[a:b] for a, b in p_tok],
            b=[candidate[a:b] for a, b in c_tok], autojunk=False).get_opcodes():
        if op == "equal":
            if cur:
                regions.append(cur)
                cur = None
            continue
        i1o, i2o = _span(p_tok, ti1, ti2, len(prepass))
        j1o, j2o = _span(c_tok, tj1, tj2, len(candidate))
        if cur is None:
            cur = [i1o, i2o, j1o, j2o]
        else:
            cur = [min(cur[0], i1o), max(cur[1], i2o), min(cur[2], j1o), max(cur[3], j2o)]
    if cur:
        regions.append(cur)
    edits = 0
    for i1, i2, j1, j2 in regions:
        edits += 1
        deleted, inserted = _norm(prepass[i1:i2]), _norm(candidate[j1:j2])
        if not deleted or not inserted:
            return False                      # not an in-place replace
        matched = False
        for lo, hi, surf in anchors:
            overlaps = not (i2 <= lo or i1 >= hi)   # edit region meets this fill's mention
            tight = surf and (inserted == surf or        # exact, or surface + minor punctuation/casing
                              (surf in inserted and len(inserted) - len(surf) <= 3))
            if overlaps and tight:
                matched = True
                break
        if not matched:
            return False                      # wrong location, wrong/mixed surface, or hallucination
    return edits <= max_edits
