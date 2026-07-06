"""Design 3 reconstructor: recover the survived-span residue the rule cascade
(cloak.extract.invert) misses — paraphrase-reworded / lossy fill mentions — by editing
out_p with the original surfaces in R. Residue-targeted constrained edit; copy-bias guard
enforces do-no-harm (only R originals may enter). Plan: docs/plans/2026-07-06-reconstructor-design3-plan.md.
"""
import difflib
import re

from cloak.extract import _rule_prepass, _finalize

_RECON = {}


def load_reconstructor(path: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    if path not in _RECON:
        tok = AutoTokenizer.from_pretrained(path)
        base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",
                                                     torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, path).eval()
        _RECON[path] = (model, tok)
    return _RECON[path]


def run_model(obj, prompt: str) -> str:
    import torch
    model, tok = obj
    PROMPT = ("Restore the original terms in the CLINICAL/LEGAL answer below. Replace each "
              "generalized mention with its original from the RESTORE map; copy everything "
              "else verbatim; if a mapped term is not present, leave the text unchanged.\n\n{input}")
    ids = tok(PROMPT.format(input=prompt), return_tensors="pt", truncation=True,
              max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1024, num_beams=1)
    return tok.decode(out[0], skip_special_tokens=True)


def reconstruct(out_p: str, R: list[dict], model=None) -> tuple[str, dict]:
    """Cascade first; model edits only the residue, gated by copy-bias (do-no-harm)."""
    prepass, stats, residue = _rule_prepass(out_p, R, semantic=True)
    text = prepass
    if residue and model is not None:
        prompt = f"{prepass}\n\n[RESTORE]\n{linearize_restore_map(residue)}"
        cand = run_model(model, prompt)
        if edit_guard(prepass, cand, residue, max_edits=2 * len(residue) + 1):
            text = cand
            stats["gen_reconstructor"] = sum(1 for e in residue if e["surface"] in cand
                                             and e["surface"] not in prepass)
        else:
            stats["gen_recon_rejected"] = 1
    stats.setdefault("gen_reconstructor", 0)
    stats["gen_absent"] += len(residue)  # cascade's residue accounting; refined by recon count
    return _finalize(text, stats)


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


def build_target(text: str, located: list[dict]) -> tuple[str, int]:
    """Splice each located mention's original into text (longest-quote-first so a short quote
    can't match inside a longer one); entries with a falsy quote are abstains (no edit).
    Returns (target_text, n_edits)."""
    edits = 0
    for e in sorted([x for x in located if x.get("quote")],
                    key=lambda x: -len(x["quote"])):
        new, ok = splice_at_quote(text, e["quote"], e["surface"])
        if ok:
            text, edits = new, edits + 1
    return text, edits


_AMBIGUOUS_TYPES = {"DATETIME", "QUANTITY", "LOC", "ORG"}  # scalar/named — high false-corr risk


def restorable(entry: dict, verdict: dict, prepass: str, nli=None) -> bool:
    """Target-admission gate (Round-2 fix — type-sanity alone is NOT correspondence: "the
    last four years" vs "three years ago" are both DATETIME-sane). Admit a splice target
    only if the judge marked it present, the quote is grounded in the text we edit, AND:
      - for scalar/named ambiguous types, a MANDATORY correspondence check: the quote must
        be entailed by / consistent with the FILL (the generalization actually sent), so the
        mention is the fill's restatement, not a different specific the model invented. This
        admits D-3 lossy (quote ⊨ fill: "Early 1980s" is consistent with fill "the early
        1980s") and rejects D-4 ("three years ago" is NOT entailed by fill "some time ago"'s
        content — it adds a specific not in doc_p).
      - other types: type-sanity suffices.
    D-3 (exact original in R) still passes; only D-4 false-correspondence is filtered."""
    from cloak.extract import _type_sane
    q = verdict.get("quote")
    if verdict.get("label") not in ("SURVIVED", "REWORDED") or not q:
        return False
    if _norm(q) not in _norm(prepass):
        return False
    if not _type_sane(entry.get("type", "MISC"), entry["replacement"], q):
        return False
    if entry.get("type", "MISC") in _AMBIGUOUS_TYPES:
        # mandatory correspondence: quote must not assert a specific beyond the fill's content
        return _corresponds(entry["replacement"], q, nli)   # NLI: quote ⊨ fill, no added specificity
    return True


def _corresponds(fill: str, quote: str, nli) -> bool:
    """Admit only if the quote adds NO information beyond the generalized fill — i.e.
    `fill ⊨ quote` (Round-3 fix: the entailment must run fill→quote, not quote→fill; the
    latter admits a model-invented specific because "three years ago" ⊨ "some time ago").
    Admits D-3 lossy ("the early 1980s" ⊨ "Early 1980s"); rejects D-4 ("some time ago" ⊭
    "three years ago") and inference leaks ("a city" ⊭ "Boston"). For DATETIME/QUANTITY,
    prefer the deterministic `_value_compatible` check first (NLI is unreliable on scalars);
    fall back to NLI only when it abstains."""
    ok = _value_compatible(fill, quote)
    if ok is not None:
        return ok
    if nli is None:
        return False                       # fail-closed
    return nli(premise=fill, hypothesis=quote) == "entailment"


def _value_compatible(fill: str, quote: str):
    """Deterministic scalar gate, FAIL-CLOSED (Round-4 fix — never admit on a bare digit
    subset: "early 1980s"→"late 1980s" shares {1980} but flips the modifier). Only two
    non-deferring outcomes:
      False — the quote introduces a digit-run the fill lacks (model-invented specific/leak);
      None  — digits are subset-compatible but NOT proven equivalent, OR no digits → NLI (or
              a real date/quantity normalizer) must still approve.
    It never returns True; equivalence is proven downstream, not assumed here."""
    fn = set(re.findall(r"\d+", fill))
    qn = set(re.findall(r"\d+", quote))
    if qn and not qn <= fn:
        return False       # hard reject: a specific number absent from the fill
    return None            # defer: subset-compatible or no digits — NLI must confirm fill ⊨ quote


def _load_nli():
    """cross-encoder/nli-deberta-v3-small -> callable(premise, hypothesis) ->
    {'entailment'|'neutral'|'contradiction'}. Local, small; loaded once."""
    from transformers import pipeline
    pipe = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
    label_map = {"entailment": "entailment", "neutral": "neutral",
                 "contradiction": "contradiction"}
    def _nli(premise: str, hypothesis: str) -> str:
        out = pipe({"text": premise, "text_pair": hypothesis})
        if isinstance(out, list):
            out = out[0]
        return label_map.get(out["label"].lower(), "neutral")
    return _nli


def classify_recovery(out_final: str, quote: str, surface: str, prepass: str) -> str:
    """Per-residue outcome, evaluated in the quote's LOCAL WINDOW (Round-3 fix — a global
    'surface present ∧ quote gone' counts a wrong-location insert as recovery). Anchor on the
    words flanking the quote in `prepass`, relocate that window in `out_final`, and judge only
    there:
      recovered    — window now holds the original surface, quote no longer stands
      wrong_insert — surface appears but the quote still stands in-window
      deletion     — quote gone but surface absent in-window (reworded away, not restored)
      miss         — quote still stands, surface absent
    """
    p, f, sn, ql = _norm(prepass), _norm(out_final), _norm(surface), _norm(quote or "")
    i = p.find(ql) if ql else -1
    if i < 0:                                   # can't locate the quote — fall back to global
        window = f
    else:
        left = " ".join(p[max(0, i - 24):i].split()[-3:])
        right = " ".join(p[i + len(ql):i + len(ql) + 24].split()[:3])
        lo = (f.find(left) + len(left)) if left and left in f else 0
        hi = f.find(right, lo) if right and right in f else -1
        window = f[lo:hi] if hi > lo else f[lo:lo + max(len(sn), len(ql)) + 40]
    has_surf = sn in window
    quote_stands = bool(ql) and ql in window and sn not in ql
    if has_surf and not quote_stands:
        return "recovered"
    if has_surf and quote_stands:
        return "wrong_insert"
    if not has_surf and not quote_stands:
        return "deletion"
    return "miss"
