"""Extractors: out_p -> out_final by local inversion of R.

1. Placeholders: exact token swap-back (<PERSON_1> -> canonical surface of its chain).
2. Generalizations: locate mentions of each replacement in out_p (exact, then
   rapidfuzz partial-ratio alignment >= FUZZ_MIN) and narrow to the original surface.
3. Semantic-window fallback: only after exact/fuzzy miss, locate fuzzy 60-90 candidate
   windows, score them with MiniLM, and invert only high-margin, type-sane matches.
4. Detector-pointer arm: run the detector on rule-prepass residue, score typed candidate
   spans against R entries, assign one-to-one, and abstain when confidence is low.
   Ambiguous replacements (same phrase for different originals) are inverted only for
   as many occurrences as R has entries, in document order.

Designs:
  docs/plans/2026-07-05-extractor-inverse-designs.md
  docs/plans/2026-07-05-detector-pointer-extractor.md
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

FUZZ_MIN = 90.0
BAND_MIN = 60.0
SEMANTIC_MIN = 0.70
SEMANTIC_MARGIN = 0.04
POINTER_MIN = 0.70
POINTER_MARGIN = 0.05
DETECTOR_POINTER_GLINER_MODEL = "data/models/pii_gliner_multidomain/checkpoint-2479"
_MAX_CANDIDATES = 12
_GENERIC_SEMANTIC_FILLS = {"something"}

_MONTHS = ("jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
           "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
           "oct", "october", "nov", "november", "dec", "december")
_NUMWORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
             "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
             "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
             "thousand", "million", "billion")
_DEM_WORDS = ("age", "aged", "year-old", "years old", "teen", "twenties", "thirties",
              "forties", "fifties", "sixties", "seventies", "profession", "national",
              "nationality", "ethnic", "religion", "citizen", "worker", "student",
              "doctor", "nurse", "lawyer", "engineer", "teacher", "manager")
_LOC_WORDS = ("city", "town", "county", "state", "province", "country", "address",
              "street", "road", "avenue", "region", "district", "neighborhood")
_ORG_WORDS = ("company", "organization", "organisation", "court", "hospital",
              "university", "school", "agency", "department", "ministry", "firm",
              "bank", "clinic", "institution")


@dataclass(frozen=True)
class _PointerCandidate:
    start: int
    end: int
    text: str
    type: str
    detector_score: float
    slot: int


def _canonical(entries: list[dict]) -> str:
    return max((e["surface"] for e in entries), key=len)


def _word_snap(text: str, lo: int, hi: int) -> tuple[int, int]:
    while lo > 0 and text[lo - 1].isalnum():
        lo -= 1
    while hi < len(text) and text[hi].isalnum():
        hi += 1
    return lo, hi


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)]


def _candidate_windows(fill: str, text: str) -> list[tuple[int, int, float, str]]:
    """Fuzzy 60-90 candidate slices for the semantic fallback, sorted best first."""
    from rapidfuzz import fuzz

    candidates: dict[tuple[int, int], tuple[float, str]] = {}
    al = fuzz.partial_ratio_alignment(fill.lower(), text.lower())
    if al and BAND_MIN <= al.score < FUZZ_MIN and al.dest_end > al.dest_start:
        lo, hi = _word_snap(text, al.dest_start, al.dest_end)
        candidates[(lo, hi)] = (float(al.score), text[lo:hi])

    toks = _token_spans(text)
    fill_len = max(1, len(re.findall(r"\S+", fill)))
    for win_len in range(max(1, fill_len - 2), fill_len + 5):
        if win_len > len(toks):
            continue
        for i in range(0, len(toks) - win_len + 1):
            lo, hi = toks[i][0], toks[i + win_len - 1][1]
            snippet = text[lo:hi]
            score = float(fuzz.partial_ratio(fill.lower(), snippet.lower()))
            if BAND_MIN <= score < FUZZ_MIN:
                prev = candidates.get((lo, hi))
                if prev is None or score > prev[0]:
                    candidates[(lo, hi)] = (score, snippet)

    ranked = [(lo, hi, score, snippet) for (lo, hi), (score, snippet) in candidates.items()]
    ranked.sort(key=lambda c: (-c[2], c[0], c[1]))
    return ranked[:_MAX_CANDIDATES]


def _has_numish(text: str) -> bool:
    low = text.lower()
    return bool(re.search(r"\d", low) or any(w in low for w in _NUMWORDS))


def _type_sane(entity_type: str, fill: str, window: str) -> bool:
    """Cheap typed guard for semantic accepts; abstain when the type has no local cue."""
    typ = (entity_type or "MISC").upper()
    wlow = window.lower()
    if typ == "DATETIME":
        return bool(re.search(r"\b\d{1,4}\b", wlow) or any(m in wlow for m in _MONTHS) or
                    re.search(r"\b(today|yesterday|tomorrow|spring|summer|fall|autumn|winter|"
                              r"morning|evening|night|week|month|year)\b", wlow))
    if typ == "QUANTITY":
        return bool(_has_numish(window) or re.search(r"[$€£%]|\b(dollars?|euros?|pounds?|"
                                                     r"percent|kg|mg|ml|miles?|hours?)\b", wlow))
    if typ == "CODE":
        return bool(re.search(r"[A-Z]{1,6}[-_/]?\d|\d[-_/]\d|#\s?\d", window))
    if typ == "DEM":
        return bool(_has_numish(window) or any(w in wlow for w in _DEM_WORDS))
    if typ == "LOC":
        return any(w in wlow for w in _LOC_WORDS)
    if typ == "ORG":
        return any(w in wlow for w in _ORG_WORDS)
    if typ == "PERSON":
        return bool(re.search(r"\b(person|patient|applicant|claimant|defendant|plaintiff|"
                              r"employee|manager|officer|doctor|dr\.|mr\.|mrs\.|ms\.)\b", wlow))
    return True


@lru_cache(maxsize=1)
def _semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _semantic_scores(fill: str, snippets: tuple[str, ...]) -> list[float]:
    model = _semantic_model()
    emb = model.encode([fill, *snippets], normalize_embeddings=True)
    return [float(emb[0] @ e) for e in emb[1:]]


def _pointer_scores(query: str, snippets: tuple[str, ...]) -> list[float]:
    return _semantic_scores(query, snippets)


def _semantic_invert(text: str, entry: dict) -> tuple[str, bool]:
    fill = entry["replacement"]
    if fill.strip().lower() in _GENERIC_SEMANTIC_FILLS:
        return text, False
    candidates = _candidate_windows(fill, text)
    if not candidates:
        return text, False
    snippets = tuple(c[3] for c in candidates)
    try:
        scores = _semantic_scores(fill, snippets)
    except Exception:
        return text, False
    ranked = sorted(zip(scores, candidates), key=lambda x: (-x[0], -x[1][2]))
    best_cos, (lo, hi, _, snippet) = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    if best_cos < SEMANTIC_MIN:
        return text, False
    if runner_up is not None and best_cos - runner_up < SEMANTIC_MARGIN:
        return text, False
    if not _type_sane(entry.get("type", "MISC"), fill, snippet):
        return text, False
    return text[:lo] + entry["surface"] + text[hi:], True


def _partition_R(R: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    ph: dict[str, list[dict]] = {}
    gen: dict[str, list[dict]] = {}
    for e in R:
        (ph if e["action"] == "placeholder" else gen).setdefault(e["replacement"], []).append(e)
    return ph, gen


def _base_stats() -> dict:
    return {"ph_swapped": 0, "gen_exact": 0, "gen_fuzzy": 0, "gen_semantic": 0,
            "gen_absent": 0, "ph_residue": 0}


def _finalize(text: str, stats: dict) -> tuple[str, dict]:
    # Any typed placeholder left in out_final is a leak of mechanism artifacts to the user
    # (exhaustion-fallback placeholders included, not only PERSON/CODE).
    stats["ph_residue"] = len(re.findall(r"<[A-Z]+_\d+>", text))
    return text, stats


def _rule_prepass(out_p: str, R: list[dict], *, semantic: bool) -> tuple[str, dict, list[dict]]:
    """Placeholder + exact/fuzzy pre-pass. Returns residue generalization entries."""
    text = out_p
    stats = _base_stats()
    residue = []
    ph, gen = _partition_R(R)

    for token, entries in ph.items():
        n = text.count(token)
        if n:
            text = text.replace(token, _canonical(entries))
            stats["ph_swapped"] += n

    for repl, entries in sorted(gen.items(), key=lambda kv: -len(kv[0])):  # longest first
        # R is injective (substitute.py used-set): one replacement -> one surface; entries
        # for the same replacement are repeat mentions of that surface
        surface = entries[0]["surface"]
        pat = re.compile(rf"\b{re.escape(repl)}\b", re.IGNORECASE)
        hits = len(pat.findall(text))
        if hits:
            text = pat.sub(surface, text)
            stats["gen_exact"] += hits
            continue
        try:  # fuzzy: locate an approximate mention and replace the aligned slice
            from rapidfuzz import fuzz
            al = fuzz.partial_ratio_alignment(repl.lower(), text.lower())
            if al and al.score >= FUZZ_MIN and al.dest_end > al.dest_start:
                lo, hi = _word_snap(text, al.dest_start, al.dest_end)
                text = text[:lo] + surface + text[hi:]
                stats["gen_fuzzy"] += 1
                continue
        except ImportError:
            pass
        if semantic:
            new_text, ok = _semantic_invert(text, entries[0])
            if ok:
                text = new_text
                stats["gen_semantic"] += 1
                continue
        residue.append(entries[0])

    return text, stats, residue


def invert(out_p: str, R: list[dict]) -> tuple[str, dict]:
    """Rule cascade: placeholder, exact/fuzzy generalization, semantic-window fallback."""
    text, stats, residue = _rule_prepass(out_p, R, semantic=True)
    stats["gen_absent"] += len(residue)
    return _finalize(text, stats)


def _span_field(span: Any, name: str, default: Any = None) -> Any:
    if isinstance(span, dict):
        return span.get(name, default)
    return getattr(span, name, default)


def _detect_spans(detector: Any, text: str, *, gliner_model: str = DETECTOR_POINTER_GLINER_MODEL) -> list[Any]:
    if detector is None:
        from cloak.detect import Detector
        detector = Detector(gliner_model=gliner_model)
    if hasattr(detector, "detect"):
        return list(detector.detect(text))
    return list(detector(text))


def _token_index_for_span(tokens: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    overlapping = [i for i, (lo, hi) in enumerate(tokens) if lo < end and start < hi]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1] + 1


def _dilate_detector_spans(text: str, spans: list[Any], tau_det: float) -> list[_PointerCandidate]:
    toks = _token_spans(text)
    out: dict[tuple[int, int, int], _PointerCandidate] = {}
    for slot, span in enumerate(spans):
        score = float(_span_field(span, "score", 1.0) or 0.0)
        if score < tau_det:
            continue
        start = int(_span_field(span, "start", 0))
        end = int(_span_field(span, "end", 0))
        typ = str(_span_field(span, "type", _span_field(span, "label", "MISC")) or "MISC").upper()
        idx = _token_index_for_span(toks, start, end)
        if idx is None:
            continue
        i0, i1 = idx
        for si in range(max(0, i0 - 2), min(len(toks), i0 + 3)):
            for ei in range(max(si + 1, i1 - 2), min(len(toks), i1 + 2) + 1):
                lo, hi = toks[si][0], toks[ei - 1][1]
                txt = text[lo:hi]
                if txt.strip():
                    out[(slot, lo, hi)] = _PointerCandidate(lo, hi, txt, typ, score, slot)
    return sorted(out.values(), key=lambda c: (c.start, c.end, c.slot))


def _compatible(entry_type: str, cand_type: str,
                type_confusions: dict[str, set[str]] | None = None) -> bool:
    entry_type = (entry_type or "MISC").upper()
    cand_type = (cand_type or "MISC").upper()
    if entry_type == cand_type:
        return True
    return bool(type_confusions and cand_type in type_confusions.get(entry_type, set()))


def _pointer_assign(residue: list[dict], candidates: list[_PointerCandidate], *,
                    score_min: float, delta: float,
                    type_confusions: dict[str, set[str]] | None = None) -> dict[int, _PointerCandidate]:
    pairs = []
    by_entry: dict[int, list[tuple[float, _PointerCandidate]]] = {}
    for i, entry in enumerate(residue):
        compatible = [c for c in candidates if _compatible(entry.get("type", "MISC"), c.type,
                                                           type_confusions)]
        if not compatible:
            continue
        query = f"{entry.get('type', 'MISC')}: {entry['replacement']}"
        scores = _pointer_scores(query, tuple(c.text for c in compatible))
        scored = list(zip(scores, compatible))
        by_entry[i] = scored
        pairs.extend((score, i, cand) for score, cand in scored)

    assigned: dict[int, _PointerCandidate] = {}
    used_slots = set()
    for score, i, cand in sorted(pairs, key=lambda p: (-p[0], p[2].start, p[2].end)):
        if i in assigned or cand.slot in used_slots or score < score_min:
            continue
        runners = [s for s, c in by_entry.get(i, []) if c.slot != cand.slot]
        runner_up = max(runners) if runners else None
        if runner_up is not None and score - runner_up < delta:
            continue
        assigned[i] = cand
        used_slots.add(cand.slot)
    return assigned


def invert_detector_pointer(out_p: str, R: list[dict], *, detector: Any = None,
                            gliner_model: str = DETECTOR_POINTER_GLINER_MODEL,
                            tau_det: float = 0.0, score_min: float = POINTER_MIN,
                            delta: float = POINTER_MARGIN,
                            type_confusions: dict[str, set[str]] | None = None) -> tuple[str, dict]:
    """Detector-pointer arm: rule pre-pass, typed assignment, explicit abstain.

    This implements the deployable inference interface from
    docs/plans/2026-07-05-detector-pointer-extractor.md. Until an FT-extractor checkpoint
    exists, the pointer scorer is MiniLM over `"{type}: {replacement}"` queries and
    detector candidate spans; `_pointer_scores` is the checkpoint-backed replacement hook.
    """
    text, stats, residue = _rule_prepass(out_p, R, semantic=False)
    stats["gen_pointer"] = 0
    stats["gen_abstain"] = 0
    if not residue:
        return _finalize(text, stats)

    candidates = _dilate_detector_spans(text, _detect_spans(detector, text, gliner_model=gliner_model),
                                        tau_det)
    assigned = _pointer_assign(residue, candidates, score_min=score_min, delta=delta,
                               type_confusions=type_confusions)
    for i, cand in sorted(assigned.items(), key=lambda kv: -kv[1].start):
        text = text[:cand.start] + residue[i]["surface"] + text[cand.end:]
        stats["gen_pointer"] += 1
    stats["gen_abstain"] = len(residue) - len(assigned)
    stats["gen_absent"] += stats["gen_abstain"]
    return _finalize(text, stats)


if __name__ == "__main__":
    R = [
        {"action": "placeholder", "surface": "Sarah Johnson", "replacement": "<PERSON_1>"},
        {"action": "placeholder", "surface": "Sarah", "replacement": "<PERSON_1>"},
        {"action": "placeholder", "surface": "12 March 2019", "replacement": "<DATETIME_1>"},
        {"action": "generalize", "surface": "Boston", "replacement": "a city in Massachusetts"},
        {"action": "generalize", "surface": "34", "replacement": "thirty-something"},
        {"action": "generalize", "surface": "120,000 dollars",
         "replacement": "between 60,000 and 240,000 dollars"},
    ]
    out_p = ("Since <PERSON_1> is thirty-something and lives in a City in Massachusetts, "
             "she should register there since <DATETIME_1>. Earning between 60,000 and "
             "240,000  dollars is above the median for <PERSON_1>'s area. "
             "Thirty-somethings thrive there. Contact <QUANTITY_3> for details.")
    final, stats = invert(out_p, R)
    print(final)
    print(stats)
    assert "Sarah Johnson" in final and "<PERSON_1>" not in final
    assert "12 March 2019" in final and "<DATETIME_1>" not in final  # exhaustion-fallback swap
    assert "Boston" in final and "34" in final
    assert "Earning 120,000 dollars" in final, final  # fuzzy slice must land on word boundary
    assert "Thirty-somethings thrive" in final, final  # \b guard: no inversion inside larger words
    assert stats["gen_fuzzy"] >= 1  # the double-space money mention needs the fuzzy path
    assert "gen_semantic" in stats
    assert stats["ph_residue"] == 1  # the stray <QUANTITY_3> counts for ALL types now
    print("extract.py self-check OK")
