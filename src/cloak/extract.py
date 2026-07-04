"""Rule extractor: out_p -> out_final by deterministic (no-model) inversion of R.

1. Placeholders: exact token swap-back (<PERSON_1> -> canonical surface of its chain).
2. Generalizations: locate mentions of each replacement in out_p (exact, then
   rapidfuzz partial-ratio alignment >= FUZZ_MIN) and narrow to the original surface.
   Ambiguous replacements (same phrase for different originals) are inverted only for
   as many occurrences as R has entries, in document order.

ponytail: exact+fuzzy only — MiniLM window matching for paraphrased mentions is the
semantic-window extractor, add when the unmatched rate justifies it. The learned extractor
(flan-t5 LoRA) trains on the residue.
Plan: docs/plans/2026-07-02-d1-prototype-implementation.md.
"""
import re

FUZZ_MIN = 90.0


def _canonical(entries: list[dict]) -> str:
    return max((e["surface"] for e in entries), key=len)


def invert(out_p: str, R: list[dict]) -> tuple[str, dict]:
    """Returns (out_final, stats)."""
    text = out_p
    stats = {"ph_swapped": 0, "gen_exact": 0, "gen_fuzzy": 0, "gen_absent": 0, "ph_residue": 0}

    ph: dict[str, list[dict]] = {}
    gen: dict[str, list[dict]] = {}
    for e in R:
        (ph if e["action"] == "placeholder" else gen).setdefault(e["replacement"], []).append(e)

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
                lo, hi = al.dest_start, al.dest_end  # snap to word boundaries
                while lo > 0 and text[lo - 1].isalnum():
                    lo -= 1
                while hi < len(text) and text[hi].isalnum():
                    hi += 1
                text = text[:lo] + surface + text[hi:]
                stats["gen_fuzzy"] += 1
                continue
        except ImportError:
            pass
        stats["gen_absent"] += 1  # remote answer never mentioned this span; nothing to invert

    # any typed placeholder left in out_final is a leak of mechanism artifacts to the user
    # (exhaustion-fallback placeholders included, not only PERSON/CODE)
    stats["ph_residue"] = len(re.findall(r"<[A-Z]+_\d+>", text))
    return text, stats


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
    assert stats["ph_residue"] == 1  # the stray <QUANTITY_3> counts for ALL types now
    print("extract.py self-check OK")
