"""Pipeline red tests — adversarial cases through the FULL production Detector.

Companion to detector_red_tests.py (which probes raw GLiNER classification on the miner path and
is blind to pipeline code). These cases target the detector pipeline itself: stop-word span
deletion, Presidio pattern misfires on clinical numerics, coref identity merges, and
chunk-boundary span loss. A case FIRES when the pipeline exhibits the defect.

Run against two checkouts to measure a fix (same model => the delta is purely pipeline code):

    # before: PYTHONPATH=<worktree-at-old-commit>/src .venv/bin/python scripts/detector_pipeline_red_tests.py --tag before
    # after:  PYTHONPATH=src .venv/bin/python scripts/detector_pipeline_red_tests.py --tag after

Uses Detector(profile="clinical") when the checkout supports profiles, else plain Detector()
(the pre-fix behavior — that IS the before-measurement, profiles are part of the fix).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloak.detect import Detector, coref_chains


def _spans_on(spans, needle, *, types=None, presidio_only=False):
    hits = [s for s in spans if needle.lower() in s.text.lower()]
    if types is not None:
        hits = [s for s in hits if s.type in types]
    if presidio_only:
        hits = [s for s in hits if s.source.startswith("presidio")]
    return hits


def _raw_gliner_word_coverage(det, text, words):
    """Chunking isolation: raw GLiNER over _chunks (pre-dedupe, no Presidio — its full-text pass
    rescues spaCy-detectable names and would mask chunk splits). Returns the words NOT fully
    covered by any raw span. The defect signature of a chunk split is a lost word, not span
    shape: this model splits even an unchunked sentence-initial name into adjacent word spans."""
    from cloak.detect import _chunks

    spans = []
    try:
        pairs = list(_chunks(text, max_words=getattr(det, "max_words", None)))
    except TypeError:  # pre-fix checkout: _chunks has no max_words parameter
        pairs = list(_chunks(text))
    offsets, texts = zip(*pairs)
    for off, ents in zip(offsets, det.gliner.batch_predict_entities(
            list(texts), det.labels, threshold=det.threshold, batch_size=det.batch_size)):
        spans += [(off + e["start"], off + e["end"]) for e in ents]
    uncovered = []
    for w in words:
        i = text.lower().index(w.lower())
        if not any(s <= i and i + len(w) <= e for s, e in spans):
            uncovered.append(w)
    return uncovered


def run_cases(det) -> list[dict]:
    out = []

    def case(name, fired, detail):
        out.append({"case": name, "fired": bool(fired), "detail": detail})

    # stop-word deletion: a detected clinical "RN" span must survive the filter
    spans = det.detect("The RN administered the evening medication to the patient.")
    rn = _spans_on(spans, "rn")
    rn = [s for s in rn if s.text.lower().strip() == "rn"]
    case("rn-span-deleted", not rn, f"RN spans surviving filter: {len(rn)}")

    # Presidio REF_CODE on vitals: blood pressure is not a reference code
    spans = det.detect("BP was 120/80 today and stable.")
    case("bp-as-code", _spans_on(spans, "120/80", types={"CODE"}, presidio_only=True),
         "presidio CODE span on blood-pressure reading")

    # Presidio REF_CODE on year ranges
    spans = det.detect("The 2021/22 season was unusually busy for the ward.")
    case("year-range-as-code", _spans_on(spans, "2021/22", types={"CODE"}, presidio_only=True),
         "presidio CODE span on year range")

    # Presidio MONEY bare-k on distances (presidio-source only; a GLiNER quantity read is not the bug)
    spans = det.detect("He ran a 5k with his brother on Saturday.")
    case("bare-k-as-money", _spans_on(spans, "5k", types={"QUANTITY"}, presidio_only=True),
         "presidio QUANTITY (MONEY) span on run distance")

    # coref: two people sharing a surname collapse into one placeholder chain
    text = "Anna Smith met Peter Smith at the clinic reception."
    spans = coref_chains(text, det.detect(text))
    anna = next((s for s in spans if "anna" in s.text.lower()), None)
    peter = next((s for s in spans if "peter" in s.text.lower()), None)
    if anna and peter:
        case("surname-identity-merge", anna.chain == peter.chain,
             f"chains anna={anna.chain} peter={peter.chain}")
    else:
        case("surname-identity-merge", False, "model did not detect both names (case inconclusive)")

    # chunk boundary, single-word entity: the 1200-char hard cut lands INSIDE the word; the fix
    # backs off to the previous whitespace so the whole word lands in the next chunk intact.
    # Raw-GLiNER coverage (see _raw_gliner_word_coverage): Presidio would mask the split.
    filler = "the patient reported ongoing fatigue and mild joint stiffness, " * 19  # no '. ' anywhere
    text = filler[:1197] + "Sarah was seen in clinic today and follow-up was arranged."
    lost = _raw_gliner_word_coverage(det, text, ["Sarah"])
    case("chunk-splits-word", lost, f"name words with no raw-gliner span: {lost or 'none'}")

    # chunk boundary, multi-word entity straddling the cut BETWEEN its words: word-boundary
    # backoff alone cannot save it; the overlap window re-presents the whole name in the next
    # chunk (downstream, _dedupe merges the duplicate detections, widest wins).
    prefix = "word " * 238  # exactly 1190 chars, ends on a word boundary, no sentence breaks
    text = prefix + "Sarah Johnson was seen in clinic today and follow-up was arranged."
    lost = _raw_gliner_word_coverage(det, text, ["Sarah", "Johnson"])
    case("chunk-splits-multiword-name", lost, f"name words with no raw-gliner span: {lost or 'none'}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="data/models/pii_gliner_multidomain/checkpoint-2479")
    ap.add_argument("--tag", required=True, help="label for the output file, e.g. before/after")
    args = ap.parse_args()

    try:
        det = Detector(gliner_model=args.model, profile="clinical")
        profile = "clinical"
    except TypeError:  # pre-fix checkout: no profile parameter — that IS the before-behavior
        det = Detector(gliner_model=args.model)
        profile = "(pre-profile code)"

    results = run_cases(det)
    fired = [r for r in results if r["fired"]]
    out_path = Path(f"results/detector_red_tests/pipeline-{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "profile": profile, "tag": args.tag,
                                    "results": results}, indent=2, sort_keys=True) + "\n")

    print(f"\npipeline red tests  tag={args.tag}  profile={profile}")
    for r in results:
        print(f"  {'FIRED' if r['fired'] else 'clean'}  {r['case']:24s} {r['detail']}")
    print(f"fired: {len(fired)}/{len(results)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
