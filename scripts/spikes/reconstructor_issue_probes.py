"""CPU-only probes: confirm/refute claimed reconstructor design flaws against CURRENT code
(no GPU, no proxy, no stale artifacts). Each probe prints CONFIRMED/REFUTED with evidence.

Probes:
  A. edit_guard over-deletion — does the inference guard accept a candidate that deletes
     non-target content around the mention (superset splice)?
  B. admission-gate holes — empty-fill/UNKNOWN admitted? fine types bypass correspondence?
  C. truncation — fraction of current training inputs exceeding flan-t5's 1024-token window.
  D. all-or-nothing — one bad edit region rejects the whole candidate incl. good edits.

Run: PYTHONPATH=src .venv/bin/python scripts/spikes/reconstructor_issue_probes.py
"""
import json
from pathlib import Path

from cloak.reconstruct import edit_guard, restorable, splice_at_quote

def probe_a_over_deletion():
    prepass = ("Two teenage males sued the organization and the state in the United States "
               "District Court under Section 1983.")
    res = [{"surface": "Hamilton County Juvenile Court",
            "replacement": "an organization", "type": "ORG"}]
    tight = prepass.replace("the organization", "Hamilton County Juvenile Court")
    over1 = prepass.replace("the organization and the state",
                            "Hamilton County Juvenile Court")
    over2 = prepass.replace("the organization and the state in the United States District Court",
                            "Hamilton County Juvenile Court")
    a = edit_guard(prepass, tight, res, max_edits=3)
    b = edit_guard(prepass, over1, res, max_edits=3)
    c = edit_guard(prepass, over2, res, max_edits=3)
    print(f"A. guard: tight splice accepted={a} (want True)")
    print(f"   guard: +3-word deletion accepted={b}  +10-word deletion accepted={c}")
    print(f"   -> over-deletion hole: {'CONFIRMED' if (b or c) else 'REFUTED'}")

def probe_b_gate_holes():
    # B1: empty fill / UNKNOWN type
    prepass = "follow-up regarding persistently high blood pressure despite adjustments."
    admitted = restorable({"surface": "pressures", "replacement": "", "type": "UNKNOWN"},
                          {"label": "REWORDED", "quote": "pressure"}, prepass, nli=None)
    print(f"B1. empty-fill UNKNOWN admitted={admitted} -> "
          f"{'CONFIRMED hole' if admitted else 'REFUTED'}")
    # B2: fine type bypasses correspondence — quote adds specificity beyond the fill,
    # and splice-at-quote would overwrite the whole compound.
    prepass2 = "History includes Heart Failure and Type 2 Diabetes Mellitus, on Metformin."
    e = {"surface": "diabetes", "replacement": "an endocrine condition",
         "type": "health-condition"}
    v = {"label": "REWORDED", "quote": "Type 2 Diabetes Mellitus"}
    admitted2 = restorable(e, v, prepass2, nli=None)   # nli=None: correspondence would fail-close
    spliced, ok = splice_at_quote(prepass2, v["quote"], e["surface"])
    print(f"B2. fine-type (health-condition) admitted WITHOUT correspondence={admitted2}")
    print(f"    resulting training target: {spliced!r}" if ok else "    (no splice)")
    print(f"    -> fine-type correspondence bypass: {'CONFIRMED' if admitted2 else 'REFUTED'}")
    # B3: sanity — ambiguous type correctly fail-closed without NLI
    ok3 = restorable({"surface": "Boston", "replacement": "a city", "type": "LOC"},
                     {"label": "REWORDED", "quote": "Boston"}, "sued in Boston court.", nli=None)
    print(f"B3. LOC fill⊨quote fail-closed w/o NLI: admitted={ok3} (want False)")

def probe_c_truncation():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    except Exception as exc:
        print(f"C. tokenizer unavailable ({exc}); skipped")
        return
    PROMPT = ("Restore the original terms in the CLINICAL/LEGAL answer below. Replace each "
              "generalized mention with its original from the RESTORE map; copy everything else "
              "verbatim; if a mapped term is not present, leave the text unchanged.\n\n")
    for c in ("clinical", "lexsum"):
        p = Path(f"data/reconstructor_{c}.jsonl")
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        lens = [len(tok(PROMPT + r["input"]).input_ids) for r in rows]
        over = sum(1 for n in lens if n > 1024)
        print(f"C. {c}: {over}/{len(rows)} inputs exceed 1024 tokens "
              f"(median {sorted(lens)[len(lens)//2]}, max {max(lens)})")

def probe_d_all_or_nothing():
    prepass = "Patient has a disease. Patient takes a drug daily."
    res = [{"surface": "arthritis", "replacement": "a disease", "type": "health-condition"},
           {"surface": "lasix", "replacement": "a drug", "type": "MISC"}]
    # one perfect edit + one paraphrase drift elsewhere -> whole output rejected
    cand = "Patient has arthritis. Patient uses a drug daily."
    ok = edit_guard(prepass, cand, res, max_edits=5)
    print(f"D. good edit + unrelated 1-word drift: accepted={ok} -> "
          f"{'CONFIRMED all-or-nothing (good edit discarded)' if not ok else 'REFUTED'}")

if __name__ == "__main__":
    probe_a_over_deletion()
    probe_b_gate_holes()
    probe_c_truncation()
    probe_d_all_or_nothing()
