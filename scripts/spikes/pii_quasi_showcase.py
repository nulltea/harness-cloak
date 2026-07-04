"""Qualitative showcase: knowledgator/gliner-pii-base STOCK (zero-shot) vs FINE-TUNED (Arm B,
checkpoint-2756) on TAB test QUASI identifiers, at each model's operating point.

Shows concrete QUASI gold spans (MISC / DEM / QUANTITY — the gap types) and whether each detector
finds them (typed match). Stock @0.3 (its zero-shot setting), fine-tuned @0.02 (its dev-selected op).

Run: PYTHONPATH=src .venv/bin/python -u scripts/spikes/pii_quasi_showcase.py
"""
import json
import re

from cloak.detect import Detector

STOCK = "knowledgator/gliner-pii-base-v1.0"
FT = "data/models/pii_gliner/checkpoint-2756"
GAP = ("MISC", "DEM", "QUANTITY")


def quasi_golds(doc):
    best = {}
    for ann in doc["annotations"].values():
        for m in ann["entity_mentions"]:
            if m["identifier_type"] != "QUASI":
                continue
            best[(m["start_offset"], m["end_offset"])] = m
    return [(k[0], k[1], m["entity_type"]) for k, m in best.items()]


def sentence(text, s, e):
    a = max(text.rfind(". ", 0, s), text.rfind("\n", 0, s)) + 1
    b = min([x for x in (text.find(". ", e), text.find("\n", e)) if x != -1] + [len(text)])
    return re.sub(r"\s+", " ", text[a:b + 1]).strip()


def detect_all(model, thr, docs):
    import torch
    det = Detector(gliner_model=model, threshold=thr)
    out = [det.detect(d["text"]) for d in docs]
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def typed_hit(s, e, t, preds):
    return any(p.start < e and s < p.end and p.type == t for p in preds)


def main():
    docs = json.load(open("corpora/tab/echr_test.json"))[:40]
    stock = detect_all(STOCK, 0.3, docs)
    ft = detect_all(FT, 0.02, docs)

    rows = []  # (doc_i, s, e, type, surface, sent, stock_hit, ft_hit)
    counts = {t: [0, 0, 0] for t in GAP}  # type -> [n_gold, stock_typed, ft_typed]
    for i, doc in enumerate(docs):
        for s, e, t in quasi_golds(doc):
            if t not in GAP:
                continue
            sh, fh = typed_hit(s, e, t, stock[i]), typed_hit(s, e, t, ft[i])
            counts[t][0] += 1; counts[t][1] += sh; counts[t][2] += fh
            rows.append((i, s, e, t, doc["text"][s:e], sentence(doc["text"], s, e), sh, fh))

    print(f"=== typed recall on gap-type QUASI, {len(docs)} TAB-test docs "
          f"(stock@0.3 vs fine-tuned@0.02) ===")
    for t in GAP:
        n, sc, f = counts[t]
        print(f"  {t:9s} n={n:3d}  stock {sc/max(n,1):.2f}  fine-tuned {f/max(n,1):.2f}")

    def show(title, pred, k=6):
        picked = [r for r in rows if pred(r)][:k]
        print(f"\n=== {title} ({sum(1 for r in rows if pred(r))} total; showing {len(picked)}) ===")
        for i, s, e, t, surf, sent, sh, fh in picked:
            print(f"  [{t}] {surf!r}")
            print(f"     ctx: …{sent[:150]}…")
            print(f"     stock {'HIT ' if sh else 'miss'} | fine-tuned {'HIT' if fh else 'miss'}")

    show("FINE-TUNED CATCHES, STOCK MISSES (the QUASI win)", lambda r: r[7] and not r[6])
    show("both catch", lambda r: r[6] and r[7], k=3)
    show("both miss (honest residual)", lambda r: not r[6] and not r[7], k=3)


if __name__ == "__main__":
    main()
