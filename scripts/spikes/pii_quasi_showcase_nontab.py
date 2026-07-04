"""Qualitative QUASI showcase on NON-TAB corpora (no PII gold — eyeball only): does the TAB
fine-tune's quasi-identifier detection transfer out of legal text?

Compares knowledgator STOCK (zero-shot @0.3) vs FINE-TUNED (Arm B checkpoint-2756 @0.02) on
clinical / email / social docs, focusing on the gap QUASI types MISC / DEM / QUANTITY.

Run: PYTHONPATH=src .venv/bin/python -u scripts/spikes/pii_quasi_showcase_nontab.py
"""
import json

from cloak.detect import Detector

STOCK, FT = "knowledgator/gliner-pii-base-v1.0", "data/models/pii_gliner/checkpoint-2756"
GAP = ("MISC", "DEM", "QUANTITY")
DOCS = [  # (label, path, line-index)
    ("clinical/mts", "corpora/clinical/mts.jsonl", 0),
    ("clinical/aci", "corpora/clinical/aci.jsonl", 0),
    ("enron/email", "corpora/enron/replies.jsonl", 0),
    ("synthpai#0", "corpora/synthpai/train.jsonl", 0),
    ("synthpai#5", "corpora/synthpai/train.jsonl", 5),
]


def load(path, idx):
    for i, line in enumerate(open(path)):
        if i == idx:
            return json.loads(line)["text"]


def run(model, thr, texts):
    import torch
    det = Detector(gliner_model=model, threshold=thr)
    out = [det.detect(t) for t in texts]
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def gap_spans(preds):
    return [(p.type, p.text) for p in preds if p.type in GAP]


def overlaps(a, b):  # same-type char overlap between two Span lists
    return any(x.start < y.end and y.start < x.end and x.type == y.type for x in a for y in b)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-thr", type=float, default=0.3)
    ap.add_argument("--ft-thr", type=float, default=0.02)
    args = ap.parse_args()
    texts = [load(p, i) for _, p, i in DOCS]
    stock = run(STOCK, args.stock_thr, texts)
    ft = run(FT, args.ft_thr, texts)
    print(f"STOCK = {STOCK} @{args.stock_thr}   |   FINE-TUNED = {FT} @{args.ft_thr}")
    for (label, _, _), text, sp, fp in zip(DOCS, texts, stock, ft):
        s_gap = [p for p in sp if p.type in GAP]
        f_gap = [p for p in fp if p.type in GAP]
        ft_only = [(p.type, p.text) for p in f_gap
                   if not any(p.start < q.end and q.start < p.end and p.type == q.type for q in s_gap)]
        st_only = [(p.type, p.text) for p in s_gap
                   if not any(p.start < q.end and q.start < p.end and p.type == q.type for q in f_gap)]
        shared = [(p.type, p.text) for p in f_gap
                  if any(p.start < q.end and q.start < p.end and p.type == q.type for q in s_gap)]
        print("=" * 84)
        print(f"{label}  (chars={len(text)})   QUASI-gap spans: stock={len(s_gap)} fine-tuned={len(f_gap)}")
        print(f"  excerpt: {text[:180].strip()!r}")
        def line(tag, items):
            items = sorted(set(items))
            print(f"  {tag}: " + ("; ".join(f"[{t}] {x!r}" for t, x in items) if items else "—"))
        line("FT-only (transfer: fine-tuned catches, stock misses)", ft_only)
        line("stock-only", st_only)
        line("shared", shared)


if __name__ == "__main__":
    main()
