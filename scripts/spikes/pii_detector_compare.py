"""Qualitative side-by-side: two GLiNER checkpoints as the LatticeCloak detector on sample docs.
Shows which spans each catches and the diff (B-only = B's wins over A). Presidio is identical in
both, so the diff is purely the GLiNER-model contribution.

Run: PYTHONPATH=src .venv/bin/python -u scripts/pii_detector_compare.py
"""
import json

from cloak.detect import Detector

DOCS = [  # (label, path, line index) — first record of each, deterministic (no cherry-picking)
    ("clinical/aci", "corpora/clinical/aci.jsonl", 0),
    ("enron/email", "corpora/enron/replies.jsonl", 0),
    ("synthpai", "corpora/synthpai/train.jsonl", 0),
]
MODEL_A = "urchade/gliner_small-v2.1"
MODEL_B = "knowledgator/gliner-pii-base-v1.0"


def load_text(path, idx):
    with open(path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)["text"]


def run(model, texts):
    import torch
    det = Detector(gliner_model=model)
    out = [det.detect(t) for t in texts]
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def overlaps_same_type(s, spans):
    return any(s.start < o.end and o.start < s.end and s.type == o.type for o in spans)


def main():
    texts = [load_text(p, i) for _, p, i in DOCS]
    print(f"A = {MODEL_A}\nB = {MODEL_B}\n")
    A = run(MODEL_A, texts)
    B = run(MODEL_B, texts)

    for (label, _, _), text, a, b in zip(DOCS, texts, A, B):
        b_only = [s for s in b if not overlaps_same_type(s, a)]  # B catches, A misses
        a_only = [s for s in a if not overlaps_same_type(s, b)]  # A catches, B misses
        print("=" * 78)
        print(f"{label}  (chars={len(text)})  |  A spans={len(a)}  B spans={len(b)}  "
              f"B-only={len(b_only)}  A-only={len(a_only)}")
        print(f"  excerpt: {text[:160].strip()!r}...")
        def show(tag, spans):
            spans = sorted(spans, key=lambda s: (s.type, s.start))
            print(f"  {tag} ({len(spans)}): " + (", ".join(
                f"[{s.type}:{s.source[0]}] {s.text!r}" for s in spans) or "—"))
        show("B-only (knowledgator catches, small misses)", b_only)
        show("A-only (small catches, knowledgator misses)", a_only)
        # type-count deltas
        from collections import Counter
        ca, cb = Counter(s.type for s in a), Counter(s.type for s in b)
        deltas = {t: cb[t] - ca[t] for t in set(ca) | set(cb) if cb[t] != ca[t]}
        print(f"  per-type count Δ (B−A): {deltas or '—'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
