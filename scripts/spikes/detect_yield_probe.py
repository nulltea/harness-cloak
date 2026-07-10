"""Spike: does knowledgator/gliner-pii-large-v1.0 (zero-shot, prompted with lattice-type label
phrases) yield spans that resolve to non-empty levels in lattice_profiles.json?

Local + free (GPU detector, no paid teacher calls). Purpose: establish the lattice-bearing span
yield per clinical doc, which sets the paid teacher-call count for the QA build.

Run: PYTHONPATH=src .venv/bin/python -u scripts/spikes/detect_yield_probe.py --n-docs 3
"""
import argparse
import collections

# zero-shot GLiNER label phrase -> lattice_profiles.json runtime_type (only lattice-bearing types)
LABELS = {
    "drug or medication": "drug",
    "disease, health condition or medical diagnosis": "health-condition",
    "medical procedure, test or imaging": "medical-procedure",
    "hospital, clinic or medical facility": "organization-medical-facility",
    "location, city or country": "LOC",
    "organization, company or institution": "ORG",
    "nationality or citizenship": "nationality",
    "profession, occupation or job title": "profession",
    "religion or religious belief": "religion",
    "gender": "gender",
    "marital status": "marital-status",
    "sexual orientation": "sexual-orientation",
}


def _chunks(text, max_words=320):
    words = text.split()
    out, off = [], 0
    for i in range(0, len(words), max_words):
        piece = " ".join(words[i:i + max_words])
        start = text.find(words[i], off) if i < len(words) else off
        out.append((max(start, 0), piece))
        off = start + len(piece)
    return out or [(0, text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--model", default="knowledgator/gliner-pii-large-v1.0")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    import torch
    from gliner import GLiNER

    from cloak.corpora import load_task_docs
    from cloak.lattice_profiles import lookup_count, lookup_levels

    g = GLiNER.from_pretrained(args.model)
    if torch.cuda.is_available():
        g = g.to("cuda")
    labels = list(LABELS)
    docs = load_task_docs("clinical", args.n_docs)

    per_type = collections.Counter()
    latticed_type = collections.Counter()
    total_spans = total_latticed = 0
    for d in docs:
        seen = set()
        doc_latticed = []
        for off, piece in _chunks(d["text"]):
            for e in g.predict_entities(piece, labels, threshold=args.threshold):
                surface = e["text"].strip()
                rtype = LABELS[e["label"]]
                key = (surface.lower(), rtype)
                if not surface or key in seen:
                    continue
                seen.add(key)
                total_spans += 1
                per_type[rtype] += 1
                levels = lookup_levels(surface, rtype)
                if levels:
                    total_latticed += 1
                    latticed_type[rtype] += 1
                    doc_latticed.append((surface, rtype, levels,
                                         [lookup_count(x, rtype) for x in levels]))
        print(f"\n[{d['id']}] {len(doc_latticed)} lattice-bearing / detected spans", flush=True)
        for surface, rtype, levels, counts in doc_latticed[:12]:
            print(f"   {rtype:28} {surface!r:24} -> {levels} {counts}", flush=True)

    print(f"\n=== yield over {len(docs)} clinical docs ===", flush=True)
    print(f"detected spans (deduped): {total_spans}; lattice-bearing: {total_latticed}", flush=True)
    print(f"by type (detected):      {dict(per_type)}", flush=True)
    print(f"by type (lattice-bearing): {dict(latticed_type)}", flush=True)
    print(f"lattice-bearing spans/doc: {total_latticed / max(len(docs), 1):.1f}", flush=True)
    print(f"=> ladder teacher calls for N docs ~= (this rate) * N; +1 decision call/doc", flush=True)


if __name__ == "__main__":
    main()
