"""Audit for missed contextual relations on ACI: run the clinical detector with extra
zero-shot types (profession, nationality) plus the existing generalizable/anchor types,
dump per-doc detected spans, and tabulate co-occurrence of the new types with the
generalizable clinical spans (drug / health-condition / medical-procedure).

Read-only analysis: no substitution, no teacher call. Output feeds a written report and a
Sonnet eyeball pass. One GPU process.

Usage: .venv/bin/python -u scripts/spikes/relation_coverage_audit.py --n 15 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.detect import Detector, QA_V2_CLINICAL_LABELS

# gliner-pii-large is zero-shot: add natural-language label phrases for the candidate
# non-generalizable anchor types the audit is probing.
EXTRA_LABELS = {
    "profession": "profession",
    "nationality": "nationality",
    "ethnicity": "ethnicity",
    "religion": "religion",
    "employer organization": "employer-organization",
}
GENERALIZABLE = {"drug", "health-condition", "medical-procedure"}
MODEL = "knowledgator/gliner-pii-large-v1.0"
THRESHOLD = 0.35


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="aci")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scratch/relation_coverage_audit")
    args = ap.parse_args(argv)

    docs = load_task_docs(args.corpus)
    rng = random.Random(args.seed)
    sample = rng.sample(docs, min(args.n, len(docs)))

    labels = dict(QA_V2_CLINICAL_LABELS)
    labels.update(EXTRA_LABELS)
    det = Detector(gliner_model=MODEL, threshold=THRESHOLD, profile="clinical",
                   label2type=labels)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = []
    type_totals: Counter[str] = Counter()
    extra_types = set(EXTRA_LABELS.values())

    for row in sample:
        text = str(row.get("text", ""))
        spans = det.detect(text)
        by_type: dict[str, list[dict]] = {}
        for s in spans:
            type_totals[s.type] += 1
            by_type.setdefault(s.type, []).append(
                {"surface": s.text, "start": s.start, "end": s.end,
                 "raw_label": s.raw_label, "score": round(s.score, 3)})
        # spans of interest for the report
        extra_present = {t: by_type[t] for t in extra_types if t in by_type}
        gen_present = {t: by_type[t] for t in GENERALIZABLE if t in by_type}
        dump.append({
            "id": row.get("id"),
            "text": text,
            "extra_types": extra_present,
            "generalizable": gen_present,
            "all_by_type": by_type,
        })

    (out / "spans.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")

    # co-occurrence: which docs have both an extra-type span and a generalizable span
    co = [d["id"] for d in dump if d["extra_types"] and d["generalizable"]]
    extra_hits = Counter()
    for d in dump:
        for t in d["extra_types"]:
            extra_hits[t] += len(d["extra_types"][t])

    # readable per-doc dump for the Sonnet eyeball pass
    lines = []
    for d in dump:
        lines.append(f"===== DOC {d['id']} =====")
        lines.append(d["text"].strip())
        lines.append("")
        lines.append("DETECTED (extra candidate types): " + (
            json.dumps({t: [x["surface"] for x in v] for t, v in d["extra_types"].items()})
            if d["extra_types"] else "none"))
        lines.append("DETECTED (generalizable): " + json.dumps(
            {t: [x["surface"] for x in v] for t, v in d["generalizable"].items()}))
        lines.append("")
    (out / "docs.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "n_docs": len(sample),
        "doc_ids": [d["id"] for d in dump],
        "type_totals": dict(type_totals.most_common()),
        "extra_type_hits": dict(extra_hits.most_common()),
        "docs_with_extra_and_generalizable": co,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
