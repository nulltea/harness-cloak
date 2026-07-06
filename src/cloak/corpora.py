"""Task-oriented, PII-rich eval corpora (built by scripts/build_task_corpora.py).

Each doc: {id, corpus, text (input), gold_ref (reference output), gold_refs? (multi-ref)}.
- clinical (aci, mts): dialogue -> visit note / section text
- aeslc: email body -> subject line (gold_refs = 3 crowd annotations)
Spec: docs/specs/benchmarks.md.
"""
import json
from pathlib import Path

CORPORA = Path("corpora")
FILES = {
    "aci": ["clinical/aci.jsonl"],
    "mts": ["clinical/mts.jsonl"],
    "clinical": ["clinical/aci.jsonl", "clinical/mts.jsonl"],
    "aeslc": ["aeslc/test.jsonl"],
    "enron": ["enron/replies.jsonl"],
    "lexsum": ["lexsum/val.jsonl"],
    "wikibio": ["wikibio/val.jsonl"],
    "qmsum": ["qmsum/val.jsonl"],
}


def load_task_docs(corpus: str, n: int | None = None) -> list[dict]:
    """Deterministic slice. Multi-file corpora (clinical) round-robin across files so a
    slice hits every sub-corpus; single-file corpora keep file order."""
    per_file = [[json.loads(l) for l in open(CORPORA / rel, encoding="utf-8")]
                for rel in FILES[corpus]]
    if len(per_file) == 1:
        rows = per_file[0]
    else:
        from itertools import chain, zip_longest
        rows = [d for d in chain.from_iterable(zip_longest(*per_file)) if d is not None]
    return rows[:n] if n else rows


def refs_of(doc: dict) -> list[str]:
    """Reference outputs for scoring (multi-ref if present)."""
    return doc.get("gold_refs") or [doc["gold_ref"]]


if __name__ == "__main__":
    for c in ["aci", "mts", "aeslc"]:
        d = load_task_docs(c, 2)
        assert d and d[0]["text"] and refs_of(d[0]), c
        print(f"{c:8} n={len(load_task_docs(c))} refs0={len(refs_of(d[0]))} "
              f"gold0={refs_of(d[0])[0][:60]!r}")
    print("corpora.py self-check OK")
