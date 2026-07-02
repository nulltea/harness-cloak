"""Materialize task-oriented, PII-rich eval slices into corpora/.

Clinical (dialogue -> note): ACI-Bench + MTS-Dialog from
  github.com/microsoft/clinical_visit_note_summarization_corpus (pass --clinical-src <clone>).
Email (body -> subject): AESLC from github.com/ryanzhumich/AESLC (pass --aeslc-src <clone>);
  each .subject file is body + `@subject` gold + 3 `@ann` crowd references.

Writes corpora/clinical/{aci,mts}.jsonl and corpora/aeslc/test.jsonl.
Spec: docs/specs/benchmarks.md.

Run: PYTHONPATH=src .venv/bin/python -u scripts/build_task_corpora.py \
       --clinical-src /path/to/clinical_visit_note_summarization_corpus
"""
import argparse
import csv
import json
import sys
from pathlib import Path

OUT = Path("corpora")


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):5d} -> {path}")


def build_clinical(src: Path, n_aci: int, n_mts: int):
    csv.field_size_limit(1 << 24)
    aci = src / "data/aci-bench/challenge_data/train.csv"
    rows = list(csv.DictReader(open(aci, encoding="utf-8")))[:n_aci]
    _write(OUT / "clinical/aci.jsonl", [
        {"id": f"aci/{r['encounter_id']}", "corpus": "clinical",
         "text": r["dialogue"].strip(), "gold_ref": r["note"].strip()}
        for r in rows if r["dialogue"].strip() and r["note"].strip()])

    mts = src / "data/mts-dialog/MTS_Dataset_TrainingSet.csv"
    rows = list(csv.DictReader(open(mts, encoding="utf-8")))[:n_mts]
    _write(OUT / "clinical/mts.jsonl", [
        {"id": f"mts/{r['ID']}", "corpus": "clinical", "section_header": r["section_header"],
         "text": r["dialogue"].strip(), "gold_ref": r["section_text"].strip()}
        for r in rows if r["dialogue"].strip() and r["section_text"].strip()])


def _parse_aeslc(text: str) -> tuple[str, str, list[str]]:
    """-> (body, gold subject, [annotator refs]). File: body \n\n@subject\n.. \n\n@ann0\n.. """
    body, _, rest = text.partition("\n\n@subject\n")
    chunks = rest.split("\n\n@ann")
    subject = chunks[0].strip()
    anns = [c.split("\n", 1)[1].strip() for c in chunks[1:] if "\n" in c]
    return body.strip(), subject, [a for a in anns if a]


def build_aeslc(src: Path, n: int, split: str = "test"):
    files = sorted((src / "enron_subject_line" / split).glob("*.subject"))
    rows = []
    for f in files:
        body, subj, anns = _parse_aeslc(f.read_text(encoding="utf-8", errors="ignore"))
        if body and subj:
            rows.append({"id": f"aeslc/{f.stem}", "corpus": "aeslc", "text": body,
                         "gold_ref": subj, "gold_refs": anns or [subj]})
        if len(rows) >= n:
            break
    _write(OUT / f"aeslc/{split}.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-src", type=Path, help="clone of the microsoft clinical corpus")
    ap.add_argument("--aeslc-src", type=Path, help="clone of github.com/ryanzhumich/AESLC")
    ap.add_argument("--n-aci", type=int, default=67)
    ap.add_argument("--n-mts", type=int, default=200)
    ap.add_argument("--n-aeslc", type=int, default=200)
    args = ap.parse_args()

    if args.clinical_src:
        build_clinical(args.clinical_src, args.n_aci, args.n_mts)
    else:
        print("no --clinical-src; skipping clinical", file=sys.stderr)
    if args.aeslc_src:
        build_aeslc(args.aeslc_src, args.n_aeslc)
    else:
        print("no --aeslc-src; skipping aeslc", file=sys.stderr)


if __name__ == "__main__":
    main()
