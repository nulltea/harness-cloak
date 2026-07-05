"""Materialize task-oriented, PII-rich eval slices into corpora/.

Clinical (dialogue -> note): ACI-Bench + MTS-Dialog from
  github.com/microsoft/clinical_visit_note_summarization_corpus (pass --clinical-src <clone>).
Email (body -> subject): AESLC from github.com/ryanzhumich/AESLC (pass --aeslc-src <clone>);
  each .subject file is body + `@subject` gold + 3 `@ann` crowd references.
Email (email -> reply): Enron maildir (pass --enron-src <maildir>). CMU Enron strips threading
  headers, so pairs come from quoted top-posted replies within one message: gold = the new (top)
  text, parent = the quoted original ("-----Original Message-----" or ">"). Heavier restatement
  than subject lines.

Case summary (long -> short): Multi-LexSum (HF allenai/multi_lexsum, pass --lexsum). doc = long
  summary, gold = short summary; condensing forces restatement of party/org names. Also writes a
  restatement-proxy report to results/lexsum_restatement.json.

Writes corpora/clinical/{aci,mts}.jsonl, corpora/aeslc/test.jsonl, corpora/lexsum/val.jsonl.
Spec: docs/specs/benchmarks.md.

Run: PYTHONPATH=src .venv/bin/python -u scripts/build_task_corpora.py \
       --clinical-src /path/to/clinical_visit_note_summarization_corpus
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

OUT = Path("corpora")
RESULTS = Path("results")


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


_SEP = __import__("re").compile(
    r"^\s*(?:-{2,}\s*Original Message|_{5,}|-{5,}\s*$|On\b.+\bwrote:|.+\bwrote:\s*$)",
    __import__("re").IGNORECASE)
_HDR = __import__("re").compile(r"^\s*(?:From|To|Sent|Cc|Bcc|Subject|Date|Importance):",
                                __import__("re").IGNORECASE)


def _split_reply(body: str) -> tuple[str, str]:
    """(reply new text, quoted parent body). '' if no top-posted quote found."""
    import re
    lines = body.splitlines()
    idx = mode = None
    for i, ln in enumerate(lines):
        if _SEP.match(ln):
            idx, mode = i, "sep"
            break
        if ln.lstrip().startswith(">"):
            idx, mode = i, "quote"
            break
    if idx is None or idx == 0:  # nothing before the quote = not a top-posted reply
        return "", ""
    reply = "\n".join(lines[:idx]).strip()
    rest = lines[idx + 1:] if mode == "sep" else lines[idx:]
    parent = []
    for ln in rest:
        if mode == "quote":
            if ln.lstrip().startswith(">"):
                parent.append(re.sub(r"^\s*>+\s?", "", ln))
        elif not _HDR.match(ln):  # drop the quoted header block, keep the original body
            parent.append(ln)
    return reply, "\n".join(parent).strip()


def build_enron(maildir: Path, n: int, scan_limit: int, min_body: int = 200,
                max_body: int = 4000, min_reply: int = 30, max_reply: int = 800):
    from email.parser import BytesParser

    rows = []
    for i, p in enumerate(maildir.rglob("*")):
        if i >= scan_limit or len(rows) >= n:
            break
        if not p.is_file():
            continue
        try:
            m = BytesParser().parse(open(p, "rb"))
        except Exception:
            continue
        payload = m.get_payload()
        if not isinstance(payload, str):
            continue
        reply, parent = _split_reply(payload)
        if min_reply <= len(reply) <= max_reply and min_body <= len(parent) <= max_body:
            rows.append({"id": f"enron/{len(rows)}", "corpus": "enron",
                         "text": parent, "gold_ref": reply})
    _write(OUT / "enron/replies.jsonl", rows)


def build_lexsum(n: int = 200, split_file: str = "dev.json",
                 min_chars: int = 800, max_chars: int = 4000):
    """Multi-LexSum (allenai/multi_lexsum): doc = long summary, gold = short summary.
    Condensing the long into the short forces restatement of party/org names by construction.
    The HF dataset ships a legacy loading script (unsupported in datasets>=3), so we fetch the
    release's validation subset file (JSONL) directly from the hub."""
    import json as _json

    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files("allenai/multi_lexsum", repo_type="dataset")
    releases = sorted({f.split("/")[1] for f in files if f.startswith("releases/")})
    release = releases[-1]  # latest release config
    rel = f"releases/{release}/{split_file}"
    print(f"lexsum: releases={releases} using={release} split={rel}", file=sys.stderr)
    path = hf_hub_download("allenai/multi_lexsum", rel, repo_type="dataset")

    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        ex = _json.loads(line)
        long_s = (ex.get("summary/long") or "").strip()
        short_s = (ex.get("summary/short") or "").strip()
        if not (long_s and short_s):
            continue
        if not (min_chars <= len(long_s) <= max_chars):
            continue
        rows.append({"id": f"lexsum/{ex.get('case_id', len(rows))}", "corpus": "lexsum",
                     "text": long_s, "gold_ref": short_s})
        if len(rows) >= n:
            break
    _write(OUT / "lexsum/val.jsonl", rows)
    _restatement_report(rows)


_CAPSEQ = re.compile(r"\b[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*)+")


def _restatement_report(rows: list[dict]):
    """Cheap proxy (pending detection): per doc, fraction of capitalized multi-word sequences
    (>=2 tokens, e.g. party/org names) in the long summary that reappear (case-insensitive) in
    the short summary. Writes results/lexsum_restatement.json."""
    per_doc = []
    for r in rows:
        names = set(_CAPSEQ.findall(r["text"]))
        short_l = r["gold_ref"].lower()
        frac = sum(1 for nm in names if nm.lower() in short_l) / len(names) if names else 0.0
        per_doc.append(frac)
    mean = sum(per_doc) / len(per_doc) if per_doc else 0.0
    lo = min(per_doc) if per_doc else 0.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lexsum_restatement.json").write_text(json.dumps(
        {"n_docs": len(per_doc), "mean": mean, "min": lo, "per_doc": per_doc}, indent=2))
    flag = "  [FLAG: mean < 0.3, restatement weak]" if mean < 0.3 else ""
    print(f"restatement: n={len(per_doc)} mean={mean:.3f} min={lo:.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-src", type=Path, help="clone of the microsoft clinical corpus")
    ap.add_argument("--aeslc-src", type=Path, help="clone of github.com/ryanzhumich/AESLC")
    ap.add_argument("--enron-src", type=Path, help="Enron maildir root (CMU enron_mail)")
    ap.add_argument("--n-aci", type=int, default=67)
    ap.add_argument("--n-mts", type=int, default=200)
    ap.add_argument("--n-aeslc", type=int, default=200)
    ap.add_argument("--n-enron", type=int, default=200)
    ap.add_argument("--enron-scan-limit", type=int, default=120000)
    ap.add_argument("--lexsum", action="store_true", help="build lexsum (HF download)")
    ap.add_argument("--n-lexsum", type=int, default=200)
    args = ap.parse_args()

    if args.clinical_src:
        build_clinical(args.clinical_src, args.n_aci, args.n_mts)
    else:
        print("no --clinical-src; skipping clinical", file=sys.stderr)
    if args.aeslc_src:
        build_aeslc(args.aeslc_src, args.n_aeslc)
    else:
        print("no --aeslc-src; skipping aeslc", file=sys.stderr)
    if args.enron_src:
        build_enron(args.enron_src, args.n_enron, args.enron_scan_limit)
    else:
        print("no --enron-src; skipping enron", file=sys.stderr)
    if args.lexsum:
        build_lexsum(args.n_lexsum)
    else:
        print("no --lexsum; skipping lexsum", file=sys.stderr)


if __name__ == "__main__":
    main()
