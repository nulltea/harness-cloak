#!/usr/bin/env python3
"""Report QA-build review flags and re-select documents worth re-processing.

Reads one or more built utility artifacts and prints their per-document review
flags (missing generalization, fixable rejections, ledger inconsistency, …).
With --fix-class, lists just the document ids carrying a flag of that class, so
after a data/teacher/reader fix you can re-run exactly the affected documents,
e.g.:

  python scripts/qa_review_flags.py /tmp/*.json --fix-class data_lattice
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloak.train.qa_builder import compute_review_flags


def _flags_for(artifact: dict) -> dict[str, list[dict]]:
    # Prefer the flags recorded at build time; fall back to recomputing so the
    # tool works on older artifacts too.
    recorded = artifact.get("review_flags")
    return recorded if isinstance(recorded, dict) else compute_review_flags(artifact)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="+", help="built utility artifact json files")
    ap.add_argument("--fix-class", default=None,
                    help="list only doc ids with a flag of this fix_class "
                         "(data_lattice / teacher_redraw / reader / ontology_review)")
    ap.add_argument("--code", default=None, help="filter to a single flag code")
    args = ap.parse_args()

    selected: list[tuple[str, str]] = []  # (artifact_path, doc_id)
    for path in args.artifacts:
        artifact = json.loads(Path(path).read_text())
        for doc_id, flags in sorted(_flags_for(artifact).items()):
            flags = [f for f in flags
                     if (args.fix_class is None or f.get("fix_class") == args.fix_class)
                     and (args.code is None or f.get("code") == args.code)]
            if not flags:
                continue
            if args.fix_class or args.code:
                selected.append((path, doc_id))
                continue
            codes = Counter(f["code"] for f in flags)
            classes = sorted({f["fix_class"] for f in flags})
            print(f"{doc_id}  [{','.join(classes)}]")
            for code, n in codes.most_common():
                detail = next((f["detail"] for f in flags
                               if f["code"] == code and f["detail"]), {})
                extra = f"  {detail}" if detail else ""
                print(f"    {code} x{n}{extra}")

    if args.fix_class or args.code:
        for path, doc_id in selected:
            print(doc_id)
        print(f"\n{len(selected)} document(s) selected", file=sys.stderr)


if __name__ == "__main__":
    main()
