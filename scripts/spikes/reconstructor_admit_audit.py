"""Pre-training admitted-target precision audit (plan gate: proceed only at >= ~0.95).
The builder persists (input, target) per row but not the per-edit detail, so recover each
admitted splice by word-diffing input->target: each replace region is (fill/quote-region ->
restored surface). Prints each admitted edit with local context for hand-judging whether the
surface is a correct restoration of that generalized mention (precision), across both corpora.

Run: PYTHONPATH=src .venv/bin/python scripts/spikes/reconstructor_admit_audit.py
"""
import difflib, json, re
from pathlib import Path


def _tokens(s):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", s)]


def admitted_edits(inp, tgt):
    """Word-diff input->target; yield (before_ctx, deleted, inserted, after_ctx) per replace."""
    pt, ct = _tokens(inp), _tokens(tgt)
    sm = difflib.SequenceMatcher(a=[inp[a:b] for a, b in pt],
                                 b=[tgt[a:b] for a, b in ct], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or i2 <= i1 or j2 <= j1:
            continue
        di1, di2 = pt[i1][0], pt[i2 - 1][1]
        ji1, ji2 = ct[j1][0], ct[j2 - 1][1]
        yield (inp[max(0, di1 - 40):di1], inp[di1:di2], tgt[ji1:ji2], inp[di2:di2 + 40])


def main():
    edits = []
    for corpus in ("clinical", "lexsum"):
        p = Path(f"data/reconstructor_{corpus}.jsonl")
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["is_noop"]:
                continue
            for before, deleted, inserted, after in admitted_edits(r["input"], r["target"]):
                edits.append((corpus, r["doc_id"], before, deleted, inserted, after))
    print(f"TOTAL admitted edits recovered by diff: {len(edits)}\n")
    for i, (corpus, doc, before, deleted, inserted, after) in enumerate(edits):
        print(f"[{i:02d}] {corpus}/{doc}")
        print(f"     fill/quote-region: ...{before!r} [{deleted!r}] {after!r}...")
        print(f"     -> restored surface: {inserted!r}\n")


if __name__ == "__main__":
    main()
