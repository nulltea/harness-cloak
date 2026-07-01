"""Build a CNN/DailyMail continuation corpus (InferDPT §VI protocol).

Per article: GPT-2-tokenize, take the first 50 tokens as the prefix (raw document) and
the next 100 tokens as the gold human continuation (the paper-faithful MAUVE reference).
Streams the test split (no full download). Writes JSONL {"prefix":..., "gold":...}.

Run: PYTHONPATH=src python scripts/build_cnndm_corpus.py --n 200 --out corpora/cnndm.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PREFIX_TOK, GOLD_TOK = 50, 100
_BOILER = re.compile(r"^\(CNN\)\s*(--\s*)?")


def build(n: int, out: str, split: str = "test") -> None:
    import datasets
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")  # paper counts tokens with GPT-2
    ds = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split=split, streaming=True)

    rows, need = [], PREFIX_TOK + GOLD_TOK
    for ex in ds:
        art = _BOILER.sub("", ex["article"]).strip()
        ids = tok(art, add_special_tokens=False).input_ids
        if len(ids) < need:
            continue
        prefix = tok.decode(ids[:PREFIX_TOK]).strip()
        gold = tok.decode(ids[PREFIX_TOK:need]).strip()
        if prefix and gold:
            rows.append({"prefix": prefix, "gold": gold})
        if len(rows) >= n:
            break

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} examples → {out}")
    print("--- sample ---")
    print("prefix:", rows[0]["prefix"])
    print("gold  :", rows[0]["gold"][:120], "...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="corpora/cnndm.jsonl")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    build(args.n, args.out, args.split)
