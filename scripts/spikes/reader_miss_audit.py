"""Audit reader misses on PRESENT spans (roberta-large + canon/containment scorer).

For each train probe whose gold surface IS present in out_hi (the ceiling note) but the
reader still scores < TH, categorize the failure and show context:
  - abstain      : reader returned "" (SQuAD2 no-answer span won)
  - wrong_entity : reader returned a non-empty span with ZERO overlap (picked something else)
  - partial      : 0 < f1 < TH (got some tokens, missed the rest)

Ceiling anchors are cached from stage 3 -> no new gemma calls. Prints per-category counts
+ examples (question / gold / reader answer / the out_hi sentence that contains the gold).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/reader_miss_audit.py [--per-corpus 10] [--examples 6]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.train.reward import _sentences, canon
from cloak.train.roundtrip import roundtrip_batch

ENV = Path("data/ranker_env_full.json")
CORPORA = ("clinical", "lexsum", "wikibio")
TH = 0.5
READER = "deepset/roberta-large-squad2"
_m = {}


def reader(question, context, max_answer_toks=30):
    import torch
    if READER not in _m:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(READER)
        mdl = AutoModelForQuestionAnswering.from_pretrained(READER)
        mdl.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _m[READER] = (tok, mdl)
    tok, mdl = _m[READER]
    enc = tok(question, context, return_tensors="pt", truncation="only_second",
              max_length=384, stride=128, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding=True)
    offs = enc.pop("offset_mapping"); enc.pop("overflow_to_sample_mapping", None)
    inp = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**inp)
    best, best_sc = "", -1e9
    for i in range(inp["input_ids"].shape[0]):
        s, e = out.start_logits[i], out.end_logits[i]
        null = (s[0] + e[0]).item()
        mask = __import__("torch").tensor([sid != 1 for sid in enc.sequence_ids(i)],
                                          device=s.device)
        s, e = s.masked_fill(mask, -1e9), e.masked_fill(mask, -1e9)
        si = int(s.argmax()); ei = si + int(e[si:si + max_answer_toks].argmax())
        sc = (s[si] + e[ei]).item()
        if sc > null and sc > best_sc:
            lo, hi = offs[i][si], offs[i][ei]
            best, best_sc = context[lo[0]:hi[1]], sc
    return best


def score_new(pred, gold):
    p = re.findall(r"\w+", canon(pred)); g = re.findall(r"\w+", canon(gold))
    if not g:
        return float(not p)
    if not p:
        return 0.0
    if set(g) <= set(p):
        return 1.0
    common = sum(min(p.count(t), g.count(t)) for t in set(g))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_ctx(gold, out_hi):
    k = canon(gold)
    for s in _sentences(out_hi):
        if k in canon(s):
            return s.strip()[:160]
    # fall back to a char window around the raw match
    i = out_hi.lower().find(gold.lower())
    return out_hi[max(0, i - 60):i + 60].replace("\n", " ") if i >= 0 else "(window n/a)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=10)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--full", type=int, default=0,
                    help="dump the full out_final (gold «marked») for the first N per cause")
    args = ap.parse_args()
    env = json.loads(ENV.read_text())["corpora"]
    buckets = defaultdict(list)   # cause -> [(corpus, doc, q, gold, ans, ctx)]
    tally = defaultdict(int)
    n_present = 0

    for corpus in CORPORA:
        docs = [(i, d) for i, d in env[corpus].items()
                if d.get("spans") and d.get("probes", {}).get("train")][:args.per_corpus]
        texts = {x["id"]: x["text"] for x in load_task_docs(corpus)}
        for doc_id, d in docs:
            out_hi = roundtrip_batch(
                [{"corpus": corpus, "doc_p": texts[doc_id], "R": [], "probes": []}],
                workers=1)[0]["out_final"]   # cached
            for p in d["probes"]["train"]:
                a, q = p["surface"], p["question"]
                if canon(a) not in canon(out_hi):
                    continue
                n_present += 1
                ans = reader(q, out_hi)
                f1 = score_new(ans, a)
                if f1 >= TH:
                    continue
                cause = ("abstain" if ans == "" else
                         "wrong_entity" if f1 == 0 else "partial")
                tally[cause] += 1
                buckets[cause].append((corpus, doc_id, q, a, ans, out_hi))

    total_miss = sum(tally.values())
    print(f"\nPRESENT spans audited: {n_present} | misses: {total_miss} "
          f"({total_miss / max(n_present,1):.1%})")
    for cause in ("abstain", "wrong_entity", "partial"):
        print(f"  {cause:<13}{tally[cause]:>4}  ({tally[cause]/max(total_miss,1):.0%} of misses)")
    for cause in ("abstain", "wrong_entity", "partial"):
        print("\n" + "=" * 96 + f"\n### {cause}  ({tally[cause]})")
        for corpus, doc, q, gold, ans, out_hi in buckets[cause][:args.examples]:
            print(f"[{corpus}] gold={gold!r}  reader={ans!r}")
            print(f"    q:   {q[:80]!r}")
            print(f"    ctx: {gold_ctx(gold, out_hi)!r}")

    # full out_final dumps (gold «marked») for eyeballing fact placement vs the question
    for cause in ("abstain", "wrong_entity", "partial"):
        for corpus, doc, q, gold, ans, out_hi in buckets[cause][:args.full]:
            marked = re.sub(rf"({re.escape(gold)})", r"«\1»", out_hi, flags=re.IGNORECASE)
            print("\n" + "#" * 96)
            print(f"[{cause}] {corpus} {doc}")
            print(f"QUESTION : {q}")
            print(f"GOLD     : {gold!r}   READER: {ans!r}")
            print(f"--- out_final ({len(out_hi)} chars, gold «marked») ---")
            print(marked)


if __name__ == "__main__":
    main()
