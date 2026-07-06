"""Reader-fidelity ablation: {roberta-base, roberta-large} x {old scorer, new scorer}.

Reuses the SAME docs/gemma outputs as qa_reader_validation_smoke (cached -> zero new gemma
calls, no proxy contention). Only the reader model and the token-scorer vary.

- new scorer: canon()-normalize both sides (milligrams->mg, Dr.->doctor) + containment
  credit -- if the gold token-set is a subset of the answer token-set, score 1.0 (a
  verbose-but-correct answer like "AT&T Corporation" for "AT&T" is a hit, not a 0.8).
- stronger reader: deepset/roberta-large-squad2 vs the pinned base.

Reports per-config: ceiling pass, floor pass, reader-miss-on-present, abstention, mean hi.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/qa_reader_ablation_smoke.py [--per-corpus 2]
"""
import argparse
import json
import re
from pathlib import Path

from train_ranker import assemble

from cloak.corpora import load_task_docs
from cloak.train.reward import canon, token_f1
from cloak.train.roundtrip import roundtrip_batch

ENV = Path("data/ranker_env_full.json")
ARMS = Path("data/task_arms_full.json")
CORPORA = ("clinical", "lexsum", "wikibio")
TH = 0.5
READERS = {"base": "deepset/roberta-base-squad2", "large": "deepset/roberta-large-squad2"}

_models = {}


def reader(model_id, question, context, max_answer_toks=30):
    """Extractive best-span answer for `model_id` ('' = SQuAD2 no-answer). Cached per model."""
    import torch
    if model_id not in _models:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        m = AutoModelForQuestionAnswering.from_pretrained(model_id)
        m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _models[model_id] = (tok, m)
    tok, model = _models[model_id]
    enc = tok(question, context, return_tensors="pt", truncation="only_second",
              max_length=384, stride=128, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding=True)
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    inp = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**inp)
    best, best_score = "", -1e9
    for i in range(inp["input_ids"].shape[0]):
        s, e = out.start_logits[i], out.end_logits[i]
        null = (s[0] + e[0]).item()
        mask = torch.tensor([sid != 1 for sid in enc.sequence_ids(i)], device=s.device)
        s, e = s.masked_fill(mask, -1e9), e.masked_fill(mask, -1e9)
        si = int(s.argmax()); ei = si + int(e[si:si + max_answer_toks].argmax())
        sc = (s[si] + e[ei]).item()
        if sc > null and sc > best_score:
            lo, hi = offsets[i][si], offsets[i][ei]
            best, best_score = context[lo[0]:hi[1]], sc
    return best


def score_new(pred, gold):
    """canon-normalized token-F1 with containment credit (gold subset of pred -> 1.0)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=2)
    args = ap.parse_args()
    env = json.loads(ENV.read_text())["corpora"]
    arms = json.loads(ARMS.read_text())

    configs = [("base", "old"), ("base", "new"), ("large", "old"), ("large", "new")]
    agg = {c: {"n": 0, "hi_pass": 0, "lo_fail": 0, "present": 0, "miss": 0,
               "abstain": 0, "hi": 0.0} for c in configs}

    for corpus in CORPORA:
        picked = [(i, d) for i, d in env[corpus].items()
                  if d.get("spans") and d.get("probes", {}).get("train")][:args.per_corpus]
        for doc_id, d in picked:
            text = next(x["text"] for x in load_task_docs(corpus) if x["id"] == doc_id)
            R_walk = arms[corpus][doc_id]["tau_walk"][1]
            spans = d["spans"]
            ph = {s["surface"].lower(): next(a for a in s["actions"]
                                             if a["mode"] == "placeholder") for s in spans}
            lo_doc, lo_R = assemble(text, R_walk, spans, ph)
            jobs = [{"corpus": corpus, "doc_p": text, "R": [], "probes": []},
                    {"corpus": corpus, "doc_p": lo_doc, "R": lo_R, "probes": []}]
            hi_r, lo_r = roundtrip_batch(jobs, workers=2)   # cached from prior smoke
            out_hi, out_lo = hi_r["out_final"], lo_r["out_final"]
            for p in d["probes"]["train"]:
                a, q = p["surface"], p["question"]
                present = canon(a) in canon(out_hi)
                for rk in ("base", "large"):
                    a_hi = reader(READERS[rk], q, out_hi)
                    a_lo = reader(READERS[rk], q, out_lo)
                    for sc in ("old", "new"):
                        f = (token_f1 if sc == "old" else score_new)
                        hi, lo = f(a_hi, a), f(a_lo, a)
                        A = agg[(rk, sc)]
                        A["n"] += 1; A["hi"] += hi
                        A["hi_pass"] += hi >= TH; A["lo_fail"] += lo < TH
                        A["present"] += present; A["abstain"] += a_hi == ""
                        if present and hi < TH:
                            A["miss"] += 1

    print(f"\n{'config':<16}{'ceil_pass':<11}{'floor_pass':<12}{'reader_miss':<13}"
          f"{'abstain':<10}{'mean_hi':<8}")
    for c in configs:
        A = agg[c]; n = max(A["n"], 1); pres = max(A["present"], 1)
        print(f"{c[0]+'+'+c[1]:<16}{A['hi_pass']/n:<11.2%}{A['lo_fail']/n:<12.2%}"
              f"{A['miss']/pres:<13.2%}{A['abstain']/n:<10.2%}{A['hi']/n:<8.3f}")


if __name__ == "__main__":
    main()
