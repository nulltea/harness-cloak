"""Validate the QA reader (deepset/roberta-base-squad2) in isolation from the task model.

Per span-bearing doc, 3 gemma conditions + the reader on each:
    out_hi = gemma(task(doc_orig))                       # ceiling: fact present
    out_lo = invert(gemma(task(all_placeholder)), R_lo)  # floor:   fact redacted
    out_tf = invert(gemma(task(assemble(doc, BC))), R)   # policy round trip (floor-walk)
    hi/lo/tf f1 = token_f1(reader(q, out_X), a)  per probe (a = gold surface)

The point is to SEPARATE reader error from task-model error:
  - present? = gold `a` is a substring of out_hi. If present but hi_f1 low -> READER miss.
    If not present -> gemma didn't restate (task model), not the reader's fault.
  - abstain  = reader returned "" (SQuAD2 no-answer/CLS span won).

Prints a per-probe table + aggregates (ceiling/floor pass rates, reader-miss-on-present
rate, abstention rate, mean dynamic range hi-lo).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/qa_reader_validation_smoke.py [--per-corpus 2] [--max-probes 6]
"""
import argparse
import json
from pathlib import Path

from train_ranker import assemble

from cloak.corpora import load_task_docs
from cloak.train.reward import _qa_answer, token_f1
from cloak.train.roundtrip import roundtrip_batch

ENV = Path("data/ranker_env_full.json")
ARMS = Path("data/task_arms_full.json")
CORPORA = ("clinical", "lexsum", "wikibio")
TH = 0.5


def trainable_docs(env_corpus, n):
    out = []
    for doc_id, d in env_corpus.items():
        if d.get("spans") and d.get("probes", {}).get("train"):
            out.append((doc_id, d))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=2)
    ap.add_argument("--max-probes", type=int, default=6, help="probe rows shown per doc")
    args = ap.parse_args()

    env = json.loads(ENV.read_text())["corpora"]
    arms = json.loads(ARMS.read_text())
    agg = {"n": 0, "hi_pass": 0, "lo_fail": 0, "present": 0, "reader_miss": 0,
           "abstain_hi": 0, "hi": 0.0, "lo": 0.0, "tf": 0.0, "dyn": 0.0}

    for corpus in CORPORA:
        for doc_id, d in trainable_docs(env[corpus], args.per_corpus):
            text = next(x["text"] for x in load_task_docs(corpus) if x["id"] == doc_id)
            R_walk = arms[corpus][doc_id]["tau_walk"][1]
            spans = d["spans"]
            probes = d["probes"]["train"]
            ph = {s["surface"].lower(): next(a for a in s["actions"]
                                             if a["mode"] == "placeholder") for s in spans}
            bc = {s["surface"].lower(): s["actions"][s["bc_action"]] for s in spans}
            lo_doc, lo_R = assemble(text, R_walk, spans, ph)
            tf_doc, tf_R = assemble(text, R_walk, spans, bc)
            jobs = [{"corpus": corpus, "doc_p": text,   "R": [],   "probes": []},
                    {"corpus": corpus, "doc_p": lo_doc, "R": lo_R, "probes": []},
                    {"corpus": corpus, "doc_p": tf_doc, "R": tf_R, "probes": []}]
            hi_r, lo_r, tf_r = roundtrip_batch(jobs, workers=3)
            out_hi, out_lo, out_tf = hi_r["out_final"], lo_r["out_final"], tf_r["out_final"]

            print("\n" + "=" * 100)
            print(f"[{corpus}] {doc_id}  probes={len(probes)}  "
                  f"out_hi[:90]={out_hi[:90]!r}")
            print(f"{'gold':<22}{'pres':<5}{'hi':<5}{'lo':<5}{'tf':<5}  reader_hi_ans")
            for p in probes:
                a, q = p["surface"], p["question"]
                a_hi, a_lo, a_tf = (_qa_answer(q, out_hi), _qa_answer(q, out_lo),
                                    _qa_answer(q, out_tf))
                hi, lo, tf = (token_f1(a_hi, a), token_f1(a_lo, a), token_f1(a_tf, a))
                present = a.lower() in out_hi.lower()
                agg["n"] += 1
                agg["hi"] += hi; agg["lo"] += lo; agg["tf"] += tf; agg["dyn"] += hi - lo
                agg["hi_pass"] += hi >= TH
                agg["lo_fail"] += lo < TH
                agg["present"] += present
                agg["abstain_hi"] += a_hi == ""
                if present and hi < TH:
                    agg["reader_miss"] += 1
                if probes.index(p) < args.max_probes:
                    print(f"{a[:20]!r:<22}{'Y' if present else 'n':<5}"
                          f"{hi:<5.2f}{lo:<5.2f}{tf:<5.2f}  {a_hi[:40]!r}")

    n = max(agg["n"], 1)
    print("\n" + "#" * 100)
    print(f"AGGREGATE over {agg['n']} probes:")
    print(f"  ceiling pass (hi>=.5):   {agg['hi_pass']/n:.2%}   mean hi_f1={agg['hi']/n:.3f}")
    print(f"  floor   pass (lo< .5):   {agg['lo_fail']/n:.2%}   mean lo_f1={agg['lo']/n:.3f}")
    print(f"  mean tf_f1 (policy):     {agg['tf']/n:.3f}")
    print(f"  mean dynamic range hi-lo:{agg['dyn']/n:.3f}")
    print(f"  fact present in out_hi:  {agg['present']/n:.2%}")
    pres = max(agg["present"], 1)
    print(f"  READER miss on present:  {agg['reader_miss']/pres:.2%} "
          f"({agg['reader_miss']}/{agg['present']}) <- reader failed on a present answer")
    print(f"  reader abstention (hi):  {agg['abstain_hi']/n:.2%}")


if __name__ == "__main__":
    main()
