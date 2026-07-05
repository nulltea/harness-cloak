"""P2 data engine: doc -> substitute(tau) -> remote LLM -> (doc, task, doc_p, R, out_p, out_ref).

Remote task model = Qwen3.6-35B-A3B via the ts-proxy (same as the InferDPT baseline's gen
model). Substitution is local/GPU (sequential); remote calls are threaded and disk-cached
via $INFERDPT_LLM_CACHE. Tuples -> data/latticecloak_tuples/tau<tau>.jsonl (append-safe by rerun:
existing (author, task_id) pairs are skipped).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/latticecloak_engine.py --limit 8 --tau 0.02
"""
import argparse
import json
import time
from pathlib import Path

from cloak.substitute import Substitutor
from cloak.synthpai import load_docs
from cloak.tasks import doc_tasks, fill, qa_pairs
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--gen-model", default="gemma 4 (E4B)")  # = roundtrip.RT_MODEL pin (2026-07-05)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_path = Path(f"data/latticecloak_tuples/tau{args.tau}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["author"], json.loads(l)["task_id"]) for l in open(out_path)}

    docs = load_docs(args.limit)
    qa = qa_pairs(docs)
    sub = Substitutor(tau=args.tau)
    remote = LLMClient(args.gen_model, temperature=0.0, max_tokens=400,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    t0 = time.time()
    rows = []
    for i, doc in enumerate(docs):  # substitution: local GPU, sequential
        doc_p, R = sub(doc["text"])
        for task in doc_tasks(doc, qa):
            if (doc["author"], task["task_id"]) in done:
                continue
            rows.append({"author": doc["author"], "task_id": task["task_id"],
                         "task": task, "doc": doc["text"], "doc_p": doc_p, "R": R,
                         "gold_attrs": doc["gold"]})
        print(f"[sub {i+1}/{len(docs)}] {time.time()-t0:.0f}s", flush=True)

    print(f"{len(rows)} task-tuples to fill; calling remote {args.gen_model}...", flush=True)
    outs_p = pmap(lambda r: remote.generate(fill(r["task"], r["doc_p"])), rows, workers=args.workers)
    outs_ref = pmap(lambda r: remote.generate(fill(r["task"], r["doc"])), rows, workers=args.workers)
    with open(out_path, "a") as f:
        for r, op, oref in zip(rows, outs_p, outs_ref):
            r["out_p"], r["out_ref"] = op, oref
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} tuples -> {out_path} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
