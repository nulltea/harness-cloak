"""Reader-parallelism smoke: correctness + wall-time delta for the round-trip reward.

The RL reward loop batches all rollout jobs into one roundtrip_batch(jobs, workers=rt_workers)
call, whose pmap fans jobs across the served slots (reader serial WITHIN a job for prefix-KV
reuse, parallel ACROSS jobs). This smoke checks the two things that claim rests on:

  1. CORRECTNESS — the same cache-cold jobs run at workers=1 and workers=W must return
     identical recall (temp-0 determinism; concurrency must not corrupt results).
  2. PERF — wall-time / throughput delta between serial and parallel on cache-cold jobs.

Each timed run uses a FRESH empty cache dir and resets the module-global clients, so both
runs actually compute (no cross-run cache hits). Small batch, GPU run.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/reader_parallelism_smoke.py \
       --env <clinical env> --arms <clinical arms> --probes data/probes_validated.json \
       --n-docs 3 --g 6 --workers 6
"""
import argparse
import importlib
import json
import os
import tempfile
import time
from pathlib import Path


def _reset_clients():
    """Force fresh LLM clients so a newly-set INFERDPT_LLM_CACHE takes effect."""
    import cloak.train.reward as rw
    import cloak.train.roundtrip as rt
    rt._client = None
    rw._qa = None


def timed_run(jobs, workers, cache_dir):
    os.environ["INFERDPT_LLM_CACHE"] = str(cache_dir)
    _reset_clients()
    from cloak.train.roundtrip import roundtrip_batch
    t0 = time.time()
    res = roundtrip_batch(jobs, workers=workers)
    wall = time.time() - t0
    recalls = [round(r["recall"], 6) if r["recall"] is not None else None for r in res]
    return wall, recalls, res


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble, derive_spans, sample_rollout

    from cloak.corpora import load_task_docs
    from cloak.train.ranker import RankerPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env_full.json")
    ap.add_argument("--arms", default="data/task_arms_full.json")
    ap.add_argument("--probes", default="data/probes_validated.json")
    ap.add_argument("--corpus", default="clinical")
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--g", type=int, default=6, help="rollouts per doc (distinct doc_p)")
    ap.add_argument("--workers", type=int, default=6, help="parallel workers to compare vs 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    probes_all = json.loads(Path(args.probes).read_text())["docs"]
    floors = dict(env["k_floors"])
    per_doc = env["corpora"][args.corpus]

    texts = {d["id"]: d["text"] for d in load_task_docs(args.corpus, 300)}
    docs = []
    for doc_id, d in per_doc.items():
        probes = probes_all.get(doc_id, {}).get("train", [])
        if doc_id not in texts or not d.get("trainable") or not d["spans"] or len(probes) < 3:
            continue
        spans, feats = derive_spans(d["spans"], floors, args.corpus, "cpu")
        docs.append({"id": doc_id, "corpus": args.corpus, "text": texts[doc_id],
                     "R_walk": art[args.corpus][doc_id]["tau_walk"][1],
                     "spans": spans, "raw_spans": d["spans"], "feats": feats,
                     "probes_train": probes})
        if len(docs) >= args.n_docs:
            break

    # build cache-cold jobs: G varied rollouts per doc (random-init policy -> varied doc_p)
    policy = RankerPolicy()
    jobs = []
    for doc in docs:
        for _ in range(args.g):
            choice, _, _, doc_p, R, _ = sample_rollout(doc, doc["spans"], doc["feats"], policy)
            jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                         "probes": doc["probes_train"]})
    n_probes = sum(len(j["probes"]) for j in jobs)
    print(f"{len(docs)} docs, {len(jobs)} jobs, {n_probes} probe-reads "
          f"({n_probes/max(len(jobs),1):.1f}/job)", flush=True)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as dW:
        wall_s, rec_s, res_s = timed_run(jobs, 1, d1)
        print(f"[workers=1]        wall={wall_s:6.1f}s  {len(jobs)/wall_s:.2f} rt/s", flush=True)
        wall_w, rec_w, res_w = timed_run(jobs, args.workers, dW)
        print(f"[workers={args.workers}]        wall={wall_w:6.1f}s  {len(jobs)/wall_w:.2f} rt/s",
              flush=True)

    # correctness: identical recalls (temp-0 determinism; concurrency must not corrupt)
    mism = [(i, a, b) for i, (a, b) in enumerate(zip(rec_s, rec_w)) if a != b]
    print(f"\nCORRECTNESS: {len(jobs) - len(mism)}/{len(jobs)} recalls identical "
          f"serial-vs-parallel", flush=True)
    for i, a, b in mism[:5]:
        print(f"  MISMATCH job {i}: serial={a} parallel={b}", flush=True)
    # stage attribution: does the REMOTE gen (out_p) already differ, or only the reader path?
    n_op = sum(1 for a, b in zip(res_s, res_w) if a["out_p"] != b["out_p"])
    n_of = sum(1 for a, b in zip(res_s, res_w) if a["out_final"] != b["out_final"])
    print(f"STAGE: out_p (gemma gen) differs on {n_op}/{len(jobs)} jobs; "
          f"out_final (after invert) differs on {n_of}/{len(jobs)}", flush=True)
    print(f"PERF: {wall_s/max(wall_w,1e-9):.2f}x speedup at workers={args.workers} "
          f"({wall_s:.1f}s -> {wall_w:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
