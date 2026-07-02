"""Task-oriented eval: corpus doc -> substitute(tau) -> remote LLM -> rung A -> ROUGE-L/BERTScore vs gold.

Replaces the prefix/summarization smoke (LatticeCloak report §5.4) with tasks whose gold output
restates the substituted spans, so rung-A inversion fires and utility is sensitive to tau.
Utility is scored on out_final and out_ctrl (no-privacy) against the corpus reference(s).

Remote = Qwen3.6-35B-A3B via ts-proxy (disk-cached via $INFERDPT_LLM_CACHE); substitution local/GPU.
Spec: docs/specs/benchmarks.md.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/latticecloak_task_eval.py --corpus aci --limit 3 --tau 0.02
Wiring check (no remote): ... scripts/latticecloak_task_eval.py --corpus aci --limit 3 --dry-run
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from cloak.corpora import load_task_docs, refs_of
from cloak.extract import invert
from cloak.score import score_batch
from cloak.substitute import Substitutor
from cloak.tasks import TASK_TEMPLATE


def _agg(sc: dict) -> dict:
    return {k: round(mean(v), 4) for k, v in sc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(TASK_TEMPLATE))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--gen-model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no remote: validate load+substitute+score")
    args = ap.parse_args()

    template = TASK_TEMPLATE[args.corpus]
    docs = load_task_docs(args.corpus, args.limit)
    t0 = time.time()

    sub = Substitutor(tau=args.tau)
    rows = []
    for i, d in enumerate(docs):
        doc_p, R = sub(d["text"])
        rows.append({"id": d["id"], "refs": refs_of(d), "doc": d["text"], "doc_p": doc_p, "R": R})
        print(f"[sub {i+1}/{len(docs)}] {d['id']} spans={len(R)} {time.time()-t0:.0f}s", flush=True)
    refs_list = [r["refs"] for r in rows]

    if args.dry_run:  # wiring check: gold echoed back must score ~1.0, invert must run
        preds = [r["refs"][0] for r in rows]
        sc = score_batch(preds, refs_list)
        assert all(x > 0.99 for x in sc["rougeL"]), sc
        for r in rows:
            _, st = invert(template.format(doc=r["doc_p"]), r["R"])
            assert set(st) >= {"ph_swapped", "gen_exact", "gen_fuzzy", "gen_absent"}, st
        print(f"dry-run OK: {len(rows)} docs, rougeL(gold|gold)={_agg(sc)}")
        return

    from inferdpt.llm import LLMClient
    from inferdpt.pipeline import pmap
    remote = LLMClient(args.gen_model, temperature=0.0, max_tokens=args.max_tokens,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    print(f"generating {len(rows)}x2 with {args.gen_model}...", flush=True)
    outs_p = pmap(lambda r: remote.generate(template.format(doc=r["doc_p"])), rows, workers=args.workers)
    outs_ctrl = pmap(lambda r: remote.generate(template.format(doc=r["doc"])), rows, workers=args.workers)

    finals, inv = [], []
    for r, op in zip(rows, outs_p):
        of, st = invert(op, r["R"])
        finals.append(of)
        inv.append(st)

    sc_final = score_batch(finals, refs_list, args.bertscore)
    sc_ctrl = score_batch(outs_ctrl, refs_list, args.bertscore)
    sc_p = score_batch(outs_p, refs_list, args.bertscore)
    inv_sum = {k: sum(s[k] for s in inv) for k in inv[0]}

    tuples_path = Path(f"data/latticecloak_task_tuples/{args.corpus}_tau{args.tau}.jsonl")
    tuples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tuples_path, "w") as f:
        for r, op, oc, of, st in zip(rows, outs_p, outs_ctrl, finals, inv):
            f.write(json.dumps({"id": r["id"], "refs": r["refs"], "doc_p": r["doc_p"],
                                "out_p": op, "out_ctrl": oc, "out_final": of, "inv": st}) + "\n")

    summary = {"corpus": args.corpus, "tau": args.tau, "n": len(rows), "gen_model": args.gen_model,
               "utility_final": _agg(sc_final), "utility_ctrl": _agg(sc_ctrl), "utility_p": _agg(sc_p),
               "inversion_totals": inv_sum, "wall_s": round(time.time() - t0, 1)}
    res_path = Path(f"results/latticecloak_task_eval_{args.corpus}_tau{args.tau}.json")
    res_path.parent.mkdir(exist_ok=True)
    res_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"tuples -> {tuples_path}\nsummary -> {res_path}")


if __name__ == "__main__":
    main()
