"""tau sweep on a task corpus with reference-scored utility (docs/specs/benchmarks.md).

Unlike the SynthPAI probe sweep (latticecloak_tau_sweep.py), utility here is scored against the
corpus gold output (the note / reply / subject), so tau's effect shows up in a real task metric
and rung-A inversion is exercised. Efficient: detect once per doc (tau-independent), out_ctrl once
per doc; substitute + generate + score per tau. Records lattice mechanics + inversion totals.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/latticecloak_task_tau_sweep.py --corpus clinical --limit 30 --bertscore
"""
import argparse
import copy
import json
import math
import time
from statistics import mean

from cloak.corpora import load_task_docs, refs_of
from cloak.detect import Detector
from cloak.extract import invert
from cloak.score import score_batch
from cloak.substitute import substitute
from cloak.tasks import TASK_TEMPLATE
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap
from inferdpt.probes.leakage import overlap, pii_leakage
from latticecloak_tau_sweep import mlm_guess_back  # span-inversion attacker probe


def _agg(sc: dict) -> dict:
    return {k: round(mean(v), 4) for k, v in sc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(TASK_TEMPLATE))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--taus", default="0.005,0.02,0.1,0.5")
    ap.add_argument("--gen-model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]

    template = TASK_TEMPLATE[args.corpus]
    docs = load_task_docs(args.corpus, args.limit)
    refs_list = [refs_of(d) for d in docs]
    det = Detector()
    t0 = time.time()
    spans_per_doc = {d["id"]: det.detect(d["text"]) for d in docs}  # tau-independent
    print(f"detected {len(docs)} docs in {time.time()-t0:.0f}s", flush=True)

    remote = LLMClient(args.gen_model, temperature=0.0, max_tokens=args.max_tokens,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    outs_ctrl = pmap(lambda d: remote.generate(template.format(doc=d["text"])), docs, workers=args.workers)
    sc_ctrl = _agg(score_batch(outs_ctrl, refs_list, args.bertscore))  # no-privacy ceiling, tau-independent
    print(f"utility_ctrl={sc_ctrl}", flush=True)

    sweep = []
    for tau in taus:
        ts = time.time()
        subbed, n_gen, lvl_specific, lvl_floor = {}, 0, 0, 0
        for d in docs:
            doc_p, R = substitute(d["text"], copy.deepcopy(spans_per_doc[d["id"]]), tau=tau)
            subbed[d["id"]] = (doc_p, R)
            for e in R:
                if e["action"] == "generalize" and e.get("lattice"):
                    n_gen += 1
                    lvl_specific += e["replacement"].lower() == e["lattice"][0].lower()
                    lvl_floor += e["replacement"].lower() == e["lattice"][-1].lower()
        outs_p = pmap(lambda d: remote.generate(template.format(doc=subbed[d["id"]][0])),
                      docs, workers=args.workers)
        finals, inv = [], []
        for d, op in zip(docs, outs_p):
            of, st = invert(op, subbed[d["id"]][1])
            finals.append(of)
            inv.append(st)
        # leakage probes (same instruments as the extend-prefix/InferDPT sweep): overlap,
        # PII survival, and the candidate-sensitive MLM guess-back attacker on doc_p
        pii = [pii_leakage(d["text"], subbed[d["id"]][0])["recall"] for d in docs]
        leak = {"overlap": round(mean(overlap(d["text"], subbed[d["id"]][0]) for d in docs), 4),
                "PII leak": round(mean(v for v in pii if not math.isnan(v)), 4),
                "MLM guess-back": round(mean(mlm_guess_back(*subbed[d["id"]]) for d in docs), 4)}
        rec = {"tau": tau, "n": len(docs),
               "leakage": leak,
               "utility_final": _agg(score_batch(finals, refs_list, args.bertscore)),
               "utility_p": _agg(score_batch(outs_p, refs_list, args.bertscore)),
               "gen_spans": n_gen, "at_most_specific": lvl_specific, "at_floor": lvl_floor,
               "inversion_totals": {k: sum(s[k] for s in inv) for k in inv[0]},
               "wall_s": round(time.time() - ts, 1)}
        sweep.append(rec)
        print(json.dumps(rec), flush=True)

    out = args.out or f"results/latticecloak_task_tau_sweep_{args.corpus}.json"
    json.dump({"corpus": args.corpus, "docs": len(docs), "gen_model": args.gen_model,
               "taus": taus, "utility_ctrl": sc_ctrl, "sweep": sweep},
              open(out, "w"), indent=2)
    print(f"total {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
