"""Surrogate-reward validation gate: does U_surr rank candidate quality like realized utility?

The tau axis carries no utility spread on the task corpora (measured 2026-07-02: realized
ROUGE-L flat within ~0.003 across tau on clinical AND email), so the discriminative test
uses CONSTRUCTED arms per doc with guaranteed quality spread:

  no_privacy (doc_orig, empty R)  >=  tau_walk  >=  all_floor (maximal coarsening)
  >=  suppression ([REDACTED], no anchor - the NaPaRe anti-extractor case)

Per (doc, arm): realized utility = real round trip (cached) -> rule-extractor invert -> ROUGE-L vs
gold; surrogate = U_surr(doc_p, R, gold, probes). Go/no-go: per-doc Spearman between the
two orderings of the 4 arms, averaged over docs — clearly positive = the surrogate ranks
the kind of variation RL will explore. Plan: docs/plans/2026-07-02-surrogate-grpo-training.md.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src \
       .venv/bin/python -u scripts/surrogate_validation.py --corpora clinical --limit 8
Needs llama-swap up (probe questions, cached) + the ts-proxy (round trip, cached).
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from cloak.corpora import load_task_docs, refs_of
from cloak.detect import Detector
from cloak.extract import invert
from cloak.score import score_batch
from cloak.substitute import substitute
from cloak.tasks import TASK_TEMPLATE
from cloak.train.probes import probes_for_docs
from cloak.train.reward import fact_recall, u_surr

ARMS = ["no_privacy", "tau_walk", "all_floor", "suppression"]


def build_arms(text: str, spans: list, tau: float) -> dict[str, tuple[str, list[dict]]]:
    arms = {"no_privacy": (text, []),
            "tau_walk": substitute(text, spans, tau=tau),
            "all_floor": substitute(text, spans, tau=-1.0)}  # risk never < -1 -> coarsest level
    out, R = text, []
    for s in sorted(spans, key=lambda s: -s.start):
        R.append({"surface": s.text, "type": s.type, "action": "generalize",
                  "replacement": "[REDACTED]"})
        out = out[:s.start] + "[REDACTED]" + out[s.end:]
    arms["suppression"] = (out, R[::-1])
    return arms


def _rank(v):
    order = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    for i, j in enumerate(order):
        r[j] = i
    return r


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 1e-9 and vy > 1e-9 else None  # None = degenerate (flat)


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["clinical", "aeslc", "enron"])
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--gen-model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--bertscore", action="store_true",
                    help="second realized ground truth (ROUGE-L is boilerplate-dominated on clinical)")
    args = ap.parse_args()

    from inferdpt.llm import LLMClient
    from inferdpt.pipeline import pmap
    remote = LLMClient(args.gen_model, temperature=0.0, max_tokens=args.max_tokens,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    det = Detector()
    t0 = time.time()
    report = {"limit": args.limit, "tau": args.tau, "arms": ARMS,
              "gen_model": args.gen_model, "corpora": {}}

    for corpus in args.corpora:
        template = TASK_TEMPLATE[corpus]
        docs = load_task_docs(corpus, args.limit)
        arms_of = {d["id"]: build_arms(d["text"], det.detect(d["text"]), args.tau)
                   for d in docs}
        probes = probes_for_docs(docs, {i: a["tau_walk"][1] for i, a in arms_of.items()},
                                 workers=6)
        print(f"[{corpus}] {len(docs)} docs, probes/doc="
              f"{mean(len(probes[d['id']]) for d in docs):.1f} {time.time()-t0:.0f}s", flush=True)

        jobs = [(d, arm) for d in docs for arm in ARMS]
        outs = pmap(lambda j: remote.generate(
            template.format(doc=arms_of[j[0]["id"]][j[1]][0])), jobs, workers=args.workers)

        finals = []
        for (d, arm), op in zip(jobs, outs):
            finals.append(invert(op, arms_of[d["id"]][arm][1])[0])
        sc = score_batch(finals, [refs_of(d) for d, _ in jobs], use_bertscore=args.bertscore)
        rows = []
        for i, ((d, arm), op) in enumerate(zip(jobs, outs)):
            doc_p, R = arms_of[d["id"]][arm]
            s = u_surr(doc_p, R, refs_of(d)[0], probes[d["id"]])
            fr = fact_recall(finals[i], probes[d["id"]])
            rows.append({"id": d["id"], "arm": arm,
                         "realized_factrecall": round(fr, 4) if fr is not None else None,
                         "realized_rougeL": round(sc["rougeL"][i], 4),
                         "realized_bert": round(sc["bertscore_f1"][i], 4)
                         if args.bertscore else None,
                         "u_surr": s["u_surr"], "u_qa": s["u_qa"], "u_nli": s["u_nli"]})
        print(f"[{corpus}] round trips + surrogate done {time.time()-t0:.0f}s", flush=True)

        gts = ["realized_factrecall", "realized_rougeL"] + \
              (["realized_bert"] if args.bertscore else [])
        rhos = {(g, s): [] for g in gts for s in ("u_surr", "u_qa")}
        arm_means = {}
        for d in docs:
            dr = {r["arm"]: r for r in rows if r["id"] == d["id"]}
            for g in gts:
                xs = [dr[a][g] for a in ARMS]
                if any(x is None for x in xs):  # no probes for this doc
                    continue
                for skey in ("u_surr", "u_qa"):
                    ys = [dr[a][skey] if dr[a][skey] is not None else 0.0 for a in ARMS]
                    rho = _spearman(xs, ys)
                    if rho is not None:
                        rhos[(g, skey)].append(rho)
        per_doc = {f"{g}~{s}": {"mean": round(mean(v), 3) if v else None, "n": len(v)}
                   for (g, s), v in rhos.items()}
        for a in ARMS:
            ar = [r for r in rows if r["arm"] == a]
            frs = [r["realized_factrecall"] for r in ar if r["realized_factrecall"] is not None]
            arm_means[a] = {
                "realized_factrecall": round(mean(frs), 4) if frs else None,
                "realized": round(mean(r["realized_rougeL"] for r in ar), 4),
                "u_surr": round(mean(r["u_surr"] for r in ar if r["u_surr"] is not None), 4),
                "u_qa": round(mean(r["u_qa"] for r in ar if r["u_qa"] is not None), 4)
                if any(r["u_qa"] is not None for r in ar) else None,
                "u_nli": round(mean(r["u_nli"] for r in ar if r["u_nli"] is not None), 4),
            }
        report["corpora"][corpus] = {
            "arm_means": arm_means,
            "per_doc_arm_spearman": per_doc,
            "mean_probes_per_doc": round(mean(len(probes[d["id"]]) for d in docs), 2),
            "rows": rows,
        }
        print(f"[{corpus}] arm means (factrecall|rougeL|u_surr): " +
              " ".join(f"{a}={arm_means[a]['realized_factrecall']}|{arm_means[a]['realized']}"
                       f"|{arm_means[a]['u_surr']}" for a in ARMS) +
              f"  per-doc arm-spearman={per_doc}", flush=True)

    out = Path("results/surrogate_validation.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
