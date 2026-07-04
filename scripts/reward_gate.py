"""Constructed-arms validation gate for the CURRENT stage-1 reward + environment.

Spec §6 / §2 Phase 0c (docs/specs/RL/surrogate-ranker-infiller.md): per doc, four artifact arms
(no_privacy / tau_walk / all_floor / suppression) + the all-placeholder diagnostic arm; per
(doc, arm): reward components (A = P6 mean, U = u_qa on TRAIN-split probes, r at --alpha) and
realized fact recall on out_final (same probes). Go: clearly positive per-doc Spearman between
the reward ordering and realized fact recall where the ground truth orders arms sanely. The
all-placeholder arm's (reward vs realized) gap is the standing echo-cost diagnostic, not go/no-go.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/reward_gate.py
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from build_arms_artifact import load_artifact
from train_ranker import assemble

from cloak.corpora import load_task_docs
from cloak.extract import invert
from cloak.probe import reward_privacy
from cloak.tasks import TASK_TEMPLATE
from cloak.train.reward import fact_recall, stage1_reward, u_qa
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap

ARMS = ["no_privacy", "tau_walk", "all_floor", "suppression", "all_placeholder"]
GEN_MODEL = "Qwen3.6-35B-A3B"


def _rank(v):
    order = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    for i, j in enumerate(order):
        r[j] = i
    return r


def _spearman(xs, ys):
    n = len(xs)
    rx, ry = _rank(xs), _rank(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 1e-9 and vy > 1e-9 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    remote = LLMClient(GEN_MODEL, temperature=0.0, max_tokens=args.max_tokens,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    report = {"alpha": args.alpha, "gen_model": GEN_MODEL, "arms": ARMS, "corpora": {}}

    for corpus, per_doc in env["corpora"].items():
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
        jobs = []
        for doc_id, d in per_doc.items():
            probes = d["probes"]["train"]
            if not probes:
                continue  # gate needs a utility signal on both sides
            a = art[corpus][doc_id]
            arms = {"no_privacy": (texts[doc_id], []),
                    "tau_walk": tuple(a["tau_walk"]),
                    "all_floor": tuple(a["all_floor"]),
                    "suppression": tuple(a["suppression"])}
            if d["spans"]:  # diagnostic arm: every decision span -> placeholder
                ph_choice = {s["surface"].lower(): s["actions"][-1] for s in d["spans"]}
                arms["all_placeholder"] = assemble(texts[doc_id], a["tau_walk"][1],
                                                   d["spans"], ph_choice)[:2]
            for arm, (doc_p, R) in arms.items():
                jobs.append({"doc_id": doc_id, "arm": arm, "doc_p": doc_p, "R": R,
                             "probes": probes})
        outs = pmap(lambda j: remote.generate(
            TASK_TEMPLATE[corpus].format(doc=j["doc_p"])), jobs, workers=args.workers)
        rows = []
        for j, op in zip(jobs, outs):
            out_final, _ = invert(op, j["R"])
            A = reward_privacy(j["R"])["mean"]
            U, _ = u_qa(j["doc_p"], j["R"], j["probes"])
            rows.append({"id": j["doc_id"], "arm": j["arm"],
                         "A": round(A, 4), "U": round(U or 0.0, 4),
                         "r": round(stage1_reward(A, U, args.alpha), 4),
                         "realized": round(fact_recall(out_final, j["probes"]) or 0.0, 4)})
        # per-doc Spearman over the four GATE arms (diagnostic arm excluded from rho)
        rhos_r, rhos_u = [], []
        by_doc = {}
        for r in rows:
            by_doc.setdefault(r["id"], {})[r["arm"]] = r
        for doc_id, dr in by_doc.items():
            gate_arms = [a for a in ARMS[:4] if a in dr]
            xs = [dr[a]["realized"] for a in gate_arms]
            rho_r = _spearman(xs, [dr[a]["r"] for a in gate_arms])
            rho_u = _spearman(xs, [dr[a]["U"] for a in gate_arms])
            if rho_r is not None:
                rhos_r.append(rho_r)
            if rho_u is not None:
                rhos_u.append(rho_u)
        arm_means = {a: {k: round(mean(r[k] for r in rows if r["arm"] == a), 4)
                         for k in ("A", "U", "r", "realized")}
                     for a in ARMS if any(r["arm"] == a for r in rows)}
        report["corpora"][corpus] = {
            "per_doc_spearman": {"r~realized": {"mean": round(mean(rhos_r), 3) if rhos_r
                                                else None, "n": len(rhos_r)},
                                 "U~realized": {"mean": round(mean(rhos_u), 3) if rhos_u
                                                else None, "n": len(rhos_u)}},
            "arm_means": arm_means, "rows": rows}
        print(f"[{corpus}] spearman={report['corpora'][corpus]['per_doc_spearman']} "
              f"{time.time()-t0:.0f}s", flush=True)
        for a, m in arm_means.items():
            print(f"    {a:16s} {m}", flush=True)

    out = Path("results/ranker_reward_gate.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
