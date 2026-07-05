"""Round-trip support scan — THE gate before any RL training run (spec Gates-1; the
round-trip descendant of probe_flip_scan.py, mandated by the 2026-07-05 pivot handoff).

From the floor-walk baseline: single-action counterfactuals (each decision span, each legal
alternative action, capped) -> full cached round trips -> per-probe realized-F1 deltas.
PASS = reward responds in BOTH directions with magnitude above the quantization step.
A support desert is a FINDING about the environment — report it, never work around it.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/roundtrip_support_scan.py \
       [--max-swaps 150] [--workers 8] [--probes data/probes_validated.json] [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

OUT = Path("results/roundtrip_support_scan.json")
FLIP = 0.5     # per-probe |delta f1| counting as a flip


def scan_verdict(rows, mean_probes: float) -> dict:
    """PASS iff swaps moved realized recall BOTH directions with a quantization-exceeding
    magnitude in EACH direction: at least one up-move >= step AND one down-move <= -step
    (step = 1/mean probes per doc). A one-sided desert or sub-step wiggle FAILs."""
    step = 1.0 / max(mean_probes, 1.0)
    n_up = sum(1 for r in rows if r["delta"] > 0)
    n_down = sum(1 for r in rows if r["delta"] < 0)
    n_up_sig = sum(1 for r in rows if r["delta"] >= step)
    n_down_sig = sum(1 for r in rows if r["delta"] <= -step)
    max_abs = max((abs(r["delta"]) for r in rows), default=0.0)
    ok = n_up_sig >= 1 and n_down_sig >= 1
    return {"n_swaps": len(rows), "n_up": n_up, "n_down": n_down,
            "n_up_sig": n_up_sig, "n_down_sig": n_down_sig,
            "max_abs_delta": round(max_abs, 4), "quant_step": round(step, 4),
            "verdict": "PASS" if ok else "FAIL"}


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble, derive_spans, floor_walk_choice

    from cloak.corpora import load_task_docs
    from cloak.train.roundtrip import RT_BASE_URL, RT_MODEL, roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-swaps", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probes", default="data/probes_validated.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=16,
                    help="docs loaded per corpus; docs beyond the frozen arms artifact are skipped")
    args = ap.parse_args()

    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    probes_art = json.loads(Path(args.probes).read_text())
    probes_all = probes_art["docs"]
    floors = dict(env["k_floors"])
    rng = random.Random(args.seed)

    # baseline floor-walk rollout per doc (dynamic collision rule: first-come keeps the
    # fill, later colliding spans fall back to placeholder — the trainer's walk-order rule)
    docs, base_jobs = [], []
    for corpus, per_doc in env["corpora"].items():
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, args.n_docs)}
        for doc_id, d in per_doc.items():
            probes = probes_all.get(doc_id, {}).get("train", [])
            if not d.get("trainable") or not d["spans"] or len(probes) < 3:
                continue
            spans, _ = derive_spans(d["spans"], floors, corpus, "cpu")
            choice = floor_walk_choice(spans)
            doc_p, R = assemble(texts[doc_id], art[corpus][doc_id]["tau_walk"][1],
                                d["spans"], choice)
            docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "spans": spans, "raw_spans": d["spans"], "choice": choice,
                         "probes": probes})
            base_jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": probes})
    base = roundtrip_batch(base_jobs, workers=args.workers)
    base_f1s = {d["id"]: b["f1s"] for d, b in zip(docs, base)}
    base_recall = {d["id"]: b["recall"] for d, b in zip(docs, base)}

    # counterfactual swaps: every (span, legal alternative), sampled down to the cap
    swaps = []
    for d in docs:
        for s in d["spans"]:
            cur = d["choice"][s["surface"].lower()]
            for i in s["legal"]:
                a = s["actions"][i]
                if a is cur or (a["mode"] == cur["mode"] and
                                a.get("fill") == cur.get("fill")):
                    continue
                swaps.append((d, s, i))
    rng.shuffle(swaps)
    swaps = swaps[:args.max_swaps]

    jobs = []
    for d, s, i in swaps:
        choice = dict(d["choice"])
        choice[s["surface"].lower()] = s["actions"][i]
        try:
            doc_p, R = assemble(d["text"], d["R_walk"], d["raw_spans"], choice)
        except AssertionError:      # injectivity collision -> unreachable action, skip
            jobs.append(None)
            continue
        jobs.append({"corpus": d["corpus"], "doc_p": doc_p, "R": R, "probes": d["probes"]})
    live = [j for j in jobs if j]
    outs = iter(roundtrip_batch(live, workers=args.workers))

    rows = []
    for (d, s, i), j in zip(swaps, jobs):
        if j is None:
            continue
        o = next(outs)
        deltas = [a - b for a, b in zip(o["f1s"], base_f1s[d["id"]])]
        # collision-RESOLVED baseline action index: the index of the action the floor-walk
        # actually took for this surface (floor_walk_choice downgrades colliders to
        # placeholder), which differs from s["bc_action"] on collision docs.
        cur = d["choice"][s["surface"].lower()]
        from_idx = next(k for k, a in enumerate(s["actions"]) if a is cur)
        rows.append({"doc_id": d["id"], "surface": s["surface"],
                     "from": from_idx, "to": i,
                     "delta": round((o["recall"] or 0) - (base_recall[d["id"]] or 0), 4),
                     "probe_flips_up": sum(x >= FLIP for x in deltas),
                     "probe_flips_down": sum(x <= -FLIP for x in deltas)})

    mean_probes = sum(len(d["probes"]) for d in docs) / max(len(docs), 1)
    v = scan_verdict(rows, mean_probes)
    v.update(mean_probes_per_doc=round(mean_probes, 2),
             meta={"rt_model": RT_MODEL, "rt_base_url": RT_BASE_URL,
                   "probes_path": args.probes, "probes_meta": probes_art.get("meta"),
                   "floors": floors, "n_docs": args.n_docs},
             rows=rows)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(v, indent=1))
    print({k: v[k] for k in ("n_swaps", "n_up", "n_down", "max_abs_delta", "verdict")})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
