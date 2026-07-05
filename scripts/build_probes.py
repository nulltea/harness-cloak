"""Validated probe build (spec Phase 0 step 4): teacher questions + anchor validation.

Per doc: candidate probes from the gemma teacher (cloak.train.probes, cached) -> two anchor
round trips through the PINNED reward model (ceiling = doc_orig, floor = all-placeholder,
both full round trips incl. inversion) -> keep iff ceiling f1 >= TH and floor f1 < TH.
The floor check drops probes the all-placeholder baseline already answers (echoed
placeholders invert perfectly — such probes have no dynamic range above the safest action).

Writes data/probes_validated.json + results/probe_health.json. Docs with < 3 surviving
train probes are listed in excluded_docs (spec: excluded from the RL reward, never
silently kept).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/build_probes.py [--corpora clinical,enron,aeslc]
       [--n-docs 16] [--workers 8] [--th 0.5] [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

TH = 0.5
OUT = Path("data/probes_validated.json")
REPORT = Path("results/probe_health.json")


def validate_probes(cands, hi_f1s, lo_f1s, th=TH):
    """Pure keep/drop: probe survives iff answerable at the ceiling anchor AND not already
    answered at the floor anchor. Returns (kept, rejected_ceiling, rejected_floor)."""
    kept, rej_c, rej_f = [], [], []
    for p, hi, lo in zip(cands, hi_f1s, lo_f1s):
        if hi < th:
            rej_c.append(p)
        elif lo >= th:
            rej_f.append(p)
        else:
            kept.append(p)
    return kept, rej_c, rej_f


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble

    from cloak.corpora import load_task_docs, refs_of
    from cloak.train.probes import probes_for_docs
    from cloak.train.reward import fact_f1s
    from cloak.train.roundtrip import roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="clinical,enron,aeslc")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--th", type=float, default=TH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    report = {"th": args.th, "corpora": {}}

    for corpus in args.corpora.split(","):
        docs = load_task_docs(corpus, args.n_docs)
        per_doc = env["corpora"].get(corpus, {})
        rows = [d for d in docs if d["id"] in per_doc and per_doc[d["id"]]["spans"]]
        # 1. candidate probes (teacher, cached; R = artifact tau_walk R)
        R_of = {d["id"]: art[corpus][d["id"]]["tau_walk"][1] for d in rows}
        cands = probes_for_docs(rows, R_of, workers=args.workers)
        # 2. anchor round trips: ceiling (doc_orig, R=[]) + floor (all-placeholder)
        jobs, meta = [], []
        for d in rows:
            spans = per_doc[d["id"]]["spans"]
            ph_choice = {s["surface"].lower(): s["actions"][-1] for s in spans}
            lo_doc, lo_R = assemble(d["text"], art[corpus][d["id"]]["tau_walk"][1],
                                    spans, ph_choice)
            for kind, doc_p, R in (("hi", d["text"], []), ("lo", lo_doc, lo_R)):
                jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=args.workers)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = r["out_final"]
        # 3. validate + split
        stats = {"docs": 0, "kept": [], "rej_c": 0, "rej_f": 0, "cand": 0,
                 "excluded_docs": []}
        for d in rows:
            ps = cands.get(d["id"], [])
            if not ps or d["id"] not in anchor:
                continue
            hi = fact_f1s(anchor[d["id"]]["hi"], ps)
            lo = fact_f1s(anchor[d["id"]]["lo"], ps)
            kept, rc, rf = validate_probes(ps, hi, lo, args.th)
            rng = random.Random(args.seed)
            rng.shuffle(kept)
            n_hold = max(1, len(kept) // 4) if len(kept) >= 2 else 0
            out[d["id"]] = {"train": kept[n_hold:], "heldout": kept[:n_hold],
                            "rejected": {"ceiling": rc, "floor": rf}}
            stats["docs"] += 1
            stats["cand"] += len(ps)
            stats["kept"].append(len(kept))
            stats["rej_c"] += len(rc)
            stats["rej_f"] += len(rf)
            if len(out[d["id"]]["train"]) < 3:
                stats["excluded_docs"].append(d["id"])
        n = max(stats["docs"], 1)
        report["corpora"][corpus] = {
            "docs": stats["docs"],
            "kept_mean": round(sum(stats["kept"]) / n, 2),
            "kept_min": min(stats["kept"], default=0),
            "ceiling_reject_rate": round(stats["rej_c"] / max(stats["cand"], 1), 3),
            "floor_reject_rate": round(stats["rej_f"] / max(stats["cand"], 1), 3),
            "excluded_docs": stats["excluded_docs"]}
        print(f"[{corpus}] {report['corpora'][corpus]}", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT} + {REPORT}")


if __name__ == "__main__":
    main()
