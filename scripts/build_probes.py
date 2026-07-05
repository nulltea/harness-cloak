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
import datetime
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


def split_by_fact(kept, seed=0):
    """Train/heldout split at FACT granularity. Kept questions are grouped by canon(surface);
    ALL questions of a fact travel together (fact leakage across splits would corrupt the
    heldout read-out). Facts are shuffled (seeded); hold out max(1, n_facts // 4) facts when
    n_facts >= 2. Returns (train_questions, heldout_questions, n_train_facts)."""
    from cloak.train.reward import canon
    facts = {}
    for p in kept:
        facts.setdefault(canon(p["surface"]), []).append(p)
    keys = list(facts)
    random.Random(seed).shuffle(keys)
    n_hold = max(1, len(keys) // 4) if len(keys) >= 2 else 0
    train = [p for k in keys[n_hold:] for p in facts[k]]
    heldout = [p for k in keys[:n_hold] for p in facts[k]]
    return train, heldout, len(keys) - n_hold


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble

    from cloak.corpora import load_task_docs, refs_of
    from cloak.train.probes import PROMPT_VERSION, TEACHER_MODEL, probes_for_docs
    from cloak.train.reward import canon, fact_f1s
    from cloak.train.roundtrip import RT_BASE_URL, RT_MODEL, roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="clinical,enron,aeslc")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--th", type=float, default=TH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    out = prev.get("docs", {}) if isinstance(prev, dict) else {}
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
            ph_choice = {s["surface"].lower():
                         s["actions"][next(i for i, a in enumerate(s["actions"])
                                           if a["mode"] == "placeholder")]
                         for s in spans}
            lo_doc, lo_R = assemble(d["text"], art[corpus][d["id"]]["tau_walk"][1],
                                    spans, ph_choice)
            for kind, doc_p, R in (("hi", d["text"], []), ("lo", lo_doc, lo_R)):
                jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=args.workers)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = r["out_final"]
        # 3. validate (per QUESTION) + split/floor (per FACT)
        stats = {"docs": 0, "kept_facts": [], "kept_questions": [], "rej_c": 0, "rej_f": 0,
                 "cand": 0, "excluded_docs": [], "hi_kept": []}
        for d in rows:
            ps = cands.get(d["id"], [])
            if not ps or d["id"] not in anchor:
                # span-bearing doc with no candidate probes (or no anchor) is excluded, not
                # silently dropped — it contributes no RL reward signal
                stats["excluded_docs"].append(d["id"])
                continue
            hi = fact_f1s(anchor[d["id"]]["hi"], ps)
            lo = fact_f1s(anchor[d["id"]]["lo"], ps)
            kept, rc, rf = validate_probes(ps, hi, lo, args.th)
            hi_kept = [h for _p, h, l in zip(ps, hi, lo) if h >= args.th and l < args.th]
            train_q, heldout_q, n_train_facts = split_by_fact(kept, args.seed)
            out[d["id"]] = {"train": train_q, "heldout": heldout_q,
                            "rejected": {"ceiling": rc, "floor": rf}}
            stats["docs"] += 1
            stats["cand"] += len(ps)
            stats["kept_questions"].append(len(kept))
            stats["kept_facts"].append(len({canon(p["surface"]) for p in kept}))
            stats["hi_kept"].extend(hi_kept)
            stats["rej_c"] += len(rc)
            stats["rej_f"] += len(rf)
            # exclusion floor: < 3 DISTINCT FACTS in the train split (not questions)
            if n_train_facts < 3:
                stats["excluded_docs"].append(d["id"])
        n = max(stats["docs"], 1)
        report["corpora"][corpus] = {
            "docs": stats["docs"],
            "kept_facts_mean": round(sum(stats["kept_facts"]) / n, 2),
            "kept_questions_mean": round(sum(stats["kept_questions"]) / n, 2),
            "kept_min": min(stats["kept_facts"], default=0),
            "ceiling_reject_rate": round(stats["rej_c"] / max(stats["cand"], 1), 3),
            "floor_reject_rate": round(stats["rej_f"] / max(stats["cand"], 1), 3),
            "reader_hi_f1_kept_mean": (round(sum(stats["hi_kept"]) / len(stats["hi_kept"]), 3)
                                       if stats["hi_kept"] else None),
            "excluded_docs": stats["excluded_docs"]}
        print(f"[{corpus}] {report['corpora'][corpus]}", flush=True)

    artifact = {"meta": {"rt_model": RT_MODEL, "rt_base_url": RT_BASE_URL,
                         "teacher": TEACHER_MODEL, "th": args.th, "pv": PROMPT_VERSION,
                         "built_at": datetime.datetime.now().isoformat(timespec="seconds")},
                "docs": out}
    OUT.write_text(json.dumps(artifact, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT} + {REPORT}")


if __name__ == "__main__":
    main()
