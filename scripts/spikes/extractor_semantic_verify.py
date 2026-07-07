"""Verify semantic-window extractor lift against the extractor_miss_audit baseline.

This is a one-off spike: same floor-walk round trips as extractor_miss_audit.py, then:
  - baseline hits = exact + fuzzy90 echo classes from the pre-semantic audit classifier
  - current hits = invert() gen_exact + gen_fuzzy + gen_semantic
  - null control = invert each out_p with the next doc's R; semantic null firings are
    wrong-surface risks, so they must stay at zero or be reported plainly.

Run:
  INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
    .venv/bin/python -u scripts/spikes/extractor_semantic_verify.py \
    --env data/ranker_env.json --arms data/task_arms_tau0.02.json \
    --corpora clinical --n-docs 16 --workers 6
"""
import argparse
import json
import re
from pathlib import Path

from build_arms_artifact import load_artifact
from train_ranker import assemble, derive_spans, floor_walk_choice

from cloak.corpora import load_task_docs
from cloak.extract import (FUZZ_MIN, SEMANTIC_MARGIN, SEMANTIC_MIN, _candidate_windows,
                           _GENERIC_SEMANTIC_FILLS, _semantic_scores, _type_sane, invert)
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/extractor_semantic_verify.json")
BAND_LO = 60.0


def classify_baseline(fill: str, out_p: str) -> dict:
    from rapidfuzz import fuzz
    if re.search(rf"\b{re.escape(fill)}\b", out_p, re.IGNORECASE):
        return {"cls": "exact", "score": 100.0, "snippet": ""}
    al = fuzz.partial_ratio_alignment(fill.lower(), out_p.lower())
    score = al.score if al else 0.0
    snippet = out_p[max(0, al.dest_start - 20):al.dest_end + 20] if al else ""
    if score >= FUZZ_MIN:
        return {"cls": "fuzzy90", "score": round(score, 1), "snippet": snippet}
    if score >= BAND_LO:
        return {"cls": "band60_90", "score": round(score, 1), "snippet": snippet}
    return {"cls": "absent", "score": round(score, 1), "snippet": ""}


def gen_fired(stats: dict) -> int:
    return stats["gen_exact"] + stats["gen_fuzzy"] + stats.get("gen_semantic", 0)


def semantic_decision(entry: dict, out_p: str) -> dict | None:
    fill = entry["replacement"]
    if fill.strip().lower() in _GENERIC_SEMANTIC_FILLS:
        return None
    cands = _candidate_windows(fill, out_p)
    if not cands:
        return None
    scores = _semantic_scores(fill, tuple(c[3] for c in cands))
    ranked = sorted(zip(scores, cands), key=lambda x: (-x[0], -x[1][2]))
    best_cos, (lo, hi, fuzz_score, snippet) = ranked[0]
    runner = ranked[1][0] if len(ranked) > 1 else None
    margin = best_cos - runner if runner is not None else None
    type_sane = _type_sane(entry.get("type", "MISC"), fill, snippet)
    accepted = (best_cos >= SEMANTIC_MIN and type_sane and
                (runner is None or best_cos - runner >= SEMANTIC_MARGIN))
    if not accepted:
        return None
    return {"surface": entry["surface"], "fill": fill, "type": entry.get("type", "MISC"),
            "candidate": snippet, "span": [lo, hi], "fuzzy_score": round(fuzz_score, 1),
            "cos": round(float(best_cos), 4),
            "runner_up_cos": None if runner is None else round(float(runner), 4),
            "margin": None if margin is None else round(float(margin), 4)}


def build_jobs(args) -> tuple[list[dict], list[dict]]:
    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    jobs, metas = [], []
    for corpus in args.corpora.split(","):
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, args.n_docs)}
        for doc_id, d in env["corpora"].get(corpus, {}).items():
            if not d.get("spans") or doc_id not in texts:
                continue
            spans, _ = derive_spans(d["spans"], dict(env["k_floors"]), corpus, "cpu")
            choice = floor_walk_choice(spans)
            doc_p, R = assemble(texts[doc_id], art[corpus][doc_id]["tau_walk"][1],
                                d["spans"], choice)
            jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
            metas.append({"corpus": corpus, "doc_id": doc_id, "R": R})
    return jobs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env.json")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json")
    ap.add_argument("--corpora", default="clinical")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)

    baseline_counts = {"exact": 0, "fuzzy90": 0, "band60_90": 0, "absent": 0}
    current_totals = {"gen_exact": 0, "gen_fuzzy": 0, "gen_semantic": 0, "gen_absent": 0}
    semantic_examples = []
    baseline_miss_fills = 0
    for m, o in zip(metas, outs):
        _, stats = invert(o["out_p"], m["R"])
        for key in current_totals:
            current_totals[key] += stats[key]
        for e in m["R"]:
            if e["action"] != "generalize":
                continue
            cls = classify_baseline(e["replacement"], o["out_p"])
            baseline_counts[cls["cls"]] += 1
            if cls["cls"] in ("band60_90", "absent"):
                baseline_miss_fills += 1
            dec = semantic_decision(e, o["out_p"]) if cls["cls"] in ("band60_90", "absent") else None
            if dec and len(semantic_examples) < 20:
                semantic_examples.append({"doc": m["doc_id"], "baseline_class": cls, **dec})

    null_base, null_current = 0, 0
    null_semantic, null_examples = 0, []
    if len(metas) > 1:
        for i, (m, o) in enumerate(zip(metas, outs)):
            other_R = metas[(i + 1) % len(metas)]["R"]
            for e in other_R:
                if e["action"] != "generalize":
                    continue
                cls = classify_baseline(e["replacement"], o["out_p"])
                null_base += int(cls["cls"] in ("exact", "fuzzy90"))
            _, st = invert(o["out_p"], other_R)
            null_current += gen_fired(st)
            null_semantic += st.get("gen_semantic", 0)
            if st.get("gen_semantic", 0):
                for e in other_R:
                    if e["action"] != "generalize":
                        continue
                    dec = semantic_decision(e, o["out_p"])
                    if dec and len(null_examples) < 10:
                        null_examples.append({"doc": m["doc_id"],
                                              "mismatched_doc": metas[(i + 1) % len(metas)]["doc_id"],
                                              **dec})

    n = sum(baseline_counts.values())
    baseline_hits = baseline_counts["exact"] + baseline_counts["fuzzy90"]
    current_hits = current_totals["gen_exact"] + current_totals["gen_fuzzy"] + current_totals["gen_semantic"]
    report = {
        "settings": vars(args),
        "n_docs": len(metas),
        "n_level_fills": n,
        "baseline_counts": baseline_counts,
        "baseline_hit_rate": round(baseline_hits / max(n, 1), 4),
        "current_totals": current_totals,
        "current_hit_rate": round(current_hits / max(n, 1), 4),
        "absolute_hit_lift": current_hits - baseline_hits,
        "semantic_hits": current_totals["gen_semantic"],
        "semantic_recovery_rate_on_baseline_misses": round(
            current_totals["gen_semantic"] / max(baseline_miss_fills, 1), 4),
        "null_control": {
            "baseline_firings": null_base,
            "current_firings": null_current,
            "semantic_firings": null_semantic,
            "current_minus_baseline": null_current - null_base,
        },
        "semantic_examples": semantic_examples,
        "null_semantic_examples": null_examples,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("semantic_examples", "null_semantic_examples")}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
