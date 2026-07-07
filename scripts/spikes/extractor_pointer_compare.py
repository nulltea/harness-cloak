"""Compare extractor variants on the extractor_miss_audit floor-walk jobs.

Same jobs / out_p as scripts/spikes/extractor_miss_audit.py, then compare:
  - rule baseline: exact + fuzzy90 audit classes
  - Design 1: invert() exact + fuzzy + semantic
  - detector-pointer: invert_detector_pointer() exact + fuzzy + pointer

Run:
  INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
    .venv/bin/python -u scripts/spikes/extractor_pointer_compare.py \
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
from cloak.detect import Detector
from cloak.extract import (DETECTOR_POINTER_GLINER_MODEL, FUZZ_MIN, invert,
                           invert_detector_pointer)
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/extractor_pointer_compare.json")
BAND_LO = 60.0


def classify_baseline(fill: str, out_p: str) -> str:
    from rapidfuzz import fuzz
    if re.search(rf"\b{re.escape(fill)}\b", out_p, re.IGNORECASE):
        return "exact"
    al = fuzz.partial_ratio_alignment(fill.lower(), out_p.lower())
    score = al.score if al else 0.0
    if score >= FUZZ_MIN:
        return "fuzzy90"
    if score >= BAND_LO:
        return "band60_90"
    return "absent"


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


def rate(hit_count: int, n: int) -> float:
    return round(hit_count / max(n, 1), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env.json")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json")
    ap.add_argument("--corpora", default="clinical")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tau-det", type=float, default=0.3)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    detector = Detector(gliner_model=DETECTOR_POINTER_GLINER_MODEL)

    baseline = {"exact": 0, "fuzzy90": 0, "band60_90": 0, "absent": 0}
    design1 = {"gen_exact": 0, "gen_fuzzy": 0, "gen_semantic": 0, "gen_absent": 0}
    pointer = {"gen_exact": 0, "gen_fuzzy": 0, "gen_pointer": 0, "gen_abstain": 0,
               "gen_absent": 0}
    pointer_examples = []

    for m, o in zip(metas, outs):
        for e in m["R"]:
            if e["action"] == "generalize":
                baseline[classify_baseline(e["replacement"], o["out_p"])] += 1

        _, st_d1 = invert(o["out_p"], m["R"])
        for key in design1:
            design1[key] += st_d1.get(key, 0)

        out_ptr, st_ptr = invert_detector_pointer(o["out_p"], m["R"], detector=detector,
                                                  tau_det=args.tau_det)
        for key in pointer:
            pointer[key] += st_ptr.get(key, 0)
        if st_ptr.get("gen_pointer", 0) and len(pointer_examples) < 20:
            pointer_examples.append({"doc": m["doc_id"], "stats": st_ptr,
                                     "out_p": o["out_p"][:600],
                                     "out_final": out_ptr[:600]})

    n = sum(baseline.values())
    baseline_hits = baseline["exact"] + baseline["fuzzy90"]
    design1_hits = design1["gen_exact"] + design1["gen_fuzzy"] + design1["gen_semantic"]
    pointer_hits = pointer["gen_exact"] + pointer["gen_fuzzy"] + pointer["gen_pointer"]
    report = {
        "settings": {**vars(args), "detector_gliner_model": DETECTOR_POINTER_GLINER_MODEL},
        "n_docs": len(metas),
        "n_level_fills": n,
        "rows": [
            {"extractor": "rule_baseline", "hits": baseline_hits,
             "misses": n - baseline_hits, "rate": rate(baseline_hits, n),
             "components": baseline},
            {"extractor": "design1_semantic", "hits": design1_hits,
             "misses": n - design1_hits, "rate": rate(design1_hits, n),
             "components": design1},
            {"extractor": "detector_pointer", "hits": pointer_hits,
             "misses": n - pointer_hits, "rate": rate(pointer_hits, n),
             "components": pointer},
        ],
        "pointer_examples": pointer_examples,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pointer_examples"}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
