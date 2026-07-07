"""Extractor-miss audit: how much echoed content does the rule extractor leave on the table?

Reward-bias mechanism under audit (third-NULL channel): invert() narrows generalization
fills back only on exact / fuzzy>=90 echo; a level fill the remote model restates with
paraphrase drift is scored ABSENT, so the reward under-credits specific fills relative to
placeholders (whose token echoes invert perfectly) — u_qa's specificity-blindness recreated
inside the realized metric.

Per level fill in floor-walk round trips: classify the echo as
  exact       — invert's word-boundary regex fires
  fuzzy90     — invert's fuzzy alignment fires (>= FUZZ_MIN)
  band60_90   — best alignment in [60, 90): RECOVERABLE MISS (the decision number), with
                cosine(fill, aligned window) when sentence-transformers is available
  absent      — no meaningful alignment: absorbed, no extractor can help
Placeholder entries: echo rate (token present) for the asymmetry comparison.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/extractor_miss_audit.py \
       [--env data/ranker_env.json] [--arms data/task_arms_tau0.02.json] \
       [--corpora clinical] [--n-docs 16] [--workers 6]
"""
import argparse
import json
import re
from pathlib import Path

from build_arms_artifact import load_artifact
from train_ranker import assemble, derive_spans, floor_walk_choice

from cloak.corpora import load_task_docs
from cloak.extract import FUZZ_MIN
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/extractor_miss_audit.json")
BAND_LO = 60.0


def _cos(a: str, b: str):
    try:
        from sentence_transformers import SentenceTransformer, util
        if not hasattr(_cos, "m"):
            _cos.m = SentenceTransformer("all-MiniLM-L6-v2")
        ea, eb = _cos.m.encode([a, b])
        return round(float(util.cos_sim(ea, eb)), 3)
    except Exception:
        return None


def classify(fill: str, out_p: str) -> dict:
    from rapidfuzz import fuzz
    if re.search(rf"\b{re.escape(fill)}\b", out_p, re.IGNORECASE):
        return {"cls": "exact", "score": 100.0, "snippet": ""}
    al = fuzz.partial_ratio_alignment(fill.lower(), out_p.lower())
    score = al.score if al else 0.0
    snippet = out_p[max(0, al.dest_start - 20):al.dest_end + 20] if al else ""
    if score >= FUZZ_MIN:
        return {"cls": "fuzzy90", "score": round(score, 1), "snippet": snippet}
    if score >= BAND_LO:
        return {"cls": "band60_90", "score": round(score, 1), "snippet": snippet,
                "cos": _cos(fill, snippet)}
    return {"cls": "absent", "score": round(score, 1), "snippet": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env.json")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json")
    ap.add_argument("--corpora", default="clinical")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

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
            metas.append({"doc_id": doc_id, "R": R})
    outs = roundtrip_batch(jobs, workers=args.workers)

    rows, counts = [], {"exact": 0, "fuzzy90": 0, "band60_90": 0, "absent": 0}
    ph = {"echoed": 0, "total": 0}
    for m, o in zip(metas, outs):
        for e in m["R"]:
            if e["action"] == "placeholder":
                ph["total"] += 1
                ph["echoed"] += int(e["replacement"] in o["out_p"])
                continue
            c = classify(e["replacement"], o["out_p"])
            counts[c["cls"]] += 1
            if c["cls"] in ("band60_90", "absent"):
                rows.append({"doc": m["doc_id"], "surface": e["surface"],
                             "fill": e["replacement"], **c})
    n = sum(counts.values())
    report = {"n_docs": len(metas), "n_level_fills": n,
              "rates": {k: round(v / max(n, 1), 3) for k, v in counts.items()},
              "counts": counts,
              "placeholder_echo_rate": round(ph["echoed"] / max(ph["total"], 1), 3),
              "placeholder_total": ph["total"],
              "recoverable_miss_examples": rows[:20]}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "recoverable_miss_examples"}, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
