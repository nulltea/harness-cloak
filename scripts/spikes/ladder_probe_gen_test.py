"""Qualitative test of ladder/decision probe generation on clinical samples, across teachers.

For a few ACI docs: build out_hi (ceiling, cached round trip), then for each candidate teacher
model generate ladder probes (per lattice-bearing span) and decision probes (per doc), report
parse/lint survival stats, and dump everything to results/ladder_probe_gen_test.json for
eyeballing. Spike-only caches (results/spike_ladder_cache.*) — never the real probe caches.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/ladder_probe_gen_test.py \
       --models "Qwen3.6-35B-A3B,LFM2.5-8B-A1B" --n-docs 3 --max-spans 5
"""
import argparse
import json
import re
from pathlib import Path

OUT = Path("results/ladder_probe_gen_test.json")


def main():
    from cloak.corpora import load_task_docs
    from cloak.train import ladder_probes as lp
    from cloak.train.roundtrip import roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen3.6-35B-A3B")
    ap.add_argument("--corpus", default="clinical")
    ap.add_argument("--id-prefix", default="", help="keep only doc ids with this prefix (aci/)")
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--max-spans", type=int, default=5, help="lattice-bearing spans per doc")
    ap.add_argument("--env", default="data/ranker_env_full.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    env = json.loads(Path(args.env).read_text())
    per_doc = env["corpora"][args.corpus]
    docs = [d for d in load_task_docs(args.corpus)
            if d["id"] in per_doc and per_doc[d["id"]]["spans"]
            and d["id"].startswith(args.id_prefix)][:args.n_docs]
    spans_of = {}
    for d in docs:
        latticed = [s for s in per_doc[d["id"]]["spans"]
                    if any(a.get("mode") == "level" for a in s.get("actions", []))]
        spans_of[d["id"]] = latticed[:args.max_spans]
    print(f"{args.corpus}: {len(docs)} docs, "
          f"{sum(map(len, spans_of.values()))} lattice-bearing spans sampled", flush=True)

    # ceiling outputs first (gemma; one llama-swap residency, cached across spike re-runs)
    jobs = [{"corpus": args.corpus, "doc_p": d["text"], "R": [], "probes": []} for d in docs]
    out_hi_of = {d["id"]: r["out_final"]
                 for d, r in zip(docs, roundtrip_batch(jobs, workers=args.workers))}

    report = {}
    for spec in [m.strip() for m in args.models.split(",")]:
        model, _, url = spec.partition("@")          # "model@https://..." overrides base_url
        base_url = url or lp.LOCAL_BASE_URL
        tag = re.sub(r"\W+", "_", model)
        print(f"\n=== teacher: {model} ({base_url}) ===", flush=True)
        ladders = lp.ladder_probes_for_docs(
            docs, spans_of, args.corpus, workers=args.workers, model=model,
            base_url=base_url, cache_path=Path(f"results/spike_ladder_cache.{tag}.json"))
        decisions = lp.decision_probes_for_docs(
            docs, out_hi_of, args.corpus, workers=args.workers, model=model,
            base_url=base_url, cache_path=Path(f"results/spike_decision_cache.{tag}.json"))
        n_spans = sum(map(len, spans_of.values()))
        n_rungs_max = sum(len([a for a in s["actions"] if a.get("mode") == "level"]) + 1
                          for ss in spans_of.values() for s in ss)
        n_kept = sum(map(len, ladders.values()))
        spans_covered = {(did, e["surface"]) for did, es in ladders.items() for e in es}
        n_dec = sum(map(len, decisions.values()))
        stats = {"spans": n_spans, "rungs_possible": n_rungs_max, "rungs_kept": n_kept,
                 "rung_survival": round(n_kept / max(n_rungs_max, 1), 2),
                 "spans_with_any_rung": len(spans_covered),
                 "decisions_kept": n_dec,
                 "decisions_per_doc": round(n_dec / max(len(docs), 1), 2)}
        report[spec] = {"stats": stats, "ladders": ladders, "decisions": decisions}
        print(json.dumps(stats, indent=1), flush=True)
        d0 = docs[0]["id"]
        for e in ladders.get(d0, [])[:6]:
            print(f"  [{e['surface']} r{e['rung']}] {e['q']}  ->  {e['a']}", flush=True)
        for e in decisions.get(d0, [])[:2]:
            print(f"  [decision] {e['q']} opts={e['options']} gold={e['gold']} "
                  f"depends_on={e['depends_on']}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
