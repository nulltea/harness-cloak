"""Distill the Qwen survival judge into reconstructor training triples.

Per doc: run the rule cascade to get out_final' + residue Q; ask the judge to locate each
residue fill's (possibly reworded) mention in out_p; build the gold target by splicing the
original at each grounded quote. Abstains (D-4/absent) keep the text unchanged, teaching
the model when NOT to edit. Emits data/reconstructor_<corpus>.jsonl.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes \
       .venv/bin/python -u scripts/build_reconstructor_data.py \
       --env data/ranker_env_pilot.json --arms data/task_arms_pilot.json \
       --corpora clinical --n-docs 80 --workers 6
"""
import argparse, json
from pathlib import Path

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, SYSTEM, JUDGE_TMPL)
from cloak.extract import _rule_prepass
from cloak.reconstruct import linearize_restore_map, build_target, restorable, _load_nli
from cloak.train.roundtrip import roundtrip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", required=True); ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge()
    nli = _load_nli()   # cloak.reconstruct helper wrapping cross-encoder/nli-deberta-v3-small
    per_corpus: dict[str, list[dict]] = {}
    degens: dict[str, list[dict]] = {}

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)  # cascade output
        if not residue:
            continue
        items = "\n".join(f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type","MISC")}]'
                          for i, e in enumerate(residue))
        verdicts = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                              system=SYSTEM), len(residue))
        located, degen = [], []
        for e, v in zip(residue, verdicts):
            # ADMISSION GATE: admit as a splice target only if grounded in the text we edit
            # AND type-consistent correspondence holds (rejects D-4 false-correspondence).
            if restorable(e, v, prepass, nli=nli):
                located.append({"surface": e["surface"], "quote": v.get("quote")})
            else:
                located.append({"surface": e["surface"], "quote": None})  # -> no-op
                if v.get("quote") and v.get("label") in ("SURVIVED", "REWORDED"):
                    degen.append({"doc_id": m["doc_id"], "surface": e["surface"],
                                  "fill": e["replacement"], "type": e.get("type", "MISC"),
                                  "quote": v["quote"], "label": v["label"],
                                  "reason": "grounded but failed type-consistency (D-4-like)"})
        inp = f"{prepass}\n\n[RESTORE]\n{linearize_restore_map(residue)}"
        target, n_edits = build_target(prepass, located)
        per_corpus.setdefault(m["corpus"], []).append(
            {"input": inp, "target": target, "corpus": m["corpus"], "doc_id": m["doc_id"],
             "n_residue": len(residue), "n_edits": n_edits, "is_noop": n_edits == 0,
             "high_risk_noop": n_edits == 0 and len(degen) > 0})  # doc had a D-4-like reject
        degens.setdefault(m["corpus"], []).extend(degen)

    for corpus, rows in per_corpus.items():
        Path(f"data/reconstructor_{corpus}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        Path(f"data/reconstructor_{corpus}_degeneracies.jsonl").write_text(
            "\n".join(json.dumps(r) for r in degens.get(corpus, [])))
        edits = sum(r["n_edits"] for r in rows)
        noops = sum(r["is_noop"] for r in rows)
        print(f"{corpus}: {len(rows)} docs w/ residue | {edits} admitted edits | "
              f"{noops} no-op targets | {len(degens.get(corpus, []))} logged degeneracies")


if __name__ == "__main__":
    main()
