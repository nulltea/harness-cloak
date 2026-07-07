"""Diagnose the never-recovered probe facts from the context-ablation labels.

For each span whose own-recall is 0.0 under EVERY action (flat-at-zero), reconstruct the
floor-walk out_final and split the cause:
  - fact_absent_from_output    : the fact's surface is NOT in out_final -> the generator never
                                 restated it -> probe/task-quality (the fact isn't in the output
                                 the reward reads, so no fill choice can ever recover it).
  - present_reader_NONE        : surface IS in out_final but the reader abstained -> reader-miss.
  - present_reader_wrong       : surface present, reader answered, fact_score still 0 -> reader /
                                 scorer miss.
Reuses the floor-walk assembly + served round trip (deterministic, cached).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/never_recovered_audit.py
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    from ablate_context_producer import resolve_choice
    from train_ranker import assemble

    from cloak.train.reward import _qa_answer, canon, fact_score
    from cloak.train.roundtrip import roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/context_ablation_labels.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    d = json.loads(Path(args.labels).read_text())
    bundles = d["assembly_bundles"]
    never = [s for s in d["spans"] if s["flat"] and s["actions"][0]["own_recall"] <= 0.001]
    docs = sorted({s["doc_id"] for s in never if s["doc_id"] in bundles})

    # floor-walk out_final per doc (all spans at realized floor-walk, collision-resolved)
    jobs = []
    for dd in docs:
        b = bundles[dd]
        doc_p, R = assemble(b["text"], b["R_walk"], b["raw_spans"], resolve_choice(b["spans"], {}))
        jobs.append({"corpus": b["corpus"], "doc_p": doc_p, "R": R, "probes": b["probes"]})
    outs = roundtrip_batch(jobs, workers=args.workers)
    out_final = {dd: o["out_final"] for dd, o in zip(docs, outs)}

    # CEILING control: Remote(task(doc_orig)), R=[] — the validation ceiling. If the fact is
    # recoverable HERE but absent at floor-walk, the loss is anonymization/marginal-baseline,
    # NOT a task defect (validated probes require ceiling f1 >= 0.5 by construction).
    from cloak.train.reward import fact_f1s
    cjobs = [{"corpus": bundles[dd]["corpus"], "doc_p": bundles[dd]["text"], "R": [],
              "probes": bundles[dd]["probes"]} for dd in docs]
    cout = {dd: o["out_final"] for dd, o in zip(docs, roundtrip_batch(cjobs, workers=args.workers))}

    cats = collections.Counter()
    by_corpus = collections.defaultdict(collections.Counter)
    examples = []
    for s in never:
        dd = s["doc_id"]
        if dd not in out_final:
            continue
        of = out_final[dd]
        surf = s["surface"]
        probes = [p for p in bundles[dd]["probes"] if canon(p["surface"]) == canon(surf)]
        present = surf.lower() in of.lower() or canon(surf) in canon(of)
        ans = [_qa_answer(p["question"], of) for p in probes]
        best = max((fact_score(a, surf) for a in ans), default=0.0)
        if not present:
            cat = "fact_absent_from_output"
        elif all(not a for a in ans):
            cat = "present_reader_NONE"
        elif best <= 0.001:
            cat = "present_reader_wrong"
        else:
            cat = "recovered_unexpected"          # would contradict own_recall=0 -> investigate
        # does the CEILING (doc_orig control) recover this fact? (validation guarantees >=0.5)
        cprobes = [p for p in bundles[dd]["probes"] if canon(p["surface"]) == canon(surf)]
        ceil_recall = max(fact_f1s(cout[dd], cprobes), default=0.0) if dd in cout else 0.0
        cats["_ceiling_recovers" if ceil_recall >= 0.5 else "_ceiling_also_lost"] += 1
        cats[cat] += 1
        by_corpus[s["corpus"]][cat] += 1
        if len(examples) < args.show:
            examples.append((cat, s["corpus"], surf, probes[0]["question"] if probes else "",
                             ans[0] if ans else "", present, of[:160].replace("\n", " ")))

    print(f"=== never-recovered diagnosis: {len(never)} spans, {len(docs)} docs ===")
    for c, n in cats.most_common():
        print(f"  {n:3d}  {c}")
    print("\n=== by corpus ===")
    for corp, cc in by_corpus.items():
        print(f"  {corp}: {dict(cc)}")
    print("\n=== examples ===")
    for cat, corp, surf, q, a, pres, ofx in examples:
        print(f"[{cat}] ({corp}) surface={surf!r} present_in_out={pres}")
        print(f"   Q: {q}")
        print(f"   reader: {a!r}")
        print(f"   out_final[:160]: {ofx}")


if __name__ == "__main__":
    main()
