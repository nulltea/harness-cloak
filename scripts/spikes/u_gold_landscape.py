"""u_gold landscape sanity check (spec §5.1 "Intended, not proven, monotonicity").

Pre-registered check that BLOCKS training the same as the §6 gate. On the 23 trainable
docs (env + artifact, floor-walk baseline with the trainer's dynamic collision rule):

1. per doc: raw u_gold(floor-walk) vs raw u_gold(all-placeholder) — floor-walk should win.
   (raw = un-normalized _score_facts; normalized u_gold pins all-placeholder to 0 by
   construction, so the endpoint comparison MUST use raw scores. all-placeholder raw = U_lo,
   the anchor u_gold_anchors already computes.)
2. single-swap directionality: from the floor-walk baseline, for each span whose baseline
   action is a level, for each OTHER legal level with strictly LARGER aset (coarser), swap
   that one span, re-assemble, re-score raw. specificity-consistent = coarsened raw strictly
   lower than the floor-walk raw.
3. excluded docs (§5.1 edge rule): empty fact mask, or |U_hi - U_lo| < 0.05 nats.

PASS (spec §5.1): floor-walk >= all-placeholder on a CLEAR majority of docs AND a clear
majority of swaps specificity-consistent. Borderline (~50-60%) prints FAIL.

Baseline/env/artifact loading copies scripts/spikes/reward_landscape_probe.py; the swap and
scoring use the Task-1 reward interfaces (u_gold_anchors / gold_fact_spans / _score_facts),
with template=TASK_TEMPLATE[corpus] passed EXPLICITLY everywhere.

Run (CPU, per-corpus chunk so no command exceeds the 10-min cap):
  HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:scripts \
    .venv/bin/python -u scripts/spikes/u_gold_landscape.py --corpus clinical
  ... --corpus enron ; ... --corpus aeslc
  ... --merge          # combines results/u_gold_landscape_*.json, prints PASS/FAIL
"""
import argparse
import glob
import json
import time
from pathlib import Path

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs, refs_of
from cloak.tasks import TASK_TEMPLATE
from cloak.train.reward import _score_facts, gold_fact_spans, u_gold_anchors

from train_ranker import assemble, derive_spans

CLEAR_MAJORITY = 2 / 3  # the "clear majority" bar; ~50-60% is borderline -> FAIL


def floor_walk_choice(spans):
    """Floor-walk baseline with the trainer's dynamic collision rule (copied from
    reward_landscape_probe.py): per span take the min-aset legal level (bc_action); if its
    fill is already claimed by an earlier span, demote to the terminal placeholder."""
    bc, used = {}, set()
    for s in spans:
        a = s["actions"][s["bc_action"]]
        if a["mode"] == "level" and a["fill"].lower() in used:
            a = next(x for x in s["actions"] if x["mode"] == "placeholder")
        if a["mode"] == "level":
            used.add(a["fill"].lower())
        bc[s["surface"].lower()] = a
    return bc


def process_corpus(corpus, per_doc, art, floors, start=0, count=None):
    texts = {d["id"]: d for d in load_task_docs(corpus, 16)}
    template = TASK_TEMPLATE[corpus]
    swaps = 0
    docs_out, swap_rows, excluded = [], [], []
    collisions = 0
    # slice the ORDERED trainable-with-spans doc ids so a heavy corpus fits the time cap
    ids = [k for k, d in per_doc.items() if d["trainable"] and d["spans"]]
    ids = ids[start:start + count] if count else ids[start:]
    for doc_id in ids:
        d = per_doc[doc_id]
        text = texts[doc_id]["text"]
        gold = refs_of(texts[doc_id])[0]
        tau_walk = art[corpus][doc_id]["tau_walk"]
        R_walk = tau_walk[1]
        spans, _ = derive_spans(d["spans"], floors, corpus, "cpu")

        facts = gold_fact_spans(gold, R_walk)
        if not facts:
            excluded.append({"doc": doc_id, "corpus": corpus, "reason": "empty_facts"})
            continue
        art_entry = {"gold": gold, "spans": spans, "tau_walk": tau_walk}
        anchors = u_gold_anchors(text, facts, art_entry, template=template)
        U_hi, U_lo = anchors["U_hi"], anchors["U_lo"]
        if U_hi is None or U_lo is None or abs(U_hi - U_lo) < 0.05:
            excluded.append({"doc": doc_id, "corpus": corpus, "reason": "anchor_sep<0.05",
                             "U_hi": U_hi, "U_lo": U_lo})
            continue

        bc = floor_walk_choice(spans)
        doc_p_bc, R_bc = assemble(text, R_walk, spans, bc)
        fw_raw, _ = _score_facts(doc_p_bc, R_bc, gold, facts, template)
        win = fw_raw >= U_lo   # all-placeholder raw == U_lo
        docs_out.append({"doc": doc_id, "corpus": corpus, "n_facts": len(facts),
                         "fw_raw": round(fw_raw, 4), "all_ph_raw": round(U_lo, 4),
                         "U_hi": round(U_hi, 4), "win": win})

        # single-swap directionality: coarsen ONE span (strictly larger aset), re-score raw
        for s in spans:
            skey = s["surface"].lower()
            if bc[skey]["mode"] != "level":
                continue
            base_idx = s["bc_action"]
            base_aset = s["actions"][base_idx].get("aset", 0.0)
            for i in s["legal"]:
                a = s["actions"][i]
                if i == base_idx or a["mode"] != "level":
                    continue
                if a.get("aset", 0.0) <= base_aset:  # need STRICTLY coarser
                    continue
                sw = dict(bc)
                sw[skey] = a
                try:
                    doc_p_sw, R_sw = assemble(text, R_walk, spans, sw)
                except AssertionError:  # coarser fill collides with another span (injectivity)
                    collisions += 1
                    continue
                sw_raw, _ = _score_facts(doc_p_sw, R_sw, gold, facts, template)
                swap_rows.append({
                    "doc": doc_id, "corpus": corpus, "surface": s["surface"],
                    "base_fill": s["actions"][base_idx]["fill"], "coarse_fill": a["fill"],
                    "base_aset": base_aset, "swap_aset": a.get("aset", 0.0),
                    "base_raw": round(fw_raw, 4), "swap_raw": round(sw_raw, 4),
                    "consistent": sw_raw < fw_raw})
                swaps += 1
        print(f"  [{corpus}] {doc_id}: facts={len(facts)} fw_raw={fw_raw:.3f} "
              f"U_lo={U_lo:.3f} win={win} swaps_so_far={swaps}", flush=True)
    return {"corpus": corpus, "per_doc": docs_out, "swaps": swap_rows,
            "excluded": excluded, "collisions_skipped": collisions}


def summarize(parts):
    per_doc = [r for p in parts for r in p["per_doc"]]
    swaps = [r for p in parts for r in p["swaps"]]
    excluded = [r for p in parts for r in p["excluded"]]
    collisions = sum(p["collisions_skipped"] for p in parts)
    n_docs = len(per_doc)
    n_swaps = len(swaps)
    wins = sum(r["win"] for r in per_doc)
    consistent = sum(r["consistent"] for r in swaps)
    win_rate = wins / n_docs if n_docs else 0.0
    spec_frac = consistent / n_swaps if n_swaps else 0.0
    verdict = ("PASS" if win_rate > CLEAR_MAJORITY and spec_frac > CLEAR_MAJORITY
               else "FAIL")
    return {
        "scorer_model": __import__("cloak.train.reward", fromlist=["GOLD_SCORER_MODEL"]
                                   ).GOLD_SCORER_MODEL,
        "clear_majority_bar": round(CLEAR_MAJORITY, 4),
        "per_doc": per_doc, "swaps": swaps, "excluded": excluded,
        "collisions_skipped": collisions,
        "summary": {
            "n_docs_scored": n_docs, "docs_floor_walk_ge_all_placeholder": wins,
            "win_rate": round(win_rate, 4),
            "n_swaps": n_swaps, "specificity_consistent": consistent,
            "spec_consistent_frac": round(spec_frac, 4),
            "excluded_count": len(excluded), "excluded": excluded,
            "verdict": verdict}}


def print_verdict(report):
    s = report["summary"]
    print("\n" + "=" * 64)
    print(f"u_gold LANDSCAPE SANITY CHECK  (scorer={report['scorer_model']})")
    print(f"floor-walk >= all-placeholder : {s['docs_floor_walk_ge_all_placeholder']}"
          f"/{s['n_docs_scored']} docs = {s['win_rate']:.3f}")
    print(f"specificity-consistent swaps : {s['specificity_consistent']}"
          f"/{s['n_swaps']} = {s['spec_consistent_frac']:.3f}")
    print(f"excluded docs (edge rule)    : {s['excluded_count']}  {s['excluded']}")
    print(f"collisions skipped (swaps)   : {report['collisions_skipped']}")
    print(f"clear-majority bar           : > {report['clear_majority_bar']:.3f}")
    print(f"VERDICT                      : {s['verdict']}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="process one corpus, write partial JSON")
    ap.add_argument("--start", type=int, default=0, help="doc-slice offset within the corpus")
    ap.add_argument("--count", type=int, default=None, help="doc-slice size (fit the time cap)")
    ap.add_argument("--merge", action="store_true",
                    help="combine results/u_gold_landscape_*.json -> final + verdict")
    args = ap.parse_args()
    t0 = time.time()

    if args.merge:
        parts = [json.loads(Path(f).read_text())
                 for f in sorted(glob.glob("results/u_gold_landscape_*.json"))]
        report = summarize(parts)
        Path("results/u_gold_landscape.json").write_text(json.dumps(report, indent=1))
        print_verdict(report)
        print(f"merged {len(parts)} corpora -> results/u_gold_landscape.json "
              f"({time.time()-t0:.0f}s)")
        return

    env = json.loads(Path("data/ranker_env.json").read_text())
    art = load_artifact()
    floors = dict(env["k_floors"])
    corpora = [args.corpus] if args.corpus else list(env["corpora"].keys())
    parts = [process_corpus(c, env["corpora"][c], art, floors,
                            start=args.start, count=args.count) for c in corpora]

    if args.corpus:
        tag = args.corpus + (f"_{args.start}" if (args.start or args.count) else "")
        out = Path(f"results/u_gold_landscape_{tag}.json")
        out.write_text(json.dumps(parts[0], indent=1))
        print(f"[{args.corpus}] docs={len(parts[0]['per_doc'])} "
              f"swaps={len(parts[0]['swaps'])} excluded={len(parts[0]['excluded'])} "
              f"-> {out} ({time.time()-t0:.0f}s)", flush=True)
    else:
        report = summarize(parts)
        Path("results/u_gold_landscape.json").write_text(json.dumps(report, indent=1))
        print_verdict(report)
        print(f"wall {time.time()-t0:.0f}s -> results/u_gold_landscape.json")


if __name__ == "__main__":
    main()
