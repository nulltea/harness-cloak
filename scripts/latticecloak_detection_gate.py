"""P0.3 detection gate: Presidio∪GLiNER recall/precision on TAB gold spans.

Gold = union of all annotators' DIRECT/QUASI mentions (strictest recall: the privacy
ceiling must cover everything any annotator called an identifier). A gold mention counts
as detected on any character overlap ("any"); "typed" additionally requires the TAB type
to match. Gate rule (plan P0.3): DIRECT any-recall < 0.95 -> controlled condition runs
on gold spans and the detector gap is a finding.

Run: PYTHONPATH=src .venv/bin/python -u scripts/latticecloak_detection_gate.py
"""
import argparse
import json
import time
from collections import defaultdict

from cloak.detect import Detector


def gold_mentions(doc):
    """Union over annotators, deduped by (start, end); DIRECT wins type ties."""
    best = {}
    for ann in doc["annotations"].values():
        for m in ann["entity_mentions"]:
            if m["identifier_type"] not in ("DIRECT", "QUASI"):
                continue
            key = (m["start_offset"], m["end_offset"])
            if key not in best or m["identifier_type"] == "DIRECT":
                best[key] = m
    return list(best.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpora/tab/echr_test.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", default="results/latticecloak_detection_gate.json")
    args = ap.parse_args()

    docs = json.load(open(args.corpus))
    if args.limit:
        docs = docs[: args.limit]
    det = Detector(threshold=args.threshold)

    hit = defaultdict(int)      # (id_class, "any"|"typed") -> hits
    tot = defaultdict(int)      # (id_class,) and (id_class, entity_type) -> golds
    ent_all, ent_hit = defaultdict(set), defaultdict(set)  # entity-level DIRECT
    n_pred = n_pred_gold = 0
    t0 = time.time()

    for i, doc in enumerate(docs):
        text = doc["text"]
        golds = gold_mentions(doc)
        preds = det.detect(text)
        n_pred += len(preds)
        # mention-level recall
        for g in golds:
            gs, ge, idc, et = g["start_offset"], g["end_offset"], g["identifier_type"], g["entity_type"]
            tot[idc] += 1
            tot[idc, et] += 1
            over = [p for p in preds if p.start < ge and gs < p.end]
            if over:
                hit[idc, "any"] += 1
                hit[(idc, et), "any"] += 1
            if any(p.type == et for p in over):
                hit[idc, "typed"] += 1
                hit[(idc, et), "typed"] += 1
            if idc == "DIRECT":
                ent_all[g["entity_id"]].add((gs, ge))
                if over:
                    ent_hit[g["entity_id"]].add((gs, ge))
        # precision proxy: predicted spans overlapping ANY annotated mention (incl. NO_MASK)
        allg = [(m["start_offset"], m["end_offset"])
                for a in doc["annotations"].values() for m in a["entity_mentions"]]
        n_pred_gold += sum(any(p.start < ge and gs < p.end for gs, ge in allg) for p in preds)
        print(f"[{i+1}/{len(docs)}] preds={len(preds)} golds={len(golds)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {
        "corpus": args.corpus, "docs": len(docs), "threshold": args.threshold,
        "recall": {idc: {"any": hit[idc, "any"] / tot[idc], "typed": hit[idc, "typed"] / tot[idc],
                         "n": tot[idc]} for idc in ("DIRECT", "QUASI")},
        "recall_by_type": {f"{idc}/{et}": {
            "any": hit[(idc, et), "any"] / tot[idc, et],
            "typed": hit[(idc, et), "typed"] / tot[idc, et], "n": tot[idc, et]}
            for (idc, et) in sorted(k for k in tot if isinstance(k, tuple))},
        "entity_level_direct_recall":
            sum(ent_hit[e] == ms for e, ms in ent_all.items()) / max(len(ent_all), 1),
        "precision_proxy": n_pred_gold / max(n_pred, 1), "n_pred": n_pred,
        "wall_s": round(time.time() - t0, 1),
    }
    res["gate_pass"] = res["recall"]["DIRECT"]["any"] >= 0.95
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("recall", "entity_level_direct_recall",
                                          "precision_proxy", "gate_pass", "wall_s")}, indent=2))


if __name__ == "__main__":
    main()
