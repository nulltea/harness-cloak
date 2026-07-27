"""Calibrate the entity-merge cross-encoder gate on ontology-derived pairs.

Positives: (name, EXACT synonym) pairs of non-obsolete DOID terms. Hard negatives:
same-parent sibling name pairs (the class that must never merge). Reports a threshold sweep
and the chosen threshold (smallest t with precision >= 0.999 and recall >= 0.10), or null -
then the gate ships disabled and the numbers are the finding (empirical-honesty rule: the
threshold is calibrated once here and frozen; never per-run tuning).

GPU job: run as
  .venv/bin/python -u scripts/calibrate_entity_merge_gate.py \
      --out results/entity_merge_gate_eval.json 2>&1 | tee results/entity_merge_gate_eval.log
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloak.lattice.producer.reference_sources import DEFAULT_DOID_OBO, load_doid_index

PRECISION_BAR = 0.999
RECALL_FLOOR = 0.10
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


def build_eval_pairs(nodes: dict, sample: int, seed: int):
    """(positives, negatives): synonym pairs vs same-parent sibling name pairs."""
    rng = random.Random(seed)
    live = {nid: n for nid, n in nodes.items() if not n.obsolete}
    positives = [(n.name, syn) for n in live.values() for syn in n.exact_synonyms]
    by_parent = defaultdict(list)
    for n in live.values():
        for p in n.parents:
            by_parent[p].append(n.name)
    negatives = [(sibs[i], sibs[j]) for sibs in by_parent.values()
                 for i in range(len(sibs)) for j in range(i + 1, len(sibs))]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:sample], negatives[:sample]


def choose_threshold(scored: list[tuple[float, bool]], precision_bar: float,
                     recall_floor: float) -> float | None:
    """scored: (score, is_positive). Smallest threshold meeting both bars, else None."""
    total_pos = sum(1 for _, y in scored if y)
    best = None
    for t in sorted({s for s, _ in scored}):
        pred = [(s, y) for s, y in scored if s >= t]
        if not pred:
            continue
        tp = sum(1 for _, y in pred if y)
        precision = tp / len(pred)
        recall = tp / total_pos if total_pos else 0.0
        if precision >= precision_bar and recall >= recall_floor:
            best = t
            break
    return best


def nli_gate_scorer(model_id: str):
    """Same-entity score for a surface pair: min of both-direction entailment probs."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    entail_idx = next(i for i, name in model.config.id2label.items()
                      if name.lower().startswith("entail"))

    def score_pairs(pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        out = []
        both = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
        probs = []
        with torch.no_grad():
            for i in range(0, len(both), batch_size):
                chunk = both[i:i + batch_size]
                enc = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                          truncation=True, return_tensors="pt").to(model.device)
                p = torch.softmax(model(**enc).logits, dim=-1)[:, entail_idx]
                probs.extend(p.tolist())
        n = len(pairs)
        for i in range(n):
            out.append(min(probs[i], probs[n + i]))
        return out

    return score_pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obo", default=str(DEFAULT_DOID_OBO))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/entity_merge_gate_eval.json")
    args = ap.parse_args(argv)

    nodes = load_doid_index(args.obo)
    positives, negatives = build_eval_pairs(nodes, args.sample, args.seed)
    print(f"eval pairs: {len(positives)} positives, {len(negatives)} negatives", flush=True)

    scorer = nli_gate_scorer(args.model)
    scored = list(zip(scorer(positives), [True] * len(positives))) + \
             list(zip(scorer(negatives), [False] * len(negatives)))

    total_pos = len(positives)
    sweep = []
    for t in [round(0.05 * i, 2) for i in range(1, 20)]:
        pred = [(s, y) for s, y in scored if s >= t]
        tp = sum(1 for _, y in pred if y)
        sweep.append({"threshold": t,
                      "precision": round(tp / len(pred), 4) if pred else None,
                      "recall": round(tp / total_pos, 4) if total_pos else 0.0,
                      "predicted_positives": len(pred)})
    chosen = choose_threshold(scored, PRECISION_BAR, RECALL_FLOOR)
    out = {"model_id": args.model, "obo": args.obo, "sample": args.sample, "seed": args.seed,
           "positives": len(positives), "negatives": len(negatives),
           "precision_bar": PRECISION_BAR, "recall_floor": RECALL_FLOOR,
           "sweep": sweep, "chosen_threshold": chosen}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"chosen_threshold={chosen} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
