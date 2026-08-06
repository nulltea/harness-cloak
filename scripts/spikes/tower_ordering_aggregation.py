"""Is the tower's 41% ordering error real, or an impossible-target artifact?

The prefix-matched diagnostic (`authority_scale_diagnostic.py`) measured 41% sign
disagreement between the tower's utility logits and measured document utility on
live pairs. Codex Sol High (round 15) identified a target mismatch that could
account for much of it:

  the tower's logit is conditioned on the document and the selected-action PREFIX;
  the measured dU is conditioned on that prefix PLUS one particular future SUFFIX.

For a single observable state and action pair, different cached suffixes can carry
OPPOSITE-signed utility effects. No prefix-conditioned predictor can be right
about both, so some disagreement is irreducible. The policy-relevant target is
Q_U(s,a) = E_{future ~ pi}[U | s, a], not utility under every fixed cached suffix.

This script separates the two. For every observable-state group
(doc, exact prefix, decision, action pair) it reports suffix multiplicity, sign
consistency, the majority-sign (Bayes) accuracy ceiling a perfect
prefix-conditioned predictor could reach, and the tower's agreement with the
AGGREGATED effect. If the ceiling is far below 100%, much of the 41% is aliasing;
if the tower falls far short of the ceiling, it is genuine ordering failure.

Two further splits, both requested and both confounded if read naively:

  * TRAINING vs HELD-OUT documents. The checkpoint records its own training set
    (four documents). With an effective cross-document sample of four, a poor
    held-out number is uninterpretable on its own -- it is equally consistent with
    "the tower cannot generalize" and "nothing generalizes from four documents".
    It is reported descriptively, and always WITHIN effect-size strata, because
    training documents are sampled ~500x more densely and therefore contribute a
    completely different |dU| mixture (near-tie pairs are coin flips by
    construction). Reading the raw train/held-out gap would measure pair density,
    not generalization.
  * DOCUMENT-CLUSTERED BOOTSTRAP. Observations are not independent: 3,062 tie
    observations collapse to ~306 distinct pairs, themselves nested in documents.
    Intervals resample documents, the top-level cluster.

Cache-only, CPU, no reward calls. Reuses the prefix walk rather than re-deriving it.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/tower_ordering_aggregation.py
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

OUT = Path("results/ranker_v2/architecture/count_to_gain")
BOOTSTRAP = 2000
SEED = 47
QUINTILES = 5


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _group_key(record: dict) -> tuple:
    return (record["doc_id"], record["prefix"], record["pair"])


def _bootstrap(by_document: dict[str, list], statistic, rounds: int = BOOTSTRAP) -> dict:
    """Percentile interval resampling DOCUMENTS, the top-level cluster."""
    doc_ids = sorted(by_document)
    if len(doc_ids) < 2:
        return {"low": None, "high": None, "documents": len(doc_ids)}
    rng = random.Random(SEED)
    draws: list[float] = []
    for _ in range(rounds):
        pooled: list = []
        for _ in doc_ids:
            pooled.extend(by_document[doc_ids[rng.randrange(len(doc_ids))]])
        value = statistic(pooled)
        if value is not None:
            draws.append(value)
    if not draws:
        return {"low": None, "high": None, "documents": len(doc_ids)}
    draws.sort()
    return {
        "low": draws[int(0.025 * (len(draws) - 1))],
        "high": draws[int(0.975 * (len(draws) - 1))],
        "documents": len(doc_ids),
    }


def _disagreement(rows: list[dict]) -> float | None:
    return (
        sum(1 for r in rows if r["delta_u"] * r["du"] < 0.0) / len(rows)
        if rows else None
    )


def _quintile_bounds(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        ordered[min(len(ordered) - 1, int(q * len(ordered) / QUINTILES))]
        for q in range(1, QUINTILES)
    ]


def _stratum(value: float, bounds: list[float]) -> int:
    return sum(1 for bound in bounds if value > bound)


def main() -> None:
    from authority_scale_diagnostic import collect_records

    collected = collect_records()
    records = collected["records"]
    training = set(collected["training_documents"])
    live = [r for r in records if not r["utility_tied"]]
    ties = [r for r in records if r["utility_tied"]]

    print("=== SETUP ===")
    print(f"  training documents (from checkpoint schedule): {sorted(training)}")
    print(f"  usable pairs {len(records)}  live {len(live)}  utility-tied {len(ties)}")
    held_out_docs = {r["doc_id"] for r in records} - training
    print(f"  held-out documents: {len(held_out_docs)}")

    # ---------- A. suffix multiplicity and the irreducible ceiling ----------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in live:
        groups[_group_key(record)].append(record)

    multiplicity = [len(rows) for rows in groups.values()]
    contested = {k: rows for k, rows in groups.items() if len(rows) > 1}
    # a perfect prefix-conditioned sign predictor picks each group's majority sign
    majority_correct = 0
    inconsistent_groups = 0
    for rows in groups.values():
        signs = [_sign(r["delta_u"]) for r in rows]
        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)
        majority_correct += max(positive, negative)
        if positive and negative:
            inconsistent_groups += 1
    ceiling = majority_correct / len(live)

    print("\n=== A. SUFFIX ALIASING: how much of the error is impossible to avoid? ===")
    print(f"  observable-state groups (doc, prefix, decision, action pair): {len(groups)}")
    print(f"    with >1 cached suffix: {len(contested)}"
          f"  ({100 * len(contested) / len(groups):.0f}%)")
    print(f"    suffix multiplicity: median {st.median(multiplicity)}"
          f"  mean {st.mean(multiplicity):.1f}  max {max(multiplicity)}")
    print(f"    SIGN-INCONSISTENT groups (both signs present): {inconsistent_groups}"
          f"  ({100 * inconsistent_groups / len(groups):.1f}% of groups,"
          f" {100 * inconsistent_groups / max(1, len(contested)):.1f}% of contested)")
    print(f"  best achievable accuracy for ANY prefix-conditioned predictor:"
          f" {100 * ceiling:.1f}%")
    print(f"    => irreducible disagreement floor: {100 * (1 - ceiling):.1f}%")
    observed = _disagreement(live)
    print(f"  observed disagreement: {100 * observed:.1f}%")
    print(f"  tower's avoidable shortfall below the ceiling:"
          f" {100 * (observed - (1 - ceiling)):.1f} points")

    # ---------- B. tower vs the AGGREGATED effect ----------
    aggregated = []
    for key, rows in groups.items():
        mean_delta = st.mean(r["delta_u"] for r in rows)
        aggregated.append({
            "doc_id": key[0], "pair": key[2], "delta_u": mean_delta,
            "du": rows[0]["du"], "training": rows[0]["training"],
            "observations": len(rows),
        })
    non_zero = [g for g in aggregated if abs(g["delta_u"]) > 1e-12]
    group_disagreement = _disagreement(non_zero)
    by_doc_groups: dict[str, list] = defaultdict(list)
    for group in non_zero:
        by_doc_groups[group["doc_id"]].append(group)
    interval = _bootstrap(by_doc_groups, _disagreement)
    print("\n=== B. TOWER vs AGGREGATED EFFECT (the policy-relevant target) ===")
    print(f"  groups with a non-zero mean effect: {len(non_zero)}")
    print(f"  sign disagreement against mean dU: {100 * group_disagreement:.1f}%"
          f"  [95% CI {100 * interval['low']:.1f}, {100 * interval['high']:.1f}]"
          f"  (document-clustered, {interval['documents']} docs)")

    # ---------- C. train vs held-out, WITHIN effect-size strata ----------
    bounds = _quintile_bounds([abs(r["delta_u"]) for r in live])
    print("\n=== C. TRAINING vs HELD-OUT, within |dU| strata ===")
    print("  (raw split is confounded with pair density; strata make it comparable)")
    print(f"  |dU| quintile bounds: {[round(b, 4) for b in bounds]}")
    header = f"  {'stratum':<10}{'train n':>9}{'train err':>11}{'held n':>9}{'held err':>10}"
    print(header)
    for stratum in range(QUINTILES):
        part = [r for r in live if _stratum(abs(r["delta_u"]), bounds) == stratum]
        train_rows = [r for r in part if r["training"]]
        held_rows = [r for r in part if not r["training"]]
        train_error = _disagreement(train_rows)
        held_error = _disagreement(held_rows)
        print(f"  Q{stratum + 1:<9}{len(train_rows):>9}"
              f"{'  n/a' if train_error is None else f'{100 * train_error:>10.1f}%'}"
              f"{len(held_rows):>9}"
              f"{'  n/a' if held_error is None else f'{100 * held_error:>9.1f}%'}")
    raw_train = _disagreement([r for r in live if r["training"]])
    raw_held = _disagreement([r for r in live if not r["training"]])
    print(f"  raw (confounded, do not read as generalization):"
          f" train {100 * raw_train:.1f}%  held-out {100 * raw_held:.1f}%")

    # ---------- D. clustered interval on the tie scale finding ----------
    scored = [r for r in ties if not r["tied_privacy"]]
    by_doc_ties: dict[str, list] = defaultdict(list)
    for record in scored:
        by_doc_ties[record["doc_id"]].append(record)

    def exceed(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return sum(
            1 for r in rows if (0.1 - r["du"]) / r["delta_p"] > r["alpha"]
        ) / len(rows)

    tie_interval = _bootstrap(by_doc_ties, exceed)
    distinct_ties = {(r["doc_id"], r["decision_id"], r["pair"]) for r in scored}
    print("\n=== D. TIE SCALE FINDING, document-clustered interval ===")
    print(f"  scored tie observations {len(scored)}"
          f"  distinct pairs {len(distinct_ties)}  documents {len(by_doc_ties)}")
    print(f"  fraction requiring more authority than available (m=0.1):"
          f" {100 * exceed(scored):.1f}%"
          f"  [95% CI {100 * tie_interval['low']:.1f}, {100 * tie_interval['high']:.1f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_documents": sorted(training),
        "held_out_documents": len(held_out_docs),
        "aliasing": {
            "groups": len(groups),
            "contested_groups": len(contested),
            "sign_inconsistent_groups": inconsistent_groups,
            "suffix_multiplicity_median": st.median(multiplicity),
            "suffix_multiplicity_mean": st.mean(multiplicity),
            "prefix_conditioned_ceiling": ceiling,
            "irreducible_floor": 1 - ceiling,
            "observed_disagreement": observed,
            "avoidable_shortfall": observed - (1 - ceiling),
        },
        "aggregated_target": {
            "groups": len(non_zero),
            "disagreement": group_disagreement,
            "ci95": [interval["low"], interval["high"]],
        },
        "train_vs_held_out": {
            "quintile_bounds": bounds,
            "raw_train": raw_train, "raw_held_out": raw_held,
            "strata": [
                {
                    "stratum": stratum + 1,
                    "train_n": sum(
                        1 for r in live
                        if r["training"] and _stratum(abs(r["delta_u"]), bounds) == stratum
                    ),
                    "train_error": _disagreement([
                        r for r in live
                        if r["training"] and _stratum(abs(r["delta_u"]), bounds) == stratum
                    ]),
                    "held_out_n": sum(
                        1 for r in live
                        if not r["training"]
                        and _stratum(abs(r["delta_u"]), bounds) == stratum
                    ),
                    "held_out_error": _disagreement([
                        r for r in live
                        if not r["training"]
                        and _stratum(abs(r["delta_u"]), bounds) == stratum
                    ]),
                }
                for stratum in range(QUINTILES)
            ],
        },
        "tie_scale": {
            "observations": len(scored), "distinct_pairs": len(distinct_ties),
            "exceeding_fraction": exceed(scored),
            "ci95": [tie_interval["low"], tie_interval["high"]],
        },
    }
    (OUT / "tower-ordering-aggregation.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'tower-ordering-aggregation.json'}")


if __name__ == "__main__":
    main()
