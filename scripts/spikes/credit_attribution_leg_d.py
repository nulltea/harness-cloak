"""Leg D — is excluding spillover from per-decision credit BIASED or just lower-variance?

Preregistration: research-wiki/experiments/2026-08-03-credit-attribution-validation.md

A credit signal is measured in the contexts the probe happened to visit, but the
policy it trains acts in OTHER contexts. So the quantity that matters is how well
a signal measured in one set of surrounding action-vectors predicts the TRUE
(total-document) effect in held-out ones. Leave-one-context-out over every
(document, decision, action-pair) group with repeats:

  target       total document delta in the HELD-OUT context
  attributed   mean attributed delta over the training contexts  (what we now use)
  total        mean total delta over the training contexts       (what we used before)
  corrected    attributed + a pooled spillover estimate, cross-fitted across
               OTHER DOCUMENTS and shrunk toward zero (never per-pair, which
               would refit the noise it is meant to average out)

Decision rules, fixed before the run:
  attributed ~= total on sign, with lower spread   -> exclusion justified, no model
  corrected  >  attributed materially              -> spillover is predictable, so
                                                      excluding it is biased
  total      >  both                               -> the revision is wrong
Cache-only: no remote calls, no reader calls, no GPU.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

OUT = Path("results/ranker_v2/architecture/credit_attribution")
ATOL = 1e-9


def _environment():
    from cloak.ranker.environment import load_ranker_environment
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from train_interactive_ranker import _demote_out_of_scope_decisions

    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents,
        DirectCountPrivacyProvider(json.loads(
            Path("results/ranker_v2/reward/profile-count-targets.json").read_text()
        )),
    )
    return {document.doc_id: document for document in documents}


def main() -> None:
    from cloak.reward.utility_credit import (
        _partitions,
        attributed_delta_utility,
        document_utility,
        excerpt_changed_assertions,
    )

    artifact = json.loads(Path("results/ranker_v2/qa/aci-full.utility").read_text())
    documents = _environment()

    rows: list[dict] = []
    cache: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for line in Path(
        "results/ranker_v2/cache/utility-results.jsonl"
    ).read_text().splitlines():
        try:
            row = json.loads(line)["result"]
        except (ValueError, KeyError):
            continue
        document = documents.get(row.get("doc_id"))
        if document is None:
            continue
        ids = [d.decision_id for d in document.policy_decisions]
        if not ids or set(row.get("action_vector", {})) != set(ids):
            continue
        cache[document.doc_id][tuple(row["action_vector"][i] for i in ids)] = row

    for doc_id, cached in sorted(cache.items()):
        ids = [d.decision_id for d in documents[doc_id].policy_decisions]
        partitions = _partitions(artifact, doc_id)
        keys = list(cached)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                differing = [p for p in range(len(ids)) if a[p] != b[p]]
                if len(differing) != 1:
                    continue
                position = differing[0]
                decision_id = ids[position]
                scores_a = cached[a]["component_scores"]
                scores_b = cached[b]["component_scores"]
                total = (
                    document_utility(scores_a, artifact, doc_id)
                    - document_utility(scores_b, artifact, doc_id)
                )
                attributed, _ = attributed_delta_utility(
                    scores_a, scores_b, artifact, doc_id, decision_id,
                    excerpt_changed_assertions(
                        artifact, doc_id, cached[a]["doc_p"], cached[b]["doc_p"],
                    ),
                )
                # orient consistently so signs are comparable across contexts
                first, second = sorted((a[position], b[position]))
                flip = 1.0 if a[position] == first else -1.0
                context = tuple(
                    v for p, v in enumerate(a) if p != position
                )
                rows.append({
                    "doc_id": doc_id,
                    "group": (doc_id, decision_id, first, second),
                    "context": context,
                    "total": flip * total,
                    "attributed": flip * attributed,
                    "route": (
                        "linked" if decision_id in partitions.linked_by_decision
                        else "document"
                    ),
                })

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    usable = {
        key: values for key, values in groups.items()
        if len({row["context"] for row in values}) >= 2
    }

    # Pooled spillover, cross-fitted by DOCUMENT: the estimate applied to a
    # document is built only from other documents, and shrunk toward zero by the
    # fraction of its own variance that is between-document signal.
    by_document: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_document[row["doc_id"]].append(row["total"] - row["attributed"])
    all_spill = [v for values in by_document.values() for v in values]
    grand = st.mean(all_spill)

    def pooled_spillover(doc_id: str) -> float:
        other = [
            v for other_doc, values in by_document.items()
            if other_doc != doc_id for v in values
        ]
        if len(other) < 2:
            return 0.0
        mean = st.mean(other)
        # shrink toward zero by the estimate's own noise (James-Stein flavour);
        # never fit a raw per-pair observation
        se = st.pstdev(other) / len(other) ** 0.5
        if abs(mean) <= se:
            return 0.0
        return mean * (1.0 - (se / abs(mean)) ** 2)

    stats: dict[str, dict[str, list]] = {
        name: {"sign": [], "abs_err": [], "pred": []}
        for name in ("attributed", "total", "corrected")
    }
    by_route: dict[str, dict[str, list]] = defaultdict(
        lambda: {name: [] for name in ("attributed", "total", "corrected")}
    )
    folds = 0
    for key, values in usable.items():
        contexts = sorted({row["context"] for row in values})
        for held in contexts:
            train = [row for row in values if row["context"] != held]
            test = [row for row in values if row["context"] == held]
            if not train or not test:
                continue
            target = st.mean(row["total"] for row in test)
            if abs(target) <= ATOL:
                continue
            folds += 1
            spill = pooled_spillover(key[0])
            predictions = {
                "attributed": st.mean(row["attributed"] for row in train),
                "total": st.mean(row["total"] for row in train),
                "corrected": st.mean(row["attributed"] for row in train) + spill,
            }
            route = test[0]["route"]
            for name, value in predictions.items():
                agree = 1.0 if value * target > 0 else 0.0
                stats[name]["sign"].append(agree)
                stats[name]["abs_err"].append(abs(value - target))
                stats[name]["pred"].append(value)
                by_route[route][name].append(agree)

    print(f"groups with repeated contexts: {len(usable)} | leave-one-out folds: {folds}")
    print(f"pooled spillover (grand mean, unshrunk): {grand:+.5f}\n")
    print(f"{'predictor':12s} {'sign agreement':>15s} {'MAE':>9s} {'pred sd':>9s}")
    for name in ("total", "attributed", "corrected"):
        row = stats[name]
        print(
            f"{name:12s} {st.mean(row['sign']):>14.3f}  {st.mean(row['abs_err']):>9.4f}"
            f" {st.pstdev(row['pred']):>9.4f}"
        )
    print("\nsign agreement by credit route:")
    for route, table in sorted(by_route.items()):
        counts = {name: len(v) for name, v in table.items()}
        print(
            f"  {route:9s} n={counts['attributed']:5d}  "
            + "  ".join(
                f"{name} {st.mean(values):.3f}" for name, values in table.items()
            )
        )

    best = max(("total", "attributed", "corrected"), key=lambda n: st.mean(stats[n]["sign"]))
    print(f"\nhighest sign agreement: {best}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leg-d-report.json").write_text(json.dumps({
        "groups": len(usable), "folds": folds,
        "pooled_spillover_grand_mean": grand,
        "predictors": {
            name: {
                "sign_agreement": st.mean(row["sign"]),
                "mae": st.mean(row["abs_err"]),
                "prediction_sd": st.pstdev(row["pred"]),
            }
            for name, row in stats.items()
        },
        "sign_agreement_by_route": {
            route: {name: st.mean(v) for name, v in table.items()}
            for route, table in by_route.items()
        },
    }, indent=1))


if __name__ == "__main__":
    main()
