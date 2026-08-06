"""How much controller authority do utility ties require? (round-14 corrected)

Supersedes the round-13 gate, whose signed results were void (both tie ledgers
store a SORTED action-pair key while orienting `delta_u` by iteration order), and
the round-13 version of this script, which carried four further defects Codex Sol
High identified. All five corrections are applied here:

1. PREFIX-MATCHED LOGITS. The previous version computed tower logits ONCE on each
   document's greedy trajectory and reused them for every cached surrounding
   context, so utilities were context-matched but logits were not. Here each
   cached pair's logits are replayed under that pair's own shared action prefix,
   walking the legal-menu/claimed-fill machinery exactly as `replay_trajectory`
   does. Pairs whose prefix is illegal under injectivity masking are skipped.
2. EFFECTIVE AUTHORITY. Authority is `softplus(alpha_raw + residual)` computed
   per decision through `decision_controller_alpha` -- the real controller path --
   not `softplus(alpha_raw)` (which ignores the trained residual and understated
   authority 2.4x) and not a hard-coded constant.
3. BOTH ORDERING-ERROR DIRECTIONS. A tower ordering error is
   `delta_measured * du < 0`, either direction. The previous version counted only
   `du > 0 and delta_measured < 0`, making 8.6% a one-sided lower bound.
   Zero-logit-difference cases are reported separately.
4. DOCUMENT-MACRO AND DISTINCT-PAIR STATISTICS. Cache coverage is dominated by
   three documents (D2N031 3421 vectors, D2N063 3124, D2N027 2059, then tens), so
   micro statistics are those three documents' geometry. Observations are also not
   independent: the same (doc, decision, action-pair) recurs under many contexts.
5. POPULATION OVERLAP. Reports whether the tie population carrying a privacy
   preference is the same population as the measured utility-equivalent pairs --
   the precondition for the claim that ties CAUSE the scale problem, which the
   literature read flagged as assumed rather than shown.

Required authority is reported at two margins: m=0 (the minimum that flips the
greedy argmax) and m=0.1 (the chosen robustness margin -- a policy choice, not a
measured threshold). Lower bounds only: the 0.044 budget constrains a COMPLETE
document vector, so per-pair upper bounds are mutually exclusive and meaningless.

Cache-only, CPU, no reward calls.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/authority_scale_diagnostic.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

BASE = Path("results/ranker_v2")
OUT = BASE / "architecture/count_to_gain"
CHECKPOINT = OUT / "coupled-s47.pt"
TIE_ATOL = 1e-9
MAGNITUDE = 1.0        # g(lambda) at lambda-max, no gap scaling
MARGINS = (0.0, 0.1)


def _quantiles(values: list[float]) -> dict:
    ordered = sorted(values)

    def q(fraction: float) -> float:
        return ordered[int(fraction * (len(ordered) - 1))]

    return {
        "n": len(ordered), "min": ordered[0], "p10": q(0.10), "median": q(0.50),
        "p90": q(0.90), "max": ordered[-1], "mean": st.mean(ordered),
    }


def _load_environment():
    from cloak.ranker.environment import LambdaProfile, load_ranker_environment
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from train_interactive_ranker import (
        _demote_out_of_scope_decisions,
        _drop_zero_signal_documents,
    )

    count_state = json.loads((BASE / "reward/profile-count-targets.json").read_text())
    artifact = json.loads((BASE / "qa/aci-full.utility").read_text())
    documents = tuple(load_ranker_environment(BASE / "environment/ranker-env.json").values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(count_state)
    )
    documents, _ = _drop_zero_signal_documents(documents, artifact)
    menu = json.loads((BASE / "preflight/lambda-menu.json").read_text())
    profiles = tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(menu["profile_names"], menu["values"], strict=True)
    )
    return documents, profiles, count_state, artifact


def _build_policy(documents, profiles, state_dict):
    """Policy over the FULL document set, so the runtime-type vocabulary that
    indexes `runtime_type_embedding` matches training rather than a subset."""
    from types import SimpleNamespace

    from cloak.ranker.semantic import enable_controller_gain
    from train_interactive_ranker import _semantic_training_policy

    args = SimpleNamespace(
        representation_manifest=str(BASE / "architecture/representation-full/manifest.json"),
        profile_count_targets=str(BASE / "reward/profile-count-targets.json"),
        privacy_checkpoint=None,
        device="cpu",
    )
    policy = _semantic_training_policy(args, documents, profiles)
    if any("gain_head" in key for key in state_dict):
        enable_controller_gain(policy, "evidence", hidden_dim=32, bound=1.5)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.float()
    return policy


def _load_cache(by_id):
    """Cached complete-coverage vectors per document, keyed by ordered actions."""
    cache: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for line in (BASE / "cache/utility-results.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)["result"]
        except (ValueError, KeyError):
            continue
        document = by_id.get(row.get("doc_id"))
        if document is None or not row.get("component_scores"):
            continue
        ids = [d.decision_id for d in document.policy_decisions]
        vector = row.get("action_vector", {})
        if set(vector) != set(ids):
            continue
        cache[document.doc_id][tuple(vector[i] for i in ids)] = row
    return cache


def _prefix_nodes(policy, document, profile, needed: set[tuple]):
    """Replay logits and effective authority at each needed action prefix.

    Walks the prefix trie depth-first, holding only the root-to-node states, and
    reproduces `replay_trajectory`'s legal-menu and claimed-fill bookkeeping so
    the tower sees the state it would actually see under that cached vector.
    Returns {prefix: {"logits": {action_id: float}, "alpha": float}}; prefixes
    that are illegal under injectivity masking are absent.
    """
    from cloak.ranker.environment import (
        _action_by_id, _fill_key, _fixed_fill_claims, _occurrence_maps,
        legal_action_ids,
    )
    from cloak.ranker.semantic import decision_controller_alpha

    tree: dict = {}
    for prefix in needed:
        node = tree
        for action_id in prefix:
            node = node.setdefault(action_id, {})

    occurrence_by_id, _ = _occurrence_maps(document)
    reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
    decisions = document.policy_decisions
    resolved: dict[tuple, dict] = {}

    def walk(node: dict, depth: int, prefix: tuple, state, claimed: dict) -> None:
        if depth >= len(decisions):
            return
        decision = decisions[depth]
        menu = legal_action_ids(decision, claimed, reserved)
        if not menu:
            return
        if prefix in needed:
            with torch.no_grad():
                distribution = policy.distribution(state, decision, menu, profile)
                alpha = decision_controller_alpha(policy, state, decision, menu)
            resolved[prefix] = {
                "logits": {
                    action_id: float(value) for action_id, value
                    in zip(menu, distribution.utility_logits, strict=True)
                },
                "alpha": float(alpha),
            }
        for action_id, child in node.items():
            if action_id not in menu:
                continue          # illegal under this prefix; drop the subtree
            action = _action_by_id(decision, action_id)
            next_claimed = dict(claimed)
            if action.mode == "level":
                assert action.fill is not None
                next_claimed.setdefault(_fill_key(action.fill), decision.decision_id)
            walk(
                child, depth + 1, prefix + (action_id,),
                policy.advance(state, decision, action_id), next_claimed,
            )

    walk(tree, 0, (), policy.begin_document(document, profile), {})
    return resolved


def collect_records(checkpoint_path: Path = CHECKPOINT) -> dict:
    """Per-pair records with prefix-matched logits and effective authority.

    Exposed so downstream diagnostics (observable-state aggregation) reuse the
    prefix walk instead of re-implementing it.
    """
    from cloak.reward.utility_credit import _partitions, document_utility
    from cloak.ranker.profile_count import ProfileCountTargets

    documents, profiles, count_state, artifact = _load_environment()
    by_id = {d.doc_id: d for d in documents}
    targets = ProfileCountTargets.from_artifact(count_state)
    cache = _load_cache(by_id)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["policy_state_dict"]
    policy = _build_policy(documents, profiles, state_dict)
    alpha_raw_only = float(torch.nn.functional.softplus(state_dict["alpha_raw"]))
    profile = profiles[-1]
    # the checkpoint records its own training documents; do not infer them
    training_documents = frozenset(checkpoint["schedule"]["offsets_by_document"])

    records: list[dict] = []
    total_pairs = skipped_no_logits = 0

    for doc_id, cached in sorted(cache.items()):
        document = by_id[doc_id]
        ids = [d.decision_id for d in document.policy_decisions]
        partitions = _partitions(artifact, doc_id)

        # single-decision pairs share BOTH prefix and suffix (a common context)
        buckets: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for vector, row in cached.items():
            for position in range(len(vector)):
                key = (position, vector[:position], vector[position + 1:])
                buckets[key][vector[position]] = row
        buckets = {key: value for key, value in buckets.items() if len(value) >= 2}
        if not buckets:
            continue

        nodes = _prefix_nodes(
            policy, document, profile, {key[1] for key in buckets}
        )

        for (position, prefix, _suffix), rows in buckets.items():
            decision_id = ids[position]
            node = nodes.get(prefix)
            action_ids = sorted(rows)
            for i in range(len(action_ids)):
                for j in range(i + 1, len(action_ids)):
                    action_a, action_b = action_ids[i], action_ids[j]
                    total_pairs += 1
                    if node is None or action_a not in node["logits"] \
                            or action_b not in node["logits"]:
                        skipped_no_logits += 1
                        continue
                    score_a = float(targets.action_scores(decision_id, (action_a,))[0])
                    score_b = float(targets.action_scores(decision_id, (action_b,))[0])
                    delta_p = abs(score_a - score_b)
                    tied_privacy = delta_p < 1e-9
                    # EXPLICIT orientation: plus = privacy-preferred
                    plus, minus = (
                        (action_a, action_b) if score_a >= score_b
                        else (action_b, action_a)
                    )
                    delta_measured = (
                        document_utility(rows[plus]["component_scores"], artifact, doc_id)
                        - document_utility(rows[minus]["component_scores"], artifact, doc_id)
                    )
                    records.append({
                        "doc_id": doc_id,
                        "decision_id": decision_id,
                        "prefix": prefix,
                        "pair": (action_a, action_b),
                        "delta_p": delta_p,
                        "tied_privacy": tied_privacy,
                        "du": node["logits"][plus] - node["logits"][minus],
                        "alpha": node["alpha"],
                        "delta_u": delta_measured,
                        "utility_tied": abs(delta_measured) <= TIE_ATOL,
                        "structural": not partitions.linked_by_decision.get(decision_id),
                        "training": doc_id in training_documents,
                    })

    return {
        "records": records, "pairs_enumerated": total_pairs,
        "pairs_skipped": skipped_no_logits, "alpha_raw_only": alpha_raw_only,
        "training_documents": sorted(training_documents),
    }


def main() -> None:
    collected = collect_records()
    _report(
        collected["records"], collected["pairs_enumerated"],
        collected["pairs_skipped"], collected["alpha_raw_only"],
    )


def _exceedance(rows: list[dict], margin: float) -> dict:
    """Required authority vs the authority the controller actually had."""
    bounds = [(margin - r["du"]) / (MAGNITUDE * r["delta_p"]) for r in rows]
    over = [b > r["alpha"] for b, r in zip(bounds, rows, strict=True)]
    by_doc: dict[str, list[bool]] = defaultdict(list)
    for r, flag in zip(rows, over, strict=True):
        by_doc[r["doc_id"]].append(flag)
    distinct: dict[tuple, list[float]] = defaultdict(list)
    for r, b in zip(rows, bounds, strict=True):
        distinct[(r["doc_id"], r["decision_id"], r["pair"])].append(b - r["alpha"])
    return {
        "bounds": _quantiles(bounds),
        "micro_exceeding": sum(over),
        "micro_fraction": sum(over) / len(over),
        "doc_macro_fraction": st.mean(
            sum(flags) / len(flags) for flags in by_doc.values()
        ),
        "documents": len(by_doc),
        "distinct_pairs": len(distinct),
        "distinct_exceeding_fraction": st.mean(
            st.median(gaps) > 0.0 for gaps in distinct.values()
        ),
    }


def _report(records, total_pairs, skipped, alpha_raw_only) -> None:
    scored = [r for r in records if r["utility_tied"] and not r["tied_privacy"]]
    ties_all = [r for r in records if r["utility_tied"]]
    live = [r for r in records if not r["utility_tied"]]
    alphas = [r["alpha"] for r in records]

    print("=== COVERAGE ===")
    print(f"  single-decision pairs enumerated: {total_pairs}")
    print(f"  usable (prefix-legal, both actions in menu): {len(records)}")
    print(f"  skipped (illegal prefix or action off-menu): {skipped}")
    print(f"  documents: {len({r['doc_id'] for r in records})}")
    print("\n=== AUTHORITY ACTUALLY AVAILABLE (per decision, real controller path) ===")
    print(f"  softplus(alpha_raw) alone: {alpha_raw_only:.3f}  <- NOT the comparator")
    if alphas:
        stats = _quantiles(alphas)
        print(f"  effective softplus(alpha_raw + residual): median {stats['median']:.3f}"
              f"  min {stats['min']:.3f}  max {stats['max']:.3f}"
              f"  spread {stats['max'] - stats['min']:.2e}")

    print("\n=== A. SCALE: authority required to win a utility tie ===")
    for label, rows in (
        ("all verified ties w/ privacy preference", scored),
        ("  structural subset (no declared link)", [r for r in scored if r["structural"]]),
    ):
        if not rows:
            print(f"  {label}: none")
            continue
        print(f"  {label}: n={len(rows)}")
        for margin in MARGINS:
            result = _exceedance(rows, margin)
            bounds = result["bounds"]
            print(f"    m={margin}: required median {bounds['median']:.2f}"
                  f"  p90 {bounds['p90']:.2f}  max {bounds['max']:.2f}")
            print(f"      exceeding available authority:"
                  f" micro {result['micro_fraction'] * 100:.0f}%"
                  f"  doc-macro {result['doc_macro_fraction'] * 100:.0f}%"
                  f"  distinct-pair {result['distinct_exceeding_fraction'] * 100:.0f}%"
                  f"  ({result['distinct_pairs']} distinct pairs,"
                  f" {result['documents']} docs)")
        # a large required authority can come from a tiny privacy gap, not a large margin
        ordered = sorted(rows, key=lambda r: r["delta_p"])
        third = max(1, len(ordered) // 3)
        for name, part in (
            ("smallest dp third", ordered[:third]), ("largest dp third", ordered[-third:]),
        ):
            result = _exceedance(part, 0.1)
            print(f"      [{name}] dp median "
                  f"{st.median([r['delta_p'] for r in part]):.4f}"
                  f" -> required median {result['bounds']['median']:.2f},"
                  f" micro {result['micro_fraction'] * 100:.0f}%")

    print("\n=== B. TOWER ORDERING ERRORS (both directions) ===")
    if live:
        errors = [r for r in live if r["delta_u"] * r["du"] < 0.0]
        zero_logit = [r for r in live if abs(r["du"]) <= 1e-12]
        prefers_losing = [r for r in live if r["du"] > 0.0 > r["delta_u"]]
        prefers_winning_less = [r for r in live if r["du"] < 0.0 < r["delta_u"]]
        by_doc: dict[str, list[bool]] = defaultdict(list)
        for r in live:
            by_doc[r["doc_id"]].append(r["delta_u"] * r["du"] < 0.0)
        print(f"  measurably live pairs: {len(live)}")
        print(f"  sign disagreements (delta_u * du < 0): {len(errors)}"
              f" ({100 * len(errors) / len(live):.1f}% micro,"
              f" {100 * st.mean(sum(f) / len(f) for f in by_doc.values()):.1f}% doc-macro)")
        print(f"    of which privacy-preferred wins the logit but loses utility:"
              f" {len(prefers_losing)}")
        print(f"    of which privacy-preferred loses the logit but gains utility:"
              f" {len(prefers_winning_less)}  <- omitted by the previous version")
        print(f"  exact logit ties among live pairs: {len(zero_logit)}")
        # a pair whose measured utility difference is near zero is a coin flip by
        # construction, so the rate must be read against effect size
        ordered = sorted(live, key=lambda r: abs(r["delta_u"]))
        fifth = max(1, len(ordered) // 5)
        for name, part in (
            ("smallest |dU| fifth", ordered[:fifth]),
            ("middle", ordered[2 * fifth:3 * fifth]),
            ("largest |dU| fifth", ordered[-fifth:]),
        ):
            wrong = sum(1 for r in part if r["delta_u"] * r["du"] < 0.0)
            print(f"    [{name}] |dU| median "
                  f"{st.median([abs(r['delta_u']) for r in part]):.4f}"
                  f" -> disagreement {100 * wrong / len(part):.1f}%")

    print("\n=== C. POPULATION OVERLAP (do ties cause the scale problem?) ===")
    print(f"  measured utility-equivalent pairs:        {len(ties_all)}"
          f" ({100 * len(ties_all) / max(1, len(records)):.1f}% of usable)")
    print(f"    carrying a privacy preference (scored): {len(scored)}")
    print(f"    privacy-tied too (no preference to win): {len(ties_all) - len(scored)}")
    print(f"  measurably live pairs:                    {len(live)}")
    print("  The scale population is the utility-equivalent set minus privacy-tied")
    print("  pairs, so it is a SUBSET, not a distinct population.")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(CHECKPOINT),
        "alpha_raw_only": alpha_raw_only,
        "effective_alpha": _quantiles(alphas) if alphas else None,
        "coverage": {
            "pairs_enumerated": total_pairs, "pairs_usable": len(records),
            "pairs_skipped": skipped,
            "documents": sorted({r["doc_id"] for r in records}),
        },
        "scale": {
            f"m={margin}": _exceedance(scored, margin) for margin in MARGINS
        } if scored else None,
        "scale_structural": {
            f"m={margin}": _exceedance([r for r in scored if r["structural"]], margin)
            for margin in MARGINS
        } if any(r["structural"] for r in scored) else None,
        "ordering": {
            "live_pairs": len(live),
            "sign_disagreements": sum(1 for r in live if r["delta_u"] * r["du"] < 0.0),
            "prefers_utility_losing": sum(1 for r in live if r["du"] > 0.0 > r["delta_u"]),
            "prefers_utility_gaining_less": sum(
                1 for r in live if r["du"] < 0.0 < r["delta_u"]
            ),
        },
        "populations": {
            "utility_equivalent": len(ties_all), "scored_ties": len(scored),
            "privacy_tied_ties": len(ties_all) - len(scored), "live": len(live),
        },
    }
    (OUT / "authority-scale-diagnostic.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'authority-scale-diagnostic.json'}")


if __name__ == "__main__":
    main()
