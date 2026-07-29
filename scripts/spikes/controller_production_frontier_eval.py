"""Frontier-regret evaluation of the production-trainer controller runs.

Applies the controller-strength fork's preregistered instrument (decision log
2026-07-28, adjudication 2026-07-29) to production `train` epoch reports:
per-(doc, profile, epoch) groups are reconstructed by joining each report's
`conditional_samples` action-vector hashes back to the utility cache; the
frontier pool is every cached full-coverage vector per document. Utility
regret = max cached utility at count score >= the group's, minus the group's
utility; the passing rule is median regret <= 0.044 per seed (reader-noise
floor). Companion run record:
research-wiki/training/2026-07-29-RL-ranker-v8-controller-production-frontier.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

from cloak.ranker.environment import load_ranker_environment
from cloak.ranker.interactive import _vector_key, stable_hash
from cloak.ranker.privacy import DirectCountPrivacyProvider
from cloak.ranker.profile_count import ProfileCountTargets
from train_interactive_ranker import _demote_out_of_scope_decisions

DOC_IDS = ("aci/D2N005", "aci/D2N027", "aci/D2N063", "aci/D2N031")
SEEDS = (17, 29, 47)
RESULTS = Path("results/ranker_v2/architecture/controller_production")
REGRET_FLOOR = 0.044


def main() -> None:
    targets_payload = json.loads(
        Path("results/ranker_v2/reward/profile-count-targets.json").read_text()
    )
    targets = ProfileCountTargets.from_artifact(targets_payload)
    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(targets_payload)
    )
    docs = {d.doc_id: d for d in documents if d.doc_id in DOC_IDS}
    menu = json.loads(
        Path("results/ranker_v2/preflight/lambda-menu.json").read_text()
    )
    lambda_by_profile = dict(zip(
        menu["profile_names"], (float(v) for v in menu["values"]), strict=True,
    ))

    # Cached (P, U) pool per doc + hash join for report groups.
    pool: dict[str, list[tuple[float, float]]] = {}
    by_hash: dict[str, dict[str, tuple[float, float]]] = {}
    for line in Path(
        "results/ranker_v2/cache/utility-results.jsonl"
    ).read_text().splitlines():
        try:
            row = json.loads(line)["result"]
        except json.JSONDecodeError:
            # trailing partial line while a trainer is appending
            continue
        doc_id = row["doc_id"]
        if doc_id not in docs:
            continue
        document = docs[doc_id]
        ids = {dec.decision_id for dec in document.policy_decisions}
        if set(row["action_vector"]) != ids:
            continue
        scores = [
            float(targets.action_scores(
                dec.decision_id, (row["action_vector"][dec.decision_id],),
            )[0])
            for dec in document.policy_decisions
        ]
        point = (sum(scores) / len(scores), float(row["utility"]))
        pool.setdefault(doc_id, []).append(point)
        vector_hash = stable_hash(list(_vector_key(document, row["action_vector"])))
        by_hash.setdefault(doc_id, {})[vector_hash] = point

    print("pool sizes:", {doc: len(rows) for doc, rows in sorted(pool.items())})

    verdicts = {}
    for seed in SEEDS:
        report_path = RESULTS / f"epochs-s{seed}.jsonl"
        if not report_path.exists():
            print(f"seed {seed}: no epoch reports yet — skipped")
            continue
        regrets: list[float] = []
        groups = []
        for line in report_path.read_text().splitlines():
            report = json.loads(line)
            if report.get("run") != "conditional":
                continue
            epoch = int(report["epoch"])
            for doc_id, profile_rows in report.get("conditional_samples", {}).items():
                for profile_name, hashes in profile_rows.items():
                    lam = lambda_by_profile[profile_name]
                    members = []
                    for vector_hash in hashes:
                        point = by_hash.get(doc_id, {}).get(vector_hash)
                        if point is None:
                            raise ValueError(
                                f"seed {seed} epoch {epoch} {doc_id} {profile_name}: "
                                f"sampled vector hash missing from utility cache"
                            )
                        members.append(point)
                    p_group = sum(p for p, _ in members) / len(members)
                    u_group = sum(u for _, u in members) / len(members)
                    groups.append({
                        "epoch": epoch, "cycle": epoch // 4, "doc_id": doc_id,
                        "profile": profile_name, "lambda": lam,
                        "P": p_group, "utility": u_group,
                    })
                    if lam > 0 and epoch // 4 >= 1:
                        frontier = [
                            u for p, u in pool[doc_id] if p >= p_group - 1e-9
                        ]
                        regrets.append(max(frontier) - u_group)
        median_regret = statistics.median(regrets)
        verdicts[seed] = median_regret

        # Supporting preregistered items from the same groups (last cycle).
        last_cycle = max(g["cycle"] for g in groups)
        final = [g for g in groups if g["cycle"] == last_cycle]
        by_profile: dict[str, list[float]] = {}
        for g in final:
            by_profile.setdefault(g["profile"], []).append(g["P"])
        profile_means = {
            name: sum(v) / len(v)
            for name, v in sorted(
                by_profile.items(), key=lambda kv: lambda_by_profile[kv[0]],
            )
        }
        ordered = [
            profile_means[name]
            for name in sorted(profile_means, key=lambda_by_profile.get)
        ]
        max_profile = max(lambda_by_profile, key=lambda_by_profile.get)
        zero_profile = min(lambda_by_profile, key=lambda_by_profile.get)
        paired = [
            next(
                (g["P"] for g in final
                 if g["doc_id"] == doc and g["profile"] == max_profile), None,
            )
            for doc in DOC_IDS
        ]
        print(
            f"seed {seed}: median regret={median_regret:.4f} (n={len(regrets)}) "
            f"{'PASS' if median_regret <= REGRET_FLOOR else 'FAIL'} | "
            f"final-cycle P by profile: "
            + ", ".join(f"{k}={v:.3f}" for k, v in profile_means.items())
            + f" | monotone={all(b >= a - 1e-9 for a, b in zip(ordered, ordered[1:]))}"
            + f" | dP(max)={profile_means[max_profile] - profile_means[zero_profile]:+.3f}"
        )
        (RESULTS / f"frontier-eval-s{seed}.json").write_text(json.dumps({
            "seed": seed, "median_regret": median_regret,
            "regret_count": len(regrets), "regret_floor": REGRET_FLOOR,
            "passes": median_regret <= REGRET_FLOOR,
            "final_cycle_profile_P": profile_means,
            "pool_sizes": {doc: len(rows) for doc, rows in sorted(pool.items())},
            "groups": groups,
        }, indent=1))

    if verdicts:
        overall = all(v <= REGRET_FLOOR for v in verdicts.values())
        print(
            f"frontier-regret rule (median <= {REGRET_FLOOR} on all seeds): "
            f"{'PASS' if overall else 'FAIL'} — "
            + ", ".join(f"s{s}={v:.4f}" for s, v in sorted(verdicts.items()))
        )


if __name__ == "__main__":
    main()
