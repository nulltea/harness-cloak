#!/usr/bin/env python3
"""Phase 1 of the objective-normalization spike (decision log, 2026-07-28).

Measures per-family alpha gradients at a shared BC-initialized policy state over
documents stratified by decision count D, then fits the preregistered diagnostic
slope b_D per candidate normalization arm. No optimizer steps; one scoring pass
shared by every arm (arms differ only in loss recomposition, and their alpha-ratio
scalings follow algebraically from the measured per-family gradients).

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/objective_normalization_spike.py
"""
import json
import math
import sys
from pathlib import Path

import torch

from cloak.ranker.environment import LambdaProfile, load_ranker_environment
from cloak.ranker.interactive import (
    behavior_clone,
    expected_profile_count_loss,
    provisional_utility_loss,
    replay_trajectory,
    sample_trajectory,
    score_trajectories,
)
from cloak.ranker.privacy import DirectCountPrivacyProvider
from cloak.ranker.profile_count import ProfileCountTargets
from cloak.reward.utility_cache import UtilityCache
from cloak.reward.utility_credit import provisional_credit

sys.path.insert(0, "scripts")
from train_interactive_ranker import (
    _demote_out_of_scope_decisions,
    _drop_zero_signal_documents,
    _semantic_training_policy,
)

SEED = 17
ROLLOUTS = 3
BINS = ((4, 7), (8, 12), (13, 18), (19, 24))
OUT = Path("results/ranker_v2/architecture/objective-normalization-spike.json")


class _Args:
    representation_manifest = "results/ranker_v2/architecture/representation-full/manifest.json"
    privacy_checkpoint = None
    profile_count_targets = "results/ranker_v2/reward/profile-count-targets.json"
    device = "auto"


def _select_documents(documents):
    by_d = sorted((len(d.policy_decisions), d.doc_id) for d in documents)
    chosen = []
    for lo, hi in BINS:
        in_bin = [(D, doc_id) for D, doc_id in by_d if lo <= D <= hi]
        chosen.append(in_bin[0])
        chosen.append(in_bin[-1])
    return chosen


def _fit_slope(points):
    xs = [math.log(D) for D, _ in points]
    ys = [y for _, y in points]
    n = len(points)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx if sxx else float("nan")


def main() -> None:
    targets_payload = json.loads(Path(_Args.profile_count_targets).read_text())
    utility_artifact = json.loads(
        Path("results/ranker_v2/qa/aci-full.utility").read_text()
    )
    environment_hash = json.loads(
        Path("results/ranker_v2/environment/ranker-env.json").read_text()
    )["frozen_environment"]["environment_hash"]
    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(targets_payload)
    )
    documents, _ = _drop_zero_signal_documents(documents, utility_artifact)
    chosen = _select_documents(documents)
    print("documents:", chosen, flush=True)
    doc_ids = [doc_id for _, doc_id in chosen]
    documents = tuple(d for d in documents if d.doc_id in doc_ids)
    documents_by_id = {d.doc_id: d for d in documents}

    menu = json.loads(Path("results/ranker_v2/preflight/lambda-menu.json").read_text())
    profiles = tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(menu["profile_names"], menu["values"], strict=True)
    )
    targets = ProfileCountTargets.from_artifact(targets_payload)

    torch.manual_seed(SEED)
    policy = _semantic_training_policy(_Args, documents, profiles)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    behavior_clone(
        policy, documents, lambda_zero=profiles[0], optimizer=optimizer, epochs=3,
    )
    policy.train()

    cache = UtilityCache("results/ranker_v2/cache/utility-results.jsonl")
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for document in documents:
        D = len(document.policy_decisions)
        for profile in profiles:
            if float(profile.value) == 0.0:
                continue  # controller absent at lambda zero
            trajectories = tuple(
                sample_trajectory(
                    policy, document, profile, greedy=False, generator=generator,
                )
                for _ in range(ROLLOUTS)
            )
            points = score_trajectories(
                (document,), trajectories,
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
                count_reward=targets, cache=cache,
                remote_workers=6, reader_workers=6,
            )
            credit = provisional_credit(
                tuple(point.component_scores for point in points),
                utility_artifact, document.doc_id,
            )
            replayed = tuple(
                replay_trajectory(policy, document, trajectory, profile)
                for trajectory in trajectories
            )
            steps = tuple(s for t in replayed for s in t.steps)
            terms = {
                "utility": provisional_utility_loss(replayed, credit),
                "count": expected_profile_count_loss(
                    steps, targets, lambda_value=float(profile.value),
                    decision_count=D, rollout_count=ROLLOUTS,
                ),
                "entropy": -0.01 * torch.stack(
                    [s.entropy for s in steps]
                ).sum() / ROLLOUTS,
            }
            grads = {}
            for name, loss in terms.items():
                policy.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                grads[name] = float(policy.alpha_raw.grad or 0.0)
            rows.append({
                "doc_id": document.doc_id, "D": D,
                "profile": profile.name, "lambda": float(profile.value),
                "alpha_grad": grads,
            })
            print(f"  {document.doc_id} D={D} {profile.name}: "
                  f"gU={grads['utility']:+.3e} gH={grads['entropy']:+.3e} "
                  f"gC={grads['count']:+.3e}", flush=True)

    eps = 1e-12
    base_points = [
        (row["D"], math.log(abs(row["alpha_grad"]["utility"])
                            + abs(row["alpha_grad"]["entropy"]) + eps)
                   - math.log(abs(row["alpha_grad"]["count"]) + eps))
        for row in rows
    ]
    b_current = _fit_slope(base_points)
    report = {
        "seed": SEED, "rollouts": ROLLOUTS, "documents": chosen,
        "rows": rows,
        "b_D": {
            "current_mix": b_current,
            "average_all": b_current - 1.0,
            "sum_all": b_current - 1.0,
            "alpha_routing": b_current - 1.0,
        },
        "note": "fix-arm slopes follow algebraically: each divides the alpha "
                "utility:count ratio by D, shifting the log-log slope by -1",
    }
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nb_D current mix = {b_current:.3f}; all fix arms = {b_current - 1:.3f}"
          f" -> {OUT}", flush=True)


# ---- Phase 2 (revised protocol, approved 2026-07-28): b_D from training dynamics ----
PHASE2_DOCS = ("aci/D2N005", "aci/D2N027", "aci/D2N063", "aci/D2N031")  # D = 4/12/18/22
PHASE2_EPOCHS = 8
PHASE2_ROLLOUTS = 8


def phase2(arm: str) -> None:
    """Train one arm for 8 epochs over 4 length-binned docs, recording per-group
    per-family alpha gradients before each optimizer step (KL eta=0, counterfactual
    budget 0 — both were negligible alpha contributors in the v7 smoke)."""
    from cloak.ranker.interactive import build_latin_cycle_schedule

    targets_payload = json.loads(Path(_Args.profile_count_targets).read_text())
    utility_artifact = json.loads(Path("results/ranker_v2/qa/aci-full.utility").read_text())
    environment_hash = json.loads(Path("results/ranker_v2/environment/ranker-env.json").read_text())[
        "frozen_environment"]["environment_hash"]
    documents = tuple(load_ranker_environment(Path("results/ranker_v2/environment/ranker-env.json")).values())
    documents, _ = _demote_out_of_scope_decisions(documents, DirectCountPrivacyProvider(targets_payload))
    documents = tuple(d for d in documents if d.doc_id in PHASE2_DOCS)
    menu = json.loads(Path("results/ranker_v2/preflight/lambda-menu.json").read_text())
    profiles = tuple(LambdaProfile(n, float(v)) for n, v in zip(menu["profile_names"], menu["values"], strict=True))
    targets = ProfileCountTargets.from_artifact(targets_payload)

    torch.manual_seed(SEED)
    policy = _semantic_training_policy(_Args, documents, profiles)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    behavior_clone(policy, documents, lambda_zero=profiles[0], optimizer=optimizer, epochs=3)
    if arm == "routing":
        policy.alpha_utility_routing = "per-decision"
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    schedule = build_latin_cycle_schedule(documents, profiles, seed=SEED)
    cache = UtilityCache("results/ranker_v2/cache/utility-results.jsonl")
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for epoch in range(PHASE2_EPOCHS):
        for document in documents:
            profile = schedule.profile_for(document.doc_id, epoch)
            D = len(document.policy_decisions)
            trajectories = tuple(
                sample_trajectory(policy, document, profile, greedy=False, generator=generator)
                for _ in range(PHASE2_ROLLOUTS)
            )
            points = score_trajectories(
                (document,), trajectories, utility_artifact=utility_artifact,
                environment_hash=environment_hash, count_reward=targets, cache=cache,
                remote_workers=6, reader_workers=6,
            )
            credit = provisional_credit(
                tuple(p.component_scores for p in points), utility_artifact, document.doc_id,
            )
            replayed = tuple(replay_trajectory(policy, document, t, profile) for t in trajectories)
            steps = tuple(s for t in replayed for s in t.steps)
            lam = float(profile.value)
            terms = {
                "utility": provisional_utility_loss(replayed, credit),
                "count": expected_profile_count_loss(
                    steps, targets, lambda_value=lam, decision_count=D,
                    rollout_count=PHASE2_ROLLOUTS,
                ) if lam > 0 else None,
                "entropy": -0.01 * torch.stack([s.entropy for s in steps]).sum() / PHASE2_ROLLOUTS,
            }
            grads = {}
            for name, loss in terms.items():
                if loss is None:
                    grads[name] = 0.0
                    continue
                policy.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                grads[name] = float(policy.alpha_raw.grad or 0.0)
            policy.zero_grad(set_to_none=True)
            total = terms["utility"] + terms["entropy"] + (terms["count"] if lam > 0 else 0.0)
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            selected_privacy = float(sum(p.count_score for p in points) / len(points))
            utility = float(sum(p.utility for p in points) / len(points))
            entropy_mean = float(torch.stack([s.entropy for s in steps]).mean())
            rows.append({
                "epoch": epoch, "doc_id": document.doc_id, "D": D,
                "profile": profile.name, "lambda": lam, "alpha_grad": grads,
                "alpha": float(torch.nn.functional.softplus(policy.alpha_raw)),
                "selected_privacy": selected_privacy, "utility": utility,
                "entropy": entropy_mean,
                "finite": bool(torch.isfinite(total)),
            })
            print(f"  ep{epoch} {document.doc_id} D={D} {profile.name}: "
                  f"gU={grads['utility']:+.2e} gC={grads['count']:+.2e} "
                  f"u={utility:.3f} priv={selected_privacy:.3f}", flush=True)
    out = Path(f"results/ranker_v2/architecture/objective-normalization-phase2-{arm}.json")
    out.write_text(json.dumps({"arm": arm, "seed": SEED, "rollouts": PHASE2_ROLLOUTS, "rows": rows}, indent=1))
    print(f"arm {arm} -> {out}", flush=True)





# ---- Phase 3: controller-strength arms (preregistered fork, 2026-07-28) ----
PHASE3_EPOCHS = int(__import__("os").environ.get("PHASE3_EPOCHS", "12"))


def _switch_thresholds(policy, documents, profiles):
    """Weighted per-doc switch thresholds from BC menus (raw and gap-normalized)."""
    raw_pairs, norm_pairs = [], []
    for document in documents:
        state = policy.begin_document(document, profiles[-1])
        per_raw, per_norm = [], []
        for decision in document.policy_decisions:
            menu = tuple(a.action_id for a in decision.actions)
            with torch.no_grad():
                row = policy.distribution(state, decision, menu, profiles[-1])
            u, pv = row.utility_logits, row.predicted_privacy
            star = int(torch.argmax(u))
            gaps = [
                (float(u[star] - u[j]), float(pv[j] - pv[star]))
                for j in range(len(menu)) if float(pv[j]) > float(pv[star])
            ]
            if not gaps:
                continue
            t_raw = min(max(du, 0.0) / dp for du, dp in gaps)
            per_raw.append(t_raw)
            span = float(u.max() - u.min())
            if span > 0:
                per_norm.append(t_raw / span)
        for values, out in ((per_raw, raw_pairs), (per_norm, norm_pairs)):
            if values:
                weight = 1.0 / len(values)
                out.extend((weight, value) for value in values)

    def weighted_median(pairs):
        pairs = sorted(pairs, key=lambda x: x[1])
        total = sum(w for w, _ in pairs)
        cum = 0.0
        for w, value in pairs:
            cum += w
            if cum >= total / 2:
                return value
        return pairs[-1][1] if pairs else float("nan")

    return weighted_median(raw_pairs), weighted_median(norm_pairs)


def _set_alpha(policy, target: float) -> None:
    with torch.no_grad():
        policy.alpha_raw.fill_(math.log(math.expm1(max(target, 1e-4))))


def phase3(arm: str, seed: int) -> None:
    targets_payload = json.loads(Path(_Args.profile_count_targets).read_text())
    utility_artifact = json.loads(Path("results/ranker_v2/qa/aci-full.utility").read_text())
    environment_hash = json.loads(Path("results/ranker_v2/environment/ranker-env.json").read_text())[
        "frozen_environment"]["environment_hash"]
    documents = tuple(load_ranker_environment(Path("results/ranker_v2/environment/ranker-env.json")).values())
    documents, _ = _demote_out_of_scope_decisions(documents, DirectCountPrivacyProvider(targets_payload))
    documents = tuple(d for d in documents if d.doc_id in PHASE2_DOCS)
    menu = json.loads(Path("results/ranker_v2/preflight/lambda-menu.json").read_text())
    profiles = tuple(LambdaProfile(n, float(v)) for n, v in zip(menu["profile_names"], menu["values"], strict=True))
    targets = ProfileCountTargets.from_artifact(targets_payload)
    from cloak.ranker.interactive import build_latin_cycle_schedule

    torch.manual_seed(seed)
    policy = _semantic_training_policy(_Args, documents, profiles)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    behavior_clone(policy, documents, lambda_zero=profiles[0], optimizer=optimizer, epochs=3)
    t_raw, t_norm = _switch_thresholds(policy, documents, profiles)
    print(f"switch thresholds: raw median {t_raw:.3f} | gap-normalized {t_norm:.3f}", flush=True)
    if arm == "init-only":
        _set_alpha(policy, t_raw)
    elif arm == "gap-scaled":
        policy.controller_gap_scaling = "utility-gap"
        _set_alpha(policy, t_norm)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    schedule = build_latin_cycle_schedule(documents, profiles, seed=seed)
    cache = UtilityCache("results/ranker_v2/cache/utility-results.jsonl")
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for epoch in range(PHASE3_EPOCHS):
        for document in documents:
            profile = schedule.profile_for(document.doc_id, epoch)
            D = len(document.policy_decisions)
            lam = float(profile.value)
            trajectories = tuple(
                sample_trajectory(policy, document, profile, greedy=False, generator=generator)
                for _ in range(PHASE2_ROLLOUTS)
            )
            points = score_trajectories(
                (document,), trajectories, utility_artifact=utility_artifact,
                environment_hash=environment_hash, count_reward=targets, cache=cache,
                remote_workers=6, reader_workers=6,
            )
            credit = provisional_credit(
                tuple(p.component_scores for p in points), utility_artifact, document.doc_id,
            )
            replayed = tuple(replay_trajectory(policy, document, t, profile) for t in trajectories)
            steps = tuple(s for t in replayed for s in t.steps)
            total = provisional_utility_loss(replayed, credit) \
                - 0.01 * torch.stack([s.entropy for s in steps]).sum() / PHASE2_ROLLOUTS
            if lam > 0:
                total = total + expected_profile_count_loss(
                    steps, targets, lambda_value=lam, decision_count=D,
                    rollout_count=PHASE2_ROLLOUTS,
                )
            policy.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            modes = {"level": 0, "keep": 0, "placeholder": 0}
            for trajectory in trajectories:
                for decision in document.policy_decisions:
                    action_id = trajectory.action_vector[decision.decision_id]
                    action = next(a for a in decision.actions if a.action_id == action_id)
                    modes[action.mode] += 1
            total_modes = sum(modes.values())
            rows.append({
                "epoch": epoch, "cycle": epoch // 4, "doc_id": document.doc_id, "D": D,
                "profile": profile.name, "lambda": lam,
                "P": float(sum(p.count_score for p in points) / len(points)),
                "utility": float(sum(p.utility for p in points) / len(points)),
                "alpha": float(torch.nn.functional.softplus(policy.alpha_raw)),
                "mode_rates": {k: v / total_modes for k, v in modes.items()},
                "unique_vectors": len({json.dumps(dict(t.action_vector), sort_keys=True) for t in trajectories}),
                "finite": bool(torch.isfinite(total)),
            })
            print(f"  ep{epoch} {document.doc_id} {profile.name}: P={rows[-1]['P']:.3f} "
                  f"u={rows[-1]['utility']:.3f} a={rows[-1]['alpha']:.3f} "
                  f"ph={rows[-1]['mode_rates']['placeholder']:.2f}", flush=True)
    out = Path(f"results/ranker_v2/architecture/controller-strength-{arm}-s{seed}.json")
    out.write_text(json.dumps({
        "arm": arm, "seed": seed,
        "thresholds": {"raw_median": t_raw, "gap_normalized_median": t_norm},
        "rows": rows,
    }, indent=1))
    print(f"arm {arm} seed {seed} -> {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "phase2":
        phase2(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "phase3":
        phase3(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else SEED)
    else:
        main()
