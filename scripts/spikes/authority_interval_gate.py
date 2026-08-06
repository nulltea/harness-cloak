"""Phase 1 — authority-interval gate for the repaired gain head (CPU, cache-only).

Why not "train the head on the tie hinge plus the count objective": with the tower
and `alpha_raw` frozen, the hinge pushes the residual UP and the count objective
pushes it UP everywhere, so nothing bounds it above. A uniformly high residual
would win that metric while learning nothing about when authority should be
WITHHELD -- the global-dial failure again, in a better-conditioned head. Design
correction by Codex Sol High (round 12).

Instead: derive a two-sided interval per decision from frozen quantities.

  z_+ - z_-  =  du + alpha_j * g(lambda) * dp

  tied pair, privacy-preferred a_+ must win by the margin m:
      alpha_j >= (m - du) / (g * dp)                      LOWER bound
  costly live pair, where switching to the privacy-preferred action would cost
  more utility than the per-document budget allows, so it must NOT win:
      alpha_j <  -du / (g * dp)                           UPPER bound

  loss_j = relu(L_j - alpha_j) + relu(alpha_j - U_j)

Counts enter only through dp. There is no count-maximisation term.

Arms: `linear` (diagnostic baseline) vs `gelu16-prenorm` (production candidate),
3-fold split DISJOINT BY DOCUMENT, paired per-fold comparison.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/authority_interval_gate.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

BASE = Path("results/ranker_v2")
OUT = BASE / "architecture/count_to_gain"
TIE_MARGIN = 0.1          # --tie-margin, unchanged
UTILITY_BUDGET = 0.044    # the per-document lambda-zero utility bound, unchanged
FOLDS = 3
SEEDS = (17, 29, 47)
STEPS = 600


def _audit_namespace() -> dict:
    source = Path("scripts/spikes/gain_field_audit.py").read_text().split("def main()")[0]
    namespace: dict = {}
    exec(compile(source, "gain_field_audit", "exec"), namespace)
    return namespace


def _heads(input_dim: int) -> dict[str, callable]:
    def linear() -> nn.Module:
        head = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, 1, bias=False),
        )
        with torch.no_grad():
            head[-1].weight.zero_()
        return head

    def gelu16() -> nn.Module:
        head = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16, elementwise_affine=False),
            nn.GELU(),
            nn.Linear(16, 1, bias=False),
        )
        with torch.no_grad():
            head[-1].weight.zero_()
        return head

    return {"linear": linear, "gelu16-prenorm": gelu16}


def _collect(namespace: dict) -> dict:
    """Frozen features and per-decision authority intervals, over every document."""
    from cloak.ranker.environment import load_ranker_environment
    from cloak.ranker.interactive import (
        bootstrap_tie_evidence_from_cache,
        replay_trajectory,
        sample_trajectory,
    )
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from cloak.ranker.profile_count import ProfileCountTargets
    from cloak.reward.utility_credit import _partitions
    from train_interactive_ranker import (
        _demote_out_of_scope_decisions,
        _drop_zero_signal_documents,
    )

    count_state = json.loads((BASE / "reward/profile-count-targets.json").read_text())
    artifact = json.loads((BASE / "qa/aci-full.utility").read_text())
    targets = ProfileCountTargets.from_artifact(count_state)
    documents = tuple(load_ranker_environment(BASE / "environment/ranker-env.json").values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(count_state)
    )
    documents, _ = _drop_zero_signal_documents(documents, artifact)

    _, profiles, _ = namespace["_environment"]()
    checkpoint = torch.load(OUT / "coupled-s47.pt", map_location="cpu", weights_only=False)
    policy = namespace["_policy"](
        documents[:4], profiles, count_state, checkpoint["policy_state_dict"],
    )
    alpha_raw = float(checkpoint["policy_state_dict"]["alpha_raw"])
    profile = profiles[-1]
    magnitude = 1.0                      # g(lambda) at lambda-max, no gap scaling

    # measured evidence: exact ties and costly live pairs, from the utility cache
    ledger = bootstrap_tie_evidence_from_cache(
        BASE / "cache/utility-results.jsonl",
        {d.doc_id: d for d in documents},
        artifact,
    )
    measured: dict[tuple, list[float]] = {}
    for (doc_id, decision_id, a, b), records in ledger.items():
        measured.setdefault((doc_id, decision_id, a, b), []).extend(
            float(r["delta_u"]) for r in records
        )

    rows: list[dict] = []
    skipped_documents = []
    for document in documents:
        try:
            greedy = sample_trajectory(policy, document, profile, greedy=True, generator=None)
            replayed = replay_trajectory(policy, document, greedy, profile)
        except Exception as error:                    # representation gaps, etc.
            skipped_documents.append((document.doc_id, type(error).__name__))
            continue
        partitions = _partitions(artifact, document.doc_id)
        state = policy.begin_document(document, profile)
        for decision, step in zip(document.policy_decisions, replayed.steps, strict=True):
            with torch.no_grad():
                pooled = namespace["_pooled_input"](policy, state, decision)
                distribution = policy.distribution(
                    state, decision, step.legal_action_ids, profile,
                )
            utility = distribution.utility_logits.detach()
            menu = list(step.legal_action_ids)
            scores = {
                action_id: float(
                    targets.action_scores(decision.decision_id, (action_id,))[0]
                )
                for action_id in menu
            }
            structural = not partitions.linked_by_decision.get(decision.decision_id)
            lower, upper, n_tie, n_costly = [], [], 0, 0
            for i, a in enumerate(menu):
                for k, b in enumerate(menu):
                    if i >= k:
                        continue
                    dp = scores[a] - scores[b]
                    if abs(dp) < 1e-9:
                        continue                       # no privacy preference to enforce
                    # orient so the privacy-preferred action is `plus`
                    plus, minus = (i, k) if dp > 0 else (k, i)
                    dp = abs(dp)
                    du = float(utility[plus] - utility[minus])
                    key = (document.doc_id, decision.decision_id,
                           *sorted((menu[plus], menu[minus])))
                    deltas = measured.get(key)
                    tied = structural or (
                        deltas is not None and max(abs(d) for d in deltas) <= 1e-9
                    )
                    if tied:
                        lower.append((TIE_MARGIN - du) / (magnitude * dp))
                        n_tie += 1
                    elif deltas is not None and min(deltas) < -UTILITY_BUDGET:
                        # switching to the privacy-preferred action costs more utility
                        # than the budget allows: it must not win
                        upper.append(-du / (magnitude * dp))
                        n_costly += 1
            if lower or upper:
                rows.append({
                    "doc_id": document.doc_id,
                    "decision_id": decision.decision_id,
                    "features": pooled,
                    "lower": max(lower) if lower else None,   # conjunction of bounds
                    "upper": min(upper) if upper else None,
                    "tie_pairs": n_tie,
                    "costly_pairs": n_costly,
                    "structural": structural,
                })
            state = policy.advance(state, decision, step.selected_action_id)
    return {
        "rows": rows, "alpha_raw": alpha_raw, "magnitude": magnitude,
        "skipped_documents": skipped_documents,
    }


def _alpha(head: nn.Module, features: torch.Tensor, alpha_raw: float) -> torch.Tensor:
    return torch.nn.functional.softplus(alpha_raw + head(features).squeeze(-1))


def _metrics(alpha: torch.Tensor, lower: list, upper: list) -> dict:
    satisfied = total = violations = costly = 0
    for value, low, high in zip(alpha.tolist(), lower, upper, strict=True):
        if low is not None:
            total += 1
            satisfied += int(value >= low)
        if high is not None:
            costly += 1
            violations += int(value >= high)
    return {
        "tie_satisfaction": satisfied / total if total else None,
        "tie_constraints": total,
        "costly_violation_rate": violations / costly if costly else None,
        "costly_constraints": costly,
        "alpha_spread": float(alpha.max() - alpha.min()),
        "alpha_std": float(alpha.std()) if alpha.numel() > 1 else 0.0,
    }


def main() -> None:
    namespace = _audit_namespace()
    data = _collect(namespace)
    rows = data["rows"]
    if data["skipped_documents"]:
        print(f"skipped {len(data['skipped_documents'])} documents "
              f"(first few: {data['skipped_documents'][:3]})")
    documents = sorted({r["doc_id"] for r in rows})
    n_lower = sum(r["lower"] is not None for r in rows)
    n_upper = sum(r["upper"] is not None for r in rows)
    print(f"decisions with constraints: {len(rows)}  documents: {len(documents)}")
    print(f"  lower bounds (ties): {n_lower}   upper bounds (costly live): {n_upper}")
    print(f"  documents with an upper bound: "
          f"{len({r['doc_id'] for r in rows if r['upper'] is not None})}")
    if n_upper == 0:
        print("  WARNING: no upper bounds. This gate can establish lower-bound "
              "authority only and CANNOT safely select a production head.")

    features = torch.stack([r["features"] for r in rows])
    folds = {d: i % FOLDS for i, d in enumerate(documents)}
    reports = []
    for name, build in _heads(features.shape[-1]).items():
        per_fold = []
        for fold in range(FOLDS):
            train = [i for i, r in enumerate(rows) if folds[r["doc_id"]] != fold]
            test = [i for i, r in enumerate(rows) if folds[r["doc_id"]] == fold]
            if not train or not test:
                continue
            seeds = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                head = build()
                optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
                x_train = features[train]
                low_train = [rows[i]["lower"] for i in train]
                up_train = [rows[i]["upper"] for i in train]
                for _ in range(STEPS):
                    optimizer.zero_grad(set_to_none=True)
                    alpha = _alpha(head, x_train, data["alpha_raw"])
                    terms = []
                    for value, low, high in zip(alpha, low_train, up_train, strict=True):
                        if low is not None:
                            terms.append(torch.relu(torch.as_tensor(low) - value))
                        if high is not None:
                            terms.append(torch.relu(value - torch.as_tensor(high)))
                    if not terms:
                        break
                    torch.stack(terms).mean().backward()
                    optimizer.step()
                with torch.no_grad():
                    alpha_test = _alpha(head, features[test], data["alpha_raw"])
                seeds.append(_metrics(
                    alpha_test,
                    [rows[i]["lower"] for i in test],
                    [rows[i]["upper"] for i in test],
                ))
            per_fold.append({"fold": fold, "held_out_documents": len(
                {rows[i]["doc_id"] for i in test}
            ), "seeds": seeds})
        satisfaction = [
            s["tie_satisfaction"] for f in per_fold for s in f["seeds"]
            if s["tie_satisfaction"] is not None
        ]
        spreads = [s["alpha_spread"] for f in per_fold for s in f["seeds"]]
        violations = [
            s["costly_violation_rate"] for f in per_fold for s in f["seeds"]
            if s["costly_violation_rate"] is not None
        ]
        report = {
            "head": name,
            "parameters": sum(p.numel() for p in build().parameters()),
            "held_out_tie_satisfaction_mean": st.mean(satisfaction) if satisfaction else None,
            "held_out_tie_satisfaction_min": min(satisfaction) if satisfaction else None,
            "held_out_costly_violation_mean": st.mean(violations) if violations else None,
            "held_out_alpha_spread_mean": st.mean(spreads) if spreads else None,
            "per_fold": per_fold,
        }
        reports.append(report)
        print(f"\n=== {name} ({report['parameters']:,} params) ===")
        print(f"  held-out tie satisfaction  mean "
              f"{report['held_out_tie_satisfaction_mean']:.3f}"
              f"  worst {report['held_out_tie_satisfaction_min']:.3f}")
        violation = report["held_out_costly_violation_mean"]
        print("  held-out costly violation  " + (
            "n/a (no upper bounds)" if violation is None else f"{violation:.3f}"
        ))
        print(f"  held-out alpha spread      {report['held_out_alpha_spread_mean']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "authority-interval-gate.json").write_text(json.dumps(reports, indent=1))
    print(f"\nwrote {OUT / 'authority-interval-gate.json'}")


if __name__ == "__main__":
    main()
