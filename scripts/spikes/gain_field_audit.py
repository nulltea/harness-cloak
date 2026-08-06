"""Path-faithful audit of the controller gain field (offline, CPU, no reward calls).

WHY: the epoch reporter logs `controller_alpha` with a spread of EXACTLY 0.00e+00
across all 56 decisions of all four documents, at every epoch. Two readings are
possible and they lead to opposite designs:

  * real model property -- the "per-decision" gain head emits one number, so the
    tie-margin hinge (which trains ONLY that residual) has never been able to
    assign decision-specific authority; and v14's preregistered "nonzero
    cross-decision gain spread" gate was violated, making its adoption verdict
    wrong rather than merely mis-worded.
  * measurement artifact -- the reporter creates `gain_state` once
    (interactive.py:1107) and never advances it, while training advances state
    after every decision (interactive.py:895). Then the field may differentiate
    fine and the uniformity is ours, not the model's.

Design adjudicated by Codex Sol High (round 10, 2026-08-04). Three independent
estimates of the same quantity must agree or the audit is void:

  stale      reproduce the reporter exactly (state never advanced)
  faithful   advance state after each selected action, as training does
  inferred   recover alpha from the policy's own logits, without touching the
             gain head at all:  alpha = median over action pairs of
             ((z_i - z_k) - (u_i - u_k)) / (p_i - p_k)   at lambda-max, g=1

`faithful` vs `inferred` disagreement means the audit is measuring the wrong
thing; report and stop rather than interpret.

Equality thresholds are NOT invented: the numerical floor comes from repeated
identical forward calls on the same input.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/gain_field_audit.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

BASE = Path("results/ranker_v2")
OUT = BASE / "architecture/count_to_gain"
CHECKPOINTS = {
    "count_to_gain/detached": OUT / "detached-s47.pt",
    "count_to_gain/coupled": OUT / "coupled-s47.pt",
    # v14: the TRAIN checkpoints carry a trained gain head; control-evidence-*.pt
    # are lambda-zero controls and have none (Sol High corrected this).
    "v14/evidence-online": BASE / "architecture/tie_ownership/train-evidence-online-s47.pt",
    "v14/evidence-cycle": BASE / "architecture/tie_ownership/train-evidence-cycle-s47.pt",
}
RUN_DOCUMENTS = ("aci/D2N005", "aci/D2N027", "aci/D2N031", "aci/D2N063")


def _environment():
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
    selected = tuple(d for d in documents if d.doc_id in RUN_DOCUMENTS)
    assert len(selected) == len(RUN_DOCUMENTS), sorted(d.doc_id for d in selected)

    menu = json.loads((BASE / "preflight/lambda-menu.json").read_text())
    profiles = tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(menu["profile_names"], menu["values"], strict=True)
    )
    return selected, profiles, count_state


def _policy(documents, profiles, count_state, state_dict):
    from train_interactive_ranker import _semantic_training_policy

    args = SimpleNamespace(
        representation_manifest=str(BASE / "architecture/representation-full/manifest.json"),
        profile_count_targets=str(BASE / "reward/profile-count-targets.json"),
        privacy_checkpoint=None,
        device="cpu",
    )
    policy = _semantic_training_policy(args, documents, profiles)
    from cloak.ranker.semantic import enable_controller_gain

    if any("gain_head" in key for key in state_dict):
        enable_controller_gain(policy, "evidence", hidden_dim=32, bound=1.5)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.float()
    return policy


def _blocks(vector: torch.Tensor, count: int = 6) -> list[torch.Tensor]:
    """The pooled input is a concatenation of equal-width feature blocks."""
    width = vector.shape[-1] // count
    return [vector[i * width:(i + 1) * width] for i in range(count)]


def _spread(rows: list[torch.Tensor]) -> dict:
    """Centered energy, effective rank, exact-duplicate count, pair distances."""
    matrix = torch.stack([r.flatten() for r in rows])
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    energy = float(centered.norm())
    if matrix.shape[0] > 1:
        singular = torch.linalg.svdvals(centered.double())
        total = float(singular.sum())
        if total > 0:
            p = (singular / singular.sum()).clamp_min(1e-30)
            effective_rank = float(torch.exp(-(p * p.log()).sum()))
        else:
            effective_rank = 0.0
        distances = torch.cdist(matrix.double(), matrix.double())
        triu = distances[torch.triu(torch.ones_like(distances), diagonal=1) > 0]
        pair = {
            "min": float(triu.min()), "median": float(triu.median()),
            "max": float(triu.max()),
        }
    else:
        effective_rank, pair = 0.0, {}
    unique = {tuple(r.flatten().tolist()) for r in rows}
    return {
        "n": len(rows),
        "centered_energy": energy,
        "effective_rank": effective_rank,
        "exact_duplicates": len(rows) - len(unique),
        "pair_distance": pair,
    }


def _numerical_floor(policy, sample_input: torch.Tensor, repeats: int = 12) -> float:
    """Largest deviation across identical forward calls: the real 'equal' floor."""
    with torch.no_grad():
        values = [float(policy.gain_head(sample_input).squeeze(-1)) for _ in range(repeats)]
    return max(values) - min(values)


def audit(label: str, path: Path, documents, profiles, count_state) -> dict:
    from cloak.ranker.interactive import replay_trajectory, sample_trajectory
    from cloak.ranker.semantic import decision_controller_alpha

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("policy_state_dict") or checkpoint.get("state_dict")
    if state_dict is None or not any("gain_head" in k for k in state_dict):
        return {"label": label, "skipped": "no trained gain_head in checkpoint"}
    policy = _policy(documents, profiles, count_state, state_dict)
    profile = profiles[-1]                      # lambda-max, g(lambda) = 1
    stages: dict[str, list[torch.Tensor]] = {
        "pooled_input": [], "projection": [], "activation": [],
        "residual": [], "alpha": [],
    }
    block_stage: dict[int, list[torch.Tensor]] = {i: [] for i in range(6)}
    rows: list[dict] = []
    saturated_units = total_units = 0
    linear1, linear2 = policy.gain_head[0], policy.gain_head[2]

    for document in documents:
        greedy = sample_trajectory(policy, document, profile, greedy=True, generator=None)
        replayed = replay_trajectory(policy, document, greedy, profile)
        stale_state = policy.begin_document(document, profile)   # never advanced
        faithful_state = policy.begin_document(document, profile)
        for decision, step in zip(document.policy_decisions, replayed.steps, strict=True):
            with torch.no_grad():
                stale = decision_controller_alpha(
                    policy, stale_state, decision, step.legal_action_ids,
                )
                faithful = decision_controller_alpha(
                    policy, faithful_state, decision, step.legal_action_ids,
                )
                pooled = _pooled_input(policy, faithful_state, decision)
                projection = linear1(pooled)
                activation = torch.tanh(projection)
                residual = linear2(activation).squeeze(-1)
                hidden_part = float(linear2.weight.flatten() @ activation)
                inferred = _inferred_alpha(policy, faithful_state, decision, step, profile)
            saturated_units += int((activation.abs() > 0.999).sum())
            total_units += activation.numel()
            stages["pooled_input"].append(pooled)
            stages["projection"].append(projection)
            stages["activation"].append(activation)
            stages["residual"].append(residual.reshape(1))
            stages["alpha"].append(torch.tensor([faithful]))
            for i, block in enumerate(_blocks(pooled)):
                block_stage[i].append(block)
            rows.append({
                "doc_id": document.doc_id,
                "decision_id": decision.decision_id,
                "alpha_stale": stale,
                "alpha_faithful": faithful,
                "alpha_inferred": inferred,
                "residual": float(residual),
                "residual_hidden_part": hidden_part,
                "residual_output_bias": float(linear2.bias),
            })
            faithful_state = policy.advance(
                faithful_state, decision, step.selected_action_id,
            )

    floor = _numerical_floor(policy, stages["pooled_input"][0])
    faithful = [r["alpha_faithful"] for r in rows]
    stale = [r["alpha_stale"] for r in rows]
    inferred = [r["alpha_inferred"] for r in rows if r["alpha_inferred"] is not None]
    agreement = [
        abs(r["alpha_faithful"] - r["alpha_inferred"])
        for r in rows if r["alpha_inferred"] is not None
    ]
    by_doc: dict[str, list[float]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r["alpha_faithful"])
    within = st.mean(
        st.pvariance(v) if len(v) > 1 else 0.0 for v in by_doc.values()
    )
    between = st.pvariance([st.mean(v) for v in by_doc.values()])
    return {
        "label": label,
        "decisions": len(rows),
        "numerical_floor": floor,
        "alpha_stale_spread": max(stale) - min(stale),
        "alpha_faithful_spread": max(faithful) - min(faithful),
        "alpha_inferred_spread": (max(inferred) - min(inferred)) if inferred else None,
        "faithful_vs_inferred_max_abs_diff": max(agreement) if agreement else None,
        "within_document_variance": within,
        "between_document_variance": between,
        "tanh_saturation_fraction": saturated_units / max(1, total_units),
        "residual_hidden_vs_bias": {
            "hidden_abs_mean": st.mean(abs(r["residual_hidden_part"]) for r in rows),
            "output_bias": rows[0]["residual_output_bias"],
        },
        "stages": {name: _spread(vals) for name, vals in stages.items()},
        "pooled_blocks": {str(i): _spread(v) for i, v in block_stage.items()},
        "rows": rows,
    }


def _pooled_input(policy, state, decision) -> torch.Tensor:
    """Exactly the vector the gain head sees (mirrors decision_controller_alpha)."""
    actions, pair_features, token_bank, features = policy._decision_inputs(state, decision)
    utility_relations = policy.utility_projection(pair_features)
    contexts = policy.context_readout(token_bank, features, utility_relations)
    histories = policy.memory(
        torch.cat([utility_relations, contexts], dim=-1), state.selected_records,
    )
    mode_ids, runtime_type_ids = policy._category_ids(actions, decision)
    interaction = policy.interaction_projection(
        utility_relations * policy.context_to_relation(contexts)
    )
    return torch.cat([
        utility_relations, contexts, interaction,
        policy.action_mode_embedding(mode_ids),
        policy.runtime_type_embedding(runtime_type_ids),
        histories,
    ], dim=-1).mean(dim=0)


def _inferred_alpha(policy, state, decision, step, profile) -> float | None:
    """Recover alpha from the policy's own logits, bypassing the gain head.

    z = u + alpha * g * p  with g = 1 at lambda-max and no gap scaling, so for any
    action pair with distinct privacy scores alpha = ((z_i-z_k)-(u_i-u_k))/(p_i-p_k).
    """
    row = policy.distribution(state, decision, step.legal_action_ids, profile)
    z = row.combined_logits.detach()
    u = row.utility_logits.detach()
    p = row.predicted_privacy.detach()
    estimates = []
    for i in range(len(z)):
        for k in range(i + 1, len(z)):
            gap = float(p[i] - p[k])
            if abs(gap) < 1e-6:
                continue
            estimates.append((float((z[i] - z[k]) - (u[i] - u[k]))) / gap)
    return st.median(estimates) if estimates else None


def main() -> None:
    documents, profiles, count_state = _environment()
    reports = []
    for label, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"{label}: MISSING {path}")
            continue
        report = audit(label, path, documents, profiles, count_state)
        reports.append(report)
        if "skipped" in report:
            print(f"{label}: skipped — {report['skipped']}")
            continue
        print(f"\n=== {label} ({report['decisions']} decisions) ===")
        print(f"  numerical floor (identical forwards) {report['numerical_floor']:.3e}")
        print(f"  alpha spread  stale {report['alpha_stale_spread']:.3e}"
              f"  faithful {report['alpha_faithful_spread']:.3e}"
              f"  inferred {report['alpha_inferred_spread']}")
        print(f"  faithful vs inferred max |diff| {report['faithful_vs_inferred_max_abs_diff']}")
        print(f"  tanh saturation fraction {report['tanh_saturation_fraction']:.3f}")
        print(f"  variance within-doc {report['within_document_variance']:.3e}"
              f"  between-doc {report['between_document_variance']:.3e}")
        print(f"  residual: hidden |mean| {report['residual_hidden_vs_bias']['hidden_abs_mean']:.4f}"
              f"  output bias {report['residual_hidden_vs_bias']['output_bias']:.4f}")
        for name, s in report["stages"].items():
            print(f"    {name:13s} energy {s['centered_energy']:.3e}"
                  f"  eff.rank {s['effective_rank']:.2f}"
                  f"  exact dups {s['exact_duplicates']}/{s['n']}")
    destination = OUT / "gain-field-audit.json"
    destination.write_text(json.dumps(reports, indent=1, default=str))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
