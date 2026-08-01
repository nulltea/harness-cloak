"""Equivalence-critic screening (RL-ranker v15, preregistered).

Offline supervised screening of the utility-equivalence critic on cache-measured
counterfactual evidence; the policy is never retrained. Preregistration:
research-wiki/experiments/2026-07-31-RL-ranker-v15-equivalence-critic.md;
design rationale: docs/specs/RL/ties-by-design.md (fork section).

Subcommands:
  mine  — corpus-wide single-decision pair mining from the utility cache with
          surrounding-context retention, document-disjoint split assignment,
          and the step-0 feasibility count (kill gate: certification must
          plausibly yield >= 45 accepted pairs).
  run   — feature extraction from the frozen policy checkpoint, the three
          preregistered arms (hurdle eps=0.044, hurdle eps=0, Balanced-MSE
          single head), calibration-frozen thresholds, and Clopper-Pearson
          certification (document-level primary, pair-level companion).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

READER_FLOOR = 0.044
EXACT_ATOL = 1e-9
# The four tie-ownership screening documents carry the RL-run ledger and the
# richest cache coverage; they are pinned to train so calibration/certification
# stay document-disjoint from everything the v14 chain ever touched.
TRAIN_DOCS = frozenset(
    {"aci/D2N005", "aci/D2N027", "aci/D2N063", "aci/D2N031"}
)
# Finite-sample-corrected empirical bar (spec: certified 5% risk at 90%
# confidence needs ~97-98% empirical precision at n in the low hundreds).
PRECISION_TARGET = 0.98
RISK = 0.05
CONFIDENCE = 0.90
MIN_CERT_ACCEPTED = 45
OUT = Path("results/ranker_v2/architecture/equivalence_critic")


def _split_for(doc_id: str) -> str:
    """Preregistered document-disjoint split: pinned train docs, then a stable
    2/5 : 3/5 calibration/certification assignment by doc-id hash (certification
    gets the larger share because the Clopper-Pearson bound needs accepted n)."""
    if doc_id in TRAIN_DOCS:
        return "train"
    bucket = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 5
    return "calibration" if bucket < 2 else "certification"


def _stratum(delta_u: float) -> str:
    if abs(delta_u) <= EXACT_ATOL:
        return "exact"
    if abs(delta_u) <= READER_FLOOR:
        return "subnoise"
    return "live"


def _load_documents():
    from cloak.ranker.environment import load_ranker_environment
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from train_interactive_ranker import _demote_out_of_scope_decisions

    targets_payload = json.loads(
        Path("results/ranker_v2/reward/profile-count-targets.json").read_text()
    )
    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(targets_payload)
    )
    return {document.doc_id: document for document in documents}


def mine(args) -> None:
    """Mine single-decision delta-U evidence under BOTH utility statistics.

    The document-level statistic (what every run through v15 used) differences
    the whole weighted assertion mass, so assertions the artifact declares
    CANNOT depend on the flipped decision contribute pure measurement noise.
    The linked-restricted statistic differences only that decision's linked
    assertions. Decisions with NO linked assertions are structurally derivable
    ties (no feedback channel can separate their actions — Bartok duplicate
    actions); they are labelled, not measured. Round-4 zoom-out, 2026-08-01.
    """
    from cloak.reward.utility_credit import _partitions

    documents = _load_documents()
    artifact = json.loads(Path(args.utility_artifact).read_text())
    vectors: dict[str, dict[tuple, Mapping]] = {}
    for line in Path(args.utility_cache).read_text().splitlines():
        try:
            row = json.loads(line)["result"]
        except (ValueError, KeyError):
            continue
        document = documents.get(row.get("doc_id"))
        if document is None:
            continue
        ids = [decision.decision_id for decision in document.policy_decisions]
        vector = row.get("action_vector", {})
        scores = row.get("component_scores")
        if not ids or set(vector) != set(ids) or not isinstance(scores, Mapping):
            continue
        vectors.setdefault(document.doc_id, {})[
            tuple(vector[i] for i in ids)
        ] = scores

    rows = []
    for doc_id, cached in vectors.items():
        ids = [
            decision.decision_id
            for decision in documents[doc_id].policy_decisions
        ]
        partitions = _partitions(artifact, doc_id)
        weights = partitions.weights
        denominator = partitions.denominator
        index: dict[tuple, list[tuple]] = {}
        for key in cached:
            for position in range(len(ids)):
                rest = key[:position] + ("*",) + key[position + 1:]
                index.setdefault((position, rest), []).append(key)
        for (position, rest), keys in index.items():
            if len(keys) < 2:
                continue
            decision_id = ids[position]
            linked = partitions.linked_by_decision.get(decision_id, frozenset())
            context = {
                ids[i]: rest[i] for i in range(len(ids)) if i != position
            }
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    first, second = sorted(
                        (keys[i][position], keys[j][position])
                    )
                    by_action = {
                        keys[i][position]: cached[keys[i]],
                        keys[j][position]: cached[keys[j]],
                    }
                    a_scores, b_scores = by_action[first], by_action[second]
                    delta_document = sum(
                        weights[x] * (a_scores[x] - b_scores[x])
                        for x in weights if x in a_scores and x in b_scores
                    ) / denominator
                    delta_linked = sum(
                        weights[x] * (a_scores[x] - b_scores[x])
                        for x in linked if x in a_scores and x in b_scores
                    ) / denominator
                    rows.append({
                        "doc_id": doc_id,
                        "decision_id": decision_id,
                        "action_a": first,
                        "action_b": second,
                        "delta_u": delta_linked,
                        "delta_u_document": delta_document,
                        "linked_assertions": len(linked),
                        "derivable_tie": not linked,
                        "context": context,
                        "split": _split_for(doc_id),
                        "stratum": "derivable" if not linked else _stratum(delta_linked),
                        "stratum_document": _stratum(delta_document),
                    })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence-rows-linked.json").write_text(json.dumps({
        "reader_floor": READER_FLOOR,
        "exact_atol": EXACT_ATOL,
        "precision_target": PRECISION_TARGET,
        "statistic": "linked-restricted (delta_u); document-level retained as delta_u_document",
        "rows": rows,
    }))

    order = ("derivable", "exact", "subnoise", "live")
    for label, key in (
        ("DOCUMENT-level (statistic used through v15)", "stratum_document"),
        ("LINKED-restricted (corrected statistic)", "stratum"),
    ):
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        docs: dict[str, set] = defaultdict(set)
        for row in rows:
            counts[row["split"]][row[key]] += 1
            docs[row["split"]].add(row["doc_id"])
        print(f"\n{label}: {len(rows)} rows")
        for split in ("train", "calibration", "certification"):
            strata = counts[split]
            ties = sum(strata[name] for name in order if name != "live")
            print(
                f"  {split:13s}: docs {len(docs[split]):3d} | "
                + " | ".join(f"{name} {strata[name]:5d}" for name in order)
                + f" | tie-band {ties:5d}"
            )
    derivable = [row for row in rows if row["derivable_tie"]]
    contradicted = sum(
        1 for row in derivable if abs(row["delta_u_document"]) > EXACT_ATOL
    )
    print(
        f"\nstructurally derivable ties: {len(derivable)} rows across "
        f"{len({row['doc_id'] for row in derivable})} documents "
        f"({len({row['split'] for row in derivable})} splits) — free, "
        f"document-diverse labels needing no probe budget"
    )
    print(
        f"document-level statistic reports a NONZERO difference on "
        f"{contradicted}/{len(derivable)} of them ({100*contradicted/max(1,len(derivable)):.0f}%) "
        f"— pure measurement contamination on provably equivalent pairs"
    )
    cert = [row for row in rows if row["split"] == "certification"]
    cert_ties = sum(1 for row in cert if row["stratum"] != "live")
    verdict = "PASS" if cert_ties >= MIN_CERT_ACCEPTED else "KILL (undersized)"
    print(
        f"feasibility (linked statistic): certification tie-band rows "
        f"{cert_ties} vs minimum {MIN_CERT_ACCEPTED} -> {verdict}"
    )


def _clopper_pearson_upper(violations: int, n: int, confidence: float) -> float:
    from scipy.stats import beta

    if n <= 0:
        return 1.0
    if violations >= n:
        return 1.0
    return float(beta.ppf(confidence, violations + 1, n - violations))


def _auc(scores, labels) -> float:
    """Mann-Whitney AUC of score for label==1 (ties handled by midranks)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _extract_features(policy, documents, rows, profile):
    """Per-action critic inputs for every evidence row, grouped per document
    and surrounding context so each walk is priced once. The assembly mirrors
    the utility-head input contract (policy_tie_pooled_inputs without the
    pooling); this path never consults the static-stack inference cache, whose
    correctness is covered by the distribution-path unit tests instead."""
    import torch

    def action_features(state, decision):
        actions, pair_features, token_bank, features = policy._decision_inputs(
            state, decision
        )
        utility_relations = policy.utility_projection(pair_features)
        contexts = policy.context_readout(
            token_bank, features, utility_relations
        )
        histories = policy.memory(
            torch.cat([utility_relations, contexts], dim=-1),
            state.selected_records,
        )
        mode_ids, runtime_type_ids = policy._category_ids(actions, decision)
        interaction = policy.interaction_projection(
            utility_relations * policy.context_to_relation(contexts)
        )
        # Same assembly as the utility head input / policy_tie_pooled_inputs,
        # WITHOUT pooling: one feature row per action.
        matrix = torch.cat(
            [
                utility_relations, contexts, interaction,
                policy.action_mode_embedding(mode_ids),
                policy.runtime_type_embedding(runtime_type_ids),
                histories,
            ],
            dim=-1,
        ).detach()
        return {
            action.action_id: matrix[position]
            for position, action in enumerate(actions)
        }

    groups: dict[tuple, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        key = (
            row["doc_id"],
            row["decision_id"],
            tuple(sorted(row["context"].items())),
        )
        groups[key].append(row_index)

    feature_a = [None] * len(rows)
    feature_b = [None] * len(rows)
    with torch.no_grad():
        for (doc_id, decision_id, context_items), row_indices in groups.items():
            document = documents[doc_id]
            context = dict(context_items)
            state = policy.begin_document(document, profile)
            target = None
            for decision in document.policy_decisions:
                if decision.decision_id == decision_id:
                    target = decision
                    break
                state = policy.advance(
                    state, decision, context[decision.decision_id]
                )
            if target is None:
                raise ValueError(f"decision not found: {doc_id}/{decision_id}")
            by_action = action_features(state, target)
            for row_index in row_indices:
                row = rows[row_index]
                feature_a[row_index] = by_action[row["action_a"]].cpu()
                feature_b[row_index] = by_action[row["action_b"]].cpu()
    return torch.stack(feature_a), torch.stack(feature_b)


class HurdleCritic:
    """Linear probes on the frozen, detached policy features (preregistered
    architecture: no new trainable representation; small-n memorization
    mitigation from the imbalanced-regression research round)."""

    def __init__(self, feature_dim: int, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.q_head = nn.Linear(feature_dim, 1)
        self.gate_head = nn.Linear(2 * feature_dim, 1)

    def parameters(self):
        for module in (self.q_head, self.gate_head):
            yield from module.parameters()

    def forward(self, features_a, features_b):
        import torch

        delta_q = (self.q_head(features_a) - self.q_head(features_b)).squeeze(-1)
        gate_logit = self.gate_head(
            torch.cat(
                [features_a + features_b, (features_a - features_b).abs()],
                dim=-1,
            )
        ).squeeze(-1)
        return delta_q, gate_logit


def _train_hurdle(features_a, features_b, targets, epsilon, seed, epochs=300):
    import torch
    import torch.nn.functional as F

    critic = HurdleCritic(features_a.shape[-1], seed)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    tie_labels = (targets.abs() <= READER_FLOOR).float()
    for _ in range(epochs):
        delta_q, gate_logit = critic.forward(features_a, features_b)
        residual = delta_q - targets
        slack = torch.clamp(residual.abs() - epsilon, min=0.0)
        # ponytail: delta=1.0 keeps the quadratic zone off the noise scale
        magnitude = F.huber_loss(
            slack, torch.zeros_like(slack), delta=1.0
        )
        gate = F.binary_cross_entropy_with_logits(gate_logit, tie_labels)
        loss = gate + magnitude
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return critic


def _train_bmc(features_a, features_b, targets, seed, epochs=300, batch=64):
    """Balanced-MSE (BMC) single-head baseline; score is -|delta_q|."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(seed)
    q_head = nn.Linear(features_a.shape[-1], 1)
    log_sigma = torch.nn.Parameter(torch.tensor(math.log(0.1)))
    params = list(q_head.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(targets), batch):
            chosen = permutation[start:start + batch]
            if len(chosen) < 2:
                continue
            delta_q = (
                q_head(features_a[chosen]) - q_head(features_b[chosen])
            ).squeeze(-1)
            noise_var = (2.0 * log_sigma).exp()
            logits = -(delta_q.unsqueeze(1) - targets[chosen].unsqueeze(0)) ** 2 / (
                2.0 * noise_var
            )
            loss = F.cross_entropy(
                logits, torch.arange(len(chosen))
            ) * noise_var.detach()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def score(fa, fb):
        with torch.no_grad():
            delta_q = (q_head(fa) - q_head(fb)).squeeze(-1)
        return delta_q

    return score


def _calibrate_threshold(scores, labels):
    """Smallest score threshold whose calibration pair-level precision meets
    the preregistered bar; None when unreachable (arm fails calibration).

    Equal scores are grouped: `score >= t` accepts every member of t's score
    group, so precision is only evaluated at complete-group boundaries (codex
    review finding: ungrouped prefixes could freeze a threshold whose full
    accepted set misses the bar)."""
    by_score: dict[float, list[bool]] = defaultdict(list)
    for score, label in zip(scores, labels):
        by_score[score].append(label)
    accepted = violations = 0
    best = None
    for score in sorted(by_score, reverse=True):
        group = by_score[score]
        accepted += len(group)
        violations += sum(1 for label in group if not label)
        if (accepted - violations) / accepted >= PRECISION_TARGET:
            best = score
    return best


def _evaluate_arm(name, scores_by_split, delta_q_by_split, rows_by_split):
    calibration_scores, calibration_rows = (
        scores_by_split["calibration"], rows_by_split["calibration"],
    )
    threshold = _calibrate_threshold(
        [float(s) for s in calibration_scores],
        [row["stratum"] != "live" for row in calibration_rows],
    )
    cert_scores = [float(s) for s in scores_by_split["certification"]]
    cert_rows = rows_by_split["certification"]
    labels = [row["stratum"] != "live" for row in cert_rows]
    report = {
        "arm": name,
        "threshold": threshold,
        "cert_auc": _auc(cert_scores, labels),
        "cert_rows": len(cert_rows),
    }
    if threshold is None:
        report["calibration"] = "FAILED (precision target unreachable)"
        return report
    accepted = [s >= threshold for s in cert_scores]
    n_accepted = sum(accepted)
    violations = sum(
        1 for accept, label in zip(accepted, labels) if accept and not label
    )
    per_doc: dict[str, list] = defaultdict(list)
    for accept, label, row in zip(accepted, labels, cert_rows):
        if accept:
            per_doc[row["doc_id"]].append(not label)
    # Document-level certificate (the PRIMARY one — accepted pairs within a
    # document are dependent, so pair-level CP overstates its own precision;
    # conformal research round + codex review blocker): mean of per-document
    # violation RATES with a one-sided Hoeffding bound over document units,
    # plus the conservative any-violation Bernoulli CP as a companion.
    doc_rates = [sum(flags) / len(flags) for flags in per_doc.values()]
    doc_violation = [any(flags) for flags in per_doc.values()]
    n_docs = len(doc_rates)
    doc_rate_mean = sum(doc_rates) / n_docs if n_docs else None
    doc_rate_hoeffding = (
        doc_rate_mean + math.sqrt(math.log(1.0 / (1.0 - CONFIDENCE)) / (2 * n_docs))
        if n_docs else None
    )
    recall = {}
    for stratum in ("exact", "subnoise"):
        members = [
            accept for accept, row in zip(accepted, cert_rows)
            if row["stratum"] == stratum
        ]
        recall[stratum] = (
            sum(members) / len(members) if members else float("nan")
        )
    live_rows = [
        (float(dq), row["delta_u"])
        for dq, row in zip(delta_q_by_split["certification"], cert_rows)
        if row["stratum"] == "live"
    ]
    report.update({
        "accepted": n_accepted,
        "violations": violations,
        "empirical_precision": (
            (n_accepted - violations) / n_accepted if n_accepted else None
        ),
        "pair_cp_upper": _clopper_pearson_upper(
            violations, n_accepted, CONFIDENCE
        ),
        "doc_units": n_docs,
        "doc_violations": sum(doc_violation),
        "doc_rate_mean": doc_rate_mean,
        "doc_rate_hoeffding_upper": doc_rate_hoeffding,
        "doc_any_violation_cp_upper": _clopper_pearson_upper(
            sum(doc_violation), len(doc_violation), CONFIDENCE
        ),
        "recall_by_stratum": recall,
        "live_mae": (
            sum(abs(dq - du) for dq, du in live_rows) / len(live_rows)
            if live_rows else None
        ),
        "gates": {
            "certified_doc_risk": (
                n_accepted >= MIN_CERT_ACCEPTED
                and doc_rate_hoeffding is not None
                and doc_rate_hoeffding <= RISK
            ),
            "certified_pair_risk": (
                n_accepted >= MIN_CERT_ACCEPTED
                and _clopper_pearson_upper(violations, n_accepted, CONFIDENCE)
                <= RISK
            ),
            "min_accepted": n_accepted >= MIN_CERT_ACCEPTED,
        },
    })
    return report


def _load_policy(args, documents):
    """Rebuild the frozen adopted policy and its lambda profiles from disk."""
    import torch
    from types import SimpleNamespace

    from cloak.ranker.environment import LambdaProfile
    from train_interactive_ranker import (
        _apply_controller_options,
        _semantic_training_policy,
    )

    menu = json.loads(Path(args.lambda_menu).read_text())
    profile_objects = tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(menu["profile_names"], menu["values"], strict=True)
    )
    policy = _semantic_training_policy(
        SimpleNamespace(
            representation_manifest=args.representation_manifest,
            privacy_checkpoint=None,
            profile_count_targets=args.profile_count_targets,
            device=args.device,
        ),
        tuple(documents.values()),
        profile_objects,
    )
    _apply_controller_options(policy, SimpleNamespace(
        alpha_utility_routing="none",
        controller_gap_scaling="none",
        utility_logit_softcap=25.0,
        controller_gain="evidence",
        controller_gain_hidden=32,
        controller_gain_bound=1.5,
    ))
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    return policy, profile_objects


def run(args) -> None:
    import torch

    documents = _load_documents()
    payload = json.loads((OUT / "evidence-rows.json").read_text())
    rows = payload["rows"]
    policy, profile_objects = _load_policy(args, documents)

    features_a, features_b = _extract_features(
        policy, documents, rows, profile_objects[0],
    )
    targets = torch.tensor([row["delta_u"] for row in rows], dtype=torch.float32)

    split_indices = {
        split: [i for i, row in enumerate(rows) if row["split"] == split]
        for split in ("train", "calibration", "certification")
    }
    rows_by_split = {
        split: [rows[i] for i in indices]
        for split, indices in split_indices.items()
    }

    def subset(tensor, split):
        return tensor[torch.tensor(split_indices[split], dtype=torch.long)]

    reports = []
    for name, epsilon in (
        ("hurdle-eps044", READER_FLOOR), ("hurdle-eps0", 0.0),
    ):
        critic = _train_hurdle(
            subset(features_a, "train"), subset(features_b, "train"),
            subset(targets, "train"), epsilon, args.seed,
        )
        scores_by_split, delta_q_by_split = {}, {}
        with torch.no_grad():
            for split in ("calibration", "certification"):
                delta_q, gate_logit = critic.forward(
                    subset(features_a, split), subset(features_b, split),
                )
                scores_by_split[split] = torch.sigmoid(gate_logit).tolist()
                delta_q_by_split[split] = delta_q.tolist()
        report = _evaluate_arm(
            name, scores_by_split, delta_q_by_split, rows_by_split,
        )
        reports.append(report)
        torch.save({
            "q_head": critic.q_head.state_dict(),
            "gate_head": critic.gate_head.state_dict(),
            "feature_dim": features_a.shape[-1],
            "threshold": report.get("threshold"),
            "epsilon": epsilon,
            "seed": args.seed,
        }, OUT / f"critic-{name}.pt")

    bmc_score = _train_bmc(
        subset(features_a, "train"), subset(features_b, "train"),
        subset(targets, "train"), args.seed,
    )
    scores_by_split, delta_q_by_split = {}, {}
    for split in ("calibration", "certification"):
        delta_q = bmc_score(subset(features_a, split), subset(features_b, split))
        scores_by_split[split] = (-delta_q.abs()).tolist()
        delta_q_by_split[split] = delta_q.tolist()
    reports.append(_evaluate_arm(
        "bmc-single-head", scores_by_split, delta_q_by_split, rows_by_split,
    ))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screening-report.json").write_text(json.dumps({
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "precision_target": PRECISION_TARGET,
        "risk": RISK,
        "confidence": CONFIDENCE,
        "arms": reports,
    }, indent=1))
    for report in reports:
        print(json.dumps(report, indent=1))


def _actor_margin(policy, features_a, features_b):
    """The margin the controller actually races today: softcapped tower logits."""
    import torch

    with torch.no_grad():
        u_a = policy.utility_head(features_a.to(policy._device)).squeeze(-1)
        u_b = policy.utility_head(features_b.to(policy._device)).squeeze(-1)
        cap = getattr(policy, "utility_logit_softcap", None)
        if cap is not None:
            u_a = float(cap) * torch.tanh(u_a / float(cap))
            u_b = float(cap) * torch.tanh(u_b / float(cap))
    return (u_a - u_b).cpu()


def _noise_ceiling(rows) -> dict:
    """Irreducible target spread: the same pair measured in different contexts.

    No predictor can explain variance the target itself does not hold stable
    across surrounding action vectors, so this bounds the achievable fit and
    is the reference the gate's R^2 must be read against."""
    import statistics as st

    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["doc_id"], row["decision_id"], row["action_a"], row["action_b"],
        )].append(row["delta_u"])
    within, weights = [], []
    for values in groups.values():
        if len(values) < 2:
            continue
        within.append(st.pvariance(values))
        weights.append(len(values))
    targets = [row["delta_u"] for row in rows]
    total = st.pvariance(targets) if len(targets) > 1 else 0.0
    mean_within = (
        sum(v * w for v, w in zip(within, weights)) / sum(weights)
        if weights else 0.0
    )
    return {
        "pair_groups_with_repeats": len(within),
        "mean_within_group_variance": mean_within,
        "total_target_variance": total,
        "max_achievable_r2": (1.0 - mean_within / total) if total > 0 else None,
    }


def _score_margins(name, margins, rows) -> dict:
    """Stratified read-out of one margin source on one held-out split."""
    import statistics as st

    def tie(row):
        return row["stratum"] != "live"

    live = [(m, row) for m, row in zip(margins, rows) if row["stratum"] == "live"]
    # Discrimination EXCLUDING structurally derivable rows: those are an easy,
    # free-label class and would inflate a pooled number (round-4 caveat).
    judged = [
        (m, row) for m, row in zip(margins, rows)
        if row["stratum"] != "derivable"
    ]
    report = {"source": name, "rows": len(rows)}
    if judged and 0 < sum(1 for _, r in judged if tie(r)) < len(judged):
        report["tie_auc_excl_derivable"] = _auc(
            [-abs(m) for m, _ in judged], [tie(r) for _, r in judged]
        )
    derivable = [
        (m, row) for m, row in zip(margins, rows)
        if row["stratum"] == "derivable"
    ]
    if derivable and live:
        report["tie_auc_derivable_only"] = _auc(
            [-abs(m) for m, _ in (*derivable, *live)],
            [tie(r) for _, r in (*derivable, *live)],
        )
    if live:
        report["live_pairs"] = len(live)
        report["live_sign_agreement"] = sum(
            1 for m, row in live if m * row["delta_u"] > 0
        ) / len(live)
        xs = [m for m, _ in live]
        ys = [row["delta_u"] for _, row in live]
        if len(xs) > 2 and st.pvariance(xs) > 0:
            mean_x, mean_y = st.mean(xs), st.mean(ys)
            report["calibration_slope"] = sum(
                (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
            ) / sum((x - mean_x) ** 2 for x in xs)
        report["live_mae"] = sum(
            abs(m - row["delta_u"]) for m, row in live
        ) / len(live)
    return report


def gate1(args) -> None:
    """Gate 1 — representation gate.

    Can the EXISTING utility representation support being a utility estimator?
    Trains a frozen-feature probe to regress the linked-restricted delta-U and
    scores it against the margin the controller races today (the softcapped
    tower logits) on the same held-out documents, same metrics. Self-
    adjudicating: if the actor margin already discriminates as well, anchoring
    buys nothing. Round-4 gate plan, docs/research/tie-ownership-root-cause-and-solution-space.md.
    """
    import torch
    import torch.nn.functional as F

    documents = _load_documents()
    payload = json.loads((OUT / "evidence-rows-linked.json").read_text())
    rows = payload["rows"]
    if args.exclude_derivable:
        rows = [row for row in rows if not row["derivable_tie"]]
    policy, profile_objects = _load_policy(args, documents)

    features_a, features_b = _extract_features(
        policy, documents, rows, profile_objects[0],
    )
    targets = torch.tensor([row["delta_u"] for row in rows], dtype=torch.float32)
    index = {
        split: torch.tensor(
            [i for i, row in enumerate(rows) if row["split"] == split],
            dtype=torch.long,
        )
        for split in ("train", "calibration", "certification")
    }
    rows_by_split = {
        split: [rows[int(i)] for i in ids] for split, ids in index.items()
    }

    ceiling = _noise_ceiling(rows_by_split["train"])
    print("noise ceiling (train split):")
    for key, value in ceiling.items():
        print(f"  {key}: {value}")

    torch.manual_seed(args.seed)
    dim = features_a.shape[-1]
    if args.probe == "linear":
        probe = torch.nn.Linear(dim, 1)
    else:
        # Match the actor head exactly (Linear-GELU-Linear); a linear probe is
        # strictly weaker than the margin it is being compared against.
        hidden = policy.utility_head[0].out_features
        probe = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )
        if args.probe == "actor-init":
            # Warm-start from the tower's own head: the direct test of
            # "repurpose the existing head as a utility estimator".
            probe.load_state_dict({
                k: v.detach().cpu().clone()
                for k, v in policy.utility_head.state_dict().items()
            })
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    train_a = features_a[index["train"]]
    train_b = features_b[index["train"]]
    train_y = targets[index["train"]]
    for _ in range(args.epochs):
        delta_q = (probe(train_a) - probe(train_b)).squeeze(-1)
        # epsilon-insensitive: never fit inside the instrument's resolution
        slack = torch.clamp((delta_q - train_y).abs() - args.epsilon, min=0.0)
        loss = F.huber_loss(slack, torch.zeros_like(slack), delta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    actor_all = _actor_margin(policy, features_a, features_b)
    with torch.no_grad():
        probe_all = (probe(features_a) - probe(features_b)).squeeze(-1)

    report = {
        "checkpoint": str(args.checkpoint),
        "probe": args.probe,
        "epsilon": args.epsilon,
        "epochs": args.epochs,
        "exclude_derivable": bool(args.exclude_derivable),
        "rows": len(rows),
        "noise_ceiling_train": ceiling,
        "splits": {},
    }
    # Train-split fit separates "cannot learn this at all" from "learns it and
    # does not transfer" — the two readings have opposite consequences.
    train_fit = _score_margins(
        "anchored-probe (TRAIN fit)",
        [float(v) for v in probe_all[index["train"]]], rows_by_split["train"],
    )
    report["train_fit"] = train_fit
    print("\n=== train split (fit check) ===")
    for key, value in train_fit.items():
        if key not in ("source", "rows"):
            print(
                f"  {key}: "
                + (f"{value:.4f}" if isinstance(value, float) else str(value))
            )
    for split in ("calibration", "certification"):
        ids = index[split]
        split_rows = rows_by_split[split]
        anchored = _score_margins(
            "anchored-probe", [float(v) for v in probe_all[ids]], split_rows,
        )
        actor = _score_margins(
            "actor-margin (current tower)",
            [float(v) for v in actor_all[ids]], split_rows,
        )
        per_document = {}
        for doc_id in sorted({row["doc_id"] for row in split_rows}):
            keep = [
                k for k, row in enumerate(split_rows) if row["doc_id"] == doc_id
            ]
            doc_rows = [split_rows[k] for k in keep]
            if len({row["stratum"] != "live" for row in doc_rows}) < 2:
                continue
            per_document[doc_id] = {
                "rows": len(doc_rows),
                "anchored_auc": _score_margins(
                    "a", [float(probe_all[ids[k]]) for k in keep], doc_rows,
                ).get("tie_auc_excl_derivable"),
                "actor_auc": _score_margins(
                    "b", [float(actor_all[ids[k]]) for k in keep], doc_rows,
                ).get("tie_auc_excl_derivable"),
            }
        report["splits"][split] = {
            "anchored": anchored, "actor": actor,
            "per_document": per_document,
        }
        print(f"\n=== {split} ({len(split_rows)} rows) ===")
        for row in (anchored, actor):
            print(f"  {row['source']}")
            for key, value in row.items():
                if key in ("source", "rows"):
                    continue
                print(
                    f"    {key}: "
                    + (f"{value:.4f}" if isinstance(value, float) else str(value))
                )
        usable = [
            (v["anchored_auc"], v["actor_auc"]) for v in per_document.values()
            if v["anchored_auc"] is not None and v["actor_auc"] is not None
        ]
        if usable:
            print(
                f"  per-document AUC (n={len(usable)}): anchored beats actor on "
                f"{sum(1 for a, b in usable if a > b)}/{len(usable)} documents"
            )

    if args.learning_curve:
        # Does held-out discrimination improve as TRAINING documents are added?
        # Rising curve => the failure is evidence support (fixable by breadth);
        # flat curve => the representation does not carry a transferable notion
        # of utility-equivalence and more documents will not help.
        train_docs = sorted(TRAIN_DOCS)
        curve = []
        cert_ids = index["certification"]
        cert_rows = rows_by_split["certification"]
        for count in range(1, len(train_docs) + 1):
            subset = set(train_docs[:count])
            keep = torch.tensor(
                [
                    i for i, row in enumerate(rows)
                    if row["split"] == "train" and row["doc_id"] in subset
                ],
                dtype=torch.long,
            )
            torch.manual_seed(args.seed)
            curve_probe = torch.nn.Sequential(
                torch.nn.Linear(dim, policy.utility_head[0].out_features),
                torch.nn.GELU(),
                torch.nn.Linear(policy.utility_head[0].out_features, 1),
            )
            curve_optimizer = torch.optim.Adam(curve_probe.parameters(), lr=1e-3)
            sub_a, sub_b = features_a[keep], features_b[keep]
            sub_y = targets[keep]
            for _ in range(args.epochs):
                delta_q = (curve_probe(sub_a) - curve_probe(sub_b)).squeeze(-1)
                slack = torch.clamp(
                    (delta_q - sub_y).abs() - args.epsilon, min=0.0
                )
                curve_loss = F.huber_loss(
                    slack, torch.zeros_like(slack), delta=1.0
                )
                curve_optimizer.zero_grad(set_to_none=True)
                curve_loss.backward()
                curve_optimizer.step()
            with torch.no_grad():
                margins = (
                    curve_probe(features_a[cert_ids])
                    - curve_probe(features_b[cert_ids])
                ).squeeze(-1)
            scored = _score_margins(
                "curve", [float(v) for v in margins], cert_rows,
            )
            curve.append({
                "train_documents": count,
                "train_rows": int(len(keep)),
                "certification_tie_auc": scored.get("tie_auc_excl_derivable"),
                "certification_live_sign_agreement": scored.get(
                    "live_sign_agreement"
                ),
            })
            print(
                f"  learning curve: {count} train doc(s), {len(keep):5d} rows -> "
                f"certification tie-AUC "
                f"{scored.get('tie_auc_excl_derivable'):.4f}"
            )
        report["learning_curve"] = curve

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.probe}" + ("-no-derivable" if args.exclude_derivable else "")
    (OUT / f"gate1-report{suffix}.json").write_text(json.dumps(report, indent=1))


def snapshot(args) -> None:
    """Behavioral leg: certified gate bolted onto the frozen checkpoint.

    Runs the synchronous profile snapshot on held-out (calibration +
    certification split) documents twice — gated and ungated — from the SAME
    checkpoint, plus a lambda-0 identity assertion. Only meaningful after
    certification passes; refuses to run on an uncertified arm."""
    import torch
    from types import SimpleNamespace

    from cloak.ranker.environment import LambdaProfile
    from cloak.ranker.interactive import synchronous_profile_snapshot
    from cloak.ranker.profile_count import ProfileCountTargets
    from train_interactive_ranker import (
        _apply_controller_options,
        _semantic_training_policy,
    )

    report = json.loads((OUT / "screening-report.json").read_text())
    arm = next(row for row in report["arms"] if row["arm"] == args.arm)
    if not args.override_gates and not arm.get("gates", {}).get(
        "certified_doc_risk"
    ):
        raise SystemExit(
            f"arm {args.arm} is not document-level certified; behavioral leg "
            "refused"
        )
    saved = torch.load(OUT / f"critic-{args.arm}.pt", map_location="cpu")
    critic = HurdleCritic(int(saved["feature_dim"]), int(saved["seed"]))
    critic.q_head.load_state_dict(saved["q_head"])
    critic.gate_head.load_state_dict(saved["gate_head"])
    threshold = float(saved["threshold"])

    documents = _load_documents()
    held_out = {
        doc_id: document for doc_id, document in documents.items()
        if _split_for(doc_id) != "train"
    }
    menu = json.loads(Path(args.lambda_menu).read_text())
    profile_objects = tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(menu["profile_names"], menu["values"], strict=True)
    )
    policy_args = SimpleNamespace(
        representation_manifest=args.representation_manifest,
        privacy_checkpoint=None,
        profile_count_targets=args.profile_count_targets,
        device=args.device,
    )
    policy = _semantic_training_policy(
        policy_args, tuple(documents.values()), profile_objects
    )
    _apply_controller_options(policy, SimpleNamespace(
        alpha_utility_routing="none",
        controller_gap_scaling="none",
        utility_logit_softcap=25.0,
        controller_gain="evidence",
        controller_gain_hidden=32,
        controller_gain_bound=1.5,
    ))
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    # Freeze the learned gain residual at zero (formalized design: the critic
    # mechanism must be identifiable — v14's interpolated tie-alpha residual
    # would confound canonicalization on held-out documents).
    with torch.no_grad():
        final_layer = list(policy.gain_head.modules())[-1]
        final_layer.weight.zero_()
        final_layer.bias.zero_()
    policy.eval()
    device = policy._device

    def gate(features: torch.Tensor) -> torch.Tensor:
        trunk = critic.trunk(features.cpu())
        q_values = critic.q_head(trunk).squeeze(-1)
        star = int(q_values.argmax())
        pair_inputs = torch.cat(
            [trunk + trunk[star], (trunk - trunk[star]).abs()], dim=-1,
        )
        probability = torch.sigmoid(
            critic.gate_head(pair_inputs).squeeze(-1)
        )
        accepted = probability >= threshold
        accepted[star] = True
        return accepted.to(device)

    targets = ProfileCountTargets(
        json.loads(Path(args.profile_count_targets).read_text())
    )
    docs = tuple(held_out.values())
    results = {}
    greedy_vectors: dict[str, dict[str, dict]] = {"ungated": {}, "gated": {}}
    from cloak.ranker.interactive import sample_trajectory
    for label, gate_fn in (("ungated", None), ("gated", gate)):
        policy.equivalence_gate = gate_fn
        results[label] = synchronous_profile_snapshot(
            policy, docs, profile_objects, targets,
            samples=args.samples, seed=args.seed,
        )
        # Max-lambda greedy action vectors, emitted so the utility leg
        # (gated-vs-ungated regression from false canonicalization) can be
        # scored separately if the arm certifies.
        with torch.no_grad():
            for document in docs:
                walk = sample_trajectory(
                    policy, document, profile_objects[-1],
                    greedy=True, generator=None,
                )
                greedy_vectors[label][document.doc_id] = dict(walk.action_vector)
    policy.equivalence_gate = None

    identity_failures = []
    zero_name = profile_objects[0].name
    for doc_id in results["ungated"]:
        before = results["ungated"][doc_id][zero_name]["greedy_P"]
        after = results["gated"][doc_id][zero_name]["greedy_P"]
        if abs(before - after) > 1e-12:
            identity_failures.append(doc_id)
    if identity_failures:
        raise AssertionError(
            f"lambda-0 identity violated on {identity_failures}"
        )
    # Tie-oracle agreement where measured labels exist: on decisions with a
    # mined tie-band pair involving the gated greedy pick, that pick must be
    # the max-count member of its measured tie set.
    mined = json.loads((OUT / "evidence-rows.json").read_text())["rows"]
    tied_partners: dict[tuple, set] = defaultdict(set)
    for row in mined:
        if row["split"] != "train" and row["stratum"] != "live":
            key = (row["doc_id"], row["decision_id"])
            tied_partners[key].update({row["action_a"], row["action_b"]})
    max_name = profile_objects[-1].name
    oracle_checked = oracle_agreed = 0
    for doc_id, per_profile in results["gated"].items():
        for entry in per_profile[max_name].get("decisions", ()):
            key = (doc_id, entry["decision_id"])
            partners = tied_partners.get(key)
            if not partners:
                continue
            oracle_checked += 1
            best = max(
                float(targets.action_scores(entry["decision_id"], (a,))[0])
                for a in partners
            )
            if entry["greedy_action_score"] >= best - 1e-9:
                oracle_agreed += 1
    summary = {
        "arm": args.arm,
        "threshold": threshold,
        "gain_residual": "frozen-zero",
        "documents": {},
        "tie_oracle": {"checked": oracle_checked, "agreed": oracle_agreed},
    }
    for doc_id in sorted(results["ungated"]):
        summary["documents"][doc_id] = {
            "split": _split_for(doc_id),
            **{
                label: {
                    "greedy_P_lambda0": results[label][doc_id][zero_name]["greedy_P"],
                    "greedy_P_lambdamax": results[label][doc_id][max_name]["greedy_P"],
                    "greedy_vector_lambdamax": greedy_vectors[label][doc_id],
                }
                for label in ("ungated", "gated")
            },
        }
    (OUT / f"snapshot-{args.arm}.json").write_text(json.dumps(summary, indent=1))
    for split in ("certification", "calibration"):
        rows = [
            row for row in summary["documents"].values()
            if row["split"] == split
        ]
        if not rows:
            continue
        gated_mean = sum(
            row["gated"]["greedy_P_lambdamax"]
            - row["gated"]["greedy_P_lambda0"] for row in rows
        ) / len(rows)
        ungated_mean = sum(
            row["ungated"]["greedy_P_lambdamax"]
            - row["ungated"]["greedy_P_lambda0"] for row in rows
        ) / len(rows)
        print(
            f"{split:13s}: docs {len(rows):2d} | mean greedy separation gated "
            f"{gated_mean:.3f} vs ungated {ungated_mean:.3f}"
        )
    print(
        f"lambda-0 identity: exact | tie-oracle agreement "
        f"{oracle_agreed}/{oracle_checked} (measured-label decisions only)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    miner = subcommands.add_parser("mine")
    miner.add_argument(
        "--utility-cache",
        default="results/ranker_v2/cache/utility-results.jsonl",
    )
    miner.add_argument(
        "--utility-artifact", default="results/ranker_v2/qa/aci-full.utility",
    )
    miner.set_defaults(func=mine)
    runner = subcommands.add_parser("run")
    runner.add_argument("--checkpoint", required=True)
    runner.add_argument(
        "--representation-manifest",
        default="results/ranker_v2/architecture/representation-full/manifest.json",
    )
    runner.add_argument(
        "--profile-count-targets",
        default="results/ranker_v2/reward/profile-count-targets.json",
    )
    runner.add_argument(
        "--lambda-menu", default="results/ranker_v2/preflight/lambda-menu.json",
    )
    runner.add_argument("--device", default="cuda")
    runner.add_argument("--seed", type=int, default=47)
    runner.set_defaults(func=run)
    gate = subcommands.add_parser("gate1")
    gate.add_argument("--checkpoint", required=True)
    gate.add_argument(
        "--representation-manifest",
        default="results/ranker_v2/architecture/representation-full/manifest.json",
    )
    gate.add_argument(
        "--profile-count-targets",
        default="results/ranker_v2/reward/profile-count-targets.json",
    )
    gate.add_argument(
        "--lambda-menu", default="results/ranker_v2/preflight/lambda-menu.json",
    )
    gate.add_argument("--device", default="cuda")
    gate.add_argument("--seed", type=int, default=47)
    gate.add_argument("--epochs", type=int, default=400)
    gate.add_argument("--epsilon", type=float, default=READER_FLOOR)
    gate.add_argument("--exclude-derivable", action="store_true")
    gate.add_argument(
        "--probe", choices=("linear", "mlp", "actor-init"), default="mlp",
    )
    gate.add_argument("--learning-curve", action="store_true")
    gate.set_defaults(func=gate1)
    snap = subcommands.add_parser("snapshot")
    snap.add_argument("--arm", default="hurdle-eps044")
    snap.add_argument("--checkpoint", required=True)
    snap.add_argument(
        "--representation-manifest",
        default="results/ranker_v2/architecture/representation-full/manifest.json",
    )
    snap.add_argument(
        "--profile-count-targets",
        default="results/ranker_v2/reward/profile-count-targets.json",
    )
    snap.add_argument(
        "--lambda-menu", default="results/ranker_v2/preflight/lambda-menu.json",
    )
    snap.add_argument("--device", default="cuda")
    snap.add_argument("--samples", type=int, default=16)
    snap.add_argument("--seed", type=int, default=0)
    snap.add_argument("--override-gates", action="store_true")
    snap.set_defaults(func=snapshot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
