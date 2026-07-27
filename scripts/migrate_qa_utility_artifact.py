"""Rebind accepted QA assertions to a count-only-compatible Ranker-v2 environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from pathlib import Path

from cloak.qa.builder import policy_routing
from cloak.qa.freeze import compare_frozen_environment_semantics


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK2_REFERENCE = _REPO_ROOT / "results/qa_v2_aci_full/ranker-env.json"
_TASK2_REFERENCE_LABEL = "results/qa_v2_aci_full/ranker-env.json"
_V16_ARTIFACT_HASH = "sha256:ca09be95dc716430bb06a70be2d8cccbff847b7bb1440f9c0f84ce187fff5e7e"


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_mapping(path: Path, label: str) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _frozen_environment(environment: Mapping) -> Mapping:
    frozen = environment.get("frozen_environment", environment)
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("documents"), Mapping):
        raise ValueError("environment has no frozen documents")
    return frozen


def _compatibility_summary(comparison: Mapping, *, reference: str) -> dict:
    differences = list(comparison.get("differences", []))
    markers = (
        "document_ids", "occurrence_ids", "decision_ids", "action_ids",
        "fills", "modes", "authored_order",
    )
    fields = sorted({
        next(
            (marker for marker in markers
             if f":{marker}:" in str(value) or str(value).startswith(f"{marker}:")),
            "other",
        )
        for value in differences
    })
    return {
        "verdict": comparison.get("verdict"),
        "reference": reference,
        "difference_count": len(differences),
        "differing_fields": fields,
        "reference_document_count": comparison.get("reference_document_count"),
        "candidate_document_count": comparison.get("candidate_document_count"),
    }


def _cached_score_vectors(assertions: Mapping) -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = {}
    for assertion_id, assertion in assertions.items():
        evidence = assertion.get("evidence") or {}
        validation = evidence.get("validation") or {}
        for name, value in (validation.get("scores") or {}).items():
            if isinstance(value, Real) and not isinstance(value, bool):
                vectors.setdefault(str(name), {})[str(assertion_id)] = float(value)
    return vectors


def _cached_document_utilities(artifact: Mapping, scores: Mapping[str, float]) -> dict[str, float]:
    assertions = artifact.get("assertions", {})
    utilities: dict[str, float] = {}
    for doc_id, document in artifact.get("documents", {}).items():
        denominator = float(document["utility_weight_denominator"])
        if denominator <= 0:
            raise ValueError(f"non-positive utility denominator for {doc_id}")
        utilities[str(doc_id)] = sum(
            float(assertions[assertion_id]["weight"]) * scores[assertion_id]
            for assertion_id in document.get("assertion_ids", [])
            if assertion_id in scores
        ) / denominator
    return utilities


def migrate_utility_artifact(
    artifact: Mapping,
    environment: Mapping,
    *,
    semantic_reference: Mapping | None = None,
    semantic_reference_label: str = "input artifact embedded inventory",
) -> tuple[dict, dict]:
    frozen = _frozen_environment(environment)
    old_documents = artifact.get("documents", {})
    reference = {
        "documents": {
            str(doc_id): {
                "occurrences": document.get("occurrences", []),
                "decisions": document.get("decisions", []),
            }
            for doc_id, document in old_documents.items()
        }
    }
    input_inventory_comparison = compare_frozen_environment_semantics(reference, frozen)
    input_inventory_audit = _compatibility_summary(
        input_inventory_comparison,
        reference="input artifact embedded inventory",
    )
    unknown_assertion_occurrences: set[str] = set()
    stale_joint_anchor_assertions: list[str] = []
    artifact_assertions = artifact.get("assertions", {})
    for doc_id, candidate_document in frozen.get("documents", {}).items():
        candidate_occurrences = {
            str(row["occurrence_id"])
            for row in candidate_document.get("occurrences", [])
        }
        candidate_actions = {
            str(decision["decision_id"]): {
                str(action["action_id"]) for action in decision.get("actions", [])
            }
            for decision in candidate_document.get("decisions", [])
        }
        for assertion_id in old_documents.get(doc_id, {}).get("assertion_ids", []):
            assertion = artifact_assertions[assertion_id]
            unknown_assertion_occurrences.update(
                set(map(str, assertion.get("occurrence_ids", []))) - candidate_occurrences
            )
            support = assertion.get("expected_action_support") or {}
            action_vector = support.get("joint_anchor_action_vector") or {}
            if action_vector and any(
                str(decision_id) not in candidate_actions
                or str(action_id) not in candidate_actions[str(decision_id)]
                for decision_id, action_id in action_vector.items()
            ):
                stale_joint_anchor_assertions.append(str(assertion_id))
    input_inventory_audit.update({
        "unknown_assertion_occurrence_links": len(unknown_assertion_occurrences),
        "stale_joint_anchor_assertions": len(stale_joint_anchor_assertions),
        "stale_joint_anchor_assertion_ids": sorted(stale_joint_anchor_assertions),
    })
    comparison = compare_frozen_environment_semantics(
        semantic_reference if semantic_reference is not None else reference,
        frozen,
    )
    compatibility = _compatibility_summary(
        comparison, reference=semantic_reference_label,
    )
    if comparison["verdict"] != "count-only compatible":
        return {}, {
            "status": "qa_rebuild_required",
            "reason": "semantic_environment_change",
            "compatibility": compatibility,
            "input_inventory_audit": input_inventory_audit,
            "hashes": {
                "old_artifact": artifact.get("artifact_hash"),
                "old_environment": artifact.get("environment_hash"),
                "new_environment": frozen.get("environment_hash"),
            },
        }

    migrated = deepcopy(dict(artifact))
    migrated["artifact_version"] = "utility-assertions-v2"
    migrated["environment_hash"] = frozen.get("environment_hash")
    assertions = migrated.get("assertions", {})
    policy_count = 0
    fixed_count = 0
    linked_count = 0
    residual_count = 0
    uncovered_count = 0
    fixed_only_ids: list[str] = []
    unexpected_fixed_only_ids: list[str] = []

    candidate_documents = frozen["documents"]
    if list(old_documents) != list(candidate_documents):
        raise ValueError("count-only migration requires identical document ordering")
    for doc_id, document in migrated.get("documents", {}).items():
        candidate = candidate_documents[doc_id]
        decisions = list(candidate.get("decisions", []))
        policy_ids = [
            str(row["decision_id"])
            for row in decisions
            if row.get("ranker_selectable") is True
        ]
        fixed_ids = [
            str(row["decision_id"])
            for row in decisions
            if row.get("ranker_selectable") is not True
        ]
        occurrences = list(candidate.get("occurrences", []))
        occurrence_to_decision = {
            str(row["occurrence_id"]): (
                None if row.get("decision_id") is None else str(row["decision_id"])
            )
            for row in occurrences
        }
        if len(occurrence_to_decision) != len(occurrences):
            raise ValueError(f"duplicate occurrence ids for {doc_id}")

        covered: set[str] = set()
        for assertion_id in document.get("assertion_ids", []):
            row = assertions[assertion_id]
            if str(row.get("doc_id")) != str(doc_id):
                raise ValueError(f"assertion document mismatch: {assertion_id}")
            dependencies, routing = policy_routing(
                row, occurrence_to_decision, policy_ids,
            )
            row["policy_dependency_decision_ids"] = dependencies
            row["credit_routing"] = routing
            covered.update(dependencies)
            if routing == "linked":
                linked_count += 1
            else:
                residual_count += 1
                mapped = {
                    occurrence_to_decision[str(occurrence_id)]
                    for occurrence_id in row.get("occurrence_ids", [])
                }
                if mapped & set(fixed_ids):
                    fixed_only_ids.append(str(assertion_id))
                    if not (
                        row.get("family") == "delivered"
                        and row.get("subtype") == "content"
                    ):
                        unexpected_fixed_only_ids.append(str(assertion_id))

        document.pop("controlled_decision_ids", None)
        document.pop("uncovered_decision_ids", None)
        document["environment_document_hash"] = candidate.get("environment_document_hash")
        document["policy_decision_ids"] = policy_ids
        document["fixed_decision_ids"] = fixed_ids
        document["uncovered_policy_decision_ids"] = [
            decision_id for decision_id in policy_ids if decision_id not in covered
        ]
        document["occurrence_to_decision"] = occurrence_to_decision
        document["occurrences"] = deepcopy(occurrences)
        document["decisions"] = deepcopy(decisions)
        policy_count += len(policy_ids)
        fixed_count += len(fixed_ids)
        uncovered_count += len(document["uncovered_policy_decision_ids"])

    old_assertions = artifact.get("assertions", {})
    assertion_ids_identical = list(old_assertions) == list(assertions)
    scoring_contracts_identical = all(
        old_assertions[key].get("scoring_contract") == assertions[key].get("scoring_contract")
        for key in old_assertions
    )
    weights_identical = all(
        old_assertions[key].get("weight") == assertions[key].get("weight")
        for key in old_assertions
    )
    cached_vectors = _cached_score_vectors(old_assertions)
    parity_by_vector: dict[str, dict] = {}
    parity_difference_ids: set[str] = set()
    for name, scores in sorted(cached_vectors.items()):
        old_utilities = _cached_document_utilities(artifact, scores)
        new_utilities = _cached_document_utilities(migrated, scores)
        differences = sorted(
            doc_id for doc_id in old_utilities
            if old_utilities[doc_id] != new_utilities.get(doc_id)
        )
        parity_difference_ids.update(differences)
        parity_by_vector[name] = {
            "cached_component_count": len(scores),
            "differing_document_ids": differences,
        }
    parity_differences = sorted(parity_difference_ids)
    cached_component_ids = {
        assertion_id
        for scores in cached_vectors.values()
        for assertion_id in scores
    }

    migrated.pop("artifact_hash", None)
    migrated["artifact_hash"] = _stable_hash(migrated)
    report = {
        "status": "count-only compatible",
        "compatibility": compatibility,
        "input_inventory_audit": input_inventory_audit,
        "hashes": {
            "old_artifact": artifact.get("artifact_hash"),
            "new_artifact": migrated["artifact_hash"],
            "old_environment": artifact.get("environment_hash"),
            "new_environment": frozen.get("environment_hash"),
        },
        "identity_checks": {
            "assertion_ids_identical": assertion_ids_identical,
            "scoring_contracts_identical": scoring_contracts_identical,
            "weights_identical": weights_identical,
        },
        "counts": {
            "documents": len(migrated.get("documents", {})),
            "assertions": len(assertions),
            "policy_decisions": policy_count,
            "fixed_decisions": fixed_count,
            "linked_assertions": linked_count,
            "residual_assertions": residual_count,
            "uncovered_policy_decisions": uncovered_count,
            "fixed_only_linked_assertions": len(fixed_only_ids),
            "unexpected_fixed_only_linked_assertions": len(unexpected_fixed_only_ids),
        },
        "fixed_only_assertion_ids": sorted(fixed_only_ids),
        "unexpected_fixed_only_assertion_ids": sorted(unexpected_fixed_only_ids),
        "document_utility_parity": {
            "identical": not parity_differences,
            "cached_component_count": len(cached_component_ids),
            "cached_score_count": sum(map(len, cached_vectors.values())),
            "cached_vector_names": sorted(cached_vectors),
            "vectors": parity_by_vector,
            "differing_document_ids": parity_differences,
        },
    }
    if not all((assertion_ids_identical, scoring_contracts_identical, weights_identical)):
        raise ValueError("migration changed assertion identity, scoring contracts, or weights")
    if parity_differences:
        raise ValueError(f"cached document utility changed for {parity_differences}")
    return migrated, report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    artifact = _load_mapping(Path(args.input), "input artifact")
    environment = _load_mapping(Path(args.environment), "environment")
    semantic_reference = None
    semantic_reference_label = "input artifact embedded inventory"
    if artifact.get("artifact_hash") == _V16_ARTIFACT_HASH:
        pinned_environment = _load_mapping(_TASK2_REFERENCE, "Task 2 reference environment")
        semantic_reference = _frozen_environment(pinned_environment)
        semantic_reference_label = _TASK2_REFERENCE_LABEL
    migrated, report = migrate_utility_artifact(
        artifact,
        environment,
        semantic_reference=semantic_reference,
        semantic_reference_label=semantic_reference_label,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] == "qa_rebuild_required":
        print("qa_rebuild_required: semantic environment change", flush=True)
        return 2
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(migrated, indent=1) + "\n")
    print(
        f"wrote {output_path}: docs={report['counts']['documents']} "
        f"assertions={report['counts']['assertions']} "
        f"policy={report['counts']['policy_decisions']} "
        f"fixed={report['counts']['fixed_decisions']} "
        f"residual={report['counts']['residual_assertions']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
