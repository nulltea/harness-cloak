"""Matched architecture-fitness diagnostics for the semantic Ranker-v2 policy."""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cloak.reward.utility_cache import stable_hash


ARTIFACT_VERSION = "ranker-v2-architecture-spike-v1"

APPROVED_ARMS = MappingProxyType({
    "relation_representation": (
        "candidate-only", "independent-bi-encoder", "joint-pair",
    ),
    "context_readout": (
        "local-cls-mean", "target-bidirectional", "full-candidate-attention",
    ),
    "action_history": (
        "none", "utility-gru", "selected-cross-attention",
    ),
    "metadata_shortcut": (
        "metadata-only", "legacy-film-gru", "semantic",
    ),
    "decision_order": (
        "first-occurrence", "reverse", "seeded-0", "seeded-1", "seeded-2",
    ),
})

SELECTED_ARMS = MappingProxyType({
    "relation_representation": "joint-pair",
    "context_readout": "full-candidate-attention",
    "action_history": "selected-cross-attention",
    "metadata_shortcut": "semantic",
    "decision_order": "first-occurrence",
})

LOAD_BEARING_METRICS = MappingProxyType({
    "relation_representation": "held_out_outcome_ranking",
    "context_readout": "contextual_utility_ranking",
    "action_history": "multi_decision_utility",
    "metadata_shortcut": "held_out_utility",
    "decision_order": "decision_order_utility",
})

INTERVENTIONS_BY_FAMILY = MappingProxyType({
    "relation_representation": (
        "candidate_swap", "source_candidate_reversal",
    ),
    "context_readout": (
        "context_swap", "target_marker_ablation", "full_document_ablation",
        "non_middle_utility",
    ),
    "action_history": (
        "relevant_prior_action", "irrelevant_prior_action",
        "memory_row_permutation",
    ),
    "metadata_shortcut": (
        "count_metadata_mutation", "lambda_zero_identity",
        "lambda_monotonicity",
    ),
    "decision_order": (),
})

REPORT_SECTIONS = (
    "relation_representation",
    "semantic_privacy_transfer",
    "context_readout",
    "action_history",
    "metadata_shortcut",
    "lambda_controller",
    "decision_order",
    "operational_cost",
    "contamination_boundary",
    "promotion_verdict",
)

REQUIRED_ABSOLUTE_THRESHOLDS = (
    "max_incremental_latency_ms",
    "max_peak_memory_bytes",
    "max_cache_bytes",
    "max_cache_build_seconds",
    "max_order_utility_range",
)

_CONTRACT_FIELDS = (
    "profile_split_hash",
    "utility_cases_hash",
    "seeds",
    "trainable_head_budget",
    "projection_width",
    "diagnostic_head_hash",
    "encoder_revision",
    "cache_manifest_hash",
)

_CONTRACT_LABELS = {
    "profile_split_hash": "profile split",
    "utility_cases_hash": "utility cases",
    "seeds": "seeds",
    "trainable_head_budget": "trainable-head budget",
    "projection_width": "projection width",
    "diagnostic_head_hash": "diagnostic head",
    "encoder_revision": "encoder revision",
    "cache_manifest_hash": "cache manifest",
}

_OPERATIONAL_FIELDS = (
    "incremental_latency_ms", "peak_memory_bytes", "cache_bytes",
    "cache_build_seconds",
)


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or nonnegative and float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite numeric value")
    return float(value)


def _plain_diagnostic_value(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_diagnostic_value(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain_diagnostic_value(item, label) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return _finite_number(value, label)
    raise ValueError(f"{label} contains an unsupported value")


@dataclass(frozen=True)
class MatchedArmContract:
    """Frozen identities that must be byte-equivalent across every spike arm."""

    profile_split_hash: str
    utility_cases_hash: str
    seeds: tuple[int, int, int]
    trainable_head_budget: int
    projection_width: int
    diagnostic_head_hash: str
    encoder_revision: str
    cache_manifest_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.seeds) != 3
            or len(set(self.seeds)) != 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds)
        ):
            raise ValueError("matched arm contract requires three distinct integer seeds")
        if (
            isinstance(self.trainable_head_budget, bool)
            or not isinstance(self.trainable_head_budget, int)
            or self.trainable_head_budget <= 0
            or isinstance(self.projection_width, bool)
            or not isinstance(self.projection_width, int)
            or self.projection_width <= 0
        ):
            raise ValueError("matched arm widths and head budget must be positive")
        for name in (
            "profile_split_hash", "utility_cases_hash", "diagnostic_head_hash",
            "encoder_revision", "cache_manifest_hash",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"matched arm contract lacks {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_split_hash": self.profile_split_hash,
            "utility_cases_hash": self.utility_cases_hash,
            "seeds": list(self.seeds),
            "trainable_head_budget": self.trainable_head_budget,
            "projection_width": self.projection_width,
            "diagnostic_head_hash": self.diagnostic_head_hash,
            "encoder_revision": self.encoder_revision,
            "cache_manifest_hash": self.cache_manifest_hash,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MatchedArmContract":
        if set(payload) != set(_CONTRACT_FIELDS):
            raise ValueError("matched arm contract fields are incomplete")
        return cls(
            profile_split_hash=payload["profile_split_hash"],
            utility_cases_hash=payload["utility_cases_hash"],
            seeds=tuple(payload["seeds"]),
            trainable_head_budget=payload["trainable_head_budget"],
            projection_width=payload["projection_width"],
            diagnostic_head_hash=payload["diagnostic_head_hash"],
            encoder_revision=payload["encoder_revision"],
            cache_manifest_hash=payload["cache_manifest_hash"],
        )


@dataclass(frozen=True)
class ArmMeasurement:
    """One matched arm's held-out outcomes, interventions, parameters, and cost."""

    family: str
    arm: str
    matched_contract: MatchedArmContract
    seed_metrics: Mapping[int, Mapping[str, float]]
    trainable_parameters: Mapping[str, int]
    interventions: Mapping[str, Mapping[str, Any]]
    operational_cost: Mapping[str, float]
    diagnostic_distributions: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.family not in APPROVED_ARMS or self.arm not in APPROVED_ARMS[self.family]:
            raise ValueError(f"unsupported architecture arm: {self.family}:{self.arm}")
        if not isinstance(self.matched_contract, MatchedArmContract):
            raise TypeError("arm measurement lacks a matched contract")
        metrics = {
            int(seed): dict(values) for seed, values in self.seed_metrics.items()
        }
        if set(metrics) != set(self.matched_contract.seeds):
            raise ValueError(f"arm seeds differ for {self.family}:{self.arm}")
        required_metrics = {
            LOAD_BEARING_METRICS[self.family], "local_utility",
        }
        for seed, values in metrics.items():
            missing = sorted(required_metrics - set(values))
            if missing:
                raise ValueError(
                    f"arm metrics are incomplete for {self.family}:{self.arm}:{seed}: "
                    f"{missing}"
                )
            for name, value in values.items():
                _finite_number(value, f"metric {name}")
        parameters = dict(self.trainable_parameters)
        if not parameters or any(
            not isinstance(name, str)
            or not name
            or "padding" in name.casefold()
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in parameters.items()
        ):
            raise ValueError("arm trainable parameter counts are invalid")
        if parameters.get("diagnostic_head") != (
            self.matched_contract.trainable_head_budget
        ):
            raise ValueError("arm diagnostic-head trainable count differs from budget")
        costs = dict(self.operational_cost)
        if set(costs) != set(_OPERATIONAL_FIELDS):
            raise ValueError("arm operational cost fields are incomplete")
        for name, value in costs.items():
            _finite_number(value, f"operational cost {name}", nonnegative=True)
        interventions = {
            str(name): dict(values) for name, values in self.interventions.items()
        }
        for name, values in interventions.items():
            if set(values) < {"passed", "observed_effect"} or not isinstance(
                values["passed"], bool
            ):
                raise ValueError(f"invalid intervention measurement: {name}")
            _finite_number(values["observed_effect"], f"intervention {name}")
        distributions = _plain_diagnostic_value(
            dict(self.diagnostic_distributions), "diagnostic distributions"
        )
        object.__setattr__(
            self,
            "seed_metrics",
            MappingProxyType({
                seed: MappingProxyType(values) for seed, values in metrics.items()
            }),
        )
        object.__setattr__(
            self, "trainable_parameters", MappingProxyType(parameters)
        )
        object.__setattr__(
            self,
            "interventions",
            MappingProxyType({
                name: MappingProxyType(values)
                for name, values in interventions.items()
            }),
        )
        object.__setattr__(self, "operational_cost", MappingProxyType(costs))
        object.__setattr__(
            self, "diagnostic_distributions", MappingProxyType(distributions)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "arm": self.arm,
            "matched_contract": self.matched_contract.to_dict(),
            "seed_metrics": {
                str(seed): dict(values)
                for seed, values in sorted(self.seed_metrics.items())
            },
            "trainable_parameters": dict(sorted(self.trainable_parameters.items())),
            "interventions": {
                name: dict(values)
                for name, values in sorted(self.interventions.items())
            },
            "operational_cost": dict(sorted(self.operational_cost.items())),
            "diagnostic_distributions": _plain_diagnostic_value(
                self.diagnostic_distributions, "diagnostic distributions"
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArmMeasurement":
        required = {
            "family", "arm", "matched_contract", "seed_metrics",
            "trainable_parameters", "interventions", "operational_cost",
            "diagnostic_distributions",
        }
        if set(payload) != required:
            raise ValueError("arm measurement fields are incomplete")
        return cls(
            family=str(payload["family"]),
            arm=str(payload["arm"]),
            matched_contract=MatchedArmContract.from_mapping(
                payload["matched_contract"]
            ),
            seed_metrics={
                int(seed): values
                for seed, values in payload["seed_metrics"].items()
            },
            trainable_parameters=payload["trainable_parameters"],
            interventions=payload["interventions"],
            operational_cost=payload["operational_cost"],
            diagnostic_distributions=payload["diagnostic_distributions"],
        )


def _validate_complete_arms(
    measurements: Sequence[ArmMeasurement],
) -> tuple[MatchedArmContract, dict[str, dict[str, ArmMeasurement]]]:
    rows = tuple(measurements)
    if not rows:
        raise ValueError("architecture spike requires arm measurements")
    grouped: dict[str, dict[str, ArmMeasurement]] = {
        family: {} for family in APPROVED_ARMS
    }
    contract = rows[0].matched_contract
    for row in rows:
        for field in _CONTRACT_FIELDS:
            if getattr(row.matched_contract, field) != getattr(contract, field):
                raise ValueError(
                    f"matched arms differ in {_CONTRACT_LABELS[field]}"
                )
        if row.arm in grouped[row.family]:
            raise ValueError(f"duplicate architecture arm: {row.family}:{row.arm}")
        grouped[row.family][row.arm] = row
    for family, approved in APPROVED_ARMS.items():
        if set(grouped[family]) != set(approved):
            missing = sorted(set(approved) - set(grouped[family]))
            extra = sorted(set(grouped[family]) - set(approved))
            raise ValueError(
                f"architecture arms incomplete for {family}: "
                f"missing={missing} extra={extra}"
            )
    return contract, grouped


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_seed_bootstrap(
    selected: ArmMeasurement,
    baseline: ArmMeasurement,
    metric: str,
) -> dict[str, Any]:
    """Return the exact paired bootstrap distribution over the three frozen seeds."""

    if selected.matched_contract.seeds != baseline.matched_contract.seeds:
        raise ValueError("paired bootstrap seeds differ")
    seeds = selected.matched_contract.seeds
    differences = tuple(
        float(selected.seed_metrics[seed][metric])
        - float(baseline.seed_metrics[seed][metric])
        for seed in seeds
    )
    bootstrap = [
        sum(differences[index] for index in indices) / len(indices)
        for indices in itertools.product(range(len(seeds)), repeat=len(seeds))
    ]
    return {
        "family": selected.family,
        "selected_arm": selected.arm,
        "baseline_arm": baseline.arm,
        "metric": metric,
        "seeds": list(seeds),
        "paired_seed_differences": list(differences),
        "mean_improvement": sum(differences) / len(differences),
        "confidence_interval_95": [
            _quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975),
        ],
        "bootstrap_distribution": sorted(bootstrap),
    }


def _family_report(
    family: str, rows: Mapping[str, ArmMeasurement]
) -> dict[str, Any]:
    return {
        "selected_candidate": SELECTED_ARMS[family],
        "load_bearing_metric": LOAD_BEARING_METRICS[family],
        "arms": {
            arm: {
                "seed_metrics": {
                    str(seed): dict(values)
                    for seed, values in sorted(row.seed_metrics.items())
                },
                "trainable_parameters": dict(
                    sorted(row.trainable_parameters.items())
                ),
                "interventions": {
                    name: dict(values)
                    for name, values in sorted(row.interventions.items())
                },
                "diagnostic_distributions": _plain_diagnostic_value(
                    row.diagnostic_distributions, "diagnostic distributions"
                ),
            }
            for arm, row in rows.items()
        },
    }


def _required_interventions(
    grouped: Mapping[str, Mapping[str, ArmMeasurement]],
) -> dict[str, dict[str, Any]]:
    observed = {}
    for family, names in INTERVENTIONS_BY_FAMILY.items():
        selected = grouped[family][SELECTED_ARMS[family]]
        for name in names:
            if name not in selected.interventions:
                raise ValueError(f"missing architecture intervention: {name}")
            observed[name] = dict(selected.interventions[name])
    return observed


def _local_regression(
    selected: ArmMeasurement, baseline: ArmMeasurement,
) -> bool:
    return any(
        float(selected.seed_metrics[seed]["local_utility"])
        < float(baseline.seed_metrics[seed]["local_utility"])
        for seed in selected.matched_contract.seeds
    )


def _operational_assessment(
    grouped: Mapping[str, Mapping[str, ArmMeasurement]],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, Any], bool]:
    arms = {
        f"{family}:{arm}": dict(row.operational_cost)
        for family, rows in grouped.items()
        for arm, row in rows.items()
    }
    maxima = {
        field: max(float(values[field]) for values in arms.values())
        for field in _OPERATIONAL_FIELDS
    }
    checks = {
        "incremental_latency_ms": (
            maxima["incremental_latency_ms"]
            <= float(thresholds["max_incremental_latency_ms"])
            if "max_incremental_latency_ms" in thresholds else None
        ),
        "peak_memory_bytes": (
            maxima["peak_memory_bytes"]
            <= float(thresholds["max_peak_memory_bytes"])
            if "max_peak_memory_bytes" in thresholds else None
        ),
        "cache_bytes": (
            maxima["cache_bytes"] <= float(thresholds["max_cache_bytes"])
            if "max_cache_bytes" in thresholds else None
        ),
        "cache_build_seconds": (
            maxima["cache_build_seconds"]
            <= float(thresholds["max_cache_build_seconds"])
            if "max_cache_build_seconds" in thresholds else None
        ),
    }
    passed = all(value is not False for value in checks.values())
    return {"arms": arms, "maxima": maxima, "budget_checks": checks}, passed


def build_architecture_spike_report(
    measurements: Sequence[ArmMeasurement],
    *,
    registered_thresholds: Mapping[str, float],
    aci_context: bool,
    non_aci_manifest_hash: str | None,
) -> dict[str, Any]:
    """Validate matched arms and build the immutable relative-promotion report."""

    if not isinstance(aci_context, bool):
        raise TypeError("aci_context must be boolean")
    if non_aci_manifest_hash is not None and (
        not isinstance(non_aci_manifest_hash, str) or not non_aci_manifest_hash
    ):
        raise ValueError("non-ACI manifest hash is invalid")
    contract, grouped = _validate_complete_arms(measurements)
    interventions = _required_interventions(grouped)
    thresholds = dict(registered_thresholds)
    for name, value in thresholds.items():
        _finite_number(value, f"registered threshold {name}", nonnegative=True)
    unknown_thresholds = sorted(set(thresholds) - set(REQUIRED_ABSOLUTE_THRESHOLDS))
    if unknown_thresholds:
        raise ValueError(f"unregistered architecture thresholds: {unknown_thresholds}")
    missing_thresholds = sorted(set(REQUIRED_ABSOLUTE_THRESHOLDS) - set(thresholds))

    comparisons = []
    for family in (
        "relation_representation", "context_readout", "metadata_shortcut",
    ):
        selected = grouped[family][SELECTED_ARMS[family]]
        metric = LOAD_BEARING_METRICS[family]
        for baseline_name in APPROVED_ARMS[family]:
            if baseline_name != selected.arm:
                comparisons.append(paired_seed_bootstrap(
                    selected, grouped[family][baseline_name], metric,
                ))

    history_rows = grouped["action_history"]
    history_candidate = history_rows["selected-cross-attention"]
    no_history = history_rows["none"]
    history_comparison = paired_seed_bootstrap(
        history_candidate,
        no_history,
        LOAD_BEARING_METRICS["action_history"],
    )
    history_interventions_passed = all(
        interventions[name]["passed"]
        for name in INTERVENTIONS_BY_FAMILY["action_history"]
    )
    history_local_regression = _local_regression(history_candidate, no_history)
    history_selected = (
        "selected-cross-attention"
        if history_comparison["confidence_interval_95"][0] > 0.0
        and not history_local_regression
        and history_interventions_passed
        else "none"
    )

    context_selected = grouped["context_readout"]["full-candidate-attention"]
    context_baseline = grouped["context_readout"]["local-cls-mean"]
    local_regression = _local_regression(context_selected, context_baseline)

    non_history_interventions = tuple(
        name
        for family, names in INTERVENTIONS_BY_FAMILY.items()
        if family != "action_history"
        for name in names
    )
    shortcut_invariance_passed = all(
        interventions[name]["passed"] for name in non_history_interventions
    ) and (
        history_interventions_passed
        if history_selected == "selected-cross-attention"
        else True
    )

    operational, operational_passed = _operational_assessment(grouped, thresholds)
    order_rows = grouped["decision_order"]
    order_metric = LOAD_BEARING_METRICS["decision_order"]
    order_ranges = {
        str(seed): max(
            float(row.seed_metrics[seed][order_metric])
            for row in order_rows.values()
        ) - min(
            float(row.seed_metrics[seed][order_metric])
            for row in order_rows.values()
        )
        for seed in contract.seeds
    }
    max_order_range = max(order_ranges.values())
    order_passed = (
        max_order_range <= float(thresholds["max_order_utility_range"])
        if "max_order_utility_range" in thresholds else True
    )

    relative_passed = all(
        row["confidence_interval_95"][0] > 0.0 for row in comparisons
    )
    contamination = {
        "label": (
            "development_only_encoder_contaminated"
            if aci_context and not non_aci_manifest_hash
            else (
                "non_aci_validation_available"
                if non_aci_manifest_hash
                else "non_aci_manifest_missing"
            )
        ),
        "non_aci_manifest_hash": non_aci_manifest_hash,
        "encoder_selection_promotion_allowed": bool(non_aci_manifest_hash),
        "out_of_corpus_generalization_allowed": bool(non_aci_manifest_hash),
    }
    hard_failure = (
        not relative_passed
        or local_regression
        or not shortcut_invariance_passed
        or not operational_passed
        or not order_passed
    )
    if hard_failure:
        verdict = "REJECT"
    elif missing_thresholds:
        verdict = "NEEDS_THRESHOLD_REGISTRATION"
    elif aci_context and not non_aci_manifest_hash:
        verdict = "DEVELOPMENT_ONLY_ENCODER_CONTAMINATED"
    elif not non_aci_manifest_hash:
        verdict = "NEEDS_NON_ACI_VALIDATION"
    else:
        verdict = "PROMOTE"

    relation_report = _family_report(
        "relation_representation", grouped["relation_representation"]
    )
    context_report = _family_report("context_readout", grouped["context_readout"])
    history_report = _family_report("action_history", history_rows)
    history_report.update({
        "selected_arm": history_selected,
        "memory_vs_no_history": history_comparison,
        "gru_auto_promotion_allowed": False,
        "local_utility_regression": history_local_regression,
    })
    shortcut_report = _family_report(
        "metadata_shortcut", grouped["metadata_shortcut"]
    )
    order_report = _family_report("decision_order", order_rows)
    order_report.update({
        "selected_arm": "first-occurrence",
        "utility_range_by_seed": order_ranges,
        "max_utility_range": max_order_range,
        "registered_limit": thresholds.get("max_order_utility_range"),
    })

    semantic_privacy = {
        "profile_held_out": True,
        "diagnostic_distributions": {
            arm: _plain_diagnostic_value(
                row.diagnostic_distributions, "diagnostic distributions"
            )
            for arm, row in grouped["relation_representation"].items()
        },
        "arms": {
            arm: {
                str(seed): {
                    name: value
                    for name, value in row.seed_metrics[seed].items()
                    if name.startswith("privacy_")
                }
                for seed in contract.seeds
            }
            for arm, row in grouped["relation_representation"].items()
        },
    }
    semantic_arm = grouped["metadata_shortcut"]["semantic"]
    lambda_controller = {
        "seed_metrics": {
            str(seed): dict(semantic_arm.seed_metrics[seed])
            for seed in contract.seeds
        },
        "interventions": {
            name: interventions[name]
            for name in ("lambda_zero_identity", "lambda_monotonicity")
        },
    }
    report = {
        "artifact_version": ARTIFACT_VERSION,
        "matched_contract": contract.to_dict(),
        "relation_representation": relation_report,
        "semantic_privacy_transfer": semantic_privacy,
        "context_readout": context_report,
        "action_history": history_report,
        "metadata_shortcut": shortcut_report,
        "lambda_controller": lambda_controller,
        "decision_order": order_report,
        "operational_cost": operational,
        "contamination_boundary": contamination,
        "promotion_verdict": {
            "verdict": verdict,
            "paired_bootstrap": comparisons,
            "local_utility_regression": local_regression,
            "shortcut_invariance_passed": shortcut_invariance_passed,
            "operational_budget_passed": operational_passed,
            "decision_order_budget_passed": order_passed,
            "missing_thresholds": missing_thresholds,
            "registered_thresholds": dict(sorted(thresholds.items())),
        },
    }
    report["artifact_hash"] = stable_hash(report)
    return report


ArmEvaluator = Callable[[str, str, MatchedArmContract], ArmMeasurement]


def run_architecture_spike(
    contract: MatchedArmContract,
    evaluator: ArmEvaluator,
    *,
    registered_thresholds: Mapping[str, float],
    aci_context: bool,
    non_aci_manifest_hash: str | None,
) -> dict[str, Any]:
    """Execute every approved mode through one injected matched evaluator."""

    measurements = []
    for family, arms in APPROVED_ARMS.items():
        for arm in arms:
            row = evaluator(family, arm, contract)
            if not isinstance(row, ArmMeasurement):
                raise TypeError(f"arm evaluator returned an invalid result: {family}:{arm}")
            if row.family != family or row.arm != arm:
                raise ValueError(f"arm evaluator identity differs: {family}:{arm}")
            if row.matched_contract != contract:
                raise ValueError(f"arm evaluator changed the runner contract: {family}:{arm}")
            measurements.append(row)
    return build_architecture_spike_report(
        measurements,
        registered_thresholds=registered_thresholds,
        aci_context=aci_context,
        non_aci_manifest_hash=non_aci_manifest_hash,
    )
