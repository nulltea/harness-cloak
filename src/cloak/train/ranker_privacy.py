"""Profile-held-out pretraining for the Ranker-v2 semantic privacy head."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from cloak.train.profile_count import ProfileCountTargets
from cloak.train.ranker_environment import RankerDocument
from cloak.train.ranker_representation import RankerRepresentationStore


SPLIT_VERSION = "ranker-v2-profile-split-v1"
CHECKPOINT_VERSION = "ranker-v2-semantic-privacy-v1"
METRIC_REPORT_VERSION = "ranker-v2-semantic-privacy-metrics-v1"
SIGMA_MIN = 1e-4
REQUIRED_BASELINES = (
    "authored_position_mode_type",
    "mode_type_only",
    "candidate_only",
    "train_profile_mean",
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrivacyPrediction:
    mu_log_count: torch.Tensor
    sigma_log_count: torch.Tensor


@dataclass(frozen=True)
class PrivacyExample:
    """One admitted level target and its frozen diagnostic representations."""

    decision_id: str
    action_id: str
    profile_id: str
    runtime_type: str
    grounding_status: str
    source_family: str
    authored_position: int
    pair_features: torch.Tensor
    candidate_only: torch.Tensor
    log_count_target: float
    profile_score_target: float


class SemanticPrivacyHead(nn.Module):
    """Privacy-only projection and distribution head over frozen relation features."""

    def __init__(
        self,
        *,
        pair_dim: int,
        projection_dim: int,
        hidden_dim: int,
        count_basis_size: int = 0,
    ):
        super().__init__()
        if min(pair_dim, projection_dim, hidden_dim) <= 0 or count_basis_size < 0:
            raise ValueError("privacy head dimensions must be positive")
        self.pair_dim = int(pair_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.count_basis_size = int(count_basis_size)
        self.privacy_projection = nn.Linear(
            pair_dim + count_basis_size, projection_dim
        )
        self.privacy_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        pair_features: torch.Tensor,
        *,
        count_basis: torch.Tensor | None = None,
    ) -> PrivacyPrediction:
        if pair_features.ndim != 2 or pair_features.shape[1] != self.pair_dim:
            raise ValueError(
                f"pair features must have shape [batch, {self.pair_dim}]"
            )
        if self.count_basis_size:
            if count_basis is None:
                count_basis = torch.zeros(
                    pair_features.shape[0],
                    dtype=torch.long,
                    device=pair_features.device,
                )
            if (
                count_basis.ndim != 1
                or count_basis.shape[0] != pair_features.shape[0]
                or count_basis.dtype != torch.long
                or torch.any(count_basis < 0)
                or torch.any(count_basis >= self.count_basis_size)
            ):
                raise ValueError("count basis indices are invalid")
            frozen_basis = F.one_hot(
                count_basis, num_classes=self.count_basis_size
            ).to(dtype=pair_features.dtype)
            pair_features = torch.cat([pair_features, frozen_basis], dim=1)
        elif count_basis is not None:
            raise ValueError("count basis was supplied to a basis-free privacy head")
        projected = self.privacy_projection(pair_features)
        mu_raw, sigma_raw = self.privacy_head(projected).chunk(2, dim=-1)
        return PrivacyPrediction(
            mu_log_count=F.softplus(mu_raw).squeeze(-1),
            sigma_log_count=SIGMA_MIN + F.softplus(sigma_raw).squeeze(-1),
        )


class MetadataPrivacyHead(nn.Module):
    """Diagnostic shortcut baseline over authored position, mode, and runtime type."""

    def __init__(
        self,
        *,
        runtime_types: Sequence[str],
        projection_dim: int,
        hidden_dim: int,
        include_authored_position: bool,
    ):
        super().__init__()
        normalized_types = tuple(sorted(set(str(value) for value in runtime_types)))
        if not normalized_types:
            raise ValueError("metadata baseline requires runtime types")
        self.runtime_types = normalized_types
        self.runtime_type_indices = {
            value: index for index, value in enumerate(normalized_types)
        }
        self.mode_indices = {"level": 0, "keep": 1, "placeholder": 2}
        self.include_authored_position = bool(include_authored_position)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        feature_dim = len(self.mode_indices) + len(self.runtime_types)
        if self.include_authored_position:
            feature_dim += 1
        self.privacy_projection = nn.Linear(feature_dim, projection_dim)
        self.privacy_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        authored_positions: torch.Tensor,
        modes: Sequence[str],
        runtime_types: Sequence[str],
    ) -> PrivacyPrediction:
        if authored_positions.ndim != 1:
            raise ValueError("authored positions must be a vector")
        if not (len(modes) == len(runtime_types) == authored_positions.shape[0]):
            raise ValueError("metadata baseline inputs are misaligned")
        features = authored_positions.new_zeros(
            (len(modes), len(self.mode_indices) + len(self.runtime_types))
        )
        for index, (mode, runtime_type) in enumerate(
            zip(modes, runtime_types, strict=True)
        ):
            if mode not in self.mode_indices:
                raise ValueError(f"unsupported action mode: {mode}")
            if runtime_type not in self.runtime_type_indices:
                raise ValueError(f"unsupported runtime type: {runtime_type}")
            features[index, self.mode_indices[mode]] = 1.0
            features[
                index,
                len(self.mode_indices) + self.runtime_type_indices[runtime_type],
            ] = 1.0
        if self.include_authored_position:
            position_feature = torch.log1p(authored_positions.clamp_min(0)).unsqueeze(1)
            features = torch.cat([features, position_feature], dim=1)
        projected = self.privacy_projection(features)
        mu_raw, sigma_raw = self.privacy_head(projected).chunk(2, dim=-1)
        return PrivacyPrediction(
            F.softplus(mu_raw).squeeze(-1),
            SIGMA_MIN + F.softplus(sigma_raw).squeeze(-1),
        )


def build_neural_privacy_models(
    *,
    pair_dim: int,
    candidate_dim: int,
    runtime_types: Sequence[str],
    projection_dim: int,
    hidden_dim: int,
    count_basis_size: int = 0,
) -> dict[str, nn.Module]:
    """Create the semantic model and equal-head-budget neural baselines."""
    return {
        "semantic": SemanticPrivacyHead(
            pair_dim=pair_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            count_basis_size=count_basis_size,
        ),
        "authored_position_mode_type": MetadataPrivacyHead(
            runtime_types=runtime_types,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            include_authored_position=True,
        ),
        "mode_type_only": MetadataPrivacyHead(
            runtime_types=runtime_types,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            include_authored_position=False,
        ),
        "candidate_only": SemanticPrivacyHead(
            pair_dim=candidate_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            count_basis_size=count_basis_size,
        ),
    }


@dataclass(frozen=True)
class TrainProfileMeanBaseline:
    """Stable type-conditioned train mean with no held-out profile identity table."""

    runtime_type_means: Mapping[str, float]
    runtime_type_scales: Mapping[str, float]
    global_mean: float
    global_scale: float

    @classmethod
    def fit(cls, rows: Sequence[PrivacyExample]) -> "TrainProfileMeanBaseline":
        if not rows:
            raise ValueError("train-profile mean requires training rows")
        by_type: dict[str, list[float]] = defaultdict(list)
        targets = []
        for row in rows:
            target = float(row.log_count_target)
            if not math.isfinite(target):
                raise ValueError("train-profile mean received a nonfinite target")
            targets.append(target)
            by_type[row.runtime_type].append(target)

        def moments(values: Sequence[float]) -> tuple[float, float]:
            average = sum(values) / len(values)
            variance = sum((value - average) ** 2 for value in values) / len(values)
            return average, max(SIGMA_MIN, math.sqrt(variance))

        global_mean, global_scale = moments(targets)
        per_type = {
            runtime_type: moments(values)
            for runtime_type, values in sorted(by_type.items())
        }
        return cls(
            runtime_type_means={key: value[0] for key, value in per_type.items()},
            runtime_type_scales={key: value[1] for key, value in per_type.items()},
            global_mean=global_mean,
            global_scale=global_scale,
        )

    def predict(self, rows: Sequence[PrivacyExample]) -> PrivacyPrediction:
        means = [
            self.runtime_type_means.get(row.runtime_type, self.global_mean)
            for row in rows
        ]
        scales = [
            self.runtime_type_scales.get(row.runtime_type, self.global_scale)
            for row in rows
        ]
        return PrivacyPrediction(
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(scales, dtype=torch.float32),
        )


def _validate_prediction(prediction: PrivacyPrediction, size: int) -> None:
    if (
        prediction.mu_log_count.ndim != 1
        or prediction.sigma_log_count.ndim != 1
        or prediction.mu_log_count.shape[0] != size
        or prediction.sigma_log_count.shape[0] != size
    ):
        raise ValueError("privacy prediction tensors must be aligned vectors")
    if not torch.isfinite(prediction.mu_log_count).all():
        raise ValueError("privacy predicted means must be finite")
    if (
        not torch.isfinite(prediction.sigma_log_count).all()
        or torch.any(prediction.sigma_log_count < SIGMA_MIN)
    ):
        raise ValueError("privacy predicted scales must be finite and bounded")


def profile_normalize_predictions(
    prediction: PrivacyPrediction,
    modes: Sequence[str],
) -> torch.Tensor:
    """Normalize one complete decision menu and overwrite its fixed endpoints."""
    modes = tuple(str(mode) for mode in modes)
    _validate_prediction(prediction, len(modes))
    if (
        modes.count("keep") != 1
        or modes.count("placeholder") != 1
        or not any(mode == "level" for mode in modes)
        or any(mode not in {"level", "keep", "placeholder"} for mode in modes)
    ):
        raise ValueError("privacy normalization requires one complete supported menu")

    result = torch.empty_like(prediction.mu_log_count)
    level_indices = [index for index, mode in enumerate(modes) if mode == "level"]
    if len(level_indices) == 1:
        result[level_indices[0]] = 1.0
    else:
        level_means = prediction.mu_log_count[level_indices]
        denominator = level_means.max()
        if denominator <= 0:
            raise ValueError("predicted multi-level menu has zero normalization denominator")
        result[level_indices] = torch.clamp(level_means / denominator, 0.0, 1.0)
    result[modes.index("keep")] = 0.0
    result[modes.index("placeholder")] = 1.0
    return result


def _validated_profile_slices(
    profile_slices: Sequence[slice], size: int
) -> tuple[slice, ...]:
    normalized = []
    cursor = 0
    for profile_slice in profile_slices:
        if not isinstance(profile_slice, slice):
            raise ValueError("profile_slices must contain slices")
        start, stop, step = profile_slice.indices(size)
        if step != 1 or start != cursor or stop <= start:
            raise ValueError("profile_slices must be a contiguous nonempty partition")
        normalized.append(slice(start, stop))
        cursor = stop
    if cursor != size:
        raise ValueError("profile_slices must cover every target exactly once")
    return tuple(normalized)


def _normalize_level_means(
    means: torch.Tensor, profile_slices: Sequence[slice]
) -> torch.Tensor:
    normalized = torch.empty_like(means)
    for profile_slice in profile_slices:
        profile_means = means[profile_slice]
        if profile_means.numel() == 1:
            normalized[profile_slice] = 1.0
            continue
        denominator = profile_means.max()
        if denominator <= 0:
            raise ValueError("predicted multi-level profile has zero denominator")
        normalized[profile_slice] = torch.clamp(
            profile_means / denominator, 0.0, 1.0
        )
    return normalized


def privacy_training_loss(
    prediction: PrivacyPrediction,
    log_count_targets: torch.Tensor,
    profile_slices: Sequence[slice],
    profile_score_targets: torch.Tensor,
    *,
    rho: float,
    gamma: float,
) -> Mapping[str, torch.Tensor]:
    """Return level-only distribution, ordering, and profile-calibration losses."""
    size = int(log_count_targets.numel())
    _validate_prediction(prediction, size)
    if (
        log_count_targets.ndim != 1
        or profile_score_targets.ndim != 1
        or profile_score_targets.shape != log_count_targets.shape
        or not torch.isfinite(log_count_targets).all()
        or not torch.isfinite(profile_score_targets).all()
    ):
        raise ValueError("privacy targets must be aligned finite vectors")
    if rho < 0 or gamma < 0 or not math.isfinite(rho) or not math.isfinite(gamma):
        raise ValueError("privacy loss weights must be finite and nonnegative")
    slices = _validated_profile_slices(profile_slices, size)

    mu = prediction.mu_log_count
    sigma = prediction.sigma_log_count
    nll = (
        torch.log(sigma)
        + 0.5 * ((log_count_targets - mu) / sigma).square()
        + 0.5 * math.log(2 * math.pi)
    ).mean()

    rank_terms = []
    for profile_slice in slices:
        indices = range(profile_slice.start, profile_slice.stop)
        for left in indices:
            for right in range(left + 1, profile_slice.stop):
                target_difference = log_count_targets[right] - log_count_targets[left]
                if target_difference == 0:
                    continue
                direction = torch.sign(target_difference)
                predicted_difference = mu[right] - mu[left]
                rank_terms.append(F.softplus(-direction * predicted_difference))
    pairwise_rank = (
        torch.stack(rank_terms).mean() if rank_terms else mu.sum() * 0.0
    )

    normalized = _normalize_level_means(mu, slices)
    profile_huber = F.smooth_l1_loss(normalized, profile_score_targets)
    total = nll + float(rho) * pairwise_rank + float(gamma) * profile_huber
    return {
        "nll": nll,
        "pairwise_rank": pairwise_rank,
        "profile_huber": profile_huber,
        "total": total,
    }


def _row_value(row: Any, name: str) -> str:
    value = row.get(name) if isinstance(row, Mapping) else getattr(row, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"split row has invalid {name}")
    return value


def _split_counts(size: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    counts = [1, 1, 1]
    for _ in range(size - 3):
        split = max(
            range(3),
            key=lambda index: (ratios[index] * size - counts[index], -index),
        )
        counts[split] += 1
    return tuple(counts)


def build_grouped_profile_split(
    rows: Sequence[Any],
    *,
    seed: int,
    source_hash: str,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Any]:
    """Build a seeded content-addressed profile split stratified by runtime type."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("split seed must be an integer")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError("split source hash is required")
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    total_ratio = sum(ratios)
    normalized_ratios = tuple(float(value) / total_ratio for value in ratios)

    profile_types: dict[str, str] = {}
    for row in rows:
        profile_id = _row_value(row, "profile_id")
        runtime_type = _row_value(row, "runtime_type")
        previous = profile_types.setdefault(profile_id, runtime_type)
        if previous != runtime_type:
            raise ValueError(f"profile crosses runtime types: {profile_id}")
    if not profile_types:
        raise ValueError("cannot split an empty profile set")

    by_type: dict[str, list[str]] = defaultdict(list)
    for profile_id, runtime_type in profile_types.items():
        by_type[runtime_type].append(profile_id)
    split_names = ("train", "dev", "test")
    assigned: dict[str, list[str]] = {name: [] for name in split_names}
    for runtime_type, profile_ids in sorted(by_type.items()):
        if len(profile_ids) < 3:
            raise ValueError(
                f"runtime type {runtime_type} requires at least three profiles"
            )
        ordered = sorted(
            profile_ids,
            key=lambda profile_id: hashlib.sha256(
                f"{seed}\0{runtime_type}\0{profile_id}".encode("utf-8")
            ).hexdigest(),
        )
        counts = _split_counts(len(ordered), normalized_ratios)
        cursor = 0
        for name, count in zip(split_names, counts, strict=True):
            assigned[name].extend(ordered[cursor:cursor + count])
            cursor += count

    profiles = {
        name: sorted(profile_ids) for name, profile_ids in assigned.items()
    }
    runtime_type_counts = {
        name: {
            runtime_type: sum(
                profile_types[profile_id] == runtime_type
                for profile_id in profiles[name]
            )
            for runtime_type in sorted(by_type)
        }
        for name in split_names
    }
    manifest = {
        "artifact_version": SPLIT_VERSION,
        "seed": seed,
        "source_hash": source_hash,
        "ratios": dict(zip(split_names, normalized_ratios, strict=True)),
        "profiles": profiles,
        "runtime_type_counts": runtime_type_counts,
    }
    manifest["artifact_hash"] = _stable_hash(manifest)
    return manifest


def _rank_values(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        average_rank = (cursor + stop - 1) / 2.0
        for ordered_index in order[cursor:stop]:
            ranks[ordered_index] = average_rank
        cursor = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = _rank_values(left)
    right_ranks = _rank_values(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return None
    correlation = numerator / (left_scale * right_scale)
    if math.isclose(correlation, 1.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(correlation, -1.0, abs_tol=1e-12):
        return -1.0
    return max(-1.0, min(1.0, correlation))


def _profile_normalized_values(
    examples: Sequence[PrivacyExample], means: Sequence[float]
) -> list[float]:
    by_profile: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        by_profile[row.decision_id].append(index)
    normalized = [0.0] * len(examples)
    for indices in by_profile.values():
        if len(indices) == 1:
            normalized[indices[0]] = 1.0
            continue
        denominator = max(means[index] for index in indices)
        if denominator <= 0:
            for index in indices:
                normalized[index] = 0.0
            continue
        for index in indices:
            normalized[index] = min(1.0, max(0.0, means[index] / denominator))
    return normalized


def _metric_subset(
    examples: Sequence[PrivacyExample],
    prediction: PrivacyPrediction,
    normalized: Sequence[float],
    selected_indices: Sequence[int],
) -> dict[str, float | None]:
    indices = tuple(selected_indices)
    if not indices:
        raise ValueError("cannot evaluate an empty privacy metric subset")
    means = [float(prediction.mu_log_count[index]) for index in indices]
    scales = [float(prediction.sigma_log_count[index]) for index in indices]
    targets = [float(examples[index].log_count_target) for index in indices]
    errors = [abs(mean - target) for mean, target in zip(means, targets, strict=True)]
    nll_values = [
        math.log(scale)
        + 0.5 * ((target - mean) / scale) ** 2
        + 0.5 * math.log(2 * math.pi)
        for mean, scale, target in zip(means, scales, targets, strict=True)
    ]

    pair_correct = []
    profile_correlations = []
    profile_regrets = []
    by_profile: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_profile[examples[index].decision_id].append(index)
    for profile_indices in by_profile.values():
        for offset, left in enumerate(profile_indices):
            for right in profile_indices[offset + 1:]:
                target_difference = (
                    examples[right].log_count_target
                    - examples[left].log_count_target
                )
                if target_difference == 0:
                    continue
                predicted_difference = (
                    float(prediction.mu_log_count[right])
                    - float(prediction.mu_log_count[left])
                )
                pair_correct.append(
                    float(predicted_difference * target_difference > 0)
                )
        correlation = _spearman(
            [float(prediction.mu_log_count[index]) for index in profile_indices],
            [examples[index].log_count_target for index in profile_indices],
        )
        if correlation is not None:
            profile_correlations.append(correlation)
        selected = max(profile_indices, key=lambda index: (normalized[index], -index))
        best_target = max(
            examples[index].profile_score_target for index in profile_indices
        )
        profile_regrets.append(
            best_target - examples[selected].profile_score_target
        )

    sorted_errors = sorted(errors)
    middle = len(sorted_errors) // 2
    median_error = (
        sorted_errors[middle]
        if len(sorted_errors) % 2
        else (sorted_errors[middle - 1] + sorted_errors[middle]) / 2
    )
    # exp is monotone, so exponentiating only the central sorted log errors is
    # order-identical and cannot overflow on one diverged prediction.
    median_multiplicative = (
        math.exp(sorted_errors[middle])
        if len(sorted_errors) % 2
        else (math.exp(sorted_errors[middle - 1]) + math.exp(sorted_errors[middle])) / 2
    )
    return {
        "nll": sum(nll_values) / len(nll_values),
        "median_absolute_log_error": median_error,
        "median_multiplicative_error": median_multiplicative,
        "interval_95_coverage": sum(
            error <= 1.96 * scale
            for error, scale in zip(errors, scales, strict=True)
        ) / len(errors),
        "within_menu_pairwise_accuracy": (
            sum(pair_correct) / len(pair_correct) if pair_correct else None
        ),
        "spearman": (
            sum(profile_correlations) / len(profile_correlations)
            if profile_correlations else None
        ),
        "profile_relative_calibration_error": sum(
            abs(normalized[index] - examples[index].profile_score_target)
            for index in indices
        ) / len(indices),
        "selected_action_regret": sum(profile_regrets) / len(profile_regrets),
    }


def evaluate_privacy_predictions(
    examples: Sequence[PrivacyExample], prediction: PrivacyPrediction
) -> dict[str, Any]:
    """Report level-only held-out metrics overall and by required provenance strata."""
    examples = tuple(examples)
    _validate_prediction(prediction, len(examples))
    if not examples:
        raise ValueError("cannot evaluate empty privacy predictions")
    means = [float(value) for value in prediction.mu_log_count]
    normalized = _profile_normalized_values(examples, means)
    all_indices = tuple(range(len(examples)))

    def stratify(field_name: str) -> dict[str, dict[str, float | None]]:
        values = sorted({str(getattr(row, field_name)) for row in examples})
        return {
            value: _metric_subset(
                examples,
                prediction,
                normalized,
                [
                    index for index, row in enumerate(examples)
                    if str(getattr(row, field_name)) == value
                ],
            )
            for value in values
        }

    return {
        "overall": _metric_subset(
            examples, prediction, normalized, all_indices
        ),
        "by_runtime_type": stratify("runtime_type"),
        "by_grounding_status": stratify("grounding_status"),
        "by_source_family": stratify("source_family"),
    }


@dataclass(frozen=True)
class PrivacyCheckpointContract:
    environment_hash: str
    profile_target_artifact_hash: str
    representation_manifest_hash: str
    encoder_revision: str
    split_manifest_hash: str
    pair_dim: int
    projection_dim: int
    hidden_dim: int
    count_basis_size: int
    count_basis_categories: tuple[str, ...]
    rho: float
    gamma: float
    seeds: tuple[int, int, int]
    training_seed: int
    metric_report_hash: str

    def __post_init__(self) -> None:
        if min(self.pair_dim, self.projection_dim, self.hidden_dim) <= 0:
            raise ValueError("privacy checkpoint dimensions must be positive")
        if self.count_basis_size != len(self.count_basis_categories):
            raise ValueError("privacy checkpoint count basis size mismatch")
        if self.count_basis_size and self.count_basis_categories[0] != "<unknown>":
            raise ValueError("privacy checkpoint count basis lacks unknown category")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("privacy checkpoint requires three distinct seeds")
        if self.training_seed not in self.seeds:
            raise ValueError("privacy checkpoint training seed is not orchestrated")
        if self.rho < 0 or self.gamma < 0:
            raise ValueError("privacy checkpoint loss weights must be nonnegative")
        for name in (
            "environment_hash",
            "profile_target_artifact_hash",
            "representation_manifest_hash",
            "encoder_revision",
            "split_manifest_hash",
            "metric_report_hash",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"privacy checkpoint contract lacks {name}")


def save_privacy_checkpoint(
    path: Path,
    model: SemanticPrivacyHead,
    contract: PrivacyCheckpointContract,
) -> None:
    """Persist a weights-only-compatible semantic privacy checkpoint."""
    model_contract = (
        model.pair_dim,
        model.projection_dim,
        model.hidden_dim,
        model.count_basis_size,
    )
    declared_contract = (
        contract.pair_dim,
        contract.projection_dim,
        contract.hidden_dim,
        contract.count_basis_size,
    )
    if model_contract != declared_contract:
        raise ValueError("privacy checkpoint model contract mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "contract": asdict(contract),
        "model_state_dict": model.state_dict(),
    }, path)


def load_privacy_checkpoint(
    path: Path,
    model: SemanticPrivacyHead,
    expected_contract: PrivacyCheckpointContract,
) -> Mapping[str, Any]:
    """Validate every frozen binding before mutating the supplied model."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("privacy checkpoint must be a mapping")
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported privacy checkpoint version")
    expected = asdict(expected_contract)
    if checkpoint.get("contract") != expected:
        raise ValueError("privacy checkpoint contract mismatch")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("privacy checkpoint is missing model state")
    model_contract = (
        model.pair_dim,
        model.projection_dim,
        model.hidden_dim,
        model.count_basis_size,
    )
    expected_model_contract = (
        expected_contract.pair_dim,
        expected_contract.projection_dim,
        expected_contract.hidden_dim,
        expected_contract.count_basis_size,
    )
    if model_contract != expected_model_contract:
        raise ValueError("privacy checkpoint model contract mismatch")
    expected_state = model.state_dict()
    if set(state_dict) != set(expected_state):
        raise ValueError("privacy checkpoint state key mismatch")
    for name, expected_tensor in expected_state.items():
        supplied = state_dict[name]
        if (
            not isinstance(supplied, torch.Tensor)
            or supplied.shape != expected_tensor.shape
            or supplied.dtype != expected_tensor.dtype
        ):
            raise ValueError(f"privacy checkpoint state shape mismatch: {name}")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


@dataclass(frozen=True)
class CountBasisVocabulary:
    """Frozen train-split vocabulary for optional provenance/source-family inputs."""

    categories: tuple[str, ...]

    @classmethod
    def fit(cls, rows: Sequence[PrivacyExample]) -> "CountBasisVocabulary":
        observed = sorted({
            f"{row.grounding_status}|{row.source_family}" for row in rows
        })
        return cls(("<unknown>", *observed))

    def encode(self, rows: Sequence[PrivacyExample]) -> torch.Tensor:
        indices = {value: index for index, value in enumerate(self.categories)}
        return torch.tensor([
            indices.get(
                f"{row.grounding_status}|{row.source_family}", 0
            )
            for row in rows
        ], dtype=torch.long)


def load_privacy_examples(
    documents: Mapping[str, RankerDocument],
    target_artifact: Mapping,
    representation_store: RankerRepresentationStore,
) -> tuple[PrivacyExample, ...]:
    """Join admitted Task 1 level targets to frozen Task 2 representations."""
    targets = ProfileCountTargets.from_artifact(target_artifact)
    actions = {}
    for document in documents.values():
        for decision in document.policy_decisions:
            for action in decision.actions:
                if action.action_id in actions:
                    raise ValueError(f"duplicate environment action: {action.action_id}")
                actions[action.action_id] = action

    examples = []
    for target in targets.target_rows(eligible_only=True):
        if target.mode != "level":
            continue
        if target.log_count is None:
            raise ValueError(f"eligible level target lacks log count: {target.action_id}")
        try:
            action = actions[target.action_id]
        except KeyError as exc:
            raise ValueError(
                f"target action is absent from environment: {target.action_id}"
            ) from exc
        if action.mode != "level" or action.authored_level_index is None:
            raise ValueError(f"level target has invalid authored action: {target.action_id}")
        relation = representation_store.relation(
            target.decision_id, target.action_id
        )
        examples.append(PrivacyExample(
            decision_id=target.decision_id,
            action_id=target.action_id,
            profile_id=target.profile_id,
            runtime_type=target.runtime_type,
            grounding_status=target.grounding_status or "unknown",
            source_family=target.source_family or "unknown",
            authored_position=action.authored_level_index,
            pair_features=relation.pair.detach().to(dtype=torch.float32, device="cpu"),
            candidate_only=relation.candidate_only.detach().to(
                dtype=torch.float32, device="cpu"
            ),
            log_count_target=float(target.log_count),
            profile_score_target=float(target.profile_score),
        ))
    if not examples:
        raise ValueError("profile target artifact has no eligible level rows")
    return tuple(sorted(examples, key=lambda row: (
        row.profile_id,
        row.decision_id,
        row.authored_position,
        row.action_id,
    )))


def _ordered_examples(
    examples: Sequence[PrivacyExample],
) -> tuple[tuple[PrivacyExample, ...], tuple[slice, ...]]:
    ordered = tuple(sorted(examples, key=lambda row: (
        row.decision_id, row.authored_position, row.action_id,
    )))
    slices = []
    cursor = 0
    while cursor < len(ordered):
        stop = cursor + 1
        while stop < len(ordered) and ordered[stop].decision_id == ordered[cursor].decision_id:
            stop += 1
        slices.append(slice(cursor, stop))
        cursor = stop
    return ordered, tuple(slices)


def _model_prediction(
    name: str,
    model: nn.Module,
    examples: Sequence[PrivacyExample],
    basis: CountBasisVocabulary | None,
) -> PrivacyPrediction:
    if name in {"semantic", "candidate_only"}:
        feature_name = "pair_features" if name == "semantic" else "candidate_only"
        features = torch.stack([getattr(row, feature_name) for row in examples])
        indices = basis.encode(examples) if basis is not None else None
        return model(features, count_basis=indices)
    positions = torch.tensor(
        [row.authored_position for row in examples], dtype=torch.float32
    )
    return model(
        positions,
        ["level"] * len(examples),
        [row.runtime_type for row in examples],
    )


def train_privacy_seed(
    examples: Sequence[PrivacyExample],
    split_manifest: Mapping,
    *,
    seed: int,
    projection_dim: int,
    hidden_dim: int,
    rho: float,
    gamma: float,
    learning_rate: float,
    max_steps: int,
    use_count_basis: bool,
) -> tuple[SemanticPrivacyHead, dict[str, Any]]:
    """Train one CPU seed and return profile-held-out dev/test reports."""
    if max_steps <= 0 or learning_rate <= 0:
        raise ValueError("privacy training steps and learning rate must be positive")
    profiles = split_manifest.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("privacy training requires a split manifest")
    split_rows = {
        split: tuple(row for row in examples if row.profile_id in set(profile_ids))
        for split, profile_ids in profiles.items()
    }
    if set(split_rows) != {"train", "dev", "test"} or any(
        not values for values in split_rows.values()
    ):
        raise ValueError("privacy split produced an empty partition")
    train_rows, train_slices = _ordered_examples(split_rows["train"])
    pair_dim = int(train_rows[0].pair_features.numel())
    candidate_dim = int(train_rows[0].candidate_only.numel())
    if any(row.pair_features.shape != train_rows[0].pair_features.shape for row in examples):
        raise ValueError("privacy pair feature dimensions are inconsistent")
    if any(row.candidate_only.shape != train_rows[0].candidate_only.shape for row in examples):
        raise ValueError("candidate-only feature dimensions are inconsistent")
    basis = CountBasisVocabulary.fit(train_rows) if use_count_basis else None
    count_basis_size = len(basis.categories) if basis is not None else 0

    torch.manual_seed(seed)
    models = build_neural_privacy_models(
        pair_dim=pair_dim,
        candidate_dim=candidate_dim,
        runtime_types=sorted({row.runtime_type for row in examples}),
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        count_basis_size=count_basis_size,
    )
    log_targets = torch.tensor(
        [row.log_count_target for row in train_rows], dtype=torch.float32
    )
    score_targets = torch.tensor(
        [row.profile_score_target for row in train_rows], dtype=torch.float32
    )
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for _ in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = _model_prediction(name, model, train_rows, basis)
            losses = privacy_training_loss(
                prediction,
                log_targets,
                train_slices,
                score_targets,
                rho=rho,
                gamma=gamma,
            )
            losses["total"].backward()
            optimizer.step()

    train_mean = TrainProfileMeanBaseline.fit(train_rows)
    split_reports = {}
    for split in ("dev", "test"):
        held_out, _ = _ordered_examples(split_rows[split])
        evaluated = {}
        with torch.inference_mode():
            for name, model in models.items():
                model.eval()
                evaluated[name] = evaluate_privacy_predictions(
                    held_out, _model_prediction(name, model, held_out, basis)
                )
        baseline_reports = {
            name: evaluated[name] for name in REQUIRED_BASELINES[:-1]
        }
        baseline_reports["train_profile_mean"] = evaluate_privacy_predictions(
            held_out, train_mean.predict(held_out)
        )
        split_reports[split] = {
            "semantic": evaluated["semantic"],
            "baselines": baseline_reports,
        }
    report = {
        "seed": seed,
        "profile_held_out": True,
        "max_steps": max_steps,
        "count_basis_categories": list(basis.categories) if basis else [],
        "splits": split_reports,
    }
    return models["semantic"], report
