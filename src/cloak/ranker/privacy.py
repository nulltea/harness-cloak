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

from cloak.ranker.profile_count import ProfileCountTargets
from cloak.ranker.environment import RankerDocument
from cloak.ranker.representation import RankerRepresentationStore


SPLIT_VERSION = "ranker-v2-profile-split-v1"
CHECKPOINT_VERSION = "ranker-v2-semantic-privacy-v2"
METRIC_REPORT_VERSION = "ranker-v2-semantic-privacy-metrics-v2"
SIGMA_FIXED_MIN = 0.3
SIGMA_FIXED_MAX = 3.0
SIGMA_MIN = SIGMA_FIXED_MIN  # Compatibility name for downstream validation.
FEATURE_SCHEMA = "type-source-candidate-hadamard-v1"
PREDICTION_TIE_TOLERANCE = 1e-6
REQUIRED_BASELINES = (
    "authored_position_mode_type",
    "mode_type_only",
    "candidate_only",
    "train_profile_mean",
)
PRIVACY_ATTRIBUTION_ARMS = ("candidate_projected", "candidate_native", "joint_candidate", "joint_candidate_source", "joint_candidate_source_hadamard", "joint_full", "bi_encoder")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrivacyPrediction:
    mu_log_count: torch.Tensor
    sigma_log_count: torch.Tensor
    standardized_mean: torch.Tensor | None = None


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
    source_identity: str = ""
    candidate_identity: str = ""
    independent_pair: torch.Tensor | None = None


def validate_privacy_signal_for_policy(
    privacy_signal: Mapping[str, Any],
    *,
    learned_contract: "PrivacyCheckpointContract | None" = None,
    allow_development_override: bool = False,
) -> None:
    """Admit strict direct counts or a learned head with promotion evidence."""
    if not isinstance(privacy_signal, Mapping):
        raise ValueError("privacy signal provenance must be a mapping")
    kind = privacy_signal.get("kind")
    if kind == "direct-count":
        required = ("targets_artifact_hash", "environment_hash")
        if any(
            not isinstance(privacy_signal.get(name), str)
            or not privacy_signal[name]
            for name in required
        ):
            raise ValueError("direct-count privacy signal provenance is incomplete")
        return
    if kind == "learned-head":
        if learned_contract is None:
            raise ValueError("learned privacy signal lacks its checkpoint contract")
        learned_contract.validate_for_policy(
            allow_development_override=allow_development_override
        )
        return
    raise ValueError(f"unsupported privacy signal kind: {kind}")


class DirectCountPrivacyProvider(nn.Module):
    """Strict exact lookup of profile-relative scores from grounded-count targets."""

    def __init__(self, targets_artifact: Mapping[str, Any]):
        super().__init__()
        if not isinstance(targets_artifact, Mapping):
            raise ValueError("direct-count targets artifact must be a mapping")
        artifact_hash = targets_artifact.get("artifact_hash")
        if not isinstance(artifact_hash, str) or not artifact_hash:
            raise ValueError("direct-count targets artifact lacks artifact_hash")
        environment_hash = targets_artifact.get("environment_hash")
        if not isinstance(environment_hash, str) or not environment_hash:
            raise ValueError("direct-count targets artifact lacks environment_hash")
        raw_targets = targets_artifact.get("action_targets")
        if not isinstance(raw_targets, Mapping):
            raise ValueError("direct-count targets artifact lacks action_targets")

        scores: dict[tuple[str, str], float] = {}
        for action_key, raw_row in raw_targets.items():
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"direct-count target row is invalid: {action_key}")
            decision_id = raw_row.get("decision_id")
            action_id = raw_row.get("action_id")
            profile_id = raw_row.get("profile_id")
            mode = raw_row.get("mode")
            if (
                not isinstance(decision_id, str)
                or not decision_id
                or not isinstance(action_id, str)
                or not action_id
                or not isinstance(profile_id, str)
                or action_id != str(action_key)
                or mode not in {"level", "keep", "placeholder"}
            ):
                missing = (
                    "profile_id"
                    if not isinstance(profile_id, str)
                    else "identity"
                )
                raise ValueError(
                    f"direct-count target {missing} is invalid: {action_key}"
                )
            profile_score = raw_row.get("profile_score")
            if (
                isinstance(profile_score, bool)
                or not isinstance(profile_score, int | float)
                or not math.isfinite(float(profile_score))
            ):
                raise ValueError(
                    f"direct-count target lacks profile_score: "
                    f"{decision_id}:{action_id}"
                )
            if mode == "level":
                log_count = raw_row.get("log_count")
                if (
                    isinstance(log_count, bool)
                    or not isinstance(log_count, int | float)
                    or not math.isfinite(float(log_count))
                ):
                    raise ValueError(
                        f"direct-count level target lacks log_count: "
                        f"{decision_id}:{action_id}"
                    )
                grounding_status = raw_row.get("grounding_status")
                if (
                    not isinstance(grounding_status, str)
                    or not grounding_status
                ):
                    raise ValueError(
                        f"direct-count level target lacks grounding_status: "
                        f"{decision_id}:{action_id}"
                    )
            pair = (decision_id, action_id)
            if pair in scores:
                raise ValueError(
                    f"duplicate direct-count target: {decision_id}:{action_id}"
                )
            scores[pair] = float(profile_score)

        self._scores = scores
        self.privacy_signal = {
            "kind": "direct-count",
            "targets_artifact_hash": artifact_hash,
            "environment_hash": environment_hash,
        }
        validate_privacy_signal_for_policy(self.privacy_signal)

    def has_targets(
        self, decision_id: str, action_ids: Sequence[str]
    ) -> bool:
        """Report whether every action of the menu has a direct-count score."""
        if isinstance(action_ids, (str, bytes)) or not action_ids:
            raise ValueError("direct-count coverage check requires action ids")
        return all(
            (str(decision_id), str(action_id)) in self._scores
            for action_id in action_ids
        )

    def forward(
        self,
        decision_ids: Sequence[str],
        action_ids: Sequence[str],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if (
            isinstance(decision_ids, (str, bytes))
            or isinstance(action_ids, (str, bytes))
            or len(decision_ids) != len(action_ids)
        ):
            raise ValueError("direct-count lookup keys must be aligned sequences")
        values = []
        for decision_id, action_id in zip(
            decision_ids, action_ids, strict=True
        ):
            pair = (str(decision_id), str(action_id))
            try:
                values.append(self._scores[pair])
            except KeyError as exc:
                raise ValueError(
                    f"unknown direct-count target: {pair[0]}:{pair[1]}"
                ) from exc
        return torch.tensor(values, dtype=dtype, device=device)


def _head_parameter_count(input_dim: int, projection_dim: int) -> int:
    if projection_dim == 1:
        return input_dim + 1
    return input_dim * 32 + 32 + 32 + 1


def _make_fixed_projection(
    native_dim: int, common_dim: int, *, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        native_dim, common_dim, generator=generator, dtype=torch.float32
    )
    return projection / projection.square().sum(dim=0, keepdim=True).sqrt().clamp_min(
        1e-6
    )


class SemanticPrivacyHead(nn.Module):
    """Fixed-sigma privacy mean head over train-normalized frozen relation blocks."""

    def __init__(
        self,
        *,
        pair_dim: int,
        projection_dim: int,
        hidden_dim: int,
        count_basis_size: int = 0,
        select_clean_blocks: bool | None = None,
    ):
        super().__init__()
        if pair_dim <= 0 or projection_dim not in {1, 32} or hidden_dim != 0:
            raise ValueError(
                "fixed-sigma head requires projection_dim 1 or 32 and hidden_dim 0"
            )
        if count_basis_size < 0:
            raise ValueError("count basis size must be nonnegative")
        self.pair_dim = int(pair_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.count_basis_size = int(count_basis_size)
        self.select_clean_blocks = (
            pair_dim % 5 == 0
            if select_clean_blocks is None
            else bool(select_clean_blocks)
        )
        if self.select_clean_blocks and pair_dim % 5:
            raise ValueError("clean pair block selection requires five equal blocks")
        self.block_dim = pair_dim // 5 if self.select_clean_blocks else pair_dim
        self.normalized_feature_dim = (
            4 * self.block_dim if self.select_clean_blocks else pair_dim
        )
        self.budget_feature_dim = self.normalized_feature_dim
        input_dim = self.budget_feature_dim + count_basis_size
        self.native_trainable_parameters = _head_parameter_count(
            input_dim, projection_dim
        )
        if projection_dim == 1:
            self.privacy_projection = nn.Linear(input_dim, 1)
            self.privacy_head = nn.Identity()
        else:
            self.privacy_projection = nn.Linear(input_dim, 32)
            self.privacy_head = nn.Sequential(nn.GELU(), nn.Linear(32, 1))
        self.register_buffer("feature_mean", torch.zeros(self.normalized_feature_dim))
        self.register_buffer("feature_std", torch.ones(self.normalized_feature_dim))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))
        self.register_buffer("sigma_fixed", torch.tensor(1.0))

    def _select_features(self, pair_features: torch.Tensor) -> torch.Tensor:
        if not self.select_clean_blocks:
            return pair_features
        blocks = pair_features.reshape(pair_features.shape[0], 5, self.block_dim)
        return torch.cat((blocks[:, 0], blocks[:, 1], blocks[:, 2], blocks[:, 4]), dim=1)

    def fit_statistics(
        self, pair_features: torch.Tensor, log_count_targets: torch.Tensor
    ) -> None:
        """Freeze train-only per-dimension block statistics and target standardization."""
        if pair_features.ndim != 2 or pair_features.shape[1] != self.pair_dim:
            raise ValueError("privacy train features have the wrong shape")
        if log_count_targets.ndim != 1 or len(log_count_targets) != len(pair_features):
            raise ValueError("privacy train targets are misaligned")
        selected = self._select_features(pair_features.detach())
        with torch.no_grad():
            self.feature_mean.copy_(selected.mean(dim=0))
            self.feature_std.copy_(
                selected.std(dim=0, unbiased=False).clamp_min(1e-6)
            )
            self.target_mean.copy_(log_count_targets.detach().mean())
            self.target_std.copy_(
                log_count_targets.detach().std(unbiased=False).clamp_min(1e-6)
            )

    def set_sigma_fixed(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("fixed sigma must be finite")
        with torch.no_grad():
            self.sigma_fixed.fill_(
                min(SIGMA_FIXED_MAX, max(SIGMA_FIXED_MIN, float(value)))
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
        elif count_basis is not None:
            raise ValueError("count basis was supplied to a basis-free privacy head")
        selected = self._select_features(pair_features)
        normalized = (selected - self.feature_mean) / self.feature_std
        if self.count_basis_size:
            normalized = torch.cat([normalized, frozen_basis], dim=1)
        standardized_mean = self.privacy_head(
            self.privacy_projection(normalized)
        ).squeeze(-1)
        mu = self.target_mean + self.target_std * standardized_mean
        return PrivacyPrediction(
            mu_log_count=mu,
            sigma_log_count=self.sigma_fixed.expand_as(mu),
            standardized_mean=standardized_mean,
        )


class CandidatePrivacyHead(SemanticPrivacyHead):
    """Candidate-only arm with a fixed projection to the shared head width."""

    def __init__(
        self,
        *,
        candidate_dim: int,
        budget_feature_dim: int,
        projection_dim: int,
        hidden_dim: int,
        count_basis_size: int,
    ):
        super().__init__(
            pair_dim=candidate_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            count_basis_size=count_basis_size,
            select_clean_blocks=False,
        )
        self.budget_feature_dim = int(budget_feature_dim)
        self.native_trainable_parameters = _head_parameter_count(
            self.normalized_feature_dim + count_basis_size, projection_dim
        )
        self.register_buffer(
            "fixed_projection",
            _make_fixed_projection(
                self.normalized_feature_dim,
                self.budget_feature_dim,
                seed=1009 + self.normalized_feature_dim + self.budget_feature_dim,
            ),
        )
        input_dim = self.budget_feature_dim + count_basis_size
        if projection_dim == 1:
            self.privacy_projection = nn.Linear(input_dim, 1)
            self.privacy_head = nn.Identity()
        else:
            self.privacy_projection = nn.Linear(input_dim, 32)
            self.privacy_head = nn.Sequential(nn.GELU(), nn.Linear(32, 1))

    def forward(
        self,
        pair_features: torch.Tensor,
        *,
        count_basis: torch.Tensor | None = None,
    ) -> PrivacyPrediction:
        if pair_features.ndim != 2 or pair_features.shape[1] != self.pair_dim:
            raise ValueError("candidate-only features have the wrong shape")
        selected = self._select_features(pair_features)
        normalized = (selected - self.feature_mean) / self.feature_std
        normalized = normalized @ self.fixed_projection
        if self.count_basis_size:
            if count_basis is None:
                count_basis = torch.zeros(
                    len(pair_features), dtype=torch.long, device=pair_features.device
                )
            frozen_basis = F.one_hot(
                count_basis, num_classes=self.count_basis_size
            ).to(dtype=pair_features.dtype)
            normalized = torch.cat([normalized, frozen_basis], dim=1)
        elif count_basis is not None:
            raise ValueError("count basis was supplied to a basis-free privacy head")
        standardized_mean = self.privacy_head(
            self.privacy_projection(normalized)
        ).squeeze(-1)
        mu = self.target_mean + self.target_std * standardized_mean
        return PrivacyPrediction(
            mu, self.sigma_fixed.expand_as(mu), standardized_mean
        )


def _privacy_relation_blocks(
    pair_features: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if pair_features.ndim != 2 or pair_features.shape[1] % 5:
        raise ValueError(
            "privacy attribution requires [batch, 5 * hidden_size] pair features"
        )
    hidden_size = pair_features.shape[1] // 5
    return tuple(pair_features[:, index * hidden_size:(index + 1) * hidden_size]
                 for index in range(5))


def select_privacy_attribution_features(
    examples: Sequence[PrivacyExample],
    arm: str,
) -> torch.Tensor:
    """Select one prescribed nested feature arm from frozen relation features."""
    if arm not in PRIVACY_ATTRIBUTION_ARMS:
        raise ValueError(f"unknown privacy attribution arm: {arm}")
    if not examples:
        raise ValueError("privacy attribution requires examples")
    pair = torch.stack([row.pair_features for row in examples])
    candidate_only = torch.stack([row.candidate_only for row in examples])
    type_block, source_block, candidate_block, _, hadamard_block = (
        _privacy_relation_blocks(pair)
    )
    hidden_size = candidate_block.shape[1]
    if candidate_only.shape != (len(examples), hidden_size):
        raise ValueError(
            "standalone candidate width differs from joint relation hidden size"
        )
    if arm in {"candidate_projected", "candidate_native"}:
        return candidate_only
    if arm == "bi_encoder":
        if any(row.independent_pair is None for row in examples):
            raise ValueError("bi-encoder arm requires independent pair features")
        independent = torch.stack([row.independent_pair for row in examples])
        if independent.shape != (len(examples), 4 * hidden_size):
            raise ValueError(
                "independent pair width differs from joint relation hidden size"
            )
        independent_source, independent_candidate, _, independent_product = (
            independent[:, index * hidden_size:(index + 1) * hidden_size]
            for index in range(4)
        )
        # Drop the linearly redundant difference block per the prescription.
        return torch.cat(
            (independent_source, independent_candidate, independent_product),
            dim=1,
        )
    if arm == "joint_candidate":
        return candidate_block
    if arm == "joint_candidate_source":
        return torch.cat((candidate_block, source_block), dim=1)
    if arm == "joint_candidate_source_hadamard":
        return torch.cat(
            (candidate_block, source_block, hadamard_block), dim=1
        )
    # Preserve the production cleaned-pair ordering for the full E arm.
    return torch.cat(
        (type_block, source_block, candidate_block, hadamard_block), dim=1
    )


def build_privacy_attribution_model(
    *,
    arm: str,
    pair_dim: int,
    candidate_dim: int,
) -> SemanticPrivacyHead:
    """Build the prescribed linear native-width head (projected candidate keeps fixed expansion)."""
    if arm not in PRIVACY_ATTRIBUTION_ARMS:
        raise ValueError(f"unknown privacy attribution arm: {arm}")
    if pair_dim <= 0 or pair_dim % 5 or candidate_dim != pair_dim // 5:
        raise ValueError("privacy attribution representation dimensions are invalid")
    hidden_size = pair_dim // 5
    if arm == "candidate_projected":
        return CandidatePrivacyHead(
            candidate_dim=candidate_dim,
            budget_feature_dim=4 * hidden_size,
            projection_dim=1,
            hidden_dim=0,
            count_basis_size=0,
        )
    width_multiplier = {
        "candidate_native": 1,
        "joint_candidate": 1,
        "joint_candidate_source": 2,
        "joint_candidate_source_hadamard": 3,
        "joint_full": 4,
        "bi_encoder": 3,
    }[arm]
    return SemanticPrivacyHead(
        pair_dim=width_multiplier * hidden_size,
        projection_dim=1,
        hidden_dim=0,
        count_basis_size=0,
        select_clean_blocks=False,
    )


class MetadataPrivacyHead(nn.Module):
    """Parameter-matched diagnostic baseline over mode/type metadata."""

    def __init__(
        self,
        *,
        runtime_types: Sequence[str],
        projection_dim: int,
        hidden_dim: int,
        include_authored_position: bool,
        budget_feature_dim: int,
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
        if projection_dim not in {1, 32} or hidden_dim != 0:
            raise ValueError("metadata baseline uses the fixed-sigma head budget")
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        feature_dim = len(self.mode_indices) + len(self.runtime_types)
        if self.include_authored_position:
            feature_dim += 1
        self.native_feature_dim = feature_dim
        self.budget_feature_dim = int(budget_feature_dim)
        self.native_trainable_parameters = _head_parameter_count(
            self.native_feature_dim, projection_dim
        )
        self.register_buffer(
            "fixed_projection",
            _make_fixed_projection(
                self.native_feature_dim,
                self.budget_feature_dim,
                seed=2003
                + self.native_feature_dim
                + self.budget_feature_dim
                + int(self.include_authored_position),
            ),
        )
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))
        self.register_buffer("sigma_fixed", torch.tensor(1.0))
        if projection_dim == 1:
            self.privacy_projection = nn.Linear(budget_feature_dim, 1)
            self.privacy_head = nn.Identity()
        else:
            self.privacy_projection = nn.Linear(budget_feature_dim, 32)
            self.privacy_head = nn.Sequential(nn.GELU(), nn.Linear(32, 1))

    def _features(
        self,
        authored_positions: torch.Tensor,
        modes: Sequence[str],
        runtime_types: Sequence[str],
    ) -> torch.Tensor:
        if authored_positions.ndim != 1:
            raise ValueError("authored positions must be a vector")
        if not (len(modes) == len(runtime_types) == authored_positions.shape[0]):
            raise ValueError("metadata baseline inputs are misaligned")
        feature_width = len(self.mode_indices) + len(self.runtime_types)
        features = authored_positions.new_zeros((len(modes), feature_width))
        for index, (mode, runtime_type) in enumerate(
            zip(modes, runtime_types, strict=True)
        ):
            if mode not in self.mode_indices or runtime_type not in self.runtime_type_indices:
                raise ValueError("unsupported metadata baseline input")
            features[index, self.mode_indices[mode]] = 1.0
            features[index, len(self.mode_indices) + self.runtime_type_indices[runtime_type]] = 1.0
        if self.include_authored_position:
            features = torch.cat(
                [features, torch.log1p(authored_positions.clamp_min(0)).unsqueeze(1)],
                dim=1,
            )
        return features

    def fit_statistics(
        self,
        authored_positions: torch.Tensor,
        modes: Sequence[str],
        runtime_types: Sequence[str],
        log_count_targets: torch.Tensor,
    ) -> None:
        features = self._features(authored_positions, modes, runtime_types)
        with torch.no_grad():
            self.feature_mean.copy_(features.mean(0))
            self.feature_std.copy_(features.std(0, unbiased=False).clamp_min(1e-6))
            self.target_mean.copy_(log_count_targets.mean())
            self.target_std.copy_(log_count_targets.std(unbiased=False).clamp_min(1e-6))

    def set_sigma_fixed(self, value: float) -> None:
        with torch.no_grad():
            self.sigma_fixed.fill_(
                min(SIGMA_FIXED_MAX, max(SIGMA_FIXED_MIN, float(value)))
            )

    def forward(
        self,
        authored_positions: torch.Tensor,
        modes: Sequence[str],
        runtime_types: Sequence[str],
    ) -> PrivacyPrediction:
        features = self._features(authored_positions, modes, runtime_types)
        normalized = (features - self.feature_mean) / self.feature_std
        expanded = normalized @ self.fixed_projection
        standardized_mean = self.privacy_head(
            self.privacy_projection(expanded)
        ).squeeze(-1)
        mu = self.target_mean + self.target_std * standardized_mean
        return PrivacyPrediction(
            mu,
            self.sigma_fixed.expand_as(mu),
            standardized_mean,
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
    """Create models with exactly matched total trainable parameter counts."""
    semantic = SemanticPrivacyHead(
        pair_dim=pair_dim,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        count_basis_size=count_basis_size,
    )
    budget_dim = semantic.budget_feature_dim + count_basis_size
    return {
        "semantic": semantic,
        "authored_position_mode_type": MetadataPrivacyHead(
            runtime_types=runtime_types,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            include_authored_position=True,
            budget_feature_dim=budget_dim,
        ),
        "mode_type_only": MetadataPrivacyHead(
            runtime_types=runtime_types,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            include_authored_position=False,
            budget_feature_dim=budget_dim,
        ),
        "candidate_only": CandidatePrivacyHead(
            candidate_dim=candidate_dim,
            budget_feature_dim=semantic.budget_feature_dim,
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
        level_means = prediction.mu_log_count[level_indices].clamp_min(0)
        # Degenerate all-nonpositive predictions normalize to zero scores rather
        # than crashing the policy forward; endpoints stay exact.
        denominator = level_means.max().clamp_min(
            torch.finfo(level_means.dtype).tiny
        )
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
    means: torch.Tensor,
    profile_slices: Sequence[slice],
) -> torch.Tensor:
    """Segment-vectorized profile-relative normalization (one kernel per op).

    Degenerate all-zero PREDICTIONS clamp to zero scores instead of aborting
    (targets are gated fail-closed upstream); singleton profiles score 1.0.
    """
    lengths = [profile_slice.stop - profile_slice.start for profile_slice in profile_slices]
    segment_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=means.device),
        torch.as_tensor(lengths, device=means.device),
    )
    means = means.clamp_min(0)
    segment_max = torch.full(
        (len(lengths),), torch.finfo(means.dtype).tiny,
        dtype=means.dtype, device=means.device,
    ).scatter_reduce(0, segment_ids, means, reduce="amax")
    denominator = segment_max.clamp_min(torch.finfo(means.dtype).tiny)
    normalized = torch.clamp(means / denominator[segment_ids], 0.0, 1.0)
    singleton = torch.as_tensor(
        [length == 1 for length in lengths], device=means.device
    )[segment_ids]
    return torch.where(singleton, torch.ones_like(normalized), normalized)


@dataclass(frozen=True)
class PrivacyTrainingStructure:
    """Static row, decision, profile, and unequal-pair indices for loss evaluation."""

    row_profile_ids: torch.Tensor
    row_decision_ids: torch.Tensor
    decision_profile_ids: torch.Tensor
    row_counts_by_decision: torch.Tensor
    decisions_by_profile: torch.Tensor
    pair_counts_by_decision: torch.Tensor
    pair_decisions_by_profile: torch.Tensor
    decision_sizes: torch.Tensor
    profile_count: int
    decision_count: int
    pair_left: torch.Tensor
    pair_right: torch.Tensor
    pair_decision_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "PrivacyTrainingStructure":
        return PrivacyTrainingStructure(**{
            name: (
                value.to(device=device) if isinstance(value, torch.Tensor) else value
            )
            for name, value in self.__dict__.items()
        })


def build_privacy_training_structure(
    *,
    decision_ids: Sequence[str],
    profile_ids: Sequence[str],
    log_count_targets: torch.Tensor,
) -> PrivacyTrainingStructure:
    """Precompute every static segment and pair index outside the update loop."""
    if (
        log_count_targets.ndim != 1
        or len(decision_ids) != len(profile_ids)
        or len(profile_ids) != len(log_count_targets)
        or not decision_ids
    ):
        raise ValueError("privacy training structure inputs are misaligned")
    profile_index = {value: index for index, value in enumerate(dict.fromkeys(profile_ids))}
    decision_index: dict[str, int] = {}
    decision_profile = []
    row_decisions = []
    row_profiles = []
    by_decision: dict[str, list[int]] = defaultdict(list)
    for row, (decision_id, profile_id) in enumerate(
        zip(decision_ids, profile_ids, strict=True)
    ):
        if decision_id not in decision_index:
            decision_index[decision_id] = len(decision_index)
            decision_profile.append(profile_index[profile_id])
        elif decision_profile[decision_index[decision_id]] != profile_index[profile_id]:
            raise ValueError("a privacy decision crosses profiles")
        row_decisions.append(decision_index[decision_id])
        row_profiles.append(profile_index[profile_id])
        by_decision[decision_id].append(row)
    lefts: list[int] = []
    rights: list[int] = []
    pair_decisions: list[int] = []
    for decision_id, indices in by_decision.items():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                if float(log_count_targets[left]) == float(log_count_targets[right]):
                    continue
                lefts.append(left)
                rights.append(right)
                pair_decisions.append(decision_index[decision_id])
    row_decision_tensor = torch.tensor(row_decisions, dtype=torch.long)
    decision_profile_tensor = torch.tensor(decision_profile, dtype=torch.long)
    pair_decision_tensor = torch.tensor(pair_decisions, dtype=torch.long)
    row_counts_by_decision = torch.bincount(
        row_decision_tensor, minlength=len(decision_index)
    )
    pair_counts_by_decision = torch.bincount(
        pair_decision_tensor, minlength=len(decision_index)
    )
    return PrivacyTrainingStructure(
        row_profile_ids=torch.tensor(row_profiles, dtype=torch.long),
        row_decision_ids=row_decision_tensor,
        decision_profile_ids=decision_profile_tensor,
        row_counts_by_decision=row_counts_by_decision,
        decisions_by_profile=torch.bincount(
            decision_profile_tensor,
            minlength=len(profile_index),
        ),
        pair_counts_by_decision=pair_counts_by_decision,
        pair_decisions_by_profile=torch.bincount(
            decision_profile_tensor[pair_counts_by_decision > 0],
            minlength=len(profile_index),
        ),
        decision_sizes=row_counts_by_decision,
        profile_count=len(profile_index),
        decision_count=len(decision_index),
        pair_left=torch.tensor(lefts, dtype=torch.long),
        pair_right=torch.tensor(rights, dtype=torch.long),
        pair_decision_ids=pair_decision_tensor,
    )


def _decision_profile_macro_mean(
    values: torch.Tensor,
    item_decision_ids: torch.Tensor,
    item_counts_by_decision: torch.Tensor,
    structure: PrivacyTrainingStructure,
    decisions_by_profile: torch.Tensor,
    active_profiles: torch.Tensor,
) -> torch.Tensor:
    decision_sums = values.new_zeros(structure.decision_count).scatter_add(
        0, item_decision_ids, values
    )
    valid_decisions = item_counts_by_decision > 0
    decision_means = values.new_zeros(structure.decision_count)
    decision_means[valid_decisions] = (
        decision_sums[valid_decisions]
        / item_counts_by_decision[valid_decisions]
    )
    profile_sums = values.new_zeros(structure.profile_count).scatter_add(
        0,
        structure.decision_profile_ids[valid_decisions],
        decision_means[valid_decisions],
    )
    valid_profiles = active_profiles & (decisions_by_profile > 0)
    if not bool(valid_profiles.any()):
        return values.sum() * 0.0
    return (
        profile_sums[valid_profiles] / decisions_by_profile[valid_profiles]
    ).mean()


def privacy_training_loss(
    prediction: PrivacyPrediction,
    log_count_targets: torch.Tensor,
    standardized_targets: torch.Tensor,
    structure: PrivacyTrainingStructure,
    profile_score_targets: torch.Tensor,
    *,
    rho: float,
    gamma: float,
    active_profiles: torch.Tensor | None = None,
) -> Mapping[str, torch.Tensor]:
    """Return profile-macro mean, bounded-rank, and controller calibration losses."""
    size = int(log_count_targets.numel())
    _validate_prediction(prediction, size)
    if (
        log_count_targets.ndim != 1
        or standardized_targets.shape != log_count_targets.shape
        or profile_score_targets.ndim != 1
        or profile_score_targets.shape != log_count_targets.shape
        or not torch.isfinite(log_count_targets).all()
        or not torch.isfinite(standardized_targets).all()
        or not torch.isfinite(profile_score_targets).all()
    ):
        raise ValueError("privacy targets must be aligned finite vectors")
    if rho < 0 or gamma < 0 or not math.isfinite(rho) or not math.isfinite(gamma):
        raise ValueError("privacy loss weights must be finite and nonnegative")
    mu = prediction.mu_log_count
    y_hat = prediction.standardized_mean
    if y_hat is None:
        raise ValueError("fixed-sigma training requires standardized predictions")
    if active_profiles is None:
        active_profiles = torch.ones(
            structure.profile_count, dtype=torch.bool, device=mu.device
        )
    if active_profiles.shape != (structure.profile_count,):
        raise ValueError("active profile mask has the wrong shape")
    row_huber = F.smooth_l1_loss(
        y_hat, standardized_targets, beta=1.0, reduction="none"
    )
    mean_huber = _decision_profile_macro_mean(
        row_huber,
        structure.row_decision_ids,
        structure.row_counts_by_decision,
        structure,
        structure.decisions_by_profile,
        active_profiles,
    )

    if structure.pair_left.numel():
        predicted_difference = mu[structure.pair_left] - mu[structure.pair_right]
        target_difference = (
            log_count_targets[structure.pair_left]
            - log_count_targets[structure.pair_right]
        )
        pair_huber = F.smooth_l1_loss(
            predicted_difference, target_difference, beta=1.0, reduction="none"
        )
        bounded_rank = _decision_profile_macro_mean(
            pair_huber,
            structure.pair_decision_ids,
            structure.pair_counts_by_decision,
            structure,
            structure.pair_decisions_by_profile,
            active_profiles,
        )
    else:
        bounded_rank = mu.sum() * 0.0

    decision_max = mu.new_full(
        (structure.decision_count,), torch.finfo(mu.dtype).tiny
    ).scatter_reduce(
        0, structure.row_decision_ids, mu.clamp_min(0), reduce="amax"
    )
    normalized = torch.clamp(
        mu.clamp_min(0)
        / decision_max.clamp_min(torch.finfo(mu.dtype).tiny)[structure.row_decision_ids],
        0.0,
        1.0,
    )
    normalized = torch.where(
        structure.decision_sizes[structure.row_decision_ids] == 1,
        torch.ones_like(normalized),
        normalized,
    )
    calibration_rows = F.smooth_l1_loss(
        normalized, profile_score_targets, beta=1.0, reduction="none"
    )
    profile_huber = _decision_profile_macro_mean(
        calibration_rows,
        structure.row_decision_ids,
        structure.row_counts_by_decision,
        structure,
        structure.decisions_by_profile,
        active_profiles,
    )
    total = mean_huber + float(rho) * bounded_rank + float(gamma) * profile_huber
    return {
        "standardized_mean_huber": mean_huber,
        "bounded_difference_rank": bounded_rank,
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


def _rank_values(
    values: Sequence[float], *, tolerance: float = 0.0
) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while (
            stop < len(order)
            and abs(values[order[stop]] - values[order[cursor]]) <= tolerance
        ):
            stop += 1
        average_rank = (cursor + stop - 1) / 2.0
        for ordered_index in order[cursor:stop]:
            ranks[ordered_index] = average_rank
        cursor = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = _rank_values(left, tolerance=PREDICTION_TIE_TOLERANCE)
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

    decision_pair_accuracy: dict[str, float] = {}
    decision_predicted_tie_rate: dict[str, float] = {}
    decision_correlations: dict[str, float] = {}
    decision_regrets: dict[str, float] = {}
    decision_calibration: dict[str, float] = {}
    decision_profiles: dict[str, str] = {}
    by_profile: dict[str, list[str]] = defaultdict(list)
    by_decision: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_decision[examples[index].decision_id].append(index)
    for decision_id, profile_indices in by_decision.items():
        pair_correct = []
        predicted_ties = 0
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
                if abs(predicted_difference) <= PREDICTION_TIE_TOLERANCE:
                    pair_correct.append(0.5)
                    predicted_ties += 1
                else:
                    pair_correct.append(
                        float(predicted_difference * target_difference > 0)
                    )
        if pair_correct:
            decision_pair_accuracy[decision_id] = sum(pair_correct) / len(pair_correct)
            decision_predicted_tie_rate[decision_id] = (
                predicted_ties / len(pair_correct)
            )
        correlation = _spearman(
            [float(prediction.mu_log_count[index]) for index in profile_indices],
            [examples[index].log_count_target for index in profile_indices],
        )
        if correlation is not None:
            decision_correlations[decision_id] = correlation
        selected = max(profile_indices, key=lambda index: (normalized[index], -index))
        best_target = max(
            examples[index].profile_score_target for index in profile_indices
        )
        decision_regrets[decision_id] = (
            best_target - examples[selected].profile_score_target
        )
        decision_calibration[decision_id] = sum(
            abs(normalized[index] - examples[index].profile_score_target)
            for index in profile_indices
        ) / len(profile_indices)
        profile_id = examples[profile_indices[0]].profile_id
        decision_profiles[decision_id] = profile_id
        by_profile[profile_id].append(decision_id)

    def profile_macro(values: Mapping[str, float]) -> float | None:
        profile_values = []
        for decision_ids in by_profile.values():
            present = [values[value] for value in decision_ids if value in values]
            if present:
                profile_values.append(sum(present) / len(present))
        return (
            sum(profile_values) / len(profile_values) if profile_values else None
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
        "within_menu_pairwise_accuracy": profile_macro(decision_pair_accuracy),
        "within_menu_predicted_tie_rate": profile_macro(
            decision_predicted_tie_rate
        ),
        "spearman": profile_macro(decision_correlations),
        "profile_relative_calibration_error": profile_macro(
            decision_calibration
        ),
        "selected_action_regret": profile_macro(decision_regrets),
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

    by_profile = {}
    for profile_id in sorted({row.profile_id for row in examples}):
        profile_indices = [
            index for index, row in enumerate(examples)
            if row.profile_id == profile_id
        ]
        metrics = _metric_subset(
            examples, prediction, normalized, profile_indices
        )
        by_profile[profile_id] = {
            name: metrics[name]
            for name in (
                "profile_relative_calibration_error",
                "within_menu_pairwise_accuracy",
                "within_menu_predicted_tie_rate",
                "selected_action_regret",
                "median_absolute_log_error",
            )
        }

    return {
        "overall": _metric_subset(
            examples, prediction, normalized, all_indices
        ),
        "by_runtime_type": stratify("runtime_type"),
        "by_grounding_status": stratify("grounding_status"),
        "by_source_family": stratify("source_family"),
        "by_profile": by_profile,
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
    seeds: tuple[int, ...]
    training_seed: int
    metric_report_hash: str
    diagnostic_manifest_hash: str
    counterexample_set_hash: str | None
    run_protocol: str
    seed_count: int
    promotion_verdict: str
    target_mean: float
    target_std: float
    sigma_fixed: float
    feature_schema: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip: float

    def __post_init__(self) -> None:
        if (
            self.pair_dim <= 0
            or self.projection_dim not in {1, 32}
            or self.hidden_dim != 0
        ):
            raise ValueError("privacy checkpoint fixed-sigma dimensions are invalid")
        if self.count_basis_size != len(self.count_basis_categories):
            raise ValueError("privacy checkpoint count basis size mismatch")
        if self.count_basis_size and self.count_basis_categories[0] != "<unknown>":
            raise ValueError("privacy checkpoint count basis lacks unknown category")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("privacy checkpoint requires distinct explicit seeds")
        if self.seed_count != len(self.seeds):
            raise ValueError("privacy checkpoint seed count mismatch")
        if self.training_seed not in self.seeds:
            raise ValueError("privacy checkpoint training seed is not orchestrated")
        if self.run_protocol not in {"iteration", "promotion"}:
            raise ValueError("privacy checkpoint run protocol is invalid")
        if self.run_protocol == "iteration" and self.seed_count != 1:
            raise ValueError("iteration checkpoints require exactly one seed")
        if self.run_protocol == "promotion" and self.seed_count < 3:
            raise ValueError("promotion checkpoints require at least three seeds")
        if self.promotion_verdict not in {
            "PROMOTE",
            "FAIL",
            "NEEDS_MULTI_SEED_EVIDENCE",
            "NEEDS_COUNTEREXAMPLE_SET",
            "NEEDS_MULTI_SEED_AND_COUNTEREXAMPLE_SET",
        }:
            raise ValueError("privacy checkpoint promotion verdict is invalid")
        if self.rho < 0 or self.gamma < 0:
            raise ValueError("privacy checkpoint loss weights must be nonnegative")
        if self.gamma != 1.0:
            raise ValueError("privacy checkpoint requires gamma 1.0")
        if not (SIGMA_FIXED_MIN <= self.sigma_fixed <= SIGMA_FIXED_MAX):
            raise ValueError("privacy checkpoint fixed sigma is out of bounds")
        if self.target_std <= 0 or not all(math.isfinite(value) for value in (
            self.target_mean, self.target_std, self.sigma_fixed,
            self.learning_rate, self.weight_decay, self.gradient_clip,
        )):
            raise ValueError("privacy checkpoint numeric protocol is invalid")
        if (
            self.feature_schema != FEATURE_SCHEMA
            or self.optimizer != "AdamW"
            or self.learning_rate != 3e-4
            or self.weight_decay != 0.01
            or self.gradient_clip != 1.0
        ):
            raise ValueError("privacy checkpoint protocol mismatch")
        for name in (
            "environment_hash",
            "profile_target_artifact_hash",
            "representation_manifest_hash",
            "encoder_revision",
            "split_manifest_hash",
            "metric_report_hash",
            "diagnostic_manifest_hash",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"privacy checkpoint contract lacks {name}")
        if (
            self.counterexample_set_hash is not None
            and (
                not isinstance(self.counterexample_set_hash, str)
                or not self.counterexample_set_hash
            )
        ):
            raise ValueError("privacy checkpoint counterexample hash is invalid")

    def validate_for_policy(
        self, *, allow_development_override: bool = False
    ) -> None:
        """Reject undeployable privacy checkpoints before tensor loading."""
        if self.count_basis_size:
            raise ValueError(
                "privacy checkpoints with a count basis cannot enter policy inference"
            )
        promoted = (
            self.run_protocol == "promotion"
            and self.seed_count >= 3
            and self.promotion_verdict == "PROMOTE"
            and self.counterexample_set_hash is not None
        )
        if not promoted and not allow_development_override:
            raise ValueError(
                "privacy checkpoint lacks promotion evidence; "
                "an explicit development override is required"
            )


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
    if not (
        math.isclose(float(model.target_mean), contract.target_mean, abs_tol=1e-6)
        and math.isclose(float(model.target_std), contract.target_std, abs_tol=1e-6)
        and math.isclose(float(model.sigma_fixed), contract.sigma_fixed, abs_tol=1e-6)
    ):
        raise ValueError("privacy checkpoint fitted statistics mismatch")
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
    fitted_contract = {
        "target_mean": expected_contract.target_mean,
        "target_std": expected_contract.target_std,
        "sigma_fixed": expected_contract.sigma_fixed,
    }
    for name, expected_value in fitted_contract.items():
        supplied = state_dict[name]
        value = float(supplied)
        if not math.isfinite(value):
            raise ValueError(f"privacy checkpoint {name} must be finite")
        if name == "target_std" and value <= 0:
            raise ValueError("privacy checkpoint target_std must be positive")
        if name == "sigma_fixed" and not (
            SIGMA_FIXED_MIN <= value <= SIGMA_FIXED_MAX
        ):
            raise ValueError("privacy checkpoint sigma_fixed is out of bounds")
        if not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"privacy checkpoint {name} differs from contract")
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
    *,
    admitted_profile_ids: Sequence[str] | None = None,
) -> tuple[PrivacyExample, ...]:
    """Join admitted Task 1 level targets to frozen Task 2 representations."""
    targets = ProfileCountTargets.from_artifact(target_artifact)
    admitted_profiles = (
        None
        if admitted_profile_ids is None
        else frozenset(str(value) for value in admitted_profile_ids)
    )
    if admitted_profiles is not None and not admitted_profiles:
        raise ValueError("privacy example profile filter must not be empty")
    actions = {}
    decisions = {}
    for document in documents.values():
        for decision in document.policy_decisions:
            decisions[decision.decision_id] = decision
            for action in decision.actions:
                if action.action_id in actions:
                    raise ValueError(f"duplicate environment action: {action.action_id}")
                actions[action.action_id] = action

    examples = []
    for target in targets.target_rows(eligible_only=True):
        if target.mode != "level":
            continue
        if (
            admitted_profiles is not None
            and target.profile_id not in admitted_profiles
        ):
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
        decision = decisions[target.decision_id]
        source_identity = _stable_hash(str(decision.canonical_key))
        candidate_identity = _stable_hash(str(action.fill))
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
            source_identity=source_identity,
            candidate_identity=candidate_identity,
            independent_pair=relation.independent_pair.detach().to(
                dtype=torch.float32, device="cpu"
            ),
        ))
    if not examples:
        raise ValueError("profile target artifact has no eligible level rows")
    return deduplicate_privacy_examples(examples)


def deduplicate_privacy_examples(
    examples: Sequence[PrivacyExample],
) -> tuple[PrivacyExample, ...]:
    """Deduplicate only identical complete decision menus within one profile."""
    by_decision: dict[str, list[PrivacyExample]] = defaultdict(list)
    for row in examples:
        by_decision[row.decision_id].append(row)
    retained: dict[tuple[Any, ...], tuple[PrivacyExample, ...]] = {}
    for decision_id in sorted(by_decision):
        menu = tuple(sorted(
            by_decision[decision_id],
            key=lambda row: (row.authored_position, row.action_id),
        ))
        signature = (
            menu[0].profile_id,
            tuple(
                (
                    row.authored_position,
                    row.runtime_type,
                    row.source_identity,
                    row.candidate_identity
                    or _stable_hash(row.candidate_only.tolist()),
                    row.log_count_target,
                    row.profile_score_target,
                    row.grounding_status,
                    row.source_family,
                )
                for row in menu
            ),
        )
        retained.setdefault(signature, menu)
    rows = tuple(row for menu in retained.values() for row in menu)
    return tuple(sorted(rows, key=lambda row: (
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


def _prediction_inputs(
    name: str,
    examples: Sequence[PrivacyExample],
    basis: CountBasisVocabulary | None,
    device: torch.device | str = "cpu",
) -> tuple[tuple, dict]:
    """Build one model-call input set; hoistable out of the step loop and device-placed."""
    if name in {"semantic", "candidate_only"}:
        feature_name = "pair_features" if name == "semantic" else "candidate_only"
        features = torch.stack(
            [getattr(row, feature_name) for row in examples]
        ).to(device=device)
        indices = basis.encode(examples) if basis is not None else None
        if isinstance(indices, torch.Tensor):
            indices = indices.to(device=device)
        return (features,), {"count_basis": indices}
    positions = torch.tensor(
        [row.authored_position for row in examples], dtype=torch.float32, device=device
    )
    return (
        positions,
        ["level"] * len(examples),
        [row.runtime_type for row in examples],
    ), {}


def _model_prediction(
    name: str,
    model: nn.Module,
    examples: Sequence[PrivacyExample],
    basis: CountBasisVocabulary | None,
) -> PrivacyPrediction:
    args, kwargs = _prediction_inputs(name, examples, basis)
    return model(*args, **kwargs)


def _privacy_selection_key(
    rows: Sequence[PrivacyExample],
    prediction: PrivacyPrediction,
) -> tuple[float, float, float]:
    detached = PrivacyPrediction(
        prediction.mu_log_count.detach().cpu(),
        prediction.sigma_log_count.detach().cpu(),
    )
    metrics = evaluate_privacy_predictions(rows, detached)["overall"]
    ordering_value = metrics["within_menu_pairwise_accuracy"]
    targets = torch.tensor(
        [row.log_count_target for row in rows],
        dtype=detached.mu_log_count.dtype,
    )
    return (
        float(metrics["profile_relative_calibration_error"]),
        float(ordering_value) if ordering_value is not None else -1.0,
        float((detached.mu_log_count - targets).abs().mean()),
    )


def _privacy_selection_improved(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float],
    *,
    minimum_improvement: float,
) -> bool:
    if candidate[0] < incumbent[0] - minimum_improvement:
        return True
    if abs(candidate[0] - incumbent[0]) <= minimum_improvement:
        if candidate[1] > incumbent[1] + minimum_improvement:
            return True
        if (
            abs(candidate[1] - incumbent[1]) <= minimum_improvement
            and candidate[2] < incumbent[2] - minimum_improvement
        ):
            return True
    return False


def _fit_privacy_model(
    model: nn.Module,
    *,
    train_args: tuple,
    train_kwargs: Mapping[str, Any],
    validation_args: tuple,
    validation_kwargs: Mapping[str, Any],
    train_rows: Sequence[PrivacyExample],
    validation_rows: Sequence[PrivacyExample],
    log_targets: torch.Tensor,
    standardized_targets: torch.Tensor,
    score_targets: torch.Tensor,
    structure: PrivacyTrainingStructure,
    profile_count: int,
    generator: torch.Generator,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    max_steps: int,
    evaluation_interval: int,
    patience: int,
    profile_batch_size: int,
    minimum_improvement: float,
    rho: float,
    gamma: float,
) -> int:
    """Run the shared profile-balanced optimizer and lexicographic dev selection."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def validation_key() -> tuple[float, float, float]:
        model.eval()
        with torch.inference_mode():
            value = _privacy_selection_key(
                validation_rows,
                model(*validation_args, **validation_kwargs),
            )
        model.train()
        return value

    best_key = validation_key()
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    selected_step = 0
    stale_evaluations = 0
    permutation = torch.randperm(profile_count, generator=generator).tolist()
    cursor = 0
    model.train()
    for step in range(1, max_steps + 1):
        if cursor >= profile_count:
            permutation = torch.randperm(
                profile_count, generator=generator
            ).tolist()
            cursor = 0
        selected_profiles = permutation[cursor:cursor + profile_batch_size]
        cursor += profile_batch_size
        active_profiles = torch.zeros(
            structure.profile_count,
            dtype=torch.bool,
            device=log_targets.device,
        )
        active_profiles[selected_profiles] = True
        optimizer.zero_grad(set_to_none=True)
        losses = privacy_training_loss(
            model(*train_args, **train_kwargs),
            log_targets,
            standardized_targets,
            structure,
            score_targets,
            rho=rho,
            gamma=gamma,
            active_profiles=active_profiles,
        )
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if step % evaluation_interval == 0 or step == max_steps:
            candidate_key = validation_key()
            if _privacy_selection_improved(
                candidate_key,
                best_key,
                minimum_improvement=minimum_improvement,
            ):
                best_key = candidate_key
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                selected_step = step
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if stale_evaluations >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_prediction = model(*validation_args, **validation_kwargs)
        validation_targets = torch.tensor(
            [row.log_count_target for row in validation_rows],
            dtype=validation_prediction.mu_log_count.dtype,
            device=validation_prediction.mu_log_count.device,
        )
        residual_rmse = torch.sqrt(
            (
                validation_prediction.mu_log_count - validation_targets
            ).square().mean()
        ).item()
    model.set_sigma_fixed(residual_rmse)
    return selected_step


def train_privacy_attribution_fold(
    examples: Sequence[PrivacyExample],
    *,
    train_profiles: Sequence[str],
    validation_profiles: Sequence[str],
    arm: str,
    seed: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    max_steps: int = 500,
    evaluation_interval: int = 10,
    patience: int = 10,
    profile_batch_size: int = 32,
    minimum_improvement: float = 1e-3,
    rho: float = 0.0,
    gamma: float = 1.0,
    device: str = "cpu",
) -> tuple[SemanticPrivacyHead, dict[str, Any]]:
    """Fit one attribution arm/fold with the production profile-macro protocol."""
    if (
        learning_rate != 3e-4
        or weight_decay != 0.01
        or gradient_clip != 1.0
        or not 0 < max_steps <= 500
        or evaluation_interval != 10
        or patience != 10
        or rho != 0.0
        or gamma != 1.0
        or profile_batch_size <= 0
    ):
        raise ValueError("privacy attribution protocol differs from its prescription")
    train_ids = frozenset(str(value) for value in train_profiles)
    validation_ids = frozenset(str(value) for value in validation_profiles)
    if (
        not train_ids
        or not validation_ids
        or train_ids & validation_ids
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("privacy attribution fold partition is invalid")
    observed = {row.profile_id for row in examples}
    if observed != train_ids | validation_ids:
        raise ValueError("privacy attribution rows do not exactly cover the fold")
    train_rows, _ = _ordered_examples(
        tuple(row for row in examples if row.profile_id in train_ids)
    )
    validation_rows, _ = _ordered_examples(
        tuple(row for row in examples if row.profile_id in validation_ids)
    )
    pair_dim = int(train_rows[0].pair_features.numel())
    candidate_dim = int(train_rows[0].candidate_only.numel())
    if any(
        row.pair_features.numel() != pair_dim
        or row.candidate_only.numel() != candidate_dim
        for row in examples
    ):
        raise ValueError("privacy attribution feature widths are inconsistent")

    torch.manual_seed(seed)
    model = build_privacy_attribution_model(
        arm=arm,
        pair_dim=pair_dim,
        candidate_dim=candidate_dim,
    )
    train_device = torch.device(device)
    model.to(train_device)
    train_features = select_privacy_attribution_features(
        train_rows, arm
    ).to(train_device)
    validation_features = select_privacy_attribution_features(
        validation_rows, arm
    ).to(train_device)
    log_targets = torch.tensor(
        [row.log_count_target for row in train_rows],
        dtype=torch.float32,
        device=train_device,
    )
    score_targets = torch.tensor(
        [row.profile_score_target for row in train_rows],
        dtype=torch.float32,
        device=train_device,
    )
    model.fit_statistics(train_features, log_targets)
    standardized_targets = (
        log_targets - model.target_mean
    ) / model.target_std
    structure = build_privacy_training_structure(
        decision_ids=[row.decision_id for row in train_rows],
        profile_ids=[row.profile_id for row in train_rows],
        log_count_targets=log_targets.detach().cpu(),
    ).to(train_device)
    profile_ids = tuple(dict.fromkeys(row.profile_id for row in train_rows))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_step = _fit_privacy_model(
        model,
        train_args=(train_features,),
        train_kwargs={},
        validation_args=(validation_features,),
        validation_kwargs={},
        train_rows=train_rows,
        validation_rows=validation_rows,
        log_targets=log_targets,
        standardized_targets=standardized_targets,
        score_targets=score_targets,
        structure=structure,
        profile_count=len(profile_ids),
        generator=generator,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        max_steps=max_steps,
        evaluation_interval=evaluation_interval,
        patience=patience,
        profile_batch_size=profile_batch_size,
        minimum_improvement=minimum_improvement,
        rho=rho,
        gamma=gamma,
    )
    with torch.inference_mode():
        evaluation = evaluate_privacy_predictions(
            validation_rows,
            PrivacyPrediction(
                model(validation_features).mu_log_count.detach().cpu(),
                model.sigma_fixed.expand(len(validation_rows)).detach().cpu(),
            ),
        )
    profile_rows: dict[str, list[PrivacyExample]] = defaultdict(list)
    for row in validation_rows:
        profile_rows[row.profile_id].append(row)
    profile_report = {}
    for profile_id, rows in sorted(profile_rows.items()):
        runtime_types = sorted({row.runtime_type for row in rows})
        source_families = sorted({row.source_family for row in rows})
        if len(runtime_types) != 1:
            raise ValueError("privacy attribution profile crosses runtime types")
        profile_report[profile_id] = {
            "runtime_type": runtime_types[0],
            "source_families": source_families,
            "metrics": evaluation["by_profile"][profile_id],
        }
    reported_metrics = (
        "within_menu_pairwise_accuracy",
        "within_menu_predicted_tie_rate",
        "profile_relative_calibration_error",
        "selected_action_regret",
        "median_absolute_log_error",
    )
    run_report = {
        "arm": arm,
        "seed": seed,
        "selected_step": selected_step,
        "feature_dim": int(train_features.shape[1]),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "sigma_fixed": float(model.sigma_fixed),
        "metrics": {
            name: evaluation["overall"][name] for name in reported_metrics
        },
        "profiles": profile_report,
    }
    model.to("cpu")
    return model, run_report


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
    device: str = "cpu",
    weight_decay: float = 0.01,
    profile_batch_size: int = 32,
    evaluation_interval: int = 10,
    patience: int = 10,
    minimum_improvement: float = 1e-3,
    gradient_clip: float = 1.0,
) -> tuple[SemanticPrivacyHead, dict[str, Any]]:
    """Train one fixed-sigma seed with profile-balanced updates and dev selection."""
    if not (0 < max_steps <= 500) or learning_rate != 3e-4:
        raise ValueError("privacy training requires lr 3e-4 and at most 500 updates")
    if (
        weight_decay != 0.01
        or gradient_clip != 1.0
        or gamma != 1.0
        or rho not in {0.0, 0.05}
    ):
        raise ValueError("privacy training protocol differs from the adopted redesign")
    if min(profile_batch_size, evaluation_interval, patience) <= 0:
        raise ValueError("privacy batching and selection values must be positive")
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
    train_rows, _ = _ordered_examples(split_rows["train"])
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
    train_device = torch.device(device)
    device_log_targets = log_targets.to(device=train_device)
    target_mean = device_log_targets.mean()
    target_std = device_log_targets.std(unbiased=False).clamp_min(1e-6)
    device_standardized_targets = (
        device_log_targets - target_mean
    ) / target_std
    device_score_targets = score_targets.to(device=train_device)
    structure = build_privacy_training_structure(
        decision_ids=[row.decision_id for row in train_rows],
        profile_ids=[row.profile_id for row in train_rows],
        log_count_targets=log_targets,
    ).to(train_device)
    dev_rows, _ = _ordered_examples(split_rows["dev"])
    profile_ids = tuple(dict.fromkeys(row.profile_id for row in train_rows))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_steps: dict[str, int] = {}
    sigma_fixed: dict[str, float] = {}
    trainable_parameters = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    native_trainable_parameters = {
        name: int(model.native_trainable_parameters)
        for name, model in models.items()
    }
    if len(set(trainable_parameters.values())) != 1:
        raise ValueError("neural privacy baselines are not parameter matched")

    def fit_model_statistics(name: str, model: nn.Module) -> None:
        args, _ = _prediction_inputs(name, train_rows, basis, device=train_device)
        if isinstance(model, MetadataPrivacyHead):
            model.fit_statistics(*args, device_log_targets)
        else:
            model.fit_statistics(args[0], device_log_targets)

    for name, model in models.items():
        model.to(train_device)
        fit_model_statistics(name, model)
        validation_args, validation_kwargs = _prediction_inputs(
            name, dev_rows, basis, device=train_device
        )
        train_args, train_kwargs = _prediction_inputs(
            name, train_rows, basis, device=train_device
        )
        selected_steps[name] = _fit_privacy_model(
            model,
            train_args=train_args,
            train_kwargs=train_kwargs,
            validation_args=validation_args,
            validation_kwargs=validation_kwargs,
            train_rows=train_rows,
            validation_rows=dev_rows,
            log_targets=device_log_targets,
            standardized_targets=device_standardized_targets,
            score_targets=device_score_targets,
            structure=structure,
            profile_count=len(profile_ids),
            generator=generator,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            max_steps=max_steps,
            evaluation_interval=evaluation_interval,
            patience=patience,
            profile_batch_size=profile_batch_size,
            minimum_improvement=minimum_improvement,
            rho=rho,
            gamma=gamma,
        )
        sigma_fixed[name] = float(model.sigma_fixed)
        model.to("cpu")

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
        "selected_steps": dict(sorted(selected_steps.items())),
        "sigma_fixed": dict(sorted(sigma_fixed.items())),
        "target_standardization": {
            "mean": float(target_mean),
            "std": float(target_std),
            "source": "train-only",
        },
        "trainable_parameters": dict(sorted(trainable_parameters.items())),
        "native_trainable_parameters": dict(
            sorted(native_trainable_parameters.items())
        ),
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "profile_batch_size": profile_batch_size,
            "evaluation_interval": evaluation_interval,
            "patience": patience,
            "minimum_improvement": minimum_improvement,
            "rho": rho,
            "gamma": gamma,
        },
        "count_basis_categories": list(basis.categories) if basis else [],
        "splits": split_reports,
    }
    return models["semantic"], report
