"""Candidate-conditioned context and selected-action memory for Ranker-v2."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cloak.ranker.environment import LambdaProfile
from cloak.ranker.environment import RankerAction, RankerDecision, RankerDocument
from cloak.ranker.privacy import (
    DirectCountPrivacyProvider,
    PrivacyCheckpointContract,
    SemanticPrivacyHead,
    load_privacy_checkpoint,
    profile_normalize_predictions,
    validate_privacy_signal_for_policy,
)
from cloak.ranker.representation import (
    DocumentTokenBank,
    RankerRepresentationStore,
    RelationFeatures,
)


ORDINARY_ROLE = 0
CURRENT_OCCURRENCE_ROLE = 1
OTHER_CONTROLLED_ROLE = 2
ROLE_COUNT = 3
RELATIVE_POSITION_BUCKETS = 17
DOCUMENT_POSITION_BINS = 16
CONTEXT_MODES = (
    "local-cls-mean",
    "target-bidirectional",
    "full-candidate-attention",
)
PRODUCTION_CONTEXT_MODE = "full-candidate-attention"
HISTORY_MODES = (
    "none",
    "utility-gru",
    "selected-cross-attention",
)
PRODUCTION_HISTORY_MODE = "selected-cross-attention"


@dataclass(frozen=True)
class DecisionTokenFeatures:
    role_ids: torch.Tensor
    relative_position_ids: torch.Tensor
    document_position_ids: torch.Tensor
    occurrence_token_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SelectedActionRecord:
    """One count-blind utility-semantic record for a previously selected decision."""

    utility_relation: torch.Tensor
    action_mode_id: int
    runtime_type_id: int
    source_position_pool: torch.Tensor


@dataclass(frozen=True)
class SemanticPolicyState:
    document: RankerDocument
    profile: LambdaProfile
    selected_records: tuple[SelectedActionRecord, ...]


@dataclass(frozen=True)
class ActionDistribution:
    action_ids: tuple[str, ...]
    utility_logits: torch.Tensor
    mu_log_count: torch.Tensor
    sigma_log_count: torch.Tensor
    predicted_privacy: torch.Tensor
    combined_logits: torch.Tensor
    log_probs: torch.Tensor
    count_log_probs: torch.Tensor


def stack_utility_relations(
    relations: Sequence[RelationFeatures],
) -> torch.Tensor:
    """Stack one decision menu's frozen ordered relation pairs for utility queries."""
    relations = tuple(relations)
    if not relations:
        raise ValueError("utility relations must be nonempty")
    if len({relation.decision_id for relation in relations}) != 1:
        raise ValueError("utility relations must belong to one decision")
    if len({relation.action_id for relation in relations}) != len(relations):
        raise ValueError("utility relations repeat an action")
    expected = relations[0].pair
    if expected.ndim != 1 or expected.numel() == 0:
        raise ValueError("utility relation pair must be a nonempty vector")
    for relation in relations:
        if (
            relation.pair.ndim != 1
            or relation.pair.shape != expected.shape
            or relation.pair.dtype != expected.dtype
            or relation.pair.device != expected.device
        ):
            raise ValueError("utility relation pairs are inconsistent")
    return torch.stack([relation.pair for relation in relations])


def signed_distance_bucket(distance: int) -> int:
    """Map a signed source-token distance to the frozen seventeen-bin scheme."""
    if distance <= -128:
        return 0
    if distance <= -64:
        return 1
    if distance <= -32:
        return 2
    if distance <= -16:
        return 3
    if distance <= -8:
        return 4
    if distance <= -4:
        return 5
    if distance <= -2:
        return 6
    if distance == -1:
        return 7
    if distance == 0:
        return 8
    if distance == 1:
        return 9
    if distance <= 3:
        return 10
    if distance <= 7:
        return 11
    if distance <= 15:
        return 12
    if distance <= 31:
        return 13
    if distance <= 63:
        return 14
    if distance <= 127:
        return 15
    return 16


def _occurrence_index(
    document: RankerDocument,
) -> dict[str, Mapping]:
    indexed = {}
    for occurrence in document.occurrences:
        occurrence_id = occurrence.get("occurrence_id")
        if not isinstance(occurrence_id, str) or not occurrence_id:
            raise ValueError("document occurrence lacks an identity")
        if occurrence_id in indexed:
            raise ValueError(f"duplicate document occurrence: {occurrence_id}")
        indexed[occurrence_id] = occurrence
    return indexed


def _token_indices_for_span(
    offsets: torch.Tensor, start: int, end: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets.tolist())
        if int(token_start) < end and int(token_end) > start
    )


def build_decision_token_features(
    token_bank: DocumentTokenBank,
    document: RankerDocument,
    decision: RankerDecision,
) -> DecisionTokenFeatures:
    """Derive frozen token roles and source-coordinate positions for one decision."""
    if token_bank.doc_id != document.doc_id:
        raise ValueError("token bank and document identities differ")
    if token_bank.offsets.ndim != 2 or token_bank.offsets.shape[1] != 2:
        raise ValueError("token bank offsets must have shape [tokens, 2]")
    token_count = int(token_bank.offsets.shape[0])
    if token_bank.states.ndim != 2 or token_bank.states.shape[0] != token_count:
        raise ValueError("token bank states and offsets are misaligned")
    if len(token_bank.chunk_membership) != token_count:
        raise ValueError("token bank chunk membership is misaligned")
    if not document.text:
        raise ValueError("document text must be nonempty")

    occurrences = _occurrence_index(document)
    current_groups = []
    for occurrence_id in decision.occurrence_ids:
        try:
            occurrence = occurrences[occurrence_id]
        except KeyError as exc:
            raise ValueError(
                f"decision occurrence is absent: {occurrence_id}"
            ) from exc
        if occurrence.get("decision_id") != decision.decision_id:
            raise ValueError(f"decision occurrence mapping differs: {occurrence_id}")
        indices = _token_indices_for_span(
            token_bank.offsets,
            int(occurrence["start"]),
            int(occurrence["end"]),
        )
        if not indices:
            raise ValueError(f"decision occurrence has no source tokens: {occurrence_id}")
        current_groups.append(indices)
    if not current_groups:
        raise ValueError("decision has no mapped occurrences")

    current_indices = {index for group in current_groups for index in group}
    other_controlled_indices = set()
    for occurrence_id, occurrence in occurrences.items():
        if occurrence_id in decision.occurrence_ids:
            continue
        if occurrence.get("controlled") is not True:
            continue
        other_controlled_indices.update(_token_indices_for_span(
            token_bank.offsets,
            int(occurrence["start"]),
            int(occurrence["end"]),
        ))

    role_ids = torch.full((token_count,), ORDINARY_ROLE, dtype=torch.int64)
    if other_controlled_indices:
        role_ids[list(sorted(other_controlled_indices))] = OTHER_CONTROLLED_ROLE
    role_ids[list(sorted(current_indices))] = CURRENT_OCCURRENCE_ROLE

    relative_ids = []
    ordered_current = tuple(sorted(current_indices))
    for token_index in range(token_count):
        nearest = min(
            ordered_current,
            key=lambda current: (abs(token_index - current), current),
        )
        relative_ids.append(signed_distance_bucket(token_index - nearest))

    document_ids = []
    document_length = len(document.text)
    for start, end in token_bank.offsets.tolist():
        midpoint = (int(start) + int(end)) / 2.0
        position_bin = int(midpoint * DOCUMENT_POSITION_BINS / document_length)
        document_ids.append(min(DOCUMENT_POSITION_BINS - 1, max(0, position_bin)))

    return DecisionTokenFeatures(
        role_ids=role_ids,
        relative_position_ids=torch.tensor(relative_ids, dtype=torch.int64),
        document_position_ids=torch.tensor(document_ids, dtype=torch.int64),
        occurrence_token_indices=tuple(current_groups),
    )


class CandidateContextReadout(nn.Module):
    """Read target, local, and complete-document evidence with candidate queries."""

    def __init__(
        self,
        *,
        token_dim: int,
        relation_dim: int,
        context_dim: int,
        num_heads: int,
        context_mode: str = PRODUCTION_CONTEXT_MODE,
        dropout: float = 0.0,
        production: bool = False,
    ):
        super().__init__()
        if min(token_dim, relation_dim, context_dim, num_heads) <= 0:
            raise ValueError("context dimensions and head count must be positive")
        if context_dim % num_heads:
            raise ValueError("context dimension must be divisible by head count")
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"unsupported context_mode: {context_mode}")
        if production and context_mode != PRODUCTION_CONTEXT_MODE:
            raise ValueError("production accepts only full-candidate-attention")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("context dropout must be in [0, 1)")

        self.token_dim = int(token_dim)
        self.relation_dim = int(relation_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.context_mode = context_mode
        self.production = bool(production)

        self.token_projection = nn.Linear(token_dim, context_dim)
        self.query_projection = nn.Linear(relation_dim, context_dim)
        self.role_embedding = nn.Embedding(ROLE_COUNT, context_dim)
        self.relative_position_embedding = nn.Embedding(
            RELATIVE_POSITION_BUCKETS, context_dim
        )
        self.document_position_embedding = nn.Embedding(
            DOCUMENT_POSITION_BINS, context_dim
        )
        self.occurrence_position_embedding = nn.Embedding(
            DOCUMENT_POSITION_BINS, context_dim
        )
        attention_kwargs = {
            "embed_dim": context_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "batch_first": True,
        }
        self.target_attention = nn.MultiheadAttention(**attention_kwargs)
        self.local_attention = nn.MultiheadAttention(**attention_kwargs)
        self.global_attention = nn.MultiheadAttention(**attention_kwargs)
        self.context_projection = nn.Linear(3 * context_dim, context_dim)

    def _validate_inputs(
        self,
        token_bank: DocumentTokenBank,
        features: DecisionTokenFeatures,
        utility_relations: torch.Tensor,
    ) -> None:
        token_count = int(token_bank.states.shape[0])
        if (
            token_bank.states.ndim != 2
            or token_bank.states.shape[1] != self.token_dim
            or token_count == 0
        ):
            raise ValueError("context token bank has an invalid state shape")
        if len(token_bank.chunk_membership) != token_count:
            raise ValueError("context chunk membership is misaligned")
        for name, values, upper in (
            ("role", features.role_ids, ROLE_COUNT),
            ("relative position", features.relative_position_ids, RELATIVE_POSITION_BUCKETS),
            ("document position", features.document_position_ids, DOCUMENT_POSITION_BINS),
        ):
            if (
                values.ndim != 1
                or values.shape[0] != token_count
                or values.dtype != torch.int64
                or torch.any(values < 0)
                or torch.any(values >= upper)
            ):
                raise ValueError(f"context {name} features are invalid")
        if not features.occurrence_token_indices or any(
            not group or any(index < 0 or index >= token_count for index in group)
            for group in features.occurrence_token_indices
        ):
            raise ValueError("context occurrence token groups are invalid")
        if (
            utility_relations.ndim != 2
            or utility_relations.shape[0] == 0
            or utility_relations.shape[1] != self.relation_dim
        ):
            raise ValueError("utility relation tensor has an invalid shape")
        devices = {
            token_bank.states.device,
            features.role_ids.device,
            features.relative_position_ids.device,
            features.document_position_ids.device,
            utility_relations.device,
        }
        if len(devices) != 1:
            raise ValueError("context inputs must share one device")

    @staticmethod
    def _attend(
        attention: nn.MultiheadAttention,
        queries: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, weights = attention(
            queries.unsqueeze(0),
            values.unsqueeze(0),
            values.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        return output.squeeze(0), weights.squeeze(0)

    def forward_with_attention(
        self,
        token_bank: DocumentTokenBank,
        decision_features: DecisionTokenFeatures,
        utility_relations: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return context vectors and diagnostic branch weights."""
        self._validate_inputs(token_bank, decision_features, utility_relations)
        augmented = (
            self.token_projection(token_bank.states)
            + self.role_embedding(decision_features.role_ids)
            + self.relative_position_embedding(
                decision_features.relative_position_ids
            )
            + self.document_position_embedding(
                decision_features.document_position_ids
            )
        )
        queries = self.query_projection(utility_relations)
        occurrence_values = []
        for indices in decision_features.occurrence_token_indices:
            index_tensor = torch.tensor(
                indices, dtype=torch.long, device=augmented.device
            )
            occurrence_state = augmented.index_select(0, index_tensor).mean(dim=0)
            occurrence_bin = int(
                torch.round(
                    decision_features.document_position_ids
                    .index_select(0, index_tensor)
                    .to(torch.float32)
                    .mean()
                ).item()
            )
            occurrence_values.append(
                occurrence_state
                + self.occurrence_position_embedding.weight[occurrence_bin]
            )
        target_values = torch.stack(occurrence_values)

        target_chunks = {
            chunk_id
            for indices in decision_features.occurrence_token_indices
            for index in indices
            for chunk_id in token_bank.chunk_membership[index]
        }
        if not target_chunks:
            raise ValueError("target occurrences have no chunk membership")
        local_indices = [
            index
            for index, membership in enumerate(token_bank.chunk_membership)
            if target_chunks.intersection(membership)
        ]
        local_values = augmented[local_indices]
        action_count = int(queries.shape[0])

        if self.context_mode == "local-cls-mean":
            target_context = target_values.mean(dim=0).expand(action_count, -1)
            local_context = local_values.mean(dim=0).expand(action_count, -1)
            global_context = torch.zeros_like(target_context)
            target_weights = torch.full(
                (action_count, target_values.shape[0]),
                1.0 / target_values.shape[0],
                dtype=augmented.dtype,
                device=augmented.device,
            )
            local_weights = torch.full(
                (action_count, local_values.shape[0]),
                1.0 / local_values.shape[0],
                dtype=augmented.dtype,
                device=augmented.device,
            )
            global_weights = torch.zeros(
                (action_count, augmented.shape[0]),
                dtype=augmented.dtype,
                device=augmented.device,
            )
        else:
            target_context, target_weights = self._attend(
                self.target_attention, queries, target_values
            )
            local_context, local_weights = self._attend(
                self.local_attention, queries, local_values
            )
            if self.context_mode == "full-candidate-attention":
                global_context, global_weights = self._attend(
                    self.global_attention, queries, augmented
                )
            else:
                global_context = torch.zeros_like(target_context)
                global_weights = torch.zeros(
                    (action_count, augmented.shape[0]),
                    dtype=augmented.dtype,
                    device=augmented.device,
                )

        combined = torch.cat(
            [target_context, local_context, global_context], dim=-1
        )
        return self.context_projection(combined), {
            "target": target_weights,
            "local": local_weights,
            "global": global_weights,
        }

    def forward(
        self,
        token_bank: DocumentTokenBank,
        decision_features: DecisionTokenFeatures,
        utility_relations: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = self.forward_with_attention(
            token_bank, decision_features, utility_relations
        )
        return context


class SelectedActionMemory(nn.Module):
    """Retrieve earlier selected utility semantics for each current candidate.

    The record interface deliberately exposes only the selected action's utility
    relation, mode, runtime type, and original occurrence positions. Privacy
    supervision is a separate branch and must not consume this module's output or
    update its projection and attention parameters.
    """

    def __init__(
        self,
        *,
        relation_dim: int,
        context_dim: int,
        history_dim: int,
        num_heads: int,
        action_mode_count: int,
        runtime_type_count: int,
        history_mode: str = PRODUCTION_HISTORY_MODE,
        dropout: float = 0.0,
        production: bool = False,
    ):
        super().__init__()
        dimensions = (
            relation_dim,
            context_dim,
            history_dim,
            num_heads,
            action_mode_count,
            runtime_type_count,
        )
        if min(dimensions) <= 0:
            raise ValueError("history dimensions and category sizes must be positive")
        if history_dim % num_heads:
            raise ValueError("history dimension must be divisible by head count")
        if history_mode not in HISTORY_MODES:
            raise ValueError(f"unsupported history_mode: {history_mode}")
        if production and history_mode != PRODUCTION_HISTORY_MODE:
            raise ValueError("production accepts only selected-cross-attention history")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("history dropout must be in [0, 1)")

        self.relation_dim = int(relation_dim)
        self.context_dim = int(context_dim)
        self.history_dim = int(history_dim)
        self.action_mode_count = int(action_mode_count)
        self.runtime_type_count = int(runtime_type_count)
        self.history_mode = history_mode
        self.production = bool(production)

        self.action_mode_embedding = nn.Embedding(action_mode_count, history_dim)
        self.runtime_type_embedding = nn.Embedding(runtime_type_count, history_dim)
        self.source_position_embedding = nn.Embedding(
            DOCUMENT_POSITION_BINS, history_dim
        )
        self.record_projection = nn.Linear(
            relation_dim + 3 * history_dim, history_dim
        )
        self.query_projection = nn.Linear(
            relation_dim + context_dim, history_dim
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=history_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.utility_gru = (
            nn.GRU(
                input_size=history_dim,
                hidden_size=history_dim,
                batch_first=True,
            )
            if history_mode == "utility-gru"
            else None
        )

    def build_record(
        self,
        *,
        utility_relation: torch.Tensor,
        action_mode_id: int,
        runtime_type_id: int,
        decision_features: DecisionTokenFeatures,
    ) -> SelectedActionRecord:
        """Build one decision-level record, pooling every original occurrence."""
        if utility_relation.ndim != 1 or utility_relation.shape[0] != self.relation_dim:
            raise ValueError("selected utility relation has an invalid shape")
        if utility_relation.device != self.record_projection.weight.device:
            raise ValueError("selected utility relation is on a different device")
        if utility_relation.dtype != self.record_projection.weight.dtype:
            raise ValueError("selected utility relation has a different dtype")
        self._validate_category_id(
            action_mode_id, self.action_mode_count, "action mode"
        )
        self._validate_category_id(
            runtime_type_id, self.runtime_type_count, "runtime type"
        )
        positions = decision_features.document_position_ids
        if (
            positions.ndim != 1
            or positions.dtype != torch.int64
            or positions.numel() == 0
            or torch.any(positions < 0)
            or torch.any(positions >= DOCUMENT_POSITION_BINS)
        ):
            raise ValueError("selected source positions are invalid")
        if positions.device != self.source_position_embedding.weight.device:
            raise ValueError("selected source positions are on a different device")
        if not decision_features.occurrence_token_indices:
            raise ValueError("selected decision has no original occurrences")

        occurrence_positions = []
        for indices in decision_features.occurrence_token_indices:
            if not indices or any(
                index < 0 or index >= positions.numel() for index in indices
            ):
                raise ValueError("selected occurrence token indices are invalid")
            index_tensor = torch.tensor(
                indices, dtype=torch.long, device=positions.device
            )
            occurrence_positions.append(
                self.source_position_embedding(
                    positions.index_select(0, index_tensor)
                ).mean(dim=0)
            )
        source_position_pool = torch.stack(occurrence_positions).mean(dim=0)
        return SelectedActionRecord(
            utility_relation=utility_relation,
            action_mode_id=action_mode_id,
            runtime_type_id=runtime_type_id,
            source_position_pool=source_position_pool,
        )

    @staticmethod
    def _validate_category_id(value: int, upper: int, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < upper
        ):
            raise ValueError(f"selected {name} id is invalid")

    def _record_rows(
        self, records: tuple[SelectedActionRecord, ...]
    ) -> torch.Tensor:
        device = self.record_projection.weight.device
        dtype = self.record_projection.weight.dtype
        utility_relations = []
        source_positions = []
        action_modes = []
        runtime_types = []
        for record in records:
            if not isinstance(record, SelectedActionRecord):
                raise TypeError(
                    "history records must be SelectedActionRecord instances"
                )
            if (
                record.utility_relation.ndim != 1
                or record.utility_relation.shape[0] != self.relation_dim
                or record.utility_relation.device != device
                or record.utility_relation.dtype != dtype
            ):
                raise ValueError("history utility relation is invalid")
            if (
                record.source_position_pool.ndim != 1
                or record.source_position_pool.shape[0] != self.history_dim
                or record.source_position_pool.device != device
                or record.source_position_pool.dtype != dtype
            ):
                raise ValueError("history source-position pool is invalid")
            self._validate_category_id(
                record.action_mode_id, self.action_mode_count, "action mode"
            )
            self._validate_category_id(
                record.runtime_type_id, self.runtime_type_count, "runtime type"
            )
            utility_relations.append(record.utility_relation)
            source_positions.append(record.source_position_pool)
            action_modes.append(record.action_mode_id)
            runtime_types.append(record.runtime_type_id)

        action_mode_ids = torch.tensor(action_modes, dtype=torch.long, device=device)
        runtime_type_ids = torch.tensor(runtime_types, dtype=torch.long, device=device)
        row_inputs = torch.cat(
            [
                torch.stack(utility_relations),
                self.action_mode_embedding(action_mode_ids),
                self.runtime_type_embedding(runtime_type_ids),
                torch.stack(source_positions),
            ],
            dim=-1,
        )
        return self.record_projection(row_inputs)

    def forward(
        self,
        candidate_queries: torch.Tensor,
        records: tuple[SelectedActionRecord, ...],
    ) -> torch.Tensor:
        """Return candidate histories with a common width for every ablation arm."""
        if (
            candidate_queries.ndim != 2
            or candidate_queries.shape[0] == 0
            or candidate_queries.shape[1] != self.relation_dim + self.context_dim
            or candidate_queries.device != self.query_projection.weight.device
            or candidate_queries.dtype != self.query_projection.weight.dtype
        ):
            raise ValueError("candidate history queries are invalid")
        if not isinstance(records, tuple):
            raise TypeError("history records must be a tuple")
        action_count = int(candidate_queries.shape[0])
        if self.history_mode == "none" or not records:
            return candidate_queries.new_zeros((action_count, self.history_dim))

        rows = self._record_rows(records)
        if self.history_mode == "utility-gru":
            assert self.utility_gru is not None
            _, hidden = self.utility_gru(rows.unsqueeze(0))
            return hidden[-1].expand(action_count, -1)

        queries = self.query_projection(candidate_queries)
        output, _ = self.cross_attention(
            queries.unsqueeze(0),
            rows.unsqueeze(0),
            rows.unsqueeze(0),
            need_weights=False,
        )
        return output.squeeze(0)


class SemanticRankerPolicy(nn.Module):
    """Compose count-blind utility semantics with a frozen privacy controller.

    The utility tower consumes only frozen relation features, candidate-conditioned
    document context, an explicit context-relation interaction, categorical
    mode/type embeddings, and selected utility history. Lambda and privacy enter
    only through the final scalar additive controller.
    """

    def __init__(
        self,
        *,
        representation_store: RankerRepresentationStore,
        privacy_head: nn.Module | None = None,
        privacy_provider: DirectCountPrivacyProvider | None = None,
        privacy_signal: Mapping[str, str] | None = None,
        supported_profiles: Sequence[LambdaProfile],
        max_lambda: float,
        token_dim: int,
        pair_dim: int,
        relation_dim: int,
        context_dim: int,
        history_dim: int,
        utility_hidden_dim: int,
        num_heads: int,
        runtime_types: Sequence[str],
        dropout: float = 0.0,
    ):
        super().__init__()
        dimensions = (
            token_dim,
            pair_dim,
            relation_dim,
            context_dim,
            history_dim,
            utility_hidden_dim,
            num_heads,
        )
        if min(dimensions) <= 0:
            raise ValueError("semantic policy dimensions must be positive")
        if not math.isfinite(max_lambda) or max_lambda <= 0.0:
            raise ValueError("semantic policy max_lambda must be positive")
        profiles = tuple(supported_profiles)
        if not profiles:
            raise ValueError("semantic policy requires supported lambda values")
        profile_values = tuple(float(profile.value) for profile in profiles)
        if (
            any(
                not isinstance(profile, LambdaProfile)
                or not math.isfinite(profile.value)
                or profile.value < 0.0
                for profile in profiles
            )
            or len(profile_values) != len(set(profile_values))
            or 0.0 not in profile_values
            or max(profile_values) != float(max_lambda)
        ):
            raise ValueError("semantic policy lambda menu is invalid")
        runtime_types = tuple(str(value) for value in runtime_types)
        if (
            not runtime_types
            or any(not value for value in runtime_types)
            or len(runtime_types) != len(set(runtime_types))
        ):
            raise ValueError("semantic policy runtime types are invalid")
        if (privacy_head is None) == (privacy_provider is None):
            raise ValueError(
                "semantic policy requires exactly one privacy head or provider"
            )
        if privacy_head is not None:
            if not isinstance(privacy_head, nn.Module):
                raise TypeError("semantic policy privacy head must be a module")
            declared_pair_dim = getattr(privacy_head, "pair_dim", pair_dim)
            if declared_pair_dim != pair_dim:
                raise ValueError("semantic policy privacy pair dimension differs")
        if privacy_provider is not None and not isinstance(
            privacy_provider, DirectCountPrivacyProvider
        ):
            raise TypeError(
                "semantic policy direct privacy provider has the wrong type"
            )
        if privacy_signal is not None:
            if privacy_signal.get("kind") == "direct-count":
                validate_privacy_signal_for_policy(privacy_signal)
            elif privacy_signal.get("kind") != "learned-head":
                raise ValueError("semantic policy privacy signal kind is invalid")

        self.representation_store = representation_store
        self.privacy_head = privacy_head
        self.privacy_provider = privacy_provider
        self.privacy_signal = (
            dict(privacy_signal) if privacy_signal is not None else None
        )
        self.supported_lambda_values = profile_values
        self.max_lambda = float(max_lambda)
        self.pair_dim = int(pair_dim)
        self.relation_dim = int(relation_dim)
        self.context_dim = int(context_dim)
        self.history_dim = int(history_dim)
        self.runtime_types = runtime_types
        self.runtime_type_ids = {
            value: index for index, value in enumerate(runtime_types)
        }
        self.action_mode_ids = {"level": 0, "keep": 1, "placeholder": 2}
        # Frozen-store inputs are static per (document, decision); cache their
        # device-converted forms so replay/rollout loops stop rebuilding them.
        # Cleared by _apply on any device/dtype move.
        self._token_bank_cache: dict[str, DocumentTokenBank] = {}
        self._decision_feature_cache: dict[tuple[str, str], DecisionTokenFeatures] = {}
        self._pair_feature_cache: dict[str, torch.Tensor] = {}

        self.utility_projection = nn.Linear(pair_dim, relation_dim)
        self.context_readout = CandidateContextReadout(
            token_dim=token_dim,
            relation_dim=relation_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            context_mode=PRODUCTION_CONTEXT_MODE,
            dropout=dropout,
            production=True,
        )
        self.context_to_relation = nn.Linear(context_dim, relation_dim)
        self.interaction_projection = nn.Linear(relation_dim, relation_dim)
        self.action_mode_embedding = nn.Embedding(
            len(self.action_mode_ids), relation_dim
        )
        self.runtime_type_embedding = nn.Embedding(
            len(runtime_types), relation_dim
        )
        self.memory = SelectedActionMemory(
            relation_dim=relation_dim,
            context_dim=context_dim,
            history_dim=history_dim,
            num_heads=num_heads,
            action_mode_count=len(self.action_mode_ids),
            runtime_type_count=len(runtime_types),
            history_mode=PRODUCTION_HISTORY_MODE,
            dropout=dropout,
            production=True,
        )
        utility_input_dim = (
            relation_dim
            + context_dim
            + relation_dim
            + relation_dim
            + relation_dim
            + history_dim
        )
        self.utility_head = nn.Sequential(
            nn.Linear(utility_input_dim, utility_hidden_dim),
            nn.GELU(),
            nn.Linear(utility_hidden_dim, 1),
        )
        self.alpha_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32)
        )

        if self.privacy_head is not None:
            for parameter in self.privacy_head.parameters():
                parameter.requires_grad_(False)
            self.privacy_head.eval()

    @classmethod
    def from_privacy_checkpoint(
        cls,
        *,
        privacy_checkpoint: Path,
        privacy_checkpoint_contract: PrivacyCheckpointContract,
        allow_development_privacy_checkpoint: bool = False,
        **kwargs,
    ) -> "SemanticRankerPolicy":
        """Load a fail-closed privacy checkpoint and freeze it in the policy."""
        privacy_signal = {
            "kind": "learned-head",
            "targets_artifact_hash": (
                privacy_checkpoint_contract.profile_target_artifact_hash
            ),
            "environment_hash": privacy_checkpoint_contract.environment_hash,
        }
        validate_privacy_signal_for_policy(
            privacy_signal,
            learned_contract=privacy_checkpoint_contract,
            allow_development_override=allow_development_privacy_checkpoint
        )
        supplied_pair_dim = kwargs.pop("pair_dim", privacy_checkpoint_contract.pair_dim)
        if supplied_pair_dim != privacy_checkpoint_contract.pair_dim:
            raise ValueError("semantic policy pair dimension differs from checkpoint")
        privacy_head = SemanticPrivacyHead(
            pair_dim=privacy_checkpoint_contract.pair_dim,
            projection_dim=privacy_checkpoint_contract.projection_dim,
            hidden_dim=privacy_checkpoint_contract.hidden_dim,
            count_basis_size=privacy_checkpoint_contract.count_basis_size,
        )
        load_privacy_checkpoint(
            privacy_checkpoint,
            privacy_head,
            privacy_checkpoint_contract,
        )
        return cls(
            privacy_head=privacy_head,
            privacy_signal=privacy_signal,
            pair_dim=supplied_pair_dim,
            **kwargs,
        )

    @classmethod
    def from_direct_count_targets(
        cls,
        *,
        profile_count_targets: Mapping[str, object],
        **kwargs,
    ) -> "SemanticRankerPolicy":
        """Build a policy whose privacy controller uses exact artifact scores."""
        provider = DirectCountPrivacyProvider(profile_count_targets)
        return cls(
            privacy_provider=provider,
            privacy_signal=provider.privacy_signal,
            **kwargs,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def train(self, mode: bool = True) -> "SemanticRankerPolicy":
        super().train(mode)
        if self.privacy_head is not None:
            self.privacy_head.eval()
        return self

    def _validate_profile(self, profile: LambdaProfile) -> float:
        if not isinstance(profile, LambdaProfile):
            raise TypeError("semantic policy requires a LambdaProfile")
        value = float(profile.value)
        if value not in self.supported_lambda_values:
            raise ValueError("unsupported lambda value")
        return value

    def begin_document(
        self, document: RankerDocument, profile: LambdaProfile
    ) -> SemanticPolicyState:
        self._validate_profile(profile)
        if not isinstance(document, RankerDocument):
            raise TypeError("semantic policy requires a RankerDocument")
        return SemanticPolicyState(
            document=document,
            profile=profile,
            selected_records=(),
        )

    def _validate_state(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
        profile: LambdaProfile,
    ) -> None:
        if not isinstance(state, SemanticPolicyState):
            raise TypeError("semantic policy state is invalid")
        profile_value = self._validate_profile(profile)
        if profile_value != float(state.profile.value):
            raise ValueError("lambda value changed within document")
        matches = tuple(
            row
            for row in state.document.policy_decisions
            if row.decision_id == decision.decision_id
        )
        if len(matches) != 1 or matches[0] != decision:
            raise ValueError("decision does not belong to semantic policy state")

    @property
    def _device(self) -> torch.device:
        return self.utility_projection.weight.device

    def _apply(self, fn, recurse=True):
        # Device/dtype moves invalidate the cached device-converted store tensors.
        self._token_bank_cache.clear()
        self._decision_feature_cache.clear()
        self._pair_feature_cache.clear()
        return super()._apply(fn, recurse)

    def _decision_inputs(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
    ) -> tuple[
        tuple[RankerAction, ...],
        torch.Tensor,
        DocumentTokenBank,
        DecisionTokenFeatures,
    ]:
        actions = tuple(decision.actions)
        if not actions or len({action.action_id for action in actions}) != len(actions):
            raise ValueError("semantic policy decision menu is invalid")
        if any(action.runtime_type != decision.runtime_type for action in actions):
            raise ValueError("semantic policy action runtime type differs")
        if decision.runtime_type not in self.runtime_type_ids:
            raise ValueError("semantic policy runtime type is unsupported")
        pair_features = self._pair_feature_cache.get(decision.decision_id)
        if pair_features is None:
            relations = tuple(
                self.representation_store.relation(
                    decision.decision_id, action.action_id
                )
                for action in actions
            )
            pair_features = stack_utility_relations(relations).to(
                device=self._device,
                dtype=self.utility_projection.weight.dtype,
            )
            if pair_features.shape[1] != self.pair_dim:
                raise ValueError("semantic policy relation pair dimension differs")
            self._pair_feature_cache[decision.decision_id] = pair_features

        token_bank = self._token_bank_cache.get(state.document.doc_id)
        if token_bank is None:
            stored_bank = self.representation_store.document(state.document.doc_id)
            token_bank = DocumentTokenBank(
                doc_id=stored_bank.doc_id,
                states=stored_bank.states.to(
                    device=self._device,
                    dtype=self.utility_projection.weight.dtype,
                ),
                offsets=stored_bank.offsets.to(device=self._device),
                chunk_membership=stored_bank.chunk_membership,
            )
            self._token_bank_cache[state.document.doc_id] = token_bank

        feature_key = (state.document.doc_id, decision.decision_id)
        device_features = self._decision_feature_cache.get(feature_key)
        if device_features is None:
            features = build_decision_token_features(
                self.representation_store.document(state.document.doc_id),
                state.document,
                decision,
            )
            device_features = DecisionTokenFeatures(
                role_ids=features.role_ids.to(device=self._device),
                relative_position_ids=features.relative_position_ids.to(
                    device=self._device
                ),
                document_position_ids=features.document_position_ids.to(
                    device=self._device
                ),
                occurrence_token_indices=features.occurrence_token_indices,
            )
            self._decision_feature_cache[feature_key] = device_features
        return actions, pair_features, token_bank, device_features

    def _category_ids(
        self, actions: Sequence[RankerAction], decision: RankerDecision
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            mode_values = [self.action_mode_ids[action.mode] for action in actions]
            runtime_value = self.runtime_type_ids[decision.runtime_type]
        except KeyError as exc:
            raise ValueError(
                "semantic policy categorical value is unsupported"
            ) from exc
        modes = torch.tensor(mode_values, dtype=torch.long, device=self._device)
        runtime_types = torch.full(
            (len(actions),), runtime_value, dtype=torch.long, device=self._device
        )
        return modes, runtime_types

    def distribution(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        profile: LambdaProfile,
    ) -> ActionDistribution:
        self._validate_state(state, decision, profile)
        actions, pair_features, token_bank, features = self._decision_inputs(
            state, decision
        )
        complete_ids = tuple(action.action_id for action in actions)
        legal_ids = tuple(str(value) for value in legal_action_ids)
        if (
            not legal_ids
            or len(legal_ids) != len(set(legal_ids))
            or any(action_id not in complete_ids for action_id in legal_ids)
        ):
            raise ValueError("semantic policy legal action menu is invalid")

        utility_relations = self.utility_projection(pair_features)
        contexts = self.context_readout(
            token_bank, features, utility_relations
        )
        histories = self.memory(
            torch.cat([utility_relations, contexts], dim=-1),
            state.selected_records,
        )
        mode_ids, runtime_type_ids = self._category_ids(actions, decision)
        interaction = self.interaction_projection(
            utility_relations * self.context_to_relation(contexts)
        )
        utility_inputs = torch.cat(
            [
                utility_relations,
                contexts,
                interaction,
                self.action_mode_embedding(mode_ids),
                self.runtime_type_embedding(runtime_type_ids),
                histories,
            ],
            dim=-1,
        )
        complete_utility = self.utility_head(utility_inputs).squeeze(-1)

        with torch.no_grad():
            if self.privacy_provider is not None:
                complete_privacy = self.privacy_provider(
                    (decision.decision_id,) * len(actions),
                    complete_ids,
                    device=self._device,
                    dtype=complete_utility.dtype,
                )
                complete_mu_log_count = torch.zeros_like(complete_privacy)
                complete_sigma_log_count = torch.zeros_like(complete_privacy)
            else:
                assert self.privacy_head is not None
                privacy_prediction = self.privacy_head(pair_features)
                complete_privacy = profile_normalize_predictions(
                    privacy_prediction,
                    tuple(action.mode for action in actions),
                )
                complete_mu_log_count = privacy_prediction.mu_log_count
                complete_sigma_log_count = privacy_prediction.sigma_log_count
        legal_indices = torch.tensor(
            [complete_ids.index(action_id) for action_id in legal_ids],
            dtype=torch.long,
            device=self._device,
        )
        utility_logits = complete_utility.index_select(0, legal_indices)
        mu_log_count = complete_mu_log_count.index_select(
            0, legal_indices
        )
        sigma_log_count = complete_sigma_log_count.index_select(
            0, legal_indices
        )
        predicted_privacy = complete_privacy.index_select(0, legal_indices)

        lambda_value = float(profile.value)
        if lambda_value == 0.0:
            combined_logits = utility_logits
            count_combined = utility_logits.detach()
        else:
            magnitude = math.log1p(lambda_value) / math.log1p(self.max_lambda)
            controller = self.alpha * magnitude * predicted_privacy.detach()
            if getattr(self, "alpha_utility_routing", None) == "per-decision":
                # Numerically identical forward pass; alpha's gradient through the
                # utility/entropy/KL channels is decision-averaged so both of the
                # controller's opposing pressures share one per-decision scale
                # (decision-log fork: objective normalization mix, 2026-07-28).
                scale = 1.0 / max(1, len(state.document.policy_decisions))
                routed = controller.detach() + (controller - controller.detach()) * scale
            else:
                routed = controller
            combined_logits = utility_logits + routed
            count_combined = utility_logits.detach() + controller
        return ActionDistribution(
            action_ids=legal_ids,
            utility_logits=utility_logits,
            mu_log_count=mu_log_count,
            sigma_log_count=sigma_log_count,
            predicted_privacy=predicted_privacy,
            combined_logits=combined_logits,
            log_probs=torch.log_softmax(combined_logits, dim=0),
            count_log_probs=torch.log_softmax(count_combined, dim=0),
        )

    def log_probs(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        profile: LambdaProfile,
    ) -> torch.Tensor:
        return self.distribution(
            state, decision, legal_action_ids, profile
        ).log_probs

    def advance(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
        action_id: str,
    ) -> SemanticPolicyState:
        self._validate_state(state, decision, state.profile)
        matches = tuple(
            action for action in decision.actions if action.action_id == action_id
        )
        if len(matches) != 1:
            raise ValueError("selected semantic policy action is unknown")
        action = matches[0]
        _, pair_features, _, features = self._decision_inputs(state, decision)
        selected_index = tuple(
            row.action_id for row in decision.actions
        ).index(action_id)
        selected_relation = self.utility_projection(
            pair_features[selected_index:selected_index + 1]
        ).squeeze(0)
        mode_ids, runtime_type_ids = self._category_ids((action,), decision)
        record = self.memory.build_record(
            utility_relation=selected_relation,
            action_mode_id=int(mode_ids[0].item()),
            runtime_type_id=int(runtime_type_ids[0].item()),
            decision_features=features,
        )
        return SemanticPolicyState(
            document=state.document,
            profile=state.profile,
            selected_records=(*state.selected_records, record),
        )


def production_decision_order(
    document: RankerDocument,
) -> tuple[RankerDecision, ...]:
    """Return the loader-frozen first-occurrence walk without copying or sorting."""
    return document.policy_decisions


def reverse_diagnostic_decision_order(
    document: RankerDocument,
) -> tuple[RankerDecision, ...]:
    """Return a reverse replay order; callers must reapply dynamic legality."""
    return tuple(reversed(production_decision_order(document)))


def seeded_diagnostic_decision_order(
    document: RankerDocument, *, seed: int
) -> tuple[RankerDecision, ...]:
    """Return a stable seeded replay order without mutating production order."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("diagnostic replay seed must be an integer")

    def replay_key(decision: RankerDecision) -> bytes:
        value = f"{seed}\0{decision.decision_id}".encode("utf-8")
        return hashlib.sha256(value).digest()

    return tuple(sorted(production_decision_order(document), key=replay_key))
