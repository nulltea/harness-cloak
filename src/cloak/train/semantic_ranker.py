"""Candidate-conditioned frozen-document context features for Ranker-v2."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from cloak.train.ranker_environment import RankerDecision, RankerDocument
from cloak.train.ranker_representation import DocumentTokenBank, RelationFeatures


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


@dataclass(frozen=True)
class DecisionTokenFeatures:
    role_ids: torch.Tensor
    relative_position_ids: torch.Tensor
    document_position_ids: torch.Tensor
    occurrence_token_indices: tuple[tuple[int, ...], ...]


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
