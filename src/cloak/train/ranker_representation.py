"""Frozen document and semantic-relation representations for Ranker-v2."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from cloak.runtime_types import RUNTIME_TYPES
from cloak.train.ranker_environment import RankerAction, RankerDecision, RankerDocument


ENCODER_ID = "thomas-sounack/BioClinical-ModernBERT-base"
ENCODER_REVISION = "c3648aa87af95837c809e6f0c5f85d08160db437"
TOKENIZER_ID = ENCODER_ID
TOKENIZER_REVISION = ENCODER_REVISION
HIDDEN_SIZE = 768
CHUNK_LENGTH = 512
SOURCE_TOKEN_OVERLAP = 64
FIELD_SERIALIZATION_VERSION = "ranker-relation-fields-v1"
OFFSET_FORMAT_VERSION = "character-span-half-open-v1"
CANDIDATE_ONLY_VERSION = "candidate-only-mean-v1"
INDEPENDENT_PAIR_VERSION = "independent-ordered-pair-v1"
ARTIFACT_VERSION = "ranker-v2-representation-store-v1"


@dataclass(frozen=True)
class DocumentTokenBank:
    doc_id: str
    states: torch.Tensor
    offsets: torch.Tensor
    chunk_membership: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RelationFeatures:
    decision_id: str
    action_id: str
    type_mean: torch.Tensor
    source_mean: torch.Tensor
    candidate_mean: torch.Tensor
    pair: torch.Tensor
    candidate_only: torch.Tensor
    independent_pair: torch.Tensor


@dataclass(frozen=True)
class SerializedRelation:
    text: str
    fields: dict[str, tuple[int, int]]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json(payload: Mapping) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"


_RUNTIME_WORDING = {
    runtime_type: runtime_type.lower().replace("-", " ")
    for runtime_type in RUNTIME_TYPES
}
_RUNTIME_WORDING.update({
    "PERSON": "person",
    "CODE": "code",
    "ORG": "organization",
    "LOC": "location",
    "DATETIME": "date or time",
    "QUANTITY": "quantity",
    "MISC": "information",
    "demographic-other": "demographic attribute",
    "organization-medical-facility": "medical organization or facility",
})


def placeholder_description(runtime_type: str) -> str:
    """Return ordinary type-specific placeholder wording for the frozen tokenizer."""
    try:
        wording = _RUNTIME_WORDING[runtime_type]
    except KeyError as exc:
        raise ValueError(f"unsupported runtime type: {runtime_type}") from exc
    return f"unspecified {wording}"


def serialize_relation(
    runtime_type: str, source: str, candidate: str
) -> SerializedRelation:
    """Serialize the three ordinary-text fields and their pre-tokenization spans."""
    if runtime_type not in RUNTIME_TYPES:
        raise ValueError(f"unsupported runtime type: {runtime_type}")
    for field_name, value in (
        ("type", runtime_type), ("source", source), ("candidate", candidate),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"relation {field_name} must be nonempty")
    type_start = len("TYPE: ")
    source_start = type_start + len(runtime_type) + len("\nSOURCE: ")
    candidate_start = source_start + len(source) + len("\nCANDIDATE: ")
    text = (
        f"TYPE: {runtime_type}\n"
        f"SOURCE: {source}\n"
        f"CANDIDATE: {candidate}"
    )
    return SerializedRelation(
        text=text,
        fields={
            "type": (type_start, type_start + len(runtime_type)),
            "source": (source_start, source_start + len(source)),
            "candidate": (candidate_start, candidate_start + len(candidate)),
        },
    )


def render_candidate(decision: RankerDecision, action: RankerAction) -> str:
    """Render one action without identity or training metadata."""
    if action.runtime_type != decision.runtime_type:
        raise ValueError(f"action runtime type mismatch: {action.action_id}")
    if action.mode == "keep":
        return decision.canonical_key
    if action.mode == "placeholder":
        return placeholder_description(decision.runtime_type)
    if action.mode == "level" and action.fill:
        return action.fill
    raise ValueError(f"unsupported relation action: {action.action_id}")


def representation_cache_key(
    *,
    environment_hash: str,
    source_hash: str,
    encoder_id: str,
    encoder_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    hidden_size: int,
    chunk_length: int,
    source_token_overlap: int,
    field_serialization_version: str,
    offset_format_version: str,
    candidate_only_version: str,
    independent_pair_version: str,
) -> str:
    """Hash every frozen input that can change persisted representations."""
    return _stable_hash({
        "environment_hash": environment_hash,
        "source_hash": source_hash,
        "encoder": {
            "id": encoder_id,
            "revision": encoder_revision,
            "hidden_size": hidden_size,
        },
        "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision},
        "document_encoding": {
            "chunk_length": chunk_length,
            "source_token_overlap": source_token_overlap,
            "offset_format_version": offset_format_version,
        },
        "relation_encoding": {
            "field_serialization_version": field_serialization_version,
            "candidate_only_version": candidate_only_version,
            "independent_pair_version": independent_pair_version,
        },
    })


def _one_dimensional(value: Any, field_name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(f"expected one token sequence for {field_name}")
    return tensor


def _offset_rows(value: Any) -> list[tuple[int, int]]:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("tokenizer returned invalid offset mappings")
    return [(int(start), int(end)) for start, end in tensor.tolist()]


class FrozenEncoderAdapter:
    """Injected tokenizer/model adapter shared by document and relation encoding."""

    def __init__(
        self,
        tokenizer,
        model,
        *,
        encoder_id: str = ENCODER_ID,
        encoder_revision: str = ENCODER_REVISION,
        tokenizer_id: str = TOKENIZER_ID,
        tokenizer_revision: str = TOKENIZER_REVISION,
        chunk_length: int = CHUNK_LENGTH,
        source_token_overlap: int = SOURCE_TOKEN_OVERLAP,
        field_serialization_version: str = FIELD_SERIALIZATION_VERSION,
        device: str | torch.device = "cpu",
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("ranker representations require a fast tokenizer")
        special_width = int(tokenizer.num_special_tokens_to_add(pair=False))
        payload_width = chunk_length - special_width
        if payload_width <= 0:
            raise ValueError("chunk length cannot hold source tokens")
        if source_token_overlap < 0 or source_token_overlap >= payload_width:
            raise ValueError("source token overlap must be smaller than chunk payload")
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("encoder is missing a positive hidden size")

        self.tokenizer = tokenizer
        self.model = model
        self.encoder_id = encoder_id
        self.encoder_revision = encoder_revision
        self.tokenizer_id = tokenizer_id
        self.tokenizer_revision = tokenizer_revision
        self.hidden_size = hidden_size
        self.chunk_length = chunk_length
        self.source_token_overlap = source_token_overlap
        self.field_serialization_version = field_serialization_version
        self.device = torch.device(device)
        self._payload_width = payload_width

        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _forward(self, tokenized: Mapping[str, Any]) -> torch.Tensor:
        model_inputs = {
            key: value.to(self.device)
            for key, value in tokenized.items()
            if key in getattr(self.tokenizer, "model_input_names", ())
            and isinstance(value, torch.Tensor)
        }
        if "input_ids" not in model_inputs:
            raise ValueError("tokenizer did not return input_ids")
        with torch.inference_mode():
            hidden = self.model(**model_inputs).last_hidden_state
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[2] != self.hidden_size
        ):
            raise ValueError("encoder returned an invalid hidden-state shape")
        return hidden[0].detach().to(device="cpu", dtype=torch.float32)

    def encode_document(self, doc_id: str, text: str) -> DocumentTokenBank:
        """Encode all source pieces, averaging repeated overlap states by source offset."""
        raw = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_tensors=None,
            truncation=False,
        )
        source_ids = [int(value) for value in raw["input_ids"]]
        source_offsets = [tuple(map(int, row)) for row in raw["offset_mapping"]]
        if len(source_ids) != len(source_offsets):
            raise ValueError("document token ids and offsets disagree")

        chunked = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.chunk_length,
            stride=self.source_token_overlap,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        chunk_ids = chunked["input_ids"]
        chunk_attention = chunked["attention_mask"]
        chunk_special = chunked["special_tokens_mask"]
        chunk_offsets = chunked["offset_mapping"]
        if not (
            len(chunk_ids)
            == len(chunk_attention)
            == len(chunk_special)
            == len(chunk_offsets)
        ):
            raise ValueError("document overflow chunks are misaligned")

        state_sums: dict[tuple[int, int], torch.Tensor] = {}
        appearances: dict[tuple[int, int], int] = {}
        memberships: dict[tuple[int, int], set[int]] = {}
        ordered_offsets: list[tuple[int, int]] = []
        for chunk_id, (ids, attention, special, offsets_for_chunk) in enumerate(zip(
            chunk_ids, chunk_attention, chunk_special, chunk_offsets, strict=True,
        )):
            prepared = {
                "input_ids": torch.tensor([ids], dtype=torch.int64),
                "attention_mask": torch.tensor([attention], dtype=torch.int64),
            }
            special_mask = torch.tensor(special, dtype=torch.bool)
            states = self._forward(prepared)
            retained = states[~special_mask]
            retained_offsets = [
                tuple(map(int, offset))
                for offset, is_special in zip(
                    offsets_for_chunk, special_mask.tolist(), strict=True,
                )
                if not is_special
            ]
            if retained.shape[0] != len(retained_offsets):
                raise ValueError("special-token mask does not preserve source token alignment")
            for offset, state in zip(retained_offsets, retained, strict=True):
                if offset == (0, 0):
                    continue
                if offset not in state_sums:
                    state_sums[offset] = torch.zeros_like(state)
                    appearances[offset] = 0
                    memberships[offset] = set()
                    ordered_offsets.append(offset)
                state_sums[offset] += state
                appearances[offset] += 1
                memberships[offset].add(chunk_id)

        expected_offsets = {offset for offset in source_offsets if offset != (0, 0)}
        if set(state_sums) != expected_offsets:
            missing = sorted(expected_offsets - set(state_sums))
            raise ValueError(f"document overflow omitted source offsets: {missing}")

        if ordered_offsets:
            states = torch.stack([
                state_sums[offset] / appearances[offset] for offset in ordered_offsets
            ]).to(dtype=torch.float32)
            offsets = torch.tensor(ordered_offsets, dtype=torch.int64)
        else:
            states = torch.empty((0, self.hidden_size), dtype=torch.float32)
            offsets = torch.empty((0, 2), dtype=torch.int64)
        return DocumentTokenBank(
            doc_id=doc_id,
            states=states.cpu(),
            offsets=offsets.cpu(),
            chunk_membership=tuple(
                tuple(sorted(memberships[offset])) for offset in ordered_offsets
            ),
        )

    def _tokenize_text(self, text: str) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=False,
        )
        offsets = _offset_rows(tokenized["offset_mapping"])
        special_mask = _one_dimensional(
            tokenized["special_tokens_mask"], "special_tokens_mask"
        ).to(torch.bool)
        input_ids = _one_dimensional(tokenized["input_ids"], "input_ids")
        if len(offsets) > self.chunk_length or input_ids.shape[0] > self.chunk_length:
            raise ValueError("relation text exceeds the frozen encoder input length")
        states = self._forward(tokenized)
        if states.shape[0] != len(offsets) or special_mask.shape[0] != len(offsets):
            raise ValueError("relation token outputs are misaligned")
        return states, offsets, special_mask

    def _standalone_mean(self, text: str, field_name: str) -> torch.Tensor:
        states, offsets, special_mask = self._tokenize_text(text)
        retained = [
            index for index, offset in enumerate(offsets)
            if not special_mask[index] and offset != (0, 0)
        ]
        if not retained:
            raise ValueError(f"{field_name} has no retained tokens")
        return states[retained].mean(dim=0)

    def encode_relation(
        self,
        decision_id: str,
        action_id: str,
        runtime_type: str,
        source: str,
        candidate: str,
    ) -> RelationFeatures:
        """Jointly encode ordered fields and separately encode diagnostic baselines."""
        serialized = serialize_relation(runtime_type, source, candidate)
        states, offsets, special_mask = self._tokenize_text(serialized.text)
        means = {}
        for field_name, (field_start, field_end) in serialized.fields.items():
            retained = [
                index for index, (start, end) in enumerate(offsets)
                if not special_mask[index]
                and end > start
                and end > field_start
                and start < field_end
            ]
            if not retained:
                raise ValueError(f"{field_name} field has no retained tokens")
            means[field_name] = states[retained].mean(dim=0)

        type_mean = means["type"]
        source_mean = means["source"]
        candidate_mean = means["candidate"]
        pair = torch.cat([
            type_mean,
            source_mean,
            candidate_mean,
            candidate_mean - source_mean,
            source_mean * candidate_mean,
        ])
        candidate_only = self._standalone_mean(candidate, "candidate baseline")
        independent_source = self._standalone_mean(source, "independent source")
        independent_candidate = self._standalone_mean(
            candidate, "independent candidate"
        )
        independent_pair = torch.cat([
            independent_source,
            independent_candidate,
            independent_candidate - independent_source,
            independent_source * independent_candidate,
        ])
        return RelationFeatures(
            decision_id=decision_id,
            action_id=action_id,
            type_mean=type_mean,
            source_mean=source_mean,
            candidate_mean=candidate_mean,
            pair=pair,
            candidate_only=candidate_only,
            independent_pair=independent_pair,
        )


def load_pinned_encoder(
    *, cache_only_model: bool = False, device: str | torch.device = "cpu"
) -> FrozenEncoderAdapter:
    """Load the exact frozen checkpoint, optionally forbidding all network lookup."""
    local_kwargs = {"local_files_only": True} if cache_only_model else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            revision=TOKENIZER_REVISION,
            **local_kwargs,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("ranker representations require a fast tokenizer")
        model = AutoModel.from_pretrained(
            ENCODER_ID,
            revision=ENCODER_REVISION,
            trust_remote_code=False,
            **(
                {"adapter_kwargs": {"local_files_only": True}}
                if cache_only_model else {}
            ),
            **local_kwargs,
        )
    except OSError as exc:
        if cache_only_model:
            raise RuntimeError(
                "pinned encoder snapshot is unavailable in the local model cache"
            ) from exc
        raise
    if getattr(model.config, "hidden_size", None) != HIDDEN_SIZE:
        raise ValueError(f"pinned encoder hidden size must be {HIDDEN_SIZE}")
    return FrozenEncoderAdapter(tokenizer, model, device=device)


def _adapter_identity(encoder: FrozenEncoderAdapter) -> dict:
    return {
        "encoder": {
            "id": encoder.encoder_id,
            "revision": encoder.encoder_revision,
            "hidden_size": encoder.hidden_size,
        },
        "tokenizer": {
            "id": encoder.tokenizer_id,
            "revision": encoder.tokenizer_revision,
        },
        "document_encoding": {
            "chunk_length": encoder.chunk_length,
            "source_token_overlap": encoder.source_token_overlap,
            "offset_format_version": OFFSET_FORMAT_VERSION,
        },
        "relation_encoding": {
            "field_serialization_version": encoder.field_serialization_version,
            "candidate_only_version": CANDIDATE_ONLY_VERSION,
            "independent_pair_version": INDEPENDENT_PAIR_VERSION,
        },
    }


def _representation_key(
    encoder: FrozenEncoderAdapter, *, environment_hash: str, source_hash: str
) -> str:
    return representation_cache_key(
        environment_hash=environment_hash,
        source_hash=source_hash,
        encoder_id=encoder.encoder_id,
        encoder_revision=encoder.encoder_revision,
        tokenizer_id=encoder.tokenizer_id,
        tokenizer_revision=encoder.tokenizer_revision,
        hidden_size=encoder.hidden_size,
        chunk_length=encoder.chunk_length,
        source_token_overlap=encoder.source_token_overlap,
        field_serialization_version=encoder.field_serialization_version,
        offset_format_version=OFFSET_FORMAT_VERSION,
        candidate_only_version=CANDIDATE_ONLY_VERSION,
        independent_pair_version=INDEPENDENT_PAIR_VERSION,
    )


def _tensor_path(root: Path, family: str, key: str) -> tuple[Path, str]:
    relative = f"{family}/{key.removeprefix('sha256:')}.pt"
    return root / relative, relative


def _save_tensor_payload(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _file_hash(path)


def build_representation_store(
    documents: Mapping[str, RankerDocument],
    *,
    environment_hash: str,
    out_dir: Path,
    encoder: FrozenEncoderAdapter,
) -> Path:
    """Encode documents and full policy menus into a content-addressed store."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = _adapter_identity(encoder)
    document_entries = {}
    relation_entries = {}
    encoded_relations: dict[str, dict[str, str]] = {}
    source_hashes = {
        doc_id: _text_hash(document.text)
        for doc_id, document in sorted(documents.items())
    }
    store_key = _representation_key(
        encoder,
        environment_hash=environment_hash,
        source_hash=_stable_hash(source_hashes),
    )

    for doc_id, document in sorted(documents.items()):
        if document.doc_id != doc_id:
            raise ValueError(f"document key mismatch: {doc_id}")
        source_hash = source_hashes[doc_id]
        document_key = _stable_hash({
            "representation_key": _representation_key(
                encoder,
                environment_hash=environment_hash,
                source_hash=source_hash,
            ),
            "doc_id": doc_id,
        })
        bank = encoder.encode_document(doc_id, document.text)
        tensor_path, relative_path = _tensor_path(root, "documents", document_key)
        tensor_sha = _save_tensor_payload(tensor_path, {
            "states": bank.states,
            "offsets": bank.offsets,
            "chunk_membership": [list(value) for value in bank.chunk_membership],
        })
        document_entries[doc_id] = {
            "cache_key": document_key,
            "source_hash": source_hash,
            "tensor_file": relative_path,
            "tensor_sha256": tensor_sha,
        }

        for decision in document.policy_decisions:
            for action in decision.actions:
                candidate = render_candidate(decision, action)
                serialized = serialize_relation(
                    decision.runtime_type, decision.canonical_key, candidate
                )
                relation_key = _representation_key(
                    encoder,
                    environment_hash=environment_hash,
                    source_hash=_stable_hash({
                        "runtime_type": decision.runtime_type,
                        "source": decision.canonical_key,
                        "candidate": candidate,
                    }),
                )
                if relation_key not in encoded_relations:
                    features = encoder.encode_relation(
                        decision.decision_id,
                        action.action_id,
                        decision.runtime_type,
                        decision.canonical_key,
                        candidate,
                    )
                    tensor_path, relative_path = _tensor_path(
                        root, "relations", relation_key
                    )
                    tensor_sha = _save_tensor_payload(tensor_path, {
                        "type_mean": features.type_mean,
                        "source_mean": features.source_mean,
                        "candidate_mean": features.candidate_mean,
                        "pair": features.pair,
                        "candidate_only": features.candidate_only,
                        "independent_pair": features.independent_pair,
                    })
                    encoded_relations[relation_key] = {
                        "tensor_file": relative_path,
                        "tensor_sha256": tensor_sha,
                    }
                relation_index_key = _stable_hash([
                    decision.decision_id, action.action_id,
                ])
                if relation_index_key in relation_entries:
                    raise ValueError(
                        f"duplicate relation identity: {decision.decision_id}:{action.action_id}"
                    )
                relation_entries[relation_index_key] = {
                    "decision_id": decision.decision_id,
                    "action_id": action.action_id,
                    "runtime_type": decision.runtime_type,
                    "mode": action.mode,
                    "cache_key": relation_key,
                    "serialized_text": serialized.text,
                    **encoded_relations[relation_key],
                }

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "environment_hash": environment_hash,
        "store_key": store_key,
        **identity,
        "documents": document_entries,
        "relations": relation_entries,
    }
    manifest["manifest_hash"] = _stable_hash(manifest)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(_canonical_json(manifest))
    return manifest_path


def _checked_relative_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"tensor path escapes representation store: {relative}")
    return path


class RankerRepresentationStore:
    """Read-only manifest-indexed access to validated CPU tensor payloads."""

    def __init__(self, manifest_path: Path, manifest: Mapping):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self._root = manifest_path.parent
        self._documents = dict(manifest["documents"])
        self._relations = {
            (str(row["decision_id"]), str(row["action_id"])): row
            for row in manifest["relations"].values()
        }
        if len(self._relations) != len(manifest["relations"]):
            raise ValueError("duplicate decision/action relation entry")
        self._payloads: dict[str, dict[str, Any]] = {}

    @classmethod
    def open(cls, manifest_path: Path) -> "RankerRepresentationStore":
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        if manifest.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported representation store artifact version")
        supplied_hash = manifest.get("manifest_hash")
        unhashed = {
            key: value for key, value in manifest.items() if key != "manifest_hash"
        }
        if supplied_hash != _stable_hash(unhashed):
            raise ValueError("representation manifest hash mismatch")
        for field_name in (
            "encoder", "tokenizer", "document_encoding", "relation_encoding",
            "documents", "relations",
        ):
            if not isinstance(manifest.get(field_name), Mapping):
                raise ValueError(f"representation manifest is missing {field_name}")
        return cls(path, manifest)

    def _load(self, row: Mapping) -> dict[str, Any]:
        relative = str(row["tensor_file"])
        if relative in self._payloads:
            return self._payloads[relative]
        path = _checked_relative_path(self._root, relative)
        if _file_hash(path) != row.get("tensor_sha256"):
            raise ValueError(f"tensor SHA-256 mismatch: {relative}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid tensor payload: {relative}")
        self._payloads[relative] = payload
        return payload

    def document(self, doc_id: str) -> DocumentTokenBank:
        try:
            row = self._documents[doc_id]
        except KeyError as exc:
            raise KeyError(f"unknown representation document: {doc_id}") from exc
        payload = self._load(row)
        states = payload.get("states")
        offsets = payload.get("offsets")
        membership = payload.get("chunk_membership")
        hidden_size = int(self.manifest["encoder"]["hidden_size"])
        if (
            not isinstance(states, torch.Tensor)
            or states.dtype != torch.float32
            or states.ndim != 2
            or states.shape[1] != hidden_size
            or states.device.type != "cpu"
        ):
            raise ValueError(f"invalid document states: {doc_id}")
        if (
            not isinstance(offsets, torch.Tensor)
            or offsets.dtype != torch.int64
            or offsets.shape != (states.shape[0], 2)
            or offsets.device.type != "cpu"
        ):
            raise ValueError(f"invalid document offsets: {doc_id}")
        if not isinstance(membership, Sequence) or len(membership) != states.shape[0]:
            raise ValueError(f"invalid document chunk membership: {doc_id}")
        offset_rows = [tuple(map(int, value)) for value in offsets.tolist()]
        if len(offset_rows) != len(set(offset_rows)):
            raise ValueError(f"duplicate document offsets: {doc_id}")
        return DocumentTokenBank(
            doc_id=doc_id,
            states=states,
            offsets=offsets,
            chunk_membership=tuple(
                tuple(int(value) for value in chunks) for chunks in membership
            ),
        )

    def relation(self, decision_id: str, action_id: str) -> RelationFeatures:
        try:
            row = self._relations[(decision_id, action_id)]
        except KeyError as exc:
            raise KeyError(
                f"unknown representation relation: {decision_id}:{action_id}"
            ) from exc
        payload = self._load(row)
        hidden_size = int(self.manifest["encoder"]["hidden_size"])
        expected_shapes = {
            "type_mean": (hidden_size,),
            "source_mean": (hidden_size,),
            "candidate_mean": (hidden_size,),
            "pair": (5 * hidden_size,),
            "candidate_only": (hidden_size,),
            "independent_pair": (4 * hidden_size,),
        }
        tensors = {}
        for name, expected_shape in expected_shapes.items():
            tensor = payload.get(name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != expected_shape
            ):
                raise ValueError(
                    f"invalid relation tensor {name}: {decision_id}:{action_id}"
                )
            tensors[name] = tensor
        return RelationFeatures(
            decision_id=decision_id,
            action_id=action_id,
            **tensors,
        )
