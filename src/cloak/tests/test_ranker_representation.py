from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)
from cloak.train.ranker_representation import (
    CANDIDATE_ONLY_VERSION,
    CHUNK_LENGTH,
    ENCODER_ID,
    ENCODER_REVISION,
    FIELD_SERIALIZATION_VERSION,
    HIDDEN_SIZE,
    INDEPENDENT_PAIR_VERSION,
    OFFSET_FORMAT_VERSION,
    SOURCE_TOKEN_OVERLAP,
    DocumentTokenBank,
    FrozenEncoderAdapter,
    RankerRepresentationStore,
    RelationFeatures,
    build_representation_store,
    load_pinned_encoder,
    placeholder_description,
    render_candidate,
    representation_cache_key,
    serialize_relation,
)


class StubTokenizer:
    is_fast = True
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *, dropped_terms: frozenset[str] = frozenset()):
        self.dropped_terms = dropped_terms
        self.seen_texts: list[str] = []

    @staticmethod
    def _token_id(token: str) -> int:
        return 3 + sum((index + 1) * ord(char) for index, char in enumerate(token)) % 997

    def _plain(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids = []
        offsets = []
        for match in re.finditer(r"\S+", text):
            if match.group() in self.dropped_terms:
                continue
            ids.append(self._token_id(match.group()))
            offsets.append(match.span())
        return ids, offsets

    @staticmethod
    def _tensorize(payload: dict, return_tensors: str | None) -> dict:
        if return_tensors != "pt":
            return payload
        return {
            key: torch.tensor([value], dtype=torch.long)
            for key, value in payload.items()
        }

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
        return_special_tokens_mask: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        stride: int = 0,
        return_overflowing_tokens: bool = False,
    ) -> dict:
        self.seen_texts.append(text)
        ids, offsets = self._plain(text)
        if return_overflowing_tokens:
            assert truncation is True and max_length is not None
            assert add_special_tokens is True and return_tensors is None
            payload_width = max_length - 2
            chunk_stride = payload_width - stride
            rows = []
            for start in range(0, len(ids), chunk_stride):
                stop = min(start + payload_width, len(ids))
                rows.append((ids[start:stop], offsets[start:stop]))
                if stop == len(ids):
                    break
            return {
                "input_ids": [[101, *row_ids, 102] for row_ids, _ in rows],
                "attention_mask": [[1] * (len(row_ids) + 2) for row_ids, _ in rows],
                "special_tokens_mask": [
                    [1, *([0] * len(row_ids)), 1] for row_ids, _ in rows
                ],
                "offset_mapping": [
                    [(0, 0), *row_offsets, (0, 0)] for _, row_offsets in rows
                ],
                "overflow_to_sample_mapping": [0] * len(rows),
            }
        assert truncation is False
        special = [0] * len(ids)
        if add_special_tokens:
            ids = [101, *ids, 102]
            offsets = [(0, 0), *offsets, (0, 0)]
            special = [1, *special, 1]
        payload = {"input_ids": ids}
        if return_attention_mask:
            payload["attention_mask"] = [1] * len(ids)
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        if return_special_tokens_mask:
            payload["special_tokens_mask"] = special
        return self._tensorize(payload, return_tensors)

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2


class StubEncoder(nn.Module):
    def __init__(self, hidden_size: int = 6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        values = input_ids.to(torch.float32)
        positions = torch.arange(values.shape[1], device=values.device).expand_as(values)
        context = (values * attention_mask).sum(dim=1, keepdim=True).expand_as(values)
        basis = [
            values,
            positions,
            context,
            values + context,
            values - positions,
            context - values,
        ]
        hidden = torch.stack(basis[: self.config.hidden_size], dim=-1) * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


def _adapter(
    *,
    tokenizer: StubTokenizer | None = None,
    encoder_revision: str = ENCODER_REVISION,
    tokenizer_revision: str = ENCODER_REVISION,
    chunk_length: int = 16,
    overlap: int = 2,
    serialization_version: str = FIELD_SERIALIZATION_VERSION,
) -> FrozenEncoderAdapter:
    return FrozenEncoderAdapter(
        tokenizer or StubTokenizer(),
        StubEncoder(),
        encoder_id=ENCODER_ID,
        encoder_revision=encoder_revision,
        tokenizer_id=ENCODER_ID,
        tokenizer_revision=tokenizer_revision,
        chunk_length=chunk_length,
        source_token_overlap=overlap,
        field_serialization_version=serialization_version,
        device="cpu",
    )


def _decision(
    decision_id: str = "decision-1",
    *,
    source: str = "kidney transplant",
    runtime_type: str = "health-condition",
) -> RankerDecision:
    actions = (
        RankerAction(f"{decision_id}-keep", "keep", source, None, runtime_type),
        RankerAction(
            f"{decision_id}-level", "level", "solid organ transplant", 0, runtime_type,
        ),
        RankerAction(f"{decision_id}-placeholder", "placeholder", None, None, runtime_type),
    )
    return RankerDecision(
        decision_id=decision_id,
        profile_id=f"{runtime_type}:{source}",
        runtime_type=runtime_type,
        canonical_key=source,
        occurrence_ids=(f"occurrence-{decision_id}",),
        actions=actions,
    )


def _document(
    doc_id: str = "fixture/doc-1", *, text: str | None = None
) -> RankerDocument:
    return RankerDocument(
        doc_id=doc_id,
        corpus="fixture",
        text=text or "one two three four five six seven eight nine ten eleven twelve",
        occurrences=(),
        policy_decisions=(_decision(),),
        fixed_decisions=(),
    )


def test_public_types_and_pins_are_exact_and_frozen():
    assert ENCODER_ID == "thomas-sounack/BioClinical-ModernBERT-base"
    assert ENCODER_REVISION == "c3648aa87af95837c809e6f0c5f85d08160db437"
    assert CHUNK_LENGTH == 512
    assert SOURCE_TOKEN_OVERLAP == 64
    assert [field.name for field in fields(DocumentTokenBank)] == [
        "doc_id", "states", "offsets", "chunk_membership",
    ]
    assert [field.name for field in fields(RelationFeatures)] == [
        "decision_id", "action_id", "type_mean", "source_mean", "candidate_mean",
        "pair", "candidate_only", "independent_pair",
    ]
    bank = DocumentTokenBank(
        "doc", torch.zeros(1, 6), torch.zeros(1, 2, dtype=torch.int64), ((0,),)
    )
    with pytest.raises(FrozenInstanceError):
        bank.doc_id = "other"


def test_relation_serialization_and_action_rendering_are_metadata_free():
    decision = _decision()
    keep, level, placeholder = decision.actions

    serialized = serialize_relation(
        decision.runtime_type, decision.canonical_key, level.fill
    )

    assert serialized.text == (
        "TYPE: health-condition\n"
        "SOURCE: kidney transplant\n"
        "CANDIDATE: solid organ transplant"
    )
    assert serialized.fields == {
        "type": (6, 22),
        "source": (31, 48),
        "candidate": (60, 82),
    }
    assert render_candidate(decision, keep) == "kidney transplant"
    assert render_candidate(decision, level) == "solid organ transplant"
    assert render_candidate(decision, placeholder) == "unspecified health condition"
    assert all(
        forbidden not in serialized.text
        for forbidden in (decision.decision_id, level.action_id, "<TYPE_", "100", "0.5")
    )


@pytest.mark.parametrize(
    ("runtime_type", "expected"),
    [
        ("health-condition", "unspecified health condition"),
        ("drug", "unspecified drug"),
        ("medical-procedure", "unspecified medical procedure"),
        ("LOC", "unspecified location"),
    ],
)
def test_placeholder_descriptions_are_type_specific(runtime_type: str, expected: str):
    assert placeholder_description(runtime_type) == expected


def test_adapter_freezes_encoder_and_rejects_slow_tokenizer():
    encoder = StubEncoder()
    adapter = FrozenEncoderAdapter(
        StubTokenizer(), encoder, chunk_length=8, source_token_overlap=2, device="cpu"
    )

    assert adapter.model.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    slow = StubTokenizer()
    slow.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        FrozenEncoderAdapter(
            slow, StubEncoder(), chunk_length=8, source_token_overlap=2, device="cpu"
        )


def test_document_bank_averages_overlap_into_one_row_per_source_offset():
    adapter = _adapter(chunk_length=6, overlap=2)

    bank = adapter.encode_document("doc", "zero one two three four five six seven eight")

    assert bank.states.shape == (9, 6)
    assert bank.states.dtype == torch.float32
    assert bank.states.device.type == "cpu"
    assert bank.offsets.dtype == torch.int64
    assert len({tuple(row) for row in bank.offsets.tolist()}) == 9
    assert bank.chunk_membership[2] == (0, 1)
    assert bank.chunk_membership[4] == (1, 2)
    assert (0, 0) not in {tuple(row) for row in bank.offsets.tolist()}


def test_document_chunking_uses_only_the_fast_tokenizer_overflow_protocol():
    tokenizer = StubTokenizer()
    assert not hasattr(tokenizer, "prepare_for_model")
    adapter = FrozenEncoderAdapter(
        tokenizer, StubEncoder(), chunk_length=6, source_token_overlap=2, device="cpu"
    )

    bank = adapter.encode_document(
        "doc", "zero one two three four five six seven eight"
    )

    assert bank.offsets.shape == (9, 2)
    assert bank.chunk_membership[2] == (0, 1)


def test_relation_pooling_is_field_aware_ordered_and_uses_separate_baselines():
    tokenizer = StubTokenizer()
    adapter = _adapter(tokenizer=tokenizer)

    forward = adapter.encode_relation(
        "d", "a", "health-condition", "kidney transplant", "solid organ transplant"
    )
    reverse = adapter.encode_relation(
        "d", "b", "health-condition", "solid organ transplant", "kidney transplant"
    )

    assert forward.type_mean.shape == (6,)
    assert forward.pair.shape == (30,)
    assert forward.candidate_only.shape == (6,)
    assert forward.independent_pair.shape == (24,)
    assert torch.equal(
        forward.pair,
        torch.cat([
            forward.type_mean,
            forward.source_mean,
            forward.candidate_mean,
            forward.candidate_mean - forward.source_mean,
            forward.source_mean * forward.candidate_mean,
        ]),
    )
    signed = slice(18, 24)
    assert not torch.equal(forward.pair[signed], reverse.pair[signed])
    assert "solid organ transplant" in tokenizer.seen_texts
    assert "kidney transplant" in tokenizer.seen_texts
    assert not torch.equal(forward.candidate_only, forward.candidate_mean)


def test_relation_rejects_nonempty_field_with_no_retained_tokens():
    adapter = _adapter(tokenizer=StubTokenizer(dropped_terms=frozenset({"kidney"})))

    with pytest.raises(ValueError, match="source field has no retained tokens"):
        adapter.encode_relation(
            "d", "a", "drug", "kidney", "medicine"
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("encoder_revision", "different-model-revision"),
        ("tokenizer_revision", "different-tokenizer-revision"),
        ("chunk_length", 256),
        ("source_token_overlap", 32),
        ("field_serialization_version", "relation-fields-v2"),
        ("source_hash", "sha256:different-source"),
    ],
)
def test_cache_identity_changes_for_every_frozen_input(field: str, replacement):
    inputs = {
        "environment_hash": "sha256:environment",
        "source_hash": "sha256:source",
        "encoder_id": ENCODER_ID,
        "encoder_revision": ENCODER_REVISION,
        "tokenizer_id": ENCODER_ID,
        "tokenizer_revision": ENCODER_REVISION,
        "hidden_size": 768,
        "chunk_length": CHUNK_LENGTH,
        "source_token_overlap": SOURCE_TOKEN_OVERLAP,
        "field_serialization_version": FIELD_SERIALIZATION_VERSION,
        "offset_format_version": OFFSET_FORMAT_VERSION,
        "candidate_only_version": CANDIDATE_ONLY_VERSION,
        "independent_pair_version": INDEPENDENT_PAIR_VERSION,
    }
    baseline = representation_cache_key(**inputs)

    inputs[field] = replacement

    assert representation_cache_key(**inputs) != baseline


def test_content_addressed_store_round_trip_and_hash_validation(tmp_path: Path):
    adapter = _adapter()
    manifest_path = build_representation_store(
        {"fixture/doc-1": _document()},
        environment_hash="sha256:environment",
        out_dir=tmp_path / "store",
        encoder=adapter,
    )

    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    assert manifest_text == json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"
    assert manifest["artifact_version"] == "ranker-v2-representation-store-v1"
    assert manifest["environment_hash"] == "sha256:environment"
    assert manifest["encoder"] == {
        "id": ENCODER_ID, "revision": ENCODER_REVISION, "hidden_size": 6,
    }
    assert manifest["document_encoding"] == {
        "chunk_length": 16,
        "source_token_overlap": 2,
        "offset_format_version": OFFSET_FORMAT_VERSION,
    }
    assert manifest["relation_encoding"] == {
        "field_serialization_version": FIELD_SERIALIZATION_VERSION,
        "candidate_only_version": CANDIDATE_ONLY_VERSION,
        "independent_pair_version": INDEPENDENT_PAIR_VERSION,
    }
    assert len(manifest["documents"]) == 1
    assert len(manifest["relations"]) == 3
    assert len({row["tensor_file"] for row in manifest["relations"].values()}) == 3
    assert all(row["source_hash"].startswith("sha256:") for row in manifest["documents"].values())
    assert all(row["tensor_sha256"].startswith("sha256:") for row in manifest["relations"].values())

    store = RankerRepresentationStore.open(manifest_path)
    bank = store.document("fixture/doc-1")
    relation = store.relation("decision-1", "decision-1-level")
    assert bank.doc_id == "fixture/doc-1"
    assert relation.decision_id == "decision-1"
    assert relation.action_id == "decision-1-level"
    assert relation.pair.shape == (30,)

    relation_row = next(iter(manifest["relations"].values()))
    tensor_path = manifest_path.parent / relation_row["tensor_file"]
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
    corrupt_store = RankerRepresentationStore.open(manifest_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        corrupt_store.relation(
            relation_row["decision_id"], relation_row["action_id"]
        )


def test_relations_enumerate_full_policy_menu_and_deduplicate_semantics(tmp_path: Path):
    first = _document("fixture/doc-1")
    duplicate_decision = _decision("decision-2")
    second = RankerDocument(
        doc_id="fixture/doc-2",
        corpus="fixture",
        text="a separate source document",
        occurrences=(),
        policy_decisions=(duplicate_decision,),
        fixed_decisions=(),
    )

    manifest_path = build_representation_store(
        {first.doc_id: first, second.doc_id: second},
        environment_hash="sha256:environment",
        out_dir=tmp_path / "store",
        encoder=_adapter(),
    )
    manifest = json.loads(manifest_path.read_text())

    assert len(manifest["relations"]) == 6
    assert len({row["tensor_file"] for row in manifest["relations"].values()}) == 3
    assert len(list((manifest_path.parent / "documents").glob("*.pt"))) == 2
    assert len(list((manifest_path.parent / "relations").glob("*.pt"))) == 3


def test_relation_cache_identity_ignores_mode_when_serialized_input_is_identical(
    tmp_path: Path,
):
    decision = _decision()
    duplicate_level = RankerAction(
        "decision-1-duplicate-level",
        "level",
        decision.canonical_key,
        1,
        decision.runtime_type,
    )
    document = RankerDocument(
        doc_id="fixture/doc-1",
        corpus="fixture",
        text="a source document",
        occurrences=(),
        policy_decisions=(RankerDecision(
            decision_id=decision.decision_id,
            profile_id=decision.profile_id,
            runtime_type=decision.runtime_type,
            canonical_key=decision.canonical_key,
            occurrence_ids=decision.occurrence_ids,
            actions=(*decision.actions, duplicate_level),
        ),),
        fixed_decisions=(),
    )

    manifest_path = build_representation_store(
        {document.doc_id: document},
        environment_hash="sha256:environment",
        out_dir=tmp_path / "store",
        encoder=_adapter(),
    )
    manifest = json.loads(manifest_path.read_text())
    rows = {
        row["action_id"]: row for row in manifest["relations"].values()
    }

    assert len(rows) == 4
    assert (
        rows["decision-1-keep"]["cache_key"]
        == rows["decision-1-duplicate-level"]["cache_key"]
    )
    assert len({row["tensor_file"] for row in rows.values()}) == 3


def test_content_keys_are_stable_when_a_tiny_slice_expands(tmp_path: Path):
    first = _document("fixture/doc-1")
    second = RankerDocument(
        doc_id="fixture/doc-2",
        corpus="fixture",
        text="a separate source document",
        occurrences=(),
        policy_decisions=(_decision("decision-2"),),
        fixed_decisions=(),
    )
    slice_path = build_representation_store(
        {first.doc_id: first},
        environment_hash="sha256:environment",
        out_dir=tmp_path / "slice",
        encoder=_adapter(),
    )
    full_path = build_representation_store(
        {first.doc_id: first, second.doc_id: second},
        environment_hash="sha256:environment",
        out_dir=tmp_path / "full",
        encoder=_adapter(),
    )
    sliced = json.loads(slice_path.read_text())
    full = json.loads(full_path.read_text())

    assert (
        sliced["documents"][first.doc_id]["cache_key"]
        == full["documents"][first.doc_id]["cache_key"]
    )
    sliced_relations = {
        row["action_id"]: row["cache_key"] for row in sliced["relations"].values()
    }
    full_relations = {
        row["action_id"]: row["cache_key"]
        for row in full["relations"].values()
        if row["decision_id"] == "decision-1"
    }
    assert sliced_relations == full_relations


def test_default_factory_is_pinned_frozen_and_cache_only(monkeypatch):
    module = importlib.import_module("cloak.train.ranker_representation")
    calls = []

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return StubTokenizer()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return StubEncoder(HIDDEN_SIZE)

    monkeypatch.setattr(module, "AutoTokenizer", TokenizerFactory)
    monkeypatch.setattr(module, "AutoModel", ModelFactory)

    adapter = load_pinned_encoder(cache_only_model=True, device="cpu")

    assert adapter.model.training is False
    assert calls == [
        ("tokenizer", ENCODER_ID, {
            "revision": ENCODER_REVISION,
            "local_files_only": True,
        }),
        ("model", ENCODER_ID, {
            "revision": ENCODER_REVISION,
            "trust_remote_code": False,
            "local_files_only": True,
        }),
    ]


def test_cache_only_factory_stops_before_model_load_when_snapshot_is_absent(monkeypatch):
    module = importlib.import_module("cloak.train.ranker_representation")

    class MissingTokenizerFactory:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            raise OSError("not cached")

    class ForbiddenModelFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("model loading must not be attempted")

    monkeypatch.setattr(module, "AutoTokenizer", MissingTokenizerFactory)
    monkeypatch.setattr(module, "AutoModel", ForbiddenModelFactory)

    with pytest.raises(RuntimeError, match="unavailable in the local model cache"):
        load_pinned_encoder(cache_only_model=True, device="cpu")


def test_cli_builds_only_requested_document_with_injected_stub(
    tmp_path: Path, monkeypatch
):
    module = importlib.import_module("build_ranker_representation_cache")
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps({
        "artifact_version": "ranker-v2-environment-v2",
        "frozen_environment": {"environment_hash": "sha256:environment"},
    }))
    documents = {
        "fixture/doc-1": _document("fixture/doc-1"),
        "fixture/doc-2": _document("fixture/doc-2", text="different document text"),
    }
    calls = []
    monkeypatch.setattr(module, "load_ranker_environment", lambda _path: documents)
    monkeypatch.setattr(
        module,
        "load_pinned_encoder",
        lambda *, cache_only_model, device: (
            calls.append((cache_only_model, device)) or _adapter()
        ),
    )
    out_dir = tmp_path / "store"
    monkeypatch.setattr(sys, "argv", [
        "build_ranker_representation_cache.py",
        "--environment", str(environment_path),
        "--out-dir", str(out_dir),
        "--doc-id", "fixture/doc-2",
        "--cache-only-model",
        "--device", "cpu",
    ])

    module.main()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert calls == [(True, "cpu")]
    assert set(manifest["documents"]) == {"fixture/doc-2"}
    assert {
        row["decision_id"] for row in manifest["relations"].values()
    } == {"decision-1"}


def test_manifest_hash_rejects_tampering(tmp_path: Path):
    manifest_path = build_representation_store(
        {"fixture/doc-1": _document()},
        environment_hash="sha256:environment",
        out_dir=tmp_path / "store",
        encoder=_adapter(),
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["environment_hash"] = "sha256:tampered"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        RankerRepresentationStore.open(manifest_path)
