"""Validated append-only cache for complete ranker-v2 utility round trips."""
from __future__ import annotations

import hashlib
import json
import math
import os
import fcntl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cloak.train.ranker_environment import RankerDocument


CACHE_VERSION = 1
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class UtilityResult:
    doc_id: str
    action_vector: Mapping[str, str]
    doc_p: str
    out_p: str
    out_final: str
    component_scores: Mapping[str, float]
    utility: float
    result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_vector", MappingProxyType(dict(self.action_vector)))
        object.__setattr__(
            self, "component_scores", MappingProxyType(dict(self.component_scores)),
        )


@dataclass(frozen=True)
class UtilityRequest:
    """One complete policy action vector against one frozen utility artifact."""

    document: RankerDocument
    action_vector: Mapping[str, str]
    utility_artifact: Mapping
    environment_hash: str


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def utility_binding(artifact: Mapping, doc_id: str) -> dict[str, Any]:
    """Freeze the exact accepted-assertion aggregation contract for one document."""
    try:
        document = artifact["documents"][doc_id]
        assertions = artifact["assertions"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"utility artifact lacks document {doc_id!r}") from error
    ids = [
        str(assertion_id) for assertion_id in document.get("assertion_ids", ())
        if assertions[str(assertion_id)].get("status", "accepted") == "accepted"
    ]
    if not ids:
        ids = sorted(
            str(assertion_id) for assertion_id, row in assertions.items()
            if row.get("doc_id") == doc_id
            and row.get("status", "accepted") == "accepted"
        )
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("utility binding requires unique accepted assertion IDs")
    rows = {assertion_id: assertions[assertion_id] for assertion_id in ids}
    weights = {assertion_id: rows[assertion_id]["weight"] for assertion_id in ids}
    binding = {
        "assertion_ids": ids,
        "weights": weights,
        "utility_weight_denominator": document["utility_weight_denominator"],
        "assertion_binding_hash": stable_hash(rows),
    }
    _validate_binding(binding)
    return binding


def result_hash(result: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in result.items() if key != "result_hash"})


def make_result(
    *,
    doc_id: str,
    action_vector: Mapping[str, str],
    doc_p: str,
    out_p: str,
    out_final: str,
    component_scores: Mapping[str, float],
    utility: float,
) -> UtilityResult:
    payload = {
        "doc_id": str(doc_id),
        "action_vector": {str(key): str(value) for key, value in action_vector.items()},
        "doc_p": str(doc_p),
        "out_p": str(out_p),
        "out_final": str(out_final),
        "component_scores": {
            str(key): float(value) for key, value in component_scores.items()
        },
        "utility": float(utility),
    }
    return UtilityResult(**payload, result_hash=result_hash(payload))


def _valid_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _validate_binding(binding: Any) -> None:
    if not isinstance(binding, Mapping):
        raise ValueError("cache identity lacks a valid utility binding")
    ids = binding.get("assertion_ids")
    weights = binding.get("weights")
    denominator = binding.get("utility_weight_denominator")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or not isinstance(weights, Mapping)
        or set(weights) != set(ids)
        or not isinstance(binding.get("assertion_binding_hash"), str)
        or not binding["assertion_binding_hash"].startswith("sha256:")
        or isinstance(denominator, bool)
        or not isinstance(denominator, int | float)
        or not math.isfinite(float(denominator))
        or float(denominator) <= 0.0
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
            for weight in weights.values()
        )
    ):
        raise ValueError("cache identity lacks a valid utility binding")


_REQUIRED_IDENTITY_FIELDS = frozenset({
    "doc_id",
    "ordered_action_vector",
    "doc_p_hash",
    "environment_hash",
    "utility_artifact_hash",
    "utility_binding",
    "task_prompt_pin",
    "remote_model_pin",
    "extractor_pin",
    "reader_pin",
    "runtime_scorer_version",
    "execution_contract_version",
    "reader_refresh",
})


def _validate_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or set(identity) != _REQUIRED_IDENTITY_FIELDS:
        raise ValueError("cache entry has incomplete request identity pins")
    canonical = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    ordered_vector = canonical["ordered_action_vector"]
    if (
        not isinstance(canonical["doc_id"], str)
        or not canonical["doc_id"]
        or not isinstance(ordered_vector, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
            for pair in ordered_vector
        )
        or len({pair[0] for pair in ordered_vector}) != len(ordered_vector)
        or not isinstance(canonical["reader_refresh"], bool)
    ):
        raise ValueError("cache entry has invalid request identity")
    for name in (
        "doc_p_hash", "environment_hash", "utility_artifact_hash",
        "runtime_scorer_version", "execution_contract_version",
    ):
        if not isinstance(canonical[name], str) or not canonical[name]:
            raise ValueError(f"cache identity has invalid {name}")
    _validate_pin_schemas(canonical)
    _validate_binding(canonical["utility_binding"])
    return canonical


def _validate_pin_schemas(identity: Mapping[str, Any]) -> None:
    task = identity.get("task_prompt_pin")
    if (
        not isinstance(task, Mapping)
        or set(task) != {"version", "corpus", "template_hash"}
        or any(not isinstance(task[key], str) or not task[key] for key in task)
        or not task["template_hash"].startswith("sha256:")
    ):
        raise ValueError("cache identity has invalid task_prompt_pin")
    remote = identity.get("remote_model_pin")
    if (
        not isinstance(remote, Mapping)
        or set(remote) != {
            "model", "endpoint", "temperature", "max_tokens", "thinking",
            "single_flight",
        }
        or not isinstance(remote["model"], str) or not remote["model"]
        or not isinstance(remote["endpoint"], str) or not remote["endpoint"]
        or isinstance(remote["temperature"], bool)
        or not isinstance(remote["temperature"], int | float)
        or not math.isfinite(float(remote["temperature"]))
        or isinstance(remote["max_tokens"], bool)
        or not isinstance(remote["max_tokens"], int)
        or remote["max_tokens"] <= 0
        or not isinstance(remote["thinking"], bool)
        or not isinstance(remote["single_flight"], bool)
    ):
        raise ValueError("cache identity has invalid remote_model_pin")
    extractor = identity.get("extractor_pin")
    if (
        not isinstance(extractor, Mapping)
        or not {
            "kind", "version", "pin_version", "semantic", "semantic_model",
            "modules", "packages", "sha256",
        } <= set(extractor)
        or not isinstance(extractor["sha256"], str)
        or extractor["sha256"] != stable_hash({
            key: value for key, value in extractor.items() if key != "sha256"
        })
    ):
        raise ValueError("cache identity has invalid extractor_pin")
    reader = identity.get("reader_pin")
    if (
        not isinstance(reader, Mapping)
        or set(reader) != {
            "model", "endpoint", "prompt_version", "response_schema", "revision",
        }
        or any(
            not isinstance(reader[key], str) or not reader[key]
            for key in ("model", "endpoint", "prompt_version", "revision")
        )
        or not isinstance(reader["response_schema"], Mapping)
        or not reader["response_schema"]
    ):
        raise ValueError("cache identity has invalid reader_pin")


def _result_mapping(result: UtilityResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, UtilityResult):
        return {
            "doc_id": result.doc_id,
            "action_vector": dict(result.action_vector),
            "doc_p": result.doc_p,
            "out_p": result.out_p,
            "out_final": result.out_final,
            "component_scores": dict(result.component_scores),
            "utility": result.utility,
            "result_hash": result.result_hash,
        }
    if isinstance(result, Mapping):
        return dict(result)
    raise ValueError("utility cache requires a complete UtilityResult")


def _validate_result(
    result: UtilityResult | Mapping[str, Any], identity: Mapping[str, Any],
) -> UtilityResult:
    payload = _result_mapping(result)
    required = {field.name for field in UtilityResult.__dataclass_fields__.values()}
    if set(payload) != required:
        raise ValueError("utility cache requires a complete UtilityResult")
    if (
        payload["doc_id"] != identity["doc_id"]
        or not isinstance(payload["doc_p"], str)
        or stable_hash(payload["doc_p"]) != identity["doc_p_hash"]
        or not isinstance(payload["out_p"], str)
        or not isinstance(payload["out_final"], str)
        or not isinstance(payload["action_vector"], Mapping)
        or dict(payload["action_vector"])
        != dict(identity["ordered_action_vector"])
        or not isinstance(payload["component_scores"], Mapping)
        or not _valid_score(payload["utility"])
    ):
        raise ValueError("utility cache result does not match its request identity")
    binding = identity["utility_binding"]
    if set(payload["component_scores"]) != set(binding["assertion_ids"]):
        raise ValueError("utility cache component assertion set does not match binding")
    if any(
        not isinstance(assertion_id, str) or not _valid_score(score)
        for assertion_id, score in payload["component_scores"].items()
    ):
        raise ValueError("utility cache has an invalid component score")
    expected = sum(
        float(binding["weights"][assertion_id])
        * float(payload["component_scores"][assertion_id])
        for assertion_id in binding["assertion_ids"]
    ) / float(binding["utility_weight_denominator"])
    if not math.isclose(
        float(payload["utility"]), expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError("utility cache result violates fixed-denominator parity")
    if payload["result_hash"] != result_hash(payload):
        raise ValueError("utility cache result hash mismatch")
    return UtilityResult(
        doc_id=payload["doc_id"],
        action_vector={
            decision_id: payload["action_vector"][decision_id]
            for decision_id, _action_id in identity["ordered_action_vector"]
        },
        doc_p=payload["doc_p"],
        out_p=payload["out_p"],
        out_final=payload["out_final"],
        component_scores={
            str(key): float(value) for key, value in payload["component_scores"].items()
        },
        utility=float(payload["utility"]),
        result_hash=payload["result_hash"],
    )


class UtilityCache:
    """Append-only JSONL store that admits only complete, self-validating results."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries = self._load()
        self.hits = 0
        self.misses = 0
        self.last_batch_metrics: dict[str, Any] = {}

    def _load(self) -> dict[str, tuple[dict[str, Any], UtilityResult]]:
        if not self.path.exists():
            return {}
        try:
            return self._decode(self.path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid utility cache {self.path}: {error}") from error

    @classmethod
    def _decode(cls, payload: bytes) -> dict[str, tuple[dict[str, Any], UtilityResult]]:
        if payload and not payload.endswith(b"\n"):
            raise ValueError("truncated JSONL record")
        entries: dict[str, tuple[dict[str, Any], UtilityResult]] = {}
        for line_number, encoded in enumerate(payload.splitlines(), 1):
            if not encoded:
                raise ValueError(f"blank JSONL record at line {line_number}")
            record = json.loads(encoded.decode("utf-8"))
            request_identity, identity, result = cls._validate_record(record)
            previous = entries.get(request_identity)
            current = (identity, result)
            if previous is not None and previous != current:
                raise ValueError(
                    f"conflicting duplicate request identity at line {line_number}"
                )
            entries.setdefault(request_identity, current)
        return entries

    @staticmethod
    def request_identity(identity: Mapping[str, Any]) -> str:
        return stable_hash(_validate_identity(identity))

    @classmethod
    def _validate_record(
        cls, record: Any,
    ) -> tuple[str, dict[str, Any], UtilityResult]:
        if not isinstance(record, Mapping) or record.get("version") != CACHE_VERSION:
            raise ValueError("invalid cache record schema")
        identity = _validate_identity(record.get("identity"))
        request_identity = cls.request_identity(identity)
        if record.get("request_identity") != request_identity:
            raise ValueError("cache request identity mismatch")
        result = _validate_result(record.get("result"), identity)
        if record.get("result_hash") != result.result_hash:
            raise ValueError("cache record result hash mismatch")
        storage_identity = stable_hash({
            "request_identity": request_identity,
            "result_hash": result.result_hash,
        })
        if record.get("storage_identity") != storage_identity:
            raise ValueError("cache storage identity mismatch")
        if set(record) != {
            "version", "identity", "request_identity", "result", "result_hash",
            "storage_identity",
        }:
            raise ValueError("invalid cache record schema")
        return request_identity, identity, result

    def lookup(self, identity: Mapping[str, Any]) -> UtilityResult | None:
        canonical = _validate_identity(identity)
        request_identity = self.request_identity(canonical)
        entry = self.entries.get(request_identity)
        if entry is None:
            self.misses += 1
            return None
        stored_identity, result = entry
        if stored_identity != canonical:
            raise ValueError("cache request hash collision")
        self.hits += 1
        return result

    def store_many(
        self,
        items: Sequence[tuple[Mapping[str, Any], UtilityResult]],
    ) -> list[UtilityResult]:
        staged: dict[str, tuple[dict[str, Any], UtilityResult]] = {}
        records: list[dict[str, Any]] = []
        output: list[UtilityResult] = []
        for raw_identity, raw_result in items:
            identity = _validate_identity(raw_identity)
            request_identity = self.request_identity(identity)
            result = _validate_result(raw_result, identity)
            current = (identity, result)
            previous = staged.get(request_identity, self.entries.get(request_identity))
            if previous is not None and previous != current:
                raise ValueError("conflicting duplicate utility cache identity")
            if request_identity not in self.entries and request_identity not in staged:
                staged[request_identity] = current
                records.append({
                    "version": CACHE_VERSION,
                    "identity": identity,
                    "request_identity": request_identity,
                    "result": _result_mapping(result),
                    "result_hash": result.result_hash,
                    "storage_identity": stable_hash({
                        "request_identity": request_identity,
                        "result_hash": result.result_hash,
                    }),
                })
            output.append(result)
        if records:
            self._append(records)
        return output

    def _append(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                current = self._decode(handle.read())
                new_records = []
                for record in records:
                    request_identity, identity, result = self._validate_record(record)
                    entry = (identity, result)
                    previous = current.get(request_identity)
                    if previous is not None and previous != entry:
                        raise ValueError("conflicting duplicate utility cache identity")
                    if previous is None:
                        current[request_identity] = entry
                        new_records.append(record)
                if new_records:
                    payload = "".join(
                        json.dumps(
                            record, sort_keys=True, separators=(",", ":"),
                            allow_nan=False,
                        ) + "\n"
                        for record in new_records
                    ).encode("utf-8")
                    handle.seek(0, os.SEEK_END)
                    written = os.write(handle.fileno(), payload)
                    if written != len(payload):
                        raise OSError("partial utility cache append")
                    os.fsync(handle.fileno())
                self.entries = current
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
