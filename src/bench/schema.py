from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
from typing import Iterable, Mapping, TypeAlias

JsonDict: TypeAlias = dict[str, object]


def stable_hash(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def jsonl_write(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[JsonDict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass(frozen=True)
class SensitiveSpan:
    span_id: str
    surface: str
    start: int
    end: int
    type: str
    identifier_class: str
    subject_id: str
    task_relevance: str
    reference_evidence: list[str]

    def to_json(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "SensitiveSpan":
        return cls(
            span_id=str(row["span_id"]),
            surface=str(row["surface"]),
            start=int(row["start"]),
            end=int(row["end"]),
            type=str(row["type"]),
            identifier_class=str(row["identifier_class"]),
            subject_id=str(row["subject_id"]),
            task_relevance=str(row["task_relevance"]),
            reference_evidence=list(row.get("reference_evidence", [])),
        )


@dataclass(frozen=True)
class PrivacyTarget:
    target_id: str
    known_to_attacker: str
    secret_attributes: list[str]

    def to_json(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "PrivacyTarget":
        return cls(
            target_id=str(row["target_id"]),
            known_to_attacker=str(row["known_to_attacker"]),
            secret_attributes=list(row.get("secret_attributes", [])),
        )


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    domain: str
    task: str
    corpus: str
    doc_orig: str
    task_prompt_template: str
    reference_outputs: list[str]
    gold_sensitive_spans: list[SensitiveSpan]
    privacy_targets: list[PrivacyTarget]

    def to_json(self) -> JsonDict:
        row = asdict(self)
        row["gold_sensitive_spans"] = [s.to_json() for s in self.gold_sensitive_spans]
        row["privacy_targets"] = [t.to_json() for t in self.privacy_targets]
        return row

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "BenchmarkItem":
        return cls(
            item_id=str(row["item_id"]),
            domain=str(row["domain"]),
            task=str(row["task"]),
            corpus=str(row["corpus"]),
            doc_orig=str(row["doc_orig"]),
            task_prompt_template=str(row["task_prompt_template"]),
            reference_outputs=list(row.get("reference_outputs", [])),
            gold_sensitive_spans=[
                SensitiveSpan.from_json(s)
                for s in row.get("gold_sensitive_spans", [])
            ],
            privacy_targets=[
                PrivacyTarget.from_json(t)
                for t in row.get("privacy_targets", [])
            ],
        )


@dataclass(frozen=True)
class StageOutput:
    detected_spans: list[JsonDict]
    R: list[JsonDict]
    doc_p: str
    out_p: str
    out_final: str
    extractor_trace: JsonDict | None = None

    def to_json(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "StageOutput":
        return cls(
            detected_spans=list(row.get("detected_spans", [])),
            R=list(row.get("R", [])),
            doc_p=str(row.get("doc_p", "")),
            out_p=str(row.get("out_p", "")),
            out_final=str(row.get("out_final", "")),
            extractor_trace=row.get("extractor_trace") or None,
        )


@dataclass(frozen=True)
class BenchmarkTrace:
    item: BenchmarkItem
    config_hash: str
    stage: StageOutput
    metrics: JsonDict

    def to_json(self) -> JsonDict:
        return {
            "item": self.item.to_json(),
            "config_hash": self.config_hash,
            "stage": self.stage.to_json(),
            "metrics": self.metrics,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "BenchmarkTrace":
        return cls(
            item=BenchmarkItem.from_json(row["item"]),
            config_hash=str(row["config_hash"]),
            stage=StageOutput.from_json(row["stage"]),
            metrics=dict(row.get("metrics", {})),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    suite: str
    limit: int | None
    seed: int
    detector_version: str
    substitutor_version: str
    privacy_setting: str
    remote_model: str | None
    extractor_version: str
    attacker_version: str
    output_dir: str
    detector_model: str | None = None
    detector_fine_dem: bool = False
    extractor_model: str | None = None
    attack_docp_model: str | None = None
    attack_reconstruction_model: str | None = None
    attack_leak_model: str | None = None

    def to_json(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "BenchmarkConfig":
        limit = row.get("limit")
        return cls(
            suite=str(row["suite"]),
            limit=None if limit is None else int(limit),
            seed=int(row["seed"]),
            detector_version=str(row["detector_version"]),
            substitutor_version=str(row["substitutor_version"]),
            privacy_setting=str(row["privacy_setting"]),
            remote_model=_optional_str(row.get("remote_model")),
            extractor_version=str(row["extractor_version"]),
            attacker_version=str(row["attacker_version"]),
            output_dir=str(row["output_dir"]),
            detector_model=_optional_str(row.get("detector_model")),
            detector_fine_dem=bool(row.get("detector_fine_dem", False)),
            extractor_model=_optional_str(row.get("extractor_model")),
            attack_docp_model=_optional_str(row.get("attack_docp_model")),
            attack_reconstruction_model=_optional_str(row.get("attack_reconstruction_model")),
            attack_leak_model=_optional_str(row.get("attack_leak_model")),
        )

    def config_hash(self) -> str:
        row = self.to_json()
        row.pop("output_dir", None)
        return stable_hash(row)

    def replace(self, **changes: object) -> "BenchmarkConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class BenchmarkScores:
    config_hash: str
    stage_metrics: list[JsonDict]
    utility_metrics: JsonDict
    privacy_metrics: JsonDict
    frontier: list[JsonDict]
    gates: list[JsonDict]

    def to_json(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "BenchmarkScores":
        return cls(
            config_hash=str(row["config_hash"]),
            stage_metrics=list(row.get("stage_metrics", [])),
            utility_metrics=dict(row.get("utility_metrics", {})),
            privacy_metrics=dict(row.get("privacy_metrics", {})),
            frontier=list(row.get("frontier", [])),
            gates=list(row.get("gates", [])),
        )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
