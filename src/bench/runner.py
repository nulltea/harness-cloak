from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Protocol

from bench.baselines import (
    make_all_placeholder_record,
    make_coarsest_text_record,
    make_no_privacy_record,
)
from bench.registry import load_items
from bench.schema import BenchmarkConfig, BenchmarkItem, BenchmarkTrace, StageOutput, jsonl_write
from cloak.extract import invert
from cloak.substitute import substitute
from cloak.tasks import TASK_TEMPLATE


class RemoteClientProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...


class StubRemote:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        doc = prompt.split("\n\n")[-1]
        first = re.split(r"(?<=[.!?])\s+", doc.strip())[0]
        return first or doc[:200]


def build_prompt(item: BenchmarkItem, doc_p: str) -> str:
    template = TASK_TEMPLATE.get(item.corpus) or TASK_TEMPLATE.get(item.task_prompt_template)
    if template is None:
        return f"Complete the requested task using only the document below.\n\n{doc_p}"
    return template.format(doc=doc_p)


def run_item(
    item: BenchmarkItem,
    config: BenchmarkConfig,
    remote: RemoteClientProtocol | None = None,
) -> BenchmarkTrace:
    detected = _detect(item, config)
    doc_p, R = _substitute(item, detected, config)
    client = remote or _remote(config)
    out_p = client.generate(build_prompt(item, doc_p))
    out_final, extractor_trace = invert(out_p, R)
    stage = StageOutput(
        detected_spans=detected,
        R=R,
        doc_p=doc_p,
        out_p=out_p,
        out_final=out_final,
        extractor_trace=extractor_trace,
    )
    return BenchmarkTrace(item=item, config_hash=config.config_hash(), stage=stage, metrics={})


def run_suite(
    config: BenchmarkConfig,
    remote: RemoteClientProtocol | None = None,
) -> list[BenchmarkTrace]:
    items = load_items(config.suite, limit=config.limit, seed=config.seed)
    return [run_item(item, config, remote=remote) for item in items]


def write_dry_run(config: BenchmarkConfig, output_dir: Path) -> list[BenchmarkItem]:
    items = load_items(config.suite, limit=config.limit, seed=config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(_json(config.to_json()), encoding="utf-8")
    jsonl_write(output_dir / "items.jsonl", [item.to_json() for item in items])
    return items


def write_traces(output_dir: Path, traces: list[BenchmarkTrace]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_write(output_dir / "traces.jsonl", [trace.to_json() for trace in traces])


def _detect(item: BenchmarkItem, config: BenchmarkConfig) -> list[dict]:
    if config.detector_version == "gold":
        return [
            {
                "start": span.start,
                "end": span.end,
                "text": span.surface,
                "surface": span.surface,
                "type": span.type,
                "score": 1.0,
                "chain": idx,
            }
            for idx, span in enumerate(item.gold_sensitive_spans)
        ]
    from cloak.detect import Detector

    return [
        {
            "start": span.start,
            "end": span.end,
            "text": span.text,
            "surface": span.text,
            "type": span.type,
            "score": span.score,
            "chain": span.chain,
        }
        for span in Detector().detect(item.doc_orig)
    ]


def _substitute(item: BenchmarkItem, detected: list[dict], config: BenchmarkConfig) -> tuple[str, list[dict]]:
    if config.substitutor_version == "no_privacy":
        return make_no_privacy_record(item.doc_orig, detected)
    if config.substitutor_version == "all_placeholder":
        return make_all_placeholder_record(item.doc_orig, detected)
    if config.substitutor_version == "coarsest_text":
        return make_coarsest_text_record(item.doc_orig, detected)
    return substitute(item.doc_orig, _span_objects(detected), tau=_tau(config.privacy_setting))


def _span_objects(detected: list[dict]):
    from cloak.detect import Span

    return [
        Span(
            start=int(row["start"]),
            end=int(row["end"]),
            text=str(row.get("text", row.get("surface", ""))),
            type=str(row["type"]),
            score=float(row.get("score", 1.0)),
            chain=int(row.get("chain", 0)),
        )
        for row in detected
    ]


def _tau(setting: str) -> float:
    if setting.startswith("tau="):
        return float(setting.split("=", 1)[1])
    return 0.02


def _remote(config: BenchmarkConfig) -> RemoteClientProtocol:
    if not os.getenv("INFERDPT_LLM_CACHE"):
        raise RuntimeError("live remote benchmark requires INFERDPT_LLM_CACHE")
    from inferdpt.llm import LLMClient

    return LLMClient(config.remote_model, temperature=0.0, max_tokens=1024)


def _json(row: object) -> str:
    import json

    return json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
