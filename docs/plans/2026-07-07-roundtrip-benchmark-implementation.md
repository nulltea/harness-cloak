---
type: plan
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [benchmark, roundtrip, privacy, utility, detector, substitutor, extractor, plan]
companion: [../specs/roundtrip-pipeline-benchmark.md, ../specs/detector-model.md, ../specs/lattice-substitutor.md, ../specs/probes.md, ../specs/attacks.md, ../specs/RL/roundtrip-ranker-infiller.md]
---

# Roundtrip Pipeline Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark harness for
`doc_orig -> detector + substitutor -> doc_p -> RemoteLLM -> out_p -> extractor -> out_final` that reports
utility on `out_final` against realized privacy on `doc_p` and leak-through from `out_final`.

**Architecture:** Add a `bench` package that owns benchmark records, stage traces, metrics,
attacker-facing exports, and reports. Reuse existing corpus loaders, task templates, detector,
substitutor, extractor, and reward scorers where their contracts are already stable; keep publication
benchmark orchestration separate from `cloak.train.*` so reward training does not accidentally become the
source of benchmark claims. The harness writes content-addressed JSONL artifacts and summary reports, with
remote calls behind an explicit cache requirement.

**Tech Stack:** Python stdlib (`argparse`, `dataclasses`, `hashlib`, `json`, `statistics`, `pathlib`,
`random`, `re`), existing `cloak` modules, pytest. Use existing optional packages already imported by the
repo (`rapidfuzz`, `rouge_score`, `bert_score`, local/served `inferdpt` clients) only through existing
wrappers unless a task explicitly extends them.

## Global Constraints

- Compare methods only at matched realized privacy and identical settings; never add a per-method
  normalization or calibration knob to make utility numbers line up.
- Privacy metrics are attacker outcomes, not surface proxies such as overlap, embedding distance,
  anonymity-set size, or epsilon.
- Detector misses are a hard privacy ceiling and must be reported as residual PII in `doc_p`.
- The substitutor may use placeholders or truthful generalizations, but every action must be legal for the
  selected privacy setting and recorded in `R`.
- The extractor may use only local inputs available at inference time: `out_p`, `R`, and the deployed local
  extraction model or rules. It must never use gold labels or call the remote model.
- `out_final` leak-through is reported separately from the remote-threat privacy claim on `doc_p`.
- Remote task execution must require `CLOAK_LLM_CACHE` and deterministic decoding before any live call.
- Heavy local model workflows run through `.venv/bin/python -u ...`; one GPU process at a time.
- External or rate-limited API attackers, including agentic web-search tiers, require explicit user approval
  before execution.
- Docs in `docs/**/*.md` need frontmatter with `type`, `status`, `created`, `updated`, and `tags`.
- Durable reusable scripts live under `scripts/`; one-off exploratory probes live under `scripts/spikes/`.

## Design Choice

Three implementation shapes were considered: extend `scripts/latticecloak_task_eval.py`, add a monolithic
`scripts/run_roundtrip_benchmark.py`, or create a small package with a thin CLI. The package-plus-CLI shape
is the right tradeoff: it keeps stage contracts testable, lets training import only the parts it needs, and
prevents benchmark metrics from being scattered across spike scripts.

## File Structure

- Create `src/bench/__init__.py`
  - Exposes the benchmark package version and stable public imports.
- Create `src/bench/schema.py`
  - Dataclasses and JSON serialization for items, spans, stage traces, scored rows, and run manifests.
- Create `src/bench/registry.py`
  - Loads benchmark items from existing task corpora and public smoke/detector corpora.
- Create `src/bench/baselines.py`
  - Converts detector spans into action records for no-privacy, all-placeholder, coarsest-text, floor-walk,
    deployed-policy, and oracle-extractor baselines.
- Create `src/bench/runner.py`
  - Orchestrates detector, substitution, remote task, extraction, cache keys, and per-item trace writing.
- Create `src/bench/metrics.py`
  - Computes detector residuals, echo/absorption labels, restoration metrics, utility scores, bootstrap
    intervals, and matched-privacy bins.
- Create `src/bench/privacy.py`
  - Provides offline deterministic privacy attackers and exports traces for later frontier/LLM attackers.
- Create `src/bench/report.py`
  - Builds JSON and Markdown summaries for per-domain tables, Pareto points, regressions, and gates.
- Create `scripts/run_roundtrip_benchmark.py`
  - Durable CLI for local slices, cached remote runs, scoring, reporting, and artifact paths.
- Modify `src/cloak/corpora.py`
  - Add PriMock57 and external detector/privacy corpus loaders only when local raw files exist.
- Modify `src/cloak/score.py`
  - Add lightweight entity/fact helpers used by the benchmark without changing existing ROUGE/BERTScore
    behavior.
- Create tests:
  - `src/cloak/tests/test_benchmark_schema.py`
  - `src/cloak/tests/test_benchmark_registry.py`
  - `src/cloak/tests/test_benchmark_baselines.py`
  - `src/cloak/tests/test_benchmark_metrics.py`
  - `src/cloak/tests/test_benchmark_privacy.py`
  - `src/cloak/tests/test_benchmark_runner.py`
  - `src/cloak/tests/test_run_roundtrip_benchmark_cli.py`

## Artifact Layout

Benchmark runs write immutable artifacts under `results/roundtrip_benchmark/<run_id>/`:

```text
results/roundtrip_benchmark/<run_id>/
  manifest.json
  items.jsonl
  traces.jsonl
  stage_metrics.json
  privacy_metrics.json
  utility_metrics.json
  matched_privacy_frontier.json
  report.md
```

`run_id` is a readable prefix plus a hash of the manifest fields:

```text
benchmark_version, item_ids, task_template, detector_version, substitutor_version,
privacy_setting, remote_model, extractor_version, attacker_version
```

Changing any field produces a new run directory. The CLI may resume a run only when the stored manifest
matches the requested manifest exactly.

## Runtime Interfaces

The package should expose these concrete signatures:

```python
def load_items(suite: str, limit: int | None = None, seed: int = 0) -> list[BenchmarkItem]: ...
def run_item(item: BenchmarkItem, config: BenchmarkConfig) -> BenchmarkTrace: ...
def run_suite(config: BenchmarkConfig) -> list[BenchmarkTrace]: ...
def score_traces(traces: list[BenchmarkTrace], config: BenchmarkConfig) -> BenchmarkScores: ...
def write_report(scores: BenchmarkScores, output_dir: Path) -> Path: ...
```

Core dataclasses are frozen where possible, and every one has `to_json()` / `from_json()` helpers that use
plain JSON-compatible dictionaries. Keep JSONL artifacts easy to inspect with `jq` and `rg`.

## Task 1 - Benchmark Schema

**Files:**
- Create: `src/bench/__init__.py`
- Create: `src/bench/schema.py`
- Test: `src/cloak/tests/test_benchmark_schema.py`

**Interfaces:**
- Produces:
  - `SensitiveSpan`
  - `PrivacyTarget`
  - `BenchmarkItem`
  - `StageOutput`
  - `BenchmarkTrace`
  - `BenchmarkConfig`
  - `BenchmarkScores`
  - `stable_hash(payload: Mapping[str, object]) -> str`
  - `jsonl_write(path: Path, rows: Iterable[JsonDict]) -> None`
  - `jsonl_read(path: Path) -> list[JsonDict]`
- Consumes:
  - Python stdlib only.

- [ ] **Step 1: Write failing schema tests**

Add `src/cloak/tests/test_benchmark_schema.py`:

```python
from pathlib import Path

from bench.schema import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkTrace,
    PrivacyTarget,
    SensitiveSpan,
    StageOutput,
    jsonl_read,
    jsonl_write,
    stable_hash,
)


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="clinical/mts/000001",
        domain="clinical",
        task="visit_note_generation",
        corpus="mts",
        doc_orig="Martha is a 50-year-old patient.",
        task_prompt_template="clinical_note",
        reference_outputs=["Martha is a 50-year-old patient."],
        gold_sensitive_spans=[
            SensitiveSpan(
                span_id="s1",
                surface="Martha",
                start=0,
                end=6,
                type="PERSON",
                identifier_class="DIRECT",
                subject_id="patient",
                task_relevance="gold_restated",
                reference_evidence=["Martha is a 50-year-old patient."],
            )
        ],
        privacy_targets=[
            PrivacyTarget(
                target_id="patient",
                known_to_attacker="document_context_only",
                secret_attributes=["name", "age"],
            )
        ],
    )


def test_item_round_trips_through_json():
    item = _item()
    got = BenchmarkItem.from_json(item.to_json())
    assert got == item
    assert got.gold_sensitive_spans[0].surface == "Martha"


def test_trace_round_trips_through_json():
    item = _item()
    stage = StageOutput(
        detected_spans=[{"surface": "Martha", "type": "PERSON", "start": 0, "end": 6}],
        R=[{"surface": "Martha", "replacement": "<PERSON_1>", "type": "PERSON"}],
        doc_p="<PERSON_1> is a 50-year-old patient.",
        out_p="<PERSON_1> is a patient.",
        out_final="Martha is a patient.",
    )
    trace = BenchmarkTrace(item=item, config_hash="abc123", stage=stage, metrics={"utility": 1.0})
    assert BenchmarkTrace.from_json(trace.to_json()) == trace


def test_stable_hash_is_order_invariant_for_dict_keys():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_jsonl_helpers(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    jsonl_write(path, [{"a": 1}, {"b": 2}])
    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]


def test_config_hash_changes_when_remote_model_changes():
    base = BenchmarkConfig(
        suite="primary_utility",
        limit=2,
        seed=0,
        detector_version="current",
        substitutor_version="current",
        privacy_setting="tau=0.02",
        remote_model="gemma 4 (E4B)",
        extractor_version="current",
        attacker_version="offline-v1",
        output_dir="results/roundtrip_benchmark/test",
    )
    changed = base.replace(remote_model="other-model")
    assert base.config_hash() != changed.config_hash()
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_schema.py -q
```

Expected: fails because `bench.schema` does not exist.

- [ ] **Step 3: Implement schema module**

Create `src/bench/__init__.py`:

```python
"""Roundtrip privacy/utility benchmark harness."""

BENCHMARK_VERSION = "roundtrip-benchmark-v1"
```

Create `src/bench/schema.py` with dataclasses matching the test. Use:

```python
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
from typing import Iterable, Mapping, TypeAlias

JsonDict: TypeAlias = dict[str, object]
```

`stable_hash()` must call `json.dumps(payload, sort_keys=True, separators=(",", ":"))`, hash with SHA-256,
and return the first 16 hex characters. `BenchmarkConfig.replace()` should delegate to
`dataclasses.replace(self, **changes)`.

- [ ] **Step 4: Run schema tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_schema.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bench/__init__.py src/bench/schema.py src/cloak/tests/test_benchmark_schema.py
git commit -m "feat: add roundtrip benchmark schema"
```

## Task 2 - Corpus Registry and Suite Selection

**Files:**
- Create: `src/bench/registry.py`
- Modify: `src/cloak/corpora.py`
- Test: `src/cloak/tests/test_benchmark_registry.py`

**Interfaces:**
- Consumes:
  - `cloak.corpora.load_task_docs(corpus: str, n: int | None = None) -> list[dict]`
  - `cloak.corpora.refs_of(doc: dict) -> list[str]`
  - `bench.schema.BenchmarkItem`
- Produces:
  - `SUITES: dict[str, list[str]]`
  - `load_items(suite: str, limit: int | None = None, seed: int = 0) -> list[BenchmarkItem]`
  - `load_detector_items(dataset: str, limit: int | None = None) -> list[BenchmarkItem]`
  - `gold_spans_from_text(doc_orig: str, refs: list[str]) -> list[SensitiveSpan]`

- [ ] **Step 1: Write failing registry tests**

Add `src/cloak/tests/test_benchmark_registry.py`:

```python
from bench.registry import SUITES, gold_spans_from_text, load_items


def test_primary_utility_suite_has_expected_domains():
    items = load_items("primary_utility", limit=6, seed=7)
    domains = {item.domain for item in items}
    assert {"clinical", "legal", "biography"} <= domains
    assert all(item.reference_outputs for item in items)
    assert all(item.doc_orig for item in items)


def test_negative_controls_are_separate_from_primary():
    assert "email_controls" in SUITES
    primary = {item.corpus for item in load_items("primary_utility", limit=12, seed=0)}
    controls = {item.corpus for item in load_items("email_controls", limit=4, seed=0)}
    assert not primary & controls
    assert controls <= {"aeslc", "enron"}


def test_gold_spans_from_text_marks_reference_restated_strings():
    doc = "Martha Collins is 50 years old and lives in Oslo."
    refs = ["Martha Collins is a 50-year-old patient."]
    spans = gold_spans_from_text(doc, refs)
    surfaces = {span.surface for span in spans}
    assert "Martha Collins" in surfaces
    assert any(span.task_relevance == "gold_restated" for span in spans)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_registry.py -q
```

Expected: fails because `bench.registry` does not exist.

- [ ] **Step 3: Implement suite registry**

Create `src/bench/registry.py` with:

```python
SUITES = {
    "primary_utility": ["clinical", "lexsum", "wikibio"],
    "clinical_smoke": ["primock57"],
    "email_controls": ["aeslc", "enron"],
    "detector_coverage": ["tab", "pii-bench", "synthetic-financial-pii"],
    "privacy_stress": ["rat-bench"],
}
```

Map corpora to domains:

```python
DOMAIN = {
    "aci": "clinical",
    "mts": "clinical",
    "clinical": "clinical",
    "primock57": "clinical",
    "lexsum": "legal",
    "wikibio": "biography",
    "aeslc": "email",
    "enron": "email",
    "tab": "legal",
    "pii-bench": "detector",
    "synthetic-financial-pii": "finance",
    "rat-bench": "privacy",
}
```

For `load_items("primary_utility", limit=6, seed=7)`, take a deterministic round-robin across suite corpora
so small limits include all three domains. Convert each existing task corpus row into `BenchmarkItem`:

```python
item_id = f"{corpus}/{row.get('id', index)}"
doc_orig = row["text"]
reference_outputs = refs_of(row)
task_prompt_template = corpus
```

Use `gold_spans_from_text()` as a cheap seed: detect capitalized multi-token names, dates/numbers, codes, and
locations/org-like title phrases from `doc_orig`; mark a span `gold_restated` when the exact surface or a
simple normalized variant appears in any reference. This is a bootstrap supply for tests and smoke runs, not
the final detector gold.

- [ ] **Step 4: Add guarded external corpus entries to `src/cloak/corpora.py`**

Extend `FILES` only for local paths that the downloader creates and keep loaders tolerant of missing optional
datasets:

```python
OPTIONAL_EXTERNAL = {
    "primock57": ["external/primock57/primock57-main.zip"],
    "rat-bench": ["external/rat-bench/benchmark/english/level_1.jsonl"],
    "synthetic-financial-pii": ["external/synthetic-financial-pii/Testing_Set.xlsx"],
    "pii-bench": ["external/pii-bench/data/test.jsonl"],
}
```

Add `available_corpora() -> set[str]` that returns built-in corpora plus optional corpora whose paths exist.
`load_task_docs()` should still raise `KeyError` for unsupported corpus names; the registry is responsible
for skipping unavailable optional datasets with a clear note in the manifest.

- [ ] **Step 5: Run registry tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_registry.py -q
```

Expected: all tests pass using existing built-in corpora.

- [ ] **Step 6: Commit**

```bash
git add src/bench/registry.py src/cloak/corpora.py src/cloak/tests/test_benchmark_registry.py
git commit -m "feat: add benchmark corpus registry"
```

## Task 3 - Baseline Action Policies

**Files:**
- Create: `src/bench/baselines.py`
- Test: `src/cloak/tests/test_benchmark_baselines.py`

**Interfaces:**
- Consumes:
  - Detector span dictionaries or `cloak.detect.Span` objects.
  - `cloak.runtime_types.placeholder_token`
  - `cloak.lattice.lattice_for`
- Produces:
  - `make_no_privacy_record(text: str, spans: list[object]) -> tuple[str, list[dict]]`
  - `make_all_placeholder_record(text: str, spans: list[object]) -> tuple[str, list[dict]]`
  - `make_coarsest_text_record(text: str, spans: list[object]) -> tuple[str, list[dict]]`
  - `make_oracle_extractor_record(out_p: str, R: list[dict]) -> str`

- [ ] **Step 1: Write failing baseline tests**

Add `src/cloak/tests/test_benchmark_baselines.py`:

```python
from dataclasses import dataclass

from bench.baselines import (
    make_all_placeholder_record,
    make_coarsest_text_record,
    make_no_privacy_record,
    make_oracle_extractor_record,
)


@dataclass
class SpanLike:
    start: int
    end: int
    text: str
    type: str
    score: float = 1.0
    chain: int = 0


def test_no_privacy_record_keeps_text_and_records_keep_actions():
    text = "Martha is 50 years old."
    doc_p, R = make_no_privacy_record(text, [SpanLike(0, 6, "Martha", "PERSON")])
    assert doc_p == text
    assert R[0]["action"] == "keep"
    assert R[0]["replacement"] == "Martha"


def test_all_placeholder_record_replaces_each_span():
    text = "Martha is in Oslo."
    spans = [SpanLike(0, 6, "Martha", "PERSON"), SpanLike(13, 17, "Oslo", "LOC")]
    doc_p, R = make_all_placeholder_record(text, spans)
    assert "Martha" not in doc_p
    assert "Oslo" not in doc_p
    assert "<PERSON_1>" in doc_p
    assert "<LOC_1>" in doc_p
    assert [r["action"] for r in R] == ["placeholder", "placeholder"]


def test_coarsest_text_record_uses_text_when_available_else_placeholder():
    text = "Martha is a cardiologist."
    spans = [SpanLike(12, 24, "cardiologist", "profession")]
    doc_p, R = make_coarsest_text_record(text, spans)
    assert "cardiologist" not in doc_p
    assert R[0]["replacement"]
    assert R[0]["action"] in {"generalize", "placeholder"}


def test_oracle_extractor_only_replaces_echoed_replacements():
    out_p = "<PERSON_1> is a healthcare worker."
    R = [
        {"surface": "Martha", "replacement": "<PERSON_1>", "action": "placeholder", "type": "PERSON"},
        {"surface": "cardiologist", "replacement": "healthcare worker", "action": "generalize", "type": "profession"},
        {"surface": "Oslo", "replacement": "a city", "action": "generalize", "type": "LOC"},
    ]
    out_final = make_oracle_extractor_record(out_p, R)
    assert out_final == "Martha is a cardiologist."
    assert "Oslo" not in out_final
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_baselines.py -q
```

Expected: fails because `bench.baselines` does not exist.

- [ ] **Step 3: Implement baseline policies**

Implement right-to-left span replacement with a helper:

```python
def _span_dict(span: object) -> dict:
    return {
        "start": span["start"] if isinstance(span, dict) else span.start,
        "end": span["end"] if isinstance(span, dict) else span.end,
        "surface": span.get("text", span.get("surface")) if isinstance(span, dict) else span.text,
        "type": span["type"] if isinstance(span, dict) else span.type,
        "score": span.get("score", 1.0) if isinstance(span, dict) else getattr(span, "score", 1.0),
        "chain": span.get("chain", 0) if isinstance(span, dict) else getattr(span, "chain", 0),
    }
```

For all-placeholder, number counters by type and use `placeholder_token(type, counter)`. For coarsest text,
call `lattice_for(surface, type, "")`, choose the last non-placeholder candidate that is not the exact
surface, and fall back to a typed placeholder if no text candidate exists.

- [ ] **Step 4: Run baseline tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_baselines.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bench/baselines.py src/cloak/tests/test_benchmark_baselines.py
git commit -m "feat: add benchmark baseline policies"
```

## Task 4 - Stage Runner With Cached Remote Calls

**Files:**
- Create: `src/bench/runner.py`
- Create: `scripts/run_roundtrip_benchmark.py`
- Test: `src/cloak/tests/test_benchmark_runner.py`
- Test: `src/cloak/tests/test_run_roundtrip_benchmark_cli.py`

**Interfaces:**
- Consumes:
  - `cloak.detect.Detector`
  - `cloak.substitute.substitute`
  - `cloak.extract.invert`
  - `cloak.tasks.TASK_TEMPLATE`
  - `bench.registry.load_items`
  - `bench.baselines`
- Produces:
  - `RemoteClientProtocol.generate(prompt: str) -> str`
  - `build_prompt(item: BenchmarkItem, doc_p: str) -> str`
  - `run_item(item: BenchmarkItem, config: BenchmarkConfig, remote: RemoteClientProtocol | None = None) -> BenchmarkTrace`
  - `run_suite(config: BenchmarkConfig, remote: RemoteClientProtocol | None = None) -> list[BenchmarkTrace]`
  - CLI command `scripts/run_roundtrip_benchmark.py`

- [ ] **Step 1: Write failing runner tests**

Add `src/cloak/tests/test_benchmark_runner.py`:

```python
from bench.runner import build_prompt, run_item
from bench.schema import BenchmarkConfig
from bench.registry import load_items


class StubRemote:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _config(policy="all_placeholder"):
    return BenchmarkConfig(
        suite="primary_utility",
        limit=1,
        seed=0,
        detector_version="stub",
        substitutor_version=policy,
        privacy_setting="tau=0.02",
        remote_model="stub-remote",
        extractor_version="current",
        attacker_version="offline-v1",
        output_dir="results/roundtrip_benchmark/test",
    )


def test_build_prompt_uses_task_template():
    item = load_items("primary_utility", limit=1, seed=0)[0]
    prompt = build_prompt(item, "PRIVATE DOC")
    assert "PRIVATE DOC" in prompt
    assert item.doc_orig not in prompt


def test_run_item_writes_stage_outputs_with_stub_remote():
    item = load_items("primary_utility", limit=1, seed=0)[0]
    remote = StubRemote(["<PERSON_1> is a patient."])
    trace = run_item(item, _config(), remote=remote)
    assert trace.stage.doc_p
    assert trace.stage.out_p == "<PERSON_1> is a patient."
    assert trace.stage.out_final
    assert remote.prompts and item.doc_orig not in remote.prompts[0]
```

Add `src/cloak/tests/test_run_roundtrip_benchmark_cli.py`:

```python
import json
import subprocess
import sys


def test_cli_dry_run_writes_manifest(tmp_path):
    out = tmp_path / "bench"
    cmd = [
        sys.executable,
        "scripts/run_roundtrip_benchmark.py",
        "--suite",
        "primary_utility",
        "--limit",
        "2",
        "--output-dir",
        str(out),
        "--dry-run",
    ]
    res = subprocess.run(cmd, check=True, text=True, capture_output=True)
    manifest = out / "manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["suite"] == "primary_utility"
    assert "dry-run" in res.stdout
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_runner.py src/cloak/tests/test_run_roundtrip_benchmark_cli.py -q
```

Expected: fails because runner and CLI do not exist.

- [ ] **Step 3: Implement runner**

Implement `run_item()` with these policy branches:

```python
if config.substitutor_version == "no_privacy":
    doc_p, R = make_no_privacy_record(item.doc_orig, detected_spans)
elif config.substitutor_version == "all_placeholder":
    doc_p, R = make_all_placeholder_record(item.doc_orig, detected_spans)
elif config.substitutor_version == "coarsest_text":
    doc_p, R = make_coarsest_text_record(item.doc_orig, detected_spans)
else:
    doc_p, R = substitute(item.doc_orig, detected_spans, tau=_tau_from_setting(config.privacy_setting))
```

Use `Detector().detect(item.doc_orig)` for normal runs. Tests can monkeypatch detector behavior later if
needed. After remote generation:

```python
out_final, extraction_stats = invert(out_p, R)
```

Store `extraction_stats` inside `StageOutput.extractor_trace`.

- [ ] **Step 4: Implement CLI dry-run and live-run modes**

`scripts/run_roundtrip_benchmark.py` arguments:

```text
--suite primary_utility|clinical_smoke|email_controls|detector_coverage|privacy_stress
--limit N
--seed N
--substitutor current|no_privacy|all_placeholder|coarsest_text
--privacy-setting tau=0.02
--remote-model gemma
--output-dir PATH
--dry-run
--workers N
```

Dry-run writes `manifest.json` and `items.jsonl`, prints `dry-run: wrote <N> items`, and performs no detector,
remote, extractor, or attacker work. Live-run must assert `CLOAK_LLM_CACHE` before constructing the real
remote client.

- [ ] **Step 5: Run runner and CLI tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_runner.py src/cloak/tests/test_run_roundtrip_benchmark_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run a local dry-run smoke**

Run:

```bash
.venv/bin/python -u scripts/run_roundtrip_benchmark.py --suite primary_utility --limit 3 --output-dir results/roundtrip_benchmark/dry-smoke --dry-run
```

Expected: prints `dry-run: wrote 3 items` and creates `manifest.json` plus `items.jsonl`.

- [ ] **Step 7: Commit**

```bash
git add src/bench/runner.py scripts/run_roundtrip_benchmark.py src/cloak/tests/test_benchmark_runner.py src/cloak/tests/test_run_roundtrip_benchmark_cli.py
git commit -m "feat: add roundtrip benchmark runner"
```

## Task 5 - Stage Metrics and Utility Scoring

**Files:**
- Create: `src/bench/metrics.py`
- Modify: `src/cloak/score.py`
- Test: `src/cloak/tests/test_benchmark_metrics.py`

**Interfaces:**
- Consumes:
  - `BenchmarkTrace`
  - `cloak.score.score_batch`
  - `cloak.train.reward.token_f1`, `canon`
- Produces:
  - `detector_residuals(trace: BenchmarkTrace) -> dict[str, object]`
  - `echo_labels(trace: BenchmarkTrace) -> list[dict]`
  - `restoration_metrics(trace: BenchmarkTrace) -> dict[str, float | int]`
  - `utility_metrics(trace: BenchmarkTrace) -> dict[str, float | int | None]`
  - `bootstrap_ci(values: list[float], seed: int = 0, samples: int = 1000) -> tuple[float, float]`
  - `score_traces(traces: list[BenchmarkTrace], config: BenchmarkConfig) -> BenchmarkScores`

- [ ] **Step 1: Write failing metric tests**

Add `src/cloak/tests/test_benchmark_metrics.py`:

```python
from bench.metrics import (
    bootstrap_ci,
    echo_labels,
    restoration_metrics,
    utility_metrics,
)
from bench.schema import BenchmarkItem, BenchmarkTrace, StageOutput


def _trace(out_p="<PERSON_1> is a healthcare worker.", out_final="Martha is a cardiologist."):
    item = BenchmarkItem(
        item_id="bio/1",
        domain="biography",
        task="bio_summary",
        corpus="wikibio",
        doc_orig="Martha is a cardiologist.",
        task_prompt_template="wikibio",
        reference_outputs=["Martha is a cardiologist."],
        gold_sensitive_spans=[],
        privacy_targets=[],
    )
    stage = StageOutput(
        detected_spans=[],
        R=[
            {"surface": "Martha", "replacement": "<PERSON_1>", "type": "PERSON", "action": "placeholder"},
            {"surface": "cardiologist", "replacement": "healthcare worker", "type": "profession", "action": "generalize"},
            {"surface": "Oslo", "replacement": "a city", "type": "LOC", "action": "generalize"},
        ],
        doc_p="<PERSON_1> is a healthcare worker.",
        out_p=out_p,
        out_final=out_final,
    )
    return BenchmarkTrace(item=item, config_hash="abc", stage=stage, metrics={})


def test_echo_labels_distinguish_echoed_and_absent():
    labels = echo_labels(_trace())
    by_surface = {row["surface"]: row["echo"] for row in labels}
    assert by_surface["Martha"] == "exact"
    assert by_surface["cardiologist"] == "exact"
    assert by_surface["Oslo"] == "absent"


def test_restoration_metrics_count_supported_recovery_only():
    metrics = restoration_metrics(_trace())
    assert metrics["echoed_span_recovery"] == 1.0
    assert metrics["unsupported_insertion_count"] == 0


def test_utility_metrics_use_references_and_sensitive_fact_recall():
    metrics = utility_metrics(_trace())
    assert 0.0 <= metrics["rougeL"] <= 1.0
    assert metrics["sensitive_fact_recall"] == 1.0


def test_bootstrap_ci_is_deterministic_and_ordered():
    lo, hi = bootstrap_ci([0.0, 0.5, 1.0], seed=123, samples=100)
    assert 0.0 <= lo <= hi <= 1.0
    assert (lo, hi) == bootstrap_ci([0.0, 0.5, 1.0], seed=123, samples=100)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_metrics.py -q
```

Expected: fails because metrics module does not exist.

- [ ] **Step 3: Implement echo, restoration, and utility metrics**

Echo labels:

```python
if replacement in out_p:
    echo = "exact"
elif fuzzy_score(replacement, out_p) >= 90:
    echo = "fuzzy"
else:
    echo = "absent"
```

Restoration counts only echoed entries as eligible. `unsupported_insertion_count` is the number of original
surfaces that appear in `out_final` when their replacement was absent from `out_p`.

Utility metrics:

```python
rouge = rouge_l(trace.stage.out_final, trace.item.reference_outputs)
sensitive_fact_recall = mean(
    token_f1(trace.stage.out_final, entry["surface"])
    for entry in trace.stage.R
    if entry["surface"] appears in any reference output after canon()
)
```

Return `None` for `sensitive_fact_recall` when no sensitive surfaces are reference-restated.

- [ ] **Step 4: Add entity/fact helper to `src/cloak/score.py`**

Add:

```python
def contains_fact(text: str, fact: str) -> bool:
    from cloak.train.reward import canon
    return canon(fact) in canon(text)
```

Keep existing `score_batch()` behavior unchanged.

- [ ] **Step 5: Run metric tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/bench/metrics.py src/cloak/score.py src/cloak/tests/test_benchmark_metrics.py
git commit -m "feat: add benchmark stage metrics"
```

## Task 6 - Offline Privacy Attackers and Matched-Privacy Bins

**Files:**
- Create: `src/bench/privacy.py`
- Test: `src/cloak/tests/test_benchmark_privacy.py`
- Modify: `src/bench/metrics.py`

**Interfaces:**
- Consumes:
  - `BenchmarkTrace`
  - `BenchmarkScores`
- Produces:
  - `closed_world_reidentifier(trace: BenchmarkTrace, roster: list[dict]) -> dict[str, object]`
  - `attribute_attacker(trace: BenchmarkTrace) -> dict[str, object]`
  - `reconstruction_attacker(trace: BenchmarkTrace) -> dict[str, object]`
  - `leak_through_attacker(trace: BenchmarkTrace) -> dict[str, object]`
  - `realized_privacy_score(attack_rows: list[dict]) -> float`
  - `matched_privacy_bins(rows: list[dict], width: float = 0.05) -> dict[str, list[dict]]`

- [ ] **Step 1: Write failing privacy tests**

Add `src/cloak/tests/test_benchmark_privacy.py`:

```python
from bench.privacy import (
    attribute_attacker,
    leak_through_attacker,
    matched_privacy_bins,
    realized_privacy_score,
    reconstruction_attacker,
)
from bench.schema import BenchmarkItem, BenchmarkTrace, StageOutput


def _trace(doc_p, out_final):
    item = BenchmarkItem(
        item_id="x/1",
        domain="clinical",
        task="note",
        corpus="clinical",
        doc_orig="Martha Collins lives in Oslo.",
        task_prompt_template="clinical",
        reference_outputs=["Martha Collins lives in Oslo."],
        gold_sensitive_spans=[],
        privacy_targets=[],
    )
    stage = StageOutput(
        detected_spans=[],
        R=[
            {"surface": "Martha Collins", "replacement": "<PERSON_1>", "type": "PERSON", "action": "placeholder"},
            {"surface": "Oslo", "replacement": "a city", "type": "LOC", "action": "generalize"},
        ],
        doc_p=doc_p,
        out_p="",
        out_final=out_final,
    )
    return BenchmarkTrace(item=item, config_hash="abc", stage=stage, metrics={})


def test_attribute_attacker_detects_surface_leaks_in_doc_p():
    row = attribute_attacker(_trace("Martha Collins lives in a city.", "Martha Collins lives in Oslo."))
    assert row["doc_p_exact_leaks"] == 1
    assert row["doc_p_attack_success"] == 1.0


def test_reconstruction_attacker_scores_recoverable_replacements():
    row = reconstruction_attacker(_trace("<PERSON_1> lives in a city.", "Martha Collins lives in Oslo."))
    assert row["replacement_echoes"] == 2
    assert row["original_surface_leaks"] == 0


def test_leak_through_attacker_scores_final_output():
    row = leak_through_attacker(_trace("<PERSON_1> lives in a city.", "Martha Collins lives in Oslo."))
    assert row["out_final_exact_leaks"] == 2
    assert row["out_final_attack_success"] == 1.0


def test_realized_privacy_score_is_one_minus_attack_success():
    assert realized_privacy_score([{"attack_success": 0.0}, {"attack_success": 0.5}]) == 0.75


def test_matched_privacy_bins_groups_rows_by_realized_privacy():
    rows = [{"method": "a", "realized_privacy": 0.91}, {"method": "b", "realized_privacy": 0.93}]
    bins = matched_privacy_bins(rows, width=0.05)
    assert list(bins.values()) == [rows]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_privacy.py -q
```

Expected: fails because privacy module does not exist.

- [ ] **Step 3: Implement deterministic offline attackers**

These attackers are not the publication frontier attacker. They are reproducible smoke/stress attackers that
catch obvious leaks and create a common interface for later LLM attackers:

- `attribute_attacker`: exact/canonical original-surface leakage in `doc_p`.
- `reconstruction_attacker`: replacement echo count and original surface recovery from `doc_p`.
- `leak_through_attacker`: exact/canonical original-surface leakage in `out_final`.
- `closed_world_reidentifier`: choose the roster candidate with the largest overlap between candidate
  attributes and `doc_p`; report top-1 hit if `item_id` or `target_id` matches.

Use `attack_success` keys consistently so `realized_privacy_score()` can compute `1 - mean(success)`.

- [ ] **Step 4: Wire privacy rows into `score_traces()`**

Update `src/bench/metrics.py` so `score_traces()` includes:

```python
privacy_rows = [
    attribute_attacker(trace),
    reconstruction_attacker(trace),
    leak_through_attacker(trace),
]
```

Store both component attack success and realized privacy in `BenchmarkScores`.

- [ ] **Step 5: Run privacy and metric tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_privacy.py src/cloak/tests/test_benchmark_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/bench/privacy.py src/bench/metrics.py src/cloak/tests/test_benchmark_privacy.py
git commit -m "feat: add benchmark privacy attackers"
```

## Task 7 - Reporting and Acceptance Gates

**Files:**
- Create: `src/bench/report.py`
- Modify: `src/bench/runner.py`
- Modify: `scripts/run_roundtrip_benchmark.py`
- Test: `src/cloak/tests/test_benchmark_report.py`

**Interfaces:**
- Consumes:
  - `BenchmarkScores`
  - `matched_privacy_bins()`
- Produces:
  - `summarize_by_domain(scores: BenchmarkScores) -> list[dict]`
  - `acceptance_gates(scores: BenchmarkScores) -> list[dict]`
  - `write_json_outputs(scores: BenchmarkScores, output_dir: Path) -> dict[str, Path]`
  - `write_markdown_report(scores: BenchmarkScores, output_dir: Path) -> Path`

- [ ] **Step 1: Write failing report tests**

Add `src/cloak/tests/test_benchmark_report.py`:

```python
import json

from bench.report import acceptance_gates, write_markdown_report
from bench.schema import BenchmarkScores


def _scores():
    return BenchmarkScores(
        config_hash="abc",
        stage_metrics=[{"domain": "clinical", "unsupported_insertion_count": 0, "rougeL": 0.7}],
        utility_metrics={"mean_rougeL": 0.7, "mean_sensitive_fact_recall": 0.6},
        privacy_metrics={"realized_privacy": 0.9, "doc_p_attack_success": 0.1},
        frontier=[{"method": "all_placeholder", "realized_privacy": 0.9, "utility": 0.4}],
        gates=[],
    )


def test_acceptance_gates_pass_clean_scores():
    gates = acceptance_gates(_scores())
    assert any(g["name"] == "unsupported_extractor_insertions" for g in gates)
    assert all(g["passed"] for g in gates)


def test_write_markdown_report_contains_frontier_and_gates(tmp_path):
    path = write_markdown_report(_scores(), tmp_path)
    text = path.read_text()
    assert "Privacy-Utility Frontier" in text
    assert "Acceptance Gates" in text
    assert "all_placeholder" in text
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_report.py -q
```

Expected: fails because report module does not exist.

- [ ] **Step 3: Implement report writer**

Markdown report sections:

```markdown
# Roundtrip Benchmark Report

## Run Manifest
## Privacy-Utility Frontier
## Utility By Domain
## Privacy Attack Results
## Stage Diagnostics
## Acceptance Gates
## Degenerate Outcomes
```

Acceptance gates:

- detector residual leak does not increase over the selected baseline;
- legality violations equal zero;
- unsupported extractor insertions equal zero;
- realized privacy exists and is attacker-derived;
- utility exists for every primary domain in the run;
- leak-through metrics exist for `out_final`.

- [ ] **Step 4: Make live CLI write traces, scores, and reports**

Update `scripts/run_roundtrip_benchmark.py` so non-dry-run writes:

```text
traces.jsonl
stage_metrics.json
privacy_metrics.json
utility_metrics.json
matched_privacy_frontier.json
report.md
```

Print the exact report path and the number of traces scored.

- [ ] **Step 5: Run report tests**

Run:

```bash
.venv/bin/python -m pytest src/cloak/tests/test_benchmark_report.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/bench/report.py src/bench/runner.py scripts/run_roundtrip_benchmark.py src/cloak/tests/test_benchmark_report.py
git commit -m "feat: add benchmark reports"
```

## Task 8 - End-to-End Smoke and Documentation Sync

**Files:**
- Modify: `docs/specs/roundtrip-pipeline-benchmark.md`
- Modify: `docs/plans/2026-07-07-roundtrip-benchmark-implementation.md`
- Test: existing benchmark tests plus a dry-run and stub-remote live run.

**Interfaces:**
- Consumes:
  - CLI from Task 4.
  - Report writer from Task 7.
- Produces:
  - A documented release-one smoke command.
  - A documented full benchmark command that is safe because it requires explicit cache and remote setup.

- [ ] **Step 1: Add implementation status to the benchmark spec**

Append a short `## Implementation Entry Points` section to
`docs/specs/roundtrip-pipeline-benchmark.md`:

```markdown
## Implementation Entry Points

The durable benchmark runner is `scripts/run_roundtrip_benchmark.py`. It writes immutable run artifacts
under `results/roundtrip_benchmark/<run_id>/` and uses `src/bench/` for schema, registry,
runner, metrics, privacy, and report code. Dry runs do not hit the detector, remote model, extractor, or
attacker suite; live runs require `CLOAK_LLM_CACHE` before constructing the remote client.
```

- [ ] **Step 2: Run focused benchmark tests**

Run:

```bash
.venv/bin/python -m pytest \
  src/cloak/tests/test_benchmark_schema.py \
  src/cloak/tests/test_benchmark_registry.py \
  src/cloak/tests/test_benchmark_baselines.py \
  src/cloak/tests/test_benchmark_runner.py \
  src/cloak/tests/test_benchmark_metrics.py \
  src/cloak/tests/test_benchmark_privacy.py \
  src/cloak/tests/test_benchmark_report.py \
  src/cloak/tests/test_run_roundtrip_benchmark_cli.py \
  -q
```

Expected: all benchmark tests pass.

- [ ] **Step 3: Run dry-run smoke**

Run:

```bash
.venv/bin/python -u scripts/run_roundtrip_benchmark.py \
  --suite primary_utility \
  --limit 3 \
  --seed 0 \
  --substitutor all_placeholder \
  --output-dir results/roundtrip_benchmark/dry-smoke \
  --dry-run
```

Expected: writes `manifest.json` and `items.jsonl`; prints `dry-run: wrote 3 items`.

- [ ] **Step 4: Run stub-remote live smoke**

Add a `--stub-remote` CLI flag that returns deterministic outputs by echoing the first sentence of `doc_p`.
Then run:

```bash
.venv/bin/python -u scripts/run_roundtrip_benchmark.py \
  --suite primary_utility \
  --limit 3 \
  --seed 0 \
  --substitutor all_placeholder \
  --remote-model stub \
  --stub-remote \
  --output-dir results/roundtrip_benchmark/stub-smoke
```

Expected: writes all seven run artifacts and a `report.md`; no external API or live model is contacted.

- [ ] **Step 5: Run formatting and docs checks**

Run:

```bash
git diff --check -- docs/plans/2026-07-07-roundtrip-benchmark-implementation.md docs/specs/roundtrip-pipeline-benchmark.md src/bench scripts/run_roundtrip_benchmark.py src/cloak/tests
```

Expected: no output.

Run:

```bash
.venv/bin/python - <<'PY'
import re
from pathlib import Path

paths = [
    Path("docs/plans/2026-07-07-roundtrip-benchmark-implementation.md"),
    Path("docs/specs/roundtrip-pipeline-benchmark.md"),
]
patterns = [
    "T" + "BD",
    "TO" + "DO",
    "FIX" + "ME",
    r"Arm [A-Z]",
    r"\b" + "D" + r"[0-9]+\b",
    "R" + r"D[0-9]+",
]
bad = []
for path in paths:
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if any(re.search(pattern, line) for pattern in patterns):
            bad.append(f"{path}:{lineno}:{line}")
if bad:
    raise SystemExit("\n".join(bad))
PY
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add docs/specs/roundtrip-pipeline-benchmark.md docs/plans/2026-07-07-roundtrip-benchmark-implementation.md
git commit -m "docs: add roundtrip benchmark implementation plan"
```

## Execution Notes

- Start with `--dry-run` and `--stub-remote`; those runs exercise schema, registry, metrics, privacy, and
  reporting without cost.
- Use live remote execution only after the stub run writes a complete report and
  `CLOAK_LLM_CACHE=data/llm_cache` is set.
- Publication-grade privacy claims require the pre-registered LLM attacker suite or an explicit statement
  that only the deterministic offline attacker tier was run.
- PriMock57, RAT-Bench, PIIBench, and synthetic financial PII enter implementation through corpus loaders
  after `scripts/download/fetch_benchmark_datasets.py` has placed raw data under `data/external`.

## Self-Review

- Spec coverage: the plan covers corpus selection, detector residuals, substitutor baselines, remote cache
  pinning, extractor recovery, utility metrics, privacy attackers, matched-privacy reporting, and acceptance
  gates from `docs/specs/roundtrip-pipeline-benchmark.md`.
- Placeholder scan: this plan intentionally uses the word "placeholder" as a benchmark mechanism term; it
  avoids open work markers and empty implementation instructions.
- Type consistency: `BenchmarkItem`, `BenchmarkConfig`, `BenchmarkTrace`, `StageOutput`, and
  `BenchmarkScores` are introduced in Task 1 and reused with the same names in later tasks.
