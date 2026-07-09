#!/usr/bin/env python3
"""Cross-process determinism gate for the frozen extractor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from cloak import frozen_extractor as fx


DEFAULT_FIXTURES = Path("data/extractor_gate_fixtures.jsonl")
DEFAULT_RUNS = 3


class StubEncoder:
    """Small deterministic encoder with aliases for extractor-gate fixtures."""

    ALIASES = {
        "municipality": "city",
        "margincity": "city",
        "typemargincity": "city",
        "individual": "person",
    }

    def __init__(self, dims: int = 32):
        self.dims = dims

    def encode(self, texts: Iterable[str]):
        rows = []
        for text in texts:
            vec = np.zeros(self.dims, dtype=np.float64)
            for token in str(text).lower().split():
                canonical = self.ALIASES.get(token.strip(".,;:!?"), token.strip(".,;:!?"))
                digest = hashlib.sha256(canonical.encode("utf-8")).digest()
                vec[int.from_bytes(digest[:4], "big") % self.dims] += 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm else vec)
        return np.vstack(rows)


class StubNLI:
    """Deterministic NLI stub that exercises both EPS_MARGIN fail-closed paths."""

    def __call__(self, premise: str, hypothesis: str) -> tuple[str, float]:
        text = f"{premise} {hypothesis}".lower()
        if "typemargincity" in str(premise).lower():
            return "entailment", float(fx.EXTRACTOR_PINS["thresholds"]["TYPE_ENTAIL"])
        if "margincity" in text and "typemargincity" not in text:
            return "entailment", float(fx.EXTRACTOR_PINS["thresholds"]["NLI_ENTAIL"])
        return "entailment", 0.99


class StubMLM:
    def pll(self, sentence: str) -> float:
        return -1.0


def stub_models() -> dict:
    return {"encoder": StubEncoder(), "nli": StubNLI(), "mlm": StubMLM()}


def install_stub_prepass() -> None:
    fx._rule_prepass = _stub_rule_prepass  # type: ignore[attr-defined]


def _stub_rule_prepass(out_p: str, R: list[dict], *, semantic: bool) -> tuple[str, dict, list[dict]]:
    del semantic
    text = out_p
    stats = {
        "ph_swapped": 0,
        "gen_exact": 0,
        "gen_fuzzy": 0,
        "gen_semantic": 0,
        "gen_absent": 0,
        "ph_residue": 0,
    }
    residue = []
    for entry in R:
        replacement = str(entry.get("replacement", ""))
        surface = str(entry.get("surface", ""))
        if entry.get("action") == "placeholder":
            count = text.count(replacement)
            if count:
                text = text.replace(replacement, surface)
                stats["ph_swapped"] += count
            continue
        pattern = re.compile(rf"\b{re.escape(replacement)}\b")
        matches = list(pattern.finditer(text)) if replacement else []
        if matches:
            count = len(matches)
            text = pattern.sub(surface, text)
            stats["gen_exact"] += count
        else:
            residue.append(entry)
    return text, stats, residue


def make_fixture_records() -> list[dict]:
    city_doc = "The hearing was in city."
    city_start = city_doc.index("city")
    person_doc = "A person filed the appeal."
    person_start = person_doc.index("person")
    return [
        {
            "doc_p": "The appeal was filed by <PERSON_1>.",
            "R": [
                {
                    "action": "placeholder",
                    "surface": "Ada Lovelace",
                    "replacement": "<PERSON_1>",
                    "type": "PERSON",
                }
            ],
            "out_p": "<PERSON_1> filed the appeal.",
        },
        {
            "doc_p": "The hearing was in a city.",
            "R": [
                {
                    "action": "generalize",
                    "surface": "Boston",
                    "replacement": "a city",
                    "type": "LOC",
                    "fill_spans": [[19, 25]],
                }
            ],
            "out_p": "The hearing was in a city.",
        },
        {
            "doc_p": city_doc,
            "R": [
                {
                    "action": "generalize",
                    "surface": "Boston",
                    "replacement": "city",
                    "type": "LOC",
                    "fill_spans": [[city_start, city_start + len("city")]],
                }
            ],
            "out_p": "The hearing was in municipality.",
        },
        {
            "doc_p": city_doc,
            "R": [
                {
                    "action": "generalize",
                    "surface": "Boston",
                    "replacement": "city",
                    "type": "LOC",
                    "fill_spans": [[city_start, city_start + len("city")]],
                }
            ],
            "out_p": "The hearing was in margincity.",
        },
        {
            "doc_p": city_doc,
            "R": [
                {
                    "action": "generalize",
                    "surface": "Boston",
                    "replacement": "city",
                    "type": "LOC",
                    "fill_spans": [[city_start, city_start + len("city")]],
                }
            ],
            "out_p": "The hearing was in typemargincity.",
        },
        {
            "doc_p": person_doc,
            "R": [
                {
                    "action": "generalize",
                    "surface": "Grace Hopper",
                    "replacement": "person",
                    "type": "PERSON",
                    "fill_spans": [[person_start, person_start + len("person")]],
                }
            ],
            "out_p": "An individual filed the appeal.",
        },
    ]


def write_fixtures(path: Path, records: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = records if records is not None else make_fixture_records()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")


def load_fixtures(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            missing = {"doc_p", "R", "out_p"} - set(record)
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys: {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no fixtures")
    return records


def canonical_outputs(fixtures: list[dict], *, models: dict) -> str:
    outputs = []
    for record in fixtures:
        out_final, stats = fx.extract(record["doc_p"], record["R"], record["out_p"], models=models)
        outputs.append({"entries": stats.get("entries", []), "out_final": out_final})
    return _canonical_json(outputs)


def run_worker(args: argparse.Namespace) -> int:
    if args.stub:
        install_stub_prepass()
        models = stub_models()
    else:
        models = fx.load_models(device=args.device)
    print(canonical_outputs(load_fixtures(args.fixtures), models=models))
    return 0


def run_parent(args: argparse.Namespace) -> int:
    if args.make_fixtures:
        write_fixtures(args.fixtures)
    fixtures = load_fixtures(args.fixtures)
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    cmd_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--fixtures",
        str(args.fixtures),
        "--device",
        args.device,
    ]
    if args.stub:
        cmd_base.append("--stub")

    outputs = []
    for _ in range(args.runs):
        result = subprocess.run(
            cmd_base,
            cwd=Path.cwd(),
            text=False,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            sys.stderr.buffer.write(result.stderr)
            return result.returncode
        outputs.append(result.stdout.strip())

    first = outputs[0]
    for idx, output in enumerate(outputs[1:], start=2):
        if output != first:
            print(f"determinism mismatch on run {idx}", file=sys.stderr)
            return 1

    print(
        _canonical_json(
            {
                "deterministic": True,
                "fixtures": len(fixtures),
                "runs": args.runs,
                "stub": bool(args.stub),
            }
        )
    )
    return 0


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
