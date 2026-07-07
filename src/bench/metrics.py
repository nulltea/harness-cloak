from __future__ import annotations

import random
import re
from statistics import mean

from bench.privacy import (
    attribute_attacker,
    leak_through_attacker,
    reconstruction_attacker,
    realized_privacy_score,
)
from bench.schema import BenchmarkConfig, BenchmarkScores, BenchmarkTrace
from cloak.score import rouge_l


def echo_labels(trace: BenchmarkTrace) -> list[dict]:
    rows = []
    for entry in trace.stage.R:
        replacement = str(entry["replacement"])
        echo = "exact" if replacement and replacement in trace.stage.out_p else "absent"
        rows.append({
            "surface": entry["surface"],
            "replacement": replacement,
            "type": entry.get("type"),
            "action": entry.get("action"),
            "echo": echo,
        })
    return rows


def restoration_metrics(trace: BenchmarkTrace) -> dict[str, float | int]:
    labels = echo_labels(trace)
    echoed = [row for row in labels if row["echo"] != "absent"]
    recovered = [row for row in echoed if _contains(trace.stage.out_final, str(row["surface"]))]
    unsupported = [
        entry for entry, label in zip(trace.stage.R, labels)
        if label["echo"] == "absent" and _contains(trace.stage.out_final, str(entry["surface"]))
    ]
    return {
        "echoed_spans": len(echoed),
        "restored_echoed_spans": len(recovered),
        "echoed_span_recovery": (len(recovered) / len(echoed)) if echoed else None,
        "unsupported_insertion_count": len(unsupported),
    }


def utility_metrics(trace: BenchmarkTrace) -> dict[str, float | int | None]:
    rouge = rouge_l(trace.stage.out_final, trace.item.reference_outputs)
    restated = [
        entry for entry in trace.stage.R
        if any(_contains(ref, str(entry["surface"])) for ref in trace.item.reference_outputs)
    ]
    recalls = [
        1.0 if _contains(trace.stage.out_final, str(entry["surface"])) else 0.0
        for entry in restated
    ]
    return {
        "rougeL": rouge,
        "sensitive_fact_recall": mean(recalls) if recalls else None,
        "reference_restated_sensitive_facts": len(restated),
    }


def detector_residuals(trace: BenchmarkTrace) -> dict[str, object]:
    detected = {(int(row["start"]), int(row["end"])) for row in trace.stage.detected_spans}
    residual = [
        span.to_json()
        for span in trace.item.gold_sensitive_spans
        if (span.start, span.end) not in detected
    ]
    return {"detector_residual_count": len(residual), "detector_residual_spans": residual}


def bootstrap_ci(values: list[float], seed: int = 0, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return (round(lo, 6), round(hi, 6))


def score_traces(traces: list[BenchmarkTrace], config: BenchmarkConfig) -> BenchmarkScores:
    stage_rows = []
    attack_rows = []
    rouge_values = []
    fact_values = []
    for trace in traces:
        rest = restoration_metrics(trace)
        util = utility_metrics(trace)
        attacks = [
            attribute_attacker(trace),
            reconstruction_attacker(trace),
            leak_through_attacker(trace),
        ]
        attack_rows.extend(attacks)
        if util["rougeL"] is not None:
            rouge_values.append(float(util["rougeL"]))
        if util["sensitive_fact_recall"] is not None:
            fact_values.append(float(util["sensitive_fact_recall"]))
        stage_rows.append({
            "item_id": trace.item.item_id,
            "domain": trace.item.domain,
            **detector_residuals(trace),
            **rest,
            **util,
        })

    realized_privacy = realized_privacy_score(attack_rows)
    utility = mean(fact_values) if fact_values else (mean(rouge_values) if rouge_values else 0.0)
    return BenchmarkScores(
        config_hash=config.config_hash(),
        stage_metrics=stage_rows,
        utility_metrics={
            "mean_rougeL": mean(rouge_values) if rouge_values else None,
            "mean_sensitive_fact_recall": mean(fact_values) if fact_values else None,
            "rougeL_ci": bootstrap_ci(rouge_values) if rouge_values else None,
        },
        privacy_metrics={
            "realized_privacy": realized_privacy,
            "attack_rows": attack_rows,
        },
        frontier=[{
            "method": config.substitutor_version,
            "privacy_setting": config.privacy_setting,
            "realized_privacy": realized_privacy,
            "utility": utility,
        }],
        gates=[],
    )


def _contains(text: str, needle: str) -> bool:
    key = _canon(needle)
    return bool(key and key in _canon(text))


def _canon(text: str) -> str:
    out = text.lower().replace("-year-old", " years old")
    out = re.sub(r"[^a-z0-9_<>]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()
