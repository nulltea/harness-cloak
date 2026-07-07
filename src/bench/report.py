from __future__ import annotations

from pathlib import Path
import json

from bench.schema import BenchmarkScores


def summarize_by_domain(scores: BenchmarkScores) -> list[dict]:
    domains = sorted({str(row["domain"]) for row in scores.stage_metrics})
    out = []
    for domain in domains:
        rows = [row for row in scores.stage_metrics if row["domain"] == domain]
        rouges = [float(row["rougeL"]) for row in rows if row.get("rougeL") is not None]
        out.append({
            "domain": domain,
            "n": len(rows),
            "mean_rougeL": sum(rouges) / len(rouges) if rouges else None,
        })
    return out


def acceptance_gates(scores: BenchmarkScores) -> list[dict]:
    unsupported = sum(int(row.get("unsupported_insertion_count", 0)) for row in scores.stage_metrics)
    return [
        {
            "name": "unsupported_extractor_insertions",
            "passed": unsupported == 0,
            "value": unsupported,
        },
        {
            "name": "attacker_realized_privacy_present",
            "passed": scores.privacy_metrics.get("realized_privacy") is not None,
            "value": scores.privacy_metrics.get("realized_privacy"),
        },
        {
            "name": "utility_present",
            "passed": bool(scores.utility_metrics),
            "value": scores.utility_metrics.get("mean_sensitive_fact_recall", scores.utility_metrics.get("mean_rougeL")),
        },
    ]


def write_json_outputs(scores: BenchmarkScores, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage_metrics": output_dir / "stage_metrics.json",
        "privacy_metrics": output_dir / "privacy_metrics.json",
        "utility_metrics": output_dir / "utility_metrics.json",
        "frontier": output_dir / "matched_privacy_frontier.json",
    }
    paths["stage_metrics"].write_text(_json(scores.stage_metrics), encoding="utf-8")
    paths["privacy_metrics"].write_text(_json(scores.privacy_metrics), encoding="utf-8")
    paths["utility_metrics"].write_text(_json(scores.utility_metrics), encoding="utf-8")
    paths["frontier"].write_text(_json(scores.frontier), encoding="utf-8")
    return paths


def write_markdown_report(scores: BenchmarkScores, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gates = acceptance_gates(scores)
    path = output_dir / "report.md"
    lines = [
        "# Roundtrip Benchmark Report",
        "",
        "## Privacy-Utility Frontier",
        "",
        "| Method | Privacy Setting | Realized Privacy | Utility |",
        "|---|---:|---:|---:|",
    ]
    for row in scores.frontier:
        lines.append(
            f"| {row.get('method')} | {row.get('privacy_setting', '')} | "
            f"{row.get('realized_privacy')} | {row.get('utility')} |"
        )
    lines.extend(["", "## Utility By Domain", ""])
    for row in summarize_by_domain(scores):
        lines.append(f"- {row['domain']}: n={row['n']}, mean_rougeL={row['mean_rougeL']}")
    lines.extend(["", "## Privacy Attack Results", ""])
    lines.append(f"- realized_privacy: {scores.privacy_metrics.get('realized_privacy')}")
    lines.extend(["", "## Acceptance Gates", ""])
    for gate in gates:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- {status}: {gate['name']} = {gate['value']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _json(row: object) -> str:
    return json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
