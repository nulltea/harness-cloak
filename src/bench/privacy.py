from __future__ import annotations

from collections import defaultdict
import re

from bench.schema import BenchmarkTrace


def attribute_attacker(trace: BenchmarkTrace) -> dict[str, object]:
    leaks = _surface_leaks(trace.stage.doc_p, trace.stage.R)
    success = 1.0 if leaks else 0.0
    return {
        "attacker": "attribute_exact_doc_p",
        "attack_success": success,
        "doc_p_attack_success": success,
        "doc_p_exact_leaks": len(leaks),
        "leaked_surfaces": leaks,
    }


def reconstruction_attacker(trace: BenchmarkTrace) -> dict[str, object]:
    echoes = [r for r in trace.stage.R if _contains(trace.stage.doc_p, str(r["replacement"]))]
    leaks = _surface_leaks(trace.stage.doc_p, trace.stage.R)
    success = 1.0 if leaks else 0.0
    return {
        "attacker": "reconstruction_exact_doc_p",
        "attack_success": success,
        "replacement_echoes": len(echoes),
        "original_surface_leaks": len(leaks),
    }


def leak_through_attacker(trace: BenchmarkTrace) -> dict[str, object]:
    leaks = _surface_leaks(trace.stage.out_final, trace.stage.R)
    success = 1.0 if leaks else 0.0
    return {
        "attacker": "leak_through_exact_out_final",
        "attack_success": success,
        "out_final_attack_success": success,
        "out_final_exact_leaks": len(leaks),
        "leaked_surfaces": leaks,
    }


def closed_world_reidentifier(trace: BenchmarkTrace, roster: list[dict]) -> dict[str, object]:
    doc = _canon(trace.stage.doc_p)
    ranked = sorted(
        roster,
        key=lambda row: _overlap(doc, _canon(" ".join(str(v) for v in row.values()))),
        reverse=True,
    )
    top = ranked[0] if ranked else {}
    hit = bool(top and str(top.get("item_id", top.get("target_id", ""))) in trace.item.item_id)
    return {
        "attacker": "closed_world_overlap",
        "attack_success": 1.0 if hit else 0.0,
        "top_candidate": top,
    }


def realized_privacy_score(attack_rows: list[dict]) -> float:
    if not attack_rows:
        return 1.0
    success = [float(row.get("attack_success", 0.0)) for row in attack_rows]
    return round(1.0 - (sum(success) / len(success)), 6)


def matched_privacy_bins(rows: list[dict], width: float = 0.05) -> dict[str, list[dict]]:
    bins: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        privacy = float(row["realized_privacy"])
        lo = int(privacy / width) * width
        bins[f"{lo:.2f}-{lo + width:.2f}"].append(row)
    return dict(bins)


def _surface_leaks(text: str, R: list[dict]) -> list[str]:
    return [str(r["surface"]) for r in R if _contains(text, str(r["surface"]))]


def _contains(text: str, needle: str) -> bool:
    key = _canon(needle)
    return bool(key and key in _canon(text))


def _canon(text: str) -> str:
    out = text.lower().replace("-year-old", " years old")
    out = re.sub(r"[^a-z0-9_<>]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _overlap(left: str, right: str) -> int:
    return len(set(left.split()) & set(right.split()))
