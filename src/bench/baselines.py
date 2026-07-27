from __future__ import annotations

import re

from cloak.lattice.core import lattice_for
from cloak.runtime_types import PLACEHOLDER_RE, placeholder_token


def make_no_privacy_record(text: str, spans: list[object]) -> tuple[str, list[dict]]:
    R = []
    for span in sorted((_span_dict(s) for s in spans), key=lambda s: s["start"]):
        entry = dict(span)
        entry.update(action="keep", replacement=span["surface"], risk=1.0)
        R.append(entry)
    return text, R


def make_all_placeholder_record(text: str, spans: list[object]) -> tuple[str, list[dict]]:
    counters: dict[str, int] = {}
    out = text
    R = []
    for span in sorted((_span_dict(s) for s in spans), key=lambda s: -s["start"]):
        typ = span["type"]
        counters[typ] = counters.get(typ, 0) + 1
        replacement = placeholder_token(typ, counters[typ])
        entry = dict(span)
        entry.update(action="placeholder", replacement=replacement, risk=0.0)
        out = out[: span["start"]] + replacement + out[span["end"] :]
        R.append(entry)
    return out, sorted(R, key=lambda e: e["start"])


def make_coarsest_text_record(text: str, spans: list[object]) -> tuple[str, list[dict]]:
    counters: dict[str, int] = {}
    out = text
    R = []
    for span in sorted((_span_dict(s) for s in spans), key=lambda s: -s["start"]):
        replacement = _coarsest_replacement(span)
        action = "generalize"
        if PLACEHOLDER_RE.fullmatch(replacement):
            action = "placeholder"
        elif replacement.lower() == span["surface"].lower():
            counters[span["type"]] = counters.get(span["type"], 0) + 1
            replacement = placeholder_token(span["type"], counters[span["type"]])
            action = "placeholder"
        entry = dict(span)
        entry.update(action=action, replacement=replacement, risk=0.0 if action == "placeholder" else None)
        out = out[: span["start"]] + replacement + out[span["end"] :]
        R.append(entry)
    return out, sorted(R, key=lambda e: e["start"])


def make_oracle_extractor_record(out_p: str, R: list[dict]) -> str:
    out = out_p
    for entry in sorted(R, key=lambda e: -len(str(e["replacement"]))):
        replacement = str(entry["replacement"])
        if replacement and replacement in out:
            out = out.replace(replacement, str(entry["surface"]))
    return out


def _coarsest_replacement(span: dict) -> str:
    lattice = lattice_for(span["surface"], span["type"], "")
    for candidate in reversed(lattice):
        if candidate and not PLACEHOLDER_RE.fullmatch(candidate):
            return candidate
    return placeholder_token(span["type"], 1)


def _span_dict(span: object) -> dict:
    if isinstance(span, dict):
        surface = span.get("surface", span.get("text", ""))
        return {
            "start": int(span["start"]),
            "end": int(span["end"]),
            "surface": str(surface),
            "type": str(span["type"]),
            "score": float(span.get("score", 1.0)),
            "chain": int(span.get("chain", 0)),
        }
    return {
        "start": int(getattr(span, "start")),
        "end": int(getattr(span, "end")),
        "surface": str(getattr(span, "text", getattr(span, "surface", ""))),
        "type": str(getattr(span, "type")),
        "score": float(getattr(span, "score", 1.0)),
        "chain": int(getattr(span, "chain", 0)),
    }
