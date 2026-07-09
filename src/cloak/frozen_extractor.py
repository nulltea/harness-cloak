"""Frozen extractor: pinned tiered recovery for doc_p/out_p round trips.

The skeleton keeps the existing rule cascade as tier 0 and records the frozen
pin table in the extractor version. Later tiers slot between tier 0 and
finalization without changing the public `extract()` entrypoint.
"""
import hashlib
import json

from cloak.extract import _finalize, _rule_prepass


EXTRACTOR_PINS = {
    "models": {
        "encoder": "BAAI/bge-small-en-v1.5",
        "nli": "cross-encoder/nli-deberta-v3-small",
        "mlm": "roberta-base",
    },
    "thresholds": {
        "SIM_MIN": 0.55,
        "ASSIGN_MARGIN": 0.05,
        "PRIOR_WEIGHT": 0.15,
        "NLI_ENTAIL": 0.80,
        "TYPE_ENTAIL": 0.70,
        "PLL_MIN_DELTA": -6.0,
        "EPS_MARGIN": 0.02,
        "CHUNK_MAX_WORDS": 6,
    },
    "type_hypotheses": {},
    "ladder_semver": "0.1.0",
}


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def extractor_version() -> str:
    digest = hashlib.sha256(_canonical_json(EXTRACTOR_PINS).encode("utf-8")).hexdigest()
    return "fx-" + digest[:12]


def _run_tier0(out_p: str, R: list[dict]) -> tuple[str, dict, list[dict]]:
    return _rule_prepass(out_p, R, semantic=True)


def _abstain_entries(residue: list[dict], *, reason: str) -> list[dict]:
    return [
        {
            "surface": entry["surface"],
            "type": entry.get("type", "MISC"),
            "outcome": "abstained",
            "reason": reason,
        }
        for entry in residue
    ]


def _record_unresolved(stats: dict, residue: list[dict], *, reason: str) -> dict:
    stats["gen_absent"] = stats.get("gen_absent", 0) + len(residue)
    stats["entries"] = _abstain_entries(residue, reason=reason)
    stats["extractor_version"] = extractor_version()
    return stats


def extract(
    doc_p: str | None,
    R: list[dict],
    out_p: str,
    *,
    models: dict | None = None,
) -> tuple[str, dict]:
    """Recover original surfaces from `out_p`; fail closed on unresolved tier-0 residue."""
    del doc_p  # Stage 1 consumes this when alignment-prior support lands.
    prepass_text, stats, residue = _run_tier0(out_p, R)
    reason = "no-models" if models is None else "stage-not-implemented"
    stats = _record_unresolved(stats, residue, reason=reason)
    return _finalize(prepass_text, stats)
