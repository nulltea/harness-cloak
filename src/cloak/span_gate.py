"""Semantic span gate: link-keep -> link-retype -> deny-list -> [opt-in] NLI-substitution.

Spec: docs/specs/detector-noise-semantic-gate.md. One decision core for every consumer of
detected spans (miner, runtime detect / RL). Fail-open is the terminal default: a span that
survives links + deny-list is KEPT. The NLI-substitution layer is opt-in (nli_verify=True) and
off by default — measured AUC 0.733 (results/nli_subst_spike.json), unusable at safe thresholds;
kept for experimentation. Record: research-wiki/experiments/detector-noise-filter-methods.md.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cloak import lattice_profiles as lp
from cloak.detect import is_noise_span
from cloak.profile_match import PROFILE_BACKED_TYPES, span_key

log = logging.getLogger(__name__)

_WARNED: set[str] = set()

# NLI-substitution generalizations: a residue span of one of these types is kept only if the
# context entails replacing the surface with the phrase. Opt-in (nli_verify=True).
TYPE_SUB_PHRASES = {
    "health-condition": "a medical condition",
    "medical-procedure": "a medical procedure",
    "drug": "a medication",
    "injury": "an injury",
    "organization-medical-facility": "a medical facility",
    "LOC": "a location",
    "profession": "an occupation",
    "nationality": "a nationality",
    "religion": "a religion",
    "ethnicity": "an ethnicity",
}


@dataclass
class GateDecision:
    action: str
    layer: str
    entry: str | None = None
    new_type: str | None = None


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning("span_gate: %s — NLI layer disabled (fail-open)", message)


def gate_fingerprint(profiles_path=None, nli_verify: bool = False,
                     nli_thresh: float = 0.9) -> str:
    """Version stamp for run configs: gate behavior changes iff this changes."""
    h = hashlib.sha256()
    p = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    h.update(p.read_bytes() if p.exists() else b"absent")
    h.update(f"{nli_verify}:{nli_thresh}".encode())
    return h.hexdigest()[:16]


def gate_spans(items, operating_point: str = "production", *, profiles_path=None,
               nli_verify: bool = False, nli_thresh: float = 0.9,
               nli_batch_fn: Callable | None = None
               ) -> dict[tuple[str, str], GateDecision]:
    """Gate detected spans. items entries are (surface, type) or (surface, type, context).

    Layers 1-3 (link-keep, link-retype, deny-list) run for every span on (surface, type).
    The residue (missed link + not deny-listed) fails open to keep/open, UNLESS nli_verify:
    then residue spans with a context and a TYPE_SUB_PHRASES type are entailment-checked and
    dropped (layer 'nli') if the substitution is not approved.
    """
    profiles_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)

    out: dict[tuple[str, str], GateDecision] = {}
    residue: list[tuple[tuple[str, str], str, str | None]] = []  # (key, surface, context)
    for item in items:
        surface, runtime_type = item[0], item[1]
        context = item[2] if len(item) > 2 else None
        key = span_key(surface, runtime_type)
        if key in out:
            continue
        got = lp.lookup_entry(surface, runtime_type, profiles_path)
        if got:
            out[key] = GateDecision("keep", "link", entry=got[0])
            continue
        retyped = None
        for other in sorted(PROFILE_BACKED_TYPES):
            if other == runtime_type:
                continue
            hit = lp.lookup_entry(surface, other, profiles_path)
            if hit:
                retyped = GateDecision("retype", "retype", entry=hit[0], new_type=other)
                break
        if retyped:
            out[key] = retyped
            continue
        if is_noise_span(surface, runtime_type):
            out[key] = GateDecision("drop", "denylist")
            continue
        out[key] = GateDecision("keep", "open")   # fail-open floor
        residue.append((key, surface, context))

    if not nli_verify:
        return out

    jobs, job_keys = [], []
    for key, surface, context in residue:
        phrase = TYPE_SUB_PHRASES.get(key[0])
        if phrase is None or not context:
            continue   # unknown type or no context -> stay keep/open
        jobs.append((surface, context, [phrase]))
        job_keys.append(key)
    if not jobs:
        return out

    if nli_batch_fn is None:
        from cloak.lattice import nli_gate_batch
        nli_batch_fn = lambda j: nli_gate_batch(j, thresh=nli_thresh)
    try:
        results = nli_batch_fn(jobs)
    except Exception as exc:
        _warn_once("nli", f"NLI batch failed ({type(exc).__name__})")
        return out
    for key, approved in zip(job_keys, results):
        if not approved:
            out[key] = GateDecision("drop", "nli")
    return out
