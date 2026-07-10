"""Semantic span gate: link-keep -> link-retype -> deny-list -> anchor-margin drop.

Spec: docs/specs/detector-noise-semantic-gate.md. One decision core for every consumer of
detected spans (miner, runtime detect / RL); operating points differ only in frozen
thresholds from results/span_gate_calibration.json. Fail-open is the terminal default.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from cloak import lattice_profiles as lp
from cloak.detect import (_NOISE_ANATOMY, _NOISE_DEVICE_SUPPLIES,
                          _NOISE_IMAGING_DIAGNOSTICS, _NOISE_LAB_TESTS, is_noise_span)
from cloak.profile_match import (DEFAULT_MODEL_ID, PROFILE_BACKED_TYPES, _index_path_for,
                                 _l2norm, _st_model, load_embindex, span_key)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_NEGATIVES_PATH = Path("data/span_gate/negatives.npz")
DEFAULT_CALIBRATION_PATH = Path("results/span_gate_calibration.json")
JUNK_EXEMPLARS_PATH = Path("data/span_gate/junk_exemplars.txt")
_WARNED: set[str] = set()


@dataclass
class GateDecision:
    action: str
    layer: str
    entry: str | None = None
    new_type: str | None = None
    pos_sim: float | None = None
    neg_sim: float | None = None


def seed_negative_surfaces() -> list[str]:
    surfaces = set().union(_NOISE_LAB_TESTS, _NOISE_IMAGING_DIAGNOSTICS,
                           _NOISE_ANATOMY, _NOISE_DEVICE_SUPPLIES)
    if JUNK_EXEMPLARS_PATH.exists():
        surfaces |= {line.strip().lower()
                     for line in JUNK_EXEMPLARS_PATH.read_text().splitlines() if line.strip()}
    return sorted(surfaces)


def anchor_seed_split(surfaces) -> tuple[list[str], list[str]]:
    """Deterministic disjoint halves: sha256(surface) even -> anchor seed, odd -> eval."""
    anchor, evalh = [], []
    for s in sorted(set(surfaces)):
        (anchor if hashlib.sha256(s.encode()).digest()[0] % 2 == 0 else evalh).append(s)
    return anchor, evalh


def build_negative_index(out_path=DEFAULT_NEGATIVES_PATH, model_id=DEFAULT_MODEL_ID,
                         embed_fn: Callable | None = None,
                         surfaces: list[str] | None = None) -> Path:
    if surfaces is None:
        surfaces, _ = anchor_seed_split(seed_negative_surfaces())
    if embed_fn is None:
        model = _st_model(model_id)
        embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
    vectors = _l2norm(embed_fn(surfaces)) if surfaces else np.zeros((0, 0), dtype=np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": SCHEMA_VERSION, "model_id": model_id, "surfaces": surfaces,
            "seed_rule": "sha256-even-anchor"}
    np.savez(out_path, vectors=vectors, meta=np.array(json.dumps(meta)))
    return out_path


def _load_negatives(path) -> tuple[np.ndarray, dict] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta"].item())
        if meta.get("schema_version") != SCHEMA_VERSION:
            return None
        return data["vectors"], meta
    except (OSError, ValueError, KeyError):
        return None


def load_thresholds(path=DEFAULT_CALIBRATION_PATH) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        artifact = json.loads(path.read_text())
        return artifact.get("points") or None
    except (OSError, ValueError):
        return None


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning("span_gate: %s — margin layer disabled (fail-open)", message)


def gate_fingerprint(profiles_path=None, negatives_path=DEFAULT_NEGATIVES_PATH,
                     calibration_path=DEFAULT_CALIBRATION_PATH) -> str:
    """Version stamp for run configs: gate behavior changes iff this changes."""
    h = hashlib.sha256()
    for p in (profiles_path or lp.DEFAULT_PROFILE_PATH, negatives_path, calibration_path):
        p = Path(p)
        h.update(p.read_bytes() if p.exists() else b"absent")
    return h.hexdigest()[:16]


def gate_spans(items, operating_point: str, *, profiles_path=None, negatives_path=None,
               calibration_path=None, embed_fn: Callable | None = None
               ) -> dict[tuple[str, str], GateDecision]:
    profiles_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    negatives_path = Path(negatives_path or DEFAULT_NEGATIVES_PATH)
    calibration_path = Path(calibration_path or DEFAULT_CALIBRATION_PATH)

    out: dict[tuple[str, str], GateDecision] = {}
    margin_todo: list[tuple[tuple[str, str], str]] = []   # (key, surface)
    for surface, runtime_type in items:
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
        out[key] = GateDecision("keep", "open")
        margin_todo.append((key, surface))

    if not margin_todo:
        return out
    points = load_thresholds(calibration_path)
    point = (points or {}).get(operating_point)
    negatives = _load_negatives(negatives_path)
    index = load_embindex(str(_index_path_for(profiles_path)), str(profiles_path))
    if point is None or negatives is None or index is None:
        reason = ("no calibration point" if point is None else
                  "negatives index missing/stale" if negatives is None else
                  "profile embindex missing/stale")
        _warn_once(f"{calibration_path}:{operating_point}", reason)
        return out
    neg_vectors, neg_meta = negatives
    # review guards: a margin drop is only trustworthy when negatives were embedded by the
    # SAME model as the positive index and are non-empty/dimension-compatible; anything else
    # is a stale artifact -> whole layer fails open.
    if (neg_meta.get("model_id") != index.model_id or neg_vectors.size == 0
            or neg_vectors.shape[1] != index.vectors.shape[1]):
        _warn_once(str(negatives_path), "negatives stale (model/dim mismatch or empty)")
        return out
    if embed_fn is None:
        model = _st_model(index.model_id)
        embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
    try:
        vectors = _l2norm(embed_fn([s for _, s in margin_todo]))
    except Exception:
        _warn_once(str(negatives_path), "embedding model failed")
        return out
    floor, margin = float(point["floor"]), float(point["margin"])
    for (key, _), q in zip(margin_todo, vectors):
        rows = index.type_rows(key[0])
        if not rows:  # no positive anchors for this type -> cannot judge, fail open
            out[key] = GateDecision("keep", "open")
            continue
        pos = float(np.max(index.vectors[rows] @ q))
        neg = float(np.max(neg_vectors @ q))
        if pos < floor and (neg - pos) >= margin:
            out[key] = GateDecision("drop", "margin", pos_sim=pos, neg_sim=neg)
        else:
            out[key] = GateDecision("keep", "open", pos_sim=pos, neg_sim=neg)
    return out
