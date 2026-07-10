---
type: plan
status: current
created: 2026-07-10
updated: 2026-07-10
tags: [detector, noise-gate, span-filtering, entity-linking, calibration, weak-supervision]
companion: [docs/specs/detector-noise-semantic-gate.md]
---

# Detector Noise Semantic Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fail-open deny-list ceiling with a semantically generalizing span gate
(link-keep → link-retype → anchor-margin drop → deny-list), run by the miner and the runtime
detector at frozen per-consumer operating points.

**Architecture:** One decision core `cloak.span_gate` consuming existing machinery: exact
entry resolution (`lookup_entry`, all profile types) for keep/retype, the profile embindex as
positive anchors, a new negative-anchor artifact seeded from the `_NOISE_*` deny-lists + a
curated junk exemplar file, and a calibration artifact holding frozen FLOOR/MARGIN per
operating point. Consumers: `scripts/build_mined_lattice_profiles.py` (miner point) and
`cloak.detect` `negative_filter` (production point — also the RL pipeline's path). Absent
artifacts degrade to layers 1/2/4 (current behavior + linking), never crash.

**Tech Stack:** Python 3.12, pytest, numpy, sentence-transformers (bge-small — the embindex
model), skweak (spike only).

## Global Constraints

- **Reuse-first:** layer 1/2 use `cloak.lattice_profiles.lookup_entry` — never a new matcher.
  Positive anchors are the existing `lattice_profiles.embindex.npz` — never re-embedded into a
  second positive index. Embedding model access via `cloak.profile_match._st_model`.
- **Fail-open terminal default:** the gate only drops what layer 3/4 positively justifies; a
  missing/stale anchor or calibration artifact disables layer 3 with a one-time
  `log.warning`, keeping layers 1/2/4.
- **No leakage between anchor seeds and eval negatives:** deny-list surfaces are split
  deterministically (sha256 of surface, even/odd) into anchor-seed and eval halves; the split
  rule is recorded in the calibration artifact.
- **Frozen thresholds (empirical honesty):** production point = largest drop-recall with
  false-drop rate ≤ 0.001 on eval keeps; miner point = drop-precision ≥ 0.99 first. Chosen
  once by `scripts/calibrate_span_gate.py`, recorded in `results/span_gate_calibration.json`;
  if a bar is unreachable, that operating point ships with layer 3 disabled and the sweep is
  the finding. No per-run tuning.
- **Term hygiene:** junk/entity term listings live in data/results files, never stdout. Test
  fixtures use synthetic invented names (`blorbitis`, `flurb disease`, `springtown`), never
  real medical terms.
- **Tests:** `PYTHONPATH=src .venv/bin/python -m pytest <files> -v` from repo root; unit
  tests never require GPU/network — `embed_fn`/lookup functions injectable or monkeypatched.
- **Commits:** path-scoped `git add <files>` only; `git diff --cached --name-only` must be
  empty before staging (shared checkout).
- **GPU:** one GPU process at a time; long runs `.venv/bin/python -u` to a log.

---

### Task G1: span_gate core + negative-anchor artifact + eval-set builder

**Files:**
- Create: `src/cloak/span_gate.py`
- Create: `data/span_gate/junk_exemplars.txt` (curated from
  `docs/issues/2026-07-10-detector-junk-and-noise-gate-limits.md` §2 "arbitrary nouns / true
  junk" list — one lowercase surface per line; file content only, never echoed to stdout)
- Test: `src/cloak/tests/test_span_gate.py`

**Interfaces:**
- Consumes: `lookup_entry(surface, runtime_type, path)` (all `PROFILE_BACKED_TYPES`);
  `cloak.profile_match.load_embindex/_index_path_for/_st_model/_l2norm`;
  `cloak.detect.is_noise_span` and the `_NOISE_LAB_TESTS`, `_NOISE_IMAGING_DIAGNOSTICS`,
  `_NOISE_ANATOMY`, `_NOISE_DEVICE_SUPPLIES` frozensets (anchor seeds).
- Produces (used by Task G2 and consumers):

```python
@dataclass
class GateDecision:
    action: str                 # "keep" | "drop" | "retype"
    layer: str                  # "link" | "retype" | "margin" | "denylist" | "open"
    entry: str | None = None    # resolved canonical (link/retype)
    new_type: str | None = None # retype target runtime type
    pos_sim: float | None = None
    neg_sim: float | None = None

def seed_negative_surfaces() -> list[str]        # deny-list sets ∪ junk_exemplars.txt
def anchor_seed_split(surfaces) -> tuple[list[str], list[str]]   # (anchor_half, eval_half)
def build_negative_index(out_path="data/span_gate/negatives.npz",
                         model_id=DEFAULT_MODEL_ID, embed_fn=None) -> Path
def load_thresholds(path="results/span_gate_calibration.json") -> dict | None
def gate_spans(items, operating_point, *, profiles_path=None, negatives_path=None,
               calibration_path=None, embed_fn=None) -> dict[tuple[str, str], GateDecision]
    # items: [(surface, runtime_type)]; keys as profile_match.span_key
def gate_fingerprint(...) -> str    # sha256 over negatives meta + calibration json + profile hash
```

- Layer order inside `gate_spans` (spec section "Gate architecture"):
  1. `lookup_entry(surface, own type)` hit → `keep/link` with entry.
  2. `lookup_entry(surface, other profile-backed type)` exact hit → `retype` (first hit in
     sorted type order for determinism; only for types in `PROFILE_BACKED_TYPES`).
  3. `is_noise_span(surface, runtime_type)` → `drop/denylist`.
  4. margin: embed uncached surfaces in ONE batch; `pos_sim` = max cosine vs the embindex rows
     of the span's type; `neg_sim` = max cosine vs the negative index; drop iff
     `pos_sim < floor AND (neg_sim - pos_sim) >= margin` for the operating point; else
     `keep/open`. Layer skipped (straight to `keep/open`) when thresholds or either index is
     missing/stale — one-time warning, mirroring `profile_match._warn_exact_only`.
- `negatives.npz` layout mirrors the embindex: `vectors` (L2-normalized float32) +
  `meta` JSON (`schema_version`, `model_id`, `surfaces`, `seed_rule`).

- [ ] **Step 1: failing tests** — create `src/cloak/tests/test_span_gate.py`:

```python
import json

import numpy as np
import pytest

from cloak import span_gate
from cloak.profile_match import span_key


@pytest.fixture()
def profile(tmp_path):
    artifact = {"schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
        "health-condition": {"blorbitis": {"aliases": ["blorb inflammation"],
                                           "levels": ["organ disease"],
                                           "source_ids": ["t:1"], "count": 10.0}},
        "medical-procedure": {"flurbectomy": {"aliases": [], "levels": ["surgical procedure"],
                                              "source_ids": ["t:2"], "count": 5.0}},
    }}
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(artifact))
    return p


def _vec(x):  # 2-d unit vectors for fake embeddings
    v = np.asarray(x, dtype=np.float32)
    return v / np.linalg.norm(v)


FAKE_SPACE = {  # surface -> direction; positives near [1,0], junk near [0,1]
    "blorbitis": [1.0, 0.0], "blorb inflammation": [1.0, 0.05],
    "weird fragment": [0.05, 1.0], "brickish thing": [0.1, 1.0],
    "ambiguous middle": [0.7, 0.7],
}


def fake_embed(texts):
    return np.stack([_vec(FAKE_SPACE.get(t, [0.5, 0.5])) for t in texts])


def test_seed_split_is_deterministic_and_disjoint():
    surfaces = [f"surface {i}" for i in range(50)]
    a1, e1 = span_gate.anchor_seed_split(surfaces)
    a2, e2 = span_gate.anchor_seed_split(list(reversed(surfaces)))
    assert set(a1) == set(a2) and set(e1) == set(e2)
    assert set(a1).isdisjoint(e1) and set(a1) | set(e1) == set(surfaces)


def test_gate_link_keep_and_retype(profile, tmp_path):
    got = span_gate.gate_spans(
        [("Blorbitis", "health-condition"), ("flurbectomy", "health-condition")],
        "production", profiles_path=profile,
        negatives_path=tmp_path / "missing.npz",   # layer 3 disabled -> still links
        calibration_path=tmp_path / "missing.json")
    d1 = got[span_key("Blorbitis", "health-condition")]
    assert (d1.action, d1.layer, d1.entry) == ("keep", "link", "blorbitis")
    d2 = got[span_key("flurbectomy", "health-condition")]
    assert (d2.action, d2.new_type) == ("retype", "medical-procedure")


def test_gate_margin_drops_junk_keeps_ambiguous(profile, tmp_path):
    neg = tmp_path / "negatives.npz"
    span_gate.build_negative_index(out_path=neg, embed_fn=fake_embed,
                                   surfaces=["weird fragment"])
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"schema_version": 1, "model_id": "fake",
        "points": {"production": {"floor": 0.6, "margin": 0.2}}}))
    # embindex for the profile with the fake embedder
    from cloak.profile_match import build_embindex
    build_embindex(profile, embed_fn=fake_embed, model_id="fake")
    got = span_gate.gate_spans(
        [("brickish thing", "health-condition"), ("ambiguous middle", "health-condition")],
        "production", profiles_path=profile, negatives_path=neg, calibration_path=calib,
        embed_fn=fake_embed)
    assert got[span_key("brickish thing", "health-condition")].action == "drop"
    assert got[span_key("ambiguous middle", "health-condition")].action == "keep"  # fail-open


def test_gate_fails_open_without_artifacts(profile, tmp_path):
    got = span_gate.gate_spans([("brickish thing", "health-condition")], "production",
                               profiles_path=profile,
                               negatives_path=tmp_path / "none.npz",
                               calibration_path=tmp_path / "none.json")
    d = got[span_key("brickish thing", "health-condition")]
    assert (d.action, d.layer) == ("keep", "open")


def test_denylist_layer_still_fires(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(span_gate, "is_noise_span",
                        lambda s, t: s == "known junk", raising=False)
    got = span_gate.gate_spans([("known junk", "health-condition")], "miner",
                               profiles_path=profile,
                               negatives_path=tmp_path / "none.npz",
                               calibration_path=tmp_path / "none.json")
    assert got[span_key("known junk", "health-condition")].layer == "denylist"
```

- [ ] **Step 2: run to fail** — `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_span_gate.py -v`
  → `ModuleNotFoundError: No module named 'cloak.span_gate'`.

- [ ] **Step 3: implement `src/cloak/span_gate.py`** (complete implementation; docstring cites
  the spec path):

```python
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
        pos = float(np.max(index.vectors[rows] @ q)) if rows else 0.0
        neg = float(np.max(neg_vectors @ q)) if neg_vectors.size else 0.0
        if pos < floor and (neg - pos) >= margin:
            out[key] = GateDecision("drop", "margin", pos_sim=pos, neg_sim=neg)
        else:
            out[key] = GateDecision("keep", "open", pos_sim=pos, neg_sim=neg)
    return out
```

- [ ] **Step 4: junk exemplars file** — extract the "arbitrary nouns / true junk" surfaces
  from the issue doc §2 into `data/span_gate/junk_exemplars.txt`, one per line, lowercase.
- [ ] **Step 5: run to green** — same pytest command; all 5 tests pass.
- [ ] **Step 6: commit** —
  `git add src/cloak/span_gate.py src/cloak/tests/test_span_gate.py data/span_gate/junk_exemplars.txt`
  `git commit -m "feat(span-gate): layered semantic gate core + negative-anchor index"`

---

### Task G2: calibration script + miner & runtime wiring

**Files:**
- Create: `scripts/calibrate_span_gate.py`
- Modify: `scripts/build_mined_lattice_profiles.py` (the `is_noise_span` call site, ~line 133)
- Modify: `src/cloak/detect.py` (`negative_filter` block, ~lines 400–402)
- Test: `src/cloak/tests/test_span_gate_wiring.py`

**Interfaces:**
- Consumes: everything Task G1 produces.
- Produces: `results/span_gate_calibration.json`:

```json
{"schema_version": 1, "model_id": "...", "seed_rule": "sha256-even-anchor",
 "eval": {"keeps": 0, "drops": 0},
 "sweep": [{"floor": 0.0, "margin": 0.0, "point": "production",
            "false_drop_rate": 0.0, "drop_recall": 0.0, "drop_precision": 0.0}],
 "points": {"production": {"floor": 0.0, "margin": 0.0},
            "miner": {"floor": 0.0, "margin": 0.0}}}
```

  A point absent from `points` = that operating point's margin layer is disabled (bar not
  reached) — `gate_spans` already fail-opens.
- Miner wiring: the `is_noise_span` branch becomes a batch `gate_spans(..., "miner")` pass
  over the unique spans before the per-span loop; decisions: `drop` → `stats["gate_dropped"]`
  (deny-list drops keep the existing `noise_skipped` counter), `retype` → reassign
  `runtime_type` + `stats["gate_retyped"]`, `keep` → proceed. Stats gain
  `"gate_fingerprint": gate_fingerprint()`.
- detect.py wiring: replace the `is_noise_span` list-comprehension with a batch
  `gate_spans([(s.text, s.type) for s in spans if s.type in _NOISE_FILTER_TYPES], "production")`
  pass; `drop` removes the span, `retype` rewrites `Span.type` (keep score/source), `keep`
  passes through. Types outside `_NOISE_FILTER_TYPES` bypass the gate entirely (current
  contract). Import `span_gate` lazily inside the branch (detect.py must not import numpy at
  module load for non-clinical profiles).

Calibration eval set (built inside `scripts/calibrate_span_gate.py`, no separate file):
- **keeps** = all profile canonical+alias surfaces of `_NOISE_FILTER_TYPES` types (they must
  never margin-drop; linking already keeps exact hits, so evaluate them with layer 1/2
  bypassed — the script scores raw margin decisions on embeddings only).
- **drops** = eval half of `anchor_seed_split(seed_negative_surfaces())` (held out from the
  anchor index by construction).
- Sweep `floor ∈ {0.4..0.8 step 0.05} × margin ∈ {0.0..0.4 step 0.05}`; select per the
  Global Constraints bars; write the full sweep + chosen points; print only counts/rates.

- [ ] **Step 1: failing tests** — `src/cloak/tests/test_span_gate_wiring.py`:

```python
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from calibrate_span_gate import choose_points, margin_scores


def test_margin_scores_and_choose_points_bars():
    # keeps at pos .9 / neg .1 ; drops at pos .1 / neg .9 -> separable
    keeps = [(0.9, 0.1)] * 100
    drops = [(0.1, 0.9)] * 50
    sweep, points = choose_points(keeps, drops,
                                  floors=[0.5], margins=[0.2],
                                  production_false_drop=0.001, miner_precision=0.99)
    assert points["production"] == {"floor": 0.5, "margin": 0.2}
    assert points["miner"] == {"floor": 0.5, "margin": 0.2}
    row = sweep[0]
    assert row["false_drop_rate"] == 0.0 and row["drop_recall"] == 1.0
    # inseparable data -> no production point
    _, points2 = choose_points([(0.1, 0.9)] * 100, [(0.1, 0.9)] * 100,
                               floors=[0.5], margins=[0.2],
                               production_false_drop=0.001, miner_precision=0.99)
    assert "production" not in points2


def test_miner_gate_wiring_drop_retype_keep(monkeypatch, tmp_path):
    import build_mined_lattice_profiles as m
    from cloak.span_gate import GateDecision
    from cloak.profile_match import span_key

    decisions = {
        span_key("junky fragment", "injury"): GateDecision("drop", "margin"),
        span_key("flurbectomy", "injury"): GateDecision("retype", "retype",
                                                        new_type="medical-procedure"),
        span_key("blorbitis", "injury"): GateDecision("keep", "open"),
    }
    monkeypatch.setattr(m, "gate_spans", lambda items, point, **kw: decisions)
    spans = [m.DetectedSpan(s, "injury", "doc1", 0.9)
             for s in ("junky fragment", "flurbectomy", "blorbitis")]
    rows, stats = m.build_rows_for_test(spans)   # thin seam added in Step 3
    assert stats["gate_dropped"] == 1 and stats["gate_retyped"] == 1
    assert ("medical-procedure" in rows) and ("blorbitis" in rows.get("injury", {}))
```

(The exact miner seam name/shape may differ — the constraint that binds: gate decisions must
be applied per unique span before profile-row creation, with the three stats above; expose
the smallest testable seam that achieves it, e.g. factoring the loop body into a helper.)

- [ ] **Step 2: run to fail** — `ModuleNotFoundError: No module named 'calibrate_span_gate'`.
- [ ] **Step 3: implement** `scripts/calibrate_span_gate.py` with pure functions
  `margin_scores(surfaces, index_vectors, neg_vectors, embed_fn) -> list[(pos, neg)]` and
  `choose_points(keeps, drops, *, floors, margins, production_false_drop, miner_precision)
  -> (sweep, points)`; a `main()` that loads the real embindex + negatives, builds the eval
  set, sweeps, writes the artifact. Then wire the miner and detect.py per the Interfaces
  block (smallest seams; keep existing stats keys intact).
- [ ] **Step 4: green + regression** — the new test file passes AND
  `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_profiles.py src/cloak/tests/test_build_mined_lattice_profiles.py -q` stays green (wiring must not change
  behavior when gate artifacts are absent — fail-open).
- [ ] **Step 5: commit** — path-scoped, message
  `"feat(span-gate): calibration script + miner/runtime wiring at frozen operating points"`.

---

### Task G3: execution — build artifacts, calibrate, measure on the re-mine

GPU + local only; no paid calls.

- [ ] **Step 1:** `PYTHONPATH=src .venv/bin/python -c "from cloak.span_gate import build_negative_index; print(build_negative_index())"`
- [ ] **Step 2:** `PYTHONPATH=src .venv/bin/python -u scripts/calibrate_span_gate.py 2>&1 | tee results/span_gate_calibration.log` — report which operating points met their bars
  (counts/rates only on stdout).
- [ ] **Step 3:** measure on the measured re-mine: rerun
  `scripts/build_mined_lattice_profiles.py` span-processing over
  `results/mined_lattice_profile_spans_large.jsonl` (detection already cached in the jsonl —
  no detector run) and report per-type `gate_dropped` / `gate_retyped` / kept counts vs the
  pre-gate baseline; listings to `results/span_gate_mine_report.json`.
- [ ] **Step 4:** full suite `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/ -q`
  (pre-existing unrelated failure excepted); commit artifacts:
  `results/span_gate_calibration.json`, `results/span_gate_mine_report.json`,
  `data/span_gate/negatives.npz` only if git-trackable (check `.gitignore`; embindex-style
  artifacts are ignored — then commit the calibration/report JSONs only).
- [ ] **Step 5: STOP** — RL-config fingerprint integration and any paid-teacher validation
  are follow-ups needing user direction.

---

### Task G4 (concurrent spike, after G1): skweak adoption spike

**Files:** Create: `scripts/spikes/skweak_gate_spike.py`

- [ ] Install check: `.venv/bin/pip install skweak` into the venv? **No** — spike must not
  mutate the shared venv without approval; use `uv pip install --python .venv/bin/python skweak`
  only after asking the user, OR run with `pipx`/isolated venv. If install is not approved,
  record the spike as blocked in the spec's adoption note.
- [ ] Fit skweak's HMM over LFs (deny-list hit, link verdict, margin bucket, detector-score
  bucket) on the 2,770 re-mine spans; compare keep/drop vs the layered gate on the
  calibration eval set; write `results/skweak_spike.json` (drop-precision/recall both
  systems). Adoption rule (spec): adopt only if skweak beats the layered gate on
  drop-precision at ≥ equal drop-recall.

---

## Execution notes for the SDD controller

- Tasks G1 → G2 → G3 serial; G4 after G1, concurrent with G2/G3 (disjoint files).
- Dedup-track leftovers (uncommitted reprocessed `lattice_profiles.json`) are **not** part of
  this plan — data state questions go to the user.
- The dedup plan's paid-teacher E2E stop still stands.
