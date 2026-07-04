---
type: plan
status: current
created: 2026-07-04
updated: 2026-07-04
tags: [privacy, lattice, anonymity-set, per-type-tau, mask, inference-architecture]
companion: ../research/inference-risk-enforcement.md
---

# Structural Lattice Risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inference-time `walk_risk` LM probe with anonymity-set counts stored on
lattice levels, enforced as per-type count floors — zero risk models on the inference path.

**Architecture:** A new `src/cloak/anonymity.py` computes, per (fill, type, original), how
many candidate values of the original's granularity are consistent with the fill (GeoNames
counts for LOC, WordNet hyponym-leaf counts for DEM/ORG/MISC, window/granularity for
DATETIME, range/step for QUANTITY). A free rescore of the existing 150-item shootout
validates `1/count` against the cached attacker labels next to `walk_risk`. The arms artifact
is annotated in place (never re-detected), a keep-original action is inserted per span, the
env/trainer legal-set derivation switches from `walk_risk < tau` to `aset >= k_floors[type]`,
and the probe retires to offline duty.

**Tech Stack:** Python 3.12, host `.venv` (GPU torch — needed only for env rebuild probes),
GeoNames files already at `data/geonames/`, NLTK WordNet (already a lattice dependency),
existing shootout machinery in `scripts/spikes/privacy_probe_shootout.py`.

## Global Constraints

- **Never re-detect**: detection is process-nondeterministic; the arms artifact
  `data/task_arms_tau0.02.json` is annotated in place, never rebuilt (CLAUDE.md / spec).
- **Testing convention**: this repo uses `__main__` assert self-checks, not pytest suites.
  Each code task ends with a runnable self-check.
- **Empirical honesty**: the count measure is *promoted only by measured attacker
  correlation* (Task 2 gate); floor values are calibrated from the measured table, never
  invented. Report regressions plainly.
- **Naming**: no spec-section identifiers in code; the measure is `aset` (anonymity-set
  count), floors are `k_floors`.
- **Runs**: all long runs `.venv/bin/python -u`; one GPU process at a time; every run in this
  plan is local (the shootout rescore makes zero remote calls).
- **Git**: commit at the end of every task; messages end with the Claude co-author line.

---

### Task 1: Anonymity-set counting module

**Files:**
- Create: `src/cloak/anonymity.py`

**Interfaces:**
- Consumes: `cloak.lattice._load_geo`, `cloak.lattice.CONTINENTS`,
  `cloak.lattice.TYPE_LABEL`, `cloak.lattice._MONTHS`, NLTK WordNet.
- Produces: `aset_count(fill: str, span_type: str, original: str, strict: bool = False) ->
  float` (1.0 = keep-original / fail-closed; `GENERIC = 1e9` for type-label coarse fills and
  placeholders; **`strict=True` is the certifying mode** — parse miss fails closed with no
  last-word WordNet fallback, so legality can never ride an over-count; permissive mode is
  diagnostics-only), `K_FLOORS: dict[str, float]` (per-type floors; provisional until Task 2
  calibration).

- [ ] **Step 1: Write the module**

```python
"""Anonymity-set counts for lattice levels — structural risk, zero models at inference.

count = number of candidate values AT THE ORIGINAL'S GRANULARITY consistent with the fill
(k-anonymity's consistency set transplanted to span substitution; design:
docs/research/inference-risk-enforcement.md). Per-type floors replace the walk_risk
tau-mask: legal iff aset >= k_floors[type] (placeholder always legal, keep-original has
aset 1 -> legal only for user-waived types). Units need only be consistent WITHIN a type —
floors are per-type, so city-counts and day-windows never compare.

Fail-closed: a specific-looking but unparseable fill counts 1 (illegal under any floor > 1,
same conservative direction as the probe's unpooled-type rule). The six TYPE_LABEL coarse
fills are known-generic -> GENERIC, checked first.

Two modes. strict=True (CERTIFYING — legality, artifact annotation, decode-time checks):
a parse miss fails closed, never falls through to a broader lookup; in particular the
WordNet last-word fallback is off ("a coastal Norwegian city" -> no full-phrase synset ->
1.0, NOT the count of all cities). Deterministic is not un-gameable: permissive fallbacks
fail OPEN on adversarial phrasings, and a generation loop would mine them. strict=False is
for diagnostics only.
"""
import re
from collections import Counter
from decimal import Decimal
from functools import lru_cache

from cloak.lattice import CONTINENTS, TYPE_LABEL, _MONTHS, _load_geo

GENERIC = 1e9

# provisional floors — Task 2 replaces these with values calibrated on the measured
# count-vs-attacker table (empirical-honesty rule: never ship invented numbers)
K_FLOORS = {"LOC": 50.0, "ORG": 50.0, "DATETIME": 30.0,
            "DEM": 1.0, "QUANTITY": 1.0, "MISC": 1.0, "OTHER": 1.0}

_geo_counts_cache = None


def _geo_counts():
    """Reverse count indexes over cities500: unique city names per admin1 region name,
    per country name, per continent name; plus countries per continent."""
    global _geo_counts_cache
    if _geo_counts_cache is None:
        cities, admin1, countries = _load_geo()
        by_region, by_country, by_cont, cont_countries = (Counter(), Counter(),
                                                          Counter(), Counter())
        for ccode, a1, _pop in cities.values():
            cn = countries.get(ccode)
            if not cn:
                continue
            by_country[cn["name"].lower()] += 1
            cont = CONTINENTS.get(cn["continent"])
            if cont:
                by_cont[cont.lower()] += 1
            region = admin1.get(f"{ccode}.{a1}")
            if region:
                by_region[region.lower()] += 1
        for cc in countries.values():
            cont = CONTINENTS.get(cc["continent"])
            if cont:
                cont_countries[cont.lower()] += 1
        _geo_counts_cache = (by_region, by_country, by_cont, cont_countries)
    return _geo_counts_cache


def _loc_count(fill: str) -> float | None:
    by_region, by_country, by_cont, cont_countries = _geo_counts()
    m = re.fullmatch(r"a city in (.+)", fill.strip(), re.IGNORECASE)
    if m:
        key = m.group(1).lower()
        n = by_cont.get(key) or by_country.get(key) or by_region.get(key)
        return float(n) if n else None
    m = re.fullmatch(r"a country in (.+)", fill.strip(), re.IGNORECASE)
    if m:
        n = cont_countries.get(m.group(1).lower())
        return float(n) if n else None
    return None


_MONTH_RE = "|".join(_MONTHS)
_AGE_FILL = {"teenaged": 7.0}  # plus "X-something" -> 10 (a decade of ages)


def _date_granularity_days(original: str) -> float | None:
    t = original.strip().lower()
    has_year = re.search(r"\b(19|20)\d{2}\b", t)
    has_month = re.search(rf"\b({_MONTH_RE})\b", t)
    has_day = re.search(r"\b([0-3]?\d)(st|nd|rd|th)?\b", t)
    if has_month and has_year and has_day and not re.fullmatch(r"(19|20)\d{2}", t):
        return 1.0       # "12 March 2019"
    if has_month and has_year:
        return 30.0      # "March 2019"
    if has_month:
        return 30.0      # "March", "November 7" (day unknown-year: month-grain)
    if has_year:
        return 365.0     # "2019"
    return None          # relative phrases ("last year"): no absolute granularity


def _date_window_days(fill: str) -> float | None:
    f = fill.strip().lower()
    if re.fullmatch(r"the (early|mid|late) (19|20)\d0s", f):
        return 3652 / 3
    if re.fullmatch(r"the (19|20)\d0s", f):
        return 3652.0
    if re.fullmatch(rf"({_MONTH_RE}) (19|20)\d{{2}}", f):
        return 30.0
    if re.fullmatch(r"the (winter|spring|summer|autumn|fall)", f):
        return 91.0
    return None


def _datetime_count(fill: str, original: str) -> float | None:
    f = fill.strip().lower()
    if f in _AGE_FILL:
        return _AGE_FILL[f]
    if re.fullmatch(r"\w+-something", f):
        return 10.0
    if f in ("at some point", "some time ago"):
        return GENERIC
    window = _date_window_days(fill)
    if window is None:
        return None
    gran = _date_granularity_days(original)
    if gran is None:
        return None
    return max(window / gran, 1.0)


def _step(v: float) -> float:
    """Magnitude of the least-significant digit: 40 -> 10, 45 -> 1, 2.5 -> 0.1."""
    d = Decimal(str(v)).normalize()
    return float(Decimal(1).scaleb(d.as_tuple().exponent))


def _quantity_count(fill: str, original: str) -> float | None:
    m = re.search(r"between ([\d,\.]+) and ([\d,\.]+)", fill, re.IGNORECASE)
    om = re.search(r"[\d][\d,.]*", original)
    if not (m and om):
        return None
    try:
        lo, hi = (float(m.group(i).replace(",", "")) for i in (1, 2))
        v = float(om.group().replace(",", ""))
    except ValueError:
        return None
    return max((hi - lo) / _step(v) + 1, 1.0)


@lru_cache(maxsize=4096)
def _wn_leaf_count(phrase: str, strict: bool = False) -> float | None:
    """Leaves of the hyponym closure under the phrase's first noun synset (the same
    sense-selection policy as lattice.wordnet_chain). strict: full-phrase synset only —
    the last-word fallback over-counts ("...city" -> all cities), a fail-open a
    generation loop would mine."""
    from nltk.corpus import wordnet as wn
    p = re.sub(r"^(an?|the) ", "", phrase.lower().strip())
    syns = wn.synsets(p.replace(" ", "_"), pos=wn.NOUN)
    if not syns and not strict and " " in p:
        syns = wn.synsets(p.split()[-1], pos=wn.NOUN)  # diagnostic-only fallback
    if not syns:
        return None
    seen, stack, leaves = set(), [syns[0]], 0
    while stack:
        s = stack.pop()
        if s.name() in seen:
            continue
        seen.add(s.name())
        subs = s.hyponyms() + s.instance_hyponyms()
        if subs:
            stack.extend(subs)
        else:
            leaves += 1
    return float(max(leaves, 1))


@lru_cache(maxsize=8192)
def aset_count(fill: str, span_type: str, original: str, strict: bool = False) -> float:
    """Anonymity-set size of `fill` for a span of `span_type` whose original text is
    `original`. 1.0 for keep-original and for fail-closed unparseable fills; GENERIC for
    type-label coarse fills. strict=True for anything that certifies legality."""
    if fill.lower().strip() == original.lower().strip():
        return 1.0
    if fill.lower().strip() in {v.lower() for v in TYPE_LABEL.values()}:
        return GENERIC
    got = None
    if span_type == "LOC":
        got = _loc_count(fill) or _wn_leaf_count(fill, strict)
    elif span_type == "DATETIME":
        got = _datetime_count(fill, original)
    elif span_type == "QUANTITY":
        got = _quantity_count(fill, original)
    else:  # DEM / ORG / MISC / OTHER — WordNet lattices
        got = _wn_leaf_count(fill, strict)
    return got if got else 1.0  # ponytail: fail-closed; per-surface overrides if too strict


if __name__ == "__main__":
    assert aset_count("Oslo", "LOC", "Oslo") == 1.0                      # keep-original
    n_no = aset_count("a city in Norway", "LOC", "Oslo")
    n_eu = aset_count("a city in Europe", "LOC", "Oslo")
    assert 4 < n_no < n_eu, (n_no, n_eu)                                 # ordering
    assert aset_count("a place", "LOC", "Oslo") == GENERIC               # TYPE_LABEL
    d_mo = aset_count("March 2019", "DATETIME", "12 March 2019")
    d_dec = aset_count("the 2010s", "DATETIME", "12 March 2019")
    assert 1 < d_mo < d_dec, (d_mo, d_dec)
    assert aset_count("thirty-something", "DATETIME", "34") == 10.0
    q = aset_count("between 20 and 80 milligrams", "QUANTITY", "40 milligrams")
    assert q == 7.0, q                                                    # step 10
    w_hd = aset_count("a heart disease", "DEM", "hypertension")
    w_d = aset_count("a disease", "DEM", "hypertension")
    assert 1 < w_hd < w_d, (w_hd, w_d)
    assert aset_count("xyzzy blorp", "DEM", "hypertension") == 1.0        # fail-closed
    # strict mode: no last-word fallback — a narrowing phrase must fail CLOSED, while
    # permissive mode may fail OPEN (that asymmetry is why legality always uses strict)
    assert aset_count("a coastal Norwegian city", "LOC", "Bergen", strict=True) == 1.0
    assert aset_count("a coastal Norwegian city", "LOC", "Bergen") > 1.0  # diagnostic
    print("anonymity.py self-check OK")
```

- [ ] **Step 2: Run the self-check**

Run: `PYTHONPATH=src .venv/bin/python src/cloak/anonymity.py`
Expected: `anonymity.py self-check OK`. If a WordNet assertion fails on sense selection or a
GeoNames name mismatch, fix the parsing (the ordering assertions are the contract; exact
numbers are not).

- [ ] **Step 3: Commit**

```bash
git add src/cloak/anonymity.py
git commit -m "feat: anonymity-set counts for lattice levels (structural risk measure)"
```

---

### Task 2: Validation shootout — 1/count vs walk_risk against cached attacker labels

**Files:**
- Create: `scripts/spikes/lattice_count_shootout.py`

**Interfaces:**
- Consumes: `results/privacy_probe_shootout.json` (items with per-item `hit1`/`hit5`
  attacker labels and stored `p4`/`p6` scores), `privacy_probe_shootout.auc`,
  `privacy_probe_shootout.per_span_rank_agreement`, `cloak.anonymity.aset_count` (Task 1).
- Produces: `results/lattice_count_shootout.json` — AUC + level-ordering for `1/aset`
  next to the stored probes, plus the per-type count-bucket vs attacker-hit-rate table
  used to calibrate `K_FLOORS`.

- [ ] **Step 1: Write the spike**

```python
"""Promotion shootout for the structural risk measure: score = 1/aset_count, evaluated
against the SAME cached attacker labels as the probe shootout (zero remote calls, zero
GPU). Decision gate for docs/research/inference-risk-enforcement.md: adopt structural
counts if level-ordering is within 0.05 of walk_risk's; otherwise counts still mask but
floors are LM-calibrated per (type, depth) offline.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/lattice_count_shootout.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from privacy_probe_shootout import auc, per_span_rank_agreement  # noqa: E402

from cloak.anonymity import aset_count  # noqa: E402

report = json.loads(Path("results/privacy_probe_shootout.json").read_text())
items = report["items"]
for it in items:
    it["aset"] = aset_count(it["fill"], it["type"], it["surface"], strict=True)
    it["inv_aset"] = 1.0 / it["aset"]
    it["fail_closed"] = (it["aset"] == 1.0
                         and it["fill"].lower().strip() != it["surface"].lower().strip())

out = {"n_items": len(items), "scores": {}}
for probe in ("p4", "p6", "inv_aset"):
    out["scores"][probe] = {
        "auc_hit1": auc([it[probe] for it in items], [it["hit1"] for it in items]),
        "auc_hit5": auc([it[probe] for it in items], [it["hit5"] for it in items]),
        "level_ordering": per_span_rank_agreement(items, probe),
    }
    print(probe, out["scores"][probe], flush=True)

# calibration table: attacker hit rate per (type, count bucket) -> floor per type
buckets = defaultdict(list)
for it in items:
    b = 0 if it["aset"] < 10 else 1 if it["aset"] < 100 else 2 if it["aset"] < 1e4 else 3
    buckets[(it["type"], b)].append(it["hit5"])
table = {f"{t}|{['<10', '10-100', '100-10k', '>=10k'][b]}":
         {"n": len(v), "attacker_hit5": round(sum(v) / len(v), 3)}
         for (t, b), v in sorted(buckets.items())}
out["calibration_table"] = table
for k, v in table.items():
    print(f"{k:24s} {v}", flush=True)

# fail-closed rate per type (strict certifying mode): sizes the utility cost of parse
# misses and therefore the case for the offline LM count-estimator (build-time tool)
fc = defaultdict(lambda: [0, 0])
for it in items:
    fc[it["type"]][1] += 1
    fc[it["type"]][0] += it["fail_closed"]
out["fail_closed_rate"] = {t: {"rate": round(a / b, 3), "n": b}
                           for t, (a, b) in sorted(fc.items())}
print("fail_closed_rate:", out["fail_closed_rate"], flush=True)

Path("results/lattice_count_shootout.json").write_text(json.dumps(out, indent=1))
print("-> results/lattice_count_shootout.json")
```

Note: `per_span_rank_agreement`'s exact signature lives in
`scripts/spikes/privacy_probe_shootout.py` (it computed the stored per-probe ordering
numbers). If it takes `(items, key)` differently, adapt the call — the reference numbers to
reproduce are walk_risk's level-ordering .86 (local referee) from
`docs/specs/RL/surrogate-ranker-infiller.md` §4.2 table.

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/lattice_count_shootout.py`
Expected: three score rows (p4, p6, inv_aset) + the calibration table, written to
`results/lattice_count_shootout.json`. Wall < 2 min, no remote calls.

- [ ] **Step 3: DECISION GATE — present to the user, do not proceed silently**

Report to the user: `inv_aset` level-ordering and AUC vs `p4`'s stored numbers, the
calibration table, and the per-type fail-closed rate (high fail-closed = the strict parser
is starving that type of legal levels — the utility argument for building the offline LM
count-estimator that populates counts for out-of-universe lattice nodes). Two pre-registered
outcomes (design doc §4-1):
- ordering within 0.05 of walk_risk → **structural flavor**: set `K_FLOORS` per type to the
  smallest bucket edge whose attacker_hit5 ≤ the tau_walk reference rate for that type.
- ordering clearly worse → **LM-calibrated flavor**: keep `aset` as the mask *variable* but
  set floors per (type, depth) from an offline walk_risk pass; record the measured deficit.

Update `K_FLOORS` in `src/cloak/anonymity.py` with the user-approved values and drop the
"provisional" comment wording.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/lattice_count_shootout.py results/lattice_count_shootout.json \
        src/cloak/anonymity.py
git commit -m "feat: count-vs-attacker shootout + calibrated k floors (gate: <outcome>)"
```

---

### Task 3: Annotate the arms artifact — aset per action + keep-original action

**Files:**
- Create: `scripts/annotate_lattice_counts.py` (durable — re-run on any lattice change)

**Interfaces:**
- Consumes: `data/task_arms_tau0.02.json` (action tables: per span
  `{surface, type, actions: [{fill, mode, walk_risk, p6}...], bc_action}` with placeholder
  last), `cloak.anonymity.aset_count`.
- Produces: the same artifact, in place, where every level action carries `"aset": float`,
  and every span has one keep-original action `{fill: surface, mode: "level", keep: true,
  walk_risk: 1.0, p6: 1.0, aset: 1.0}` inserted at index `len(actions)-1` (before the
  placeholder, so placeholder stays `actions[-1]` and level indices are unchanged);
  `bc_action` remapped when it pointed at the placeholder.

- [ ] **Step 1: Write the annotator**

```python
"""Annotate the arms artifact with anonymity-set counts and keep-original actions.

In-place, idempotent, NO re-detection (the artifact's spans/lattices/walk decisions are
frozen; spec §3.3-5). Level actions gain "aset"; each span gains a keep-original action
inserted before the trailing placeholder (level indices unchanged; bc_action remapped when
it pointed at the placeholder). Placeholder actions carry no aset — they are always legal.

Run: PYTHONPATH=src .venv/bin/python scripts/annotate_lattice_counts.py
"""
import json
from pathlib import Path

from cloak.anonymity import aset_count

PATH = Path("data/task_arms_tau0.02.json")

art = json.loads(PATH.read_text())
n_spans = n_keep = 0
for corpus, per_doc in art.items():
    for doc_id, entry in per_doc.items():
        for key, span in entry.get("action_table", {}).items():
            acts = span["actions"]
            assert acts[-1]["mode"] == "placeholder", (doc_id, key)
            n_spans += 1
            for a in acts:
                if a["mode"] == "level":
                    a["aset"] = aset_count(a["fill"], span["type"], span["surface"],
                                           strict=True)  # certifying mode, always
            if not any(a.get("keep") for a in acts):  # idempotency
                keep = {"fill": span["surface"], "mode": "level", "keep": True,
                        "walk_risk": 1.0, "p6": 1.0, "aset": 1.0}
                acts.insert(len(acts) - 1, keep)
                if span["bc_action"] == len(acts) - 1 - 1:  # pointed at old placeholder
                    span["bc_action"] = len(acts) - 1
                n_keep += 1
            assert acts[-1]["mode"] == "placeholder"
            bc = acts[span["bc_action"]]
            assert bc["mode"] == "placeholder" or not bc.get("keep"), (doc_id, key)

PATH.write_text(json.dumps(art))
print(f"annotated {n_spans} spans, inserted {n_keep} keep actions -> {PATH}")
```

Check the artifact's actual JSON shape first (`action_table` key name and whether
`bc_action` lives on the span dict) against `scripts/build_arms_artifact.py:action_table` —
the annotator must match it exactly.

- [ ] **Step 2: Run it, then verify nothing downstream broke**

```bash
PYTHONPATH=src .venv/bin/python scripts/annotate_lattice_counts.py
# BC reproduction must still hold (bc_action remap correctness):
PYTHONPATH=src:scripts .venv/bin/python -u scripts/train_ranker.py --smoke 2>&1 | \
    grep "BC reproduction"
```
Expected: `annotated 177 spans, inserted ~177 keep actions`, then
`BC reproduction verified ... on all 2 docs` (smoke uses 2 docs). Run the annotator a second
time and confirm `inserted 0 keep actions` (idempotent).

- [ ] **Step 3: Commit**

```bash
git add scripts/annotate_lattice_counts.py data/task_arms_tau0.02.json
git commit -m "feat: aset counts + keep-original actions in the arms artifact"
```

---

### Task 4: Switch env + trainer legal sets to per-type count floors

**Files:**
- Modify: `scripts/build_ranker_env.py` (env carries `k_floors`, spans copy `aset`/`keep`)
- Modify: `scripts/train_ranker.py` (legal-set derivation, floor-walk BC teacher,
  `--floors` / `--randomize-floors` args)
- Modify: `src/cloak/train/ranker.py` (`action_features` gains `log_aset` and active-floor
  features; `N_FEAT` += 2)

**Interfaces:**
- Consumes: annotated artifact (Task 3), `cloak.anonymity.K_FLOORS` (Task 2 values).
- Produces: `data/ranker_env.json` with `"k_floors": {...}` (replacing `"tau"` as the
  operating knob; tau kept in the file for provenance only); trainer flag
  `--floors "LOC=50,ORG=50,DATETIME=30,DEM=1,QUANTITY=1,MISC=1,OTHER=1"` (default:
  env values) and `--randomize-floors` (per-episode log-uniform k_T in [1, 10·k_T],
  active floors appended to features — the tau-conditioned-training design).

- [ ] **Step 1: env builder** — add to the env dict in `scripts/build_ranker_env.py`:

```python
from cloak.anonymity import K_FLOORS
env["k_floors"] = K_FLOORS          # alongside the legacy "tau" (provenance only)
env["risk_measure"] = "aset (anonymity-set count); walk_risk retained offline-only"
```
(the spans/action tables are copied from the artifact and now already carry `aset`/`keep`).

- [ ] **Step 2: trainer legal sets + BC teacher** — in `scripts/train_ranker.py` replace the
tau-derived block inside `main()`:

```python
# legal = per-type count floor (structural risk; walk_risk is offline-only).
# floor-walk teacher = most specific legal level, else placeholder — replaces the
# artifact's tau-walk bc_action, which described the retired tau mask.
floors = dict(env["k_floors"])
if args.floors:
    floors.update((t, float(k)) for t, k in
                  (kv.split("=") for kv in args.floors.split(",")))
for s in d["spans"]:
    s = dict(s)
    k = floors.get(s["type"], 1.0)
    s["legal"] = [i for i, a in enumerate(s["actions"])
                  if a["mode"] == "placeholder" or a.get("aset", 0) >= k]
    s["bc_action"] = next((i for i, a in enumerate(s["actions"])
                           if a["mode"] == "level" and a.get("aset", 0) >= k),
                          len(s["actions"]) - 1)
    spans.append(s)
```
Gate `verify_bc_reproduction` to run only when the floor-walk choice equals the artifact's
stored `bc_action` on every span (the reproduction reference is the *tau*-walk's doc_p; under
new floors the teacher legitimately differs — assert instead that every `bc_action` is in
`legal` and one full `assemble()` per doc raises no injectivity error).

- [ ] **Step 3: features** — in `src/cloak/train/ranker.py`:

```python
import math
# Action features: [is_placeholder, walk_risk, p6, level_index/4, n_levels/4,
#                   log10_aset/9, log10_active_floor/9, type one-hot (7), corpus one-hot (3)]
N_FEAT = 7 + len(TYPES) + len(CORPORA)

def action_features(span: dict, corpus: str, floor: float = 1.0) -> torch.Tensor:
    ...
    rows.append([1.0 if a["mode"] == "placeholder" else 0.0,
                 a["walk_risk"], a["p6"], min(i, 4) / 4.0, min(n_lvl, 4) / 4.0,
                 math.log10(max(a.get("aset", 1e9), 1.0)) / 9.0,
                 math.log10(max(floor, 1.0)) / 9.0]
                + t_oh + c_oh)
```
Update the `__main__` self-check's expected shape, and every `action_features(s, corpus)`
call site in `scripts/train_ranker.py` to pass the active floor. With `--randomize-floors`,
features must be rebuilt per episode from the sampled floors (move the `feats` construction
from doc-load time into the epoch loop for that mode; the fixed-floor path keeps precomputed
features).

- [ ] **Step 4: Run the checks**

```bash
PYTHONPATH=src .venv/bin/python src/cloak/train/ranker.py     # self-check, new N_FEAT
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_ranker_env.py
PYTHONPATH=src:scripts .venv/bin/python -u scripts/train_ranker.py --smoke
```
Expected: self-check OK; env rebuild reports the same doc/probe counts as before (probes are
cached and the split seed is unchanged); smoke run prints the train-set line with a
`floors=` summary and completes 2 epochs. Report the new legal-set statistics (spans with
≥ 2 legal actions — compare against the 72/177 measured under tau=0.02).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_ranker_env.py scripts/train_ranker.py src/cloak/train/ranker.py \
        data/ranker_env.json
git commit -m "feat: per-type count-floor legal sets, floor-walk BC, floor-conditioned features"
```

---

### Task 5: Retire the probe from the inference path + docs

**Files:**
- Modify: `src/cloak/probe.py` (docstring only: walk_risk = offline calibration/validation
  and reward-side diagnostics; never on the inference path)
- Modify: `docs/specs/RL/surrogate-ranker-infiller.md` (§3.3-2 tau ceiling → per-type count
  floors; §2 Phase-0 `legal[s]` pseudocode; the Phase-2 decode loop is replaced by
  **grammar-constrained decoding**: the infiller generates inside the node's canonical
  template grammar with closed slot vocabularies, every producible string parses under
  `aset_count(strict=True)`, and the count-≥-floor check is recomputed online per
  instantiation (the infiller may legally pick a different slot than the ranker's default —
  the certificate is deterministic and per-string); loop = sample-within-grammar →
  injectivity → strict count ≥ floor → NLI truthfulness → accept/resample/placeholder.
  Proposer/verifier separation: the infiller never emits or certifies its own count —
  a predicted-count head, if ever added, is a training-time calibration aid only. The
  offline LM count-estimator is scoped to build time: authoring slot vocabularies /
  populating counts for out-of-universe lattice nodes, shootout-calibrated)
- Modify: `research-wiki/training/2026-07-04-RL-ranker-stage1-bandit.md` (Observations:
  correct "optimization-regime null" to reward-support null — cite
  `scripts/spikes/probe_flip_scan.py` (3/106 flippable probes) and the reordered lever
  ladder; link this plan as successor)
- Modify: `docs/research/inference-risk-enforcement.md` (frontmatter `updated:`, note §2.1
  adopted with plan link)

- [ ] **Step 1: Apply the four doc edits** — keep the spec amendment surgical: the tau-mask
  invariant text is replaced by the count-floor invariant (same load-bearing enumeration:
  training env, deployed inference, BC teacher, eval control group), with walk_risk moved to
  the offline-instruments section next to the shootout.
- [ ] **Step 2: Sanity-grep** — `grep -rn "walk_risk" scripts/ src/ | grep -v spikes` and
  confirm every remaining use is build-time (arms artifact builder), offline calibration, or
  a feature/diagnostic — none on a deployment decision path.
- [ ] **Step 3: Commit**

```bash
git add src/cloak/probe.py docs/specs/RL/surrogate-ranker-infiller.md \
        research-wiki/training/2026-07-04-RL-ranker-stage1-bandit.md \
        docs/research/inference-risk-enforcement.md
git commit -m "docs: adopt structural lattice risk — spec amendment, probe retired offline"
```

---

## Out of scope (next plans)

- The floor-randomized retraining sweep itself (a training-experiment record under
  `research-wiki/training/`, spec-then-results, after this plan lands).
- Population-weighted counts and per-surface overrides — pre-registered escalations, only
  on measured eval-attacker hits through the floors.
- The infiller build (grammar-constrained decoding per the Task-5 spec amendment; the
  offline LM count-estimator for out-of-universe nodes, sized by Task 2's fail-closed rate).

## Self-review notes

- Task 2's gate deliberately blocks Tasks 3–5 on user sign-off of measured numbers
  (empirical-honesty rule).
- Type consistency: `aset_count(fill, span_type, original) -> float` is the only new public
  function; `K_FLOORS` the only new constant; both defined in Task 1 and consumed by name in
  Tasks 2–4. `N_FEAT = 7 + 7 + 3 = 17` after Task 4.
- The artifact is edited in place exactly once per content change (idempotent annotator);
  no task re-detects.
