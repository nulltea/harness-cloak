---
type: reference
status: current
created: 2026-07-10
updated: 2026-07-10
tags: [detector, noise-gate, span-filtering, entity-linking, weak-supervision, calibration,
       lattice-profiles, mining]
companion: [docs/issues/2026-07-10-detector-junk-and-noise-gate-limits.md,
            docs/specs/lattice-entry-dedup-and-span-resolution.md,
            docs/specs/substitutor-profile-match-retrieve-verify.md]
---

# Detector noise — semantic gate (link, retype, margin-drop, aggregate)

## Purpose

The current negative filter (`is_noise_span`, `src/cloak/detect.py`) is a fail-open deny-list:
it removes only enumerated noise families and generalizes to nothing
([issue](../issues/2026-07-10-detector-junk-and-noise-gate-limits.md)). The detector's
over-generation therefore reaches the producer (wasted work), the substitutor (non-sensitive
spans get ranked/generalized/placeholded), and the reward (polluted credit). This spec replaces
the deny-list *ceiling* — not the deny-list itself — with a semantically generalizing gate built
mostly from machinery the entry-dedup work already provides
([spec](lattice-entry-dedup-and-span-resolution.md)).

Three detector error families, named throughout:

- **F1 non-entity noise** — arbitrary common nouns and fragments.
- **F2 out-of-scope real entities** — real-world things of families the schema does not treat
  as sensitive (anatomy, devices, lab artifacts, foods…).
- **F3 wrong-type routing** — real in-scope entities put in the wrong type bucket.

## Reuse-first mandate

Where a principled, maintained implementation exists and fits, it is used rather than
reimplemented:

- **Link-or-drop / NIL rejection** (the BLINK/ReFinED pattern,
  [arXiv 1911.03814](https://arxiv.org/abs/1911.03814),
  [arXiv 2207.04108](https://arxiv.org/abs/2207.04108)) — implemented by reusing the repo's own
  retrieve-then-verify matcher (`cloak.profile_match.match_spans_batch`) and `lookup_entry`
  against lattice profiles; no new linker.
- **Weak-supervision aggregation** — [skweak](https://arxiv.org/abs/2104.09683) (pip library,
  span-specialized Snorkel successor): labeling functions + HMM label model produce one
  keep/drop posterior per span. Adopted for the **miner** path if the adoption spike passes
  (below); the margin rule is the fallback, not a parallel implementation.
- **Embeddings** — the existing embindex model/infra (`cloak.profile_match`, bge-small); no new
  embedding stack.
- **Calibration** — split-conformal-style threshold selection is ~20 lines on top of the
  existing calibration-artifact pattern (`scripts/calibrate_entity_merge_gate.py` skeleton); no
  calibration library.

## Definitions

- **Positive anchors** — embedded canonical+alias surfaces of a runtime type from the profile
  embindex; the type's known-good exemplars.
- **Negative anchors** — embedded surfaces of known-noise families; seeded from the existing
  `_NOISE_*` deny-list sets plus junk exemplars from the measured re-mine
  (`results/mined_lattice_profile_spans_large.jsonl`).
- **Link verdict** — the matcher's output for a span: exact hit, certified semantic hit, or
  abstain (`MatchResult | None`).
- **Anchor margin** — `sim(span, nearest negative anchor) − sim(span, nearest positive anchor
  of the span's type)`; positive margin = looks more like noise than like any known entity.
- **Labeling function (LF)** — a weak, possibly-abstaining vote on keep/drop (skweak sense).
- **Operating point** — a frozen threshold set for one consumer; never tuned per run.

## Gate architecture

One shared decision core, `cloak.span_gate` (new module), consumed at three call sites. Layers
in order; **fail-open remains the terminal default** — the gate only drops what it can
positively justify:

1. **Link-keep (F1/F2 reject by absence, keep by presence).** Span resolves via
   `lookup_entry`/matcher to a profile entry of its detected type → KEEP, carrying entry
   identity. Shared verbatim with the dedup span-resolution path — in the QA build this call
   already exists and costs nothing extra.
2. **Link-retype (F3).** Span exact-resolves to an entry of a *different* profile type → emit
   RETYPE(new_type) instead of drop. Linking is the only surveyed technique that natively fixes
   routing rather than discarding the span.
3. **Anchor-margin drop (F1/F2).** Embed the surface (same model, cached); DROP iff
   `nearest-positive similarity < FLOOR` **and** `anchor margin ≥ MARGIN`, both frozen at the
   consumer's operating point. Generalizes beyond enumeration: an unlisted member of a noise
   family embeds near its enumerated siblings; an arbitrary noun sits far from all positive
   anchors.
4. **Deny-list short-circuit.** `is_noise_span` stays as the cheap confirmed-noise exact check
   (it is precise; its ceiling was coverage). Its sets double as negative-anchor seeds.

### Per-consumer operating points (the asymmetry is the design)

Every pipeline that consumes detected spans runs the **full gate** — miner, RL
training/reward, and production inference alike. What differs is only the frozen operating
point:

| Consumer | Layers | Operating point | Rationale |
|---|---|---|---|
| Miner (`build_mined_lattice_profiles`) | 1–4 (+ aggregation) | precision-leaning | junk rows poison the profile; a missed real entity costs one lost row |
| RL training env + reward (span sets feeding ranker decisions and per-span credit) | 1–4 (+ aggregation) | **identical to production** | junk spans split reward credit and force the substitutor to rank/placehold non-sensitive spans |
| Production inference (`negative_filter` path in `detect.py`, feeding the substitutor) | 1–4 (+ aggregation) | calibrated at ≈zero measured false-drop rate | a false drop is a privacy leak, so the drop bar is strict — but layer 3 stays *active*: never dropping anything is the current failure, not safety |
| QA build (`_detect_docs`) | 1 (via dedup Task B) | unchanged | matcher abstain already excludes non-probe spans |

**Train/deploy consistency (hard requirement):** the RL env/reward and production inference
use the *same gate version and the same operating point* — otherwise the policy trains on a
different span distribution than deployment serves. The gate's artifact hashes (anchor
indexes, aggregation model, thresholds) are recorded in RL run configs; a mismatch fails the
run, not silently degrades.

### Aggregation (skweak) — fitted offline, applied everywhere

Layer outputs become LFs — deny-list hit, link verdict, anchor margin bucket, detector
confidence bucket, label-stability vote (span survives detection under the type's alternate
label phrasings — the agreement-filter idea from
[CrossWeigh](https://arxiv.org/abs/1909.01441)) — and skweak's HMM label model aggregates them
into a keep/drop posterior; accept thresholds on the posterior are the per-consumer operating
points.

Fitting is corpus-level and happens **once, offline** (on the mined span corpus); the fitted
label model is then a frozen, versioned artifact applied at *every* call site — miner, RL, and
production — like the embindex. Per-span application is cheap (the LF signals are computed
anyway; aggregation is a lookup-scale computation).

**Adoption spike (gates skweak in or out):** install skweak, fit on the measured re-mine span
set, compare keep/drop quality against the plain layered gate (fixed layer precedence) on the
labeled eval set. Adopt only if it beats the layered gate on drop-precision at equal or better
drop-recall; otherwise record the numbers and ship the layered gate alone at all call sites.
Either way, all consumers run the same decision core — the spike chooses the core, not a
per-pipeline split.

### LLM triage band (miner only, optional)

Spans the gate neither keeps (no link) nor drops (margin inconclusive) may go to the producer's
existing teacher for a typed yes/no/retype verdict (the
[GPT-NER self-verification](https://arxiv.org/abs/2304.10428) pattern), cached under the
teacher-harvest convention, results to a review JSON. Off by default; enabled per run with the
usual paid-call approval.

## Calibration & evaluation

- **Labeled eval set:** the measured junk/real split of the large re-mine (the issue documents
  ~200 surviving entries with a known noise fraction) as drop-positives; profile surfaces and
  their certified variants as keep-positives. Frozen file under `data/`, versioned.
- `FLOOR`/`MARGIN` (per operating point) and the skweak posterior thresholds are chosen
  **once** on this set and frozen — the shared RL/production point at ≈zero measured
  false-drop rate (report the achieved rate), the miner point precision-first. Published as a
  calibration artifact JSON (sweep + chosen values per operating point), same shape as the
  entity-merge gate eval.
- **Empirical honesty:** if layer 3 cannot reach the RL/production bar at any useful recall,
  those paths ship with layers 1/2/4 only and the sweep is the finding. No per-run threshold
  tuning, ever.
- Detector scores are used only as bucketed LF input, never as a calibrated probability
  (GLiNER-class scores are not assumed calibrated;
  cf. [arXiv 1706.04599](https://arxiv.org/abs/1706.04599)).

## Shared machinery with the entry-dedup work (implement once, there)

| Piece | Provided by |
|---|---|
| `lookup_entry` canonical identity | dedup Task 1 (committed) |
| Matcher link verdict + NLI certification | `cloak.profile_match` (exists) |
| Positive anchors | embindex artifact (exists; rebuilt by dedup reprocess) |
| Calibration-artifact pattern | dedup Task B calibration script |
| Review-file convention | dedup Task A entity-merge report |

Net-new work in this spec: negative-anchor index, `cloak.span_gate` layers 2–3, the two
consumer wirings, the labeled eval set + calibration script, the skweak adoption spike, the
optional triage hook. Files touched (`detect.py`, `build_mined_lattice_profiles.py`, new
modules) are disjoint from the dedup tasks' remaining files, so the two tracks can be
implemented concurrently.

## Non-goals

- No detector retraining or threshold change at the miner (orthogonal lever, measured
  separately).
- No remote calls in any runtime path; the triage band is miner-only, cached, gated on paid
  approval.
- No deny-list expansion campaigns — enumerations grow only as negative-anchor seeds.
- No replacement of the substitutor's certifier; the gate decides span admission, not
  replacement legality.
- No claim of improved end-to-end privacy/utility without the standard attacker-measured
  comparison.
