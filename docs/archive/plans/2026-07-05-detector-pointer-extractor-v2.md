---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [extractor, invert, detector-aligned, pointer, verification, nli, lexical-anchor]
companion: [2026-07-05-detector-pointer-extractor.md, ../../research-wiki/experiments/extractor-pointer-by-type.md]
supersedes: 2026-07-05-detector-pointer-extractor.md
---

# Detector-pointer extractor v2: dual-proposer + verification gate

v1 (`2026-07-05-detector-pointer-extractor.md`) was implemented as a zero-training
detector-pointer and measured in `research-wiki/experiments/extractor-pointer-by-type.md`.
Result: it **ties** the semantic-window fallback (28/60 vs 27/60 rule) and moves only
DATETIME (10%→20%, one span). It does not touch the two largest error buckets, DEM
(18/32) and MISC (3–4/12). This doc redesigns around the three measured causes of that
ceiling.

## What v1 got wrong (measured, not speculative)

1. **Polluted denominator.** `present_in_out_p` = exact+fuzzy90+**band60_90**, but band60_90
   is three different populations (from `results/extractor_miss_audit.json`):
   - *real generic echo* — `lasix`→`a drug`→"a new drug" (cos 0.46), `polycystic
     kidneys`→`an organ`→"some organ" (cos 0.46). Recoverable.
   - *templating artifact* — `arthritis`→`a disease`→"[Disease]" (cos 0.45). The remote
     model refused; there is no original mention to recover. Not recoverable.
   - *alignment noise* — `dragon`→`a mythical being` fuzzy-hitting "fiscal year-end data"
     (cos 0.02). Not an echo at all.
   Rates over this denominator are not a real ceiling. **v2 requires a labeled denominator.**
2. **Detector ceiling on generic fills.** `_pointer_assign` scores R entries only against
   *detected typed spans*. GLiNER does not emit "a drug"/"an organ"/"something" as entities,
   so `compatible` is empty for the DEM/MISC bulk and the pointer cannot fire. The
   generalization mechanism produces mostly generic hypernyms — exactly the case the
   detector cannot propose.
3. **Embedding score is noise on short generic phrases.** `_pointer_scores` cos sits at
   0.01–0.46 for real matches, indistinguishable from spurious; `POINTER_MIN=0.70` then
   only fires on near-identical text the rule path already caught.

## Core reframe

Extraction has **two echo modes**, and they need different candidate proposers and one
shared acceptance test:

- **Generic hypernym echo** (the DEM/MISC bulk): the fill's *head noun survives* in out_p,
  possibly with a swapped determiner/modifier (`a drug`→"a new drug", `an organ`→"some
  organ"). Proposer = **lexical head-anchor**, not the detector.
- **Typed reparaphrase** (DATETIME/LOC/QUANTITY/PERSON): the model rewords into a
  differently-worded *typed* mention the detector can see (`last weekend`→"at some point").
  Proposer = **detector spans** (v1's mechanism, kept).

Recall (proposing a candidate) is now easy in both modes. The hard, high-value problem is
**precision**: accepting a low-similarity true echo without splicing a wrong surface into
out_final (asserting a wrong fact to the user is worse than a miss). v2's central new
component is a **verification gate**, not a better recall scorer.

## Components

```
out_p ──► [rule pre-pass]  ──► exact/fuzzy-90 resolved; residue Q
out_p ──► [proposer A: lexical head-anchor]  ─┐
out_p ──► [proposer B: detector typed spans] ─┴► candidate windows C (typed, deduped)
(Q, C) ──► [verification gate: entailment + hyponymy]  ──► accept | reject
        ──► [splice executor] ──► out_final + stats
```

### Proposer A — lexical head-anchor (new; owns generic fills)

For each residue fill, extract its content head (drop leading determiner/quantifier:
`a`, `an`, `some`, `the`, `a new`, `between … and …`). Locate occurrences of the head in
out_p by fuzzy partial-ratio ≥ 60 (the audit's own band lower bound) **restricted to the
head noun**, not the whole phrase — this is what separates "a new drug" (head *drug*
matches) from alignment noise ("fiscal year-end", head *drug* absent). Window = the
determiner-phrase around the matched head, snapped to noun-phrase boundaries.

### Proposer B — detector typed spans (v1, kept)

`_detect_spans` + `_dilate_detector_spans` unchanged. Owns the typed-reparaphrase mode
where no lexical head survives. Candidates from A and B are unioned and de-duplicated by
overlapping char span.

### Verification gate (new; the precision mechanism)

Replaces the bare `cos ≥ POINTER_MIN` threshold. For a candidate (fill f, window w,
original surface s), accept the splice iff **both**:
1. **Echo check** — `w` is a mention of `f`: NLI entailment `w ⊨ f` OR high lexical-head
   overlap (proposer-A candidates pass this by construction; proposer-B candidates need the
   NLI check because the words differ).
2. **Hyponymy check** — `s` is-a `f` (arthritis ⊑ a disease, lasix ⊑ a drug): confirms the
   generalization direction so we only ever *specialize* a genuine hypernym mention.
   Cheap realization: reuse the substitution lattice that produced `f` from `s`
   (`substitute.py` already knows `s→f` is a generalization edge — it is in R's provenance),
   so hyponymy is **free from R** for the common case; fall back to the NLI model only when
   provenance is absent.

Model recommendation: a small cross-encoder NLI checkpoint
(`cross-encoder/nli-deberta-v3-small`, ~0.14B, local, one forward per candidate) for the
echo check. No generation. Operating point: threshold chosen on the labeled dev denominator
at a **precision floor ≥ the rule extractor** (hard constraint). Report the recall it buys
per echo mode.

### Splice executor + stats (v1, extended)

Existing word-boundary splice. New stat keys `gen_lexical`, `gen_detector`, `gen_verify_reject`
(candidates a proposer found but the gate refused — the precision-vs-recall knob's audit
trail, and part of the RL gaming guard).

## Pseudocode (the changed core)

```python
def invert_v2(out_p, R, nli, detector):
    text, stats, Q = rule_prepass(out_p, R)          # extract.py verbatim
    cands = lexical_head_anchor(text, Q) + detector_spans(text, detector)   # A ∪ B
    cands = dedupe_overlapping(cands)
    accepted = {}
    for i, entry in enumerate(Q):
        for w in compatible_windows(cands, entry):    # type OR head-noun match
            echo = head_overlap(entry.fill, w) or nli_entails(w.text, entry.fill)
            hypo = lattice_is_a(entry.surface, entry.fill, entry.provenance) \
                   or nli_entails(entry.surface, entry.fill)     # s ⊑ f
            if echo and hypo and w.slot not in used:
                accepted[i] = w; used.add(w.slot); break
    for i, w in sorted(accepted.items(), key=lambda kv: -kv[1].start):
        text = splice(text, w, Q[i].surface)
    stats["gen_verify_reject"] = candidates_seen - len(accepted)
    return finalize(text, stats)
```

## Evaluation (fixes the v1 metric)

1. **Label the denominator first.** Extend `extractor_miss_audit.py`: for each band60_90
   fill, tag {real-echo, templating-artifact, alignment-noise} — heuristic seed (bracket
   `[\w+]` → artifact; head-noun-present → real; else noise) then a one-pass manual
   correction on the ~33 band fills (small, one-time). Real-recoverable becomes the true
   denominator. Report v1 and v2 against **this** number, not the polluted 60.
2. Per-echo-mode recall/precision vs rule and vs semantic-window at identical settings.
3. `gen_verify_reject` rate = the safety margin; end-to-end out_final utility.
4. **Detector-ceiling-by-type on answer prose** — the honest question the experiment
   flagged: run the detector over out_p and report, per type, the fraction of real-echo
   fills for which a typed candidate exists. This prices proposer B and confirms proposer
   A is mandatory for DEM/MISC (expected from the v1 empty-`compatible` finding).

## Learned upgrade (unchanged plan, better seeded)

The learned FT-extractor from v1 still applies, but v2 gives it a **clean label source**:
proposer A∪B candidates + verification-gate decisions are exactly (window, fill, surface,
accept?) triples. Train the pointer/verifier jointly on these once the zero-training v2
establishes the recall ceiling and the labeled denominator. Backbone/init recommendation
from v1 stands (FT-detector checkpoint for proposer B's typed matching).

## Risks

- **Hyponymy from R provenance may be stale** if substitution didn't record the lattice
  edge; the NLI fallback covers it but is the slower path — measure how often provenance is
  present before assuming the fast path dominates.
- **Head extraction on multi-word fills** ("between 60,000 and 240,000 dollars") has no
  single head; route numeric/quantity fills to proposer B (detector QUANTITY) instead.
- **NLI on clinical/legal prose** may be out-of-domain; the labeled dev set is the gate,
  and a low ceiling there is a finding (prices a domain-tuned verifier), not a reason to
  loosen the precision floor.
- **Gaming guard** (shared): `gen_verify_reject` trajectory + exact-vs-(lexical+detector)
  gap per RL checkpoint; a collapsing reject rate = the policy learned the gate's soft spot.
```
