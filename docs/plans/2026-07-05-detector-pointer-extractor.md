---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [extractor, invert, detector-aligned, pointer, edit-tagger, gliner, architecture]
companion: [2026-07-05-extractor-inverse-designs.md, ../research/learned-PII-detection.md]
---

# Detector-pointer extractor: detailed design

The selected long-term `invert()` architecture from the designs survey
(`2026-07-05-extractor-inverse-designs.md`): a learned pointer model with the FT-detector
backbone, operating over typed candidate spans proposed by a detector pass on out_p. It
substitutes only surfaces copied from R (cannot hallucinate), abstains explicitly, inherits
per-user-type tailorability from the detector composition, and runs as one encoder pass
per document — local, zero privacy cost by post-processing immunity.

## Definitions

- **R** — the substitution record from `substitute.py`: entries
  `{action: placeholder|generalize, surface, replacement, type}` (type = 8-type TAB schema;
  injective: one replacement ↔ one surface).
- **out_p** — remote model's answer to doc_p. **out_final** — reconstructed answer.
- **fill / mention** — a generalization's replacement text / its (possibly drifted)
  occurrence in out_p.
- **abstain** — the pointer's explicit "this entry has no mention here" outcome; distinct
  from a low-score miss.

## Components

```
out_p ──► [1 detector pass] ──► typed spans S = {(start,end,type,score)}
   R ──► [2 rule pre-pass]  ──► exact/fuzzy-90 hits resolved; residue entries Q
(out_p tagged with S) + Q ──► [3 pointer model] ──► per-entry: span ∈ S⁺ | ABSTAIN
                              [4 splice executor] ──► out_final + stats
```

### 1. Candidate proposer (detector pass)

Run the span detector over out_p once; keep all spans above the detector's operating
threshold, typed. **Checkpoint deferred** behind the pre-registered gating spike (production
`detect()` vs latest FT-detector on the audit's 75 labeled fills' out_p; select by drifted-
mention recall per type at a fixed precision floor). S⁺ = S augmented with a small dilation:
each detected span also yields boundary variants snapped to word/sentence boundaries ±2
tokens, because detector boundaries on drifted mentions are approximate and the splice must
land exactly.

Per-user types extend here for free: a user-registered type enters the detector's zero-shot
label path, and its R entries then have same-type candidates like any built-in type.

### 2. Rule pre-pass (unchanged code)

`extract.py` placeholder swap-back and exact/fuzzy-90 narrowing run first, verbatim — they
already invert 36% with near-zero false positives and keep the re-gate story clean. Spans
consumed by the pre-pass are removed from S⁺. The pointer sees only the **residue queries
Q**: generalization entries the rules scored absent. (Measured option, later: let the
pointer own all generalizations and delete the fuzzy path — adopt only if the pointer beats
fuzzy-90 on its own hits at equal precision.)

### 3. Pointer model

**Framing.** GLiNER is already a query-against-span matcher: it embeds label phrases and
scores them against span representations. We reuse exactly that machinery with **R entries
as queries instead of label phrases** — which is why the FT-detector checkpoint is the
natural init (both the encoder AND the matching head transfer; a plain-encoder init must
learn the head from scratch).

- **Query encoding**: entry i → text `"{type}: {replacement}"` (the fill text, type-tagged).
- **Span side**: out_p encoded once; candidate representations pooled over each s ∈ S⁺.
- **Score**: `score(i, s) = cos(q_i, h_s) / T` restricted to type-compatible pairs
  (same type, plus a small learned type-confusion allowance — see Risks). A learned null
  vector per type gives `score(i, ABSTAIN)`.
- **Assignment**: greedy one-to-one over all (i, s) pairs by descending score (R is
  injective; each span serves ≤1 entry, overlapping boundary-variants of one detected span
  count as one slot). Accept iff the winner beats ABSTAIN **and** clears margin δ over the
  runner-up compatible span; else ABSTAIN.

**Base model recommendation (ordered):**
1. **FT-detector checkpoint** (GLiNER on `knowledgator/gliner-pii-base-v1.0`,
   DeBERTa-v3-base encoder, ~0.2B) — primary; maximal transfer, one shared backbone for
   detect+extract (memory win on the local GPU).
2. **Stock `gliner-pii-base-v1.0`** — fallback if the gating spike shows fine-tuning hurt
   out-of-distribution (answer-prose) recall.
3. **DeBERTa-v3-small + multi-query extractive-QA head** — ablation arm proving what the
   GLiNER head transfer buys; also the escape hatch if GLiNER's fixed pooling fights the
   boundary-variant scheme.
   Do NOT reach for a generative model here; pointing, not generation, is the design.

### 4. Splice executor

For each accepted (entry, span): replace out_p[span] with entry.surface using the existing
word-boundary splice from `extract.py`'s fuzzy path. Then the standard `ph_residue` check.
Stats gain `gen_pointer` and `gen_abstain` (abstentions reported separately from
rule-absent — they are the pointer's calibrated refusals and feed the gaming guard).

## Pseudocode

```python
def invert(out_p, R, detector, pointer, tau_det, delta):
    text, stats = rule_prepass(out_p, R)              # extract.py verbatim
    Q = [e for e in R if e.action == "generalize" and not stats.hit(e)]
    if not Q:
        return finalize(text, stats)
    S = [s for s in detector(text) if s.score >= tau_det]
    Splus = dilate_boundaries(S, text)                 # word/sentence-snapped variants
    H = encode(pointer, text, Splus)                   # one encoder pass
    q = [embed_query(pointer, f"{e.type}: {e.replacement}") for e in Q]
    pairs = [(score(q[i], H[s]), i, s) for i in range(len(Q))
             for s in compatible(Splus, Q[i].type)]
    used_slots, out = set(), {}
    for sc, i, s in sorted(pairs, reverse=True):
        if i in out or slot(s) in used_slots:
            continue
        if sc <= score_abstain(q[i]) or sc - runner_up(pairs, i, s) < delta:
            continue
        out[i] = s; used_slots.add(slot(s))
    for i, s in sorted(out.items(), key=lambda kv: -kv[1].start):  # right-to-left splice
        text = splice(text, s, Q[i].surface)           # word-boundary rule from extract.py
        stats["gen_pointer"] += 1
    stats["gen_abstain"] = len(Q) - len(out)
    return finalize(text, stats)                       # ph_residue check
```

Training step (per round-trip example):

```python
# example: (out_p, R, gold)  where gold[i] ∈ {char-span in out_p, ABSENT}
S = detector(out_p); Splus = dilate_boundaries(S, out_p)
target[i] = argmax_overlap(Splus, gold[i])   # IoU >= 0.5, else:
#   gold span exists but no candidate overlaps -> DROP from loss, count as
#   detector-ceiling miss (reported metric, not trained noise)
#   gold is ABSENT -> target[i] = ABSTAIN
loss = mean_i CE(scores(q_i, Splus + [ABSTAIN]), target[i]) \
     + lambda_pair * margin_loss(same_type_confusables)      # two dates, two names
```

## Training data

Same seed corpus as the designs doc, aligned to detector candidates:
- **Silver positives** — rule-pre-pass exact/fuzzy-90 hits (abundant, free). Trained
  through the pointer even though inference skips them: they teach the matching head and
  cost nothing.
- **Hard positives** — synthetic paraphrase drift: local pin (gemma E4B) restates
  out_p sentences containing fills; drifted mention location known by construction.
- **Negatives / ABSTAIN** — audit `absent` rows, plus fills whose sentences were deleted
  in restatement.
- **Confusables** — construct docs with ≥2 same-type entries (two dates, two persons) so
  the margin loss has signal; this is the known hardest case.

Split discipline: train clinical → eval lexsum and vice versa (open-label generality);
eval docs disjoint from every data-generation step.

## Selection, eval, success criteria

Training record: **FT-extractor v1** (spec before run). Operating point: pick (τ_det, δ,
abstain calibration) on the dev split at a **precision floor ≥ the rule extractor's**
(a wrong surface asserted to the user is worse than a miss — hard constraint).
- Primary: inversion recall/precision on held-out audit-labeled fills vs rule extractor
  and vs the semantic-window matcher at identical settings; per corpus, never averaged.
- End-to-end: out_final utility at identical upstream settings.
- **Detector-ceiling metric**: fraction of gold mentions with no overlapping candidate —
  the pointer cannot beat this; it prices the next FT-detector iteration (fine-tuning on
  answer-prose is the detector track's job; this doc only pre-registers the interface).
- Gaming guard: exact-vs-pointer recall gap per RL checkpoint; abstain-rate trajectory
  (a collapsing abstain rate under RL = the policy found the extractor's soft spot).

## Ops

- **Re-gate**: shared rule from the designs doc — land before a probe build; anchors,
  probes, scan verdicts, policies invalidate together; budget one probe+scan rebuild.
- **GPU**: one process rule; pointer training is a small-encoder FT, hours not days.
- **Latency**: detector pass + one pointer pass per doc, only when the rule pre-pass
  leaves residue.

## Risks

- **Detector ceiling on answer prose** (the gating spike's question) — if drifted-mention
  recall is low, the pointer is capped regardless of quality. Mitigation is a detector-track
  fine-tune on out_p-style text, not pointer-side hacks. Report the ceiling either way.
- **Type disagreement**: detector types a drifted mention differently than the original
  span (DEM vs MISC is the likely confusion). The type-compatibility mask therefore allows
  a learned small cross-type allowance rather than a hard equality; measure per-type
  confusion in the gating spike before fixing the mask.
- **Boundary misalignment**: dilation variants cover ±2 tokens; beyond that a hit splices
  wrong text. Count boundary-IoU in eval; if it is the dominant error, add a start/end
  refinement head (deferred — YAGNI until measured).
- **Silver-label bias**: pointer learns the fuzzy path's blind spots; the synthetic hard
  positives exist precisely to break this. Ablate: train without them, report the gap.
- **Fused mentions** (one span realizing two fills): structurally unpointable — counted in
  the audit; if material, that share routes to the denoise seq2seq contingency, not here.
