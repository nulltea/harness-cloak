---
type: experiment
node_id: exp:context-injection-surface-ablation
title: "Context-injection surface for π_rank: CLS vs attention-pool vs biencoder"
idea_id: "idea:context-injection-surface"
verdict: "INVALID as-run — Level-1 baseline (others at floor-walk) sits at the utility-collapse point; flat rate + producer null are artifacts. Re-run with ceiling/realistic others-baseline before concluding."
confidence: "n/a — operating-point flaw found (all 36 never-recovered facts recover at the ceiling)"
date: 2026-07-07
hardware: "host .venv iGPU (gfx1151): frozen encoders + head fit; served reward (:8060) for labels"
duration: ""
provenance: "scripts/spikes/context_ablation_labels.py + scripts/spikes/ablate_context_producer.py; companion spec docs/specs/RL/roundtrip-ranker-infiller.md (§ π_rank — features)"
added: 2026-07-06T00:00:00Z
tags: ["ranker", "context", "ablation", "encoder", "attention-pool", "biencoder", "rl", "offline-judge"]
---

# Context-injection surface for π_rank: CLS vs attention-pool vs biencoder

**verdict:** `pending` · tests `idea:context-injection-surface`

## Objective & hypothesis

π_rank must be **context-aware**: under a utility-only reward with privacy enforced by the
`aset` floor gate, the policy's job is to pick, among floor-legal actions, the fill that best
preserves downstream utility — which depends on the span's surrounding text. The **only**
context channel in the policy is the encoder's context vector, concatenated to each action's
feature row and consumed by the shared MLP head (`docs/specs/RL/roundtrip-ranker-infiller.md`
§ π_rank). walk_risk and the corpus one-hot are being removed (train/deploy skew + orphaned
under utility-only reward), so **the encoder is the sole context source** — this ablation asks
which producer fills that slot best.

**H1** — filling the context slot beats the context-free floor (else context representation is
not the bottleneck). **H2** — a better producer than raw ModernBERT-CLS (learned attention
pooling, or a contrastive biencoder) lowers realized action-regret at matched everything-else.

## Arms (one variable: the context producer; A0 as the necessary reference)

The MLP head, objective, optimizer, epochs, seed, split, and label set are **identical** across
arms. The context vector is concatenated to the per-action feature row (injection point fixed).

| arm | producer → context vector | frozen? | tests |
|-----|---------------------------|---------|-------|
| **A0** | none (feature-only head) | — | reference: is the slot worth filling? not a deploy candidate (context is non-negotiable) |
| **a. CLS** | frozen ModernBERT-base, `[CLS]` | yes (precomputed) | the current design; control |
| **b. attn-pool** | frozen ModernBERT token states + small trained attention query | encoder frozen, pool trains | does learned pooling of pretrained features beat CLS? |
| **c. biencoder** | frozen contrastive embedder (bge-small / gte-small / `modernbert-embed`), mean-pool | yes (precomputed) | does a better pretrained rep beat ModernBERT? |

Not tested (with rationale): standalone from-scratch context net (too little data, ~267 docs);
encoder fine-tune / LoRA (data-hungry, defeats frozen+precompute, couples to reward);
walk_risk-scalar and "both" arms (walk_risk removed by decision).

## Labels (built once, arm-independent — `context_ablation_labels.py`)

The round-trip reward is deterministic + disk-cached, so the per-span optimum is *knowable*, not
estimated. For each trainable span: **hold all other spans at the floor-walk baseline, sweep
this span's legal actions through `R_rt`**, record realized recall per action. Signal =
**own-probe recall** (probes whose surface is this span's fact; sharper than whole-doc recall
because only this span's fills move). `oracle(span) = argmax_a own_recall`. This is the support
scan (`roundtrip_support_scan.py`) generalized from sampled flip-signs to the full per-action
profile. Each label row: `{doc_id, corpus, surface, span_context_text, action_idx,
action_feats (context-free part), own_recall}` + the floor-walk baseline action.

## Judge (offline, no RL) — two levels

**Level 1 — per-span action-regret (primary screen).** Fit each arm's head (supervised, shared
objective) to the reward profiles; on **doc-held-out** spans:
`regret(span) = R_rt(oracle) − R_rt(head_argmax)` (recall units). Also top-1 accuracy and
NDCG/Spearman of predicted-vs-true per-span action ordering.

**Level 2 — doc-level greedy realized reward (cross-span confirmation).** Let each fitted head
choose every span of a held-out doc greedily; compute the **actual `R_rt`** of that full joint
assignment (a few cached round trips). Captures the cross-span coupling Level 1 omits; the
deployment-like number.

**References:** oracle = 0 (upper bound); **floor-walk** (current deployed min-aset policy — an
arm is worthless unless it beats this); random-legal (sanity). Report each arm on the ladder
`oracle ≤ best ≤ … ≤ floor-walk ≤ random`.

## Selection, controls, success criteria

- **Split by doc** (reuse the probe train/held split) — same-doc spans are correlated.
- **Paired bootstrap over held-out docs** — CI on the *difference* `(arm_i − arm_j)` on the same
  docs; "beats" only if the paired CI excludes 0.
- **Matched capacity** — identical head arch / LR / epochs / init-seed set (~5 seeds; producer
  embeddings deterministic). **No per-arm tuning** (the per-model-knob trap; CLAUDE.md).
- **Per-corpus breakdown, never averaged** — context value differs (structured clinical/legal vs
  flat email).
- **Decision:** (1) if no arm beats floor-walk per corpus → representation is not the bottleneck,
  escalate (reward proxy? context genuinely weak). (2) winner = lowest regret whose edge over the
  next-cheaper producer has a CI excluding 0 **on the Level-1 screen AND confirmed by Level 2**
  (a winner must not lose the doc-level joint comparison). (3) tie-break on cost/deployability
  (smaller producer wins). No winner is declared on Level 1 alone. Winner is carried into the
  pilot for the both-ways confirmation.
- **Flat spans** (all legal actions give identical own-recall) carry no signal: primary
  regret/top-1/NDCG are reported on **non-flat** held-out spans, with the flat-span rate as a
  separate failure-mode statistic (a high rate means the reward can't separate this span's
  actions — a finding about the reward/probe, not the producer).

## Limitations (empirical honesty)

- **Necessary-condition screen, not the RL outcome.** Level 1 is per-span marginal (others at
  floor-walk); full RL optimizes jointly. Level 2 recovers the joint choice but still uses a
  supervised-fit head, not RL exploration.
- **Screening proxy, NOT a formal upper bound.** The RL reward `R_rt` is doc-level and
  **non-additive** across spans, so dense per-span oracle labels do not upper-bound RL — they are
  a *necessary-condition screen*: an arm that ties floor-walk on the screen is a weak candidate;
  a winner is promising, not proven. Confirm cross-span with Level 2 (below) + the interaction
  audit before declaring a winner, then in-pilot.
- **Interaction audit** (guards the additivity gap): on held-out docs compare the summed
  per-span own-recall gains against the realized joint-assignment `R_rt` (Level 2); a large gap
  means per-span labels mislead and the screen's ranking must be read with caution.
- **Inherits the reward's validity** — if fact-recall is a weak utility proxy (placeholder-gaming
  issue), all arms inherit it; orthogonal to the producer comparison.
- **Interaction-audit is partial** — `summed_span_gain` skips a Level-2 greedy pick whose own-recall
  was never measured (an action not forced-reachable from the floor-walk baseline during labeling).
  The *realized* joint gain (from the actual round trip) is unaffected; only the additivity-gap
  audit is partial when such picks occur.

## Artifacts (paths)

- Labels: `results/context_ablation_labels.json` (gitignored)
- Results: `results/context_ablation.json` (per-arm, per-corpus regret/top-1/NDCG + CIs)
- Code: `scripts/spikes/context_ablation_labels.py`, `scripts/spikes/ablate_context_producer.py`
- Companion spec: `docs/specs/RL/roundtrip-ranker-infiller.md`

## Results (n-docs 60 offline probe, 2026-07-07)

**Scope.** Label build over 60 docs/corpus → 167 probe-bearing spans, 277 round trips, **87 spans**
(≥2 legal actions) kept across 67 docs. Corpora present: clinical 41, lexsum 36, wikibio 10;
**enron/aeslc contributed no probe-bearing decision spans** (coverage gap — those tasks' facts
aren't restated as separable decision spans here). Level-1 held-out: 22 spans / 12 docs, **9
non-flat**. This is a validation-scope probe, not a powered study — the pipeline ran clean live
end-to-end (label build + Level 1); the model-free `--selfcheck` passes.

> **[SUPERSEDED — see the corrected follow-up below: the flat rate is a floor-walk-baseline
> artifact, not intrinsic. All 36 never-recovered facts recover at the ceiling.]**

**Finding 1 — the reward separates actions on only ~⅓ of probe spans (robust).** 64% flat
(56/87; 65% at n-docs 20, so not an artifact). Decomposition of the flat set by their constant
own-recall:
- **36/87 (41%) never recovered** (own-recall 0.0 for *every* action) — the reader misses the
  fact even at the most specific legal fill;
- 17/87 (20%) always recovered (1.0 for every action) — no decision to make;
- 31/87 (36%) **action-dependent** (mean action-separation **0.79**) — the only spans where the
  fill choice changes recovery;
- 3 mid-flat.
This *reconciles* the RL smoke's `r_heldout ≈ 0.33–0.42` rather than contradicting it: that mean
recall = ~20% ones + ~41% zeros + ~36% partial. The smoke's learning (ph 1.0→0.28, gradient each
epoch) rides on the 36% action-dependent facts. Same `roundtrip_batch`/`fact_f1s`/`canon`
machinery as the RL reward — the zeros here are the same facts dragging the smoke's mean below 1.0.

> **[SUPERSEDED — measured at the collapsed floor-walk operating point; not trustworthy. See
> corrected follow-up.]**

**Finding 2 — no context producer beats floor-walk or context-free, at this scope.** On the 9
non-flat held-out spans (5-seed-averaged, primary metric):

| arm | non-flat regret ↓ | top-1 | Δregret vs floor-walk (ALL) | trainable params |
|-----|-------------------|-------|-----------------------------|------------------|
| attn | 0.387 | 0.467 | −0.010, CI [−0.17, 0.16] | 768 |
| cls | 0.427 | 0.422 | −0.008, CI [−0.41, 0.23] | 0 |
| A0 (no context) | 0.453 | 0.444 | +0.043, CI [−0.07, 0.22] | 0 |
| biencoder | 0.491 | 0.289 | +0.070, CI [−0.18, 0.27] | 0 |

Every paired CI — vs floor-walk **and** vs A0 — includes 0: no producer separates from the rule
baseline or from a context-free head. The n-docs-20 directional hint (A0 worst / cls best) did
**not** replicate. n=9 non-flat held is underpowered; per-corpus CIs are wide or degenerate.

**Verdict (producer question): pre-registered null — representation is not the bottleneck here.**
Per the decision rule, no arm beats floor-walk → escalate to the reward, not the encoder. The
leverage is Finding 1's 41% never-recovered facts (a fact missed even when kept specific is a
reader-miss / probe-quality problem — 0 reward, no gradient — not a privacy↔utility tradeoff).
Diagnosing reader-miss vs probe-quality on those 36 spans is the immediate follow-up (done below).

### Follow-up diagnosis — CORRECTED: the never-recovered facts are a BASELINE ARTIFACT, not a task defect (`scripts/spikes/never_recovered_audit.py`)

An initial read ("30/36 fact absent from `out_final` → the task emits templates") was **wrong** —
it only inspected the floor-walk output. Checking the **ceiling** `out_hi = Remote(task(doc_orig))`,
R=[]: **all 36/36 never-recovered-at-floor-walk facts ARE recovered at the ceiling** (consistent
with probe validation, `build_probes.py:28-30`: a probe is kept only if `ceiling_f1 ≥ 0.5` AND
`floor_f1 < 0.5`). The model produces every one of these facts on `doc_orig`.

**Root cause = the Level-1 marginal-label baseline, not the task.** The labels hold every *other*
span at **floor-walk** (≈ the all-placeholder FLOOR anchor) — exactly where validation established
these facts are lost. At that near-maximal anonymization gemma collapses to a bracketed template
(`[Insert Date]`, `[PERSON_2]`…) and the fact vanishes regardless of the swept span's own fill →
`own_recall = 0` for every action → "flat". The templating is gemma's response to a
maximally-anonymized `doc_p`, not unconditional task behavior.

**This invalidates two conclusions above — corrected:**
- **Finding 1 (64% flat) is largely a baseline artifact**, NOT the reward's intrinsic
  action-separation. These facts are validated *action-sensitive* by construction; they were
  measured at the utility-collapse operating point.
- **Finding 2 (producer null) is measured at a broken operating point** and is not trustworthy.

**Methodology fix (the actual next step):** change the marginal baseline so a swept span's action
has room to matter — hold other spans at the **ceiling** (`doc_orig`, R=[]) or a realistic
mid-anonymization / the policy's own greedy choices, rather than floor-walk — or rely on the joint
Level 2. Same machinery, cheap re-run. Only after that is the producer comparison meaningful.
(The floor-walk-everywhere baseline was inherited from the support scan, which only needs a few
movers to PASS and so never surfaced that most facts sit at the collapse point.)

**Not run (deliberately):** Level 2 (joint confirmation) — moot given the Level-1 null; full-corpus
scope — would reconfirm the flat rate at cost without changing the producer conclusion.

## Connections
_Edges in `graph/edges.jsonl`; summary for humans:_ tests the π_rank context-injection design in
the round-trip ranker spec; gates the walk_risk + corpus feature removal (both already decided);
feeds the RL pilot (`research-wiki/training/2026-07-06-RL-ranker-v3-roundtrip-pilot.md`).
