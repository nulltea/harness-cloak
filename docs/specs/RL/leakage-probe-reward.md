---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [rl, reward-design, privacy, leakage-probe, re-identification, attacker, pareto,
       operating-point, spec]
companion: [docs/specs/RL/roundtrip-ranker-infiller.md,
            docs/specs/RL/training-task-env.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
---

# Leakage-probe reward — reintroducing privacy pressure into round-trip RL (design options)

**Status: design options, NOT implemented.** This spec pins the candidate designs and their
analysis so the choice can be made deliberately later. Nothing here is wired.

## Context — why privacy pressure must move into the reward

Decision 2026-07-12: the per-type legality floors (`K_FLOORS`) and the hand-written
`_APPROVED_FINE_COUNTS` are removed from the runtime path. Rationale: the floors made
keep-original (aset 1) illegal everywhere and pruned genuine lattice levels off junk counts
(the canonical artifact prices many levels at fail-closed 1.0 or the legacy 1000.0 default),
they are context/task-blind, and the floor-randomized training protocol conditioned the policy
on a preference distribution no real user expresses. KEEP is now a legal action; the ranker
must learn to weigh low-anonymity actions rather than have them masked away.

Consequence: the training reward is currently **utility-only**, and KEEP is the per-span
utility optimum by construction (an exact answer entails every coarser rung in
`entail_score`'s acceptance set, so KEEP earns full echo *and* semantic credit). Without an
opposing term the trained policy is expected to drift toward KEEP-everything on
lattice-bearing types. PERSON/CODE stay placeholder-by-rule (outside the learned loop), so
direct identifiers cannot leak via drift — the exposure is quasi-identifiers.

The eval-side attacker (Phase 5) remains the only *measure* of privacy (empirical-honesty
rule). This spec is about the *training-side pressure* that makes the measured Pareto
non-degenerate.

## Definitions

- **Leakage probe** — a reward component that estimates, per rollout, how much sensitive
  content `doc_p` exposes; the privacy analogue of the utility probes.
- **Span-recovery attack** — an attacker model recovers the original span text given `doc_p`
  context around a replacement (inference/reconstruction attack). Crude: recovering a value is
  not identifying a person.
- **Profile-matching attack (re-identification)** — the attacker receives `doc_p` (or a
  derivative) plus a candidate set of person profiles (one true, k−1 distractors) and must
  pick the matching profile. Operationalizes k-anonymity/linkage — the project's headline
  privacy notion.
- **Distractor set / hardness** — the k−1 non-matching profiles; how similar they are to the
  true profile controls attack difficulty and is itself a pinned design choice.
- **λ (privacy weight)** — the scalarization weight between utility and the privacy term.
  Declared operating-point knob (sweeps trace the Pareto curve); never a per-model
  calibration.
- **Chance-normalized accuracy** — attacker accuracy minus 1/k, floored at 0; the graded
  re-identification penalty.
- Everything else (R_rt, out_p/out_final, ExIt/RLOO, single-flight, reader_refresh) as in the
  [round-trip RL spec](roundtrip-ranker-infiller.md).

## Design constraints (all options inherit these)

1. **Channel: probe `doc_p`, not `out_p`.** Provider exposure is what the remote model
   *receives*; `out_p` is a post-processing of `doc_p` (data-processing inequality), so its
   leakage is bounded by `doc_p`'s and probing it separately buys nothing for the training
   signal. (`out_final` leak-through stays an eval-side measurement of the user-facing
   channel.)
2. **Determinism pins.** Any attacker in the reward loop joins the pinned-components table:
   temp-0, non-thinking where possible, content-addressed cache, **single-flight** if served
   on llama.cpp, `refresh` support for winner re-verification. A sampled attacker breaks the
   "R_rt deterministic given doc_p" invariant exactly the way the gen stage did.
3. **Frozen per cycle, never co-trained.** The attacker is frozen for a whole
   gate → train → eval cycle (re-gate on change). Adversarial co-training is out of scope:
   unstable, and it turns the realized privacy level into a moving target mid-run.
4. **Anti-Goodhart pricing.** A policy trained against a fixed attacker learns fills that fool
   *that* attacker. This is priced, not prevented: Phase-5 runs the strongest (frontier)
   attacker, and the gap between train-attacker and eval-attacker success is reported as a
   finding — never calibrated away.
5. **Family separation.** The training attacker must not share a family with the reward/task
   model or the probe teacher (same rule as the grader/teacher separation).
6. **Cost realism.** The reward path is the measured wall. Options are judged per-rollout:
   extra remote/served calls multiply by docs × G × rounds.
7. **Credit.** Per-span terms compose with exact counterfactual credit; doc-level terms enter
   the combine like the decision/schema components and rely on counterfactuals for span
   attribution.

## Option count-shaped — reward higher-count levels directly

Per span, a privacy bonus increasing in the chosen action's anonymity count; e.g.
`priv(s) = log10(max(aset(a_s), 1)) / log10(GENERIC)` (KEEP → 0, generic → 1), doc term =
mean over spans, reward `= u − λ·(1 − priv)` or equivalently `u + λ·priv` (sign convention
fixed at pre-registration).

- **Pros:** zero marginal cost (counts precomputed in the arms artifact); deterministic;
  per-span credit exact without counterfactuals; smooth in lattice depth; λ is a clean,
  honest Pareto knob that restores the operating-point story the floors' retirement removed.
- **Cons:** inherits count quality wholesale — today that means rewarding the artifact's
  bimodal defaults (1.0 vs 1000.0) for domain types; **blocked on the certifying-counts
  decision (issue register §3)** exactly as the floors were. Context/task-blind (the same
  count for a rare condition in a village note and a metropolis note) — the same critique
  that retired the floors, now in reward form. The policy can also Goodhart the count
  *artifact structure* (it already sees `log10_aset` as a feature).
- **Honest role:** dense shaping term, not a privacy claim. Acceptable as stage one once
  counts are grounded; meaningless before that.

## Option span-recovery probe — attacker recovers the original span

Frozen local attacker LLM; per probed span, given `doc_p` with the replacement located (or
just the surrounding window), it outputs its best guess of the original value;
`leak(s) = fact_score(guess, original)` (the deployed scorer). Reward
`= u − λ·mean_s leak(s)`.

- **Pros:** **context-aware** — this is the first privacy signal that sees the famous-context
  gap the counts are structurally blind to (a coarse fill that context still pins down scores
  as leaked). Per-span credit stays exact. Moderate cost: one short attacker call per probed
  span, batchable, cacheable on `hash(doc_p, span)`; prompt-caches the shared doc prefix like
  the reader does.
- **Cons:** crude objective — recovering a value ≠ identifying a person (a recovered
  "hypertension" in a k=1e6 population is not a re-identification); over-penalizes common
  values unless the score is normalized by a prior (an attacker that guesses "hypertension"
  from base rates leaks nothing — consider penalizing only the *margin over a context-free
  guess*, i.e. leak = score(guess | doc_p) − score(guess | type alone); this differential
  form is my recommended variant and mirrors the ceiling/floor anchor logic the utility
  probes already use). Attacker strength defines the realized pressure (pin + re-gate).
- **Honest role:** the practical middle option — the cheapest signal that is actually about
  *this document's* residual information rather than a global count.

## Option profile-matching probe — re-identification against a candidate set

Per doc, build one true attribute profile (from the detected spans / gold) and k−1
distractors; the frozen attacker gets `doc_p` + the k profiles (order shuffled per call,
seeded — `mc_shuffle` machinery) and picks the match. Penalty = chance-normalized accuracy,
optionally averaged over a small number of fixed distractor draws. Reward `= u − λ·penalty`.

- **Pros:** the most deployment-faithful objective — it *is* the headline metric
  (LLM re-identification) moved into training; measures linkage, not value recovery; graded
  and interpretable (margin over 1/k); naturally doc-level, matching how privacy is actually
  claimed.
- **Cons:** expensive (long prompt: k profiles + doc_p, per rollout; k and profile length
  multiply tokens); needs a whole profile-construction pipeline (attribute extraction,
  distractor sampling) that becomes part of the pinned environment; **distractor hardness is
  a hidden knob** — easy distractors overstate privacy (the "weak attacker" failure), so the
  sampling policy must be pinned as hard negatives (nearest-neighbor profiles by embedding
  similarity), pre-registered; doc-level credit smears across spans (counterfactual
  round trips required for attribution, at additional cost).
- **Honest role:** the end-state reward and the strongest claim; premature as the first
  implementation because every one of its knobs (k, distractor policy, profile schema) enters
  the re-gate surface at once.

## Additional analysis and ideas (coordinator)

- **Hybrid staging (recommended shape):** dense-cheap + sparse-honest. Use the count-shaped
  term (once counts are grounded) or the span-recovery term as the *per-span dense* signal,
  and the profile-matching probe *sparsely* — e.g. only in the RLOO refiner stage, only on
  ExIt winners as a verification-style gate, or at checkpoint selection — so the realistic
  objective disciplines the cheap proxy without paying its cost per rollout.
- **Distilled risk model (privacy-side RM analog):** fit a cheap regressor
  (embedding + counts + span features → attacker outcome) on cached profile-matching results,
  and use it for *candidate screening only, never gradients* — the same rule the spec already
  imposes on the utility-side distilled RM. If it ever feeds gradients it becomes a learned
  reward with all the Goodhart surface that implies.
- **Differential scoring for span-recovery** (see option above): penalize only the attacker's
  improvement over its context-free guess. This kills the base-rate false-leak problem and is
  cheap (one extra cached context-free call per span *type-value*, amortized corpus-wide).
- **λ is the new floor.** Whichever option lands, λ (or the attacker-target level) is the
  declared operating-point knob replacing the retired floors: training may condition on it
  (the `log10_active_floor` feature slot is already there and currently inert), eval sweeps
  it, and the Pareto claim is made at matched *realized* privacy as always. What is NOT
  acceptable is tuning λ per model/corpus to make a secondary number look right.
- **Guarantee honesty:** a reward term is pressure, not a guarantee. The mask used to
  guarantee "no detected span survives verbatim"; that guarantee is gone for
  lattice-bearing types. If deployment needs a hard rail (e.g. regulated settings), it
  returns as a *user-set* per-type constraint at inference time — a product feature, not a
  training-environment fiction.
- **Sequencing dependency:** the count-shaped option is blocked on the certifying-counts
  fork (issue register §3: wire Mondo/DOID / ChEMBL-ATC member sets, or re-scope). The
  attacker options are not — they need no counts at all, which is an argument for
  span-recovery-first if §3 stays unresolved.

## Comparison

| option | signal quality (privacy realism) | context-aware | credit | marginal cost/rollout | blocked on |
|---|---|---|---|---|---|
| count-shaped | low (proxy; artifact-sensitive) | no | per-span exact | ~0 | grounded counts (§3) |
| span-recovery | medium (value recovery, differential variant recommended) | **yes** | per-span exact | +1 short call/span (cached) | attacker pin |
| profile-matching | **high (re-identification itself)** | yes | doc-level (+counterfactuals) | +1 long call (+profile pipeline) | profile pipeline, distractor policy, cost |

## Open decisions (for when this is picked up)

1. Which option (or hybrid staging) — and the pre-registered λ sweep grid.
2. Training attacker binding (family-separated from gemma/Qwen3.6 roles; served vs local).
3. Span-recovery: plain vs differential scoring.
4. Profile-matching: k, profile schema, distractor-hardness policy, per-rollout vs sparse
   placement.
5. Whether the inert floor-conditioning feature slot is repurposed for λ-conditioning or
   removed.
