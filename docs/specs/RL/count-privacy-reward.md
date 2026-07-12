---
type: reference
status: stale
created: 2026-07-12
updated: 2026-07-12
tags: [rl, reward-design, privacy, anonymity-counts, lambda, pareto, operating-point, spec]
companion: [docs/specs/RL/leakage-probe-reward.md,
            docs/specs/RL/roundtrip-ranker-infiller.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
superseded_by: docs/specs/RL/interactive-ranker-v2.md
archive_reason: Superseded by the full interactive-ranker v2 redesign; its convex-lambda decision is retired.
---

# Count-maximization privacy reward — design (in progress)

Implements the "count-shaped" option of the
[leakage-probe reward survey](leakage-probe-reward.md): after the 2026-07-12 floor
retirement, privacy pressure re-enters training as a reward term that pays the ranker for
choosing higher-anonymity-count actions, so it learns to balance utility against privacy
instead of having low-count actions masked away.

**Status: superseded.** This draft records an earlier convex-lambda design. The normative
replacement is [interactive ranker v2](interactive-ranker-v2.md), which uses additive
count shaping with distinct utility and count credit paths.

## Definitions

- **U** — the round-trip utility reward (graded fact recall / carrier reward on the round
  trip), in [0,1]. λ-independent; cached on `hash(doc_p)` as today.
- **P** — the count-based privacy score of a rollout's chosen actions, in [0,1]; analytic
  (computed from the arms artifact's per-action `aset`, no model calls).
- **priv(s)** — the per-span privacy score of span s's chosen action (see Working shape).
- **λ** — the privacy weight: a **fixed, pre-registered constant per run**, the declared
  operating-point knob (the ε-analogue under the empirical-honesty rules). Never learned,
  never tuned mid-run or post-hoc: a learnable λ has no gradient signal of its own, would
  collapse toward the cheapest term (utility, via KEEP), and is the definition of a hidden
  calibration knob. Realized privacy at any λ is measured by the eval attacker, never
  inferred from λ.
- **Indifference point λ\*** — the λ at which the reward is indifferent between two policies
  (or two actions for one span): `λ* = ΔU / (ΔU + ΔP)`.
- **GENERIC** — the sentinel count for known-generic fills (1e9, `cloak.anonymity`).

## FORK (DECIDED 2026-07-12): how λ trades utility against count-based privacy

**Chosen: convex combination `R = (1−λ)·U + λ·P`, λ ∈ [0,1].**

Why: clean endpoints — λ=0 is bit-identical to the utility-only reward (the baseline arm and
a mandatory regression gate), λ=1 is pure privacy (degenerates to all-placeholder; a
diagnostic endpoint, not an operating point). R stays in [0,1] so RLOO advantage scales stay
comparable across λ; the Pareto grid is a bounded sweep.

Recorded alternatives:
- **Penalty form `R = U − λ·(1−P)`** — same frontier, but λ is unbounded with no natural
  scale, R goes negative, and advantage magnitudes drift with λ. Rejected.
- **Constrained form (maximize U s.t. P ≥ target, dual ascent on λ)** — the most
  interpretable knob ("privacy ≥ 0.7") and the only principled sense in which λ is ever
  adapted (as a dual variable serving a fixed declared target). Deferred: adds constrained-RL
  machinery the pilot doesn't need; revisit if convex-λ sweeps prove hard to steer.

## Structural properties (why this term is cheap and exact)

- **λ never touches the round trip.** U depends only on doc_p; P only on the chosen actions.
  The reward cache keeps storing pure U; P is recomputed client-side. Determinism of R_rt is
  preserved trivially on the P side.
- **Exact per-span decomposition.** P is a mean of per-span terms, so counterfactual span
  advantages get an *analytic* privacy component (no extra round trips):
  `ΔR_s = (1−λ)·ΔU_s + λ·Δpriv(s)/n_spans`.
- **Zero marginal remote cost.** The counts are precomputed into the arms artifact.

## Working shape of P (defaults; open forks below)

```python
priv(s)  = log10(max(aset(a_s), 1)) / log10(GENERIC)   # KEEP -> 0; generic -> 1
priv(s)  = 1.0 if a_s is placeholder                    # nothing sent = max privacy
P(doc)   = mean(priv(s) for s in ranker_spans(doc))     # PERSON/CODE excluded (rule-masked,
                                                        # constant, would dilute the mean)
R        = (1 - lam) * U + lam * P
```

The log scale matches the existing `log10_aset` policy-feature normalization; a linear scale
would let one GENERIC action dominate the doc mean.

## λ calibration protocol (recorded 2026-07-12)

Full RL per candidate λ is unnecessary; calibration exploits "λ never touches the round
trip". Three stages, increasing cost:

1. **Replay cached rewards across a dense λ grid (free).** Recompute `R_λ` over any existing
   scored-rollout pool (support scan, ExIt group samples) for ~50 λ values — pure arithmetic.
   Output: the **active band** of λ (where within-group argmax, floor-walk rank, and
   selection direction actually change). Running RL outside the active band is wasted
   compute.
2. **Indifference points from rule-policy anchors (cached round trips, no training).**
   Evaluate fixed rule policies — KEEP-walk, min-aset non-KEEP (the BC teacher), a
   mid-lattice walk, all-placeholder — as (U, P) points; adjacent-policy indifference
   `λ* = ΔU/(ΔU+ΔP)` gives principled grid candidates. Per-span variant: from the support
   scan's counterfactual deltas, each span has a flip threshold
   `λ_s = ΔU_s/(ΔU_s + ΔP_s)`; the quantiles of the λ_s distribution ("flips 25/50/75% of
   span decisions") are candidates too.
3. **Train only at the survivors.** Either 2–3 fixed-λ runs at the selected candidates (the
   round-trip cache is shared across λ, so later runs are much cheaper), or a λ-conditioned
   policy (λ sampled per episode, fed through the input slot the retired floor feature
   vacated) read out across the grid at eval — see open fork.

Honesty boundary: calibration **selects and pre-registers** the λ values to run. It never
adjusts λ during or after a run to make an outcome look right; each operating point's
realized privacy is measured by the attacker at eval.

## Gates

- **λ=0 bit-identity:** with λ=0 the reward, selection, and gradients must be bit-identical
  to the utility-only path (regression-tested) — the baseline arm costs nothing.
- The support scan gates the U side as before; P needs no scan (analytic, full support by
  construction).
- λ values for any claimed run are pre-registered in the training record before launch.

## Open forks (brainstorm continues)

1. **Count-quality policy** — proceed on current mixed-provenance counts vs gate on the
   grounded-count merge (issue register §3 / nemotron artifact) vs source-aware credit (only
   grounded counts earn priv). The term inherits count quality wholesale; junk counts make P
   partly artifact-shaped.
2. **λ deployment** — fixed-λ runs at calibrated candidates vs one λ-conditioned policy
   (conditioning must then be validated).
3. **P details** — placeholder priv = 1.0 (working default); the normalization constant
   (log10 GENERIC = 9); span set (all ranker spans, PERSON/CODE excluded); whether
   probe-less spans count.
