---
type: research
status: current
created: 2026-07-30
updated: 2026-07-30
tags: [rl, ranker-v2, reward-ties, entropy-collapse, controllability, kl-anchor,
       failure-taxonomy, literature]
companion: [docs/specs/RL/interactive-ranker-v2-decision-log.md,
            research-wiki/training/2026-07-30-RL-ranker-v11-kl-anchor-spike.md]
---

# Reward ties and controller authority: why the privacy dial flips coins on small documents

A failure taxonomy for the λ-conditioned ranker policy, assembled from three
independent analyses (this session's systematic-debugging investigation, a
Codex Sol High pass, and a 27-source literature sweep) after the λ3 privacy
behavior on the smallest training document was measured to flip between
seeds AND between training cycles within one run, despite a monotone,
well-formed count-score reward. Every claim below is either a measurement
from this repo's artifacts or a cited external result.

## Definitions

- **Policy decision / menu**: one controlled entity occurrence group in a
  document; its menu holds generalization-level actions (L0 = finest), a KEEP
  action (retain the original surface), and a placeholder action.
- **Utility logits u(a)**: the shared tower's per-action scores. The tower is
  deliberately λ-blind; λ enters only through the controller.
- **Controller / additive shift**: at λ>0, sampling logits are
  `z(a) = u(a) + α·g(λ)·gap·p̂(a)` where p̂(a) is the action's frozen
  profile-relative count score, `gap` is the detached per-menu utility-logit
  range, and α is a global scalar initialized by switch-threshold calibration.
  λ=0 is an exact identity branch (`z = u`).
- **Count score P**: mean per-decision profile score of selected actions — the
  privacy *shaping proxy* (never a privacy claim; realized privacy is attacker
  success).
- **Reward tie**: two actions whose measured utility is exactly equal in every
  observed context (not merely close).
- **Margin**: the utility-logit difference between two actions; **sharpening**
  = margin growth during training; **saturation** = margins so large the
  softmax is effectively deterministic and gradients of entropy/KL/count terms
  vanish.
- **LOO credit**: leave-one-out advantage over a group's rollouts — the
  utility gradient's source.
- **Synchronous snapshot**: evaluation of all λ profiles from ONE checkpoint
  (64 sampled vectors per profile plus greedy per-decision statistics);
  Latin-cycle group means are confounded because each profile is observed
  several optimizer steps apart.
- **Controllability**: measured sensitivity of realized behavior to the
  conditioning input λ — reported per document, seed, and cycle.
- **Reference / anchor**: the calibrated warm-start policy snapshot used by
  the KL term.

## The observed phenomenon

On the 4-document controller-strength set, the two large documents (19-25
policy decisions) are seed-invariant (final-cycle ΔP(λ3) spread ±0.03 across
seeds), while the smallest (aci/D2N005, 4 decisions) spans +0.05..+0.42 across
seeds and lurches between cycles within single runs (e.g. 0.53 → 0.27 → 0.49).
Fixed-checkpoint replay (128 trajectories) proves this is real policy
divergence, not sampling noise: final E[P|λ3] = 0.498 / 0.271 / 0.084 across
seeds with 8-rollout standard errors ≤ 0.017. In the worst seed the λ3 policy
selects KEEP nearly everywhere — the action with the *minimum* privacy score —
under maximum privacy pressure.

## Root-cause chain (all five steps measured in this repo)

1. **Tie-silent reward.** The utility scorer returns *exactly* equal scores
   for KEEP and L0 on all four D2N005 decisions (113/113 cached
   single-decision contexts; delta exactly 0.000), and one decision is flat
   across its entire menu. Level-0 generalization is utility-free privacy —
   but no utility gradient can ever order it against KEEP.
2. **Drift fills the silence, with a consistent direction.** At BC
   initialization the KEEP−L0 margins are negative (−5.6..−7.6; BC clones
   levels). After RL they are +41..+225 on every seed. The corpus-wide truth
   "KEEP is utility-safe" leaks through the shared action-mode embeddings and
   utility head into exactly the pairs where this document's reward is silent.
3. **Sharpening is unbounded.** Real signal elsewhere in the menu (deep levels
   genuinely cost utility) drives softmax margins without limit: per-decision
   logit ranges grew from ~9-12 at initialization to 277-327 on the collapsed
   seed. The fixed entropy bonus (β=0.01) is structurally outrun (see
   taxonomy), and the KL safety trigger is aggregate-level — it never fired.
4. **A bounded shift races unbounded margins.** The controller can add at most
   `α·Δp̂·range` (~22-43% of the logit range for KEEP→L0/L1). The drifted
   KEEP-margin fraction crosses that line back and forth during training —
   cycle flips — and on the collapsed seed exceeds it permanently. The final
   checkpoints' argmax reproduces every observed behavior deterministically.
5. **Nothing can push back.** The count objective's gradient reaches only the
   global scalar α by design (count_combined detaches u, p̂, and gap), and the
   forward-KL direction loses its gradient exactly under the saturation it is
   meant to correct.

Refuted candidate causes (measured): profile count targets (monotone,
well-separated, no inversions), α-calibration error (switchable-decision
fraction ~50% on every document), and sampling noise.

## The anchor experiment and the structural dilemma

An always-on weak KL to the calibrated reference (η=0.01, forward and reverse
directions, RL-ranker v11) fixed steps 2-4 — logit ranges bounded at 7-18,
flat-decision privacy held, λ3 kept near the reference instead of collapsing —
but pinned the λ0 profile to reference behavior (P stuck ~0.45 instead of
converging to utility-optimal ~0), so the profile separation the dial exists
to express never opened. Both directions failed identically at the first
cycle. The tension is structural, not a coefficient problem: one leash
strength cannot serve a profile whose job is to *leave* the reference (λ0) and
a profile whose job is to *stay near* it (λ3). This exact fixed-reference vs
current-policy regularization trade-off is the stated premise of
[he2026_unifying_stable_optimization_reference_regularization](../../research-wiki/papers/he2026_unifying_stable_optimization_reference_regularization.md)
([arXiv 2602.11523](https://arxiv.org/abs/2602.11523)).

## Taxonomy: each step is a named phenomenon

| Step | Named phenomenon | Canonical source |
|---|---|---|
| 1 | Advantage collapse / value-function interference from a many-to-one utility; reward-homogeneous groups are gradient-dead | [vamplew2024_value_function_interference](../../research-wiki/papers/vamplew2024_value_function_interference.md) ([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)); [yu2025_dapo_open_source_llm_rl](../../research-wiki/papers/yu2025_dapo_open_source_llm_rl.md) ([arXiv 2503.14476](https://arxiv.org/abs/2503.14476)) |
| 2 | Underspecification (seed decides what the loss is indifferent to); policy churn through shared parameters | [damour2020_underspecification_ml](../../research-wiki/papers/damour2020_underspecification_ml.md) ([arXiv 2011.03395](https://arxiv.org/abs/2011.03395)); [schaul2022_policy_churn](../../research-wiki/papers/schaul2022_policy_churn.md) ([arXiv 2206.00730](https://arxiv.org/abs/2206.00730)) |
| 3 | Entropy collapse: entropy decay is advantage-proportional covariance, so fixed bonuses are outrun by construction | [cui2025_entropy_mechanism_rl](../../research-wiki/papers/cui2025_entropy_mechanism_rl.md) ([arXiv 2505.22617](https://arxiv.org/abs/2505.22617)) |
| 4 | Preference-conditioning controllability failure: agents ace aggregate metrics while the conditioning input is behaviorally inert | [delasheras2026_controllability_preference_morl](../../research-wiki/papers/delasheras2026_controllability_preference_morl.md) ([arXiv 2605.10585](https://arxiv.org/abs/2605.10585)) |
| 5 | Fixed-reference vs current-policy regularization tension in RLHF | [he2026_unifying_stable_optimization_reference_regularization](../../research-wiki/papers/he2026_unifying_stable_optimization_reference_regularization.md) ([arXiv 2602.11523](https://arxiv.org/abs/2602.11523)) |

Two prescriptions carry over directly. Vamplew et al. show random or drifting
tie-breaking is itself an instability source and demand tie resolution that is
**deterministic and stated**. de las Heras et al. make **controllability a
first-class reported quantity** — a stabilizer can pass every aggregate gate
while leaving λ inert, so the synchronous per-document λ-sensitivity readout
is the acceptance instrument for any fix.

## Solution space (three-way analysis)

Constraints throughout: no pre-baked per-decision constants (training must
stay meaningful); the count gradient must not enter the utility tower (u keeps
utility semantics; λ0 stays an exact identity); no per-model calibration
knobs; auditable mechanisms only.

- **Entropy floor (auto-tuned target entropy).** Enforce a small minimum
  per-menu entropy (target normalized entropy ≈ 0.10) via a dual variable —
  the constrained formulation of
  [haarnoja2018_sac_algorithms_applications](../../research-wiki/papers/haarnoja2018_sac_algorithms_applications.md)
  ([arXiv 1812.05905](https://arxiv.org/abs/1812.05905)) — replacing the dead
  fixed bonus. Bounded margins restore the calibrated controller's authority:
  on exact ties the additive shift then decides **deterministically**, meeting
  the Vamplew prescription with zero new capacity and no reference to go
  stale. This session's top pick; the literature sweep rates the
  floor/covariance-clamp family "the best structural fit" for a bounded
  additive shift on shared logits. Open risk: it makes ties controller-owned
  but does not give privacy any *learned* per-decision authority.
- **Learned state-conditioned controller gain.** Codex Sol High's top pick:
  `α_j = softplus(α_raw + δ_φ(stopgrad(h_j)))`, zero-initialized, trained by
  the count and high-λ utility gradients, with u untouched. Rationale:
  *conditional negative transfer* — four λ-tasks share one λ-blind tower and
  only the controller can separate them, so controller capacity (not just
  optimization stability) is load-bearing; entropy floors "say remain
  uncertain, not choose more private". Risks: capacity/overfit on 63
  documents, gain explosion if margins stay unbounded (it composes with, and
  arguably wants, the entropy floor), audit complexity, and recalibration of
  frozen artifacts.
- **λ-gated fixed-reference KL** (η(λ) = η·g(λ)). Cheapest patch for the v11
  failure; demoted by both other analyses because high-λ KL gradients still
  update the shared u (indirect λ0 pinning) and it preserves reference
  behavior rather than deriving privacy from the count reward. Closest
  literature precedent for anchor-selectivity is runtime-statistic-gated KL —
  [lin2026_tepo_token_level_policy_optimization](../../research-wiki/papers/lin2026_tepo_token_level_policy_optimization.md)
  ([arXiv 2604.12736](https://arxiv.org/abs/2604.12736)) — but anchoring
  specifically on the reward-indifference set is unpublished.
- **Trust region to the previous policy / churn mitigation.** Bounds per-step
  change, not cumulative drift; can entrench an early seed-selected mistake.
  Stabilizer, not an owner of ties.
- **ε-lexicographic constrained controller** (maximize count subject to
  utility loss ≤ the measured 0.044 reader-noise floor). The faithful
  formalization of "free privacy"; second-line escalation because it leans on
  counterfactual attribution quality.

## Novelty gaps (found by the sweep, relevant for eventual write-up)

1. "Bounded additive controller authority vs unbounded logit scale" as a
   design quantity — the required ratio between controller range and margin
   scale — appears unpublished; step 4 of the chain is an original synthesis.
2. Regularizing specifically on the reward-indifference set ("KL-on-ties")
   has no direct precedent.
3. Exact ties produced by a graded (non-binary) scorer are not treated as a
   distinct phenomenon anywhere in the verified literature.
4. No standard metric tracks logit *range/margin* directly (entropy is the
   universal proxy); our 9 → 300 measurement has no published counterpart.

## Adjudication protocol (preregistered in the decision log)

Two arms test the two live hypotheses head-to-head — Arm E (entropy floor,
no KL) vs Arm G (learned monotone gain, no KL) — seeds 17+47, 8-epoch
screening with the synchronous snapshot, early kill, v11 and unanchored runs
as frozen controls. Gates: per-document λ0 utility loss ≤ 0.044 vs the fixed
control; D2N005 synchronous ΔP(λ3) ≥ 0.20 by the final cycle with cycle range
≤ 0.10; cross-seed final difference ≤ 0.10; no menu-logit range > 50 and no
persistent action probability > 0.999; frontier regret reported. Decision
rule and escalations are recorded in the decision-log fork entry
(2026-07-30). The general lesson already adopted: **controllability is
reported per document, seed, and cycle from synchronous snapshots — never
from aggregate or Latin-cycle-confounded readouts.**

## Sources

Registered wiki pages (all cited inline above):
[vamplew2024_value_function_interference](../../research-wiki/papers/vamplew2024_value_function_interference.md) ([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)) ·
[cui2025_entropy_mechanism_rl](../../research-wiki/papers/cui2025_entropy_mechanism_rl.md) ([arXiv 2505.22617](https://arxiv.org/abs/2505.22617)) ·
[delasheras2026_controllability_preference_morl](../../research-wiki/papers/delasheras2026_controllability_preference_morl.md) ([arXiv 2605.10585](https://arxiv.org/abs/2605.10585)) ·
[he2026_unifying_stable_optimization_reference_regularization](../../research-wiki/papers/he2026_unifying_stable_optimization_reference_regularization.md) ([arXiv 2602.11523](https://arxiv.org/abs/2602.11523)) ·
[lin2026_tepo_token_level_policy_optimization](../../research-wiki/papers/lin2026_tepo_token_level_policy_optimization.md) ([arXiv 2604.12736](https://arxiv.org/abs/2604.12736)) ·
[haarnoja2018_sac_algorithms_applications](../../research-wiki/papers/haarnoja2018_sac_algorithms_applications.md) ([arXiv 1812.05905](https://arxiv.org/abs/1812.05905)) ·
[yu2025_dapo_open_source_llm_rl](../../research-wiki/papers/yu2025_dapo_open_source_llm_rl.md) ([arXiv 2503.14476](https://arxiv.org/abs/2503.14476)) ·
[damour2020_underspecification_ml](../../research-wiki/papers/damour2020_underspecification_ml.md) ([arXiv 2011.03395](https://arxiv.org/abs/2011.03395)) ·
[schaul2022_policy_churn](../../research-wiki/papers/schaul2022_policy_churn.md) ([arXiv 2206.00730](https://arxiv.org/abs/2206.00730)).

Repo evidence: decision-log entries of 2026-07-29/30
(docs/specs/RL/interactive-ranker-v2-decision-log.md), training records
RL-ranker v8-v11 (research-wiki/training/), result artifacts under
results/ranker_v2/architecture/{controller_production,credit_support,kl_anchor}/.
