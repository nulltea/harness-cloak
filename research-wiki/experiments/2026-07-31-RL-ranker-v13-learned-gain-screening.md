---
type: experiment
node_id: exp:RL-ranker-v13-learned-gain-screening
verdict: ""
confidence: ""
created: 2026-07-31
model: semantic-v1 policy (controller_production s47 BC/ExIt warm start, softcap 25, no gap-scaling, alpha-init switch-calibrated raw, sensitivity reg 0.1, LEARNED state-conditioned controller gain)
dataset: aci 4-doc controller-strength spike set, frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: pending
tags: [rl, ranker-v2, tie-ownership, learned-controller-gain, escalation]
companion: ../../docs/research/reward-ties-and-controller-authority.md
---

# RL-ranker v13 — learned controller gain screening

The tie-ownership fork's preregistered escalation, running on the adopted infrastructure base (softcap 25 + no-gap + sensitivity 0.1 — the configuration that uniquely found a stable equilibrium in v12 but at insufficient separation, final Delta P +0.12 vs the 0.20 gate).

## Objective & hypothesis

A zero-initialized, bounded, state-conditioned gain residual (alpha_j = softplus(alpha_raw + 1.5*tanh(delta_phi(stopgrad(h_j))/1.5)), trained by the count objective and lambda>0 sampling gradients) gives the count reward its first action-specific pathway: decisions whose utility resists cheaply keep gain near the calibrated alpha, while reward-silent/tied decisions accumulate gain until they switch — lifting the no-gap equilibrium's separation past the gate without destabilizing it. Guardrails per the registered literature: bound in raw-logit space (attested failure = unbounded upward growth), gradient training not dual ascent (multiplier-network oscillation), per-decision controller_alpha recorded every epoch in the synchronous snapshots (gain-field smoothness gate), and a RANDOM-gain control arm (deterministic parameter-free per-decision offsets) mandatory before any adoption claim.

## Config

Production trainer, 8 epochs, seed 47 first, base 8 rollouts, Arm C counterfactual coverage, `--utility-logit-softcap 25 --alpha-init switch-calibrated --profile-sensitivity-reg 0.1 --controller-gain learned --controller-gain-bound 1.5 --controller-gain-hidden 32`, NO gap-scaling, NO KL-anchor flags, synchronous profile eval. Implementation commit 06c6c56.

## Evaluation & success criteria

v12 gate set (lambda-zero utility <= 0.044 vs control and free convergence; D2N005 synchronous Delta P(lambda-3) >= 0.20 final, cycle range <= 0.10 on sampled AND greedy paths; ranges <= 50; no persistent p > 0.999; flat-decision expected privacy >= 0.50) PLUS gain-field smoothness (per-decision controller_alpha varies smoothly across epochs, no oscillation/divergence toward the bound). If learned passes: run the random-gain control (same everything, `--controller-gain random`) — adoption requires learned to beat random on the gates it passes (the random-weighting litmus). If learned fails: the fork's remaining options are the epsilon-lexicographic constrained controller or accepting a documented tiny-doc variance bound.

## Results

- **gain-learned s47 at shared lr 1e-4: MECHANISM NULL.** Per-decision controller_alpha moved 5.35 -> 5.39 over 8 epochs, UNIFORM across decisions (no differentiation); the trajectory replicates no-gap-s47 nearly exactly (dP series +0.12/+0.14/+0.07/+0.03/+0.36/+0.14/+0.11/+0.19 vs no-gap +0.12/+0.14/+0.07/+0.03/+0.34/+0.15/+0.11/+0.12). The hypothesis was NOT tested: the zero-init head at the shared lr learns orders of magnitude too slowly for a screening window — the same learning-timescale trap that motivated calibrated alpha-init. No smoothness or divergence issues (the field is simply flat). Ranges bounded (max 31.9); ep7 hints late divergence from no-gap (+0.19 final, greedy 0.34/0.52, lambda-zero drifting to L0-ish selections — utility-free by the tie structure).
- AMENDMENT (pre-rerun): --controller-gain-lr param group added (gain head at 1e-2, 100x the shared lr; everything else unchanged); rerun as the actual mechanism test.
- **gain-fastlr s47 (8 epochs): mechanism ALIVE but GLOBAL-ONLY; gates fail on D2N005.** The alpha field now moves decisively (5.1 -> 4.77 -> 6.13) but stays UNIFORM across every decision of every document — the count gradient's common component dominates the zero-init head, so it learns a faster global alpha first; per-decision differentiation (which requires the opposing lambda>0 utility gradients to accumulate: zero on ties, negative on live decisions) has not yet emerged. Split outcome: the large documents reach their strongest separation ever recorded (D2N027 +0.42/+0.55, D2N031 +0.43/+0.49 at epochs 6-7 — the count pathway demonstrably works where authority suffices), while D2N005 goes fully lambda-inert for epochs 2-6 (all profiles 0.00) and only begins recovering as alpha climbs (+0.07 final). Ranges bounded (31.2 max); no oscillation or bound-divergence in the gain field (smoothness trivially satisfied — it is flat across decisions).
- AMENDMENT 2: 16-epoch extension of gain-fastlr (same config) to test whether per-decision differentiation emerges in the later phase — the classic global-first/differentiate-later learning ordering predicts it; if the field is still uniform at 16 epochs, the state-conditioned head (pooled-feature input) lacks discriminative signal and the fork falls to epsilon-lexicographic or the variance bound.

## Artifacts

results/ranker_v2/architecture/tie_ownership/{train,control,kl-ref,epochs}-gain-{learned,random}-s47.*, logs alongside. Predecessors: v12 (all capacity-free arms), decision log 2026-07-30/31.
