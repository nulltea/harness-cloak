---
type: training-experiment
status: done
created: 2026-07-05
model: RankerPolicy feature-MLP (64-h, N_FEAT 17 incl. log10_aset + active-floor), BC'd from the floor-walk
dataset: ranker_env.json (floor environment 2026-07-04 — aset-annotated actions, keep-original, k_floors all types 100) — 23 trainable docs, 177 decision spans, 106 train probes
result: "NULL again at these settings — byte-identical greedy read-outs across alphas; support re-measurement shows why: 7/106 flipping probes, ALL downward (min-aset BC init is u_qa-optimal by construction) — utility is a penalty-only term around this init"
tags: [ranker, stage1, bandit, reinforce, surrogate-reward, count-floors, latticecloak]
companion: ../../docs/specs/RL/surrogate-ranker-infiller.md
---

# Stage-1 ranker bandit on the count-floor environment — fixed floors, REINFORCE + KL

## Objective & hypothesis
Does learned per-span level selection move the surrogate reward off the floor-walk
behavior-clone init, now that the environment's reward support is restored? Predecessor
([2026-07-04 bandit](2026-07-04-RL-ranker-v1-stage1-bandit.md)) was a NULL whose corrected
diagnosis is reward-support starvation at the retired tau=0.02 mask (3/106 probes flippable;
`scripts/spikes/probe_flip_scan.py`). The count-floor environment roughly doubles decision
freedom (140/177 spans ≥2-legal at the all-100 floors vs 72/177). H: with more live probes,
REINFORCE finds per-(type, context-feature) trades the walk cannot express; movement off the
BC init at fixed alpha is the success signal.

This is the **fixed-floor arm** of the pre-registered protocol (spec §5.4-2): it precedes the
floor-randomized run at the same update budget, so a randomized NULL later is attributable to
nonstationarity, not to the environment.

## Pre-flight (spec §6 gate, chained before the run in the same job)
1. **Reward gate re-run** on the floor environment (`scripts/reward_gate.py`): required
   because the gate certifies a (reward, environment) pair. Gate inputs (constructed arms,
   probes, u_qa, extractor) are bit-identical to the last PASS (U~realized 0.354/0.511/0.760,
   all corpora positive, incl. identity_only arm), so this is expected to reproduce; the run
   makes it a measured fact. Bar: mean per-doc Spearman positive on every corpus.
2. **Probe-flip support re-measurement** (`scripts/spikes/probe_flip_scan.py`, floor-based):
   flippable and actually-flipping probe counts at the new BC point. Predecessor baseline:
   27 flippable / 3 flipping of 106. No numeric go/no-go pre-set — the measured value is
   context for interpreting the training outcome (a persistent NULL with support ≈ 3 is the
   old desert; a NULL despite support ≫ 3 is a genuine optimization/learnability finding).

## Training data
data/ranker_env.json (floor environment, 2026-07-04): aset-annotated action tables,
keep-original actions (illegal at the default all-100 floors — keep-legal count 0),
per-type k_floors all 100 (default-deny incl. MISC/OTHER), persisted probe splits (seed 0).
23 trainable docs / 177 decision spans / 106 train probes; ≥2-legal spans 140/177.

## Training config
scripts/train_ranker.py @ 233cae7 — REINFORCE, group advantage (G=8), KL leash to the
floor-walk BC init (coef 0.05), Adam lr 3e-4, 5 epochs, BC pretrain 30 epochs, seed 0,
alphas {0.3, 0.5, 0.7}, **fixed floors (no randomization)**. Reward r = alpha(1-A) +
(1-alpha) u_qa; echo unpriced (spec §5.2). BC teacher = min-aset floor-walk; static-teacher
injectivity collisions accepted and masked in rollouts (spec §3.3-1).

## Selection & operating point
Greedy read-out per alpha at the env floors (deployment operating point); one policy = one
(alpha, floor-config) point; three alphas = the training-time Pareto sample. No cross-floor
averaging (fixed floors throughout).

## Evaluation & success criteria
Training-side only (surrogate reward). Success = reward/component trajectories move
coherently off the BC init at fixed alpha, and greedy read-outs differ across alphas
(the predecessor's read-outs were byte-identical — that is the failure signature to beat).
Kill = flat at init again; interpretation then depends on the measured probe-flip support
(see Pre-flight 2). Realized evaluation (fact recall on out_final + frontier attacker vs
floor-walk Pareto) is spec Phase 3 and a separate run — no realized claims from this record.

## Results (measured 2026-07-05; logs in Artifacts)

- **Gate re-run: PASSED** — U~realized per-doc Spearman 0.354 clinical (n=13) / 0.511 enron
  (n=9) / 0.760 aeslc (n=5), positive on every corpus; reproduced the prior pass exactly
  (gate inputs bit-identical, as predicted in Pre-flight).
- **Probe-flip support (floor env):** flippable-in-principle 27 → **60**/106; actually
  flipping 3 → **7**/106. **All 7 flips are downward** (bc_f1 0.5–1.0 → 0.0, every
  counterfactual = PLACEHOLDER): the min-aset floor-walk init already holds the
  u_qa-maximal choice on every flippable probe. Baseline f1 hist: 88 zero / 7 mid / 11 one.
- **Training: NULL, same signature as the predecessor.** Greedy read-outs **byte-identical
  across alphas** (A 0.2662, U 0.1032, ph 0.4854); per-epoch means flat (a=0.3 r
  0.2923→0.2923; a=0.5 0.4194→0.4199; a=0.7 0.5464→0.5474); KL(pi‖pi_0) ≤ 0.0053. Wall
  ~1285 s/alpha (u_qa reader unbatched). The *operating point* moved with the environment
  (U 0.0909→0.1032, ph 0.639→0.485 vs 2026-07-04) — the *learning* did not.

**Interpretation (supersedes "reward desert" as the whole story).** The environment fix
worked — support doubled — but revealed the deeper structure: u_qa's coarsening invariance
plus a max-findability (min-aset) init makes the utility term **penalty-only**: no action
the policy can take raises u_qa above the init, 7 probes punish deviation, and the only
strictly-improving direction left is mean-shaping of A (placeholdering high-proximity
non-probe spans, ΔA ≈ (p6_s − Ā)/(n−1) ≈ 0.02–0.05 per span) — the same order as the
rollout reward std (~0.024), which 115 REINFORCE updates at lr 3e-4 cannot integrate.
This is now a reward-structure finding, not an environment finding.

## Landscape-probe addendum (measured 2026-07-05, scripts/spikes/reward_landscape_probe.py)

The "cannot integrate" reading above is superseded by a complete mechanical account:

1. **The reward's optimum at alpha > 0 is the degenerate all-placeholder point**: mean
   r(all-placeholder) − r(BC init) = **+0.062 / +0.119 / +0.177** at alpha 0.3/0.5/0.7 —
   and 43–46 of 96 single level→placeholder swaps are locally r-positive (mean +0.013–0.029,
   max 0.19). A real climb exists and points at collapse.
2. **The KL leash prices that climb out of reach — by design**: the BC init assigns
   placeholders mean **−log p = 4.42 nats**; at kl_coef 0.05 the marginal leash cost of
   shifting mass to placeholder (~0.22/span) exceeds the marginal reward gain (≤0.03/span)
   by ~10×; equilibrium mass shift ≈ 2% — invisible in greedy read-outs. Matches the
   measured KL ≤ 0.005 exactly.
3. **At alpha = 0 the BC init is the utility optimum outright** (r_bc 0.1032 vs
   all-placeholder 0.0786; zero positive swaps): u_qa is penalty-only around min-aset, as
   the flip scan showed.

**Net: the NULL is the correct equilibrium of this (reward, leash, init) triple, not an
optimizer failure.** The privacy term rewards only the degenerate direction (leash-blocked,
per anti-Goodhart §5.3 intent); the utility term is blind to the non-degenerate axis
(specificity trades — inversion-invariance); nothing leash-affordable improves r. This
elevates spec §7-2's reward escalation from open tension to the binding constraint.

## Observations — lever ladder for the successor (ranked)
1. **Coarse-init ablation (cheapest, most diagnostic) — CORRECTED by the landscape probe:**
   BC from an all-placeholder teacher, **alpha = 0 (pure u_qa), KL leash off (or coef ≤
   0.005)**. The original framing (any alpha, leash on) was confounded twice: at alpha > 0
   the all-placeholder init is near the reward optimum (staying put is correct, proves
   nothing), and the leash blocks movement from ANY init at these signal sizes (−log
   p(other-action) ≈ 4+ nats ≫ gains). At alpha=0/no-leash the upward gradient is real and
   large (U climb 0.0786 → 0.1032 available; per-placement gains ~Δf1/n_probes ≈ 0.1–0.2
   through 60 flippable probes): climb → optimizer healthy, NULL fully attributed to
   reward+leash design; no climb → implementation/optimizer bug hunt.
2. **Reward-side:** a utility term that differentiates specificity above bare
   invertibility (graded reader confidence / margin instead of 0-1 F1 through inversion) —
   a spec §5 change, re-gate required.
3. **Optimization throughput:** batch the u_qa reader (~90% of wall) + per-span
   counterfactual credit for the A-shaping direction, then 5–10× more updates.

## Chain-integrity note (empirical honesty)
The probe-flip step crashed on the first chain run (static floor-walk teacher is
non-injective — "a certain amount"; the spike lacked the trainer's dynamic collision rule)
and a missing `pipefail` let the chain continue to training anyway. The spike was fixed
(baseline now applies the trainer's walk-order collision→placeholder rule) and re-run
standalone; training was unaffected (its own guards ran). Gate and training numbers are
from the chain run; probe-flip numbers from the fixed re-run.

## Ablations
Fixed-floor vs floor-randomized (same budget) is the pre-registered next comparison; the
conditioning ablation (floor-blind vs conditioned at held-out floors) attaches to the
randomized run, not this one.

## Cost
Local iGPU only, zero remote calls (gate uses cached round trips). Estimate: gate ~6 min,
probe-flip ~4 min, training ~22 min/alpha (u_qa reader unbatched — known ceiling; batch the
reader before any larger sweep), total ≈ 75 min.

## Risks & caveats
u_qa quantization at ~2.5 probes/doc (advantage noise); opposing placeholder biases in the
reward (spec §7-3); keep-original absent from the action space at these floors — the
identity-only-style trades are reachable only under user waivers, not tested here; corpus
mix dominated by clinical.

## Artifacts
results/gate_floor_env.log, results/probe_flip_floor_env.log, results/train_floor_env.log,
results/ranker_train_a{0.3,0.5,0.7}.json, data/ranker_policy_a{alpha}.pt,
results/ranker_reward_gate.json (refreshed).

## Sources
Spec: [surrogate-ranker-infiller](../../docs/specs/RL/surrogate-ranker-infiller.md) (§2
Phase 1, §5.4, §6). Predecessor: [2026-07-04 bandit](2026-07-04-RL-ranker-v1-stage1-bandit.md).
Migration plan: [structural-lattice-risk](../../docs/plans/2026-07-04-structural-lattice-risk.md).
