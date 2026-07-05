---
type: training-experiment
status: done
created: 2026-07-04
model: RankerPolicy feature-MLP (64-h, ~5k params), behavior-cloned from the tau-walk
dataset: ranker_env.json (DETECTOR v2 pii_gliner_multidomain@0.3, switched pre-run) — 23 trainable docs (12 clinical / 8 enron / 3 aeslc), 177 decision spans, 106 train probes
result: "NULL at these settings — policy never left the BC init at any alpha (identical greedy read-outs); diagnosis: weak per-span credit, rare probe flips, not entropy collapse; superseded diagnosis: reward-support null (3/106 flippable probes)"
tags: [ranker, stage1, bandit, reinforce, surrogate-reward, latticecloak]
companion: ../../docs/specs/RL/surrogate-ranker-infiller.md
---

# Stage-1 ranker bandit — feature policy, REINFORCE + KL, fully local reward

## Objective & hypothesis
Does learned per-span level selection move the surrogate reward off the tau-walk
behavior-clone init? (H: the policy finds per-(type, context-feature) placeholder/level trades
a single global tau cannot express.) This is the plan's ablation-floor policy promoted to v0 —
a null here, with flat features, does NOT kill selection learning (the frozen-encoder feature
upgrade is pre-registered); a positive is a cheap existence proof.

## Training data
data/ranker_env.json (Phase-0 artifact) rebuilt 2026-07-04 on the v2 detector
(data/models/pii_gliner_multidomain/checkpoint-2479 @0.3 — deployment decision; TAB QUASI 0.979):
action tables with stored P4 walk_risk + P6 proximity, tau-legal masks (tau=0.02),
behavior-clone labels, persisted probe splits (seed 0). Trainable = probe-bearing AND
span-bearing: 23 docs / 177 decision spans / 106 train probes (v2's DEM/MISC recall nearly
doubled probe supply). BC reproduction verified 23/23. Reward gate re-passed on this
environment: U~realized 0.354/0.511/0.760 (results/ranker_reward_gate.json).

## Training config
scripts/train_ranker.py — REINFORCE, group advantage (G=8), KL leash to BC init (coef 0.05),
Adam lr 3e-4, 5 epochs, BC pretrain 30 epochs, seed 0, alphas {0.3, 0.5, 0.7}.
Reward r = alpha(1-A_P6) + (1-alpha) u_qa(train probes); echo channel unpriced (spec §5.2).

## Selection & operating point
Greedy read-out per alpha after training (logged as greedy_final); one policy = one operating
point; three alphas = the training-time Pareto sample.

## Evaluation & success criteria
This run's own metrics are TRAINING-side only (surrogate reward). Success = reward and
component (A, U, placeholder-rate) trajectories move coherently vs the BC init at fixed alpha;
kill = flat at init (then: frozen-encoder features next, per plan ladder). Realized evaluation
(fact recall on out_final + frontier attacker vs tau-walk Pareto) is Phase-3 of the spec and a
separate run — no realized claims from this record.

## Results (measured 2026-07-04, results/ranker_train_a{0.3,0.5,0.7}.json)

**Training-side NULL: the RL phase produced no movement off the behavior-clone init at any
alpha.** Full trajectories (per-epoch means over 23 docs, G=8 sampled rollouts):

| alpha | epoch 0 -> 4 (r) | A (e0 -> e4) | U (e0 -> e4) | ph_rate (e0 -> e4) | KL (e4) | wall |
|---|---|---|---|---|---|---|
| 0.3 | 0.2808 -> 0.2805 | 0.2778 -> 0.2768 | 0.0917 -> 0.0907 | 0.667 -> 0.657 | 0.0009 | 1317 s |
| 0.5 | 0.4070 -> 0.4069 | 0.2778 -> 0.2764 | 0.0917 -> 0.0901 | 0.667 -> 0.661 | 0.0013 | 1296 s |
| 0.7 | 0.5331 -> 0.5341 | 0.2778 -> 0.2754 | 0.0917 -> 0.0896 | 0.667 -> 0.662 | 0.0021 | 1297 s |

**Greedy read-outs are byte-identical across alphas** — A 0.2813, U 0.0909, ph_rate 0.6394 —
i.e. every alpha's final deterministic policy makes exactly the BC init's decisions; r differs
only through the trivial alpha-rescaling of the same (A, U). No alpha Pareto spread exists.
KL(pi || pi_0) <= 0.0021 everywhere: the leash never engaged; the policy did not move.

**Post-hoc diagnostics (trained a=0.5 policy, 8 docs, seed 1):**
- Policy entropy on multi-action spans: **0.52 nats** (uniform-over-2 = 0.69) — sampling
  exploration is present; NOT entropy collapse.
- Per-doc reward std over G=8: **mean 0.024, max 0.043, nonzero on 8/8 docs** — an advantage
  signal exists but is small.
- Scale comparison: one QA-probe flip moves r by ~0.1 at alpha=0.5 (1/n_probes ~ 0.2, utility
  weight 0.5); observed rollout std 0.024 << 0.1 ⇒ **rollout variations almost never flip a
  probe** — most of the variance is small P6 wiggles. The utility term contributes nearly no
  rollout-level gradient; the doc-level REINFORCE advantage is then diluted over ~8 span
  decisions, and 115 updates x lr 3e-4 cannot integrate what remains.

## Observations

- **This is an optimization-regime null, not (yet) a feature-ceiling or selection-learning
  null.** The pre-registered frozen-encoder upgrade triggers on "flat WITH healthy optimization";
  here the optimization itself was starved (weak, diluted gradient). The cheap levers must be
  exhausted first, in order: (1) **per-span counterfactual credit assignment** — the privacy
  side is exactly computable per span from the cached P6 table (swap one action, hold the rest),
  and a leave-one-out utility estimate is one extra reader pass per span; (2) **batch the QA
  reader** (~90% of wall time; 22 min/alpha for only 115 updates) then train 5-10x longer;
  (3) entropy bonus / higher lr — secondary, exploration is not the binding constraint.
- **The alpha-clustering fallback (lexicographic reward) is NOT triggered either** — it
  presupposes trained policies whose realized privacy clusters; here the policies never left
  the init, so alpha placement was never actually tested.
- **Why the walk is hard to beat at its own reward:** the BC init inherits per-span decisions
  that are already risk-optimal under the same walk_risk table the reward's privacy term is
  correlated with (P4 and P6 agree on ordering more often than not). The learnable residual is
  the cross-span and utility-aware part — and that is exactly the axis the utility gradient
  must carry, which measurement shows is near-silent (rare probe flips). Lever (1) attacks this
  directly.
- **The v2-detector environment is a better training substrate than v1's** independent of this
  null: probe supply doubled (54 -> 106 train probes; v2's DEM/MISC recall creates more
  gold-restated spans), 177 decision spans, gate held (U~realized 0.354/0.511/0.760). The BC
  baseline operating point is placeholder-heavy (ph 0.639 greedy) at tau=0.02 — consistent with
  the strict-tau regime (chance ~ 1/16 in the contrastive probe's candidate set).
- **Harness validity held throughout**: BC reproduction 23/23 (assemble == deployed
  substitute), dynamic injectivity mask (zero collisions by construction), persisted probe
  splits — the null is attributable to the learning signal, not to environment bugs, which is
  what all that Phase-0/review machinery was for.

### Superseded diagnosis (2026-07-04, same day)

**The "optimization-regime null" diagnosis above is superseded by a reward-support diagnosis
measured the same day** (empirical-honesty rule — records are append-corrected, not rewritten;
the original text stays intact above). A single-action counterfactual scan
(`scripts/spikes/probe_flip_scan.py`) found that **only 3/106 train probes could flip under any
single-action swap at tau=0.02**: the reward was structurally near-constant over the policy's
entire action space, not merely served by a weak/diluted gradient. This means the record's lever
ladder — per-span counterfactual credit assignment, QA-reader batching + longer training, entropy
bonus — **could not have cured the null**: there was no reward signal to integrate, however many
updates or however clean the credit assignment. The binding constraint was reward *support*
(the strict-tau mask left almost nothing flippable), not optimization dynamics.

**Successor environment** (fixes the support desert at its root — the mask itself): per-type
anonymity-set **count floors** replace the tau=0.02 walk_risk mask, roughly doubling decision
freedom (**146/177 spans with ≥2 legal actions vs 72/177** under tau=0.02) and legalizing
**keep-original** as a first-class action so probe-bearing attribute spans gain keep/level trades.
walk_risk itself is retired to offline calibration/diagnostics. Migration and spec:
[structural-lattice-risk plan](../../docs/plans/2026-07-04-structural-lattice-risk.md) ·
[RL spec (rewritten)](../../docs/specs/RL/surrogate-ranker-infiller.md). Re-measuring probe-flip
support on the floor environment is a pre-registered part of the next run's gate.

## Ablations
alpha sweep is built in; feature-only vs frozen-encoder features is the pre-registered follow-up.

## Cost
Local only (iGPU). Smoke: 40 s. Full: ~30 min/alpha (u_qa reader unbatched — known ceiling;
batch the reader if a second sweep is needed). Zero remote calls.

## Risks & caveats
u_qa quantization at 2.5 probes/doc (advantage noise); opposing placeholder biases (A excludes
placeholders, u_qa under-credits them) — equilibrium is alpha-governed, spec §7-2; corpus mix
dominated by clinical.

## Artifacts
data/ranker_policy_a{alpha}.pt, results/ranker_train_a{alpha}.json, env: data/ranker_env.json,
arms artifact data/task_arms_tau0.02.json.

## Successor
[2026-07-05 floor-env rerun](2026-07-05-RL-ranker-v2-stage1-floor-env.md) — NULL persisted at
doubled reward support; diagnosis refined to a reward-structure finding (u_qa is
penalty-only around a min-aset init).

## Sources
Spec: docs/specs/RL/surrogate-ranker-infiller.md (§2 Phase 1, §5). Plan:
docs/plans/2026-07-02-surrogate-grpo-training.md (policy-scope decision, ablation floor).
