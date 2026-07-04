---
type: training-experiment
status: done
created: 2026-07-04
model: RankerPolicy feature-MLP (64-h, ~5k params), behavior-cloned from the tau-walk
dataset: ranker_env.json (DETECTOR v2 pii_gliner_multidomain@0.3, switched pre-run) — 23 trainable docs (12 clinical / 8 enron / 3 aeslc), 177 decision spans, 106 train probes
result: "NULL at these settings — policy never left the BC init at any alpha (identical greedy read-outs); diagnosis: weak per-span credit, rare probe flips, not entropy collapse"
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
alpha.** All three greedy read-outs are identical (A 0.2813, U 0.0909, ph_rate 0.6394 — the BC
policy's own decisions); per-epoch means flat (e.g. a=0.3: r 0.2808 -> 0.2805 over 5 epochs);
KL(pi||pi_0) <= 0.002 everywhere — the leash never engaged, the policy simply did not move.
No alpha Pareto spread exists (all alphas converge to one operating point).

**Diagnosis (measured on the trained a=0.5 policy):** NOT entropy collapse — mean policy entropy
0.52 nats on multi-action spans (uniform-2 = 0.69), and per-doc reward std over G=8 is nonzero
on 8/8 docs (mean 0.024, max 0.043). The gradient exists but is weak: (a) doc-level REINFORCE
dilutes the advantage over ~8 spans/doc; (b) rollout variations rarely flip a QA probe (a flip
would move r by ~0.1 at alpha=0.5; observed std 0.024 means most rollouts share identical U), so
the utility term contributes almost no rollout-level gradient; (c) 115 updates x lr 3e-4 is a
tiny optimization budget against that signal. This is an optimization-regime null, not yet a
feature-ceiling null — the pre-registered frozen-encoder upgrade is NOT triggered until the
cheap optimization levers are exhausted (per-span/leave-one-out credit assignment, more epochs
with a batched reader, entropy bonus / higher lr).

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

## Sources
Spec: docs/specs/RL/surrogate-ranker-infiller.md (§2 Phase 1, §5). Plan:
docs/plans/2026-07-02-surrogate-grpo-training.md (policy-scope decision, ablation floor).
