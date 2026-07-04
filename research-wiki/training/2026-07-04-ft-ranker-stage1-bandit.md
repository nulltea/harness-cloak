---
type: training-experiment
status: planned
created: 2026-07-04
model: RankerPolicy feature-MLP (64-h, ~5k params), behavior-cloned from the tau-walk
dataset: ranker_env.json — 20 trainable docs (9 clinical / 7 enron / 4 aeslc; probe- AND span-bearing), 139 decision spans, 54 train probes
result: pending
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
data/ranker_env.json (Phase-0 artifact): action tables with stored P4 walk_risk + P6 proximity,
tau-legal masks (tau=0.02), behavior-clone labels, persisted probe splits (seed 0). Trainable =
probe-bearing AND span-bearing: 20 docs / 139 decision spans / 54 train probes (the runner logs
the realized counts at startup and verifies assemble(bc) == artifact tau_walk doc_p, 20/20).

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

## Results
pending

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
