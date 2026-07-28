---
type: training-experiment
status: done
created: 2026-07-28
model: semantic-v1 policy (BC warm start, pinned frozen encoder
  thomas-sounack/BioClinical-ModernBERT-base@c3648aa8, direct-count privacy signal)
dataset: aci 2-doc smoke set (D2N001, D2N002), frozen environment
  sha256:4cc754a7, utility artifact qa-utility-runtime-v2 policy denominator
result: "TRAIN PASS — first real hybrid optimizer steps: opposing finite alpha
  gradients at every nonzero lambda, exact lambda-zero identity (0 failures),
  utility/counterfactual/entropy terms all live, loadable pinned checkpoint"
tags: [rl, ranker-v2, hybrid, direct-counts, smoke]
companion: ../../docs/issues/2026-07-27-pre-rl-reward-audit.md
---

# RL-ranker v7 — hybrid direct-count training smoke

Supersedes the v6 mechanics slice (which stopped at the cache-only remote
boundary and preregistered the retired learned privacy head). Executed
2026-07-28 in-session after the pre-RL reward audit; spec and results recorded
together — the run doubled as the audited pipeline's first live validation.

## Objective & hypothesis

Mechanics smoke for the full hybrid loop under the final architecture:
semantic-v1 policy, direct grounded-count privacy signal, policy-role utility
denominator (qa-utility-runtime-v2), frozen preflight calibration. Pass
criteria preregistered in v6: nontrivial utility components, finite opposing
alpha gradients, exact lambda-zero identity, loadable semantic checkpoint.

## Training data / config

- Docs: aci/D2N001 + aci/D2N002 (D2N048 dropped by the zero-signal filter).
- BC warm start: 3 epochs, lr 1e-4, seed 17 (policy utilities 0.9677 / 0.9526).
- ExIt: 4 rollouts/doc, 8 unique candidates, 0 winners (BC reference near
  ceiling — expected).
- Hybrid: 4 epochs x Latin cycle over the frozen 4-profile lambda menu
  (results/ranker_v2/preflight/lambda-menu.json), 4 rollouts/group,
  counterfactual budget 5 (frozen manifest), beta 0.01, eta 0.01, lr 1e-4,
  KL reference = BC checkpoint (published to kl-reference-smoke.pt), device cuda.

## Results (measured)

- TRAIN PASS documents=2 profiles=4 epochs=4; conditional train-smoke.pt +
  fixed lambda-zero control control-smoke.pt published, full pin structure.
- Opposing alpha gradients at every nonzero lambda, all finite: count-side
  alpha grad norms 0.012-0.038 vs utility-side (linked) 0.003-0.023; net alpha
  gradient -0.0035..-0.037; alpha drifted 1.0000 -> 1.0003 over 4 epochs.
- Exact lambda-zero identity: 0 failures in all 8 epoch reports.
- Utility term live and growing (linked gradient mass 0.49 -> 2.62);
  counterfactual substitutions executed each epoch; entropy term active;
  KL never enabled (no collapse trigger) — consistent with 4 smoke epochs.
- Two latent defects found and fixed by this run (each pinned by test):
  credit-pair coverage vs load-time-demoted decisions (fb6a3ae), and the
  kl-reference-checkpoint output path clobbering the BC checkpoint when
  pointed at it (operator error documented here: it is an OUTPUT path).

## Cost

All remote work local (medgemma via llama-swap) and largely cache-hit;
hybrid chain wall time ~13 min on the shared iGPU.

## Artifacts

results/ranker_v2/architecture/policy/{train-smoke.pt, control-smoke.pt,
kl-reference-smoke.pt, train-smoke-epochs.json, train-smoke.log};
lambda menu + threshold manifest under results/ranker_v2/preflight/ (hashes
inside); utility cache at 442+ identities. Code at commit fb6a3ae.

## Risks & caveats

2-doc smoke: no learning claims, no promotion evidence, ACI contamination
bars generalization statements. Alpha movement is mechanical evidence only.
The v6 record remains as the mechanics-slice predecessor.
